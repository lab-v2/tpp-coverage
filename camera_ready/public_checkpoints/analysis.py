#!/usr/bin/env python3
"""Budget-matched CO-versus-NC re-analysis of public released losses.

Implements CRITERIA.md exactly. Every numeric primitive - the law forms, the
multi-start fitter, the bounding-box NC selector, and V_K / tau_K / regime - is
imported from the paper's code in src/tpp_analysis/; nothing is re-derived here.

Writes ``results.csv`` (or ``--out``) under the work directory (paths.WORK): one row per
(cell, window, ray, seed, law) design pair.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass, field
from typing import Iterator, Sequence

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

from paths import ARTIFACTS, REPO_SRC, WORK

sys.path.insert(0, str(REPO_SRC))
sys.path.insert(0, str(REPO_SRC / "subset_analysis"))

import subset_analysis_bbox_seeded as repo  # noqa: E402  (path set above)
from scaling_laws.laws import chinchilla_loss, kaplan_loss  # noqa: E402

# ── pre-registered constants (CRITERIA.md) ──────────────────────────────────
HOLDOUT_TAIL_FRAC = 0.15
N_RESTARTS = 80
SEEDS = (0, 1, 2, 3, 4)
BETA_EFF = 0.28
BETA_GRID = (0.10, 0.20, 0.28, 0.35, 0.50)
KAPPA_TARGET = 1e2
KAPPA_GRID = (1e2, 1e4, 1e5, 1e6)
MERGE_TOL = 0.01
MIN_DISTINCT_N_IN_DESIGN = 4

LAWS = {
    "Chinchilla": (repo._fit_chinchilla, chinchilla_loss, 5),
    "Kaplan": (repo._fit_kaplan, kaplan_loss, 4),
}

# P3 screening thresholds
MIN_DISTINCT_N_CELL = 6
MIN_POPULATED_BINS = 3
MIN_N_PER_BIN = 4


def ray_bin(k: float) -> int:
    """CRITERIA C1: factor-2 log bins anchored at k = 1."""
    return int(math.floor(math.log2(k)))


def merged_ratios(ks: Sequence[float]) -> list[float]:
    """CRITERIA C1b: distinct k merged at 1 percent relative to their geometric mean."""
    out: list[float] = []
    group: list[float] = []
    for k in sorted(float(x) for x in ks):
        if group and (k - group[0]) / group[0] <= MERGE_TOL:
            group.append(k)
        else:
            if group:
                out.append(float(np.exp(np.mean(np.log(group)))))
            group = [k]
    if group:
        out.append(float(np.exp(np.mean(np.log(group)))))
    return out


# ════════════════════════════════════════════════════════════════════════════
# Screening and window construction
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CellScreen:
    cell: str
    n_points: int
    n_distinct_N: int
    n_crossed_bins: int
    included: bool
    reason: str


def screen_cell(df: pd.DataFrame, cell: str) -> CellScreen:
    """CRITERIA P3: >= 6 distinct N and >= 3 k-bins each populated at >= 4 distinct N."""
    n_N = df["N"].nunique()
    bins = df.assign(b=df["k"].map(ray_bin)).groupby("b")["N"].nunique()
    crossed = int((bins >= MIN_N_PER_BIN).sum())
    if n_N < MIN_DISTINCT_N_CELL:
        return CellScreen(cell, len(df), n_N, crossed, False,
                          f"P3: only {n_N} distinct N (< {MIN_DISTINCT_N_CELL})")
    if crossed < MIN_POPULATED_BINS:
        return CellScreen(cell, len(df), n_N, crossed, False,
                          f"P3: only {crossed} k-bins populated at >= {MIN_N_PER_BIN} sizes")
    return CellScreen(cell, len(df), n_N, crossed, True, "included")


@dataclass(frozen=True)
class Window:
    cell: str
    index: int
    sizes: tuple[float, ...]

    @property
    def n_gm(self) -> float:
        return float(np.exp(np.mean(np.log(self.sizes))))


def build_windows(cell: str, sizes: Sequence[float]) -> list[Window]:
    """CRITERIA W1: contiguous size windows holding the within-window N-span fixed."""
    ordered = sorted(sizes)
    n = len(ordered)
    if n >= 12:
        width, stride = 8, 2
    elif n >= 8:
        width, stride = 5, 1
    else:
        return [Window(cell, 0, tuple(ordered))]
    return [
        Window(cell, i // stride, tuple(ordered[i:i + width]))
        for i in range(0, n - width + 1, stride)
    ]


# ════════════════════════════════════════════════════════════════════════════
# Designs
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Split:
    """CRITERIA W3: design-independent off-ray holdout, fixed before any carving."""

    core: pd.DataFrame
    holdout: pd.DataFrame

    @classmethod
    def from_window(cls, pts: pd.DataFrame) -> "Split":
        ranked = pts.sort_values("k")  # index must stay global: it addresses `table`
        n = len(ranked)
        tail = int(round(HOLDOUT_TAIL_FRAC * n))
        if tail < 1 or n - 2 * tail < 5:
            return cls(ranked.iloc[0:0], ranked.iloc[0:0])
        holdout = pd.concat([ranked.iloc[:tail], ranked.iloc[n - tail:]])
        return cls(ranked.iloc[tail:n - tail].copy(), holdout)


@dataclass
class DesignPair:
    """One budget-matched CO / NC comparison inside one window."""

    window: Window
    ray: int
    n_rays: int
    seed: int
    co_idx: list[int]
    nc_idx: list[int]
    ncm_idx: list[int] = field(default_factory=list)


def _grid_info(core: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Build the (sorted N) x (sorted D) index grid the repo's selector consumes."""
    runs = [{"N": float(r.N), "D": float(r.D), "_row": int(r.Index)} for r in core.itertuples()]
    all_N = sorted({r["N"] for r in runs})
    all_D = sorted({r["D"] for r in runs})
    n_idx = {v: i for i, v in enumerate(all_N)}
    d_idx = {v: i for i, v in enumerate(all_D)}
    grid: dict[tuple[int, int], list[dict]] = {}
    for r in runs:
        grid.setdefault((n_idx[r["N"]], d_idx[r["D"]]), []).append(r)
    return {"m": len(all_N), "n": len(all_D), "grid": grid}, runs


def enumerate_pairs(window: Window, split: Split, n_rays_variants=(1, 2)) -> Iterator[DesignPair]:
    """CRITERIA W5/W6: every admissible ray, budget-matched NC from the same core."""
    core = split.core
    if core.empty:
        return
    core = core.copy()
    core["b"] = core["k"].map(ray_bin)
    grid_info, runs = _grid_info(core)
    run_tpp = {id(r): ray_bin(r["D"] / r["N"]) for r in runs}

    for n_rays in n_rays_variants:
        for b0 in sorted(core["b"].unique()):
            bands = list(range(b0, b0 + n_rays))
            co = core[core["b"].isin(bands)]
            if len(co) < 4 or co["N"].nunique() < MIN_DISTINCT_N_IN_DESIGN:
                continue
            if n_rays > 1 and co["b"].nunique() < n_rays:
                continue
            n = len(co)
            # N-matched pool (CRITERIA Amendment 2): same model sizes as CO, so the
            # only thing that differs between the designs is ratio coverage.
            pool_m = core[core["N"].isin(set(co["N"]))]
            grid_m, runs_m = _grid_info(pool_m)
            run_tpp_m = {id(r): ray_bin(r["D"] / r["N"]) for r in runs_m}

            for seed in SEEDS:
                bbox_seed = repo.stable_hash(window.cell, window.index, b0, n_rays, seed) & 0x7FFFFFFF
                nc_runs, *_ = repo.select_nc_bbox(
                    target_n_points=n, target_tpps=bands, run_tpp=run_tpp,
                    grid_info=grid_info, bbox_seed=bbox_seed,
                )
                ncm_runs, *_ = repo.select_nc_bbox(
                    target_n_points=n, target_tpps=bands, run_tpp=run_tpp_m,
                    grid_info=grid_m, bbox_seed=bbox_seed,
                )
                if len(nc_runs) != n or len(ncm_runs) != n:
                    continue
                yield DesignPair(
                    window=window, ray=b0, n_rays=n_rays, seed=seed,
                    co_idx=list(co.index),
                    nc_idx=[r["_row"] for r in nc_runs],
                    ncm_idx=[r["_row"] for r in ncm_runs],
                )


# ════════════════════════════════════════════════════════════════════════════
# Fitting
# ════════════════════════════════════════════════════════════════════════════

def _rmse(loss_fn, params, sub: pd.DataFrame) -> float:
    if len(sub) == 0 or params is None:
        return float("nan")
    pred = loss_fn(sub["N"].to_numpy(float), sub["D"].to_numpy(float), *params)
    if not np.all(np.isfinite(pred)):
        return float("nan")
    return float(np.sqrt(np.mean((pred - sub["loss"].to_numpy(float)) ** 2)))


def _diagnostics(ks: Sequence[float]) -> dict:
    rats = merged_ratios(ks)
    vk = float(repo.compute_VK(rats, BETA_EFF))
    tk = float(repo.compute_tauK(rats, BETA_EFF, KAPPA_TARGET))
    out = {
        "K": len(rats),
        "V_K": vk,
        "tau_K": tk,
        "regime": repo.classify_regime(vk, tk),
        "log10_VK_over_tauK": float(np.log10(vk / tk)) if vk > 0 and tk > 0 else float("-inf"),
    }
    for beta in BETA_GRID:
        v = float(repo.compute_VK(rats, beta))
        t = float(repo.compute_tauK(rats, beta, KAPPA_TARGET))
        out[f"regime_beta{beta}"] = repo.classify_regime(v, t)
    for kap in KAPPA_GRID:
        t = float(repo.compute_tauK(rats, BETA_EFF, kap))
        out[f"regime_kappa{kap:.0e}"] = repo.classify_regime(vk, t)
    return out


ONRAY_TRAIN_FRAC = 0.8


def _subsample(idx: Sequence[int], seed: int) -> tuple[list[int], list[int]]:
    """Split a design into a kept fraction and a withheld remainder, seeded."""
    rng = np.random.RandomState(seed & 0x7FFFFFFF)
    perm = rng.permutation(len(idx))
    n_keep = int(np.ceil(ONRAY_TRAIN_FRAC * len(idx)))
    keep = [idx[i] for i in sorted(perm[:n_keep])]
    drop = [idx[i] for i in sorted(perm[n_keep:])]
    return keep, drop


_STATE: dict = {}


def _init_worker(table: pd.DataFrame, n_restarts: int) -> None:
    _STATE["table"] = table
    repo.N_RESTARTS = n_restarts


def _fit_one(job) -> dict | None:
    pair, holdout_idx, law_name = job
    table: pd.DataFrame = _STATE["table"]
    fit_fn, loss_fn, min_pts = LAWS[law_name]
    holdout = table.loc[holdout_idx]

    repo.GLOBAL_SEED = pair.seed
    row = {
        "cell": pair.window.cell,
        "window": pair.window.index,
        "N_gm": pair.window.n_gm,
        "N_min": min(pair.window.sizes),
        "N_max": max(pair.window.sizes),
        "n_sizes": len(pair.window.sizes),
        "ray_bin": pair.ray,
        "n_rays": pair.n_rays,
        "seed": pair.seed,
        "law": law_name,
        "n_train": len(pair.co_idx),
        "n_holdout_offray": len(holdout),
    }

    # On-ray (interpolation) arm: withhold the same fraction of each design's own
    # points, refit on the remainder - still identical cardinality for CO and NC -
    # and score BOTH on the points withheld from the CO ray. See CRITERIA Amendment 1.
    onray_seed = repo.stable_hash(pair.window.cell, pair.window.index, pair.ray, pair.seed)
    co_keep, co_drop = _subsample(pair.co_idx, onray_seed)
    nc_keep, _ = _subsample(pair.nc_idx, onray_seed)
    ncm_keep, _ = _subsample(pair.ncm_idx, onray_seed)
    onray_hold = table.loc[co_drop]
    onray_ok = len(co_drop) >= 2 and len(co_keep) >= min_pts and len(nc_keep) == len(co_keep)
    row["n_holdout_onray"] = len(co_drop) if onray_ok else 0

    for label, idx, keep in (
        ("CO", pair.co_idx, co_keep),
        ("NC", pair.nc_idx, nc_keep),
        ("NCM", pair.ncm_idx, ncm_keep),
    ):
        sub = table.loc[idx]
        if len(sub) < min_pts:
            return None
        diags = _diagnostics(sub["k"].tolist())
        params, _, _ = fit_fn(
            sub["N"].to_numpy(float), sub["D"].to_numpy(float), sub["loss"].to_numpy(float)
        )
        row.update({f"{label}_{k}": v for k, v in diags.items()})
        row[f"{label}_k_min"] = float(sub["k"].min())
        row[f"{label}_k_max"] = float(sub["k"].max())
        row[f"{label}_n_sizes"] = int(sub["N"].nunique())
        row[f"{label}_rmse_train"] = _rmse(loss_fn, params, sub)
        row[f"{label}_rmse_offray"] = _rmse(loss_fn, params, holdout)

        if onray_ok:
            sub_k = table.loc[keep]
            p_on, _, _ = fit_fn(
                sub_k["N"].to_numpy(float), sub_k["D"].to_numpy(float), sub_k["loss"].to_numpy(float)
            )
            row[f"{label}_rmse_onray"] = _rmse(loss_fn, p_on, onray_hold)
        else:
            row[f"{label}_rmse_onray"] = np.nan

    def _gap(num: float, den: float) -> float:
        return float(np.log10(num / den)) if np.isfinite(num) and np.isfinite(den) and den > 0 else np.nan

    row["gap_offray"] = _gap(row["CO_rmse_offray"], row["NC_rmse_offray"])
    row["gap_offray_ncm"] = _gap(row["CO_rmse_offray"], row["NCM_rmse_offray"])
    row["gap_onray"] = _gap(row["CO_rmse_onray"], row["NC_rmse_onray"])
    row["gap_onray_ncm"] = _gap(row["CO_rmse_onray"], row["NCM_rmse_onray"])
    return row


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fit budget-matched CO and NC designs on public runs.")
    ap.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 8))
    ap.add_argument("--restarts", type=int, default=N_RESTARTS)
    ap.add_argument("--farseer-loss", choices=["loss", "loss_smooth", "loss_val"], default="loss",
                    help="CRITERIA C5 sensitivity variant for the Farseer cell only")
    ap.add_argument("--runs", default="runs_normalised.csv",
                    help="normalised run table under artifacts/")
    ap.add_argument("--corpus", choices=["E", "X", "all"], default="all",
                    help="CRITERIA A4.2: E = terminal-observation cells only, X = E plus the "
                         "cells admitted by P2'")
    ap.add_argument("--screening", default="screening.csv")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--cells", nargs="+", default=None,
                    help="fit only these cells (row indices and seeds are unchanged, so the "
                         "rows equal the matching rows of a full run)")
    args = ap.parse_args(argv)

    runs = pd.read_csv(ARTIFACTS / args.runs)
    if args.corpus != "all" and "corpus" in runs.columns:
        keep = {"E": {"E"}, "X": {"E", "X"}}[args.corpus]
        runs = runs[runs["corpus"].isin(keep)]
    runs = runs.reset_index(drop=True)
    if args.farseer_loss != "loss":
        mask = runs["cell"].str.startswith("farseer/")
        runs.loc[mask, "loss"] = runs.loc[mask, args.farseer_loss]
        runs = runs[runs["loss"] > 0]
        print(f"Farseer loss column swapped to {args.farseer_loss}")
    runs = runs.reset_index(drop=True)

    screens, jobs = [], []
    for cell, grp in runs.groupby("cell"):
        if args.cells and cell not in args.cells:
            continue
        screen = screen_cell(grp, cell)
        screens.append(screen)
        if not screen.included:
            continue
        for window in build_windows(cell, grp["N"].unique()):
            pts = grp[grp["N"].isin(window.sizes)]
            split = Split.from_window(pts)
            if split.core.empty or len(split.holdout) < 3:
                continue
            hold_idx = list(split.holdout.index)
            for pair in enumerate_pairs(window, split):
                for law in LAWS:
                    jobs.append((pair, hold_idx, law))

    pd.DataFrame([s.__dict__ for s in screens]).to_csv(WORK / args.screening, index=False)
    print("Cells screened:")
    for s in screens:
        print(f"  {'IN ' if s.included else 'OUT'} {s.cell:36s} pts={s.n_points:5d} "
              f"#N={s.n_distinct_N:3d} crossed_bins={s.n_crossed_bins:2d}  {s.reason}")
    print(f"\n{len(jobs)} fits queued on {args.workers} workers")

    ctx = mp.get_context("fork")
    with ctx.Pool(
        args.workers,
        initializer=_init_worker,
        initargs=(runs, args.restarts),
    ) as pool:
        rows = [r for r in pool.imap_unordered(_fit_one, jobs, chunksize=4) if r]

    out = pd.DataFrame(rows).sort_values(["cell", "window", "n_rays", "ray_bin", "law", "seed"])

    # Integrity checks: a CO design must actually be collinear, and the two designs
    # must be budget-matched on an identical holdout.
    span = out["CO_k_max"] / out["CO_k_min"]
    bad_span = int((span > 2.0 ** out["n_rays"] * 1.001).sum())
    assert bad_span == 0, f"{bad_span} CO designs span more than their ray width"
    assert (out["CO_n_sizes"] >= MIN_DISTINCT_N_IN_DESIGN).all(), "CO design lost its N-lever"
    print(f"integrity OK: max CO k-span within a 1-ray design = "
          f"{span[out['n_rays'] == 1].max():.3f} (must be <= 2)")

    out.to_csv(WORK / args.out, index=False)
    print(f"wrote {args.out}  rows={len(out)}")


if __name__ == "__main__":
    main()
