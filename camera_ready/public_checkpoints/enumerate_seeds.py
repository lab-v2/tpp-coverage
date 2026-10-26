#!/usr/bin/env python3
"""CRITERIA Amendment 5, A5.1 / A5.3: re-fit the E-corpus at many more NC-carve seeds,
and under alternative window constructions.

Nothing about the carving, the fitting or the diagnostics is re-implemented: the windows,
the off-ray holdout, the CO enumeration, ``select_nc_bbox``, ``compute_VK``,
``compute_tauK`` and the two laws all come from `analysis.py`, which in turn imports them
from the paper's code in src/tpp_analysis/. The only things this file changes are the SEED
RANGE (A5.1), the window width/stride (A5.3), and the fact that it writes the
identifiability margins in the same row rather than reconstructing them later.

Two economies are declared in A5.1: single-ray CO only, and the off-ray arm only. The
on-ray secondary of Amendment 1 is untouched and is not recomputed here.

Writes a slim one-row-per-(cell, window, ray, seed, law) CSV.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from typing import Sequence

import numpy as np
import pandas as pd

import analysis as A

from paths import ARTIFACTS, WORK
ARMS = ("CO", "NC", "NCM")


# ════════════════════════════════════════════════════════════════════════════
# Window constructions (A5.3).  W1 = (8, 2) is the pre-registered one.
# ════════════════════════════════════════════════════════════════════════════

CONSTRUCTIONS = {
    "W1": (8, 2),   # pre-registered
    "V1": (8, 1),
    "V2": (8, 8),
    "V3": (5, 5),
    "V4": (5, 1),
    "V5": (4, 1),   # CRITERIA Amendment 6: narrowest admissible width, stride 1
    "V7": (4, 4),
}


def build_windows(cell: str, sizes: Sequence[float], width: int, stride: int) -> list[A.Window]:
    """W1 reproduces `analysis.build_windows` exactly; the rest are the A5.3 variants.

    W1 delegates rather than re-deriving, because the pre-registered W1 is not a single
    (width, stride): CRITERIA W1 prescribes (8, 2) only for cells with >= 12 distinct N and
    (5, 1) for cells with 8 <= #N < 12. No E-corpus cell falls in that band, so this changes
    nothing already computed; one X-corpus cell (gpt3, 8 sizes) does, and without the
    delegation the X-corpus run would silently use a construction W1 does not prescribe.
    """
    if (width, stride) == CONSTRUCTIONS["W1"]:
        return A.build_windows(cell, sizes)
    ordered = sorted(sizes)
    n = len(ordered)
    if n < width:
        return [A.Window(cell, 0, tuple(ordered))]
    return [
        A.Window(cell, j, tuple(ordered[i:i + width]))
        for j, i in enumerate(range(0, n - width + 1, stride))
    ]


# ════════════════════════════════════════════════════════════════════════════
# Task construction
# ════════════════════════════════════════════════════════════════════════════

class Task:
    """One (cell, window, ray) design family over a contiguous block of seeds."""

    __slots__ = ("cell", "win_index", "sizes", "hold_idx", "ray", "seed_lo", "seed_hi")

    def __init__(self, cell, win_index, sizes, hold_idx, ray, seed_lo, seed_hi):
        self.cell, self.win_index, self.sizes = cell, win_index, sizes
        self.hold_idx, self.ray = hold_idx, ray
        self.seed_lo, self.seed_hi = seed_lo, seed_hi


def make_tasks(runs: pd.DataFrame, width: int, stride: int, n_seeds: int,
               block: int, max_ngm: float = np.inf,
               cells: Sequence[str] | None = None) -> list[Task]:
    """`max_ngm` is the CRITERIA A6.2a compute economy: fit only the windows of one coarse
    band. Windows are carved and scored independently, so this changes no estimate inside
    the band - it only leaves the other bands unfitted for that construction."""
    tasks: list[Task] = []
    for cell, grp in runs.groupby("cell"):
        if cells and cell not in cells:
            continue
        if not A.screen_cell(grp, cell).included:
            continue
        for window in build_windows(cell, grp["N"].unique(), width, stride):
            if window.n_gm >= max_ngm:
                continue
            split = A.Split.from_window(grp[grp["N"].isin(window.sizes)])
            if split.core.empty or len(split.holdout) < 3:
                continue
            core = split.core.copy()
            core["b"] = core["k"].map(A.ray_bin)
            hold = tuple(split.holdout.index)
            for b0 in sorted(core["b"].unique()):
                co = core[core["b"] == b0]
                if len(co) < 4 or co["N"].nunique() < A.MIN_DISTINCT_N_IN_DESIGN:
                    continue
                for lo in range(0, n_seeds, block):
                    tasks.append(Task(cell, window.index, tuple(window.sizes), hold,
                                      b0, lo, min(lo + block, n_seeds)))
    return tasks


# ════════════════════════════════════════════════════════════════════════════
# Worker
# ════════════════════════════════════════════════════════════════════════════

_STATE: dict = {}


def _init_worker(runs: pd.DataFrame, n_restarts: int) -> None:
    _STATE["runs"] = runs
    A.repo.N_RESTARTS = n_restarts


def _margins(ks: Sequence[float]) -> dict:
    rats = A.merged_ratios(list(ks))
    out = {"K": len(rats)}
    for beta in A.BETA_GRID:
        vk = float(A.repo.compute_VK(rats, beta))
        tk = float(A.repo.compute_tauK(rats, beta, A.KAPPA_TARGET))
        out[f"m_beta{beta}"] = float(np.log10(vk / tk)) if vk > 0 else float("-inf")
    return out


def _rows_for(task: Task) -> list[dict]:
    runs: pd.DataFrame = _STATE["runs"]
    grp = runs[runs["cell"] == task.cell]
    split = A.Split.from_window(grp[grp["N"].isin(task.sizes)])
    core = split.core.copy()
    core["b"] = core["k"].map(A.ray_bin)
    grid_info, rr = A._grid_info(core)
    run_tpp = {id(r): A.ray_bin(r["D"] / r["N"]) for r in rr}

    co = core[core["b"] == task.ray]
    n = len(co)
    co_idx = list(co.index)
    pool_m = core[core["N"].isin(set(co["N"]))]
    grid_m, rm = A._grid_info(pool_m)
    run_tpp_m = {id(r): A.ray_bin(r["D"] / r["N"]) for r in rm}

    holdout = runs.loc[list(task.hold_idx)]
    n_gm = float(np.exp(np.mean(np.log(task.sizes))))
    co_marg = _margins(runs.loc[co_idx, "k"].tolist())

    out: list[dict] = []
    for seed in range(task.seed_lo, task.seed_hi):
        bbox_seed = A.repo.stable_hash(task.cell, task.win_index, task.ray, 1, seed) & 0x7FFFFFFF
        nc_runs, *_ = A.repo.select_nc_bbox(
            target_n_points=n, target_tpps=[task.ray], run_tpp=run_tpp,
            grid_info=grid_info, bbox_seed=bbox_seed)
        ncm_runs, *_ = A.repo.select_nc_bbox(
            target_n_points=n, target_tpps=[task.ray], run_tpp=run_tpp_m,
            grid_info=grid_m, bbox_seed=bbox_seed)
        if len(nc_runs) != n or len(ncm_runs) != n:
            continue
        nc_idx = [r["_row"] for r in nc_runs]
        ncm_idx = [r["_row"] for r in ncm_runs]

        A.repo.GLOBAL_SEED = seed
        for law_name, (fit_fn, loss_fn, min_pts) in A.LAWS.items():
            row = {
                "cell": task.cell, "window": task.win_index, "N_gm": n_gm,
                "N_min": min(task.sizes), "N_max": max(task.sizes),
                "ray_bin": task.ray, "seed": seed, "law": law_name, "n_train": n,
                "nc_key": A.repo.stable_hash(tuple(sorted(nc_idx))),
                "ncm_key": A.repo.stable_hash(tuple(sorted(ncm_idx))),
            }
            ok = True
            for label, idx, marg in (("CO", co_idx, co_marg),
                                     ("NC", nc_idx, None),
                                     ("NCM", ncm_idx, None)):
                sub = runs.loc[idx]
                if len(sub) < min_pts:
                    ok = False
                    break
                m = marg if marg is not None else _margins(sub["k"].tolist())
                row.update({f"{label}_{k}": v for k, v in m.items()})
                row[f"{label}_n_sizes"] = int(sub["N"].nunique())
                params, _, _ = fit_fn(sub["N"].to_numpy(float), sub["D"].to_numpy(float),
                                      sub["loss"].to_numpy(float))
                row[f"{label}_rmse_offray"] = A._rmse(loss_fn, params, holdout)
            if not ok:
                continue
            row["CO_k_min"] = float(runs.loc[co_idx, "k"].min())
            row["CO_k_max"] = float(runs.loc[co_idx, "k"].max())

            def gap(num, den):
                return float(np.log10(num / den)) if np.isfinite(num) and np.isfinite(den) and den > 0 else np.nan

            row["gap_offray"] = gap(row["CO_rmse_offray"], row["NC_rmse_offray"])
            row["gap_offray_ncm"] = gap(row["CO_rmse_offray"], row["NCM_rmse_offray"])
            out.append(row)
    return out


# ════════════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Re-fit the designs over many NC-carve seeds.")
    ap.add_argument("--construction", default="W1", choices=sorted(CONSTRUCTIONS))
    ap.add_argument("--seeds", type=int, default=512)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--workers", type=int, default=min(480, os.cpu_count() or 8))
    ap.add_argument("--runs", default="runs_expanded.csv")
    ap.add_argument("--corpus", default="E")
    ap.add_argument("--max-ngm", type=float, default=float("inf"),
                    help="CRITERIA A6.2a: fit only windows with N_gm below this")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", nargs="+", default=None, help="fit only these cells")
    args = ap.parse_args(argv)

    width, stride = CONSTRUCTIONS[args.construction]
    runs = pd.read_csv(ARTIFACTS / args.runs)
    if args.corpus != "all":
        runs = runs[runs["corpus"].isin({"E": {"E"}, "X": {"E", "X"}}[args.corpus])]
    runs = runs.reset_index(drop=True)

    tasks = make_tasks(runs, width, stride, args.seeds, args.block, args.max_ngm, args.cells)
    print(f"construction {args.construction} = (width {width}, stride {stride}); "
          f"seeds 0-{args.seeds - 1}; {len(tasks)} tasks on {args.workers} workers", flush=True)

    t0 = time.time()
    ctx = mp.get_context("fork")
    rows: list[dict] = []
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(runs, A.N_RESTARTS)) as pool:
        for i, chunk in enumerate(pool.imap_unordered(_rows_for, tasks, chunksize=1), 1):
            rows.extend(chunk)
            if i % 200 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(tasks)} tasks  {len(rows)} rows  {el:.0f}s "
                      f"(eta {el / i * (len(tasks) - i):.0f}s)", flush=True)

    out = pd.DataFrame(rows).sort_values(["cell", "window", "ray_bin", "law", "seed"])

    span = out["CO_k_max"] / out["CO_k_min"]
    assert (span <= 2.0 * 1.001).all(), "a single-ray CO design spans more than a factor of 2"
    assert (out["CO_n_sizes"] >= A.MIN_DISTINCT_N_IN_DESIGN).all(), "CO design lost its N-lever"
    print(f"integrity OK: max CO k-span = {span.max():.4f} (must be <= 2)")

    out.to_csv(WORK / args.out, index=False)
    print(f"wrote {args.out} rows={len(out)}  wall={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
