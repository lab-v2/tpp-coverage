#!/usr/bin/env python3
"""CRITERIA Amendment 6: every reported quantity at 5 seeds and at 512 seeds, both corpora,
plus the A6.2 treatment of the <100M band.

The estimator is untouched. This file changes only the enumeration depth handed to it (A6.1)
and adds the two intervals A6.2b declares for a band where most cells carry one window. The
fast bootstrap engine of `cluster_bootstrap.py` is asserted against `conditional.py`'s own code
path on BOTH corpora before any Amendment-6 number is printed.

Writes `estimates_seeds512.csv`, `doseresponse_seeds512.csv`, `small_band.csv` to the work
directory. The headline numbers of the paper are rows of `estimates_seeds512.csv`.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import zlib
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import conditional as C
from cluster_bootstrap import ClusterOLS, Estimate
from paths import WORK
BIG = 200_000          # A5.2 resample depth, inherited
MED_BOOT = C.N_BOOT    # 2000, the pre-registered depth for a median (E1)
STRATA = ["<100M", "100M-1B", ">=1B"]
CONSTRUCTIONS = {"W1": (8, 2), "V1": (8, 1), "V2": (8, 8), "V3": (5, 5),
                 "V4": (5, 1), "V5": (4, 1), "V7": (4, 4)}


def pct(b: float) -> float:
    return 100.0 * (10.0 ** b - 1.0)


def prepare(path: str | Path) -> pd.DataFrame:
    """Attach the A3.1 regressors to a slim `enumerate_seeds.py` frame.

    Identical to `conditional.add_margins` in every respect that matters: the same
    winsorisation of a non-finite m_CO, the same complete-case column kept alongside, the
    same Delta definition, the same coarse strata.
    """
    df = pd.read_csv(path)
    b = C.BETA_PRIMARY
    for arm in ("CO", "NC", "NCM"):
        df[f"m_{arm}"] = df[f"{arm}_m_beta{b}"].replace([np.inf, -np.inf], np.nan)
    floor = df["m_CO"].min()
    df["m_CO_degenerate"] = df["m_CO"].isna()
    df["m_CO_cc"] = df["m_CO"]
    df["m_CO"] = df["m_CO"].fillna(floor)
    for arm in ("NC", "NCM"):
        df[f"Delta_{arm}"] = df[f"m_{arm}"] - df["m_CO"]
        df[f"Delta_{arm}_cc"] = df[f"m_{arm}"] - df["m_CO_cc"]
    df["stratum"] = df["N_gm"].map(C.coarse_stratum)
    df["logN"] = np.log10(df["N_gm"])
    df["suite"] = C.suite_of(df["cell"])
    df["cell_win"] = df["cell"] + "#" + df["window"].astype(str)
    df["ray_id"] = df["cell_win"] + "#" + df["ray_bin"].astype(str)
    df["_one"] = "all"
    df["row_id"] = np.arange(len(df))
    return df


def beta_delta(df: pd.DataFrame, beta: float) -> pd.DataFrame:
    """Re-derive the Delta regressors at another C2 beta, on a slim frame."""
    out = df.copy()
    for arm in ("CO", "NC", "NCM"):
        out[f"m_{arm}"] = out[f"{arm}_m_beta{beta}"].replace([np.inf, -np.inf], np.nan)
    floor = out["m_CO"].min()
    out["m_CO"] = out["m_CO"].fillna(floor)
    for arm in ("NC", "NCM"):
        out[f"Delta_{arm}"] = out[f"m_{arm}"] - out["m_CO"]
    return out


def suite_weights(df: pd.DataFrame) -> np.ndarray:
    """A4.4: every SUITE contributes equally regardless of how many pairs it supplies."""
    share = df["suite"].map(df["suite"].value_counts())
    return 1.0 / share.to_numpy(float)


def est(df: pd.DataFrame, y: str, x: str | list[str], fe: str = "cell",
        cluster=("cell", "window"), group: str = "cell", weights=None,
        n_boot: int = BIG, block: int = 1, seed: int = 20260802,
        index: int = 0, keep: bool = False) -> Estimate:
    xs = [x] if isinstance(x, str) else list(x)
    eng = ClusterOLS(df, y, xs, cluster_cols=cluster, weights=weights,
                     order_col="window", fe_col=fe, group_col=group)
    if eng.n_obs < len(xs) + 2 or eng.n_clusters < 1:
        return Estimate(np.nan, np.nan, np.nan, eng.n_obs, eng.n_clusters)
    return eng.estimate(np.random.default_rng(seed), n_boot, block=block,
                        index=index, keep_draws=keep)


# ════════════════════════════════════════════════════════════════════════════
# Medians for the dose-response (A3.2 / A3.3), with both pre-registered bootstraps
# ════════════════════════════════════════════════════════════════════════════

def _median_job(job):
    tag, vals, groups, seedkey = job
    # crc32, not hash(): str hashes are salted per process, which made these CIs irreproducible.
    rng = np.random.default_rng(zlib.crc32(seedkey.encode()))
    if len(vals) == 0:
        return tag, dict(n_pairs=0, n_windows=0, median=np.nan, lo=np.nan, hi=np.nan,
                         lo_cl=np.nan, hi_cl=np.nan, win=np.nan, signp=np.nan)
    med = float(np.median(vals))
    n = len(vals)
    chunk = max(1, min(MED_BOOT, int(2e7 // max(1, n))))
    draws = np.empty(MED_BOOT)
    for lo_i in range(0, MED_BOOT, chunk):
        b = min(chunk, MED_BOOT - lo_i)
        draws[lo_i:lo_i + b] = np.median(vals[rng.integers(0, n, size=(b, n))], axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    n_cl = sum(len(g) for g in groups)
    if n_cl >= 2:
        d2 = np.empty(MED_BOOT)
        for b in range(MED_BOOT):
            picked = [g[i] for g in groups for i in rng.integers(0, len(g), size=len(g))]
            d2[b] = np.median(vals[np.concatenate(picked)])
        lo_cl, hi_cl = np.percentile(d2, [2.5, 97.5])
    else:
        lo_cl = hi_cl = np.nan
    wins = int((vals > 0).sum())
    return tag, dict(n_pairs=len(vals), n_windows=n_cl, median=med, lo=float(lo), hi=float(hi),
                     lo_cl=float(lo_cl), hi_cl=float(hi_cl), win=wins / len(vals),
                     signp=float(binomtest(wins, len(vals)).pvalue))


def dose_jobs(df: pd.DataFrame, label: str):
    """One job per (arm, kappa, joint regime cell) - the A3.2 table, unmodified."""
    jobs = []
    for arm, gap in (("NC", "gap_offray"), ("NCM", "gap_offray_ncm")):
        m_co = df["m_CO_cc"].fillna(-np.inf).to_numpy(float)
        m_arm = df[f"m_{arm}"].to_numpy(float)
        good = np.isfinite(df[gap].to_numpy(float))
        for kappa in C.KAPPA_GRID:
            shift = np.log10(kappa / 100.0)
            co = np.where(m_co + shift >= 0, "B", "A")
            nc = np.where(m_arm + shift >= 0, "B", "A")
            name = pd.Series([f"CO={a},NC={b}" for a, b in zip(co, nc)], index=df.index)
            for cellname in C.CELL_ORDER:
                sub = df[(name == cellname) & good]
                v = sub[gap].to_numpy(float)
                by_cell: dict[str, list] = {}
                for (cc, _), ix in sub.reset_index(drop=True).groupby(["cell", "window"]).indices.items():
                    by_cell.setdefault(cc, []).append(np.asarray(ix))
                jobs.append(((label, arm, kappa, cellname), v, list(by_cell.values()),
                             f"{label}|{arm}|{kappa}|{cellname}"))
    return jobs


# ════════════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Every reported estimate at 5 and at 512 seeds (CRITERIA Amendment 6).")
    ap.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 8))
    ap.add_argument("--skip-x", action="store_true")
    args = ap.parse_args(argv)

    rows: list[dict] = []

    def rec(section, spec, arm, corpus, stratum, e5: Estimate, e512: Estimate, note=""):
        chg = ""
        if np.isfinite(e5.lo) and np.isfinite(e512.lo):
            if e5.excludes_zero != e512.excludes_zero:
                chg = "SIGNIFICANCE CHANGED"
            if np.isfinite(e5.value) and np.isfinite(e512.value) and \
                    np.sign(e5.value) != np.sign(e512.value):
                chg = (chg + " SIGN CHANGED").strip()
        rows.append({
            "section": section, "spec": spec, "arm": arm, "corpus": corpus,
            "stratum": stratum,
            "beta_5": e5.value, "lo_5": e5.lo, "hi_5": e5.hi,
            "beta_512": e512.value, "lo_512": e512.lo, "hi_512": e512.hi,
            "pct_5": pct(e5.value), "pct_lo_5": pct(e5.lo), "pct_hi_5": pct(e5.hi),
            "pct_512": pct(e512.value), "pct_lo_512": pct(e512.lo), "pct_hi_512": pct(e512.hi),
            "n_pairs_5": e5.n_obs, "n_pairs_512": e512.n_obs,
            "n_windows": e512.n_clusters,
            "excl0_5": e5.excludes_zero, "excl0_512": e512.excludes_zero,
            "change": chg, "note": note,
        })
        print(f"  {corpus:2s} {spec:46s} {e5}  ->  {e512}"
              f"{'   ** ' + chg if chg else ''}")

    # ── load ────────────────────────────────────────────────────────────────
    print("=" * 118)
    print("0.  VERIFICATION - the Amendment-6 arithmetic against the pre-registered code path")
    print("=" * 118)
    frames: dict[str, pd.DataFrame] = {}
    frames["E"] = prepare(WORK / "results_E_seeds512.csv")
    if not args.skip_x:
        frames["X"] = prepare(WORK / "results_X_seeds512.csv.gz")

    for corpus in frames:
        pre = C.add_margins(C.load(1, f"results_{corpus}.csv", f"margins_{corpus}.csv"),
                            C.BETA_PRIMARY)
        five = frames[corpus][frames[corpus]["seed"] < 5]
        for spec, fe in (("cell FE (PRIMARY)", "cell"), ("no FE (pooled)", "_one")):
            o = C.regress(pre, "gap_offray", ["Delta_NC"], None if fe == "_one" else "cell",
                          C.rng_for("primary", "NC") if fe == "cell"
                          else C.rng_for("companion", "NC", "no fixed effect"))["Delta_NC"]
            n = ClusterOLS(five, "gap_offray", ["Delta_NC"], fe_col=fe).point()[0]
            ok = abs(o.value - n) < 1e-8
            print(f"  {corpus}: {spec:22s} conditional.py {o.value:+.6f}   "
                  f"fast engine on seeds 0-4 {n:+.6f}   {'MATCH' if ok else 'MISMATCH'}")
            assert ok, (corpus, spec, o.value, n)
        for stratum in STRATA:
            s = pre[pre["stratum"] == stratum]
            if s.empty:
                continue
            o = C.regress(s, "gap_offray", ["Delta_NC"], "cell",
                          C.rng_for("stratum", "NC", stratum))["Delta_NC"]
            n = ClusterOLS(five[five["stratum"] == stratum], "gap_offray", ["Delta_NC"]).point()[0]
            print(f"  {corpus}: stratum {stratum:9s} conditional.py {o.value:+.6f}   "
                  f"fast {n:+.6f}   {'MATCH' if abs(o.value - n) < 1e-8 else 'MISMATCH'}")
            assert abs(o.value - n) < 1e-8, (corpus, stratum, o.value, n)
        print(f"  {corpus}-corpus enlarged frame: {len(frames[corpus])} rows, "
              f"{frames[corpus].groupby(['cell', 'window']).ngroups} windows, "
              f"{frames[corpus]['cell'].nunique()} cells, "
              f"{frames[corpus]['suite'].nunique()} suites, "
              f"seeds 0-{frames[corpus]['seed'].max()}")

    def pair(fn, *a, **kw):
        """The same statistic on seeds 0-4 and on seeds 0-511."""
        return fn(5, *a, **kw), fn(512, *a, **kw)

    # ── 1. primary and companions ───────────────────────────────────────────
    print("\n" + "=" * 118)
    print("1.  A3.1 PRIMARY AND COMPANIONS   5 seeds -> 512 seeds   (B = 200000 both sides)")
    print("=" * 118)
    for corpus, big in frames.items():
        print(f"\n  --- {corpus}-corpus ---")
        specs = [
            ("cell FE (PRIMARY)", "gap_offray", "Delta_NC", "cell", None, "NC"),
            ("no fixed effect (pooled)", "gap_offray", "Delta_NC", "_one", None, "NC"),
            ("no FE, suite-weighted (A4.4)", "gap_offray", "Delta_NC", "_one", "sw", "NC"),
            ("cell FE, suite-weighted (A4.4)", "gap_offray", "Delta_NC", "cell", "sw", "NC"),
            ("(cell,window) FE", "gap_offray", "Delta_NC", "cell_win", None, "NC"),
            ("cell FE, complete case", "gap_offray", "Delta_NC_cc", "cell", None, "NC"),
            ("secondary regressor m_CO (pred < 0)", "gap_offray", "m_CO", "cell", None, "NC"),
            ("NCM arm, cell FE", "gap_offray_ncm", "Delta_NCM", "cell", None, "NCM"),
            ("NCM arm, (cell,window) FE", "gap_offray_ncm", "Delta_NCM", "cell_win", None, "NCM"),
            ("NCM arm, no fixed effect", "gap_offray_ncm", "Delta_NCM", "_one", None, "NCM"),
        ]
        for spec, y, x, fe, w, arm in specs:
            def f(ns, y=y, x=x, fe=fe, w=w):
                d = big[big["seed"] < ns]
                return est(d, y, x, fe=fe, weights=suite_weights(d) if w else None)
            rec("1 primary", spec, arm, corpus, "all", *pair(f))
        for law in sorted(big["law"].unique()):
            def f(ns, law=law):
                d = big[(big["seed"] < ns) & (big["law"] == law)]
                return est(d, "gap_offray", "Delta_NC")
            rec("1 primary", f"law = {law}", "NC", corpus, "all", *pair(f))
        for b in C.BETA_GRID:
            def f(ns, b=b):
                return est(beta_delta(big[big["seed"] < ns], b), "gap_offray", "Delta_NC")
            rec("1 primary", f"C2 beta_eff = {b}", "NC", corpus, "all", *pair(f))

    # ── 2. per-suite, leave-one-suite-out, DataDecide ───────────────────────
    print("\n" + "=" * 118)
    print("2.  A4.5 PER-SUITE SLOPES AND THE POST-HOC LEAVE-ONE-SUITE-OUT")
    print("=" * 118)
    for corpus, big in frames.items():
        print(f"\n  --- {corpus}-corpus: per suite (cell FE inside the suite) ---")
        for suite in sorted(big["suite"].unique()):
            sd = float(big.loc[big["suite"] == suite, "Delta_NC"].std())
            def f(ns, suite=suite):
                d = big[(big["seed"] < ns) & (big["suite"] == suite)]
                fe = "cell" if d["cell"].nunique() > 1 else "_one"
                return est(d, "gap_offray", "Delta_NC", fe=fe, n_boot=50_000)
            note = "NOT IDENTIFIED: Delta sd = %.1e" % sd if sd < 1e-3 else ""
            rec("2 per suite", f"suite = {suite}", "NC", corpus, "all", *pair(f), note)
        print(f"\n  --- {corpus}-corpus: leave-one-suite-out, pooled no-FE (POST-HOC) ---")
        for suite in sorted(big["suite"].unique()):
            def f(ns, suite=suite):
                d = big[(big["seed"] < ns) & (big["suite"] != suite)]
                return est(d, "gap_offray", "Delta_NC", fe="_one")
            rec("2 LOSO pooled", f"drop {suite}", "NC", corpus, "all", *pair(f), "POST-HOC")
        print(f"\n  --- {corpus}-corpus: leave-one-suite-out, cell FE (POST-HOC) ---")
        for suite in sorted(big["suite"].unique()):
            def f(ns, suite=suite):
                d = big[(big["seed"] < ns) & (big["suite"] != suite)]
                return est(d, "gap_offray", "Delta_NC", fe="cell")
            rec("2 LOSO cellFE", f"drop {suite}", "NC", corpus, "all", *pair(f), "POST-HOC")

    if "X" in frames:
        print("\n  --- X-corpus: the MAD-excluded / CO-ray k >= 1 restriction (POST-HOC) ---")
        big = frames["X"]
        for spec, fe, w in (("pooled no-FE", "_one", None), ("cell FE", "cell", None),
                            ("no FE, suite-weighted", "_one", "sw")):
            def f(ns, fe=fe, w=w):
                d = big[(big["seed"] < ns) & (big["CO_k_min"] >= 1.0)]
                return est(d, "gap_offray", "Delta_NC", fe=fe,
                           weights=suite_weights(d) if w else None)
            rec("2 k>=1", f"CO ray k >= 1, {spec}", "NC", "X", "all", *pair(f), "POST-HOC")

    # ── 3. scale: interaction and the three coarse strata ───────────────────
    print("\n" + "=" * 118)
    print("3.  A3.4 SCALE - interaction term and the three coarse strata of W2")
    print("=" * 118)
    for corpus, big in frames.items():
        print(f"\n  --- {corpus}-corpus interaction: "
              f"gap ~ b0*Delta + b1*(Delta x logN_c) + b2*logN_c + cell FE ---")
        for arm, gap in (("NC", "gap_offray"), ("NCM", "gap_offray_ncm")):
            for i, term in enumerate(("b0 Delta", "b1 Delta x logN", "b2 logN")):
                def f(ns, arm=arm, gap=gap, i=i):
                    d = big[big["seed"] < ns].copy()
                    d["logN_c"] = d["logN"] - big["logN"].mean()
                    d["_ix"] = d[f"Delta_{arm}"] * d["logN_c"]
                    return est(d, gap, [f"Delta_{arm}", "_ix", "logN_c"], index=i)
                rec("3 interaction", term, arm, corpus, "all", *pair(f))
        print(f"\n  --- {corpus}-corpus within-stratum beta_delta ---")
        for arm, gap in (("NC", "gap_offray"), ("NCM", "gap_offray_ncm")):
            for stratum in STRATA:
                if (big["stratum"] == stratum).sum() == 0:
                    continue
                def f(ns, arm=arm, gap=gap, stratum=stratum):
                    d = big[(big["seed"] < ns) & (big["stratum"] == stratum)]
                    return est(d, gap, f"Delta_{arm}")
                nw = big.loc[big["stratum"] == stratum].groupby(["cell", "window"]).ngroups
                nc = big.loc[big["stratum"] == stratum, "cell"].nunique()
                rec("3 stratum", f"stratum {stratum}", arm, corpus, stratum, *pair(f),
                    f"{nw} windows in {nc} cells")
            def fb(ns, arm=arm, gap=gap):
                d = big[(big["seed"] < ns) & (big["stratum"] != ">=1B")]
                return est(d, gap, f"Delta_{arm}")
            rec("3 stratum", "stratum <1B (combined)", arm, corpus, "<1B", *pair(fb))

    pd.DataFrame(rows).to_csv(WORK / "estimates_seeds512.csv", index=False)

    # ── 4. dose-response and the same-regime falsification arm ──────────────
    print("\n" + "=" * 118)
    print("4.  A3.2 DOSE-RESPONSE AND A3.3b SAME-REGIME, EVERY KAPPA   5 seeds -> 512 seeds")
    print("=" * 118)
    jobs = []
    for corpus, big in frames.items():
        jobs += dose_jobs(big[big["seed"] < 5], f"{corpus}|5")
        jobs += dose_jobs(big, f"{corpus}|512")
    print(f"  {len(jobs)} regime cells on {args.workers} workers")
    with mp.get_context("fork").Pool(args.workers) as pool:
        res = dict(pool.map(_median_job, jobs, chunksize=1))
    drows = []
    for corpus in frames:
        for arm in ("NC", "NCM"):
            for kappa in C.KAPPA_GRID:
                for cellname in C.CELL_ORDER:
                    a = res[(f"{corpus}|5", arm, kappa, cellname)]
                    b = res[(f"{corpus}|512", arm, kappa, cellname)]
                    drows.append({
                        "corpus": corpus, "arm": arm, "kappa": kappa,
                        "regime_cell": cellname, "role": C.CELL_ROLE[cellname],
                        "pairs_5": a["n_pairs"], "pairs_512": b["n_pairs"],
                        "windows_5": a["n_windows"], "windows_512": b["n_windows"],
                        "gap_5": a["median"], "gap_512": b["median"],
                        "lo_pair_5": a["lo"], "hi_pair_5": a["hi"],
                        "lo_pair_512": b["lo"], "hi_pair_512": b["hi"],
                        "lo_wnd_5": a["lo_cl"], "hi_wnd_5": a["hi_cl"],
                        "lo_wnd_512": b["lo_cl"], "hi_wnd_512": b["hi_cl"],
                        "win_5": a["win"], "win_512": b["win"],
                        "signp_5": a["signp"], "signp_512": b["signp"],
                    })
    dr = pd.DataFrame(drows)
    dr.to_csv(WORK / "doseresponse_seeds512.csv", index=False)
    for corpus in frames:
        print(f"\n  --- {corpus}-corpus, NC arm ---")
        print(f"  {'kappa':>6s} {'regime cell':11s} {'pairs512':>9s} {'wnd':>4s} "
              f"{'gap5':>8s} {'CI(wnd) 5 seeds':>21s} {'gap512':>8s} "
              f"{'CI(wnd) 512 seeds':>21s} {'win%':>6s}")
        for r in dr[(dr["corpus"] == corpus) & (dr["arm"] == "NC")].itertuples():
            if r.pairs_512 == 0 and r.pairs_5 == 0:
                continue
            print(f"  {r.kappa:6.0e} {r.regime_cell:11s} {r.pairs_512:9d} {r.windows_512:4d} "
                  f"{r.gap_5:8.3f} [{r.lo_wnd_5:8.3f},{r.hi_wnd_5:8.3f}] "
                  f"{r.gap_512:8.3f} [{r.lo_wnd_512:8.3f},{r.hi_wnd_512:8.3f}] "
                  f"{100 * r.win_512:5.1f}%")

    # ── 5. the <100M band (A6.2) ────────────────────────────────────────────
    print("\n" + "=" * 118)
    print("5.  A6.2  THE <100M BAND: every construction, three intervals each")
    print("=" * 118)
    srows = []

    def small_row(corpus, name, d, seeds):
        width, stride = CONSTRUCTIONS[name]
        block = int(np.ceil(width / stride))
        s = d[d["stratum"] == "<100M"]
        if s.empty:
            print(f"  {corpus} {name} (w={width},s={stride}): NO <100M window exists")
            srows.append({"corpus": corpus, "construction": name, "width": width,
                          "stride": stride, "seeds": seeds, "n_windows": 0, "n_cells": 0,
                          "n_pairs": 0})
            return
        per_cell = s.groupby("cell")["window"].nunique()
        ew = est(s, "gap_offray", "Delta_NC")
        eb = est(s, "gap_offray", "Delta_NC", block=block)
        ec = est(s, "gap_offray", "Delta_NC", cluster=("cell",), group="_one")
        srows.append({
            "corpus": corpus, "construction": name, "width": width, "stride": stride,
            "seeds": seeds, "n_windows": int(per_cell.sum()), "n_cells": int(len(per_cell)),
            "cells_with_2plus_windows": int((per_cell >= 2).sum()),
            "n_pairs": ew.n_obs, "block_len": block,
            "beta": ew.value, "pct": pct(ew.value),
            "wnd_lo": ew.lo, "wnd_hi": ew.hi, "pct_wnd_lo": pct(ew.lo), "pct_wnd_hi": pct(ew.hi),
            "mb_lo": eb.lo, "mb_hi": eb.hi, "pct_mb_lo": pct(eb.lo), "pct_mb_hi": pct(eb.hi),
            "cell_lo": ec.lo, "cell_hi": ec.hi,
            "pct_cell_lo": pct(ec.lo), "pct_cell_hi": pct(ec.hi),
            "excl0_window": ew.excludes_zero, "excl0_movingblock": eb.excludes_zero,
            "excl0_cell": ec.excludes_zero,
            "windows_per_cell": ";".join(f"{c}={v}" for c, v in per_cell.items()),
        })
        print(f"  {corpus} {name} (w={width},s={stride})  windows={int(per_cell.sum()):3d} "
              f"in {len(per_cell)} cells ({int((per_cell >= 2).sum())} with >=2)  "
              f"pairs={ew.n_obs:6d}\n"
              f"        window-cluster {ew}   moving-block(L={block}) "
              f"[{eb.lo:+.4f}, {eb.hi:+.4f}]   cell-cluster [{ec.lo:+.4f}, {ec.hi:+.4f}]")
        for cellname, g in s.groupby("cell"):
            sd = float(g["Delta_NC"].std())
            b = (ClusterOLS(g, "gap_offray", ["Delta_NC"], fe_col="_one").point()[0]
                 if sd > 1e-3 else np.nan)
            print(f"          {cellname:40s} wnds={g['window'].nunique():2d} "
                  f"pairs={len(g):6d} N_gm=[{g['N_gm'].min():.2e},{g['N_gm'].max():.2e}] "
                  f"gap_med={g['gap_offray'].median():+.4f} "
                  f"NCwin={100 * (g['gap_offray'] > 0).mean():5.1f}%  "
                  f"within-cell beta={b:+.4f}" if np.isfinite(b) else
                  f"          {cellname:40s} wnds={g['window'].nunique():2d} "
                  f"pairs={len(g):6d} gap_med={g['gap_offray'].median():+.4f} "
                  f"NCwin={100 * (g['gap_offray'] > 0).mean():5.1f}%  beta NOT IDENTIFIED")
            srows.append({"corpus": corpus, "construction": f"{name} composition",
                          "seeds": seeds, "cell_detail": cellname,
                          "n_windows": int(g["window"].nunique()), "n_pairs": len(g),
                          "beta": b, "gap_median": float(g["gap_offray"].median()),
                          "nc_win_rate": float((g["gap_offray"] > 0).mean()),
                          "N_gm_min": float(g["N_gm"].min()),
                          "N_gm_max": float(g["N_gm"].max())})

    for corpus, big in frames.items():
        print(f"\n  --- {corpus}-corpus, 512 seeds ---")
        small_row(corpus, "W1", big, 512)
        if corpus == "E":
            for name in ("V1", "V2", "V3", "V4", "V5", "V7"):
                p = WORK / f"results_E_win{name}_small512.csv.gz"
                if not p.exists():
                    p = WORK / f"results_E_win{name}.csv"
                if not p.exists():
                    print(f"  E {name}: no fit file, skipped")
                    continue
                d = prepare(p)
                small_row("E", name, d, int(d["seed"].max()) + 1)

    print("\n  --- A6.2c borrowed strength: the smooth-in-log-N slope read below 100M ---")
    for corpus, big in frames.items():
        d = big.copy()
        ctr = d["logN"].mean()
        d["logN_c"] = d["logN"] - ctr
        d["_ix"] = d["Delta_NC"] * d["logN_c"]
        eng = ClusterOLS(d, "gap_offray", ["Delta_NC", "_ix", "logN_c"], order_col="window")
        pts = [("<100M centroid", d.loc[d["stratum"] == "<100M", "logN"].mean())]
        pts += [(f"log10 N = {v}", v) for v in (7.5, 8.0)]
        for label, lg in pts:
            if not np.isfinite(lg):
                continue
            combo = np.array([1.0, lg - ctr, 0.0])
            e = eng.estimate_combo(np.random.default_rng(4242), BIG, combo)
            print(f"    {corpus} {label:18s} {e}  = {pct(e.value):+.0f}% "
                  f"[{pct(e.lo):+.0f}%, {pct(e.hi):+.0f}%]"
                  f"{'   EXCLUDES ZERO' if e.excludes_zero else ''}")
            srows.append({"corpus": corpus, "construction": "smooth-in-logN",
                          "seeds": 512, "n_windows": e.n_clusters, "n_pairs": e.n_obs,
                          "beta": e.value, "wnd_lo": e.lo, "wnd_hi": e.hi,
                          "pct": pct(e.value), "pct_wnd_lo": pct(e.lo), "pct_wnd_hi": pct(e.hi),
                          "excl0_window": e.excludes_zero,
                          "windows_per_cell": f"read at {label}"})

    pd.DataFrame(srows).to_csv(WORK / "small_band.csv", index=False)
    print(f"\nwrote estimates_seeds512.csv, doseresponse_seeds512.csv, small_band.csv to {WORK}")


if __name__ == "__main__":
    main()
