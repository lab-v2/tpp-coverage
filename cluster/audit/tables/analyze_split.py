#!/usr/bin/env python3
"""Win rates + CIs + size interaction from split_rows.csv (produced by split_holdout.py).

This is the code under audit. It computes, per stratum and per holdout split:
  * NC win rate with a bootstrap CI clustered on FIT SEED
  * the two-stratum win-rate contrast (high - low)
  * a windowed regression of log10(RMSE_CO/RMSE_NC) on log10 N
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Input CSV: the first argument, else the committed copy in ../results/.
RESULTS = Path(__file__).resolve().parents[1] / "results"

d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else RESULTS / "split_rows.csv")
rng = np.random.default_rng(20260801)
B = 5000
SPLITS = ["off_ray", "on_ray", "pooled"]
wr = lambda c, n: float((c > n).mean())

def cluster_boot(sub, cc, nc, stat):
    seeds = sub["seed"].unique(); out = []
    for _ in range(B):
        pick = rng.choice(seeds, len(seeds), replace=True)
        rows = pd.concat([sub[sub["seed"] == s] for s in pick])
        out.append(stat(rows[cc].values, rows[nc].values))
    return np.percentile(out, [2.5, 97.5])

print(f"{'stratum':<8}{'split':<9}{'n':>5}{'winrate':>9}{'  95% CI':<18}{'median log10(CO/NC)':>21}")
res = {}
for strat in ["low", "high", "all"]:
    for sp in SPLITS:
        sub = d[d.stratum == strat]
        if sub.empty: continue
        c, n = sub[f"co_{sp}"].values, sub[f"nc_{sp}"].values
        lo, hi = cluster_boot(sub, f"co_{sp}", f"nc_{sp}", wr)
        res[(strat, sp)] = wr(c, n)
        print(f"{strat:<8}{sp:<9}{len(c):>5}{wr(c,n):>9.3f}  [{lo:.3f}, {hi:.3f}]"
              f"{float(np.median(np.log10(c/n))):>+21.3f}")

print("\nSIZE INTERACTION (two-stratum contrast)")
for sp in SPLITS:
    lo_s, hi_s = d[d.stratum == "low"], d[d.stratum == "high"]
    seeds = lo_s["seed"].unique(); diffs = []
    for _ in range(B):
        pick = rng.choice(seeds, len(seeds), replace=True)
        L = pd.concat([lo_s[lo_s.seed == s] for s in pick])
        H = pd.concat([hi_s[hi_s.seed == s] for s in pick])
        diffs.append(wr(H[f"co_{sp}"].values, H[f"nc_{sp}"].values)
                     - wr(L[f"co_{sp}"].values, L[f"nc_{sp}"].values))
    a, b = np.percentile(diffs, [2.5, 97.5])
    print(f"  high-low {sp:<8}: {res[('high',sp)]-res[('low',sp)]:+.3f}  95% CI [{a:+.3f}, {b:+.3f}]")

print("\nWINDOWED REGRESSION on log10 N  (SUSPECT: Delta is collinear with logN)")
win = d[d.stratum.str.startswith("win")].copy()
for sp in SPLITS:
    win["y"] = np.log10(win[f"co_{sp}"] / win[f"nc_{sp}"])
    g = win.groupby("stratum").agg(x=("logN_gm", "first"), y=("y", "median"))
    pt = np.polyfit(g.x, g.y, 1)[0]
    seeds = win.seed.unique(); sl = []
    for _ in range(2000):
        pick = rng.choice(seeds, len(seeds), replace=True)
        w2 = pd.concat([win[win.seed == s] for s in pick])
        gg = w2.groupby("stratum").agg(x=("logN_gm", "first"), y=("y", "median"))
        if len(gg) > 1: sl.append(np.polyfit(gg.x, gg.y, 1)[0])
    a, b = np.percentile(sl, [2.5, 97.5])
    print(f"  slope {sp:<8}: {pt:+.3f}  95% CI [{a:+.3f}, {b:+.3f}]")
