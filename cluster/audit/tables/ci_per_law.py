#!/usr/bin/env python3
"""Per-law bootstrap CIs on the between-stratum off-ray difference.

Input: bbox_strata.csv (raw per-comparison rows, one per CO-subset x law x seed).
Cluster unit = the CO subset S, i.e. the design. Comparisons sharing a design are
dependent; they ALSO share trained cells across designs, so this remains
anti-conservative and is labelled as such.

Vectorised: per design we precompute (wins, count), so a bootstrap replicate is
one gather plus two sums, and all B replicates are drawn as a single
(B, n_designs) index matrix. The whole computation is ~15 ms, so it is
deliberately SERIAL -- a process pool here costs minutes of worker startup to
save microseconds. The heavy fitting that produces bbox_strata.csv is what uses
every core.
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Input CSV: the first argument, else the committed copy in ../results/.
RESULTS = Path(__file__).resolve().parents[1] / "results"

CSV = sys.argv[1] if len(sys.argv) > 1 else RESULTS / "bbox_strata.csv"
B, SEED = 4000, 20260802
d = pd.read_csv(CSV)
d["win"] = (d.co_off_ray > d.nc_off_ray).astype(float)
rng = np.random.default_rng(SEED)

def agg(sub):
    g = sub.groupby("S")["win"].agg(["sum", "count"])
    return g["sum"].to_numpy(float), g["count"].to_numpy(float)

def boot(w, c):
    idx = rng.integers(0, len(w), size=(B, len(w)))
    return w[idx].sum(1) / c[idx].sum(1)          # ratio-of-sums, cluster bootstrap

q = lambda a: np.percentile(a, [2.5, 97.5])
print(f"{len(d)} comparisons, {d.S.nunique()} distinct CO subsets, {d.law.nunique()} laws")
print("cluster = CO subset S (anti-conservative: designs share trained cells)\n")
print(f"{'law':<15}{'low-N':>16}{'high-N':>16}{'  diff (high-low)':>22}{'  designs':>13}")
for law in ["Chinchilla", "Kaplan", "Droppo-Elibol"]:
    L = d[(d.law == law) & (d.stratum == "low")]
    H = d[(d.law == law) & (d.stratum == "high")]
    if L.empty or H.empty:
        continue
    lw, lc = agg(L); hw, hc = agg(H)
    bl, bh = boot(lw, lc), boot(hw, hc)
    pl, ph = lw.sum() / lc.sum(), hw.sum() / hc.sum()
    cl, ch, cd = q(bl), q(bh), q(bh - bl)
    print(f"{law:<15}{100*pl:6.1f}% [{100*cl[0]:4.1f},{100*cl[1]:5.1f}]"
          f"{100*ph:8.1f}% [{100*ch[0]:4.1f},{100*ch[1]:5.1f}]"
          f"{100*(ph-pl):+9.1f}pp [{100*cd[0]:+6.1f},{100*cd[1]:+6.1f}]"
          f"{len(lw):>7}/{len(hw):<5}")

print(f"\ncorr(Delta, logN) = {np.corrcoef(d.Delta, d.logN)[0,1]:+.3f}   "
      f"(was -0.993 before select_nc_bbox was applied per stratum)")
L, H = d[d.stratum == "low"], d[d.stratum == "high"]
print(f"FOOTNOTE ONLY - pooled: low {100*L.win.mean():.1f}% -> high {100*H.win.mean():.1f}% "
      f"(+{100*(H.win.mean()-L.win.mean()):.1f}pp), a composition effect driven by Kaplan.")
