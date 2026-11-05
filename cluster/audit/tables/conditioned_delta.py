#!/usr/bin/env python3
"""Condition the size comparison on the design contrast Delta.

select_nc_bbox removed the WITHIN-pair collinearity (corr(Delta, logN) -0.993 ->
-0.197) but not the BETWEEN-stratum mean shift: Delta median is +0.870 at low-N
and +0.711 at high-N, 18% weaker, and within high-N corr(win, Delta) = -0.377.
So an unconditioned low-vs-high difference confounds "the effect attenuated with
N" with "the NC design got weaker". Two conditioned estimates:

  A. Common-support trim: keep only comparisons whose Delta lies in the overlap
     of the two strata's IQRs, then recompute the difference.
  B. Linear probability model  win ~ logN + Delta, per law. The logN coefficient
     is the size effect holding contrast fixed; reported in pp per decade.

Both use the design-clustered bootstrap (cluster = CO subset S), which remains
anti-conservative because the 226 designs are drawn from 39 trained cells.
Vectorised and serial: ~1 s total.
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Input CSV: the first argument, else the committed copy in ../results/.
RESULTS = Path(__file__).resolve().parents[1] / "results"

B, SEED = 4000, 20260802
d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else RESULTS / "bbox_strata.csv")
d["win"] = (d.co_off_ray > d.nc_off_ray).astype(float)
rng = np.random.default_rng(SEED)
LAWS = ["Chinchilla", "Kaplan", "Droppo-Elibol"]

lo_q = d[d.stratum == "low"].Delta.quantile([.25, .75]).to_numpy()
hi_q = d[d.stratum == "high"].Delta.quantile([.25, .75]).to_numpy()
band = (max(lo_q[0], hi_q[0]), min(lo_q[1], hi_q[1]))
print("Delta by stratum:")
for s in ("low", "high"):
    x = d[d.stratum == s].Delta
    print(f"  {s:5s} median {x.median():+.3f}  IQR [{x.quantile(.25):+.3f}, {x.quantile(.75):+.3f}]  n={len(x)}")
print(f"common-support band = [{band[0]:+.3f}, {band[1]:+.3f}]\n")

def clu_boot(sub, stat):
    """Bootstrap resampling whole designs (CO subsets)."""
    keys = sub.S.unique()
    idx = {k: sub.index[sub.S == k].to_numpy() for k in keys}
    out = np.empty(B)
    for b in range(B):
        pick = rng.choice(keys, len(keys), replace=True)
        rows = np.concatenate([idx[k] for k in pick])
        out[b] = stat(sub.loc[rows])
    return np.percentile(out, [2.5, 97.5])

print("=== A. common-support trim (Delta restricted to the overlap) ===")
t = d[(d.Delta >= band[0]) & (d.Delta <= band[1])]
print(f"{'law':<15}{'low-N':>18}{'high-N':>18}{'  diff':>20}")
for law in LAWS:
    L = t[(t.law == law) & (t.stratum == "low")]; H = t[(t.law == law) & (t.stratum == "high")]
    if len(L) < 20 or len(H) < 20:
        print(f"{law:<15}  insufficient after trim (low {len(L)}, high {len(H)})"); continue
    pl, ph = L.win.mean(), H.win.mean()
    cl = clu_boot(L, lambda s: s.win.mean()); ch = clu_boot(H, lambda s: s.win.mean())
    both = pd.concat([L, H])
    cd = clu_boot(both, lambda s: (s[s.stratum=="high"].win.mean() - s[s.stratum=="low"].win.mean())
                  if len(s[s.stratum=="high"]) and len(s[s.stratum=="low"]) else np.nan)
    print(f"{law:<15}{100*pl:6.1f}% [{100*cl[0]:4.1f},{100*cl[1]:5.1f}]"
          f"{100*ph:8.1f}% [{100*ch[0]:4.1f},{100*ch[1]:5.1f}]"
          f"{100*(ph-pl):+8.1f}pp [{100*cd[0]:+6.1f},{100*cd[1]:+6.1f}]   n={len(L)}/{len(H)}")

print("\n=== B. win ~ logN + Delta   (logN coeff = size effect, contrast held fixed) ===")
def ols(df, cols):
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in cols])
    return np.linalg.lstsq(X, df.win.to_numpy(float), rcond=None)[0]
print(f"{'law':<15}{'logN coeff (pp/decade)':>28}{'Delta coeff (pp/dex)':>26}")
for law in LAWS:
    s = d[d.law == law]
    pt = ols(s, ["logN", "Delta"])
    keys = s.S.unique(); idx = {k: s.index[s.S == k].to_numpy() for k in keys}
    bs = np.empty((B, 3))
    for b in range(B):
        pick = rng.choice(keys, len(keys), replace=True)
        bs[b] = ols(s.loc[np.concatenate([idx[k] for k in pick])], ["logN", "Delta"])
    ln_ci = np.percentile(bs[:, 1], [2.5, 97.5]); dl_ci = np.percentile(bs[:, 2], [2.5, 97.5])
    print(f"{law:<15}{100*pt[1]:+9.1f} [{100*ln_ci[0]:+7.1f},{100*ln_ci[1]:+7.1f}]"
          f"{100*pt[2]:+12.1f} [{100*dl_ci[0]:+6.1f},{100*dl_ci[1]:+6.1f}]")

print("\n=== cell-count imbalance (the second concern) ===")
for s in ("low", "high"):
    x = d[d.stratum == s]
    print(f"  {s:5s} n cells: median {x.n.median():.0f}  range {x.n.min()}-{x.n.max()}  designs {x.S.nunique()}")
