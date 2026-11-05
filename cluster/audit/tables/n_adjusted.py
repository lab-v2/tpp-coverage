#!/usr/bin/env python3
"""Adjust the size comparison for SUBSET SIZE (n_cells), the dominant confound.

High-N subsets are systematically smaller (median 9 cells vs 15 at low-N) and
small subsets lose, so part of the apparent "attenuation with N" is attenuation
with sample size. Delta is used only as a BALANCE DIAGNOSTIC, not as a control:
Theorem 1 gives an ordering, not a dose-response in Delta, so conditioning on it
can inject bias rather than remove it.

Estimates reported per law, as the between-stratum effect over the actual 0.574
decade separation, with the design-clustered bootstrap.
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
DLOG = d[d.stratum=="high"].logN.mean() - d[d.stratum=="low"].logN.mean()

print("=== verify the confound ===")
print(f"  corr(n_cells, logN) = {np.corrcoef(d.n, d.logN)[0,1]:+.3f}")
print(f"  corr(win,     n_cells) = {np.corrcoef(d.win, d.n)[0,1]:+.3f}")
print("  win rate by n_cells bin:")
for lo, hi in [(6,8),(8,11),(11,14),(14,18),(18,24)]:
    s = d[(d.n>=lo)&(d.n<hi)]
    if len(s): print(f"    {lo:2d}-{hi:2d}  {100*s.win.mean():5.1f}%   (n={len(s)})")
for s_ in ("low","high"):
    x = d[d.stratum==s_]
    print(f"  {s_:5s} n_cells median {x.n.median():.0f}")

def ols(df, cols):
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in cols])
    return np.linalg.lstsq(X, df.win.to_numpy(float), rcond=None)[0]

def clustered(s, cols):
    keys = s.S.unique(); idx = {k: s.index[s.S==k].to_numpy() for k in keys}
    bs = np.empty((B, len(cols)+1))
    for b in range(B):
        pick = rng.choice(keys, len(keys), replace=True)
        bs[b] = ols(s.loc[np.concatenate([idx[k] for k in pick])], cols)
    return ols(s, cols), bs

print(f"\n=== between-stratum effect (scaled to the actual {DLOG:.3f} decade separation) ===")
print(f"{'law':<15}{'logN only':>26}{'+ Delta':>26}{'+ Delta + n_cells':>28}")
for law in LAWS:
    s = d[d.law==law]
    out = []
    for cols in (["logN"], ["logN","Delta"], ["logN","Delta","n"]):
        pt, bs = clustered(s, cols)
        ci = np.percentile(bs[:,1], [2.5,97.5])
        out.append(f"{100*pt[1]*DLOG:+7.1f} [{100*ci[0]*DLOG:+6.1f},{100*ci[1]*DLOG:+6.1f}]")
    print(f"{law:<15}" + "".join(f"{o:>26}" for o in out))

print("\n=== did the Delta trim incidentally balance n? (the hypothesis) ===")
lo_q = d[d.stratum=="low"].Delta.quantile([.25,.75]).to_numpy()
hi_q = d[d.stratum=="high"].Delta.quantile([.25,.75]).to_numpy()
band = (max(lo_q[0],hi_q[0]), min(lo_q[1],hi_q[1]))
t = d[(d.Delta>=band[0])&(d.Delta<=band[1])]
for s_ in ("low","high"):
    a = d[d.stratum==s_].n.median(); b = t[t.stratum==s_].n.median()
    print(f"  {s_:5s} n_cells median: full {a:.0f}  ->  trimmed {b:.0f}")
print(f"  n gap: full {d[d.stratum=='low'].n.median()-d[d.stratum=='high'].n.median():.0f} cells"
      f"  ->  trimmed {t[t.stratum=='low'].n.median()-t[t.stratum=='high'].n.median():.0f} cells")
