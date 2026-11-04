#!/usr/bin/env python3
"""Scale test, both audit fixes applied. Preregistered before any outcome was seen.

FIX 1 (balancing): the previous rule kept the cells nearest each row's median
log-D, which shaved 20.8% off NC's V_K -- i.e. it eroded the treatment. New rule
is symmetric and extreme-preserving: whichever arm must shed cells in a row keeps
its min and max log-D and fills the rest by farthest-point spacing. Both arms
keep their bounding box; cardinality is still matched per N row.

FIX 2 (convergence): Droppo-Elibol (6 params, bounds spanning 11 decades) failed
to find its basin in 20/30 seeds at 300 restarts, so the win rate was measuring
optimizer luck. Restarts raised to 2000, and a fit counts only if its TRAINING
SSE is within CONV_TOL of the best SSE for that (stratum, law, arm). Holdout is
never used to filter -- that would be outcome-dependent selection. A seed-pair
is retained only if BOTH arms converged, preserving the seed pairing.

No cell is ever dropped. Only failed optimizations are.
"""
import sys
import os
import csv
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import wiki300m_paths  # the paper's code on sys.path, and the run directory
from get_results.get_results import extract_from_local
from scaling_laws.data import load_and_prepare, prepare_arrays
from subset_analysis.subset_analysis_bbox_seeded import compute_VK, compute_tauK, classify_regime
import seed_variance as sv

B = wiki300m_paths.RUN_DIR
SEEDS   = int(sys.argv[1]) if len(sys.argv) > 1 else 30
RESTARTS= int(sys.argv[2]) if len(sys.argv) > 2 else 2000
CONV_TOL= 0.01          # 1% of best training SSE
BETA, KT = 0.28, 100.0

out = f"{B}/interim/extracted"; os.makedirs(out, exist_ok=True)
extract_from_local(f"{B}/shared/training_metrics", out)
data = load_and_prepare(f"{out}/wikipedia.csv", dataset_name="WIKI300M", verbose=False)
Nc, Dc, Lc = prepare_arrays(data.colinear_train,      mode="first")
Nn, Dn, Ln = prepare_arrays(data.noncolinear_train,   mode="first")
Nh, Dh, Lh = prepare_arrays(data.unified_holdout,     mode="first")
Non,Don,Lon= prepare_arrays(data.colinear_holdout,    mode="first")
Nof,Dof,Lof= prepare_arrays(data.noncolinear_holdout, mode="first")
HOLD = {"pooled": (Nh,Dh,Lh), "on_ray": (Non,Don,Lon), "off_ray": (Nof,Dof,Lof)}

def diag(N, D):
    k = D/N; v = compute_VK(k, BETA); t = compute_tauK(k, BETA, KT)
    return v, t, classify_regime(v, t)

# ---------------- FIX 1: extreme-preserving balanced subsample ----------------
def keep_spread(idx, logD, k):
    """Deterministically keep k of idx preserving the bounding box."""
    if k >= len(idx): return list(idx)
    order = sorted(idx, key=lambda i: logD[i])
    keep = [order[0], order[-1]]
    while len(keep) < k:                       # farthest-point fill
        best, bd = None, -1.0
        for i in order:
            if i in keep: continue
            d = min(abs(logD[i] - logD[j]) for j in keep)
            if d > bd: best, bd = i, d
        keep.append(best)
    return sorted(keep, key=lambda i: logD[i])

def by_row(N):
    d = defaultdict(list)
    for i in range(len(N)): d[int(round(N[i]/1e5))].append(i)
    return d

lc, ln = np.log(Dc), np.log(Dn)
rc, rn = by_row(Nc), by_row(Nn)
kc, kn = [], []
for row in sorted(set(rc) & set(rn)):
    m = min(len(rc[row]), len(rn[row]))
    kc += keep_spread(rc[row], lc, m)
    kn += keep_spread(rn[row], ln, m)
NcB,DcB,LcB = Nc[kc],Dc[kc],Lc[kc]
NnB,DnB,LnB = Nn[kn],Dn[kn],Ln[kn]

vco_raw,_,_ = diag(Nc,Dc); vnc_raw,_,_ = diag(Nn,Dn)
vco,tco,gco = diag(NcB,DcB); vnc,tnc,gnc = diag(NnB,DnB)
print(f"balanced arms: CO {len(LcB)} = NC {len(LnB)} over {len(set(np.round(NcB/1e5).astype(int)))} N rows")
print(f"  V_K  CO {vco:.4g} (tau {tco:.4g}) regime {gco}   [was {vco_raw:.4g} unbalanced]")
print(f"  V_K  NC {vnc:.4g} (tau {tnc:.4g}) regime {gnc}   [was {vnc_raw:.4g} unbalanced]")
print(f"  NC V_K retained by new balancing: {100*vnc/vnc_raw:.1f}%  (old rule kept 79.2%)")
print(f"  V_K ratio NC/CO = {vnc/vco:.2f}x")
print(f"  holdouts: pooled {len(Lh)}, on-ray {len(Lon)}, off-ray {len(Lof)}")

rows_all = sorted(set(np.round(NcB/1e5).astype(int)))
med = np.median(np.concatenate([NcB, NnB]))
specs = [("all", np.ones(len(NcB),bool), np.ones(len(NnB),bool)),
         ("low", NcB<=med, NnB<=med), ("high", NcB>med, NnB>med)]
for i in range(len(rows_all)-2):
    keep = set(rows_all[i:i+3])
    specs.append((f"win{i}",
                  np.array([int(round(x/1e5)) in keep for x in NcB]),
                  np.array([int(round(x/1e5)) in keep for x in NnB])))

def run(job):
    tag, law, s, nc,dc,lc_, nn,dn,ln_ = job
    fitter, lossfn = sv._SEED_FITTERS[law]
    pc, sse_c, _ = fitter(nc,dc,lc_, n_restarts=RESTARTS, seed=1000+s)
    pn, sse_n, _ = fitter(nn,dn,ln_, n_restarts=RESTARTS, seed=1000+s)
    if pc is None or pn is None: return None
    r = {"stratum":tag,"law":law,"seed":s,"sse_co":float(sse_c),"sse_nc":float(sse_n),
         "n_cells":int(len(lc_)),
         "logN_gm":float(np.mean(np.log10(np.concatenate([nc,nn]))))}
    for hk,(HN,HD,HL) in HOLD.items():
        if len(HL)==0: continue
        r[f"co_{hk}"]=float(np.sqrt(np.mean((lossfn(HN,HD,*pc)-HL)**2)))
        r[f"nc_{hk}"]=float(np.sqrt(np.mean((lossfn(HN,HD,*pn)-HL)**2)))
    return r

jobs=[]
for tag,mc,mn in specs:
    if mc.sum()<6 or mn.sum()<6: continue
    for law in sv._SEED_FITTERS:
        for s in range(SEEDS):
            jobs.append((tag,law,s,NcB[mc],DcB[mc],LcB[mc],NnB[mn],DnB[mn],LnB[mn]))
W = max(1, len(os.sched_getaffinity(0))-2)
print(f"\n{len(jobs)} fit-pairs x {RESTARTS} restarts on {W} workers ...", flush=True)
with ProcessPoolExecutor(max_workers=W) as ex:
    res=[r for r in ex.map(run, jobs) if r]

# ---------------- FIX 2: convergence rule on TRAINING SSE ----------------
best=defaultdict(lambda: np.inf)
for r in res:
    for a in ("co","nc"):
        k=(r["stratum"],r["law"],a); best[k]=min(best[k], r[f"sse_{a}"])
for r in res:
    ok_c = r["sse_co"] <= best[(r["stratum"],r["law"],"co")]*(1+CONV_TOL)
    ok_n = r["sse_nc"] <= best[(r["stratum"],r["law"],"nc")]*(1+CONV_TOL)
    r["converged"] = bool(ok_c and ok_n)
with open(f"{B}/rows_fixed.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)

print("\n=== convergence retention (seed-pairs kept / 30) ===")
for tag in [s[0] for s in specs]:
    line=[]
    for law in sv._SEED_FITTERS:
        sub=[r for r in res if r["stratum"]==tag and r["law"]==law]
        if not sub: continue
        line.append(f"{law[:4]} {sum(r['converged'] for r in sub):2d}/{len(sub):2d}")
    if line: print(f"  {tag:6s} " + "   ".join(line))

print("\n=== NC win rate, converged pairs only ===")
for tag in ["all","low","high"]:
    for sp in ["off_ray","on_ray","pooled"]:
        sub=[r for r in res if r["stratum"]==tag and r["converged"]]
        if not sub: continue
        per=[]
        for law in sv._SEED_FITTERS:
            L=[r for r in sub if r["law"]==law]
            if L: per.append(f"{law[:4]} {100*np.mean([r[f'co_{sp}']>r[f'nc_{sp}'] for r in L]):3.0f}%({len(L)})")
        w_=100*np.mean([r[f"co_{sp}"]>r[f"nc_{sp}"] for r in sub])
        print(f"  {tag:5s} {sp:8s} POOLED {w_:5.1f}%  n={len(sub):3d}   " + "  ".join(per))
print(f"\nwrote {B}/rows_fixed.csv")
