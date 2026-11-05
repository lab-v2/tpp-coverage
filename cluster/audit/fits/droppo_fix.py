#!/usr/bin/env python3
"""Re-fit Droppo-Elibol only, with N_c/D_c bounds scaled to the observed data.

Diagnosis: at 300 AND 2000 restarts, Droppo NC high-N diverged (off-ray RMSE
0.40-5.76, cv 0.758) while every other law/stratum cell had cv <= 0.06. Cause is
not restart count: the released fitter puts N_c, D_c in (1e3, 1e14) -- eleven
decades -- with the sampler drawing log-uniform over only 1e5-1e12, while this
grid has N in [2.3e7, 2.0e8] and D in [1e8, 1e10]. Most restarts start far from
any basin.

Fix: bounds and sampler for the two scale parameters are tied to the data,
N_c in [min(N)/1e3, max(N)*1e3] and likewise for D_c. Everything else -- the law,
_fit_generic, the exponent bounds, the seeds, the arms -- is the repo's and
unchanged, and the change is applied IDENTICALLY to CO and NC.
"""
import sys
import os
import csv
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import wiki300m_paths  # the paper's code on sys.path, and the run directory
from scaling_laws.data import load_and_prepare, prepare_arrays
from scaling_laws.laws import droppo_elibol_loss, _fit_generic     # repo's law + fitter

B = wiki300m_paths.RUN_DIR
SEEDS=int(sys.argv[1]) if len(sys.argv)>1 else 30
RESTARTS=int(sys.argv[2]) if len(sys.argv)>2 else 2000

def fit_droppo_scaled(N, D, L, n_restarts, seed):
    nlo, nhi = N.min()/1e3, N.max()*1e3
    dlo, dhi = D.min()/1e3, D.max()*1e3
    bounds=[(1e-6, min(L)*0.99), (nlo,nhi), (dlo,dhi), (0.01,2.0),(0.01,2.0),(0.01,2.0)]
    ln, ld = (np.log10(nlo), np.log10(nhi)), (np.log10(dlo), np.log10(dhi))
    def sampler(rng, L):
        return [rng.uniform(0.01, min(L)*0.8),
                10**rng.uniform(*ln), 10**rng.uniform(*ld),
                rng.uniform(0.05,1.5), rng.uniform(0.05,1.5), rng.uniform(0.05,1.5)]
    return _fit_generic(droppo_elibol_loss, N, D, L, bounds, sampler, n_restarts, seed)

data=load_and_prepare(f"{B}/interim/extracted/wikipedia.csv","W",verbose=False)
Nc,Dc,Lc=prepare_arrays(data.colinear_train,mode="first")
Nn,Dn,Ln=prepare_arrays(data.noncolinear_train,mode="first")
HOLD={"pooled":prepare_arrays(data.unified_holdout,mode="first"),
      "on_ray":prepare_arrays(data.colinear_holdout,mode="first"),
      "off_ray":prepare_arrays(data.noncolinear_holdout,mode="first")}
# identical extreme-preserving balance as rerun_fixed.py
def keep_spread(idx, logD, k):
    if k>=len(idx): return list(idx)
    o=sorted(idx,key=lambda i:logD[i]); keep=[o[0],o[-1]]
    while len(keep)<k:
        best,bd=None,-1.0
        for i in o:
            if i in keep: continue
            dd=min(abs(logD[i]-logD[j]) for j in keep)
            if dd>bd: best,bd=i,dd
        keep.append(best)
    return sorted(keep,key=lambda i:logD[i])
def by_row(N):
    d=defaultdict(list)
    for i in range(len(N)): d[int(round(N[i]/1e5))].append(i)
    return d
lc,ln_=np.log(Dc),np.log(Dn); rc,rn=by_row(Nc),by_row(Nn); kc,kn=[],[]
for row in sorted(set(rc)&set(rn)):
    m=min(len(rc[row]),len(rn[row])); kc+=keep_spread(rc[row],lc,m); kn+=keep_spread(rn[row],ln_,m)
NcB,DcB,LcB,NnB,DnB,LnB=Nc[kc],Dc[kc],Lc[kc],Nn[kn],Dn[kn],Ln[kn]
med=np.median(np.concatenate([NcB,NnB]))
specs=[("all",np.ones(len(NcB),bool),np.ones(len(NnB),bool)),
       ("low",NcB<=med,NnB<=med),("high",NcB>med,NnB>med)]

def run(j):
    tag,s,nc,dc,lc_,nn,dn,ln2=j
    pc,sc,_=fit_droppo_scaled(nc,dc,lc_,RESTARTS,1000+s)
    pn,sn,_=fit_droppo_scaled(nn,dn,ln2,RESTARTS,1000+s)
    if pc is None or pn is None: return None
    r={"stratum":tag,"law":"Droppo-Elibol","seed":s,"sse_co":float(sc),"sse_nc":float(sn)}
    for hk,(HN,HD,HL) in HOLD.items():
        r[f"co_{hk}"]=float(np.sqrt(np.mean((droppo_elibol_loss(HN,HD,*pc)-HL)**2)))
        r[f"nc_{hk}"]=float(np.sqrt(np.mean((droppo_elibol_loss(HN,HD,*pn)-HL)**2)))
    return r
jobs=[(t,s,NcB[mc],DcB[mc],LcB[mc],NnB[mn],DnB[mn],LnB[mn])
      for t,mc,mn in specs if mc.sum()>=6 and mn.sum()>=6 for s in range(SEEDS)]
W=max(1,len(os.sched_getaffinity(0))-2)
print(f"{len(jobs)} Droppo fit-pairs x {RESTARTS} restarts on {W} workers (data-scaled N_c/D_c)",flush=True)
with ProcessPoolExecutor(max_workers=W) as ex: res=[r for r in ex.map(run,jobs) if r]
with open(f"{B}/droppo_fixed.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)
import pandas as pd
d=pd.DataFrame(res)
print("\n=== divergence check (old: high-N NC cv 0.758, max 5.76) ===")
for t in ["all","low","high"]:
    s=d[d.stratum==t]
    if s.empty: continue
    print(f"  {t:5s} NC off-ray: min {s.nc_off_ray.min():.3f} max {s.nc_off_ray.max():.3f} "
          f"cv {s.nc_off_ray.std()/s.nc_off_ray.mean():.3f}"
          + ("   STILL DIVERGED" if s.nc_off_ray.max()>1.0 else "   converged"))
print("\n=== Droppo NC win rate (data-scaled bounds) ===")
for t in ["all","low","high"]:
    s=d[d.stratum==t]
    if s.empty: continue
    row=[f"{sp} {100*float((s[f'co_{sp}']>s[f'nc_{sp}']).mean()):5.1f}%" for sp in ["off_ray","on_ray","pooled"]]
    print(f"  {t:5s}  " + "   ".join(row))
print(f"\nwrote {B}/droppo_fixed.csv")
