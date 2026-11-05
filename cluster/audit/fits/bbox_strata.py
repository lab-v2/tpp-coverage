#!/usr/bin/env python3
"""Per-stratum budget-matched CO/NC comparison using the paper's own select_nc_bbox.

Why: the trained stratum cannot identify a size interaction by regression, because
NC's k_max = D_max/N falls as 1/N by bbox construction, so the design contrast
Delta is collinear with log N (r = -0.993). Adding cells cannot fix it -- any new
D column still falls as 1/N.

Instead, hold the design contrast FIXED BY CONSTRUCTION: for every CO subset S,
select_nc_bbox builds a budget-matched NC design around it (the machinery behind
the paper's Table 6). Run that separately within each N stratum, and the residual
difference between strata IS the size effect, with contrast held fixed.

Everything here is the repo's: assign_tpp_bins, select_nc_bbox, bbox_seed_for,
_SEED_FITTERS, MIN_POINTS_FOR_FIT. Droppo uses the ratified data-scaled N_c/D_c.
"""
import sys
import os
import csv
import itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import wiki300m_paths  # the paper's code on sys.path, and the run directory
from scaling_laws.data import load_and_prepare, prepare_arrays
from scaling_laws.laws import droppo_elibol_loss, _fit_generic
from incremental_coverage.coverage import assign_tpp_bins
from subset_analysis.subset_analysis_bbox_seeded import (
    select_nc_bbox, bbox_seed_for, MIN_POINTS_FOR_FIT, compute_VK, compute_tauK)
import seed_variance as sv

B = wiki300m_paths.RUN_DIR
SEEDS=int(sys.argv[1]) if len(sys.argv)>1 else 5
RESTARTS=int(sys.argv[2]) if len(sys.argv)>2 else 2000
BETA,KT=0.28,100.0

def fit_droppo_scaled(N,D,L,n_restarts,seed):           # ratified deviation
    nlo,nhi=N.min()/1e3,N.max()*1e3; dlo,dhi=D.min()/1e3,D.max()*1e3
    bounds=[(1e-6,min(L)*0.99),(nlo,nhi),(dlo,dhi),(0.01,2.0),(0.01,2.0),(0.01,2.0)]
    ln,ld=(np.log10(nlo),np.log10(nhi)),(np.log10(dlo),np.log10(dhi))
    def sampler(rng,L):
        return [rng.uniform(0.01,min(L)*0.8),10**rng.uniform(*ln),10**rng.uniform(*ld),
                rng.uniform(0.05,1.5),rng.uniform(0.05,1.5),rng.uniform(0.05,1.5)]
    return _fit_generic(droppo_elibol_loss,N,D,L,bounds,sampler,n_restarts,seed)
def fitter_for(law):
    if law=="Droppo-Elibol": return fit_droppo_scaled, droppo_elibol_loss
    f,l=sv._SEED_FITTERS[law]; return (lambda N,D,L,n,s: f(N,D,L,n_restarts=n,seed=s)), l

d=load_and_prepare(f"{B}/interim/extracted/wikipedia.csv","WIKI300M",verbose=False)
HOLD={"pooled":prepare_arrays(d.unified_holdout,mode="first"),
      "on_ray":prepare_arrays(d.colinear_holdout,mode="first"),
      "off_ray":prepare_arrays(d.noncolinear_holdout,mode="first")}
med=np.median([r["N"] for r in d.colinear_train+d.noncolinear_train])

def margin(runs):
    N,D,_=prepare_arrays(runs,mode="first"); k=D/N
    return np.log10(compute_VK(k,BETA)/compute_tauK(k,BETA,KT))

JOBS=[]
for lab,sel in (("low",lambda r:r["N"]<=med),("high",lambda r:r["N"]>med)):
    co=[r for r in d.colinear_train if sel(r)]; nc=[r for r in d.noncolinear_train if sel(r)]
    cb=assign_tpp_bins(co,mode="first"); nb=assign_tpp_bins(nc,mode="first")
    run_tpp={id(r):t for r,t in zip(nc,nb)}
    aN=sorted(set(r["N"] for r in nc)); aD=sorted(set(r["D"] for r in nc))
    Ni={v:i for i,v in enumerate(aN)}; Di={v:i for i,v in enumerate(aD)}
    grid={}
    for r in nc: grid.setdefault((Ni[r["N"]],Di[r["D"]]),[]).append(r)
    gi={"m":len(aN),"n":len(aD),"grid":grid}
    bins=sorted(set(cb))
    for K in range(2,len(bins)+1):
        for S in itertools.combinations(bins,K):
            keep=set(S); co_runs=[r for r,t in zip(co,cb) if t in keep]
            if len(co_runs)<max(MIN_POINTS_FOR_FIT.values()): continue
            mask=sum(1<<bins.index(x) for x in S)
            nc_runs=select_nc_bbox(target_n_points=len(co_runs),
                                   target_tpps=[t for t in cb if t in keep],
                                   run_tpp=run_tpp, grid_info=gi,
                                   bbox_seed=bbox_seed_for("WIKI300M","first",mask))[0]
            if len(nc_runs)!=len(co_runs): continue
            JOBS.append((lab,tuple(S),co_runs,nc_runs))
print(f"{len(JOBS)} budget-matched (CO subset, NC bbox) pairs: "
      f"low {sum(1 for j in JOBS if j[0]=='low')}, high {sum(1 for j in JOBS if j[0]=='high')}")

def run(a):
    lab,S,co_runs,nc_runs,law,s=a
    fit,loss=fitter_for(law)
    Nc,Dc,Lc=prepare_arrays(co_runs,mode="first"); Nn,Dn,Ln=prepare_arrays(nc_runs,mode="first")
    if len(Lc)<MIN_POINTS_FOR_FIT[law] or len(Ln)<MIN_POINTS_FOR_FIT[law]: return None
    pc,sc,_=fit(Nc,Dc,Lc,RESTARTS,1000+s); pn,sn,_=fit(Nn,Dn,Ln,RESTARTS,1000+s)
    if pc is None or pn is None: return None
    r={"stratum":lab,"S":"|".join(map(str,S)),"law":law,"seed":s,"n":len(Lc),
       "m_co":margin(co_runs),"m_nc":margin(nc_runs),
       "logN":float(np.mean(np.log10(np.concatenate([Nc,Nn]))))}
    r["Delta"]=r["m_nc"]-r["m_co"]
    for hk,(HN,HD,HL) in HOLD.items():
        r[f"co_{hk}"]=float(np.sqrt(np.mean((loss(HN,HD,*pc)-HL)**2)))
        r[f"nc_{hk}"]=float(np.sqrt(np.mean((loss(HN,HD,*pn)-HL)**2)))
    return r

work=[(l,S,c,n,law,s) for (l,S,c,n) in JOBS for law in sv._SEED_FITTERS for s in range(SEEDS)]
W=max(1,len(os.sched_getaffinity(0))-2)
print(f"{len(work)} fit-pairs x {RESTARTS} restarts on {W} workers ...",flush=True)
with ProcessPoolExecutor(max_workers=W) as ex: res=[r for r in ex.map(run,work) if r]
with open(f"{B}/bbox_strata.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)
import pandas as pd
df=pd.DataFrame(res)
print("\n=== is the design contrast actually held fixed across strata? ===")
for lab in ["low","high"]:
    s=df[df.stratum==lab]
    print(f"  {lab:5s} Delta: median {s.Delta.median():+.3f}  IQR [{s.Delta.quantile(.25):+.3f},{s.Delta.quantile(.75):+.3f}]  n={len(s)}")
print(f"  corr(Delta, logN) over all pairs = {np.corrcoef(df.Delta,df.logN)[0,1]:+.3f}   (was -0.993)")
print("\n=== OFF-RAY NC win rate, contrast held fixed by construction ===")
for law in list(sv._SEED_FITTERS)+["POOLED"]:
    out=[]
    for lab in ["low","high"]:
        s=df[df.stratum==lab] if law=="POOLED" else df[(df.stratum==lab)&(df.law==law)]
        if s.empty: out.append(f"{lab} n/a"); continue
        out.append(f"{lab} {100*float((s.co_off_ray>s.nc_off_ray).mean()):5.1f}% (n={len(s)})")
    print(f"  {law:14s} " + "   ".join(out))
print(f"\nwrote {B}/bbox_strata.csv")
