#!/usr/bin/env python3
"""Cell-level bootstrap, flat-parallel.

Resamples TRAINED CELLS (within N row), rebuilds the CO subsets and matched NC
bbox from the resampled pool, refits. Cell-level sampling uncertainty therefore
propagates, unlike the design-clustered bootstrap which treats 226 subsets built
from ~39 cells as independent.

PERFORMANCE: design construction (resample -> bins -> subsets -> select_nc_bbox)
is pure bookkeeping and runs serially in a few seconds. Every (replicate, subset)
FIT is then a separate job in ONE flat pool, so all cores stay busy. Mapping over
replicates instead leaves cores idle -- 60 replicates cannot fill hundreds of workers.

GATE: the cell-level interval must be WIDER than the design-clustered one.
"""
import sys
import os
import itertools
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import wiki300m_paths  # the paper's code on sys.path, and the run directory
from scaling_laws.data import load_and_prepare, prepare_arrays
from scaling_laws.laws import droppo_elibol_loss, _fit_generic
from incremental_coverage.coverage import assign_tpp_bins
from subset_analysis.subset_analysis_bbox_seeded import select_nc_bbox, bbox_seed_for, MIN_POINTS_FOR_FIT
import seed_variance as sv

B_REPS   = int(sys.argv[1]) if len(sys.argv)>1 else 60
RESTARTS = int(sys.argv[2]) if len(sys.argv)>2 else 400
LAW      = sys.argv[3] if len(sys.argv)>3 else "Chinchilla"
BASE = wiki300m_paths.RUN_DIR

def fit_droppo_scaled(N,D,L,n,s):
    nlo,nhi=N.min()/1e3,N.max()*1e3; dlo,dhi=D.min()/1e3,D.max()*1e3
    b=[(1e-6,min(L)*0.99),(nlo,nhi),(dlo,dhi),(0.01,2.0),(0.01,2.0),(0.01,2.0)]
    ln,ld=(np.log10(nlo),np.log10(nhi)),(np.log10(dlo),np.log10(dhi))
    def sam(rng,L): return [rng.uniform(0.01,min(L)*0.8),10**rng.uniform(*ln),10**rng.uniform(*ld),
                            rng.uniform(0.05,1.5),rng.uniform(0.05,1.5),rng.uniform(0.05,1.5)]
    return _fit_generic(droppo_elibol_loss,N,D,L,b,sam,n,s)
def fitter_for(law):
    if law=="Droppo-Elibol": return fit_droppo_scaled, droppo_elibol_loss
    f,l=sv._SEED_FITTERS[law]; return (lambda N,D,L,n,s: f(N,D,L,n_restarts=n,seed=s)), l

d=load_and_prepare(f"{BASE}/interim/extracted/wikipedia.csv","W",verbose=False)
HOFF=prepare_arrays(d.noncolinear_holdout,mode="first")
MED=np.median([r["N"] for r in d.colinear_train+d.noncolinear_train])
MINPTS=MIN_POINTS_FOR_FIT[LAW]

def build_jobs():
    """Serial, cheap: every (rep, stratum, subset) -> raw arrays ready to fit."""
    jobs=[]
    for rep in range(B_REPS):
        rng=np.random.RandomState(90000+rep)
        for lab,sel in (("low",lambda r:r["N"]<=MED),("high",lambda r:r["N"]>MED)):
            co0=[r for r in d.colinear_train if sel(r)]; nc0=[r for r in d.noncolinear_train if sel(r)]
            def resample(pool):
                byrow=defaultdict(list)
                for r in pool: byrow[round(r["N"]/1e5)].append(r)
                o=[]
                for v in byrow.values(): o+=[v[i] for i in rng.randint(0,len(v),len(v))]
                return o
            co,nc=resample(co0),resample(nc0)
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
                    keep=set(S); cr=[r for r,t in zip(co,cb) if t in keep]
                    if len(cr)<MINPTS: continue
                    mask=sum(1<<bins.index(x) for x in S)
                    nr=select_nc_bbox(target_n_points=len(cr),target_tpps=[t for t in cb if t in keep],
                                      run_tpp=run_tpp,grid_info=gi,
                                      bbox_seed=bbox_seed_for("W","first",mask))[0]
                    if len(nr)!=len(cr): continue
                    Nc,Dc,Lc=prepare_arrays(cr,mode="first"); Nn,Dn,Ln=prepare_arrays(nr,mode="first")
                    jobs.append((rep,lab,Nc,Dc,Lc,Nn,Dn,Ln))
    return jobs

def run(j):
    rep,lab,Nc,Dc,Lc,Nn,Dn,Ln=j
    fit,loss=fitter_for(LAW)
    pc,_,_=fit(Nc,Dc,Lc,RESTARTS,1000); pn,_,_=fit(Nn,Dn,Ln,RESTARTS,1000)
    if pc is None or pn is None: return None
    eC=float(np.sqrt(np.mean((loss(HOFF[0],HOFF[1],*pc)-HOFF[2])**2)))
    eN=float(np.sqrt(np.mean((loss(HOFF[0],HOFF[1],*pn)-HOFF[2])**2)))
    return (rep,lab,int(eC>eN))

if __name__=="__main__":
    import time; t0=time.time()
    jobs=build_jobs(); W=max(1,len(os.sched_getaffinity(0))-2)
    print(f"{LAW}: {B_REPS} replicates -> {len(jobs)} FIT jobs (flat) on {W} workers "
          f"[build {time.time()-t0:.1f}s]",flush=True)
    with ProcessPoolExecutor(max_workers=W) as ex:
        res=[r for r in ex.map(run,jobs,chunksize=max(1,len(jobs)//(W*4))) if r]
    acc=defaultdict(lambda: [0,0])
    for rep,lab,w in res: a=acc[(rep,lab)]; a[0]+=w; a[1]+=1
    lo=np.array([v[0]/v[1] for (r,l),v in acc.items() if l=="low"  and v[1]])
    hi=np.array([v[0]/v[1] for (r,l),v in acc.items() if l=="high" and v[1]])
    n=min(len(lo),len(hi)); lo,hi=lo[:n],hi[:n]; diff=hi-lo
    q=lambda a: np.percentile(a,[2.5,97.5])
    print(f"\nreplicates: {n}   fits: {len(res)}   elapsed {time.time()-t0:.1f}s")
    print(f"  low  {100*lo.mean():5.1f}%  CI [{100*q(lo)[0]:5.1f},{100*q(lo)[1]:5.1f}]")
    print(f"  high {100*hi.mean():5.1f}%  CI [{100*q(hi)[0]:5.1f},{100*q(hi)[1]:5.1f}]")
    print(f"  diff {100*diff.mean():+5.1f}pp CI [{100*q(diff)[0]:+5.1f},{100*q(diff)[1]:+5.1f}]  <- CELL-level")
    w=100*(q(diff)[1]-q(diff)[0]); ref=9.7
    print(f"\n  cell-level diff CI width {w:.1f}pp  vs design-clustered {ref:.1f}pp")
    print("  GATE:", "PASS (wider)" if w>ref else "FAIL (not wider) - estimator suspect")
    np.save(f"{BASE}/cell_boot_{LAW}.npy", np.vstack([lo,hi]))
