#!/usr/bin/env python3
"""Required check: is the Droppo N_c/D_c bounds rescaling INERT on the published data?

If the paper's own 324-run Wikipedia numbers move under the rescaled bounds, the
released fitter has a latent problem and the camera-ready must say so. If they do
not move, the deviation is inert and only matters at the WIKI300M grid's scale.

Both variants are run on identical arms, identical seeds, identical restarts.
"""
import sys
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import wiki300m_paths  # the paper's code on sys.path, and the run directory
from scaling_laws.data import load_and_prepare, prepare_arrays
from scaling_laws.laws import droppo_elibol_loss, _fit_generic

B = wiki300m_paths.RUN_DIR
SEEDS=int(sys.argv[1]) if len(sys.argv)>1 else 30
RESTARTS=int(sys.argv[2]) if len(sys.argv)>2 else 2000

def fit_scaled(N,D,L,n_restarts,seed):
    nlo,nhi=N.min()/1e3,N.max()*1e3; dlo,dhi=D.min()/1e3,D.max()*1e3
    bounds=[(1e-6,min(L)*0.99),(nlo,nhi),(dlo,dhi),(0.01,2.0),(0.01,2.0),(0.01,2.0)]
    ln,ld=(np.log10(nlo),np.log10(nhi)),(np.log10(dlo),np.log10(dhi))
    def sampler(rng,L):
        return [rng.uniform(0.01,min(L)*0.8),10**rng.uniform(*ln),10**rng.uniform(*ld),
                rng.uniform(0.05,1.5),rng.uniform(0.05,1.5),rng.uniform(0.05,1.5)]
    return _fit_generic(droppo_elibol_loss,N,D,L,bounds,sampler,n_restarts,seed)

def fit_released(N,D,L,n_restarts,seed):
    # released fitter hardcodes seed=42 inside _fit_generic; reproduce it with the
    # released bounds/sampler but the seed plumbed, so the comparison is seed-paired
    bounds=[(1e-6,min(L)*0.99),(1e3,1e14),(1e3,1e14),(0.01,2.0),(0.01,2.0),(0.01,2.0)]
    def sampler(rng,L):
        return [rng.uniform(0.01,min(L)*0.8),10**rng.uniform(5,12),10**rng.uniform(5,12),
                rng.uniform(0.05,1.5),rng.uniform(0.05,1.5),rng.uniform(0.05,1.5)]
    return _fit_generic(droppo_elibol_loss,N,D,L,bounds,sampler,n_restarts,seed)

d=load_and_prepare(f"{B}/FINAL_wiki_epoch_results.csv","WIKI_PUBLISHED",verbose=True)
ARMS={"CO":prepare_arrays(d.colinear_train,mode="first"),
      "NC":prepare_arrays(d.noncolinear_train,mode="first")}
H=prepare_arrays(d.unified_holdout,mode="first")
print(f"published arms: CO {len(ARMS['CO'][2])}, NC {len(ARMS['NC'][2])}, holdout {len(H[2])}")
print(f"  CO N range {ARMS['CO'][0].min():.3g}-{ARMS['CO'][0].max():.3g}  D {ARMS['CO'][1].min():.3g}-{ARMS['CO'][1].max():.3g}")

def run(a):
    variant,arm,s=a
    N,D,L=ARMS[arm]
    f = fit_scaled if variant=="scaled" else fit_released
    p,sse,_=f(N,D,L,RESTARTS,1000+s)
    if p is None: return None
    return {"variant":variant,"arm":arm,"seed":s,"sse":float(sse),
            "rmse_hold":float(np.sqrt(np.mean((droppo_elibol_loss(H[0],H[1],*p)-H[2])**2)))}

work=[(v,a,s) for v in ("released","scaled") for a in ("CO","NC") for s in range(SEEDS)]
W=max(1,len(os.sched_getaffinity(0))-2)
print(f"{len(work)} fits x {RESTARTS} restarts on {W} workers ...",flush=True)
with ProcessPoolExecutor(max_workers=W) as ex: res=[r for r in ex.map(run,work) if r]
import pandas as pd
df=pd.DataFrame(res); df.to_csv(f"{B}/droppo_published_check.csv",index=False)
print("\n=== does the paper's Droppo fit move under rescaled bounds? ===")
print(f"{'arm':<5}{'variant':<10}{'SSE med':>12}{'holdout RMSE med':>19}{'RMSE range':>22}")
for arm in ("CO","NC"):
    for v in ("released","scaled"):
        s=df[(df.arm==arm)&(df.variant==v)]
        print(f"{arm:<5}{v:<10}{s.sse.median():>12.5g}{s.rmse_hold.median():>19.5f}"
              f"{f'[{s.rmse_hold.min():.4f}, {s.rmse_hold.max():.4f}]':>22}")
print("\n=== seed-paired CO-vs-NC win rate (NC better on holdout) ===")
for v in ("released","scaled"):
    co=df[(df.variant==v)&(df.arm=="CO")].set_index("seed").rmse_hold
    nc=df[(df.variant==v)&(df.arm=="NC")].set_index("seed").rmse_hold
    j=co.to_frame("co").join(nc.to_frame("nc"),how="inner")
    print(f"  {v:<9}: NC wins {int((j.co>j.nc).sum())}/{len(j)} = {100*float((j.co>j.nc).mean()):.1f}%")
print("\n=== VERDICT ===")
for arm in ("CO","NC"):
    a=df[(df.arm==arm)&(df.variant=="released")].rmse_hold.median()
    b=df[(df.arm==arm)&(df.variant=="scaled")].rmse_hold.median()
    print(f"  {arm}: median holdout RMSE {a:.5f} -> {b:.5f}   ({100*(b-a)/a:+.2f}%)"
          + ("   INERT" if abs(b-a)/a < 0.01 else "   MOVES"))
