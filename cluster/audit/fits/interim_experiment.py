#!/usr/bin/env python3
"""
Interim run of the actual experiment on cells finished so far.

Protocol care taken:
  * budget-matching: arms are balanced PER N-ROW (take min(n_CO,n_NC) from each
    row, lowest-|k-median| kept deterministically) so neither design gets a
    cardinality edge -- the published design's defining constraint.
  * seed-paired: same fit seed for both designs, repo fitters, 3 laws.
  * ONE pooled holdout, scored identically for both.
  * per-stratum split by N: the scale-sensitivity question itself.
Caveats printed at the end. This is interim, not the final number.
"""
import sys
import os
import numpy as np
from collections import defaultdict
import wiki300m_paths  # the paper's code on sys.path, and the run directory
from get_results.get_results import extract_from_local
from scaling_laws.data import load_and_prepare, prepare_arrays
import seed_variance as sv

B = wiki300m_paths.RUN_DIR
SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
RESTARTS = int(sys.argv[2]) if len(sys.argv) > 2 else 80

out = f"{B}/interim/extracted"; os.makedirs(out, exist_ok=True)
extract_from_local(f"{B}/shared/training_metrics", out)
data = load_and_prepare(f"{out}/wikipedia.csv", dataset_name="WIKI300M", verbose=False)
Nc, Dc, Lc = prepare_arrays(data.colinear_train, mode="first")
Nn, Dn, Ln = prepare_arrays(data.noncolinear_train, mode="first")
Nh, Dh, Lh = prepare_arrays(data.unified_holdout, mode="first")

# ---- balance arms per N row -------------------------------------------------
def by_row(N, D, L):
    d = defaultdict(list)
    for i in range(len(N)):
        d[int(round(N[i] / 1e5))].append(i)
    return d
rc, rn = by_row(Nc, Dc, Lc), by_row(Nn, Dn, Ln)
keep_c, keep_n = [], []
for row in sorted(set(rc) & set(rn)):
    ic, inn = rc[row], rn[row]
    k = min(len(ic), len(inn))
    # deterministic: keep the k cells closest to that row's median log-D
    ic = sorted(ic, key=lambda i: abs(np.log(Dc[i]) - np.median(np.log(Dc[ic]))))[:k]
    inn = sorted(inn, key=lambda i: abs(np.log(Dn[i]) - np.median(np.log(Dn[inn]))))[:k]
    keep_c += ic; keep_n += inn
Nc, Dc, Lc = Nc[keep_c], Dc[keep_c], Lc[keep_c]
Nn, Dn, Ln = Nn[keep_n], Dn[keep_n], Ln[keep_n]
print(f"\nbalanced arms: CO {len(Lc)} = NC {len(Ln)} cells over "
      f"{len(set(np.round(Nc/1e5).astype(int)))} N rows;  pooled holdout {len(Lh)}")

ROWS = []   # per (stratum, law, seed) raw errors, for CIs + size interaction

def _one(job):
    """Pure function of its args: (law, seed) -> holdout RMSEs. Deterministic."""
    law, s, Nc, Dc, Lc, Nn, Dn, Ln, Nh, Dh, Lh = job
    fitter, lossfn = sv._SEED_FITTERS[law]
    pc, _, _ = fitter(Nc, Dc, Lc, n_restarts=RESTARTS, seed=1000 + s)
    pn, _, _ = fitter(Nn, Dn, Ln, n_restarts=RESTARTS, seed=1000 + s)
    if pc is None or pn is None:
        return (law, s, None, None)
    eC = float(np.sqrt(np.mean((lossfn(Nh, Dh, *pc) - Lh) ** 2)))
    eN = float(np.sqrt(np.mean((lossfn(Nh, Dh, *pn) - Lh) ** 2)))
    return (law, s, eC, eN)

def tally(Nc, Dc, Lc, Nn, Dn, Ln, Nh, Dh, Lh, label):
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(law, s, Nc, Dc, Lc, Nn, Dn, Ln, Nh, Dh, Lh)
            for law in sv._SEED_FITTERS for s in range(SEEDS)]
    workers = min(len(jobs), max(1, len(os.sched_getaffinity(0)) - 2))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_one, jobs))          # map preserves input order
    per = {}; tot = n_all = 0
    for law in sv._SEED_FITTERS:
        wins, ratios = 0, []
        for (l, s, eC, eN) in res:
            if l != law or eC is None: continue
            wins += eC > eN; ratios.append(eC / eN)
            ROWS.append({"stratum": label, "law": law, "seed": s,
                         "rmse_co": eC, "rmse_nc": eN,
                         "n_co": int(len(Lc)), "n_nc": int(len(Ln)),
                         "logN_co_gm": float(np.mean(np.log10(Nc))),
                         "logN_nc_gm": float(np.mean(np.log10(Nn)))})
        r = np.array(ratios); per[law] = (wins, len(r), np.median(r) if len(r) else np.nan)
        tot += wins; n_all += len(r)
    print(f"\n--- {label} ---  [{workers} workers]")
    for law, (w, n, m) in per.items():
        print(f"  {law:14s} NC {w}/{n} ({100*w/max(n,1):3.0f}%)  median RMSE_CO/RMSE_NC {m:.2f}x")
    print(f"  POOLED  {tot}/{n_all} = {100*tot/max(n_all,1):.1f}% NC win rate")
    return tot, n_all

tally(Nc, Dc, Lc, Nn, Dn, Ln, Nh, Dh, Lh, "ALL ROWS (interim, balanced)")

# ---- per-stratum: the scale-sensitivity question --------------------------
med = np.median(np.concatenate([Nc, Nn]))
for lab, sel_c, sel_n in (
    ("LOW-N stratum  (N <= %.0fM)" % (med / 1e6), Nc <= med, Nn <= med),
    ("HIGH-N stratum (N >  %.0fM)" % (med / 1e6), Nc > med, Nn > med)):
    if sel_c.sum() < 6 or sel_n.sum() < 6:
        print(f"\n--- {lab} --- too few cells ({sel_c.sum()},{sel_n.sum()}), skipped")
        continue
    tally(Nc[sel_c], Dc[sel_c], Lc[sel_c], Nn[sel_n], Dn[sel_n], Ln[sel_n],
          Nh, Dh, Lh, lab + f"  [CO {sel_c.sum()} = NC {sel_n.sum()}]")

print("""
CAVEATS - interim, not the final number:
  * arms are ~60% complete and were balanced by subsampling, so both designs
    are fit on fewer points than the design specifies (weakens both fits).
  * the 380.6M row and most of the seed-43/44 replicates are still queued.
  * fit-seed pairs only; training-seed replication is what the final run adds.
""")

import csv
with open(f"{B}/rows.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(ROWS[0].keys())); w.writeheader(); w.writerows(ROWS)
print(f"wrote {B}/rows.csv  ({len(ROWS)} rows)")
