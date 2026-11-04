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


# ============ split-holdout + windowed size-interaction (additive) ============
# Fits are identical to interim_experiment.py (same fitters, seeds, restarts,
# same balanced arms). Each fit is scored on THREE holdouts instead of one:
#   pooled = data.unified_holdout, on-ray = colinear_holdout, off-ray = holdout
# The question is the off-ray gap specifically; the paper reports 72.1% on-ray
# vs 98.7% off-ray (Table 14), so a pooled number understates the effect.
from concurrent.futures import ProcessPoolExecutor
import csv

Non, Don, Lon = prepare_arrays(data.colinear_holdout, mode="first")   # on-ray
Noff, Doff, Loff = prepare_arrays(data.noncolinear_holdout, mode="first")         # off-ray
HOLDOUTS = {"pooled": (Nh, Dh, Lh), "on_ray": (Non, Don, Lon), "off_ray": (Noff, Doff, Loff)}
print(f"holdouts: pooled {len(Lh)}, on-ray {len(Lon)}, off-ray {len(Loff)}")

# N rows present in the balanced arms, ascending
rows_all = sorted(set(np.round(Nc/1e5).astype(int)))
print(f"N rows in balanced arms: {len(rows_all)} -> {[r/10 for r in rows_all]} (M)")

def job_list():
    jobs = []
    # strata: all, low, high (median split, same as the script), plus sliding
    # windows of 3 consecutive N rows for the interaction regression
    med = np.median(np.concatenate([Nc, Nn]))
    specs = [("all", np.ones_like(Nc, bool), np.ones_like(Nn, bool)),
             ("low", Nc <= med, Nn <= med),
             ("high", Nc > med, Nn > med)]
    W = 3
    for i in range(len(rows_all) - W + 1):
        keep = set(rows_all[i:i+W])
        mc = np.array([int(round(x/1e5)) in keep for x in Nc])
        mn = np.array([int(round(x/1e5)) in keep for x in Nn])
        specs.append((f"win{i}", mc, mn))
    for tag, mc, mn in specs:
        if mc.sum() < 6 or mn.sum() < 6: continue
        for law in sv._SEED_FITTERS:
            for s in range(SEEDS):
                jobs.append((tag, law, s, Nc[mc], Dc[mc], Lc[mc], Nn[mn], Dn[mn], Ln[mn]))
    return jobs

def run_job(j):
    tag, law, s, nc, dc, lc, nn, dn, ln = j
    fitter, lossfn = sv._SEED_FITTERS[law]
    pc, _, _ = fitter(nc, dc, lc, n_restarts=RESTARTS, seed=1000 + s)
    pn, _, _ = fitter(nn, dn, ln, n_restarts=RESTARTS, seed=1000 + s)
    if pc is None or pn is None: return None
    out = {"stratum": tag, "law": law, "seed": s, "n_cells": int(len(lc)),
           "logN_gm": float(np.mean(np.log10(np.concatenate([nc, nn]))))}
    for hk, (HN, HD, HL) in HOLDOUTS.items():
        if len(HL) == 0: continue
        eC = float(np.sqrt(np.mean((lossfn(HN, HD, *pc) - HL) ** 2)))
        eN = float(np.sqrt(np.mean((lossfn(HN, HD, *pn) - HL) ** 2)))
        out[f"co_{hk}"] = eC; out[f"nc_{hk}"] = eN
    return out

jobs = job_list()
workers = max(1, len(os.sched_getaffinity(0)) - 2)
print(f"{len(jobs)} fit-pairs on {workers} workers")
with ProcessPoolExecutor(max_workers=workers) as ex:
    res = [r for r in ex.map(run_job, jobs) if r]
with open(f"{B}/split_rows.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)
print(f"wrote {B}/split_rows.csv ({len(res)} rows)")
