#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════
EXHAUSTIVE TPP SUBSET ANALYSIS  -  BOUNDING-BOX NC GRID  (seeded)
════════════════════════════════════════════════════════════════════

Identical to subset_analysis_bbox.py except it takes a --seed N flag
which offsets the fit RNG and the NC-bbox RNG so the entire run is
reproducible AND varies consistently across seeds.

Seed-zero reproduces the original paper run exactly:
  FIT_SEED_BASE = 42      (was hardcoded 42 in original)
  bbox_seed = hash(...)   (args.seed XORs with 0, no change)

Any nonzero seed produces a fully independent realization:
  FIT_SEED = 42 + args.seed * 100003
  bbox_seed = hash(...) ^ (args.seed * 2654435761)

The holdout sampling seed is intentionally NOT varied - holdout
composition is part of the experimental definition, not a pipeline
knob.

Output layout:
  <out-dir>/seed_<N>/subset_bbox_<dataset>.csv

Aggregator reads <out-dir>/seed_*/subset_bbox_*.csv.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import hashlib
import multiprocessing as mp
import time
from itertools import combinations

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from scaling_laws.data import load_and_prepare, prepare_arrays
from scaling_laws.laws import (
    _fit_generic,
    kaplan_loss, chinchilla_loss, droppo_elibol_loss,
)
from incremental_coverage.coverage import (
    assign_tpp_bins,
)


# ════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════

DATASETS = {
    "Wikipedia":  "wikipedia.csv",
    "Cosmopedia": "cosmopedia.csv",
    "RedPajama":  "redpajama.csv",
    "Pes2o":      "pes2o.csv",
    "C4":         "c4.csv",
}

MODES = ["final", "first", "second"]
N_RESTARTS = 300

FIT_SEED_BASE      = 42
FIT_SEED_STRIDE    = 100003            # large prime, odd
BBOX_SEED_STRIDE   = 0x9E3779B1        # 32-bit golden-ratio constant
GLOBAL_SEED        = 0                 # set by --seed at runtime

HOLDOUT_FRACTION  = 0.70
HOLDOUT_SEED_BASE = 7777               # NOT varied across seeds

KAPPA_TARGETS = [1e4, 1e5, 1e6]

BETA_EFF_INDEX = {
    "Chinchilla":    4,
    "Kaplan":        3,
    "Droppo-Elibol": 4,
}

MIN_POINTS_FOR_FIT = {
    "Chinchilla":    5,
    "Kaplan":        4,
    "Droppo-Elibol": 6,
}


def current_fit_seed() -> int:
    return FIT_SEED_BASE + GLOBAL_SEED * FIT_SEED_STRIDE


def stable_hash(*args) -> int:
    """Deterministic 31-bit hash independent of PYTHONHASHSEED.
    Python's built-in hash() is salted per interpreter process, so it
    cannot be used to reproduce results across machines or across
    invocations. We use blake2b truncated to 31 bits."""
    h = hashlib.blake2b(digest_size=8)
    for a in args:
        h.update(repr(a).encode("utf-8"))
        h.update(b"\x00")
    return int.from_bytes(h.digest()[:4], "big") & 0x7FFFFFFF


def bbox_seed_for(ds_name: str, mode: str, mask_int: int) -> int:
    base = stable_hash(ds_name, mode, mask_int)
    return (base ^ (GLOBAL_SEED * BBOX_SEED_STRIDE)) & 0x7FFFFFFF


# ════════════════════════════════════════════════════════════════
# FIT WRAPPERS - now pull FIT_SEED dynamically
# ════════════════════════════════════════════════════════════════

def _fit_kaplan(N, D, L):
    bounds = [(1e3, 1e14), (1e3, 1e14), (0.01, 2.0), (0.01, 2.0)]
    def sampler(rng, L):
        return [10**rng.uniform(5,12), 10**rng.uniform(5,12),
                rng.uniform(0.05,1.5), rng.uniform(0.05,1.5)]
    return _fit_generic(kaplan_loss, N, D, L, bounds, sampler,
                        N_RESTARTS, seed=current_fit_seed())

def _fit_chinchilla(N, D, L):
    bounds = [(0.0,10.0),(1e-2,1e10),(0.01,2.0),(1e-2,1e10),(0.01,2.0)]
    def sampler(rng, L):
        return [rng.uniform(0.0, min(L)*0.9), 10**rng.uniform(0,8),
                rng.uniform(0.1,1.5), 10**rng.uniform(0,8),
                rng.uniform(0.1,1.5)]
    return _fit_generic(chinchilla_loss, N, D, L, bounds, sampler,
                        N_RESTARTS, seed=current_fit_seed())

def _fit_droppo_elibol(N, D, L):
    bounds = [(1e-6, min(L)*0.99),(1e3,1e14),(1e3,1e14),
              (0.01,2.0),(0.01,2.0),(0.01,2.0)]
    def sampler(rng, L):
        return [rng.uniform(0.01, min(L)*0.8), 10**rng.uniform(5,12),
                10**rng.uniform(5,12), rng.uniform(0.05,1.5),
                rng.uniform(0.05,1.5), rng.uniform(0.05,1.5)]
    return _fit_generic(droppo_elibol_loss, N, D, L, bounds, sampler,
                        N_RESTARTS, seed=current_fit_seed())

FITTERS = {
    "Chinchilla":    (_fit_chinchilla,    chinchilla_loss),
    "Kaplan":        (_fit_kaplan,        kaplan_loss),
    "Droppo-Elibol": (_fit_droppo_elibol, droppo_elibol_loss),
}


# ════════════════════════════════════════════════════════════════
# V_K  /  τ_K  /  REGIME
# ════════════════════════════════════════════════════════════════

def compute_VK(ks, beta_eff):
    vals = np.asarray(ks, dtype=float) ** (-beta_eff)
    return np.mean(vals**2) - np.mean(vals)**2

def compute_tauK(ks, beta_eff, kappa_target):
    K = len(ks)
    s2b = np.sum(np.asarray(ks, dtype=float) ** (-2 * beta_eff))
    return (K + s2b)**2 / (K**2 * kappa_target)

def classify_regime(VK, tauK):
    return "B" if VK >= tauK else "A"


# ════════════════════════════════════════════════════════════════
# CO HELPERS
# ════════════════════════════════════════════════════════════════

def discover_co_tpp_levels(co_train, mode):
    binned = assign_tpp_bins(co_train, mode=mode)
    return sorted(set(binned))

def select_co_runs(co_train, tpp_levels_to_keep, mode):
    binned = assign_tpp_bins(co_train, mode=mode)
    keep = set(tpp_levels_to_keep)
    return [r for r, t in zip(co_train, binned) if t in keep]


# ════════════════════════════════════════════════════════════════
# BOUNDING-BOX NC GRID SELECTION - unchanged
# ════════════════════════════════════════════════════════════════

def _priority_subsample(ring, needed, run_tpp, target_tpp_set,
                        covered_tpps, covered_any, rng):
    bucket_a, bucket_b, bucket_c = [], [], []
    for r in ring:
        t = run_tpp[id(r)]
        if t in target_tpp_set and t not in covered_tpps:
            bucket_a.append(r)
        elif t not in covered_any:
            bucket_b.append(r)
        else:
            bucket_c.append(r)
    rng.shuffle(bucket_a); rng.shuffle(bucket_b); rng.shuffle(bucket_c)
    return (bucket_a + bucket_b + bucket_c)[:needed]


def select_nc_bbox(target_n_points, target_tpps, run_tpp, grid_info,
                   bbox_seed=42):
    rng = np.random.RandomState(bbox_seed)
    target_tpp_set = set(target_tpps)

    m, n, grid = grid_info["m"], grid_info["n"], grid_info["grid"]
    if m < 2 or n < 2 or target_n_points == 0:
        return [], 0, 0, 0, [], 0, []

    cn = m // 2 if m % 2 == 1 else m // 2 + rng.choice([-1, 0])
    cd = n // 2 if n % 2 == 1 else n // 2 + rng.choice([-1, 0])

    selected_ids = set(); selected_list = []
    covered_tpps = set(); covered_any  = set()

    def _collect_new(nl, nh, dl, dh):
        out = []
        for ni in range(nl, nh + 1):
            for di in range(dl, dh + 1):
                for r in grid.get((ni, di), []):
                    if id(r) not in selected_ids:
                        out.append(r)
        return out

    def _absorb(runs):
        selected_list.extend(runs)
        for r in runs:
            selected_ids.add(id(r))
            t = run_tpp[id(r)]
            covered_any.add(t)
            if t in target_tpp_set:
                covered_tpps.add(t)

    def _score_ring(ring):
        new_target, new_any = set(), set()
        for r in ring:
            t = run_tpp[id(r)]
            if t not in covered_any:
                new_any.add(t)
            if t in target_tpp_set and t not in covered_tpps:
                new_target.add(t)
        return len(new_target), len(new_any)

    n_lo = max(0, cn - 1); n_hi = min(m - 1, n_lo + 1); n_lo = max(0, n_hi - 1)
    d_lo = max(0, cd - 1); d_hi = min(n - 1, d_lo + 1); d_lo = max(0, d_hi - 1)

    init_runs = _collect_new(n_lo, n_hi, d_lo, d_hi)

    if len(init_runs) > target_n_points:
        pick = _priority_subsample(init_runs, target_n_points, run_tpp,
                                     target_tpp_set, covered_tpps,
                                     covered_any, rng)
        _absorb(pick)
        t_hit = sorted(covered_tpps); a_hit = sorted(covered_any)
        return (selected_list, n_hi - n_lo + 1, d_hi - d_lo + 1,
                len(t_hit), t_hit, len(a_hit), a_hit)

    _absorb(init_runs)
    if len(selected_list) == target_n_points:
        t_hit = sorted(covered_tpps); a_hit = sorted(covered_any)
        return (selected_list, n_hi - n_lo + 1, d_hi - d_lo + 1,
                len(t_hit), t_hit, len(a_hit), a_hit)

    while len(selected_list) < target_n_points:
        candidates = []
        new_nl = max(0, n_lo - 1); new_nh = min(m - 1, n_hi + 1)
        if new_nl < n_lo or new_nh > n_hi:
            ring = []
            if new_nl < n_lo: ring += _collect_new(new_nl, new_nl, d_lo, d_hi)
            if new_nh > n_hi: ring += _collect_new(new_nh, new_nh, d_lo, d_hi)
            s1, s2 = _score_ring(ring)
            candidates.append((s1, s2, rng.random(), ring,
                                (new_nl, new_nh, d_lo, d_hi)))
        new_dl = max(0, d_lo - 1); new_dh = min(n - 1, d_hi + 1)
        if new_dl < d_lo or new_dh > d_hi:
            ring = []
            if new_dl < d_lo: ring += _collect_new(n_lo, n_hi, new_dl, new_dl)
            if new_dh > d_hi: ring += _collect_new(n_lo, n_hi, new_dh, new_dh)
            s1, s2 = _score_ring(ring)
            candidates.append((s1, s2, rng.random(), ring,
                                (n_lo, n_hi, new_dl, new_dh)))
        if not candidates:
            break
        candidates.sort(key=lambda c: (-c[0], -c[1], -c[2]))
        _, _, _, best_ring, best_bounds = candidates[0]
        if len(best_ring) == 0:
            n_lo, n_hi, d_lo, d_hi = best_bounds; continue
        space_left = target_n_points - len(selected_list)
        if len(best_ring) <= space_left:
            _absorb(best_ring); n_lo, n_hi, d_lo, d_hi = best_bounds
        else:
            pick = _priority_subsample(best_ring, space_left, run_tpp,
                                         target_tpp_set, covered_tpps,
                                         covered_any, rng)
            _absorb(pick)
            n_lo, n_hi, d_lo, d_hi = best_bounds
            break

    act_a = n_hi - n_lo + 1; act_b = d_hi - d_lo + 1
    t_hit = sorted(covered_tpps); a_hit = sorted(covered_any)
    return (selected_list, act_a, act_b,
            len(t_hit), t_hit, len(a_hit), a_hit)


# ════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ════════════════════════════════════════════════════════════════

_G = {}

def _init_worker(prepared_dict, holdout_dict, co_tpp_sorted,
                 nc_train_dict, n_restarts, modes_list, global_seed):
    _G["prepared"] = prepared_dict
    _G["holdouts"] = holdout_dict
    _G["co_tpp"]   = co_tpp_sorted
    _G["nc_train"] = nc_train_dict
    global N_RESTARTS, GLOBAL_SEED
    N_RESTARTS  = n_restarts
    GLOBAL_SEED = global_seed

    _G["nc_grid_info"] = {}
    for ds_name, nc_train in nc_train_dict.items():
        all_N = sorted(set(r["N"] for r in nc_train))
        all_D = sorted(set(r["D"] for r in nc_train))
        N_idx = {v: i for i, v in enumerate(all_N)}
        D_idx = {v: i for i, v in enumerate(all_D)}
        grid = {}
        for r in nc_train:
            grid.setdefault((N_idx[r["N"]], D_idx[r["D"]]), []).append(r)
        _G["nc_grid_info"][ds_name] = {
            "m": len(all_N), "n": len(all_D), "grid": grid,
        }

    _G["nc_run_tpp"] = {}
    _G["co_binned"]  = {}
    for ds_name in nc_train_dict:
        nc_train = nc_train_dict[ds_name]
        co_train = prepared_dict[ds_name]["co_train"]
        for mode in modes_list:
            nc_bins = assign_tpp_bins(nc_train, mode=mode)
            _G["nc_run_tpp"][(ds_name, mode)] = {
                id(r): t for r, t in zip(nc_train, nc_bins)
            }
            _G["co_binned"][(ds_name, mode)] = assign_tpp_bins(co_train,
                                                                  mode=mode)


# ════════════════════════════════════════════════════════════════
# WORKER
# ════════════════════════════════════════════════════════════════

def _worker(job):
    ds_name, law_name, mode, idx_tuple = job
    indices  = list(idx_tuple); K = len(indices)
    mask_int = sum(1 << i for i in indices)

    co_train_full = _G["prepared"][ds_name]["co_train"]
    co_tpp_all    = _G["co_tpp"][(ds_name, mode)]

    N_h, D_h, L_h = _G["holdouts"][(ds_name, mode)]
    if len(L_h) == 0:
        return []

    co_tpp_subset = [co_tpp_all[i] for i in indices]
    keep = set(co_tpp_subset)
    co_bins = _G["co_binned"][(ds_name, mode)]
    co_runs = [r for r, t in zip(co_train_full, co_bins) if t in keep]
    n_co = len(co_runs)
    if n_co == 0:
        return []

    # Seeded bbox construction
    bbox_seed = bbox_seed_for(ds_name, mode, mask_int)
    run_tpp   = _G["nc_run_tpp"][(ds_name, mode)]
    grid_info = _G["nc_grid_info"][ds_name]

    (nc_runs, bbox_a, bbox_b,
     n_target_hit, target_hit_list,
     n_total_tpp, all_tpp_list) = select_nc_bbox(
        target_n_points=n_co,
        target_tpps=co_tpp_subset,
        run_tpp=run_tpp,
        grid_info=grid_info,
        bbox_seed=bbox_seed,
    )

    fit_fn, loss_fn = FITTERS[law_name]
    min_pts = MIN_POINTS_FOR_FIT[law_name]
    results = []

    for design_label, runs, extra in [
        ("CO", co_runs, {
            "co_tpp_values":      str(co_tpp_subset),
            "bbox_a": np.nan,     "bbox_b": np.nan,
            "n_target_tpp_covered": K,
            "target_tpp_hit":     str(co_tpp_subset),
            "tpp_coverage_pct":   100.0,
            "n_total_tpp":        K,
            "all_tpp_hit":        str(co_tpp_subset),
        }),
        ("NC_BBOX", nc_runs, {
            "co_tpp_values":      "",
            "bbox_a":             bbox_a,
            "bbox_b":             bbox_b,
            "n_target_tpp_covered": n_target_hit,
            "target_tpp_hit":     str(target_hit_list),
            "tpp_coverage_pct":   100.0 * n_target_hit / K if K else 0.0,
            "n_total_tpp":        n_total_tpp,
            "all_tpp_hit":        str(all_tpp_list),
        }),
    ]:
        row = {
            "dataset":               ds_name,
            "law":                   law_name,
            "mode":                  mode,
            "design":                design_label,
            "subset_mask":           mask_int,
            "subset_size":           K,
            "n_runs":                len(runs),
            "seed":                  GLOBAL_SEED,
            "target_tpps":           str(co_tpp_subset),
            "co_tpp_values":         extra["co_tpp_values"],
            "bbox_a":                extra["bbox_a"],
            "bbox_b":                extra["bbox_b"],
            "n_target_tpp_covered":  extra["n_target_tpp_covered"],
            "target_tpp_hit":        extra["target_tpp_hit"],
            "tpp_coverage_pct":      extra["tpp_coverage_pct"],
            "n_total_tpp":           extra["n_total_tpp"],
            "all_tpp_hit":           extra["all_tpp_hit"],
            "r2_unified":            np.nan,
            "rmse_unified":          np.nan,
        }

        row["V_K"] = np.nan
        row["beta_eff_fitted"] = np.nan
        for kt in KAPPA_TARGETS:
            tag = f"{kt:.0e}"
            row[f"tau_K_{tag}"]  = np.nan
            row[f"regime_{tag}"] = ""

        N_tr, D_tr, L_tr = prepare_arrays(runs, mode)
        if len(L_tr) < min_pts:
            results.append(row); continue

        try:
            params, sse, success = fit_fn(N_tr, D_tr, L_tr)
            if params is None:
                results.append(row); continue

            pred_h = loss_fn(N_h, D_h, *params)
            row["r2_unified"]   = r2_score(L_h, pred_h)
            row["rmse_unified"] = np.sqrt(np.mean((L_h - pred_h) ** 2))

            if design_label == "CO":
                bidx = BETA_EFF_INDEX[law_name]
                beta_eff = params[bidx]
                row["beta_eff_fitted"] = beta_eff
                VK = compute_VK(co_tpp_subset, beta_eff)
                row["V_K"] = VK
                for kt in KAPPA_TARGETS:
                    tag = f"{kt:.0e}"
                    tK = compute_tauK(co_tpp_subset, beta_eff, kt)
                    row[f"tau_K_{tag}"] = tK
                    row[f"regime_{tag}"] = classify_regime(VK, tK)

            row["fitted_params"] = str(list(params))

        except Exception as e:
            row["error"] = str(e)

        results.append(row)

    return results


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Exhaustive TPP subset analysis (seeded bbox)")
    ap.add_argument("--seed",             type=int,   required=True,
                    help="Integer seed offset. 0 reproduces the "
                         "original bbox run exactly.")
    ap.add_argument("--data-dir",         default=None,
                    help="Directory with per-dataset CSVs (defaults to "
                         "$DATA_DIR if set, else current directory). "
                         "These CSVs come from get_results.py.")
    ap.add_argument("--out-dir",          default="subset_bbox_seeds",
                    help="Base output directory. Per-seed results go "
                         "into <out-dir>/seed_<N>/")
    ap.add_argument("--n-workers",        type=int,   default=None)
    ap.add_argument("--datasets",         nargs="*",  default=None)
    ap.add_argument("--laws",             nargs="*",  default=None)
    ap.add_argument("--modes",            nargs="*",  default=None)
    ap.add_argument("--n-restarts",       type=int,   default=300)
    ap.add_argument("--min-subset-size",  type=int,   default=1)
    ap.add_argument("--max-subset-size",  type=int,   default=None)
    ap.add_argument("--holdout-fraction", type=float, default=0.70)
    args = ap.parse_args()

    # Resolve data-dir: explicit flag > $DATA_DIR > current dir
    if args.data_dir is None:
        args.data_dir = os.environ.get("DATA_DIR", ".")

    global N_RESTARTS, HOLDOUT_FRACTION, GLOBAL_SEED
    N_RESTARTS       = args.n_restarts
    HOLDOUT_FRACTION = args.holdout_fraction
    GLOBAL_SEED      = args.seed

    n_workers    = args.n_workers or mp.cpu_count()
    ds_to_run    = args.datasets  or list(DATASETS.keys())
    laws_to_run  = args.laws      or list(FITTERS.keys())
    modes_to_run = args.modes     or MODES

    seed_out_dir = os.path.join(args.out_dir, f"seed_{args.seed}")
    os.makedirs(seed_out_dir, exist_ok=True)

    print("╔══════════════════════════════════════════════════════════╗")
    print(f"║  EXHAUSTIVE TPP SUBSET - BBOX NC GRID  (SEED {args.seed:>3d})       ║")
    print(f"║  Workers: {n_workers:<4d}  Restarts: {N_RESTARTS:<4d}                   ║")
    print(f"║  Fit seed: {current_fit_seed():<10d}                           ║")
    print(f"║  Data dir: {args.data_dir:<46s}║")
    print(f"║  Out dir: {seed_out_dir:<47s}║")
    print("╚══════════════════════════════════════════════════════════╝")

    # [1/4] Load
    print("\n[1/4] Loading datasets...")
    prepared_raw = {}
    for name in ds_to_run:
        fpath = os.path.join(args.data_dir, DATASETS[name])
        if not os.path.exists(fpath):
            print(f"  ⚠  Skipping {name}: {fpath} not found"); continue
        prepared_raw[name] = load_and_prepare(fpath, name)
        print(f"  ✓ {name}")
    if not prepared_raw:
        print("No datasets loaded."); return

    # [2/4] Discover
    print("\n[2/4] Discovering CO TPP levels & NC grid geometry...")
    co_tpp_sorted  = {}; holdout_arrays = {}
    prepared_light = {}; nc_train_dict  = {}

    for name, data in prepared_raw.items():
        nc_train = data.noncolinear_train
        nc_train_dict[name] = nc_train
        prepared_light[name] = {"co_train": data.colinear_train}

        all_N_nc = sorted(set(r["N"] for r in nc_train))
        all_D_nc = sorted(set(r["D"] for r in nc_train))
        print(f"  {name}: NC grid {len(all_N_nc)}×{len(all_D_nc)} "
              f"({len(nc_train)} runs)")

        for mode in modes_to_run:
            co_tpps = discover_co_tpp_levels(data.colinear_train, mode)
            co_tpp_sorted[(name, mode)] = co_tpps
            N_h, D_h, L_h = prepare_arrays(data.unified_holdout, mode)
            n_full = len(L_h)
            n_sample = max(1, int(HOLDOUT_FRACTION * n_full))
            # Holdout seed NOT varied by global seed
            ho_seed = (HOLDOUT_SEED_BASE +
                        stable_hash(name, mode) % 100_000) & 0x7FFFFFFF
            idx = np.random.RandomState(ho_seed).choice(
                n_full, n_sample, replace=False)
            holdout_arrays[(name, mode)] = (N_h[idx], D_h[idx], L_h[idx])

    # [3/4] Jobs
    print("\n[3/4] Enumerating subsets...")
    all_jobs = []; cached_masks = {}

    for name in prepared_light:
        csv_path = os.path.join(seed_out_dir, f"subset_bbox_{name}.csv")
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path)
            cached = set()
            for _, row in existing.iterrows():
                cached.add((row["law"], str(row.get("mode", "final")),
                            int(row["subset_mask"])))
            cached_masks[name] = cached
            print(f"  {name}: {len(cached)//2} cached combos")
        else:
            cached_masks[name] = set()

        n_new = 0
        for mode in modes_to_run:
            K = len(co_tpp_sorted[(name, mode)])
            max_k = min(args.max_subset_size or K, K)
            for size in range(args.min_subset_size, max_k + 1):
                for idx_combo in combinations(range(K), size):
                    mask_int = sum(1 << i for i in idx_combo)
                    for law in laws_to_run:
                        if (law, mode, mask_int) in cached_masks[name]:
                            continue
                        all_jobs.append((name, law, mode, idx_combo))
                        n_new += 1
        print(f"  {name}: {n_new} new jobs")

    total_fits = len(all_jobs) * 2
    print(f"\n  Total new jobs: {len(all_jobs)}  ({total_fits} fits)")
    if total_fits == 0:
        print("  All cached!"); return

    # [4/4] Run
    print(f"\n[4/4] Running {len(all_jobs)} jobs across {n_workers} workers...")
    t0 = time.time()
    buffers = {name: [] for name in prepared_light}
    flush_every = 500

    def flush(ds):
        if not buffers[ds]:
            return 0
        csv_path = os.path.join(seed_out_dir, f"subset_bbox_{ds}.csv")
        df_new = pd.DataFrame(buffers[ds])
        write_header = not os.path.exists(csv_path)
        df_new.to_csv(csv_path, mode="a", header=write_header, index=False)
        cnt = len(buffers[ds]); buffers[ds] = []; return cnt

    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(prepared_light, holdout_arrays, co_tpp_sorted,
                  nc_train_dict, N_RESTARTS, modes_to_run,
                  GLOBAL_SEED),
    ) as pool:
        done = 0
        for result_rows in pool.imap_unordered(_worker, all_jobs,
                                                 chunksize=16):
            if result_rows:
                buffers[result_rows[0]["dataset"]].extend(result_rows)
            done += 1
            if done % flush_every == 0:
                for name in buffers:
                    flush(name)
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(all_jobs) - done) / rate if rate > 0 else 0
                print(f"  [seed {args.seed}]  [{done:>6d}/{len(all_jobs)}]  "
                      f"{rate:.1f} jobs/s  ETA {eta/60:.1f}min", flush=True)

    for name in buffers:
        flush(name)
    elapsed = time.time() - t0
    print(f"\n  Seed {args.seed} done in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()