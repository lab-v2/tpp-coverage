#!/usr/bin/env python3
"""
Bootstrap CIs on per-law log-weighted winrate averaging across
datasets, treating each seed as a unit.

Reads <seed-dir>/seed_*/subset_bbox_*.csv produced by
subset_analysis_bbox_seeded.py (typically launched via dispatch_seeds.sh)
and computes per-law mean ± 95% bootstrap CI for the κ-averaged winrate.

Usage
─────
  # Default - reads from ./subset_bbox_seeds/ (matches dispatch_seeds.sh default)
  python per_law_winrate_ci.py --out-dir subset_bbox_seeds/aggregated

  # Explicit seed dir
  python per_law_winrate_ci.py \\
    --seed-dir subset_bbox_seeds \\
    --out-dir  subset_bbox_seeds/aggregated
"""

import argparse
import re
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd


LIT_BETA_EFF = {
    "Chinchilla":    0.28,
    "Kaplan":        0.095,
    "Droppo-Elibol": 0.08,
}

KT_LOG_MIN = 1.0
KT_LOG_MAX = 1e9
N_POINTS   = 2000
N_BOOT     = 5000


_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_NP_WRAPPER_RE = re.compile(r"np\.\w+\(([^)]*?)\)")

def parse_tpp_list(s):
    if not isinstance(s, str) or not s.strip():
        return None
    clean = _NP_WRAPPER_RE.sub(r"\1", s)
    nums = _NUM_RE.findall(clean)
    if not nums:
        return None
    try:
        return np.array([float(x) for x in nums], dtype=float)
    except Exception:
        return None


def per_seed_log_avg_winrate(g: pd.DataFrame, law: str, kappa_targets):
    """Returns this (seed, cell) slice's log-averaged winrate over
    the alive κ range."""
    beta = LIT_BETA_EFF[law]
    tpps_list = [parse_tpp_list(s) for s in g["target_tpps"].values]
    good = np.array([t is not None for t in tpps_list])
    if good.sum() == 0: return np.nan

    V_Ks = np.empty(good.sum())
    noK2 = np.empty(good.sum())
    i = 0
    for t, is_good in zip(tpps_list, good):
        if not is_good: continue
        K = len(t)
        k_neg = t ** (-beta)
        V_Ks[i] = float(np.mean(k_neg**2) - np.mean(k_neg)**2)
        noK2[i] = (K + float(np.sum(t ** (-2*beta))))**2 / (K*K)
        i += 1

    rmse_co = g["rmse_co"].values[good]
    rmse_nc = g["rmse_nc"].values[good]
    nc_wins = (rmse_nc < rmse_co).astype(np.int8)

    n_kt = len(kappa_targets)
    winrates = np.full(n_kt, np.nan)
    for idx, kt in enumerate(kappa_targets):
        tau_K = noK2 / kt
        in_A = V_Ks < tau_K
        a_size = int(in_A.sum())
        if a_size == 0: break
        winrates[idx] = 100.0 * nc_wins[in_A].sum() / a_size
    alive = np.where(np.isfinite(winrates))[0]
    return float(np.mean(winrates[alive])) if len(alive) else np.nan


def load_all_seeds(seed_dir: Path):
    frames = []
    for sd in sorted(seed_dir.glob("seed_*")):
        m = re.match(r"seed_(\d+)$", sd.name)
        if not m: continue
        seed = int(m.group(1))
        for csv in sorted(sd.glob("subset_bbox_*.csv")):
            try: df = pd.read_csv(csv)
            except Exception: continue
            if len(df) == 0: continue
            co = df[df["design"] == "CO"][["dataset","law","mode",
                "subset_mask","target_tpps","rmse_unified"]].rename(
                columns={"rmse_unified":"rmse_co"})
            nc = df[df["design"] == "NC_BBOX"][["dataset","law","mode",
                "subset_mask","rmse_unified"]].rename(
                columns={"rmse_unified":"rmse_nc"})
            key = ["dataset","law","mode","subset_mask"]
            m2 = co.merge(nc, on=key, how="inner")
            m2["seed"] = seed
            frames.append(m2)
    if not frames:
        raise SystemExit(
            f"No data found under {seed_dir}/seed_*/subset_bbox_*.csv\n"
            f"Did you run dispatch_seeds.sh first?")
    return pd.concat(frames, ignore_index=True).dropna(
        subset=["rmse_co","rmse_nc"]).reset_index(drop=True)


def worker(args):
    law, dataset, mode, g, kappa_targets = args
    seeds = sorted(g["seed"].unique())
    out = {}
    for seed in seeds:
        out[seed] = per_seed_log_avg_winrate(
            g[g["seed"] == seed], law, kappa_targets)
    return (law, dataset, mode, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-dir", type=Path,
                    default=Path("subset_bbox_seeds"),
                    help="Parent dir containing seed_*/ subdirs "
                         "(default: subset_bbox_seeds - matches "
                         "dispatch_seeds.sh default).")
    ap.add_argument("--out-dir",  required=True, type=Path)
    ap.add_argument("--mode",     default="first")
    ap.add_argument("--n-workers", type=int, default=8)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    kappa_targets = np.logspace(np.log10(KT_LOG_MIN),
                                 np.log10(KT_LOG_MAX), N_POINTS)
    df = load_all_seeds(args.seed_dir)
    df = df[df["mode"] == args.mode]

    cells = []
    for (law, dataset, mode), g in df.groupby(["law","dataset","mode"]):
        cells.append((law, dataset, mode, g, kappa_targets))

    print(f"Analyzing {len(cells)} cells across "
          f"{df['seed'].nunique()} seeds…")
    with Pool(processes=args.n_workers) as pool:
        results = pool.map(worker, cells)

    # Build (seed, law, dataset) → log-avg winrate
    tidy = []
    for law, dataset, mode, per_seed in results:
        for seed, wr in per_seed.items():
            tidy.append({"seed": seed, "law": law, "dataset": dataset,
                          "log_avg_winrate": wr})
    tidy_df = pd.DataFrame(tidy)

    # Per-law: first average across datasets (within a seed),
    # then bootstrap CI across seeds
    rng = np.random.default_rng(20260423)
    per_law_ci = []
    per_dataset_ci = []
    for law in sorted(tidy_df["law"].unique()):
        sub = tidy_df[tidy_df["law"] == law]
        # Per-seed, average across 5 datasets
        by_seed = sub.groupby("seed")["log_avg_winrate"].mean().values
        by_seed = by_seed[np.isfinite(by_seed)]
        if len(by_seed) == 0:
            continue
        idx = rng.integers(0, len(by_seed), size=(N_BOOT, len(by_seed)))
        boot = np.mean(by_seed[idx], axis=1)
        mean = float(np.mean(by_seed))
        lo   = float(np.percentile(boot, 2.5))
        hi   = float(np.percentile(boot, 97.5))
        per_law_ci.append({"law": law, "mean": mean, "lo": lo, "hi": hi,
                            "n_seeds": len(by_seed)})

        # Per-dataset: bootstrap CI across seeds
        for dataset in sorted(sub["dataset"].unique()):
            vals = sub[sub["dataset"] == dataset]["log_avg_winrate"].values
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0: continue
            idx = rng.integers(0, len(vals), size=(N_BOOT, len(vals)))
            boot = np.mean(vals[idx], axis=1)
            per_dataset_ci.append({
                "law": law, "dataset": dataset,
                "mean": float(np.mean(vals)),
                "lo":   float(np.percentile(boot, 2.5)),
                "hi":   float(np.percentile(boot, 97.5)),
                "n_seeds": len(vals),
            })

    law_df = pd.DataFrame(per_law_ci)
    cell_df = pd.DataFrame(per_dataset_ci)
    law_df.to_csv(args.out_dir / "per_law_ci.csv", index=False)
    cell_df.to_csv(args.out_dir / "per_dataset_ci.csv", index=False)

    print("\n" + "=" * 95)
    print(f"  PER-LAW LOG-WEIGHTED AVG WINRATE (mean ± 95% bootstrap CI, "
          f"n_seeds={int(law_df['n_seeds'].iloc[0])})")
    print("=" * 95)
    print(f"  {'Law':<16s}  {'Dataset':<14s}  "
          f"{'Mean winrate':>15s}  {'95% CI':>20s}")
    print("  " + "─" * 91)
    for law in sorted(law_df["law"].unique()):
        cells_ = cell_df[cell_df["law"] == law].sort_values("dataset")
        for _, r in cells_.iterrows():
            ci = f"[{r['lo']:.2f}, {r['hi']:.2f}]"
            print(f"  {r['law']:<16s}  {r['dataset']:<14s}  "
                  f"{r['mean']:>14.2f}%  {ci:>20s}")
        law_row = law_df[law_df["law"] == law].iloc[0]
        ci = f"[{law_row['lo']:.2f}, {law_row['hi']:.2f}]"
        print(f"  {'':<16s}  {'── AVG ──':<14s}  "
              f"{law_row['mean']:>14.2f}%  {ci:>20s}")
        print("  " + "─" * 91)
    print(f"\n  Artifacts: {args.out_dir}/per_{{law,dataset}}_ci.csv")


if __name__ == "__main__":
    main()