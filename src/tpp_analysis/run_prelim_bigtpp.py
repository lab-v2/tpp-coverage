#!/usr/bin/env python3
"""
Preliminary Chinchilla-only winrate analysis for the BigTPP Wikipedia partition.

This is a compute-limited preliminary experiment, not a full main-table replication.
Scope deliberately narrowed:
  - Only the Chinchilla scaling law (not Kaplan / Droppo-Elibol / Parametric-Epoch)
  - Only first-epoch fits (`mode="first"` - BigTPP has 1 epoch of data)
  - Only the unified holdout
  - Reports Wilson 95% CI on the NC win rate

Data source: extracted_losses/wikipedia_bigtpp.csv, produced by
download_and_extract.py with --include-special.

Usage:
    python run_prelim_bigtpp.py --data-dir extracted_losses/ --n-workers 32
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
# Config
# ============================================================

DATASET_NAME = "Wikipedia_BigTPP"
DATASET_CSV  = "wikipedia_bigtpp.csv"
LAW          = "Chinchilla"
MODE         = "first"                # BigTPP only has 1 epoch

DEFAULT_N_SEEDS    = 30
DEFAULT_N_RESTARTS = 100
DEFAULT_BASE_SEED  = 0
SEED_STRIDE        = 137              # match BF16 protocol for consistency

CHINCHILLA_PARAM_NAMES = ["E", "A", "alpha", "B", "beta"]


# ============================================================
# Seed-aware Chinchilla fitter
# ============================================================

def _fit_chinchilla_seed(N, D, L, n_restarts=300, seed=42):
    from scaling_laws.laws import _fit_generic, chinchilla_loss
    bounds = [(0.0, 10.0), (1e-2, 1e10), (0.01, 2.0), (1e-2, 1e10), (0.01, 2.0)]
    def sampler(rng, L):
        return [
            rng.uniform(0.0, min(L) * 0.9),
            10 ** rng.uniform(0, 8),
            rng.uniform(0.1, 1.5),
            10 ** rng.uniform(0, 8),
            rng.uniform(0.1, 1.5),
        ]
    return _fit_generic(chinchilla_loss, N, D, L, bounds, sampler, n_restarts, seed=seed)


# ============================================================
# Worker: one (design, seed) combo
# ============================================================

def _run_seed(args):
    """Fit Chinchilla once for a given (design, seed). Returns one row."""
    design_label, csv_path, seed_idx, base_seed, n_restarts = args
    from scaling_laws.data import load_and_prepare, prepare_arrays
    from scaling_laws.laws import chinchilla_loss

    try:
        data = load_and_prepare(csv_path, DATASET_NAME, verbose=False)
    except Exception as e:
        return {"error": f"load: {e}", "design": design_label, "seed": seed_idx}

    train_runs = (data.colinear_train if design_label == "CO"
                  else data.noncolinear_train)
    if not train_runs:
        return {"error": f"no {design_label} train runs",
                "design": design_label, "seed": seed_idx}

    try:
        N_arr, D_arr, L_arr = prepare_arrays(train_runs, mode=MODE)
    except Exception as e:
        return {"error": f"prepare: {e}", "design": design_label, "seed": seed_idx}
    if len(N_arr) < 5:
        return {"error": f"only {len(N_arr)} training points",
                "design": design_label, "seed": seed_idx}

    actual_seed = base_seed + seed_idx * SEED_STRIDE
    try:
        params, sse, success = _fit_chinchilla_seed(
            N_arr, D_arr, L_arr, n_restarts=n_restarts, seed=actual_seed,
        )
    except Exception as e:
        return {"error": f"fit: {e}", "design": design_label, "seed": seed_idx}
    if params is None:
        return {"error": "fit returned None", "design": design_label, "seed": seed_idx}

    row = {
        "dataset": DATASET_NAME,
        "law": LAW,
        "mode": MODE,
        "design": design_label,
        "seed": seed_idx,
    }

    # Train metrics
    try:
        pred_tr = chinchilla_loss(N_arr, D_arr, *params)
        row["r2_train"] = float(r2_score(L_arr, pred_tr))
        row["rmse_train"] = float(np.sqrt(mean_squared_error(L_arr, pred_tr)))
    except Exception:
        pass

    # Unified holdout only - preliminary scope
    try:
        hN, hD, hL = prepare_arrays(data.unified_holdout, mode=MODE)
        if len(hL) >= 2:
            pred_h = chinchilla_loss(hN, hD, *params)
            row["r2_unified"] = float(r2_score(hL, pred_h))
            row["rmse_unified"] = float(np.sqrt(mean_squared_error(hL, pred_h)))
    except Exception:
        pass

    # Named param columns (for forward-compat with appendix table generators)
    for pi, pval in enumerate(params):
        row[f"param_{pi}"] = float(pval)
        if pi < len(CHINCHILLA_PARAM_NAMES):
            row[f"param_{CHINCHILLA_PARAM_NAMES[pi]}"] = float(pval)

    return row


# ============================================================
# Wilson CI for winrate
# ============================================================

def _wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p_hat = k / n
    denom = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
    return p_hat * 100, (center - margin) * 100, (center + margin) * 100


# ============================================================
# Build winrate table
# ============================================================

def build_winrate_table(df, save_dir):
    """Compute seed-paired NC vs CO winrate on unified holdout. Emits LaTeX."""
    if "r2_unified" not in df.columns or df["r2_unified"].dropna().empty:
        print("  No r2_unified values to compute winrate from.")
        return

    co = df[df["design"] == "CO"].set_index("seed")["r2_unified"]
    nc = df[df["design"] == "NC"].set_index("seed")["r2_unified"]
    paired_co, paired_nc = co.align(nc, join="inner")
    paired = pd.DataFrame({"co": paired_co, "nc": paired_nc}).dropna()
    if len(paired) == 0:
        print("  No paired seeds available for winrate.")
        return

    n_total = len(paired)
    nc_wins = int((paired["nc"] > paired["co"]).sum())
    pct, lo, hi = _wilson_ci(nc_wins, n_total)

    print("\n  BigTPP preliminary NC win rate (Chinchilla, first epoch, "
          "unified holdout):")
    print(f"    {nc_wins}/{n_total} = {pct:.1f}% [{lo:.1f}%, {hi:.1f}%]")

    # CSV
    pd.DataFrame([{
        "Law": LAW, "Mode": MODE,
        "NC_Wins": nc_wins, "Total": n_total,
        "NC_Pct": round(pct, 2),
        "CI_Lo": round(lo, 2), "CI_Hi": round(hi, 2),
    }]).to_csv(os.path.join(save_dir, "winrate_bigtpp_prelim.csv"), index=False)

    # LaTeX
    tex = "\n".join([
        r"\begin{table}[!htbp]",
        r"\small", r"\centering",
        r"\caption{Preliminary BigTPP NC win rate (Chinchilla, first epoch). "
        r"Compute-limited preliminary experiment; not a full main-table replication.}",
        r"\label{tab:winrate_bigtpp_prelim}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Law} & \textbf{NC Wins} & \textbf{Total} & "
        r"\textbf{NC \% [95\% CI]} \\",
        r"\midrule",
        f"Chinchilla & {nc_wins} & {n_total} & "
        f"{pct:.1f}\\% [{lo:.1f}\\%, {hi:.1f}\\%] \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    out_path = os.path.join(save_dir, "table_winrate_bigtpp_prelim.tex")
    with open(out_path, "w") as f:
        f.write(tex)
    print(f"\n  Wrote {out_path}")
    print(tex)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preliminary BigTPP Chinchilla-only winrate analysis "
                    "(compute-limited).")
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing wikipedia_bigtpp.csv")
    parser.add_argument("--n-workers", type=int, default=os.cpu_count())
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--n-restarts", type=int, default=DEFAULT_N_RESTARTS)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("-o", "--output-dir", default="prelim_bigtpp_results")
    args = parser.parse_args()

    csv_path = os.path.join(args.data_dir, DATASET_CSV)
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found.")
        print("  Did you run download_and_extract.py with --include-special?")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("  PRELIMINARY BigTPP analysis")
    print("  Compute-limited; Chinchilla-only, first-epoch, unified holdout.")
    print(f"  CSV: {csv_path}")
    print(f"  Seeds: {args.n_seeds} × stride {SEED_STRIDE}")
    print(f"  Workers: {args.n_workers}")
    print(f"  Restarts per fit: {args.n_restarts}")
    print(f"{'='*60}\n")

    # Build job list: 2 designs × n_seeds
    jobs = []
    for design in ["CO", "NC"]:
        for seed_idx in range(args.n_seeds):
            jobs.append((design, csv_path, seed_idx, args.base_seed,
                         args.n_restarts))

    t0 = time.time()
    rows = []
    errors = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {pool.submit(_run_seed, j): j for j in jobs}
        for future in as_completed(futures):
            try:
                row = future.result()
                if "error" in row:
                    errors.append(row)
                else:
                    rows.append(row)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  [ERROR] worker crash: {e}")

    print(f"\n  Completed in {time.time()-t0:.1f}s")
    print(f"  Successful fits: {len(rows)}/{len(jobs)}")
    if errors:
        print(f"  Failures: {len(errors)}")
        for e in errors[:5]:
            print(f"    [{e.get('design')}/seed_{e.get('seed')}] "
                  f"{e.get('error')}")
        if len(errors) > 5:
            print(f"    ... and {len(errors)-5} more")

    if not rows:
        print("\nNo successful fits - nothing to write.")
        return

    df = pd.DataFrame(rows)
    parquet_path = os.path.join(args.output_dir, "prelim_bigtpp.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"\n  Saved: {parquet_path} ({len(df)} rows)")

    # Quick summary
    for design in ["CO", "NC"]:
        sub = df[df["design"] == design]
        if sub.empty:
            continue
        if "r2_unified" in sub.columns:
            r2u = sub["r2_unified"].dropna()
            print(f"  {design}: R² unified mean={r2u.mean():.4f} "
                  f"median={r2u.median():.4f} std={r2u.std():.4f} "
                  f"(n={len(r2u)})")

    build_winrate_table(df, args.output_dir)
    print(f"\nDone. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()