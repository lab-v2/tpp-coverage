#!/usr/bin/env python3
"""
Parallelized full-coverage CI and seed-variance analysis.

Distributes (dataset × law × mode) combos across worker processes.
Each worker is independent - loads data, fits, saves its own parquet.
Main process aggregates into summary tables AND LaTeX tables matching
the paper's tab:summary_full and tab:winrate format.

Usage:
    python run_analysis.py full_coverage --data-dir extracted_losses/ --n-workers 16
    python run_analysis.py seed_variance --data-dir extracted_losses/ --n-workers 16
    python run_analysis.py all --data-dir extracted_losses/ --n-workers 16
    python run_analysis.py all --data-dir extracted_losses/ --clear-cache

Prerequisites:
    The scaling_laws/ and incremental_coverage/ packages must be importable.
    CSVs must be in --data-dir (one per dataset, from get_results.py).
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

DATASETS = {
    "Wikipedia":  "wikipedia.csv",
    "Cosmopedia": "cosmopedia.csv",
    "RedPajama":  "redpajama.csv",
    "Pes2o":      "pes2o.csv",
    "C4":         "c4.csv",
}

# Per-law mode lists. Parametric-Epoch uses the special "Epoch-Expanded" mode
# which expands each run into per-epoch rows with (N, U_D, R, L) instead of
# collapsing to a single epoch. The other three laws use the standard
# epoch-collapsing modes.
LAW_MODES = {
    "Chinchilla":       ["final", "first", "second"],
    "Kaplan":           ["final", "first", "second"],
    "Droppo-Elibol":    ["final", "first", "second"],
    "Parametric-Epoch": ["Epoch-Expanded"],
}
LAWS = list(LAW_MODES.keys())
MODES = ["final", "first", "second"]  # kept for backward-compat callers

# Parameter names for each law (used to write named param columns).
# Must match what generate_appendix_tables.py expects in LAW_TABLE_CONFIG.
LAW_PARAM_NAMES = {
    "Kaplan":           ["N_c", "D_c", "alpha_N", "alpha_D"],
    "Chinchilla":       ["E", "A", "alpha", "B", "beta"],
    "Droppo-Elibol":    ["L_inf", "N_c", "D_c", "alpha_N", "alpha_D", "alpha"],
    "Parametric-Epoch": ["A", "alpha", "B", "beta", "E", "R_star_D", "R_star_N"],
}

DEFAULT_N_SEEDS    = 30
DEFAULT_N_RESTARTS = 100
DEFAULT_BASE_SEED  = 0


# ============================================================
# Seed-aware fit wrappers (for the three (N, D, L)-interface laws)
#
# Parametric-Epoch is NOT in here - it uses a different data shape
# (N, U_D, R, L) and is handled directly in the worker via
# fit_parametric_epoch + prepare_epoch_expanded.
# ============================================================

def _get_seed_fitters():
    """Build seed-aware fitters for the three (N, D, L)-shape laws.
    Import inside function for fork safety.

    Returns: dict[law_name -> (fit_fn, predict_fn)]
        fit_fn(N, D, L, n_restarts, seed) -> (params_array, sse, success)
        predict_fn(N, D, *params) -> L_pred
    """
    from scaling_laws.laws import (
        _fit_generic,
        kaplan_loss,
        chinchilla_loss,
        droppo_elibol_loss,
    )

    def _fit_kaplan_seed(N, D, L, n_restarts=300, seed=42):
        bounds = [(1e3, 1e14), (1e3, 1e14), (0.01, 2.0), (0.01, 2.0)]
        def sampler(rng, L):
            return [
                10 ** rng.uniform(5, 12),
                10 ** rng.uniform(5, 12),
                rng.uniform(0.05, 1.5),
                rng.uniform(0.05, 1.5),
            ]
        return _fit_generic(kaplan_loss, N, D, L, bounds, sampler, n_restarts, seed=seed)

    def _fit_chinchilla_seed(N, D, L, n_restarts=300, seed=42):
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

    def _fit_droppo_elibol_seed(N, D, L, n_restarts=300, seed=42):
        bounds = [
            (1e-6, 10.0),
            (1e3, 1e14), (1e3, 1e14),
            (0.01, 2.0), (0.01, 2.0), (0.01, 2.0),
        ]
        def sampler(rng, L):
            return [
                rng.uniform(0.01, min(L) * 0.8),
                10 ** rng.uniform(5, 12),
                10 ** rng.uniform(5, 12),
                rng.uniform(0.05, 1.5),
                rng.uniform(0.05, 1.5),
                rng.uniform(0.05, 1.5),
            ]
        return _fit_generic(droppo_elibol_loss, N, D, L, bounds, sampler, n_restarts, seed=seed)

    return {
        "Kaplan":        (_fit_kaplan_seed, kaplan_loss),
        "Chinchilla":    (_fit_chinchilla_seed, chinchilla_loss),
        "Droppo-Elibol": (_fit_droppo_elibol_seed, droppo_elibol_loss),
    }


# ============================================================
# Worker: Full Coverage
# ============================================================

def _run_full_coverage_combo(args):
    """Worker for one (dataset, law, mode) combo.

    Two code paths:
      - Standard laws (Chinchilla / Kaplan / Droppo-Elibol): use prepare_arrays
        with the given mode → (N, D, L), then call the law's seed-aware fitter.
      - Parametric-Epoch (mode = 'Epoch-Expanded'): use prepare_epoch_expanded
        → (N, U_D, R, epoch, L), then call fit_parametric_epoch directly.
    """
    (dataset_name, csv_path, law_name, mode,
     n_seeds, n_restarts, base_seed, save_dir) = args

    from scaling_laws.data import load_and_prepare, prepare_arrays, prepare_epoch_expanded

    tag = f"{dataset_name}_{law_name}_{mode}"
    cache_path = os.path.join(save_dir, f"fullcov_{tag}.parquet")

    if os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            print(f"  [cached] {tag}: {len(df)} rows")
            return df.to_dict("records")
        except Exception:
            pass

    try:
        data = load_and_prepare(csv_path, dataset_name)
    except Exception as e:
        print(f"  [FAIL] {tag}: load error: {e}")
        return []

    is_pe = (law_name == "Parametric-Epoch")
    rows = []

    if is_pe:
        # ── PARAMETRIC-EPOCH PATH ──
        try:
            from scaling_laws.laws import parametric_epoch_loss, fit_parametric_epoch
        except ImportError:
            print(f"  [SKIP] {tag}: parametric_epoch_loss not importable")
            return []
        param_names = LAW_PARAM_NAMES["Parametric-Epoch"]

        for design_label, train_runs in [
            ("CO", data.colinear_train),
            ("NC", data.noncolinear_train),
        ]:
            try:
                N_arr, UD_arr, R_arr, _ep, L_arr = prepare_epoch_expanded(train_runs)
            except Exception as e:
                print(f"  [FAIL] {tag} {design_label}: epoch-expand error: {e}")
                continue
            if len(N_arr) < 4:
                continue

            # Pre-expand holdouts
            holdout_arrays = {}
            for hname, hlist in [
                ("unified", data.unified_holdout),
                ("colinear", data.colinear_holdout),
                ("noncolinear", data.noncolinear_holdout),
            ]:
                try:
                    hN, hUD, hR, _hep, hL = prepare_epoch_expanded(hlist)
                    if len(hL) >= 2:
                        holdout_arrays[hname] = (hN, hUD, hR, hL)
                except Exception:
                    pass

            for seed_idx in range(n_seeds):
                actual_seed = base_seed + seed_idx
                try:
                    params_dict, loss, success = fit_parametric_epoch(
                        N_arr, UD_arr, R_arr, L_arr,
                        n_restarts=n_restarts, seed=actual_seed,
                    )
                    if params_dict is None:
                        continue
                except Exception:
                    continue

                row = {
                    "dataset": dataset_name,
                    "law": law_name,
                    "mode": mode,
                    "design": design_label,
                    "seed": seed_idx,
                }

                # Train metrics
                try:
                    pred_tr = parametric_epoch_loss(
                        N_arr, UD_arr, R_arr,
                        params_dict["A"], params_dict["alpha"],
                        params_dict["B"], params_dict["beta"],
                        params_dict["E"],
                        params_dict["R_star_D"], params_dict["R_star_N"],
                    )
                    row["r2_train"] = float(r2_score(L_arr, pred_tr))
                    row["rmse_train"] = float(np.sqrt(mean_squared_error(L_arr, pred_tr)))
                except Exception:
                    pass

                # Holdout metrics
                for hname, (hN, hUD, hR, hL) in holdout_arrays.items():
                    try:
                        pred_h = parametric_epoch_loss(
                            hN, hUD, hR,
                            params_dict["A"], params_dict["alpha"],
                            params_dict["B"], params_dict["beta"],
                            params_dict["E"],
                            params_dict["R_star_D"], params_dict["R_star_N"],
                        )
                        row[f"r2_{hname}"] = float(r2_score(hL, pred_h))
                        row[f"rmse_{hname}"] = float(np.sqrt(mean_squared_error(hL, pred_h)))
                    except Exception:
                        pass

                # Store params with NAMED columns (matches generate_appendix_tables)
                for pname in param_names:
                    row[f"param_{pname}"] = float(params_dict[pname])

                rows.append(row)

    else:
        # ── STANDARD PATH (Chinchilla / Kaplan / Droppo-Elibol) ──
        seed_fitters = _get_seed_fitters()
        if law_name not in seed_fitters:
            print(f"  [SKIP] {tag}: {law_name} not available")
            return []

        fit_fn, loss_fn = seed_fitters[law_name]
        param_names = LAW_PARAM_NAMES.get(law_name, [])

        for design_label, train_runs in [
            ("CO", data.colinear_train),
            ("NC", data.noncolinear_train),
        ]:
            try:
                N_arr, D_arr, L_arr = prepare_arrays(train_runs, mode=mode)
            except Exception:
                continue
            if len(N_arr) < 4:
                continue

            holdout_arrays = {}
            for hname, hlist in [
                ("unified", data.unified_holdout),
                ("colinear", data.colinear_holdout),
                ("noncolinear", data.noncolinear_holdout),
            ]:
                try:
                    hN, hD, hL = prepare_arrays(hlist, mode=mode)
                    if len(hL) >= 2:
                        holdout_arrays[hname] = (hN, hD, hL)
                except Exception:
                    pass

            for seed_idx in range(n_seeds):
                actual_seed = base_seed + seed_idx
                try:
                    params, sse, success = fit_fn(
                        N_arr, D_arr, L_arr,
                        n_restarts=n_restarts,
                        seed=actual_seed,
                    )
                    if params is None:
                        continue
                except Exception:
                    continue

                row = {
                    "dataset": dataset_name,
                    "law": law_name,
                    "mode": mode,
                    "design": design_label,
                    "seed": seed_idx,
                }

                try:
                    pred_tr = loss_fn(N_arr, D_arr, *params)
                    row["r2_train"] = float(r2_score(L_arr, pred_tr))
                    row["rmse_train"] = float(np.sqrt(mean_squared_error(L_arr, pred_tr)))
                except Exception:
                    pass

                for hname, (hN, hD, hL) in holdout_arrays.items():
                    try:
                        pred_h = loss_fn(hN, hD, *params)
                        row[f"r2_{hname}"] = float(r2_score(hL, pred_h))
                        row[f"rmse_{hname}"] = float(np.sqrt(mean_squared_error(hL, pred_h)))
                    except Exception:
                        pass

                # Store params: positional AND named (named for appendix table generator)
                for pi, pval in enumerate(params):
                    row[f"param_{pi}"] = float(pval)
                for pi, pval in enumerate(params):
                    if pi < len(param_names):
                        row[f"param_{param_names[pi]}"] = float(pval)

                rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        os.makedirs(save_dir, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"  [done] {tag}: {len(rows)} rows")
    else:
        print(f"  [empty] {tag}: no results")

    return rows


# ============================================================
# Worker: Seed Variance
# ============================================================

def _run_seed_variance_combo(args):
    """Worker for one (dataset, law, mode) seed-variance combo.

    Parametric-Epoch is skipped here - it's a 4D law (N × U_D × R × L) that
    isn't plotted on the convergence-fraction figures, so seed-variance
    sweeps are unnecessary and would add hours to the run.
    """
    (dataset_name, csv_path, law_name, mode,
     n_seeds, n_restarts, base_seed, save_dir) = args

    if law_name == "Parametric-Epoch":
        return []

    from scaling_laws.data import load_and_prepare, prepare_arrays
    from incremental_coverage.coverage import (
        build_colinear_steps,
        build_noncolinear_steps,
    )

    tag = f"{dataset_name}_{law_name}_{mode}"
    cache_path = os.path.join(save_dir, f"seed_variance_{tag}.parquet")

    if os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            print(f"  [cached] {tag}: {len(df)} rows")
            return df.to_dict("records")
        except Exception:
            pass

    try:
        data = load_and_prepare(csv_path, dataset_name)
    except Exception as e:
        print(f"  [FAIL] {tag}: load error: {e}")
        return []

    seed_fitters = _get_seed_fitters()
    if law_name not in seed_fitters:
        return []

    fit_fn, loss_fn = seed_fitters[law_name]

    try:
        co_steps = build_colinear_steps(data.colinear_train, mode=mode)
        nc_steps = build_noncolinear_steps(data.noncolinear_train)
    except Exception as e:
        print(f"  [FAIL] {tag}: step build error: {e}")
        return []

    holdout_map = {
        "unified": data.unified_holdout,
        "colinear": data.colinear_holdout,
        "noncolinear": data.noncolinear_holdout,
    }

    rows = []
    for design_label, steps in [("Colinear", co_steps), ("Non-Colinear", nc_steps)]:
        n_total_steps = len(steps)
        for step_idx, step in enumerate(steps):
            step_data = step if isinstance(step, list) else step.get("data", step.get("runs", []))
            coverage_frac = (step_idx + 1) / n_total_steps

            try:
                N_arr, D_arr, L_arr = prepare_arrays(step_data, mode=mode)
            except Exception:
                continue
            if len(N_arr) < 4:
                continue

            for holdout_name, holdout_runs in holdout_map.items():
                seed_r2s = []
                try:
                    hN, hD, hL = prepare_arrays(holdout_runs, mode=mode)
                except Exception:
                    continue
                if len(hN) < 2:
                    continue

                for seed_idx in range(n_seeds):
                    actual_seed = base_seed + seed_idx
                    try:
                        params, sse, success = fit_fn(
                            N_arr, D_arr, L_arr,
                            n_restarts=n_restarts,
                            seed=actual_seed,
                        )
                        if params is None:
                            continue
                    except Exception:
                        continue

                    try:
                        h_pred = loss_fn(hN, hD, *params)
                        ss_res = np.sum((hL - h_pred)**2)
                        ss_tot = np.sum((hL - hL.mean())**2)
                        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
                        if r2 is not None:
                            seed_r2s.append(r2)
                    except Exception:
                        continue

                if seed_r2s:
                    arr = np.array(seed_r2s)
                    rows.append({
                        "design": design_label,
                        "step": step_idx + 1,
                        "coverage_frac": coverage_frac,
                        "holdout_split": holdout_name,
                        "r2_mean": float(arr.mean()),
                        "r2_std": float(arr.std()),
                        "r2_p5": float(np.percentile(arr, 5)),
                        "r2_p95": float(np.percentile(arr, 95)),
                        "r2_min": float(arr.min()),
                        "r2_max": float(arr.max()),
                        "n_seeds_ok": len(seed_r2s),
                    })

    if rows:
        df = pd.DataFrame(rows)
        os.makedirs(save_dir, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"  [done] {tag}: {len(rows)} rows")
    else:
        print(f"  [empty] {tag}: no results")

    return rows


# ============================================================
# Stats helper
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
# LaTeX tables matching the paper's tab:summary_full and tab:winrate
# ============================================================

def build_paper_tables(df: pd.DataFrame, save_dir: str):
    """Emit summary + winrate tables in the paper's format.

    Summary table: Train (D) and Holdout (H) splits only, R² + RMSE stats.
    Winrate table: 3-column block - Dataset | Scaling Law | Epoch.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ==============================================================
    # SUMMARY TABLE
    # ==============================================================
    split_label_map = {
        "train":       ("Train",      r"Train ($\mathcal{D}$)"),
        "unified":     ("Holdout",    r"Holdout ($\mathcal{H}$)"),
        "colinear":    ("Holdout-CO", r"Holdout--CO ($\mathcal{H}_{\mathrm{col}}$)"),
        "noncolinear": ("Holdout-NC", r"Holdout--NC ($\mathcal{H}_{\mathrm{nc}}$)"),
    }

    summary_rows = []
    for split_key, (csv_label, tex_label) in split_label_map.items():
        r2_col = f"r2_{split_key}"
        rmse_col = f"rmse_{split_key}"
        if r2_col not in df.columns:
            continue

        for design in ["CO", "NC"]:
            sub = df[df["design"] == design]
            if sub[r2_col].dropna().empty:
                continue

            combos = sub.groupby(["dataset", "law", "mode"])
            r2_means = combos[r2_col].mean()
            r2_stds = combos[r2_col].std()
            rmse_means = combos[rmse_col].mean()
            rmse_stds = combos[rmse_col].std()
            mean_n = combos[r2_col].count().mean() or 1

            r2_ci_half = 1.96 * r2_stds.mean() / np.sqrt(mean_n)
            rmse_ci_half = 1.96 * rmse_stds.mean() / np.sqrt(mean_n)

            summary_rows.append({
                "Split_csv": csv_label,
                "Split_tex": tex_label,
                "Design": design,
                "R2_Mean": float(r2_means.mean()),
                "R2_CI_lo": float(r2_means.mean() - r2_ci_half),
                "R2_CI_hi": float(r2_means.mean() + r2_ci_half),
                "R2_Median": float(r2_means.median()),
                "R2_Std": float(r2_means.std()),
                "RMSE_Mean": float(rmse_means.mean()),
                "RMSE_CI_lo": float(rmse_means.mean() - rmse_ci_half),
                "RMSE_CI_hi": float(rmse_means.mean() + rmse_ci_half),
                "RMSE_Median": float(rmse_means.median()),
                "RMSE_Std": float(rmse_means.std()),
            })

    summary = pd.DataFrame(summary_rows)
    summary_csv = summary.drop(columns=["Split_tex"]).rename(columns={"Split_csv": "Split"})
    summary_csv.to_csv(os.path.join(save_dir, "summary_table.csv"), index=False)

    # Best (highest R²) per split for bolding
    best = {}
    for tex_label in summary["Split_tex"].unique():
        s = summary[summary["Split_tex"] == tex_label]
        best[tex_label] = s.loc[s["R2_Mean"].idxmax(), "Design"]

    tex_lines = [
        r"\begin{table*}[!htbp]",
        r"\small",
        r"\centering",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{$R^2$ and RMSE summary across training designs with different "
        r"seed-to-seed optimizer setup. 95\% CI: confidence interval on the mean. "
        r"Best per split in \textbf{bold}.}",
        r"\label{tab:summary_full}",
        r"\begin{tabular}{ll cccc cccc}",
        r"\toprule",
        r"& & \multicolumn{4}{c}{\textbf{$R^2$}} & \multicolumn{4}{c}{\textbf{RMSE}} \\",
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}",
        r"\textbf{Split} & \textbf{Design} & Mean & 95\% CI & Median & Std & "
        r"Mean & 95\% CI & Median & Std \\",
        r"\midrule",
    ]

    last_split = None
    for _, row in summary.iterrows():
        tex_label = row["Split_tex"]
        design = row["Design"]
        is_best = (best.get(tex_label) == design)

        def _w(val, fmt=".4f"):
            s = f"{val:{fmt}}"
            return rf"\win{{{s}}}" if is_best else s

        def _wci(lo, hi):
            s = f"[{lo:.3f}, {hi:.3f}]"
            return rf"\win{{{s}}}" if is_best else s

        if tex_label != last_split:
            if last_split is not None:
                tex_lines.append(r"\midrule")
            tex_lines.append(rf"\multirow{{2}}{{*}}{{{tex_label}}}")
            last_split = tex_label

        tex_lines.append(
            rf" & {design}  & {_w(row['R2_Mean'])}  & {_wci(row['R2_CI_lo'], row['R2_CI_hi'])} "
            rf"& {_w(row['R2_Median'])} & {_w(row['R2_Std'])} "
            rf"& {_w(row['RMSE_Mean'])} & {_wci(row['RMSE_CI_lo'], row['RMSE_CI_hi'])} "
            rf"& {_w(row['RMSE_Median'])} & {_w(row['RMSE_Std'])} \\"
        )

    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    tex_summary = "\n".join(tex_lines)
    summary_path = os.path.join(save_dir, "table_summary_full.tex")
    with open(summary_path, "w") as f:
        f.write(tex_summary)
    print(f"\n  Wrote {summary_path}")
    print(tex_summary)
    print("\n  Summary preview:")
    print(summary[["Split_csv", "Design", "R2_Mean", "R2_Median", "R2_Std",
                   "RMSE_Mean", "RMSE_Median"]].round(4).to_string(index=False))

    # ==============================================================
    # WINRATE TABLE
    # ==============================================================
    if "r2_unified" not in df.columns:
        return summary, None

    wins_rows = []
    for (ds, law, mode), group in df.groupby(["dataset", "law", "mode"]):
        co = group[group["design"] == "CO"].set_index("seed")["r2_unified"]
        nc = group[group["design"] == "NC"].set_index("seed")["r2_unified"]
        paired_co, paired_nc = co.align(nc, join="inner")
        if len(paired_co) == 0:
            continue
        nc_wins = int((paired_nc > paired_co).sum())
        wins_rows.append({
            "Dataset": ds, "Law": law, "Mode": mode,
            "NC_wins": nc_wins,
            "N_paired": len(paired_co),
        })
    wins_df = pd.DataFrame(wins_rows)
    wins_df.to_csv(os.path.join(save_dir, "winrate_per_combo.csv"), index=False)

    if wins_df.empty:
        return summary, wins_df

    total_nc = int(wins_df["NC_wins"].sum())
    total_n = int(wins_df["N_paired"].sum())
    overall_pct, overall_lo, overall_hi = _wilson_ci(total_nc, total_n)

    slice_columns = ["Dataset", "Law", "Mode"]
    slices = {}
    for col in slice_columns:
        entries = []
        for val, grp in wins_df.groupby(col):
            nc_w = int(grp["NC_wins"].sum())
            tot = int(grp["N_paired"].sum())
            pct, _, _ = _wilson_ci(nc_w, tot)
            entries.append((val, pct))
        entries.sort(key=lambda x: str(x[0]))
        slices[col] = entries

    print(f"\n  Overall NC win rate (unified): {total_nc}/{total_n} "
          f"({overall_pct:.1f}% [{overall_lo:.1f}%, {overall_hi:.1f}%])")
    for col in slice_columns:
        print(f"  By {col}:")
        for val, pct in slices[col]:
            print(f"    {val}: {pct:.1f}%")

    tex_lines = [
        r"\begin{table*}[!htbp]",
        r"\small",
        r"\centering",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{Win rate breakdown (NC vs.\ CO design) on unified holdout "
        rf"with seed-paired comparisons. Overall NC (non-colinear) win rate: "
        rf"{overall_pct:.1f}\% ({total_nc}/{total_n}). 95\% CI: "
        rf"[{overall_lo:.1f}\%, {overall_hi:.1f}\%].}}",
        r"\label{tab:winrate}",
        r"\begin{tabular}{lr|lr|lr}",
        r"\toprule",
        r"\multicolumn{2}{c|}{\textbf{Dataset}} & "
        r"\multicolumn{2}{c|}{\textbf{Scaling Law}} & "
        r"\multicolumn{2}{c}{\textbf{Epoch}} \\",
        r"\midrule",
    ]

    max_rows = max(len(slices[c]) for c in slice_columns)
    for i in range(max_rows):
        parts = []
        for col in slice_columns:
            entries = slices[col]
            if i < len(entries):
                val, pct = entries[i]
                parts.append(rf"{val} & {pct:.1f}\%")
            else:
                parts.append(r" & ")
        tex_lines.append(" & ".join(parts) + r" \\")

    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    tex_winrate = "\n".join(tex_lines)
    winrate_path = os.path.join(save_dir, "table_winrate.tex")
    with open(winrate_path, "w") as f:
        f.write(tex_winrate)
    print(f"\n  Wrote {winrate_path}")
    print(tex_winrate)

    # ==============================================================
    # APPENDIX WINRATE TABLE - broken down by holdout split definition
    # ==============================================================
    build_holdout_split_winrate(df, save_dir)

    return summary, wins_df


def build_holdout_split_winrate(df: pd.DataFrame, save_dir: str,
                                 label: str = "appendix",
                                 include_per_dataset: bool = True):
    """Emit a winrate table broken down by holdout-split definition.

    Top section: aggregate over all (law × mode × dataset) for each of
    the three holdout splits (unified / colinear / noncolinear).

    Bottom section (optional): same splits sliced by dataset.
    """
    SPLIT_NAMES = {
        "unified":     r"$\mathcal{H}$",
        "colinear":    r"$\mathcal{H}_{\mathrm{col}}$",
        "noncolinear": r"$\mathcal{H}_{\mathrm{nc}}$",
    }

    available = [s for s in SPLIT_NAMES if f"r2_{s}" in df.columns]
    if not available:
        print("  No holdout columns found - skipping appendix winrate")
        return

    def _pair_wins(group, r2_col):
        co = group[group["design"] == "CO"].set_index("seed")[r2_col]
        nc = group[group["design"] == "NC"].set_index("seed")[r2_col]
        co_a, nc_a = co.align(nc, join="inner")
        if len(co_a) == 0:
            return 0, 0
        nc_wins = int((nc_a > co_a).sum())
        return nc_wins, len(co_a)

    # Aggregate rows
    agg_rows = []
    for split in available:
        r2_col = f"r2_{split}"
        nc_total, n_total = 0, 0
        for _, group in df.groupby(["dataset", "law", "mode"]):
            nw, np_ = _pair_wins(group, r2_col)
            nc_total += nw
            n_total += np_
        agg_rows.append({
            "split": split,
            "label": SPLIT_NAMES[split],
            "nc_wins": nc_total,
            "co_wins": n_total - nc_total,
            "total": n_total,
            "pct": (nc_total / n_total * 100) if n_total else 0.0,
        })

    # Per-dataset rows (only if multiple datasets)
    per_ds_rows = []
    datasets_in_df = sorted(df["dataset"].unique())
    if include_per_dataset and len(datasets_in_df) > 1:
        for split in available:
            r2_col = f"r2_{split}"
            for ds in datasets_in_df:
                sub = df[df["dataset"] == ds]
                nc_total, n_total = 0, 0
                for _, group in sub.groupby(["law", "mode"]):
                    nw, np_ = _pair_wins(group, r2_col)
                    nc_total += nw
                    n_total += np_
                per_ds_rows.append({
                    "split": split,
                    "label": f"{SPLIT_NAMES[split]} / {ds}",
                    "nc_wins": nc_total,
                    "co_wins": n_total - nc_total,
                    "total": n_total,
                    "pct": (nc_total / n_total * 100) if n_total else 0.0,
                })

    # CSV (clean labels)
    csv_rows = []
    for r in agg_rows + per_ds_rows:
        csv_rows.append({
            "Holdout_Split": r["split"],
            "Label": r["label"].replace("$", "").replace(r"\mathcal", "").replace(r"\mathrm", ""),
            "NC_Wins": r["nc_wins"],
            "CO_Wins": r["co_wins"],
            "Total": r["total"],
            "NC_Pct": round(r["pct"], 2),
        })
    pd.DataFrame(csv_rows).to_csv(
        os.path.join(save_dir, f"winrate_by_holdout_split_{label}.csv"), index=False,
    )

    # LaTeX
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Win rate breakdown by holdout split definition.}",
        r"\label{tab:winrate_appendix}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Holdout Split} & \textbf{NC Wins} & \textbf{CO Wins} & "
        r"\textbf{Total} & \textbf{NC \%} \\",
        r"\midrule",
    ]
    for r in agg_rows:
        tex_lines.append(
            f"{r['label']} & {r['nc_wins']} & {r['co_wins']} & "
            f"{r['total']} & {r['pct']:.1f}\\% \\\\"
        )
    if per_ds_rows:
        tex_lines.append(r"\midrule")
        for r in per_ds_rows:
            tex_lines.append(
                f"  {r['label']} & {r['nc_wins']} & {r['co_wins']} & "
                f"{r['total']} & {r['pct']:.1f}\\% \\\\"
            )
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex_str = "\n".join(tex_lines)
    out_path = os.path.join(save_dir, f"table_winrate_by_holdout_split_{label}.tex")
    with open(out_path, "w") as f:
        f.write(tex_str)
    print(f"\n  Wrote {out_path}")
    print(tex_str)


# ============================================================
# Drivers
# ============================================================

def run_full_coverage(data_dir, output_dir, n_workers, n_seeds, n_restarts, base_seed):
    print(f"\n{'='*60}\nFULL COVERAGE ANALYSIS")
    print(f"  Workers: {n_workers}  Seeds: {n_seeds}  Restarts: {n_restarts}")
    print(f"  Laws: {LAWS}\n{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    combos = []
    for ds_name, csv_file in DATASETS.items():
        csv_path = os.path.join(data_dir, csv_file)
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found, skipping {ds_name}")
            continue
        for law in LAWS:
            for mode in LAW_MODES[law]:
                combos.append((ds_name, csv_path, law, mode,
                              n_seeds, n_restarts, base_seed, output_dir))

    print(f"Total combos: {len(combos)}")
    t0 = time.time()

    all_rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_full_coverage_combo, c): c for c in combos}
        for future in as_completed(futures):
            try:
                all_rows.extend(future.result())
            except Exception as e:
                combo = futures[future]
                print(f"  [ERROR] {combo[0]}_{combo[2]}_{combo[3]}: {e}")

    print(f"\nCompleted in {(time.time()-t0)/60:.1f} minutes")

    if all_rows:
        df = pd.DataFrame(all_rows)
        master_path = os.path.join(output_dir, "full_coverage_all_seeds.parquet")
        df.to_parquet(master_path, index=False)
        print(f"Master file: {master_path} ({len(df)} rows)")
        build_paper_tables(df, output_dir)
    else:
        print("No results!")


def run_seed_variance(data_dir, output_dir, n_workers, n_seeds, n_restarts, base_seed):
    print(f"\n{'='*60}\nSEED VARIANCE ANALYSIS")
    print(f"  Workers: {n_workers}  Seeds: {n_seeds}  Restarts: {n_restarts}\n{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    combos = []
    for ds_name, csv_file in DATASETS.items():
        csv_path = os.path.join(data_dir, csv_file)
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found, skipping {ds_name}")
            continue
        for law in LAWS:
            if law == "Parametric-Epoch":
                continue  # 4D law, not plotted in seed-variance figures
            for mode in MODES:
                combos.append((ds_name, csv_path, law, mode,
                              n_seeds, n_restarts, base_seed, output_dir))

    print(f"Total combos: {len(combos)}")
    t0 = time.time()

    all_rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_seed_variance_combo, c): c for c in combos}
        for future in as_completed(futures):
            try:
                all_rows.extend(future.result())
            except Exception as e:
                combo = futures[future]
                print(f"  [ERROR] {combo[0]}_{combo[2]}_{combo[3]}: {e}")

    print(f"\nCompleted in {(time.time()-t0)/60:.1f} minutes")

    if all_rows:
        df = pd.DataFrame(all_rows)
        master_path = os.path.join(output_dir, "seed_variance_all.parquet")
        df.to_parquet(master_path, index=False)
        print(f"Master file: {master_path} ({len(df)} rows)")

        uni = df[df["holdout_split"] == "unified"]
        full = uni[uni["coverage_frac"] == 1.0]
        if not full.empty:
            print("\nSeed variance summary (unified, full coverage):")
            for design in ["Colinear", "Non-Colinear"]:
                dsub = full[full["design"] == design]
                if not dsub.empty:
                    print(f"  {design}: mean σ(R²) = {dsub['r2_std'].mean():.4f}, "
                          f"max σ(R²) = {dsub['r2_std'].max():.4f}")
    else:
        print("No results!")


def main():
    parser = argparse.ArgumentParser(
        description="Parallelized scaling law analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=["full_coverage", "seed_variance", "all"])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--n-workers", type=int, default=os.cpu_count())
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--n-restarts", type=int, default=DEFAULT_N_RESTARTS)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("-o", "--output-dir", default=None)
    parser.add_argument("--clear-cache", action="store_true")
    args = parser.parse_args()

    def _maybe_clear(out_dir):
        if not args.clear_cache or not os.path.isdir(out_dir):
            return
        import glob
        removed = 0
        for f in glob.glob(os.path.join(out_dir, "*.parquet")):
            os.remove(f)
            removed += 1
        if removed:
            print(f"  Cleared {removed} cached parquet file(s) from {out_dir}/")

    if args.mode in ("full_coverage", "all"):
        out = args.output_dir or "full_coverage_results"
        _maybe_clear(out)
        run_full_coverage(args.data_dir, out, args.n_workers,
                         args.n_seeds, args.n_restarts, args.base_seed)

    if args.mode in ("seed_variance", "all"):
        out = args.output_dir or "seed_variance_results"
        _maybe_clear(out)
        run_seed_variance(args.data_dir, out, args.n_workers,
                         args.n_seeds, args.n_restarts, args.base_seed)


if __name__ == "__main__":
    main()