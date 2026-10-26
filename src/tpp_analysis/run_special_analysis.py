#!/usr/bin/env python3
"""
Full-coverage analysis on the BF16 Wikipedia special dataset.

Mirrors the same pattern as run_analysis.py:
  - Standard laws (Chinchilla / Kaplan / Droppo-Elibol) use _fit_generic
    on (N, D, L) arrays from prepare_arrays(mode=...)
  - Parametric-Epoch uses fit_parametric_epoch on (N, U_D, R, L) arrays
    from prepare_epoch_expanded()

Seed protocol per the appendix's BF16 replication entry:
  - 30 optimizer seeds, base seed b=0, stride 137: s_i = 137 * i
  - Stored seed column = seed index (0..29), not the actual RNG seed

Usage:
    python run_special_analysis.py --data-dir extracted_losses/ --n-workers 16
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

SPECIAL_DATASETS = {
    "bf16": {
        "name": "Wikipedia_BF16",
        "csv": "wikipedia_bf16.csv",
        "laws": ["Chinchilla", "Kaplan", "Droppo-Elibol", "Parametric-Epoch"],
        "description": "BF16 precision replication of Wikipedia",
    },
}

# Per-law mode lists (mirrors run_analysis.py).
# Parametric-Epoch only uses the special "Epoch-Expanded" mode that exposes
# (N, U_D, R, L) instead of collapsing epochs.
LAW_MODES = {
    "Chinchilla":       ["final", "first", "second"],
    "Kaplan":           ["final", "first", "second"],
    "Droppo-Elibol":    ["final", "first", "second"],
    "Parametric-Epoch": ["Epoch-Expanded"],
}

# Parameter names per law - used for named param columns
# (matches generate_appendix_tables.py LAW_TABLE_CONFIG)
LAW_PARAM_NAMES = {
    "Kaplan":           ["N_c", "D_c", "alpha_N", "alpha_D"],
    "Chinchilla":       ["E", "A", "alpha", "B", "beta"],
    "Droppo-Elibol":    ["L_inf", "N_c", "D_c", "alpha_N", "alpha_D", "alpha"],
    "Parametric-Epoch": ["A", "alpha", "B", "beta", "E", "R_star_D", "R_star_N"],
}

DEFAULT_N_SEEDS    = 30
DEFAULT_N_RESTARTS = 100
DEFAULT_BASE_SEED  = 0
SEED_STRIDE        = 137  # BF16 uses stride 137 per appendix


# ============================================================
# Seed-aware fit wrappers (for the three (N, D, L)-shape laws)
#
# Parametric-Epoch is NOT here - handled directly in the worker
# via prepare_epoch_expanded + fit_parametric_epoch.
# ============================================================

def _get_seed_fitters():
    """Build seed-aware fitters for the three (N, D, L)-shape laws."""
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
# Worker: Full Coverage (one combo)
# ============================================================

def _run_full_coverage_combo(args):
    """Worker for one (dataset, law, mode) combo.

    Branches:
      - Standard laws: prepare_arrays(mode=mode) → (N, D, L) → fitter
      - Parametric-Epoch: prepare_epoch_expanded() → (N, U_D, R, L) → fit_parametric_epoch
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
                actual_seed = base_seed + seed_idx * SEED_STRIDE
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

                # Named param columns (matches generate_appendix_tables)
                for pname in param_names:
                    row[f"param_{pname}"] = float(params_dict[pname])

                rows.append(row)

    else:
        # ── STANDARD PATH ──
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
                actual_seed = base_seed + seed_idx * SEED_STRIDE
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

                # Both positional and named (forward-compat)
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
        print(f"  [empty] {tag}: no successful fits")

    return rows


# ============================================================
# Stats helpers and table builders
# ============================================================

def _wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p_hat = k / n
    denom = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
    return p_hat * 100, (center - margin) * 100, (center + margin) * 100


def build_tables(df, save_dir, label):
    """Build summary + winrate tables with LaTeX output."""

    # SUMMARY TABLE
    # Internal split key → ("CSV label", "LaTeX label")
    SPLIT_LABELS = {
        "train":       ("Train",      r"Train ($\mathcal{D}$)"),
        "unified":     ("Holdout",    r"Holdout ($\mathcal{H}$)"),
        "colinear":    ("Holdout-CO", r"Holdout--CO ($\mathcal{H}_{\mathrm{col}}$)"),
        "noncolinear": ("Holdout-NC", r"Holdout--NC ($\mathcal{H}_{\mathrm{nc}}$)"),
    }

    summary_rows = []
    for split, (csv_label, tex_label) in SPLIT_LABELS.items():
        r2_col = f"r2_{split}"
        rmse_col = f"rmse_{split}"
        if r2_col not in df.columns:
            continue

        for design in ["CO", "NC"]:
            sub = df[df["design"] == design]
            if sub[r2_col].dropna().empty:
                continue

            combos = sub.groupby(["law", "mode"])
            r2_means = combos[r2_col].mean()
            r2_stds = combos[r2_col].std()
            rmse_means = combos[rmse_col].mean()
            rmse_stds = combos[rmse_col].std()
            mean_n = combos[r2_col].count().mean() or 1

            r2_ci_half = 1.96 * r2_stds.mean() / np.sqrt(mean_n)
            rmse_ci_half = 1.96 * rmse_stds.mean() / np.sqrt(mean_n)

            summary_rows.append({
                "Split": csv_label,
                "Split_tex": tex_label,
                "Design": design,
                "R2_Mean": float(r2_means.mean()),
                "R2_CI_lo": float(r2_means.mean() - r2_ci_half),
                "R2_CI_hi": float(r2_means.mean() + r2_ci_half),
                "R2_Median": float(r2_means.median()),
                "R2_Std": float(r2_means.std()),
                "R2_Sigma": float(r2_stds.mean()),
                "RMSE_Mean": float(rmse_means.mean()),
                "RMSE_CI_lo": float(rmse_means.mean() - rmse_ci_half),
                "RMSE_CI_hi": float(rmse_means.mean() + rmse_ci_half),
                "RMSE_Median": float(rmse_means.median()),
                "RMSE_Std": float(rmse_means.std()),
            })

    summary = pd.DataFrame(summary_rows)
    # CSV gets clean labels (no LaTeX label column)
    summary.drop(columns=["Split_tex"]).to_csv(os.path.join(save_dir, f"summary_{label}.csv"), index=False)

    best = {}
    for tex_label in summary["Split_tex"].unique():
        s = summary[summary["Split_tex"] == tex_label]
        best[tex_label] = s.loc[s["R2_Mean"].idxmax(), "Design"]

    tex_lines = [
        r"\begin{table*}[!htbp]",
        r"\small", r"\centering",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{$R^2$ and RMSE summary for {label} across training designs.",
        r"$\sigma$: seed-to-seed optimizer std (30 seeds). 95\% CI: confidence interval on the mean.",
        r"Best per split in \textbf{bold}.}",
        rf"\label{{tab:summary_{label}}}",
        r"\begin{tabular}{ll cccc cccc}",
        r"\toprule",
        r"& & \multicolumn{4}{c}{\textbf{$R^2$}} & \multicolumn{4}{c}{\textbf{RMSE}} \\",
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}",
        r"\textbf{Split} & \textbf{Design} & Mean & 95\% CI & Median & Std & Mean & 95\% CI & Median & Std \\",
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
    summary_path = os.path.join(save_dir, f"table_summary_{label}.tex")
    with open(summary_path, "w") as f:
        f.write(tex_summary)
    print(f"\n  Wrote {summary_path}")
    print(tex_summary)
    print(f"\n  {label} R² summary:")
    print(summary[["Split", "Design", "R2_Mean", "R2_Sigma", "R2_Median"]].round(4).to_string(index=False))

    # WINRATE TABLE
    wins_rows = []
    for (law, mode), group in df.groupby(["law", "mode"]):
        r2_col = "r2_unified"
        if r2_col not in group.columns:
            continue
        co = group[group["design"] == "CO"].set_index("seed")[r2_col]
        nc = group[group["design"] == "NC"].set_index("seed")[r2_col]
        paired_co, paired_nc = co.align(nc, join="inner")
        if len(paired_co) == 0:
            continue
        nc_wins = int((paired_nc > paired_co).sum())
        wins_rows.append({
            "Law": law, "Mode": mode,
            "NC_wins": nc_wins, "N_paired": len(paired_co),
            "NC_pct": nc_wins / len(paired_co) * 100,
        })

    wins_df = pd.DataFrame(wins_rows)
    wins_df.to_csv(os.path.join(save_dir, f"winrate_{label}.csv"), index=False)
    if wins_df.empty:
        return

    total_nc = int(wins_df["NC_wins"].sum())
    total_n = int(wins_df["N_paired"].sum())
    overall_pct, overall_lo, overall_hi = _wilson_ci(total_nc, total_n)

    print(f"\n  {label} NC win rate (unified): {total_nc}/{total_n} "
          f"({overall_pct:.1f}% [{overall_lo:.1f}%, {overall_hi:.1f}%])")

    slices = {}
    for col in ["Law", "Mode"]:
        entries = []
        for val, grp in wins_df.groupby(col):
            nc_w = int(grp["NC_wins"].sum())
            tot = int(grp["N_paired"].sum())
            pct, _, _ = _wilson_ci(nc_w, tot)
            entries.append((val, pct))
        entries.sort(key=lambda x: str(x[0]))
        slices[col] = entries

    tex_lines = [
        r"\begin{table*}[!htbp]",
        r"\small", r"\centering",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{Win rate breakdown (NC vs.\ CO design) on unified holdout for {label} "
        rf"with seed-paired comparisons. Overall NC win rate: {overall_pct:.1f}\% "
        rf"[{overall_lo:.1f}\%, {overall_hi:.1f}\%] ({total_nc}/{total_n}).}}",
        rf"\label{{tab:winrate_{label}}}",
    ]

    cols_available = [c for c in ["Law", "Mode"] if slices.get(c)]
    n_cols = len(cols_available)
    col_spec = "|".join(["lr"] * n_cols)
    tex_lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    tex_lines.append(r"\toprule")

    header_parts = []
    for i, col in enumerate(cols_available):
        sep = "|" if i < n_cols - 1 else ""
        header_parts.append(rf"\multicolumn{{2}}{{c{sep}}}{{\textbf{{{col}}}}}")
    tex_lines.append(" & ".join(header_parts) + r" \\")

    cmidrule_parts = []
    for i, col in enumerate(cols_available):
        start = i * 2 + 1
        end = start + 1
        cmidrule_parts.append(rf"\cmidrule(lr){{{start}-{end}}}")
    tex_lines.append(" ".join(cmidrule_parts))

    sub_header = []
    for col in cols_available:
        sub_header.append(r"& \textbf{NC\%}")
    tex_lines.append(" & ".join(sub_header) + r" \\")
    tex_lines.append(r"\midrule")

    max_rows = max(len(slices[c]) for c in cols_available) if cols_available else 0
    for i in range(max_rows):
        parts = []
        for col in cols_available:
            entries = slices[col]
            if i < len(entries):
                val, pct = entries[i]
                parts.append(rf"{val} & {pct:.1f}\%")
            else:
                parts.append("& ")
        tex_lines.append(" & ".join(parts) + r" \\")

    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    tex_winrate = "\n".join(tex_lines)
    winrate_path = os.path.join(save_dir, f"table_winrate_{label}.tex")
    with open(winrate_path, "w") as f:
        f.write(tex_winrate)
    print(f"  Wrote {winrate_path}")
    print(tex_winrate)

    # APPENDIX WINRATE TABLE - broken down by holdout split definition
    build_holdout_split_winrate(df, save_dir, label=label)


def build_holdout_split_winrate(df, save_dir, label="appendix"):
    """Emit a winrate table broken down by holdout-split definition.

    For run_special_analysis.py there's only one dataset, so we omit the
    per-dataset breakdown that run_analysis.py produces.
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

    rows = []
    for split in available:
        r2_col = f"r2_{split}"
        nc_total, n_total = 0, 0
        for _, group in df.groupby(["dataset", "law", "mode"]):
            nw, np_ = _pair_wins(group, r2_col)
            nc_total += nw
            n_total += np_
        rows.append({
            "split": split,
            "tex": SPLIT_NAMES[split],
            "nc_wins": nc_total,
            "co_wins": n_total - nc_total,
            "total": n_total,
            "pct": (nc_total / n_total * 100) if n_total else 0.0,
        })

    # CSV
    pd.DataFrame([{
        "Holdout_Split": r["split"],
        "NC_Wins": r["nc_wins"],
        "CO_Wins": r["co_wins"],
        "Total": r["total"],
        "NC_Pct": round(r["pct"], 2),
    } for r in rows]).to_csv(
        os.path.join(save_dir, f"winrate_by_holdout_split_{label}.csv"), index=False,
    )

    # LaTeX
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Win rate breakdown by holdout split definition for {label}.}}",
        rf"\label{{tab:winrate_appendix_{label}}}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Holdout Split} & \textbf{NC Wins} & \textbf{CO Wins} & "
        r"\textbf{Total} & \textbf{NC \%} \\",
        r"\midrule",
    ]
    for r in rows:
        tex_lines.append(
            f"{r['tex']} & {r['nc_wins']} & {r['co_wins']} & "
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
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full-coverage analysis for the BF16 special Wikipedia dataset")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset", choices=["bf16"], default="bf16",
                       help="Special dataset to run (only bf16 is supported)")
    parser.add_argument("--n-workers", type=int, default=os.cpu_count())
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--n-restarts", type=int, default=DEFAULT_N_RESTARTS)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("-o", "--output-dir", default="special_results")
    parser.add_argument("--clear-cache", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.clear_cache:
        import glob
        removed = 0
        for f in glob.glob(os.path.join(args.output_dir, "*.parquet")):
            os.remove(f)
            removed += 1
        print(f"Cleared {removed} cached parquet file(s)")

    cfg = SPECIAL_DATASETS[args.dataset]
    csv_path = os.path.join(args.data_dir, cfg["csv"])

    if not os.path.exists(csv_path):
        print(f"\n  ERROR: {csv_path} not found.")
        return

    print(f"\n{'='*60}")
    print(f"  {cfg['description']}")
    print(f"  CSV: {csv_path}")
    print(f"  Laws: {cfg['laws']}")
    print(f"  Seed stride: {SEED_STRIDE}")
    print(f"{'='*60}")

    combos = []
    for law in cfg["laws"]:
        for mode in LAW_MODES.get(law, ["final"]):
            combos.append((cfg["name"], csv_path, law, mode,
                          args.n_seeds, args.n_restarts, args.base_seed,
                          args.output_dir))

    print(f"\n  Full coverage: {len(combos)} combos × {args.n_seeds} seeds")
    t0 = time.time()

    all_rows = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {pool.submit(_run_full_coverage_combo, c): c for c in combos}
        for future in as_completed(futures):
            try:
                all_rows.extend(future.result())
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  [ERROR] {e}")

    if all_rows:
        df = pd.DataFrame(all_rows)
        parquet_path = os.path.join(args.output_dir, f"full_coverage_{args.dataset}.parquet")
        df.to_parquet(parquet_path, index=False)
        print(f"\n  Saved: {parquet_path} ({len(df)} rows, {time.time()-t0:.1f}s)")
        build_tables(df, args.output_dir, args.dataset)
    else:
        print("\n  No results!")

    print(f"\nDone. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()