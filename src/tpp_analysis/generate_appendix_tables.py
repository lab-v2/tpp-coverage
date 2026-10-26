#!/usr/bin/env python3
"""
Generate per-dataset full metric breakdown tables (Appendix E).

Reads a full-coverage parquet and produces LaTeX tables for each
dataset × holdout split, with seed-averaged R²/RMSE and 95% CI
annotations. Also produces the fitted parameter tables (Appendix F)
from a single reference seed.

Two modes:
  --main (default): reads full_coverage_all_seeds.parquet
                    (output of run_analysis.py - 5 datasets)
  --special:        reads full_coverage_bf16.parquet
                    (output of run_special_analysis.py - Wikipedia BF16)

Usage:
    # Main (5 datasets)
    python generate_appendix_tables.py full_coverage_results/

    # BF16 special analysis
    python generate_appendix_tables.py special_results/ --special

    # Pin a different reference seed for fitted params
    python generate_appendix_tables.py full_coverage_results/ --param-seed 0
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import argparse
import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════

DATASETS = ["Wikipedia", "Cosmopedia", "Red Pajama", "RedPajama", "Pes2o", "C4"]

# Canonical display names
DS_DISPLAY = {
    "RedPajama": "Red Pajama",
    "Red Pajama": "Red Pajama",
    "Wikipedia_BF16": "Wikipedia (BF16)",
}

# Holdout splits - two labels per split:
#   header_label : short math notation for table column headers
#   caption_label: descriptive form for table captions
HOLDOUT_SPLITS = {
    "unified": {
        "header":  r"$\mathcal{H}$",
        "caption": r"Holdout ($\mathcal{H}$)",
    },
    "colinear": {
        "header":  r"$\mathcal{H}_{\mathrm{col}}$",
        "caption": r"Holdout--CO ($\mathcal{H}_{\mathrm{col}}$)",
    },
    "noncolinear": {
        "header":  r"$\mathcal{H}_{\mathrm{nc}}$",
        "caption": r"Holdout--NC ($\mathcal{H}_{\mathrm{nc}}$)",
    },
}

DESIGN_DISPLAY = {"CO": "Colinear", "NC": "Non-Colinear"}

# Mode display order
MODE_ORDER = ["Epoch-Expanded", "Final", "First", "Second"]

# Law display order
LAW_ORDER = ["Parametric-Epoch", "Chinchilla", "Droppo-Elibol", "Kaplan"]

# Per-mode parquet filename (relative to results_dir)
PARQUET_NAMES = {
    "main":    "full_coverage_all_seeds.parquet",
    "special": "full_coverage_bf16.parquet",
}

# Per-mode output filename suffix and dataset ordering
RUN_CONFIG = {
    "main": {
        "metrics_out": "appendix_full_metrics.tex",
        "params_out":  "appendix_fitted_params.tex",
        "dataset_order": ["Wikipedia", "Cosmopedia", "RedPajama", "Pes2o", "C4"],
        "label_suffix": "",
    },
    "special": {
        "metrics_out": "appendix_full_metrics_bf16.tex",
        "params_out":  "appendix_fitted_params_bf16.tex",
        "dataset_order": ["Wikipedia_BF16"],
        "label_suffix": "-bf16",
    },
}


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════

def _fmt(val, prec=4):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "---"
    return f"{val:.{prec}f}"


def _fmt_sci(val):
    """Format large/small numbers in LaTeX scientific notation."""
    if val is None or np.isnan(val):
        return "---"
    if abs(val) < 1e-2 or abs(val) > 1e4:
        b, e = f"{val:.2e}".split("e")
        b = float(b)
        exp = int(e)
        if abs(b - 1.0) < 0.05:
            return f"$10^{{{exp}}}$"
        return f"${b:.2f}\\times10^{{{exp}}}$"
    return f"${val:.4f}$"


def _win(val, prec=4):
    return f"\\win{{{_fmt(val, prec)}}}"


# ════════════════════════════════════════════════════════════════════════
# Per-dataset metric breakdown (Appendix E)
# ════════════════════════════════════════════════════════════════════════

def _fmt_ci(val, std, ci, prec=4):
    """Format a value with tiny σ/CI annotation."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "---"
    base = f"{val:.{prec}f}"
    ann = r"{\tiny\,$\sigma$=" + f"{std:.3f}" + r", CI=" + f"{ci:.3f}" + "}"
    return base + ann


def build_metric_table(df, dataset, split_key, split_header, split_caption,
                        n_seeds_label=30):
    """
    Build one LaTeX table for a (dataset, holdout_split) pair.

    Each row = (Epoch Mode, Law, Design) with mean R², RMSE across seeds.
    Includes σ and 95% CI as tiny annotations.
    \\win{} marks best R² per (Mode, Law) group.

    split_header  : compact math notation used in column headers
                    (e.g., r"$\\mathcal{H}$")
    split_caption : descriptive form used in the table caption
                    (e.g., r"Holdout ($\\mathcal{H}$)")
    n_seeds_label : number to print in caption (e.g., "Mean across N seeds")
    """
    r2_col = f"r2_{split_key}"
    rmse_col = f"rmse_{split_key}"

    sub = df[df["dataset"] == dataset].copy()
    if sub.empty or r2_col not in sub.columns:
        return None

    # Aggregate across seeds
    agg = sub.groupby(["mode", "law", "design"]).agg(
        r2_mean=(r2_col, "mean"),
        r2_std=(r2_col, "std"),
        rmse_mean=(rmse_col, "mean"),
        rmse_std=(rmse_col, "std"),
        r2_train_mean=("r2_train", "mean"),
        r2_train_std=("r2_train", "std"),
        rmse_train_mean=("rmse_train", "mean"),
        rmse_train_std=("rmse_train", "std"),
        n_seeds=(r2_col, "count"),
    ).reset_index()

    # 95% CI on the mean
    agg["r2_ci"] = 1.96 * agg["r2_std"] / np.sqrt(agg["n_seeds"])
    agg["rmse_ci"] = 1.96 * agg["rmse_std"] / np.sqrt(agg["n_seeds"])
    agg["r2_train_ci"] = 1.96 * agg["r2_train_std"] / np.sqrt(agg["n_seeds"])
    agg["rmse_train_ci"] = 1.96 * agg["rmse_train_std"] / np.sqrt(agg["n_seeds"])

    # Gap
    agg["gap"] = agg["r2_train_mean"] - agg["r2_mean"]

    # Sort
    mode_rank = {m: i for i, m in enumerate(MODE_ORDER)}
    law_rank = {l: i for i, l in enumerate(LAW_ORDER)}
    design_rank = {"CO": 0, "NC": 1}
    agg["_mode_rank"] = agg["mode"].map(mode_rank).fillna(99)
    agg["_law_rank"] = agg["law"].map(law_rank).fillna(99)
    agg["_design_rank"] = agg["design"].map(design_rank).fillna(99)
    agg = agg.sort_values(["_mode_rank", "_law_rank", "_design_rank"])

    # Find winners per (mode, law)
    winners = set()
    for (mode, law), grp in agg.groupby(["mode", "law"]):
        if len(grp) >= 2:
            best_idx = grp["r2_mean"].idxmax()
            winners.add(best_idx)

    ds_display = DS_DISPLAY.get(dataset, dataset)

    lines = [
        r"\begin{table}[ht!]",
        r"\centering",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{ll l c c c c c}",
        r"\toprule",
        r"\textbf{Epoch Mode} & \textbf{Law} & \textbf{Design} "
        r"& \textbf{R$^2$ Train} & \textbf{R$^2$ " + split_header + r"} "
        r"& \textbf{RMSE Train} & \textbf{RMSE " + split_header + r"} "
        r"& \textbf{Gap} \\",
        r"\midrule",
    ]

    grouped_modes = list(agg.groupby("mode", sort=False))
    for mi, (mode, mode_grp) in enumerate(grouped_modes):
        grouped_laws = list(mode_grp.groupby("law", sort=False))
        mode_nrows = len(mode_grp)
        first_mode = True

        for li, (law, law_grp) in enumerate(grouped_laws):
            law_nrows = len(law_grp)
            first_law = True

            for idx, row in law_grp.iterrows():
                is_winner = idx in winners
                design_label = DESIGN_DISPLAY.get(row["design"], row["design"])

                # R² holdout with CI annotation
                r2_val = row["r2_mean"]
                if is_winner:
                    r2_hold_str = f"\\win{{{_fmt_ci(r2_val, row['r2_std'], row['r2_ci'])}}}"
                else:
                    r2_hold_str = _fmt_ci(r2_val, row["r2_std"], row["r2_ci"])

                # Other cells with CI
                r2_train_str = _fmt_ci(row["r2_train_mean"], row["r2_train_std"], row["r2_train_ci"])
                rmse_train_str = _fmt_ci(row["rmse_train_mean"], row["rmse_train_std"], row["rmse_train_ci"])
                rmse_hold_str = _fmt_ci(row["rmse_mean"], row["rmse_std"], row["rmse_ci"])

                # Mode cell
                mode_cell = f"\\multirow{{{mode_nrows}}}{{*}}{{{mode}}}" if first_mode else ""
                first_mode = False

                # Law cell
                law_cell = f"\\multirow{{{law_nrows}}}{{*}}{{{law}}}" if first_law else ""
                first_law = False

                lines.append(
                    f"{mode_cell} & {law_cell} & {design_label} "
                    f"& {r2_train_str} & {r2_hold_str} "
                    f"& {rmse_train_str} & {rmse_hold_str} "
                    f"& {_fmt(row['gap'])} \\\\"
                )

            if li < len(grouped_laws) - 1:
                lines.append(r"\cmidrule{2-8}")

        if mi < len(grouped_modes) - 1:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        f"\\caption{{{ds_display} --- {split_caption}. "
        f"Mean across {n_seeds_label} seeds; "
        r"$\sigma$ = seed-to-seed std, CI = 95\% CI on the mean.}}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# Fitted parameter table (Appendix F) - one table per law, all datasets
# ════════════════════════════════════════════════════════════════════════

# Per-law config: (param_columns_in_parquet, latex_header_names, caption, label)
LAW_TABLE_CONFIG = {
    "Chinchilla": {
        "mode": "Final",
        "param_cols": ["param_0", "param_1", "param_2", "param_3", "param_4"],
        "param_headers": ["$E$", "$A$", r"$\alpha$", "$B$", r"$\beta$"],
        "caption": r"Chinchilla fitted parameters: $L(N,D)= E + A\,N^{-\alpha} + B\,D^{-\beta}$. Final epoch.",
        "label": "tab:chinchilla-params",
    },
    "Kaplan": {
        "mode": "Final",
        "param_cols": ["param_0", "param_1", "param_2", "param_3"],
        "param_headers": ["$N_c$", "$D_c$", r"$\alpha_N$", r"$\alpha_D$"],
        "extra_cols": [("ratio", r"$\alpha_N/\alpha_D$", lambda r: r["param_2"] / r["param_3"] if r["param_3"] != 0 else float("nan"))],
        "caption": r"Kaplan fitted parameters: $L(N,D)=\bigl[(N_c/N)^{\alpha_N/\alpha_D} + D_c/D\bigr]^{\alpha_D}$. Final epoch.",
        "label": "tab:kaplan-params",
    },
    "Droppo-Elibol": {
        "mode": "Final",
        "param_cols": ["param_0", "param_1", "param_2", "param_3", "param_4", "param_5"],
        "param_headers": [r"$L_\infty$", "$N_C$", "$D_C$", r"$\alpha_N$", r"$\alpha_D$", r"$\alpha$"],
        "caption": r"Droppo--Elibol fitted parameters. Final epoch.",
        "label": "tab:droppo-params",
    },
    "Parametric-Epoch": {
        "mode": "Epoch-Expanded",
        "param_cols": ["param_A", "param_alpha", "param_B", "param_beta", "param_E", "param_R_star_D", "param_R_star_N"],
        "param_headers": ["$A$", r"$\alpha$", "$B$", r"$\beta$", "$E$", "$R^*_D$", "$R^*_N$"],
        "caption": r"Parametric-Epoch fitted parameters. Epoch-Expanded mode.",
        "label": "tab:epoch-params",
    },
}


def build_param_table_per_law(df, law_name, dataset_order, label_suffix="",
                                seed=0):
    """
    Build a fitted-parameters table for one law across all datasets.
    Matches the paper's Appendix F format (Tables 41-44).

    Per-split holdout R² columns now use the standardized notation:
    $R^2_{\\mathcal{H}}$, $R^2_{\\mathcal{H}_{\\mathrm{col}}}$,
    $R^2_{\\mathcal{H}_{\\mathrm{nc}}}$.

    label_suffix: appended to LaTeX label so main and special tables don't
                  collide (e.g., "tab:chinchilla-params" vs
                  "tab:chinchilla-params-bf16").
    """
    cfg = LAW_TABLE_CONFIG.get(law_name)
    if cfg is None:
        return None

    mode = cfg["mode"]
    param_cols = cfg["param_cols"]
    param_headers = cfg["param_headers"]
    extra_cols = cfg.get("extra_cols", [])

    sub = df[(df["law"] == law_name) & (df["mode"] == mode) & (df["seed"] == seed)].copy()
    if sub.empty:
        return None

    # Header
    n_param_cols = len(param_headers) + len(extra_cols)
    col_spec = "ll" + "c" * n_param_cols + "cccc"
    header_params = " & ".join(param_headers + [h for _, h, _ in extra_cols])
    r2_headers = (
        r"$R^2_{\text{train}}$ & $R^2_{\mathcal{H}}$ & "
        r"$R^2_{\mathcal{H}_{\mathrm{col}}}$ & $R^2_{\mathcal{H}_{\mathrm{nc}}}$"
    )

    label_with_suffix = cfg["label"] + label_suffix

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{{cfg['caption']}}}",
        f"\\label{{{label_with_suffix}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        f"\\textbf{{Dataset}} & \\textbf{{Design}} & {header_params} & {r2_headers} \\\\",
        r"\midrule",
    ]

    for di, ds in enumerate(dataset_order):
        ds_sub = sub[sub["dataset"] == ds]
        if ds_sub.empty:
            continue

        ds_display = DS_DISPLAY.get(ds, ds)

        for design_idx, design in enumerate(["CO", "NC"]):
            row = ds_sub[ds_sub["design"] == design]
            if row.empty:
                continue
            row = row.iloc[0]

            # Dataset name only on first row
            ds_cell = ds_display if design_idx == 0 else ""

            # Param values
            param_cells = []
            for col in param_cols:
                val = row.get(col, np.nan)
                if pd.isna(val):
                    param_cells.append("---")
                else:
                    param_cells.append(_fmt_sci(val))

            # Extra computed columns (e.g., αN/αD for Kaplan)
            for _, _, fn in extra_cols:
                val = fn(row)
                param_cells.append(_fmt_sci(val))

            # R² values
            r2_train = _fmt(row.get("r2_train", np.nan))
            r2_uni = _fmt(row.get("r2_unified", np.nan))
            r2_co = _fmt(row.get("r2_colinear", np.nan))
            r2_nc = _fmt(row.get("r2_noncolinear", np.nan))

            # Short design label for table
            design_short = "CO" if design == "CO" else "NC"

            param_str = " & ".join(param_cells)
            lines.append(
                f"{ds_cell:14s} & {design_short} & {param_str} "
                f"& {r2_train} & {r2_uni} & {r2_co} & {r2_nc} \\\\"
            )

        if di < len(dataset_order) - 1:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate appendix tables from full_coverage parquet")
    parser.add_argument("results_dir",
                        help="Directory containing the input parquet "
                             "(full_coverage_all_seeds.parquet for --main, "
                             "full_coverage_bf16.parquet for --special)")
    parser.add_argument("--special", action="store_true",
                        help="Run on the BF16 special analysis output "
                             "(reads full_coverage_bf16.parquet, writes "
                             "*_bf16.tex)")
    parser.add_argument("--param-seed", type=int, default=0,
                        help="Seed to use for fitted parameter tables (default: 0)")
    args = parser.parse_args()

    run_mode = "special" if args.special else "main"
    cfg = RUN_CONFIG[run_mode]
    parquet_name = PARQUET_NAMES[run_mode]
    parquet_path = os.path.join(args.results_dir, parquet_name)

    if not os.path.exists(parquet_path):
        print(f"Error: {parquet_path} not found")
        if run_mode == "main":
            print("  Did you mean --special? "
                  "(reads full_coverage_bf16.parquet)")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    print(f"[{run_mode}] Loaded {len(df)} rows from {parquet_path}")
    print(f"  Datasets: {sorted(df['dataset'].unique())}")
    print(f"  Laws: {sorted(df['law'].unique())}")
    print(f"  Modes: {sorted(df['mode'].unique())}")
    n_seeds = df['seed'].nunique()
    print(f"  Seeds: {n_seeds}")

    # Normalize mode capitalization for display
    mode_map = {"final": "Final", "first": "First", "second": "Second",
                "Epoch-Expanded": "Epoch-Expanded"}
    df["mode"] = df["mode"].map(lambda m: mode_map.get(m, m))

    # Pick datasets to render: configured order, restricted to what's
    # actually present in the parquet
    datasets_present = set(df["dataset"].unique())
    dataset_order = [d for d in cfg["dataset_order"] if d in datasets_present]
    # Fallback: anything not in the configured order, alphabetically
    extras = sorted(datasets_present - set(dataset_order))
    dataset_order = dataset_order + extras

    # ── Appendix E: Per-dataset metric breakdown ──
    all_metric_tables = []
    for dataset in dataset_order:
        for split_key, labels in HOLDOUT_SPLITS.items():
            tex = build_metric_table(
                df, dataset, split_key,
                split_header=labels["header"],
                split_caption=labels["caption"],
                n_seeds_label=n_seeds,
            )
            if tex:
                all_metric_tables.append(tex)

    metric_path = os.path.join(args.results_dir, cfg["metrics_out"])
    with open(metric_path, "w") as f:
        f.write("\n\n\n".join(all_metric_tables))
    print(f"\nSaved: {metric_path} ({len(all_metric_tables)} tables)")

    # ── Appendix F: Fitted parameters (single seed, one table per law) ──
    all_param_tables = []
    for law in ["Chinchilla", "Kaplan", "Droppo-Elibol", "Parametric-Epoch"]:
        tex = build_param_table_per_law(
            df, law,
            dataset_order=dataset_order,
            label_suffix=cfg["label_suffix"],
            seed=args.param_seed,
        )
        if tex:
            all_param_tables.append(tex)

    param_path = os.path.join(args.results_dir, cfg["params_out"])
    with open(param_path, "w") as f:
        f.write("\n\n\n".join(all_param_tables))
    print(f"Saved: {param_path} ({len(all_param_tables)} tables)")

    # Print to terminal
    print("\n" + "=" * 60)
    print("METRIC TABLES (first dataset)")
    print("=" * 60)
    if all_metric_tables:
        for t in all_metric_tables[:3]:
            print(t)
            print()

    print("\n" + "=" * 60)
    print(f"PARAMETER TABLES (seed={args.param_seed})")
    print("=" * 60)
    for t in all_param_tables:
        print(t)
        print()


if __name__ == "__main__":
    main()