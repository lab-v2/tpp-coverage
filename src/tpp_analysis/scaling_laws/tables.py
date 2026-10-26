"""
Summary tables with \\win{} winner highlighting for LaTeX.

Produces LaTeX table strings showing R² competition between Colinear
and Non-Colinear designs, evaluated on each of the three holdout sets.

Also produces aggregated stats tables (R²/RMSE summary by holdout type)
and win-rate breakdown tables.

Win-rate headline uses the **unified holdout** (the most general case),
not the average across all three holdout splits.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════════════════
# Build a tidy DataFrame from pipeline results
# ════════════════════════════════════════════════════════════════════════════

def build_summary_df(results: List[Dict]) -> pd.DataFrame:
    """
    Convert the list of result dicts (from `run_dataset_fits`) into a
    tidy DataFrame with one row per (dataset, law, design, mode) and
    columns for R² on each holdout type.
    """
    rows = []
    for r in results:
        rows.append({
            "Dataset":          r["dataset"],
            "Epoch Mode":       r["mode"],
            "Law":              r["law"],
            "Design":           r["design"],
            "R² Train":         r.get("r2_train", np.nan),
            "RMSE Train":       r.get("rmse_train", np.nan),
            "R² Unified":       r.get("r2_unified", np.nan),
            "RMSE Unified":     r.get("rmse_unified", np.nan),
            "R² Colinear":      r.get("r2_colinear", np.nan),
            "RMSE Colinear":    r.get("rmse_colinear", np.nan),
            "R² Non-Colinear":  r.get("r2_noncolinear", np.nan),
            "RMSE Non-Colinear":r.get("rmse_noncolinear", np.nan),
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# Holdout column registry
# ════════════════════════════════════════════════════════════════════════════

HOLDOUT_COLUMNS = {
    "unified":      "R² Unified",
    "colinear":     "R² Colinear",
    "noncolinear":  "R² Non-Colinear",
}


# ════════════════════════════════════════════════════════════════════════════
# LaTeX helpers
# ════════════════════════════════════════════════════════════════════════════

def _fmt(val: float, precision: int = 4) -> str:
    """Format a float for LaTeX, returning ' - ' for NaN."""
    if np.isnan(val):
        return " - "
    return f"{val:.{precision}f}"


def _win(val: float, precision: int = 4) -> str:
    """Wrap a formatted value in \\win{}."""
    return f"\\win{{{_fmt(val, precision)}}}"


def _pct_win(val: float) -> str:
    """Wrap a percentage in \\win{}."""
    return f"\\win{{{val:.1f}\\%}}"


def _pct_fmt(val: float) -> str:
    """Format a percentage without \\win."""
    return f"{val:.1f}\\%"


# ════════════════════════════════════════════════════════════════════════════
# Per-dataset comparison table (LaTeX)
# ════════════════════════════════════════════════════════════════════════════

def _latex_one_table(
    df: pd.DataFrame,
    r2_col: str,
    caption: str,
) -> str:
    """
    Build a LaTeX table for one (dataset, holdout_type) combination.

    Uses \\multirow for Epoch Mode and Law grouping, \\win{} for best
    R² holdout per (Epoch Mode, Law) group.
    """
    rmse_col = r2_col.replace("R²", "RMSE")
    holdout_short = r2_col.replace("R² ", "")

    # Prepare view
    view = df.copy()
    view["Gap"] = view["R² Train"] - view[r2_col]
    view = view.sort_values(["Epoch Mode", "Law", "Design"])

    # Find winners per (Epoch Mode, Law)
    winners = set()
    for (mode, law), grp in view.groupby(["Epoch Mode", "Law"]):
        if len(grp) >= 2 and grp[r2_col].notna().any():
            best_idx = grp[r2_col].idxmax()
            winners.add(best_idx)

    # Build rows grouped by (Epoch Mode, Law)
    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\renewcommand{\arraystretch}{1.2}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{ll l c c c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Epoch Mode} & \textbf{Law} & \textbf{Design} "
        r"& \textbf{R$^2$ Train} & \textbf{R$^2$ " + holdout_short + r"} "
        r"& \textbf{RMSE Train} & \textbf{RMSE " + holdout_short + r"} "
        r"& \textbf{Gap} \\"
    )
    lines.append(r"\midrule")

    grouped_modes = list(view.groupby("Epoch Mode", sort=False))

    for mode_idx, (mode, mode_grp) in enumerate(grouped_modes):
        grouped_laws = list(mode_grp.groupby("Law", sort=False))
        mode_row_count = len(mode_grp)

        first_mode_row = True
        for law_idx, (law, law_grp) in enumerate(grouped_laws):
            law_row_count = len(law_grp)
            first_law_row = True

            for _, row in law_grp.iterrows():
                is_winner = row.name in winners
                r2_hold_str = _win(row[r2_col]) if is_winner else _fmt(row[r2_col])
                rmse_hold_str = _fmt(row[rmse_col]) if rmse_col in row.index else " - "

                if first_mode_row:
                    mode_cell = f"\\multirow{{{mode_row_count}}}{{*}}{{{mode}}}"
                    first_mode_row = False
                else:
                    mode_cell = ""

                if first_law_row:
                    law_cell = f"\\multirow{{{law_row_count}}}{{*}}{{{law}}}"
                    first_law_row = False
                else:
                    law_cell = ""

                lines.append(
                    f"{mode_cell} & {law_cell} & {row['Design']} "
                    f"& {_fmt(row['R² Train'])} & {r2_hold_str} "
                    f"& {_fmt(row['RMSE Train'])} & {rmse_hold_str} "
                    f"& {_fmt(row['Gap'])} \\\\"
                )

            if law_idx < len(grouped_laws) - 1:
                lines.append(r"\cmidrule{2-8}")

        if mode_idx < len(grouped_modes) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# Aggregated stats tables (LaTeX)
# ════════════════════════════════════════════════════════════════════════════

def _latex_stats_table(
    df: pd.DataFrame,
    r2_col: str,
    rmse_col: str,
    caption: str,
) -> str:
    """
    Build one of the 4 aggregated stats tables (Train / Unified / Colinear / NC holdout).

    Each table has R² and RMSE sections with Mean, Median, Std per design.
    \\win{} marks the best value in each column.
    """
    designs = ["Colinear", "Non-Colinear"]

    stats = {}
    for design in designs:
        sub = df[df["Design"] == design]
        stats[design] = {
            "r2_mean":   sub[r2_col].mean(),
            "r2_median": sub[r2_col].median(),
            "r2_std":    sub[r2_col].std(),
            "rmse_mean":   sub[rmse_col].mean(),
            "rmse_median": sub[rmse_col].median(),
            "rmse_std":    sub[rmse_col].std(),
        }

    def _best(metric_key: str, higher_better: bool) -> str:
        vals = {d: stats[d][metric_key] for d in designs}
        if higher_better:
            return max(vals, key=vals.get)
        else:
            return min(vals, key=vals.get)

    best_r2_mean   = _best("r2_mean", True)
    best_r2_median = _best("r2_median", True)
    best_r2_std    = _best("r2_std", False)
    best_rmse_mean   = _best("rmse_mean", False)
    best_rmse_median = _best("rmse_median", False)
    best_rmse_std    = _best("rmse_std", False)

    def _cell(design, key, best_design):
        v = stats[design][key]
        return _win(v) if design == best_design else _fmt(v)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{l l c c c}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Metric} & \textbf{Design} & \textbf{Mean} & \textbf{Median} & \textbf{Std} \\")
    lines.append(r"\midrule")

    lines.append(r"\multirow{2}{*}{$R^2$}")
    for design in designs:
        lines.append(
            f" & {design:15s} & {_cell(design, 'r2_mean', best_r2_mean):15s} "
            f"& {_cell(design, 'r2_median', best_r2_median):15s} "
            f"& {_cell(design, 'r2_std', best_r2_std):15s} \\\\"
        )

    lines.append(r"\midrule")

    lines.append(r"\multirow{2}{*}{RMSE}")
    for design in designs:
        lines.append(
            f" & {design:15s} & {_cell(design, 'rmse_mean', best_rmse_mean):15s} "
            f"& {_cell(design, 'rmse_median', best_rmse_median):15s} "
            f"& {_cell(design, 'rmse_std', best_rmse_std):15s} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# Win-rate breakdown table (LaTeX)
# ════════════════════════════════════════════════════════════════════════════

def _winrate_ci(n_wins: int, n_total: int, confidence: float = 0.95) -> tuple:
    """Wilson score interval for binomial proportion."""
    if n_total == 0:
        return 0.0, 0.0, 0.0
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = n_wins / n_total
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    lo = max(0.0, centre - margin)
    hi = min(1.0, centre + margin)
    return 100 * p, 100 * lo, 100 * hi


def _latex_winrate_table(winners: pd.DataFrame, win_threshold: float = 60.0) -> str:
    """
    Main winrate table (Table 4): unified holdout only.

    All breakdown columns (Dataset, Law, Mode) computed on unified holdout.
    Caption includes 95% Wilson CI on the overall unified win rate.
    """
    if winners.empty:
        return "% No winners data available."

    # Filter to unified holdout only
    uni = winners[winners["Holdout"] == "Unified"]
    if uni.empty:
        return "% No unified holdout winners."

    def _pct_row(sub):
        n = len(sub)
        nc = (sub["Winner"] == "Non-Colinear").sum()
        co = n - nc
        pct, lo, hi = _winrate_ci(nc, n)
        return nc, co, n, pct, lo, hi

    # Overall
    nc_all, co_all, total_all, pct_all, lo_all, hi_all = _pct_row(uni)

    # Build breakdown rows: Dataset, Law, Mode
    rows_data = []
    for label, col in [("Dataset", "Dataset"), ("Law", "Law"), ("Mode", "Epoch Mode")]:
        for val in sorted(uni[col].unique()):
            nc, co, total, pct, lo, hi = _pct_row(uni[uni[col] == val])
            rows_data.append((f"{label}: {val}", nc, co, total, pct))

    # Find max rows per breakdown for column layout
    ds_rows = [(v, r) for v, r in [(val, _pct_row(uni[uni["Dataset"] == val]))
                for val in sorted(uni["Dataset"].unique())]]
    law_rows = [(v, r) for v, r in [(val, _pct_row(uni[uni["Law"] == val]))
                 for val in sorted(uni["Law"].unique())]]
    mode_rows = [(v, r) for v, r in [(val, _pct_row(uni[uni["Epoch Mode"] == val]))
                  for val in sorted(uni["Epoch Mode"].unique())]]

    # Build side-by-side table matching paper Table 4 format
    n_rows = max(len(ds_rows), len(law_rows), len(mode_rows))
    def pad(lst):
        return lst + [("", (0, 0, 0, 0.0, 0.0, 0.0))] * (n_rows - len(lst))
    ds_rows, law_rows, mode_rows = pad(ds_rows), pad(law_rows), pad(mode_rows)

    lines = []
    lines.append(r"\begin{table*}[!htbp]")
    lines.append(r"\small")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(
        f"\\caption{{Win rate breakdown (NC vs.\\ CO design) on unified holdout "
        f"with seed-paired comparisons. Overall NC win rate: "
        f"{pct_all:.1f}\\% [{lo_all:.1f}\\%, {hi_all:.1f}\\%] "
        f"({nc_all}/{total_all}).}}"
    )
    lines.append(r"\label{tab:winrate}")
    lines.append(r"\begin{tabular}{lr|lr|lr}")
    lines.append(r"\toprule")
    lines.append(
        r"\multicolumn{2}{c|}{\textbf{Dataset}} & "
        r"\multicolumn{2}{c|}{\textbf{Scaling Law}} & "
        r"\multicolumn{2}{c}{\textbf{Eval.\ Mode}} \\"
    )
    lines.append(r"\cmidrule(lr){1-2} \cmidrule(lr){3-4} \cmidrule(lr){5-6}")
    lines.append(
        r"& \textbf{NC\%} & & \textbf{NC\%} & & \textbf{NC\%} \\"
    )
    lines.append(r"\midrule")

    for i in range(n_rows):
        parts = []
        for col_rows in [ds_rows, law_rows, mode_rows]:
            label, (nc, co, total, pct, lo, hi) = col_rows[i]
            if label:
                pct_str = f"{pct:.1f}\\%"
            else:
                pct_str = ""
            parts.append(f"{label} & {pct_str}")
        lines.append(" & ".join(parts) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])

    return "\n".join(lines)


def _latex_winrate_table_appendix(winners: pd.DataFrame) -> str:
    """
    Appendix winrate table: full breakdown across all 3 holdout split types.
    """
    if winners.empty:
        return "% No winners data available."

    def _pct_row(sub):
        n = len(sub)
        nc = (sub["Winner"] == "Non-Colinear").sum()
        co = n - nc
        pct = nc / n * 100 if n > 0 else 0.0
        return nc, co, n, pct

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Win rate breakdown by holdout split definition.}")
    lines.append(r"\label{tab:winrate_appendix}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Category} & \textbf{NC Wins} & \textbf{CO Wins} & \textbf{Total} & \textbf{NC \%} \\")
    lines.append(r"\midrule")

    # Per holdout type
    for holdout in ["Unified", "Colinear", "Noncolinear"]:
        sub = winners[winners["Holdout"] == holdout]
        if sub.empty:
            continue
        nc, co, total, pct = _pct_row(sub)
        lines.append(f"\\textbf{{{holdout} Holdout}} & {nc} & {co} & {total} & {pct:.1f}\\% \\\\")

    lines.append(r"\midrule")

    # Per holdout × dataset
    for holdout in ["Unified", "Colinear", "Noncolinear"]:
        sub_h = winners[winners["Holdout"] == holdout]
        for ds in sorted(sub_h["Dataset"].unique()):
            nc, co, total, pct = _pct_row(sub_h[sub_h["Dataset"] == ds])
            lines.append(f"  {holdout} / {ds} & {nc} & {co} & {total} & {pct:.1f}\\% \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════

def display_comparison_tables(
    results: List[Dict],
    dataset_filter: Optional[str] = None,
) -> str:
    """
    Generate LaTeX comparison tables with \\win{} winner highlighting.

    Produces **three tables per dataset** (Unified, Colinear, Non-Colinear
    holdout) so you can see the competition from every angle.

    Parameters
    ----------
    results : list of dicts from `run_dataset_fits` or `run_full_analysis`
    dataset_filter : if given, only show tables for this dataset name

    Returns
    -------
    str : concatenated LaTeX table code
    """
    df_all = build_summary_df(results)
    datasets = [dataset_filter] if dataset_filter else sorted(df_all["Dataset"].unique())

    tables = []
    for ds in datasets:
        df_ds = df_all[df_all["Dataset"] == ds].copy()
        if df_ds.empty:
            tables.append(f"% No results for dataset: {ds}")
            continue

        for holdout_key, r2_col in HOLDOUT_COLUMNS.items():
            caption = f"{ds} - {holdout_key.replace('_', ' ').title()} Holdout"
            tables.append(_latex_one_table(df_ds, r2_col, caption))

    latex = "\n\n\n".join(tables)
    print(latex)
    return latex


def display_aggregated_stats(
    results: List[Dict],
    win_threshold: float = 60.0,
) -> str:
    """
    Generate the aggregated stats LaTeX tables: 4 summary tables
    (Train, Unified holdout, Colinear holdout, Non-Colinear holdout)
    plus one win-rate breakdown table.

    Win-rate headline uses unified holdout (the most general case).

    Parameters
    ----------
    results : list of dicts from pipeline
    win_threshold : NC% above this value gets \\win{} in the win-rate table

    Returns
    -------
    str : concatenated LaTeX table code
    """
    df = build_summary_df(results)
    winners = get_winners_summary(results)

    holdout_specs = [
        ("R² Train",         "RMSE Train",         "Train - $R^2$ and RMSE summary (black = best)"),
        ("R² Unified",       "RMSE Unified",       "Unified holdout - $R^2$ and RMSE summary (black = best)"),
        ("R² Colinear",      "RMSE Colinear",      "Colinear holdout - $R^2$ and RMSE summary (black = best)"),
        ("R² Non-Colinear",  "RMSE Non-Colinear",  "Non-colinear holdout - $R^2$ and RMSE summary (black = best)"),
    ]

    tables = []
    for r2_col, rmse_col, caption in holdout_specs:
        tables.append(_latex_stats_table(df, r2_col, rmse_col, caption))

    tables.append(_latex_winrate_table(winners, win_threshold))
    tables.append(_latex_winrate_table_appendix(winners))

    latex = "\n\n\n".join(tables)
    print(latex)
    return latex


def get_winners_summary(results: List[Dict]) -> pd.DataFrame:
    """
    Return a compact DataFrame showing the winner (Colinear vs Non-Colinear)
    for each (Dataset, Holdout Type, Epoch Mode, Law) combination.
    """
    df_all = build_summary_df(results)
    rows = []

    for ds in sorted(df_all["Dataset"].unique()):
        df_ds = df_all[df_all["Dataset"] == ds]

        for holdout_key, r2_col in HOLDOUT_COLUMNS.items():
            for (mode, law), grp in df_ds.groupby(["Epoch Mode", "Law"]):
                if len(grp) < 2:
                    continue
                best_row = grp.loc[grp[r2_col].idxmax()]
                rows.append({
                    "Dataset": ds,
                    "Holdout": holdout_key.title(),
                    "Epoch Mode": mode,
                    "Law": law,
                    "Winner": best_row["Design"],
                    "R²": best_row[r2_col],
                })

    return pd.DataFrame(rows)