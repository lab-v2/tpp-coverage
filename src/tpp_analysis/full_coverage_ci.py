"""
Full-coverage seed variance analysis with parameter confidence intervals.

For each (dataset, law, mode, design), fits the scaling law N_SEEDS times
with different optimizer seeds at full coverage (all training data).

Produces:
  - R² and RMSE for train + all holdout splits, with ± σ on each
  - Fitted parameter values per seed, enabling parameter CIs

Cache prefix: pci_ (parameter CI) - won't collide with old fullcov_ caches.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scaling_laws.data import prepare_arrays
from scaling_laws.laws import (
    _fit_generic,
    kaplan_loss,
    chinchilla_loss,
    droppo_elibol_loss,
)


# ════════════════════════════════════════════════════════════════
# Parameter name registry
# ════════════════════════════════════════════════════════════════

# Canonical parameter names for each law (order matches fit_fn output)
PARAM_NAMES: Dict[str, List[str]] = {
    "Kaplan":        ["N_c", "D_c", "alpha_N", "alpha_D"],
    "Chinchilla":    ["E", "A", "alpha", "B", "beta"],
    "Droppo-Elibol": ["L_inf", "N_c", "D_c", "alpha_N", "alpha_D", "alpha"],
}

# LaTeX-friendly names for tables
PARAM_NAMES_LATEX: Dict[str, List[str]] = {
    "Kaplan":        [r"$N_c$", r"$D_c$", r"$\alpha_N$", r"$\alpha_D$"],
    "Chinchilla":    [r"$E$", r"$A$", r"$\alpha$", r"$B$", r"$\beta$"],
    "Droppo-Elibol": [r"$L_\infty$", r"$N_c$", r"$D_c$", r"$\alpha_N$", r"$\alpha_D$", r"$\alpha$"],
}

# Maximum number of parameters across all laws (for uniform columns)
MAX_PARAMS = max(len(v) for v in PARAM_NAMES.values())


# ════════════════════════════════════════════════════════════════
# Seed-aware fit wrappers (same bounds/samplers as laws.py)
# ════════════════════════════════════════════════════════════════

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


# Map law names to seed-aware fitters and loss functions
_SEED_FITTERS = {
    "Kaplan":        (_fit_kaplan_seed, kaplan_loss),
    "Chinchilla":    (_fit_chinchilla_seed, chinchilla_loss),
    "Droppo-Elibol": (_fit_droppo_elibol_seed, droppo_elibol_loss),
}


# ════════════════════════════════════════════════════════════════
# Core: run full coverage with multiple seeds
# ════════════════════════════════════════════════════════════════

def run_full_coverage_seeds(
    train_list,
    holdouts,
    law_name,
    mode,
    n_seeds=30,
    n_restarts=100,
    base_seed=0,
):
    """
    Fit the same training data n_seeds times with different optimizer seeds.

    Returns list of dicts, one per successful seed, with R² and RMSE
    for train and each holdout split, plus the fitted parameter values.
    """
    fit_fn, loss_fn = _SEED_FITTERS[law_name]
    n_params = len(PARAM_NAMES[law_name])

    N_tr, D_tr, L_tr = prepare_arrays(train_list, mode)
    if len(L_tr) < 5:
        return []

    # Prepare holdout arrays once
    holdout_arrays = {}
    for hname, hlist in holdouts.items():
        N_h, D_h, L_h = prepare_arrays(hlist, mode)
        if len(L_h) > 0:
            holdout_arrays[hname] = (N_h, D_h, L_h)

    seed_rows = []

    for s in range(n_seeds):
        seed = base_seed + s * 137

        try:
            params, sse, success = fit_fn(
                N_tr, D_tr, L_tr,
                n_restarts=n_restarts,
                seed=seed,
            )
            if params is None:
                continue

            row = {"seed": s}

            # ── Store fitted parameters ──
            for pi in range(MAX_PARAMS):
                if pi < n_params:
                    row[f"param_{pi}"] = float(params[pi])
                else:
                    row[f"param_{pi}"] = np.nan

            # ── Train metrics ──
            pred_tr = loss_fn(N_tr, D_tr, *params)
            row["r2_train"] = r2_score(L_tr, pred_tr)
            row["rmse_train"] = np.sqrt(mean_squared_error(L_tr, pred_tr))

            # ── Holdout metrics ──
            for hname, (N_h, D_h, L_h) in holdout_arrays.items():
                pred_h = loss_fn(N_h, D_h, *params)
                row[f"r2_{hname}"] = r2_score(L_h, pred_h)
                row[f"rmse_{hname}"] = np.sqrt(mean_squared_error(L_h, pred_h))

            seed_rows.append(row)

        except Exception:
            continue

    return seed_rows


# ════════════════════════════════════════════════════════════════
# Run all combos
# ════════════════════════════════════════════════════════════════

def run_all_full_coverage(
    prepared,
    laws=None,
    modes=None,
    n_seeds=30,
    n_restarts=100,
    base_seed=0,
    save_dir=None,
    verbose=True,
):
    """
    Run full-coverage seed variance for all datasets × laws × modes × designs.

    Returns a DataFrame with one row per (dataset, law, mode, design, seed)
    containing R², RMSE, and fitted parameter values.

    Cache prefix: pci_ (won't collide with old fullcov_ caches).
    """
    if laws is None:
        laws = ["Chinchilla", "Kaplan", "Droppo-Elibol"]
    if modes is None:
        modes = ["final", "first", "second"]

    all_rows = []

    for ds_name, data in prepared.items():
        for law_name in laws:
            for mode in modes:
                for design_label, train_source in [
                    ("CO", data.colinear_train),
                    ("NC", data.noncolinear_train),
                ]:
                    if not train_source:
                        continue

                    tag = f"{ds_name}_{law_name}_{mode}_{design_label}"

                    # Check cache (pci_ prefix)
                    if save_dir:
                        cache_path = os.path.join(save_dir, f"pci_fullcov_{tag}.parquet")
                        if os.path.exists(cache_path):
                            if verbose:
                                print(f"  ✅ Cached: {tag}")
                            cached = pd.read_parquet(cache_path)
                            all_rows.extend(cached.to_dict("records"))
                            continue

                    holdouts = {
                        "unified": data.unified_holdout,
                        "colinear": data.colinear_holdout,
                        "noncolinear": data.noncolinear_holdout,
                    }

                    if verbose:
                        print(f"  {tag} | {n_seeds} seeds × {n_restarts} restarts ...",
                              end=" ", flush=True)

                    t0 = time.time()
                    seed_rows = run_full_coverage_seeds(
                        train_source, holdouts, law_name, mode,
                        n_seeds=n_seeds,
                        n_restarts=n_restarts,
                        base_seed=base_seed,
                    )
                    elapsed = time.time() - t0

                    for row in seed_rows:
                        row["dataset"] = ds_name
                        row["law"] = law_name
                        row["mode"] = mode
                        row["design"] = design_label

                    all_rows.extend(seed_rows)

                    if verbose:
                        ok = len(seed_rows)
                        if ok > 0:
                            r2s = [r["r2_unified"] for r in seed_rows
                                   if "r2_unified" in r]
                            if r2s:
                                print(f"R²={np.mean(r2s):.4f} ± {np.std(r2s):.4f} "
                                      f"({ok}/{n_seeds} ok) [{elapsed:.0f}s]")
                            else:
                                print(f"({ok}/{n_seeds} ok) [{elapsed:.0f}s]")
                        else:
                            print(f"FAILED (0/{n_seeds}) [{elapsed:.0f}s]")

                    # Cache this combo
                    if save_dir and seed_rows:
                        combo_df = pd.DataFrame(seed_rows)
                        combo_df.to_parquet(cache_path, index=False)

    df = pd.DataFrame(all_rows)
    return df


# ════════════════════════════════════════════════════════════════
# Summary table (Table 3 replacement)
# ════════════════════════════════════════════════════════════════

def build_summary_table(df):
    """
    Build summary table aggregated across all datasets × laws × modes.

    For each (Split, Design), computes:
      - R² Mean, σ, CI95, Median, Std
      - RMSE Mean, σ, CI95, Median, Std
    """
    split_map = {
        "train":       "Train",
        "unified":     "Unified",
        "colinear":    "CO",
        "noncolinear": "NC",
    }

    rows = []
    for split_key, split_label in split_map.items():
        r2_col = f"r2_{split_key}"
        rmse_col = f"rmse_{split_key}"

        if r2_col not in df.columns:
            continue

        for design in ["CO", "NC"]:
            sub = df[df["design"] == design]
            if sub.empty:
                continue

            combos = sub.groupby(["dataset", "law", "mode"])
            combo_r2_means = combos[r2_col].mean()
            combo_r2_stds = combos[r2_col].std()
            combo_rmse_means = combos[rmse_col].mean()
            combo_rmse_stds = combos[rmse_col].std()
            combo_counts = combos[r2_col].count()
            mean_n_seeds = combo_counts.mean()

            r2_ci95 = combo_r2_stds.mean() * 1.96 / np.sqrt(mean_n_seeds)
            rmse_ci95 = combo_rmse_stds.mean() * 1.96 / np.sqrt(mean_n_seeds)

            rows.append({
                "Split": split_label,
                "Design": design,
                "R² Mean": combo_r2_means.mean(),
                "R² σ": combo_r2_stds.mean(),
                "R² CI95": r2_ci95,
                "R² Median": combo_r2_means.median(),
                "R² Std": combo_r2_means.std(),
                "RMSE Mean": combo_rmse_means.mean(),
                "RMSE σ": combo_rmse_stds.mean(),
                "RMSE CI95": rmse_ci95,
                "RMSE Median": combo_rmse_means.median(),
                "RMSE Std": combo_rmse_means.std(),
            })

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
# Per-combo table (Appendix detail)
# ════════════════════════════════════════════════════════════════

def build_per_combo_table(df):
    """
    One row per (dataset, law, mode, design) with R² and RMSE
    mean, std, p5, p95 for each split.
    """
    split_keys = ["train", "unified", "colinear", "noncolinear"]
    rows = []

    for (ds, law, mode, design), grp in df.groupby(
        ["dataset", "law", "mode", "design"]
    ):
        row = {
            "Dataset": ds,
            "Law": law,
            "Mode": mode,
            "Design": design,
            "N_seeds": len(grp),
        }

        for sk in split_keys:
            r2_col = f"r2_{sk}"
            rmse_col = f"rmse_{sk}"
            label = sk.capitalize()

            if r2_col in grp.columns:
                vals = grp[r2_col].dropna()
                if len(vals) > 0:
                    row[f"R²_{label}_mean"] = vals.mean()
                    row[f"R²_{label}_std"] = vals.std()
                    row[f"R²_{label}_p5"] = vals.quantile(0.05)
                    row[f"R²_{label}_p95"] = vals.quantile(0.95)

            if rmse_col in grp.columns:
                vals = grp[rmse_col].dropna()
                if len(vals) > 0:
                    row[f"RMSE_{label}_mean"] = vals.mean()
                    row[f"RMSE_{label}_std"] = vals.std()

        rows.append(row)

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
# Win rate table (Table 4 replacement)
# ════════════════════════════════════════════════════════════════

def build_winrate_table(df):
    """
    Seed-paired win rates: at each seed, compare CO vs NC R².

    Returns (per_combo_df, aggregated_df).
    """

    split_keys = ["unified", "colinear", "noncolinear"]
    split_labels = {"unified": "Unified", "colinear": "CO", "noncolinear": "NC"}

    per_combo_rows = []

    for (ds, law, mode) in df.groupby(["dataset", "law", "mode"]).groups:
        co = df[(df["dataset"] == ds) & (df["law"] == law) &
                (df["mode"] == mode) & (df["design"] == "CO")].sort_values("seed")
        nc = df[(df["dataset"] == ds) & (df["law"] == law) &
                (df["mode"] == mode) & (df["design"] == "NC")].sort_values("seed")

        if co.empty or nc.empty:
            continue

        # Pair by seed index
        n_paired = min(len(co), len(nc))
        co_arr = co.iloc[:n_paired]
        nc_arr = nc.iloc[:n_paired]

        for sk in split_keys:
            r2_col = f"r2_{sk}"
            if r2_col not in co_arr.columns:
                continue

            co_vals = co_arr[r2_col].values
            nc_vals = nc_arr[r2_col].values

            valid = ~(np.isnan(co_vals) | np.isnan(nc_vals))
            co_v = co_vals[valid]
            nc_v = nc_vals[valid]
            n = len(co_v)

            if n == 0:
                continue

            nc_wins = int(np.sum(nc_v > co_v))
            co_wins = int(np.sum(co_v > nc_v))
            adv = nc_v - co_v

            per_combo_rows.append({
                "Dataset": ds,
                "Law": law,
                "Mode": mode,
                "Split": split_labels[sk],
                "NC_wins": nc_wins,
                "CO_wins": co_wins,
                "N_paired": n,
                "NC_win_pct": nc_wins / n * 100,
                "Mean_NC_advantage": adv.mean(),
                "Std_NC_advantage": adv.std(),
            })

    per_combo_df = pd.DataFrame(per_combo_rows)

    # Aggregate by various slices
    agg_rows = []
    z = 1.96

    def _wilson(k, n):
        if n == 0:
            return 0, 0, 0
        p_hat = k / n
        denom = 1 + z ** 2 / n
        center = (p_hat + z ** 2 / (2 * n)) / denom
        margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
        return p_hat * 100, (center - margin) * 100, (center + margin) * 100

    for slice_name, col in [
        ("Split", "Split"),
        ("Dataset", "Dataset"),
        ("Law", "Law"),
        ("Mode", "Mode"),
    ]:
        for val, grp in per_combo_df.groupby(col):
            nc_w = int(grp["NC_wins"].sum())
            total = int(grp["N_paired"].sum())
            pct, lo, hi = _wilson(nc_w, total)
            agg_rows.append({
                "Slice": slice_name,
                "Value": val,
                "NC_wins": nc_w,
                "Total": total,
                "NC_pct": pct,
                "CI_lo_95": lo,
                "CI_hi_95": hi,
            })

    # Overall
    nc_total = int(per_combo_df["NC_wins"].sum())
    n_total = int(per_combo_df["N_paired"].sum())
    pct, lo, hi = _wilson(nc_total, n_total)
    agg_rows.append({
        "Slice": "Overall",
        "Value": "All",
        "NC_wins": nc_total,
        "Total": n_total,
        "NC_pct": pct,
        "CI_lo_95": lo,
        "CI_hi_95": hi,
    })

    agg_df = pd.DataFrame(agg_rows)
    return per_combo_df, agg_df


# ════════════════════════════════════════════════════════════════
# Parameter CI table (NEW - Appendix)
# ════════════════════════════════════════════════════════════════

def build_param_ci_table(df):
    """
    Build a table of fitted-parameter confidence intervals across seeds.

    For each (dataset, law, mode, design), reports per-parameter:
      - Mean, Std, 95% CI (mean ± 1.96 σ/√n), p2.5, p97.5

    Returns a DataFrame with one row per (dataset, law, mode, design, param_name).
    """
    rows = []

    for (ds, law, mode, design), grp in df.groupby(
        ["dataset", "law", "mode", "design"]
    ):
        pnames = PARAM_NAMES.get(law, [])
        _n_seeds = len(grp)

        for pi, pname in enumerate(pnames):
            col = f"param_{pi}"
            if col not in grp.columns:
                continue

            vals = grp[col].dropna()
            if len(vals) < 2:
                continue

            mean_val = vals.mean()
            std_val = vals.std()
            n = len(vals)
            ci95 = 1.96 * std_val / np.sqrt(n)

            rows.append({
                "Dataset": ds,
                "Law": law,
                "Mode": mode,
                "Design": design,
                "Param": pname,
                "Param_idx": pi,
                "N_seeds": n,
                "Mean": mean_val,
                "Std": std_val,
                "CI95_lo": mean_val - ci95,
                "CI95_hi": mean_val + ci95,
                "P2.5": vals.quantile(0.025),
                "P97.5": vals.quantile(0.975),
                "P5": vals.quantile(0.05),
                "P95": vals.quantile(0.95),
                "Median": vals.median(),
            })

    return pd.DataFrame(rows)


def build_param_ci_summary(df):
    """
    Condensed parameter CI table: one row per (law, design, param),
    averaging across datasets and modes.

    Reports Mean ± σ (averaged across combos) and the mean 95% CI width.
    """
    param_df = build_param_ci_table(df)
    if param_df.empty:
        return pd.DataFrame()

    rows = []
    for (law, design, param, pidx), grp in param_df.groupby(
        ["Law", "Design", "Param", "Param_idx"]
    ):
        rows.append({
            "Law": law,
            "Design": design,
            "Param": param,
            "Param_idx": pidx,
            "Mean (across combos)": grp["Mean"].mean(),
            "Std (across combos)": grp["Std"].mean(),
            "Mean CI95 width": (grp["CI95_hi"] - grp["CI95_lo"]).mean(),
            "Mean P2.5": grp["P2.5"].mean(),
            "Mean P97.5": grp["P97.5"].mean(),
            "N_combos": len(grp),
        })

    result = pd.DataFrame(rows).sort_values(["Law", "Design", "Param_idx"])
    return result


# ════════════════════════════════════════════════════════════════
# LaTeX generators
# ════════════════════════════════════════════════════════════════

def summary_to_latex(summary_df):
    """Generate LaTeX for the summary table (Table 3 replacement)."""
    lines = [
        r"\begin{table*}[t]",
        r"\small",
        r"\centering",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{$R^2$ and RMSE summary across evaluation splits and training designs.",
        r"Each mean reports seed-to-seed optimizer variance: $\sigma$ (single-fit spread) "
        r"and 95\% CI on the mean (from 30 optimizer seeds). Best per split in \textbf{bold}.}",
        r"\label{tab:summary}",
        r"\begin{tabular}{ll ccc ccc}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{\textbf{$R^2$}} & \multicolumn{3}{c}{\textbf{RMSE}} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}",
        r"\textbf{Split} & \textbf{Design} & Mean & Median & Std & Mean & Median & Std \\",
        r"\midrule",
    ]

    splits_seen = set()
    for _, row in summary_df.iterrows():
        split = row["Split"]
        design = row["Design"]

        if split not in splits_seen:
            if splits_seen:
                lines.append(r"\addlinespace")
            lines.append(rf"\multirow{{2}}{{*}}{{{split}}}")
            splits_seen.add(split)

        r2_mean = row["R² Mean"]
        r2_sig = row.get("R² σ", 0)
        r2_ci = row.get("R² CI95", 0)
        r2_med = row["R² Median"]
        r2_std = row["R² Std"]

        rmse_mean = row["RMSE Mean"]
        rmse_sig = row.get("RMSE σ", 0)
        rmse_ci = row.get("RMSE CI95", 0)
        rmse_med = row["RMSE Median"]
        rmse_std = row["RMSE Std"]

        r2_str = (
            rf"{r2_mean:.4f}{{\tiny\,$\sigma$={r2_sig:.3f}, CI={r2_ci:.3f}}}"
        )
        rmse_str = (
            rf"{rmse_mean:.4f}{{\tiny\,$\sigma$={rmse_sig:.3f}, CI={rmse_ci:.3f}}}"
        )

        lines.append(
            rf" & {design}  & {r2_str} & {r2_med:.4f} & {r2_std:.4f} "
            rf"& {rmse_str} & {rmse_med:.4f} & {rmse_std:.4f} \\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def winrate_to_latex(agg_df):
    """Generate LaTeX for the win rate table (Table 4 replacement)."""
    overall = agg_df[agg_df["Slice"] == "Overall"]
    if not overall.empty:
        ov = overall.iloc[0]
        caption_extra = (
            rf"Overall NC win rate: {ov['NC_pct']:.1f}\% "
            rf"({int(ov['NC_wins'])}/{int(ov['Total'])})."
        )
    else:
        caption_extra = ""

    lines = [
        r"\begin{table*}[t]",
        r"\small",
        r"\centering",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{Win rate breakdown (NC vs.\ CO design) with seed-paired comparisons. {caption_extra}}}",
        r"\label{tab:winrate}",
        r"\begin{tabular}{lr|lr|lr|lr}",
        r"\toprule",
        r"\multicolumn{2}{c|}{\textbf{Holdout Split}} & "
        r"\multicolumn{2}{c|}{\textbf{Dataset}} & "
        r"\multicolumn{2}{c|}{\textbf{Scaling Law}} & "
        r"\multicolumn{2}{c}{\textbf{Eval.\ Mode}} \\",
        r"\cmidrule(lr){1-2} \cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}",
        r"& \textbf{NC\%} & & \textbf{NC\%} & & \textbf{NC\%} & & \textbf{NC\%} \\",
        r"\midrule",
    ]

    # Gather per slice
    slices = {}
    for _, row in agg_df[agg_df["Slice"] != "Overall"].iterrows():
        s = row["Slice"]
        if s not in slices:
            slices[s] = []
        slices[s].append((row["Value"], row["NC_pct"]))

    # Build rows side by side
    cols = ["Split", "Dataset", "Law", "Mode"]
    max_rows = max(len(slices.get(c, [])) for c in cols) if slices else 0

    for i in range(max_rows):
        parts = []
        for c in cols:
            entries = slices.get(c, [])
            if i < len(entries):
                val, pct = entries[i]
                parts.append(rf"{val} & {pct:.1f}\%")
            else:
                parts.append("& ")
        lines.append(" & ".join(parts) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def param_ci_to_latex(df, per_combo=False):
    """
    Generate LaTeX for the parameter CI appendix table.

    If per_combo=False (default), produces a condensed table:
      one row per (Law, Design, Param) averaged across datasets/modes.

    If per_combo=True, produces the full table:
      one row per (Dataset, Law, Mode, Design, Param).
    """
    if per_combo:
        param_df = build_param_ci_table(df)
        return _param_ci_latex_full(param_df)
    else:
        summary = build_param_ci_summary(df)
        return _param_ci_latex_condensed(summary)


def _param_ci_latex_condensed(summary_df):
    """Condensed: Law × Design × Param, averaged across combos."""
    lines = [
        r"\begin{table*}[t]",
        r"\small",
        r"\centering",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Fitted parameter estimates (mean $\pm$ $\sigma$ across 30 optimizer seeds, "
        r"averaged over all dataset $\times$ epoch-mode combinations). "
        r"CI$_{95}$ width is the mean 95\% confidence interval width on the parameter mean.}",
        r"\label{tab:param_ci}",
        r"\begin{tabular}{ll l rrr}",
        r"\toprule",
        r"\textbf{Law} & \textbf{Design} & \textbf{Param} "
        r"& \textbf{Mean} & \textbf{$\sigma$} & \textbf{CI$_{95}$ width} \\",
        r"\midrule",
    ]

    last_law = None
    for _, row in summary_df.iterrows():
        law = row["Law"]
        design = row["Design"]
        param = row["Param"]
        mean_v = row["Mean (across combos)"]
        std_v = row["Std (across combos)"]
        ci_w = row["Mean CI95 width"]

        if law != last_law:
            if last_law is not None:
                lines.append(r"\addlinespace")
            last_law = law

        # Format large values in scientific notation
        def _fmt(v):
            if abs(v) >= 1e4:
                exp = int(np.floor(np.log10(abs(v))))
                mantissa = v / 10 ** exp
                return rf"{mantissa:.2f}\times10^{{{exp}}}"
            elif abs(v) < 0.001 and v != 0:
                exp = int(np.floor(np.log10(abs(v))))
                mantissa = v / 10 ** exp
                return rf"{mantissa:.2f}\times10^{{{exp}}}"
            else:
                return f"{v:.4f}"

        lines.append(
            rf"  {law} & {design} & {param} "
            rf"& ${_fmt(mean_v)}$ & ${_fmt(std_v)}$ & ${_fmt(ci_w)}$ \\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def _param_ci_latex_full(param_df):
    """Full: one row per (Dataset, Law, Mode, Design, Param)."""
    lines = [
        r"\begin{table*}[t]",
        r"\scriptsize",
        r"\centering",
        r"\setlength{\tabcolsep}{2pt}",
        r"\caption{Fitted parameter estimates with 95\% confidence intervals "
        r"(from 30 optimizer seeds) for every dataset $\times$ law $\times$ "
        r"mode $\times$ design combination. CI bounds are percentile-based (2.5th--97.5th).}",
        r"\label{tab:param_ci_full}",
        r"\begin{tabular}{lllll rrrr}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Law} & \textbf{Mode} & \textbf{Design} "
        r"& \textbf{Param} & \textbf{Mean} & \textbf{$\sigma$} "
        r"& \textbf{P$_{2.5}$} & \textbf{P$_{97.5}$} \\",
        r"\midrule",
    ]

    last_key = None
    for _, row in param_df.iterrows():
        key = (row["Dataset"], row["Law"], row["Mode"], row["Design"])

        if key != last_key:
            if last_key is not None:
                lines.append(r"\addlinespace")
            last_key = key

        def _fmt(v):
            if abs(v) >= 1e4:
                exp = int(np.floor(np.log10(abs(v))))
                mantissa = v / 10 ** exp
                return rf"{mantissa:.2f}\times10^{{{exp}}}"
            elif abs(v) < 0.001 and v != 0:
                exp = int(np.floor(np.log10(abs(v))))
                mantissa = v / 10 ** exp
                return rf"{mantissa:.2f}\times10^{{{exp}}}"
            else:
                return f"{v:.4f}"

        lines.append(
            rf"  {row['Dataset']} & {row['Law']} & {row['Mode']} & {row['Design']} "
            rf"& {row['Param']} "
            rf"& ${_fmt(row['Mean'])}$ & ${_fmt(row['Std'])}$ "
            rf"& ${_fmt(row['P2.5'])}$ & ${_fmt(row['P97.5'])}$ \\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)