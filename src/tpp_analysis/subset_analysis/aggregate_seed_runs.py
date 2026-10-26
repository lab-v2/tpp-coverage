#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
AGGREGATE SEED-VARIANCE RESULTS FOR REGIME A WINRATE SWEEP
═══════════════════════════════════════════════════════════════════════════

Reads <seed-dir>/seed_*/subset_bbox_*.csv produced by
subset_analysis_bbox_seeded.py (typically launched via dispatch_seeds.sh)
across multiple seeds, then:

  1. Computes per-seed NC-win rate on the κ_target sweep
     (identical sweep logic as winrate_vs_kappa_target.py)
  2. Aggregates mean + 95% CI (percentile bootstrap) across seeds
  3. Produces paper-ready plots with shaded CI bands:
       per_law_dualaxis_with_ci_first.pdf
       small_multiples_with_ci_first.pdf
       winrate_bars_with_ci_first.pdf
       raw_wins_combined_<DATASET>_<mode>.pdf  ← per-dataset 3-law overlay
  4. Emits per-seed-per-cell diagnostic CSV for outlier detection:
       per_seed_diagnostics.csv

Usage
─────
  # Default - reads from ./subset_bbox_seeds/ (matches dispatch_seeds.sh default)
  python aggregate_seed_runs.py --out-dir subset_bbox_seeds/aggregated

  # Explicit seed dir
  python aggregate_seed_runs.py \\
    --seed-dir subset_bbox_seeds \\
    --out-dir  subset_bbox_seeds/aggregated

NOTE: --seed-dir points at the PARENT containing seed_0/, seed_1/, ...,
NOT a single seed_N/ directory.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# Font sizing: 1.5× for paper readability, captions do the titling
_FS = 1.5
plt.rcParams.update({
    "font.size":         10 * _FS,
    "axes.labelsize":    10 * _FS,
    "axes.titlesize":    11 * _FS,
    "xtick.labelsize":    9 * _FS,
    "ytick.labelsize":    9 * _FS,
    "legend.fontsize":    9 * _FS,
})

LIT_BETA_EFF = {
    "Chinchilla":    0.28,
    "Kaplan":        0.095,
    "Droppo-Elibol": 0.08,
}

DEFAULT_KT_MIN   = 1e2
DEFAULT_KT_MAX   = 1e7
DEFAULT_KT_STEPS = 100

MIN_CELL_N         = 50    # truncate pooled (per-law) winrate if |A|<this
MIN_CELL_N_PERDS   = 10    # per-dataset cells: lower threshold since |A| is 1/5 of pooled
MIN_SEEDS_PER_CELL = 5     # warn if a (law, mode, cell) has fewer than this

MODE_ORDER = ["first", "second", "final"]
LAW_COLORS = {
    "Chinchilla":    "#1f77b4",
    "Droppo-Elibol": "#ff7f0e",
    "Kaplan":        "#2ca02c",
}


# ─────────────────────────────────────────────────────────────────────────
# Notebook-style constants (scoped to plot_raw_wins_combined_per_dataset_ci
# only - applied locally so they do not affect the other plots in this
# file, which keep their _FS-based styling).
#
# Locked to the "Shared Sizing Constants" section of
# Matches the figure notebooks: 9×9 panels, 25/20/20 font sizes,
# #D8D8D8 panel face with #a8a8a8 1.6pt gridlines, 1.5pt spines on all
# four sides, big inward ticks, square white legend frame.
# ─────────────────────────────────────────────────────────────────────────
_NB_PANEL_W       = 9
_NB_PANEL_H       = 9
_NB_AXIS_LABEL_FS = 25
_NB_TICK_LABEL_FS = 20
_NB_LEGEND_FS     = 20
_NB_TICK_MAJ_LEN  = 10
_NB_TICK_MAJ_WID  = 2.0
_NB_TICK_MIN_LEN  = 6
_NB_TICK_MIN_WID  = 1.2
_NB_LINE_WIDTH    = 2.5
_NB_SPINE_LW      = 1.5
_NB_CI_ALPHA      = 0.30
_NB_GRID_BG       = "#D8D8D8"
_NB_GRID_COLOR    = "#a8a8a8"
_NB_GRID_LW       = 1.6
_NB_LEGEND_ALPHA  = 0.9


# ─────────────────────────────────────────────────────────────────────────
# TPP parsing (same regex used by single-seed script)
# ─────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────
# Load per-seed bbox CSVs
# ─────────────────────────────────────────────────────────────────────────
def load_seed_dir(seed_dir: Path):
    """Combine CO and NC_BBOX rows per seed into one wide dataframe
    with columns: dataset, law, mode, subset_mask, target_tpps, rmse_co,
    rmse_nc, seed. Returns empty DataFrame if no CSVs found."""
    csvs = sorted(seed_dir.glob("subset_bbox_*.csv"))
    if not csvs:
        return pd.DataFrame()

    frames = []
    for csv in csvs:
        try:
            df = pd.read_csv(csv)
        except Exception as e:
            print(f"  ! failed to read {csv}: {e}", file=sys.stderr)
            continue
        if len(df) == 0: continue

        co = df[df["design"] == "CO"][
            ["dataset","law","mode","subset_mask","target_tpps","rmse_unified"]
        ].rename(columns={"rmse_unified": "rmse_co"})
        nc = df[df["design"] == "NC_BBOX"][
            ["dataset","law","mode","subset_mask","rmse_unified"]
        ].rename(columns={"rmse_unified": "rmse_nc"})

        key = ["dataset","law","mode","subset_mask"]
        merged = co.merge(nc, on=key, how="inner")
        frames.append(merged)

    if not frames: return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df


def load_all_seeds(seed_runs_dir: Path):
    """Loads seed_runs_dir/seed_*/subset_bbox_*.csv for all seeds found.
    Returns long-format DataFrame with an added 'seed' column."""
    seed_dirs = sorted(seed_runs_dir.glob("seed_*"))
    if not seed_dirs:
        raise SystemExit(
            f"No seed_*/ subdirectories in {seed_runs_dir}\n"
            f"Did you run dispatch_seeds.sh first? It populates "
            f"{seed_runs_dir}/seed_N/ for N in [start, end].")

    all_frames = []
    for sd in seed_dirs:
        m = re.match(r"seed_(\d+)$", sd.name)
        if not m: continue
        seed = int(m.group(1))
        df = load_seed_dir(sd)
        if df.empty:
            print(f"  ! seed_{seed} has no data, skipping")
            continue
        df["seed"] = seed
        all_frames.append(df)
        print(f"  seed_{seed}: {len(df):,} paired rows")

    if not all_frames:
        raise SystemExit("No valid seed data found")
    combined = pd.concat(all_frames, ignore_index=True)
    # Drop rows where either RMSE is NaN (fit failures)
    combined = combined.dropna(subset=["rmse_co","rmse_nc"]).reset_index(drop=True)
    return combined


# ─────────────────────────────────────────────────────────────────────────
# Per-seed sweep: reused from single-seed script
# ─────────────────────────────────────────────────────────────────────────
def sweep_seed(g: pd.DataFrame, kappa_targets: np.ndarray):
    """Given a single-seed slice (usually grouped by law/mode/dataset),
    compute (winrates, |Regime A|, NC wins count) at each κ_target."""
    tpps_list = [parse_tpp_list(s) for s in g["target_tpps"].values]
    good = np.array([t is not None for t in tpps_list])
    if good.sum() == 0:
        K = len(kappa_targets)
        return (np.full(K, np.nan),
                np.zeros(K, dtype=int),
                np.zeros(K, dtype=int))

    g2 = g[good].reset_index(drop=True)
    tpps_list = [t for t, m in zip(tpps_list, good) if m]
    laws = g2["law"].values

    V_Ks = np.array([
        float(np.mean((t**(-LIT_BETA_EFF[law]))**2)
               - np.mean(t**(-LIT_BETA_EFF[law]))**2)
        for t, law in zip(tpps_list, laws)
    ])
    num_over_K2 = np.array([
        (len(t) + float(np.sum(t**(-2.0 * LIT_BETA_EFF[law]))))**2
            / (len(t)**2)
        for t, law in zip(tpps_list, laws)
    ])
    nc_wins = (g2["rmse_nc"].values < g2["rmse_co"].values).astype(int)

    winrates  = np.zeros(len(kappa_targets))
    counts    = np.zeros(len(kappa_targets), dtype=int)
    wins_cnt  = np.zeros(len(kappa_targets), dtype=int)
    for i, kt in enumerate(kappa_targets):
        tau_K = num_over_K2 / kt
        in_A = V_Ks < tau_K
        counts[i] = int(in_A.sum())
        wins_cnt[i] = int(nc_wins[in_A].sum())
        winrates[i] = (100 * wins_cnt[i] / counts[i]) if counts[i] > 0 else np.nan
    return winrates, counts, wins_cnt


# ─────────────────────────────────────────────────────────────────────────
# Per-seed diagnostics: long DataFrame indexed by (seed, law, mode, kt).
# For per-(law, dataset) also track which dataset.
# ─────────────────────────────────────────────────────────────────────────
def build_per_seed_table(df: pd.DataFrame, kappa_targets: np.ndarray,
                           include_dataset=False):
    group_cols = ["seed","law","mode"] + (["dataset"] if include_dataset else [])
    rows = []
    for keys, g in df.groupby(group_cols):
        wr, cnt, wn = sweep_seed(g, kappa_targets)
        for kt, r, c, w in zip(kappa_targets, wr, cnt, wn):
            rec = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            rec.update({
                "kappa_target":      float(kt),
                "log10_kappa_target": float(np.log10(kt)),
                "regime_A_size":     int(c),
                "nc_wins":           int(w),
                "nc_wins_pct":       float(r) if np.isfinite(r) else np.nan,
            })
            rows.append(rec)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────
# Aggregate across seeds: mean + 95% CI via percentile bootstrap.
# Mask out per-seed values where |A| < MIN_CELL_N first.
# ─────────────────────────────────────────────────────────────────────────
def bootstrap_ci(vals, n_boot=2000, alpha=0.05, rng=None):
    """Return (mean, lower, upper) 95% CI via percentile bootstrap.
    `vals` is a 1D array of per-seed winrates; NaNs dropped."""
    if rng is None:
        rng = np.random.default_rng(12345)
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan, 0)
    if len(v) == 1:
        return (v[0], v[0], v[0], 1)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boot_means = np.mean(v[idx], axis=1)
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return (float(np.mean(v)), float(lo), float(hi), len(v))


def aggregate(per_seed_df, group_cols, kappa_targets, min_cell_n=0):
    """Collapse per-seed values to mean/CI per (group, κ_target) cell.
    Reports CI on: (1) winrate %, (2) raw NC-win count (numerator),
    (3) |Regime A| itself (which also varies per seed since fitted β
    affects regime classification).

    NaN-masking is optional via min_cell_n (default 0, no masking).
    When 0, every κ_target value is reported; the widening CI band
    at small |A| honestly signals where the estimate is noisy."""
    rng = np.random.default_rng(12345)
    out_rows = []
    for keys, g in per_seed_df.groupby(group_cols + ["kappa_target",
                                                       "log10_kappa_target"]):
        key_dict = dict(zip(group_cols + ["kappa_target", "log10_kappa_target"],
                            keys))
        # Winrate % CI (optionally masked)
        if min_cell_n > 0:
            g_wr = g[g["regime_A_size"] >= min_cell_n]
        else:
            g_wr = g
        wr_mean, wr_lo, wr_hi, n_valid = bootstrap_ci(
            g_wr["nc_wins_pct"].values, rng=rng)
        # Raw NC-win count CI (never masked)
        wins_mean, wins_lo, wins_hi, _ = bootstrap_ci(
            g["nc_wins"].values.astype(float), rng=rng)
        # |Regime A| CI (never masked - |A| is always well-defined)
        rsz_mean, rsz_lo, rsz_hi, _ = bootstrap_ci(
            g["regime_A_size"].values.astype(float), rng=rng)
        key_dict.update({
            "mean_nc_wins_pct":   wr_mean,
            "ci_lo":               wr_lo,
            "ci_hi":               wr_hi,
            "n_seeds_valid":       n_valid,
            "mean_nc_wins":        wins_mean,
            "nc_wins_ci_lo":       wins_lo,
            "nc_wins_ci_hi":       wins_hi,
            "mean_regime_A_size":  rsz_mean,
            "regime_A_ci_lo":      rsz_lo,
            "regime_A_ci_hi":      rsz_hi,
        })
        out_rows.append(key_dict)
    return pd.DataFrame(out_rows)


# ─────────────────────────────────────────────────────────────────────────
# Raw NC-wins COUNT plots with CI (numerator only)
# ─────────────────────────────────────────────────────────────────────────
def plot_raw_wins_per_law_ci(agg_law: pd.DataFrame, kappa_targets,
                                out_dir: Path, mode="first"):
    """Per-law overlay, Y axis = NC-wins count (not %).
    Monotone-decreasing curves from max at κ=10^2 to ~0 at κ=10^7,
    with shaded CI band across seeds."""
    g = agg_law[agg_law["mode"] == mode]
    laws = [l for l in ("Chinchilla","Droppo-Elibol","Kaplan")
            if l in g["law"].unique()]
    if not laws: return

    fig, ax = plt.subplots(figsize=(10 * 1.6, 5.5 * 1.6))
    xmin, xmax = np.log10(kappa_targets).min(), np.log10(kappa_targets).max()
    for law in laws:
        sub = g[g["law"] == law].sort_values("kappa_target")
        color = LAW_COLORS.get(law, "gray")
        ax.plot(sub["log10_kappa_target"], sub["mean_nc_wins"],
                 color=color, linewidth=2.0 * 1.5,
                 label=f"{law} (max = {sub['mean_nc_wins'].max():,.0f})")
        ax.fill_between(sub["log10_kappa_target"],
                         sub["nc_wins_ci_lo"], sub["nc_wins_ci_hi"],
                         color=color, alpha=0.25)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(r"$\log_{10}(\kappa_{\mathrm{target}})$")
    ax.set_ylabel("NC-wins count within Regime A  (mean ± 95% CI)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.5))
    fig.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(out_dir / f"raw_wins_per_law_with_ci_{mode}.{ext}",
                    bbox_inches="tight")
    plt.close(fig)


def plot_raw_wins_small_multiples_ci(agg_cell: pd.DataFrame,
                                         kappa_targets,
                                         out_dir: Path, mode="first"):
    """Per-(law, dataset) small multiples of NC-wins count with CI.
    Paper-ready styling: thicker shaded CI bands, no in-axis text
    annotations, clean row/column labels."""
    g = agg_cell[agg_cell["mode"] == mode]
    laws = sorted(g["law"].unique())
    datasets = sorted(g["dataset"].unique())
    if not laws or not datasets: return

    fig, axes = plt.subplots(len(laws), len(datasets),
                              figsize=(3.2 * 1.6 * len(datasets),
                                        2.8 * 1.6 * len(laws)),
                              sharex=True)
    if len(laws) == 1: axes = axes.reshape(1, -1)
    if len(datasets) == 1: axes = axes.reshape(-1, 1)
    xmin, xmax = np.log10(kappa_targets).min(), np.log10(kappa_targets).max()

    for i, law in enumerate(laws):
        for j, ds in enumerate(datasets):
            ax = axes[i, j]
            cell = g[(g["law"] == law) & (g["dataset"] == ds)].sort_values(
                "kappa_target")
            if len(cell) == 0:
                ax.set_visible(False); continue
            color = LAW_COLORS.get(law, "black")

            # Shaded CI band - thicker (alpha 0.45) so it's visible
            # in the paper PDF even where seed variance is tight.
            ax.fill_between(cell["log10_kappa_target"],
                              cell["nc_wins_ci_lo"],
                              cell["nc_wins_ci_hi"],
                              color=color, alpha=0.45, linewidth=0)
            # Mean line on top, slightly thicker
            ax.plot(cell["log10_kappa_target"], cell["mean_nc_wins"],
                     color=color, linewidth=2.2 * 1.5,
                     solid_capstyle="round")

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.20, linewidth=0.4)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
            ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.5))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if i == 0:
                ax.text(0.5, 1.04, ds, transform=ax.transAxes,
                         ha="center", va="bottom", fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{law}\nNC-wins count")
            if i == len(laws) - 1:
                ax.set_xlabel(r"$\log_{10}(\kappa_{\mathrm{target}})$")

    fig.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(out_dir / f"raw_wins_small_multiples_with_ci_{mode}.{ext}",
                    bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Per-dataset combined-law plots - landscape, notebook-style standardized.
# Five files per mode (one per dataset), each overlaying all 3 laws on the
# same axes with shaded CI bands. Used to demonstrate the phase shift
# (Regime A → Regime B) is dataset-agnostic: every law's NC-win count
# collapses around the same κ_target threshold.
# ─────────────────────────────────────────────────────────────────────────
def _nb_style_axes(ax):
    """Apply notebook-style 2-D axes: grey panel, grey grid, all four
    spines visible, big inward ticks. Mirrors style_ticks_2d() from the
    triptych notebook (visuals.ipynb)."""
    ax.tick_params(axis="both", which="major",
                    length=_NB_TICK_MAJ_LEN, width=_NB_TICK_MAJ_WID,
                    labelsize=_NB_TICK_LABEL_FS, direction="in",
                    top=True, right=True)
    ax.tick_params(axis="both", which="minor",
                    length=_NB_TICK_MIN_LEN, width=_NB_TICK_MIN_WID,
                    direction="in", top=True, right=True)
    for s in ("top", "bottom", "left", "right"):
        sp = ax.spines[s]
        sp.set_visible(True)
        sp.set_linewidth(_NB_SPINE_LW)
        sp.set_color("black")
    ax.set_facecolor(_NB_GRID_BG)
    ax.grid(True, which="major",
             color=_NB_GRID_COLOR, linewidth=_NB_GRID_LW,
             alpha=1.0, zorder=0)


def _nb_style_legend(legend):
    """White square frame with black border at fixed alpha - mirrors
    _style_legend() from the notebook."""
    if legend is None:
        return
    fr = legend.get_frame()
    fr.set_facecolor("white")
    fr.set_edgecolor("black")
    fr.set_linewidth(_NB_SPINE_LW)
    fr.set_alpha(_NB_LEGEND_ALPHA)
    try:
        fr.set_boxstyle("square", pad=0.4)
    except Exception:
        pass


def plot_raw_wins_combined_per_dataset_ci(agg_cell: pd.DataFrame,
                                              kappa_targets,
                                              out_dir: Path, mode="first"):
    """One PDF+PNG per dataset; all 3 laws overlaid with CI bands.

    Filename pattern:  raw_wins_combined_<DATASET>_<MODE>.{pdf,png}
    Panel size:        9 × 9 inches (notebook surface-plot baseline)
    Style:             locked to notebook's "Shared Sizing Constants",
                       scoped via rc_context so it does not affect the
                       other plots in this file.
    """
    g = agg_cell[agg_cell["mode"] == mode]
    datasets = sorted(g["dataset"].unique())
    if not datasets: return
    laws_order = [l for l in ("Chinchilla", "Droppo-Elibol", "Kaplan")
                   if l in g["law"].unique()]
    if not laws_order: return

    xmin = np.log10(kappa_targets).min()
    xmax = np.log10(kappa_targets).max()

    # rc_context scopes the STIX math font / family swap to this function
    # only. The file's other plots keep their default-mathtext styling.
    with matplotlib.rc_context({"mathtext.fontset": "stix",
                                  "font.family":      "STIXGeneral"}):
        for ds in datasets:
            sub_ds = g[g["dataset"] == ds]
            fig, ax = plt.subplots(figsize=(_NB_PANEL_W, _NB_PANEL_H))

            # CI bands first, mean lines on top - gives a clean stack
            # where overlapping bands blend without obscuring the lines.
            for law in laws_order:
                cell = sub_ds[sub_ds["law"] == law].sort_values("kappa_target")
                if len(cell) == 0: continue
                color = LAW_COLORS.get(law, "gray")
                ax.fill_between(cell["log10_kappa_target"],
                                  cell["nc_wins_ci_lo"],
                                  cell["nc_wins_ci_hi"],
                                  color=color, alpha=_NB_CI_ALPHA,
                                  linewidth=0, zorder=2)
                ax.plot(cell["log10_kappa_target"], cell["mean_nc_wins"],
                          color=color, linewidth=_NB_LINE_WIDTH,
                          solid_capstyle="round",
                          label=law, zorder=3)

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(bottom=0)
            ax.set_xlabel(r"$\log_{10}(\kappa_{\mathrm{target}})$",
                            fontsize=_NB_AXIS_LABEL_FS, labelpad=10)
            ax.set_ylabel("NC-wins count  (mean ± 95% CI)",
                            fontsize=_NB_AXIS_LABEL_FS, labelpad=10)

            # Integer-decade major / half-decade minor - same as the
            # other κ_target plots in this file.
            ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
            ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.5))

            _nb_style_axes(ax)

            leg = ax.legend(loc="upper right",
                             fontsize=_NB_LEGEND_FS,
                             frameon=True,
                             framealpha=_NB_LEGEND_ALPHA)
            _nb_style_legend(leg)

            # No title - captions handle naming downstream.
            fig.tight_layout()

            ds_tag = ds.replace(" ", "_")
            stem = f"raw_wins_combined_{ds_tag}_{mode}"
            for ext in ("pdf", "png"):
                fig.savefig(out_dir / f"{stem}.{ext}",
                              bbox_inches="tight", dpi=300)
            plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────
def plot_per_law_ci(agg_law: pd.DataFrame, kappa_targets: np.ndarray,
                     out_dir: Path, mode="first"):
    """Stacked 3-panel dual-axis plot: line with shaded CI + |Regime A|
    with its own lighter-grey CI band."""
    g = agg_law[agg_law["mode"] == mode]
    laws = [l for l in ("Chinchilla","Droppo-Elibol","Kaplan")
            if l in g["law"].unique()]
    if not laws: return

    fig, axes = plt.subplots(len(laws), 1,
                              figsize=(9 * 1.6, 3.2 * 1.6 * len(laws)),
                              sharex=True)
    if len(laws) == 1: axes = [axes]
    xmin, xmax = np.log10(kappa_targets).min(), np.log10(kappa_targets).max()

    for ax_w, law in zip(axes, laws):
        sub = g[g["law"] == law].sort_values("kappa_target")
        color = LAW_COLORS.get(law, "gray")

        ax_w.plot(sub["log10_kappa_target"], sub["mean_nc_wins_pct"],
                   color=color, linewidth=2.0 * 1.5)
        ax_w.fill_between(sub["log10_kappa_target"],
                           sub["ci_lo"], sub["ci_hi"],
                           color=color, alpha=0.25)
        ax_w.axhline(50, color="black", linestyle=":",
                      linewidth=1.0, alpha=0.5)
        ax_w.set_ylim(0, 100)
        ax_w.set_xlim(xmin, xmax)
        ax_w.set_ylabel(f"{law}\nNC-wins % in Regime A",
                         color=color)
        ax_w.tick_params(axis="y", labelcolor=color)
        ax_w.grid(True, alpha=0.25, linewidth=0.5)
        ax_w.xaxis.set_major_locator(mticker.MultipleLocator(1))
        ax_w.xaxis.set_minor_locator(mticker.MultipleLocator(0.5))

        ax_n = ax_w.twinx()
        # Mean |Regime A| as filled shape on secondary axis.
        ax_n.fill_between(sub["log10_kappa_target"], 0,
                            sub["mean_regime_A_size"],
                            color="gray", alpha=0.20, step="mid")
        ax_n.plot(sub["log10_kappa_target"], sub["mean_regime_A_size"],
                    color="gray", linewidth=1.2, alpha=0.7)
        ax_n.set_yscale("symlog", linthresh=10)
        ax_n.set_ylabel(r"$|\mathrm{Regime\,A}|$", color="gray")
        ax_n.tick_params(axis="y", labelcolor="gray")
        ax_n.set_ylim(bottom=0)

    axes[-1].set_xlabel(r"$\log_{10}(\kappa_{\mathrm{target}})$")
    fig.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(out_dir / f"per_law_dualaxis_with_ci_{mode}.{ext}",
                    bbox_inches="tight")
    plt.close(fig)


def plot_small_multiples_ci(agg_cell: pd.DataFrame, kappa_targets,
                              out_dir: Path, mode="first"):
    """5-dataset × 3-law small multiples, each with shaded CI band
    for winrate % and a lighter-grey CI band for |Regime A|."""
    g = agg_cell[agg_cell["mode"] == mode]
    laws = sorted(g["law"].unique())
    datasets = sorted(g["dataset"].unique())
    if not laws or not datasets: return

    fig, axes = plt.subplots(len(laws), len(datasets),
                              figsize=(3.2 * 1.6 * len(datasets),
                                        3.0 * 1.6 * len(laws)),
                              sharex=True, sharey=True)
    if len(laws) == 1: axes = axes.reshape(1, -1)
    if len(datasets) == 1: axes = axes.reshape(-1, 1)
    xmin, xmax = np.log10(kappa_targets).min(), np.log10(kappa_targets).max()

    for i, law in enumerate(laws):
        for j, ds in enumerate(datasets):
            ax = axes[i, j]
            cell = g[(g["law"] == law) & (g["dataset"] == ds)].sort_values(
                "kappa_target")
            if len(cell) == 0:
                ax.set_visible(False); continue
            color = LAW_COLORS.get(law, "black")
            ax.plot(cell["log10_kappa_target"], cell["mean_nc_wins_pct"],
                     color=color, linewidth=1.8 * 1.5)
            ax.fill_between(cell["log10_kappa_target"],
                              cell["ci_lo"], cell["ci_hi"],
                              color=color, alpha=0.25)
            ax.axhline(50, color="black", linestyle=":",
                         linewidth=0.8, alpha=0.5)
            ax.set_ylim(0, 100)
            ax.set_xlim(xmin, xmax)
            ax.grid(True, alpha=0.25, linewidth=0.4)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
            ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.5))

            if i == 0:
                ax.text(0.5, 1.02, ds, transform=ax.transAxes,
                         ha="center", va="bottom", fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{law}\nNC-wins %")
            if i == len(laws) - 1:
                ax.set_xlabel(r"$\log_{10}(\kappa_{\mathrm{target}})$")

            # |Regime A| on secondary axis (deterministic; no CI shown)
            ax_n = ax.twinx()
            ax_n.fill_between(cell["log10_kappa_target"], 0,
                                cell["mean_regime_A_size"],
                                color="gray", alpha=0.20, step="mid")
            ax_n.set_yscale("symlog", linthresh=10)
            ax_n.set_ylim(bottom=0)
            if j == len(datasets) - 1:
                ax_n.set_ylabel(r"$|\mathrm{Regime\,A}|$", color="gray")
                ax_n.tick_params(axis="y", labelcolor="gray")
            else:
                ax_n.set_yticks([])

    fig.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(out_dir / f"small_multiples_with_ci_{mode}.{ext}",
                    bbox_inches="tight")
    plt.close(fig)


def plot_bars_ci(agg_law: pd.DataFrame, out_dir: Path, mode="first",
                  kt_values=(1e2, 1e3, 1e4, 1e5, 1e6, 1e7)):
    """Bar chart at decade κ_target values, error bars = 95% CI."""
    g = agg_law[agg_law["mode"] == mode]
    laws = [l for l in ("Chinchilla","Droppo-Elibol","Kaplan")
            if l in g["law"].unique()]
    if not laws: return

    bar_data = {}
    for law in laws:
        sub = g[g["law"] == law].set_index("kappa_target")
        nearest = [sub.index[np.argmin(np.abs(sub.index - kt))]
                   for kt in kt_values]
        rows = sub.loc[nearest]
        bar_data[law] = {
            "kt": np.array(kt_values),
            "mean": rows["mean_nc_wins_pct"].values,
            "lo":   rows["ci_lo"].values,
            "hi":   rows["ci_hi"].values,
            "n_seeds": rows["n_seeds_valid"].values,
            "regime_A": rows["mean_regime_A_size"].values,
        }

    fig, ax = plt.subplots(figsize=(11 * 1.6, 6 * 1.6))
    n_kt = len(kt_values); n_laws = len(laws)
    bar_w = 0.8 / n_laws
    x_base = np.arange(n_kt)

    for i, law in enumerate(laws):
        d = bar_data[law]
        color = LAW_COLORS.get(law, "gray")
        xs = x_base + (i - (n_laws - 1) / 2) * bar_w
        y = d["mean"]
        yerr_lo = y - d["lo"]; yerr_hi = d["hi"] - y
        yerr = np.vstack([yerr_lo, yerr_hi])
        ax.bar(xs, y, width=bar_w * 0.95,
                color=color, alpha=0.85, label=law,
                edgecolor="black", linewidth=0.3 * 1.5,
                yerr=yerr, capsize=4,
                error_kw=dict(linewidth=1.5, ecolor="black"))
        for j, (xi, yi) in enumerate(zip(xs, y)):
            if not np.isfinite(yi): continue
            ax.text(xi, min(yi + yerr_hi[j] + 2, 105),
                    f"n={int(d['regime_A'][j])}",
                    ha="center", va="bottom", fontsize=8)

    ax.axhline(50, color="black", linestyle=":",
                linewidth=1.0, alpha=0.5, zorder=0)
    ax.set_xticks(x_base)
    ax.set_xticklabels([fr"$10^{{{int(np.log10(k))}}}$" for k in kt_values])
    ax.set_xlabel(r"$\kappa_{\mathrm{target}}$")
    ax.set_ylabel("NC-wins % within predicted Regime A (mean ± 95% CI)")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(out_dir / f"winrate_bars_with_ci_{mode}.{ext}",
                    bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-dir", type=Path,
                    default=Path("subset_bbox_seeds"),
                    help="Parent dir containing seed_*/ subdirs "
                         "(default: subset_bbox_seeds - matches "
                         "dispatch_seeds.sh default).")
    ap.add_argument("--out-dir",  required=True, type=Path,
                    help="Where to write aggregated CSVs + plots")
    ap.add_argument("--kt-min",    type=float, default=DEFAULT_KT_MIN)
    ap.add_argument("--kt-max",    type=float, default=DEFAULT_KT_MAX)
    ap.add_argument("--n-points",  type=int,   default=DEFAULT_KT_STEPS)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    kappa_targets = np.logspace(np.log10(args.kt_min),
                                 np.log10(args.kt_max),
                                 args.n_points)

    # ── [1/4] Load all seeds ──────────────────────────────────────
    print(f"[1/4] Loading seeds from {args.seed_dir}/")
    df = load_all_seeds(args.seed_dir)
    seeds_present = sorted(df["seed"].unique())
    print(f"\n  Seeds loaded: {seeds_present}  "
          f"({len(seeds_present)} total)")
    print(f"  Total paired rows: {len(df):,}")
    print(f"  Laws: {sorted(df['law'].unique())}")
    print(f"  Datasets: {sorted(df['dataset'].unique())}")
    print(f"  Modes: {sorted(df['mode'].unique())}")

    # ── [2/4] Per-seed tables ─────────────────────────────────────
    print("\n[2/4] Computing per-seed sweep tables...")
    per_seed_law = build_per_seed_table(df, kappa_targets,
                                           include_dataset=False)
    per_seed_cell = build_per_seed_table(df, kappa_targets,
                                            include_dataset=True)
    per_seed_law.to_csv(args.out_dir / "per_seed_sweep_by_law.csv",
                         index=False)
    per_seed_cell.to_csv(args.out_dir / "per_seed_sweep_by_cell.csv",
                          index=False)

    # ── [3/4] Aggregate ───────────────────────────────────────────
    print("[3/4] Aggregating across seeds (bootstrap 95% CI)...")
    agg_law  = aggregate(per_seed_law,  ["law","mode"], kappa_targets)
    agg_cell = aggregate(per_seed_cell, ["law","mode","dataset"],
                            kappa_targets)
    agg_law.to_csv(args.out_dir / "agg_sweep_by_law.csv", index=False)
    agg_cell.to_csv(args.out_dir / "agg_sweep_by_cell.csv", index=False)

    # ── [4/4] Plots ───────────────────────────────────────────────
    print("[4/4] Generating paper-ready plots...")
    for mode in [m for m in MODE_ORDER if m in df["mode"].unique()]:
        plot_per_law_ci(agg_law, kappa_targets, args.out_dir, mode=mode)
        plot_small_multiples_ci(agg_cell, kappa_targets, args.out_dir,
                                 mode=mode)
        plot_raw_wins_per_law_ci(agg_law, kappa_targets, args.out_dir,
                                    mode=mode)
        plot_raw_wins_small_multiples_ci(agg_cell, kappa_targets,
                                             args.out_dir, mode=mode)
        plot_raw_wins_combined_per_dataset_ci(agg_cell, kappa_targets,
                                                  args.out_dir, mode=mode)
    plot_bars_ci(agg_law, args.out_dir, mode="first")

    # ── Summary at every κ_target in the sweep ──────────────────
    print("\n" + "=" * 80)
    print(f"  MEAN ± 95% CI AT ALL {len(kappa_targets)} κ_target VALUES "
          f"(first epoch)")
    print("=" * 80)
    print(f"  {'log10(κ)':>10s}  {'κ_target':>12s}  "
          f"{'mean%':>7s}  {'[lo, hi]':>16s}  "
          f"{'|A|':>8s}  [|A| CI]")
    for law in sorted(agg_law["law"].unique()):
        print(f"\n  ── {law} ──")
        sub = agg_law[(agg_law["law"] == law) &
                       (agg_law["mode"] == "first")].sort_values(
                       "kappa_target")
        for _, r in sub.iterrows():
            mean  = r["mean_nc_wins_pct"]
            lo    = r["ci_lo"]; hi = r["ci_hi"]
            rsize = r["mean_regime_A_size"]
            rlo   = r["regime_A_ci_lo"]; rhi = r["regime_A_ci_hi"]
            mean_str = f"{mean:5.1f}%" if np.isfinite(mean) else "   -  "
            ci_str = (f"[{lo:5.1f},{hi:5.1f}]" if np.isfinite(lo) else
                       "       -      ")
            print(f"  {r['log10_kappa_target']:>10.3f}  "
                  f"{r['kappa_target']:>12.2e}  "
                  f"{mean_str}  {ci_str}  "
                  f"{rsize:>8.0f}  [{rlo:.0f},{rhi:.0f}]")

    print(f"\nArtifacts written to {args.out_dir}/")
    print("  Per-seed:      per_seed_sweep_by_{law,cell}.csv")
    print("  Aggregated:    agg_sweep_by_{law,cell}.csv")
    print("  Plots (winrate %):")
    print("    per_law_dualaxis_with_ci_*.{pdf,png}")
    print("    small_multiples_with_ci_*.{pdf,png}")
    print("    winrate_bars_with_ci_first.{pdf,png}")
    print("  Plots (raw win count / numerator):")
    print("    raw_wins_per_law_with_ci_*.{pdf,png}")
    print("    raw_wins_small_multiples_with_ci_*.{pdf,png}")
    print("    raw_wins_combined_<DATASET>_*.{pdf,png}    (5 per mode, 9×9)")


if __name__ == "__main__":
    main()