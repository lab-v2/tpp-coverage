"""
Visualization for incremental coverage analysis.

All plots follow the scaling_laws style (STIX fonts, individual figures).
Every plot function accepts a ``mode_filter`` parameter to show only
one epoch mode at a time (e.g. "Final", "First", "Second", "Epoch-Expanded").
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

# ── Palette ──────────────────────────────────────────────────────────────

LAW_COLORS = {
    "Kaplan": "#1f77b4",
    "Chinchilla": "#ff7f0e",
    "Droppo-Elibol": "#2ca02c",
    "Parametric-Epoch": "#d62728",
}

LAW_MARKERS = {
    "Kaplan": "o",
    "Chinchilla": "s",
    "Droppo-Elibol": "D",
    "Parametric-Epoch": "^",
}

DESIGN_COLORS = {
    "Colinear": "#1f77b4",
    "Non-Colinear": "#ff7f0e",
}


def _style_ax(ax, xlabel: str, ylabel: str, title: str = ""):
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)


def _law_legend(ax, laws_present=None, loc="lower right"):
    keys = laws_present if laws_present else list(LAW_COLORS.keys())
    handles = [
        Line2D([0], [0], color=LAW_COLORS.get(law, "gray"),
               marker=LAW_MARKERS.get(law, "."),
               markersize=6, linewidth=1.5, label=law)
        for law in keys if law in LAW_COLORS
    ]
    if handles:
        ax.legend(handles=handles, loc=loc, fontsize=9, framealpha=0.9)


def _apply_ylim(ax, y_min: Optional[float], y_max: Optional[float]):
    if y_min is not None or y_max is not None:
        current_min, current_max = ax.get_ylim()
        new_min = y_min if y_min is not None else current_min
        new_max = y_max if y_max is not None else current_max
        ax.set_ylim(new_min, new_max)


def _filter_mode(df: pd.DataFrame, mode_filter: Optional[str]) -> pd.DataFrame:
    """Filter dataframe to a single mode if specified."""
    if mode_filter is None:
        return df
    return df[df["mode"] == mode_filter]


def _mode_label(mode_filter: Optional[str]) -> str:
    """Return a label string for the mode."""
    if mode_filter is None:
        return "All Modes"
    return mode_filter


# ═══════════════════════════════════════════════════════════════════════════
# 1. R² convergence
# ═══════════════════════════════════════════════════════════════════════════

def plot_r2_convergence(
    df: pd.DataFrame,
    holdout_col: str = "r2_unified",
    x_col: str = "coverage_frac",
    mode_filter: Optional[str] = None,
    min_coverage: Optional[float] = None,
    figsize: Tuple[float, float] = (14, 5),
    suptitle: str = "",
    y_min: Optional[float] = -0.5,
    y_max: Optional[float] = 1.05,
) -> plt.Figure:
    """
    R² vs. coverage fraction, one line per law.
    Two side-by-side panels: Colinear-trained (left), NC-trained (right).
    """
    df = _filter_mode(df, mode_filter)
    if min_coverage is not None:
        df = df[df[x_col] >= min_coverage]

    designs = sorted(df["design"].unique())
    fig, axes = plt.subplots(1, len(designs), figsize=figsize, sharey=True)
    if len(designs) == 1:
        axes = [axes]

    ylabel = holdout_col.replace("r2_", "R² ").replace("_", " ").title()

    for ax, design in zip(axes, designs):
        sub = df[(df["design"] == design) & df["success"]]
        laws_seen = []
        for law_name, grp in sub.groupby("law"):
            grp = grp.sort_values(x_col)
            ax.plot(
                grp[x_col], grp[holdout_col],
                color=LAW_COLORS.get(law_name, "gray"),
                marker=LAW_MARKERS.get(law_name, "."),
                markersize=5, linewidth=1.5,
                label=law_name,
            )
            laws_seen.append(law_name)
        _style_ax(ax, "Coverage Fraction", ylabel if ax == axes[0] else "",
                   f"{design}-Trained")
        ax.set_xlim(-0.02, 1.05)
        _apply_ylim(ax, y_min, y_max)
        _law_legend(ax, laws_seen)

    title = suptitle or f"R² Convergence - {_mode_label(mode_filter)}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 2. ΔR² from full coverage
# ═══════════════════════════════════════════════════════════════════════════

def plot_delta_convergence(
    df: pd.DataFrame,
    holdout_col: str = "r2_unified",
    x_col: str = "coverage_frac",
    mode_filter: Optional[str] = None,
    min_coverage: Optional[float] = None,
    figsize: Tuple[float, float] = (14, 5),
    suptitle: str = "",
    y_min: Optional[float] = -1.0,
    y_max: Optional[float] = 0.5,
) -> plt.Figure:
    """
    ΔR² (difference from full-coverage R²) vs. coverage.
    """
    delta_col = f"delta_{holdout_col}"
    if delta_col not in df.columns:
        raise KeyError(f"Column {delta_col!r} not found - run compute_deltas() first.")

    df = _filter_mode(df, mode_filter)
    if min_coverage is not None:
        df = df[df[x_col] >= min_coverage]

    designs = sorted(df["design"].unique())
    fig, axes = plt.subplots(1, len(designs), figsize=figsize, sharey=True)
    if len(designs) == 1:
        axes = [axes]

    for ax, design in zip(axes, designs):
        sub = df[(df["design"] == design) & df["success"]]
        laws_seen = []
        for law_name, grp in sub.groupby("law"):
            grp = grp.sort_values(x_col)
            ax.plot(
                grp[x_col], grp[delta_col],
                color=LAW_COLORS.get(law_name, "gray"),
                marker=LAW_MARKERS.get(law_name, "."),
                markersize=5, linewidth=1.5,
            )
            laws_seen.append(law_name)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        label = holdout_col.replace("r2_", "").replace("_", " ").title()
        _style_ax(
            ax, "Coverage Fraction",
            f"ΔR² ({label})" if ax == axes[0] else "",
            f"{design}-Trained",
        )
        ax.set_xlim(-0.02, 1.05)
        _apply_ylim(ax, y_min, y_max)
        _law_legend(ax, laws_seen, loc="lower right")

    title = suptitle or f"ΔR² from Full Coverage - {_mode_label(mode_filter)}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 3. CO vs NC direct overlay
# ═══════════════════════════════════════════════════════════════════════════

def plot_co_vs_nc_overlay(
    df: pd.DataFrame,
    holdout_col: str = "r2_unified",
    x_col: str = "coverage_frac",
    mode_filter: Optional[str] = None,
    min_coverage: Optional[float] = None,
    figsize: Tuple[float, float] = (14, 8),
    suptitle: str = "",
    y_min: Optional[float] = -0.5,
    y_max: Optional[float] = 1.05,
) -> plt.Figure:
    """
    One subplot per law. Each overlays CO vs NC R² curves.
    """
    df = _filter_mode(df, mode_filter)
    if min_coverage is not None:
        df = df[df[x_col] >= min_coverage]

    laws = sorted(df["law"].unique())
    n = len(laws)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    axes = np.atleast_2d(axes)

    ylabel = holdout_col.replace("r2_", "R² ").replace("_", " ").title()

    for idx, law_name in enumerate(laws):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        for design in sorted(df["design"].unique()):
            sub = df[(df["law"] == law_name) & (df["design"] == design) & df["success"]]
            sub = sub.sort_values(x_col)
            ax.plot(
                sub[x_col], sub[holdout_col],
                color=DESIGN_COLORS.get(design, "gray"),
                marker="o" if design == "Colinear" else "s",
                markersize=5, linewidth=1.5,
                label=design,
            )
        _style_ax(ax, "Coverage Fraction", ylabel if c == 0 else "", law_name)
        ax.set_xlim(-0.02, 1.05)
        _apply_ylim(ax, y_min, y_max)
        ax.legend(fontsize=9, loc="lower right", framealpha=0.9)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    title = suptitle or f"CO vs NC - {_mode_label(mode_filter)}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 4. Parameter trajectories
# ═══════════════════════════════════════════════════════════════════════════

def _extract_param_series(df: pd.DataFrame, law_name: str, design: str) -> pd.DataFrame:
    sub = df[(df["law"] == law_name) & (df["design"] == design) & df["success"]].copy()
    sub = sub.sort_values("step")

    if sub.empty:
        return pd.DataFrame()

    sample = sub["params"].iloc[0]
    if isinstance(sample, dict):
        keys = list(sample.keys())
        for k in keys:
            sub[f"param_{k}"] = sub["params"].apply(lambda p: p[k] if isinstance(p, dict) else np.nan)
    elif isinstance(sample, (list, np.ndarray)):
        for i in range(len(sample)):
            sub[f"param_{i}"] = sub["params"].apply(
                lambda p: p[i] if isinstance(p, (list, np.ndarray)) and len(p) > i else np.nan
            )
    return sub


def plot_parameter_trajectories(
    df: pd.DataFrame,
    law_name: str,
    param_names: Optional[List[str]] = None,
    mode_filter: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 8),
) -> plt.Figure:
    """
    Show how fitted parameters drift as coverage increases.
    """
    df = _filter_mode(df, mode_filter)

    from scaling_laws.laws import LAWS as _LAWS
    law_info = next((l for l in _LAWS if l.name == law_name), None)
    if param_names is None and law_info is not None:
        param_names = law_info.param_names

    designs = sorted(df["design"].unique())
    dfs_by_design = {d: _extract_param_series(df, law_name, d) for d in designs}

    sample_df = next((v for v in dfs_by_design.values() if not v.empty), pd.DataFrame())
    if sample_df.empty:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, f"No data for {law_name}", ha="center", va="center")
        return fig

    pcols = sorted([c for c in sample_df.columns if c.startswith("param_")])
    n = len(pcols)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, pcol in enumerate(pcols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        nice_name = pcol.replace("param_", "")
        if param_names and idx < len(param_names):
            nice_name = param_names[idx]

        for design in designs:
            sub = dfs_by_design[design]
            if sub.empty or pcol not in sub.columns:
                continue
            ax.plot(
                sub["coverage_frac"], sub[pcol],
                color=DESIGN_COLORS.get(design, "gray"),
                marker="o" if design == "Colinear" else "s",
                markersize=4, linewidth=1.2,
                label=design,
            )
        _style_ax(ax, "Coverage", nice_name, "")
        ax.legend(fontsize=8, loc="best")

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    mode_str = f" ({_mode_label(mode_filter)})" if mode_filter else ""
    fig.suptitle(f"{law_name} - Parameter Trajectories{mode_str}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 5. R² heatmap (law × step)
# ═══════════════════════════════════════════════════════════════════════════

def plot_r2_heatmap(
    df: pd.DataFrame,
    design: str,
    holdout_col: str = "r2_unified",
    mode_filter: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 4),
) -> plt.Figure:
    """
    Heatmap: rows = laws, columns = coverage steps.
    """
    df = _filter_mode(df, mode_filter)
    sub = df[(df["design"] == design) & df["success"]].copy()
    pivot = sub.pivot_table(
        index="law", columns="step", values=holdout_col, aggfunc="first",
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    cmap = plt.cm.RdYlGn
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_xlabel("Step", fontsize=11)
    mode_str = f" - {_mode_label(mode_filter)}" if mode_filter else ""
    ax.set_title(f"{design} - R² Heatmap{mode_str}", fontsize=12, fontweight="bold")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="R²")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 6. Summary dashboard (multi-panel)
# ═══════════════════════════════════════════════════════════════════════════

def plot_summary_dashboard(
    df: pd.DataFrame,
    holdout_col: str = "r2_unified",
    mode_filter: Optional[str] = None,
    min_coverage: Optional[float] = None,
    figsize: Tuple[float, float] = (16, 12),
    suptitle: str = "",
    y_min_r2: Optional[float] = -0.5,
    y_max_r2: Optional[float] = 1.05,
    y_min_delta: Optional[float] = -1.0,
    y_max_delta: Optional[float] = 0.5,
) -> plt.Figure:
    """
    4-panel summary:
        (0,0) R² convergence - CO      (0,1) R² convergence - NC
        (1,0) ΔR² convergence - CO     (1,1) ΔR² convergence - NC
    """
    delta_col = f"delta_{holdout_col}"
    df = _filter_mode(df, mode_filter)
    if min_coverage is not None:
        df = df[df["coverage_frac"] >= min_coverage]

    designs = sorted(df["design"].unique())

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ylabel_r2 = holdout_col.replace("r2_", "R² ").replace("_", " ").title()
    label_short = holdout_col.replace("r2_", "").replace("_", " ").title()

    for col_idx, design in enumerate(designs[:2]):
        sub = df[(df["design"] == design) & df["success"]]

        # Top row: R²
        ax = axes[0, col_idx]
        laws_seen = []
        for law_name, grp in sub.groupby("law"):
            grp = grp.sort_values("coverage_frac")
            ax.plot(
                grp["coverage_frac"], grp[holdout_col],
                color=LAW_COLORS.get(law_name, "gray"),
                marker=LAW_MARKERS.get(law_name, "."),
                markersize=5, linewidth=1.5,
            )
            laws_seen.append(law_name)
        _style_ax(ax, "", ylabel_r2 if col_idx == 0 else "", f"{design}-Trained")
        ax.set_xlim(-0.02, 1.05)
        _apply_ylim(ax, y_min_r2, y_max_r2)
        _law_legend(ax, laws_seen)

        # Bottom row: ΔR²
        ax = axes[1, col_idx]
        if delta_col in df.columns:
            for law_name, grp in sub.groupby("law"):
                grp = grp.sort_values("coverage_frac")
                if delta_col in grp.columns:
                    ax.plot(
                        grp["coverage_frac"], grp[delta_col],
                        color=LAW_COLORS.get(law_name, "gray"),
                        marker=LAW_MARKERS.get(law_name, "."),
                        markersize=5, linewidth=1.5,
                    )
            ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        _style_ax(
            ax, "Coverage Fraction",
            f"ΔR² ({label_short})" if col_idx == 0 else "",
            "",
        )
        ax.set_xlim(-0.02, 1.05)
        _apply_ylim(ax, y_min_delta, y_max_delta)

    for col_idx in range(len(designs), 2):
        axes[0, col_idx].set_visible(False)
        axes[1, col_idx].set_visible(False)

    title = suptitle or f"Summary - {_mode_label(mode_filter)}"
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 7. Rate of change
# ═══════════════════════════════════════════════════════════════════════════

def plot_r2_rate_of_change(
    df: pd.DataFrame,
    holdout_col: str = "r2_unified",
    mode_filter: Optional[str] = None,
    min_coverage: Optional[float] = None,
    figsize: Tuple[float, float] = (14, 5),
    suptitle: str = "",
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> plt.Figure:
    """
    Discrete derivative: Δ(R²) / Δ(coverage) at each step.
    """
    df = _filter_mode(df, mode_filter)
    if min_coverage is not None:
        df = df[df["coverage_frac"] >= min_coverage]

    designs = sorted(df["design"].unique())
    fig, axes = plt.subplots(1, len(designs), figsize=figsize, sharey=True)
    if len(designs) == 1:
        axes = [axes]

    for ax, design in zip(axes, designs):
        sub = df[(df["design"] == design) & df["success"]]
        laws_seen = []
        for law_name, grp in sub.groupby("law"):
            grp = grp.sort_values("coverage_frac").reset_index(drop=True)
            if len(grp) < 2:
                continue
            cov = grp["coverage_frac"].values
            r2 = grp[holdout_col].values
            dr2 = np.diff(r2) / np.diff(cov)
            mid_cov = (cov[:-1] + cov[1:]) / 2
            ax.plot(
                mid_cov, dr2,
                color=LAW_COLORS.get(law_name, "gray"),
                marker=LAW_MARKERS.get(law_name, "."),
                markersize=4, linewidth=1.2,
            )
            laws_seen.append(law_name)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        label = holdout_col.replace("r2_", "").replace("_", " ").title()
        _style_ax(ax, "Coverage Fraction",
                   f"dR²/d(coverage) - {label}" if ax == axes[0] else "",
                   f"{design}-Trained")
        ax.set_xlim(-0.02, 1.05)
        _apply_ylim(ax, y_min, y_max)
        _law_legend(ax, laws_seen, loc="upper right")

    title = suptitle or f"Rate of Change - {_mode_label(mode_filter)}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 8. Mode comparison - all modes on one plot for a single law
# ═══════════════════════════════════════════════════════════════════════════

MODE_COLORS = {
    "Final": "#1b9e77",
    "First": "#d95f02",
    "Second": "#7570b3",
    "Epoch-Expanded": "#e7298a",
}

MODE_MARKERS = {
    "Final": "o",
    "First": "s",
    "Second": "D",
    "Epoch-Expanded": "^",
}


def plot_mode_comparison(
    df: pd.DataFrame,
    law_name: str,
    holdout_col: str = "r2_unified",
    x_col: str = "coverage_frac",
    min_coverage: Optional[float] = None,
    figsize: Tuple[float, float] = (14, 5),
    suptitle: str = "",
    y_min: Optional[float] = -0.5,
    y_max: Optional[float] = 1.05,
) -> plt.Figure:
    """
    For a single law, overlay all modes (Final/First/Second/Epoch-Expanded)
    to compare how epoch handling affects convergence.
    """
    sub = df[(df["law"] == law_name) & df["success"]].copy()
    if min_coverage is not None:
        sub = sub[sub[x_col] >= min_coverage]

    designs = sorted(sub["design"].unique())
    fig, axes = plt.subplots(1, len(designs), figsize=figsize, sharey=True)
    if len(designs) == 1:
        axes = [axes]

    ylabel = holdout_col.replace("r2_", "R² ").replace("_", " ").title()

    for ax, design in zip(axes, designs):
        dsub = sub[sub["design"] == design]
        modes_seen = []
        for mode_name, grp in dsub.groupby("mode"):
            grp = grp.sort_values(x_col)
            ax.plot(
                grp[x_col], grp[holdout_col],
                color=MODE_COLORS.get(mode_name, "gray"),
                marker=MODE_MARKERS.get(mode_name, "."),
                markersize=5, linewidth=1.5,
                label=mode_name,
            )
            modes_seen.append(mode_name)
        _style_ax(ax, "Coverage Fraction", ylabel if ax == axes[0] else "",
                   f"{design}-Trained")
        ax.set_xlim(-0.02, 1.05)
        _apply_ylim(ax, y_min, y_max)
        handles = [
            Line2D([0], [0], color=MODE_COLORS.get(m, "gray"),
                   marker=MODE_MARKERS.get(m, "."),
                   markersize=6, linewidth=1.5, label=m)
            for m in modes_seen
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)

    title = suptitle or f"{law_name} - Mode Comparison"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 9. Single focused comparison - CO vs NC on one square plot
# ═══════════════════════════════════════════════════════════════════════════

def plot_single_comparison(
    df: pd.DataFrame,
    law_name: str,
    mode_filter: str,
    holdout_col: str = "r2_unified",
    x_col: str = "coverage_frac",
    min_coverage: Optional[float] = None,
    figsize: Tuple[float, float] = (7, 7),
    y_min: Optional[float] = -0.5,
    y_max: Optional[float] = 1.05,
    suptitle: str = "",
    legend_fontsize: float = 12,
    legend_loc: str = "lower right",
    title_fontsize: float = 15,
    label_fontsize: float = 13,
    tick_fontsize: float = 11,
    marker_size: float = 7,
    line_width: float = 2.0,
) -> plt.Figure:
    """
    Single square plot: CO and NC R² on the same axes for one (law, mode).

    Parameters
    ----------
    df             : full results DataFrame (all datasets combined or single)
    law_name       : e.g. "Chinchilla", "Kaplan", "Droppo-Elibol", "Parametric-Epoch"
    mode_filter    : e.g. "Final", "First", "Second", "Epoch-Expanded"
    holdout_col    : which R² to plot
    min_coverage   : skip early degenerate steps (e.g. 0.15)
    figsize        : square by default (7, 7)
    legend_fontsize: control legend text size
    legend_loc     : legend placement
    title_fontsize : title text size
    label_fontsize : axis label text size
    tick_fontsize  : tick label text size
    marker_size    : data point size
    line_width     : line thickness
    """
    sub = df[
        (df["law"] == law_name)
        & (df["mode"] == mode_filter)
        & df["success"]
    ].copy()

    if min_coverage is not None:
        sub = sub[sub[x_col] >= min_coverage]

    if sub.empty:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.text(0.5, 0.5, f"No data for {law_name} / {mode_filter}",
                ha="center", va="center", fontsize=14)
        return fig

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for design in ["Colinear", "Non-Colinear"]:
        dsub = sub[sub["design"] == design].sort_values(x_col)
        if dsub.empty:
            continue
        ax.plot(
            dsub[x_col], dsub[holdout_col],
            color=DESIGN_COLORS.get(design, "gray"),
            marker="o" if design == "Colinear" else "s",
            markersize=marker_size,
            linewidth=line_width,
            label=design,
        )

    ylabel = holdout_col.replace("r2_", "R² ").replace("_", " ").title()
    ax.set_xlabel("Coverage Fraction", fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.05)
    _apply_ylim(ax, y_min, y_max)

    ax.legend(fontsize=legend_fontsize, loc=legend_loc, framealpha=0.9)

    title = suptitle or f"{law_name} - {mode_filter}"
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold")

    fig.tight_layout()
    return fig