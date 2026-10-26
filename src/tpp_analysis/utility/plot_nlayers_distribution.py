#!/usr/bin/env python3
"""
Plot a frequency bar chart of n_layers across the published architectures.

Two modes:
  --mode unique     One bar per architecture (14 total). Counts how many
                    distinct architectures use each n_layers value.
  --mode runs       One bar per checkpoint (~1,933). Counts how many
                    actually-trained models use each n_layers value.
                    Requires --audit-csv path.

Usage:
    # From the 14-architecture table (no CSV needed)
    python plot_nlayers_distribution.py --mode unique --out fig_nlayers

    # From the audit CSV (run-count weighted)
    python plot_nlayers_distribution.py --mode runs \\
        --audit-csv architecture_audit_e3.csv --out fig_nlayers

Style is locked to the triptych notebook's "Shared Sizing Constants"
(the figure notebooks): STIX fonts, 9 × 9 panels,
25/20/20 font sizes, #D8D8D8 panel face with #a8a8a8 1.6 pt gridlines,
all four spines at 1.5 pt, big inward ticks, no plot title.
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────
# Notebook-style locked constants
# ─────────────────────────────────────────────────────────────────────────
_NB_PANEL_W       = 9
_NB_PANEL_H       = 9
_NB_AXIS_LABEL_FS = 25
_NB_TICK_LABEL_FS = 20
_NB_VALUE_LBL_FS  = 16   # bar-top value annotations
_NB_TICK_MAJ_LEN  = 10
_NB_TICK_MAJ_WID  = 2.0
_NB_TICK_MIN_LEN  = 6
_NB_TICK_MIN_WID  = 1.2
_NB_SPINE_LW      = 1.5
_NB_BAR_EDGE_LW   = 1.0
_NB_GRID_BG       = "#D8D8D8"
_NB_GRID_COLOR    = "#a8a8a8"
_NB_GRID_LW       = 1.6
_NB_BAR_COLOR     = "steelblue"
_NB_BAR_EDGE      = "#222222"
_NB_DPI           = 300


# 14 published architectures, copied from the paper's Table tab:model-architectures
PUBLISHED_ARCHITECTURES = [
    # (params, n_layers, d_model, d_ff, n_heads, d_head)
    (5_035_656,   24, 24,  96,   12, 2),
    (10_070_160,  12, 48,  192,  1,  48),
    (14_411_456,  24, 64,  256,  32, 2),
    (19_400_928,  1,  96,  384,  1,  96),
    (22_796_832,  24, 96,  384,  24, 4),
    (28_819_840,  12, 128, 512,  4,  32),
    (32_498_720,  1,  160, 640,  16, 10),
    (35_368_160,  8,  160, 640,  4,  40),
    (40_277_184,  3,  192, 768,  12, 16),
    (47_949_888,  16, 192, 768,  1,  192),
    (55_538_432,  4,  256, 1024, 32, 8),
    (63_931_136,  12, 256, 1024, 32, 8),
    (68_127_488,  16, 256, 1024, 32, 8),
    (76_520_192,  24, 256, 1024, 1,  256),
]


def counts_unique():
    """Distribution across the 14 unique architectures."""
    return Counter(n_layers for (_, n_layers, *_) in PUBLISHED_ARCHITECTURES)


def counts_runs(audit_csv: Path):
    """Distribution across all trained checkpoints in the audit CSV."""
    import csv
    counts = Counter()
    with open(audit_csv, newline="") as f:
        for row in csv.DictReader(f):
            v = row.get("actual_n_layers")
            if v:
                counts[int(v)] += 1
    return counts


# ─────────────────────────────────────────────────────────────────────────
# Style helper - bar-chart variant: y-only grid, all four spines.
# Mirrors style_ticks_2d() from the triptych notebook (§ 7) with the x-axis
# grid suppressed since the x-axis is categorical (one tick per bar).
# ─────────────────────────────────────────────────────────────────────────
def _nb_style_axes(ax):
    ax.tick_params(
        axis="both", which="major",
        length=_NB_TICK_MAJ_LEN, width=_NB_TICK_MAJ_WID,
        labelsize=_NB_TICK_LABEL_FS, direction="in",
        top=True, right=True,
    )
    ax.tick_params(
        axis="y", which="minor",
        length=_NB_TICK_MIN_LEN, width=_NB_TICK_MIN_WID,
        direction="in", right=True,
    )
    for s in ("top", "bottom", "left", "right"):
        sp = ax.spines[s]
        sp.set_visible(True)
        sp.set_linewidth(_NB_SPINE_LW)
        sp.set_color("black")
    ax.set_facecolor(_NB_GRID_BG)
    # Y-axis grid only - x-axis is categorical (n_layers values), so a
    # vertical grid would just bisect the bars uselessly.
    ax.grid(
        True, axis="y", which="major",
        color=_NB_GRID_COLOR, linewidth=_NB_GRID_LW, alpha=1.0, zorder=0,
    )
    ax.set_axisbelow(True)


def plot(counts: Counter, out_stem: Path):
    layers = sorted(counts.keys())
    freqs  = [counts[k] for k in layers]

    # rc_context scopes the STIX font swap to this function only.
    with matplotlib.rc_context({"mathtext.fontset": "stix",
                                  "font.family":      "STIXGeneral"}):
        fig, ax = plt.subplots(figsize=(_NB_PANEL_W, _NB_PANEL_H))

        bars = ax.bar(
            [str(l) for l in layers], freqs,
            color=_NB_BAR_COLOR,
            edgecolor=_NB_BAR_EDGE, linewidth=_NB_BAR_EDGE_LW,
            zorder=3,
        )
        for b, f in zip(bars, freqs):
            ax.text(
                b.get_x() + b.get_width() / 2, b.get_height(),
                f"{f}",
                ha="center", va="bottom",
                fontsize=_NB_VALUE_LBL_FS, fontweight="bold",
            )

        ax.set_xlabel(r"Number of layers ($n_{\mathrm{layers}}$)",
                       fontsize=_NB_AXIS_LABEL_FS, labelpad=10)
        ax.set_ylabel("Number of models",
                       fontsize=_NB_AXIS_LABEL_FS, labelpad=10)

        # Headroom so the bar-top value labels don't collide with the
        # top spine.
        ax.set_ylim(0, max(freqs) * 1.15)

        _nb_style_axes(ax)

        # No title, no figure-level footer - captions handle naming
        # and the count summary is printed to stdout.
        fig.tight_layout()

        for ext in ("pdf", "png"):
            path = out_stem.with_suffix(f".{ext}")
            fig.savefig(path, dpi=_NB_DPI, bbox_inches="tight")
            print(f"  Wrote {path}")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["unique", "runs"], default="unique")
    p.add_argument("--audit-csv", type=Path,
                   help="architecture_audit_e3.csv (required for --mode runs)")
    p.add_argument("--out", type=Path, default=Path("fig_nlayers"),
                   help="Output filename stem (without extension)")
    args = p.parse_args()

    if args.mode == "unique":
        counts = counts_unique()
        mode_label = "unique architectures"
    else:
        if not args.audit_csv:
            p.error("--audit-csv is required for --mode runs")
        if not args.audit_csv.exists():
            p.error(f"audit CSV not found: {args.audit_csv}")
        counts = counts_runs(args.audit_csv)
        mode_label = "trained checkpoints"

    plot(counts, args.out)

    # Sanity table to stdout (mode label included so console output still
    # tells you what you plotted, even though the figure has no title).
    print(f"\n  Mode: {mode_label}")
    print(f"  {'n_layers':>10} {'count':>8}")
    print(f"  {'-' * 20}")
    for n in sorted(counts):
        print(f"  {n:>10} {counts[n]:>8}")
    print(f"  {'total':>10} {sum(counts.values()):>8}")


if __name__ == "__main__":
    main()
