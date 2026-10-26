"""
scaling_laws - Modular scaling law analysis for LLM training experiments.

Usage:
    from scaling_laws import run_full_analysis

    results = run_full_analysis(
        filepath="wiki_epoch_results.csv",
        dataset_name="Wikipedia",
        n_restarts=300,
    )
"""

from scaling_laws.pipeline import run_full_analysis, run_dataset_fits
from scaling_laws.tables import (
    display_comparison_tables, display_aggregated_stats,
    build_summary_df, get_winners_summary,
)
from scaling_laws.laws import (
    kaplan_loss, chinchilla_loss, droppo_elibol_loss,
    fit_kaplan, fit_chinchilla, fit_droppo_elibol,
    parametric_epoch_loss, effective_data, effective_params,
    fit_parametric_epoch, evaluate_parametric_epoch,
)
from scaling_laws.data import load_and_prepare, prepare_epoch_expanded
from scaling_laws.visualization import plot_3d_overview, plot_overlay, plot_isoflops, plot_isoflop_single

__all__ = [
    "run_full_analysis",
    "run_dataset_fits",
    "display_comparison_tables",
    "display_aggregated_stats",
    "build_summary_df",
    "get_winners_summary",
    "load_and_prepare",
    "prepare_epoch_expanded",
    "kaplan_loss", "chinchilla_loss", "droppo_elibol_loss",
    "fit_kaplan", "fit_chinchilla", "fit_droppo_elibol",
    "parametric_epoch_loss", "effective_data", "effective_params",
    "fit_parametric_epoch", "evaluate_parametric_epoch",
    "plot_3d_overview", "plot_overlay", "plot_isoflops", "plot_isoflop_single",
]