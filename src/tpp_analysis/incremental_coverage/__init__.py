"""
incremental_coverage - Measure how scaling-law fit quality evolves
as the experimental design's TPP / D coverage grows.

Separate from ``scaling_laws`` so the analysis code lives in its own directory.
"""

from incremental_coverage.coverage import (
    BASE_TPP_BINS,
    MODE_MULTIPLIER,
    EPOCH_MODES,
    effective_tpp_bins,
    compute_tpp,
    snap_to_nearest,
    assign_tpp_bins,
    build_colinear_steps,
    build_noncolinear_steps,
    run_incremental_analysis,
    compute_deltas,
    run_full_incremental,
)
from incremental_coverage.viz import (
    plot_r2_convergence,
    plot_delta_convergence,
    plot_co_vs_nc_overlay,
    plot_parameter_trajectories,
    plot_summary_dashboard,
    plot_r2_heatmap,
    plot_r2_rate_of_change,
    plot_mode_comparison,
    plot_single_comparison,
)

__all__ = [
    "BASE_TPP_BINS",
    "MODE_MULTIPLIER",
    "EPOCH_MODES",
    "effective_tpp_bins",
    "compute_tpp",
    "snap_to_nearest",
    "assign_tpp_bins",
    "build_colinear_steps",
    "build_noncolinear_steps",
    "run_incremental_analysis",
    "compute_deltas",
    "run_full_incremental",
    "plot_r2_convergence",
    "plot_delta_convergence",
    "plot_co_vs_nc_overlay",
    "plot_parameter_trajectories",
    "plot_summary_dashboard",
    "plot_r2_heatmap",
    "plot_r2_rate_of_change",
    "plot_mode_comparison",
    "plot_single_comparison",
]