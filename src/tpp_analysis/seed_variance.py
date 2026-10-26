# ════════════════════════════════════════════════════════════════
# SEED-VARIANCE R² CONVERGENCE
# ════════════════════════════════════════════════════════════════
#
# At each coverage step, the training data is fixed (deterministic
# incremental procedure). The only randomness is the optimizer:
# random restarts in L-BFGS-B and DE initialization.
#
# This module runs the same fit N_SEEDS times with different
# optimizer seeds, producing a distribution of R² at each step.
#
# What it shows:
#   - If CO holdout R² varies across seeds → sloppy landscape,
#     optimizer lands in different spots along the flat valley
#   - If NC holdout R² is tight → well-conditioned objective,
#     clear unique minimum
#   - If both are tight → 300 restarts + DE is thorough enough
#     that optimizer randomness is negligible (also valid)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from scaling_laws.data import prepare_arrays
from scaling_laws.laws import (
    _fit_generic,
    kaplan_loss, chinchilla_loss, droppo_elibol_loss,
)


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
        (1e-6, min(L) * 0.99),
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
    "Kaplan": (_fit_kaplan_seed, kaplan_loss),
    "Chinchilla": (_fit_chinchilla_seed, chinchilla_loss),
    "Droppo-Elibol": (_fit_droppo_elibol_seed, droppo_elibol_loss),
}


# ════════════════════════════════════════════════════════════════
# Core: run one step with multiple seeds
# ════════════════════════════════════════════════════════════════

def _run_seeds_one_step(
    law_name,
    train_list,
    holdouts,
    mode,
    n_seeds=30,
    n_restarts=100,
    base_seed=0,
):
    """
    Fit the same training data n_seeds times with different
    optimizer seeds. Evaluate R² on fixed holdout each time.

    Returns dict of {split_name: array of R² values}.
    """
    fit_fn, loss_fn = _SEED_FITTERS[law_name]

    N_tr, D_tr, L_tr = prepare_arrays(train_list, mode)
    if len(L_tr) < 5:
        return None

    # Prepare holdout arrays once
    holdout_arrays = {}
    for hname, hlist in holdouts.items():
        N_h, D_h, L_h = prepare_arrays(hlist, mode)
        holdout_arrays[hname] = (N_h, D_h, L_h)

    results = {k: [] for k in holdouts.keys()}
    results["train"] = []

    for s in range(n_seeds):
        seed = base_seed + s * 137  # spread seeds apart

        try:
            params, sse, success = fit_fn(
                N_tr, D_tr, L_tr,
                n_restarts=n_restarts,
                seed=seed,
            )
            if params is None:
                continue

            # Training R²
            pred_tr = loss_fn(N_tr, D_tr, *params)
            results["train"].append(r2_score(L_tr, pred_tr))

            # Holdout R²
            for hname, (N_h, D_h, L_h) in holdout_arrays.items():
                if len(L_h) == 0:
                    results[hname].append(np.nan)
                    continue
                pred_h = loss_fn(N_h, D_h, *params)
                results[hname].append(r2_score(L_h, pred_h))

        except Exception:
            continue

    return {k: np.array(v) for k, v in results.items()}


# ════════════════════════════════════════════════════════════════
# Run across all coverage steps
# ════════════════════════════════════════════════════════════════

def run_seed_variance_convergence(
    data,
    law_name="Chinchilla",
    mode="final",
    n_seeds=30,
    n_restarts=100,
    base_seed=0,
    verbose=True,
):
    """
    Run seed-variance R² convergence for one law and mode.

    At each coverage step (deterministic), fits the law n_seeds
    times with different optimizer seeds. Reports mean, std,
    and percentiles of R².

    Parameters
    ----------
    data        : PreparedData object
    law_name    : "Chinchilla", "Kaplan", or "Droppo-Elibol"
    mode        : "final", "first", or "second"
    n_seeds     : number of optimizer seeds per step
    n_restarts  : random restarts per fit (can be lower than 300)
    base_seed   : starting seed

    Returns DataFrame with columns:
        design, step, coverage_frac, holdout_split,
        r2_mean, r2_std, r2_p5, r2_p25, r2_p50, r2_p75, r2_p95,
        r2_min, r2_max, n_seeds_ok
    """
    from incremental_coverage.coverage import (
        build_colinear_steps,
        build_noncolinear_steps,
    )

    if law_name not in _SEED_FITTERS:
        raise ValueError(f"Unsupported law: {law_name}. "
                         f"Supported: {list(_SEED_FITTERS.keys())}")

    holdouts = {
        "unified": data.unified_holdout,
        "colinear": data.colinear_holdout,
        "noncolinear": data.noncolinear_holdout,
    }

    rows = []

    for design_label, train_source in [
        ("Colinear", data.colinear_train),
        ("Non-Colinear", data.noncolinear_train),
    ]:
        if not train_source:
            continue

        # CO steps are mode-dependent (TPP bins scale with epoch);
        # NC steps are mode-independent (D-based pairing)
        if design_label == "Colinear":
            steps = build_colinear_steps(train_source, mode=mode)
        else:
            steps = build_noncolinear_steps(train_source)

        n_steps = len(steps)

        for si, step in enumerate(steps):
            step_train = step["data"]
            coverage = step["coverage_frac"]

            if verbose:
                print(f"  {design_label:14s} | {law_name:16s} | {mode:6s} | "
                      f"step {si+1:2d}/{n_steps} | cov={coverage:.2f} | "
                      f"{n_seeds} seeds ...", end=" ", flush=True)

            seed_results = _run_seeds_one_step(
                law_name, step_train, holdouts, mode,
                n_seeds=n_seeds,
                n_restarts=n_restarts,
                base_seed=base_seed + si * 1000,
            )

            if seed_results is None:
                if verbose:
                    print("SKIP (insufficient data)")
                continue

            # Record stats for each split
            for split_name in ["train", "unified", "colinear", "noncolinear"]:
                r2_arr = seed_results.get(split_name, np.array([]))
                r2_arr = r2_arr[~np.isnan(r2_arr)]

                if len(r2_arr) == 0:
                    continue

                rows.append({
                    "design": design_label,
                    "law": law_name,
                    "mode": mode.capitalize(),
                    "step": si + 1,
                    "coverage_frac": coverage,
                    "holdout_split": split_name,
                    "r2_mean": np.mean(r2_arr),
                    "r2_std": np.std(r2_arr),
                    "r2_p5": np.percentile(r2_arr, 5),
                    "r2_p25": np.percentile(r2_arr, 25),
                    "r2_p50": np.percentile(r2_arr, 50),
                    "r2_p75": np.percentile(r2_arr, 75),
                    "r2_p95": np.percentile(r2_arr, 95),
                    "r2_min": np.min(r2_arr),
                    "r2_max": np.max(r2_arr),
                    "n_seeds_ok": len(r2_arr),
                })

            # Print quick summary
            if verbose:
                uni = seed_results.get("unified", np.array([]))
                uni = uni[~np.isnan(uni)]
                if len(uni) > 0:
                    print(f"R²={np.mean(uni):.4f} ± {np.std(uni):.4f} "
                          f"[{np.min(uni):.4f}, {np.max(uni):.4f}] "
                          f"({len(uni)}/{n_seeds} ok)")
                else:
                    print("done")

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════

def plot_convergence_with_errorbars(
    sv_df,
    holdout_split="unified",
    ax=None,
    y_min=-0.5,
    y_max=1.05,
    band_type="percentile",   # "percentile" for p5-p95, "std" for mean±std
    title=None,
    axis_label_fs=25,
    tick_label_fs=20,
    legend_fs=20,
    marker_size=8,
    line_width=2.5,
    min_coverage=None,
):
    """
    Plot R² convergence with classic error bars (caps) from seed variance.

    band_type:
        "percentile" - bars span 5th to 95th percentile
        "std"        - bars span mean ± 1 std
    """
    import matplotlib.pyplot as plt

    DESIGN_COLORS = {
        "Colinear": "steelblue",
        "Non-Colinear": "indianred",
    }
    DESIGN_MARKERS = {
        "Colinear": "o",
        "Non-Colinear": "s",
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))
    else:
        fig = ax.figure

    sub = sv_df[sv_df["holdout_split"] == holdout_split].copy()

    if min_coverage is not None:
        sub = sub[sub["coverage_frac"] >= min_coverage]

    for design in ["Colinear", "Non-Colinear"]:
        dsub = sub[sub["design"] == design].sort_values("coverage_frac")
        if dsub.empty:
            continue

        color = DESIGN_COLORS.get(design, "gray")
        marker = DESIGN_MARKERS.get(design, "o")
        x = dsub["coverage_frac"].values
        y = dsub["r2_mean"].values

        # Mean line + markers
        ax.plot(
            x, y,
            color=color,
            marker=marker,
            markersize=marker_size,
            linewidth=line_width,
            label=design,
            zorder=2,
        )

        # Error bars (classic caps)
        if band_type == "percentile":
            lo = y - dsub["r2_p5"].values
            hi = dsub["r2_p95"].values - y
        else:
            lo = dsub["r2_std"].values
            hi = dsub["r2_std"].values

        ax.errorbar(
            x, y,
            yerr=[lo, hi],
            fmt='none',
            ecolor='#333333',
            elinewidth=1.5,
            capsize=6,
            capthick=1.5,
            alpha=0.7,
            zorder=3,
        )

    ylabel = f"R² {holdout_split.title()}"
    ax.set_xlabel("Coverage Fraction", fontsize=axis_label_fs)
    ax.set_ylabel(ylabel, fontsize=axis_label_fs)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(labelsize=tick_label_fs)
    ax.legend(fontsize=legend_fs, loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title, fontsize=axis_label_fs)

    return fig


# ════════════════════════════════════════════════════════════════
# Convenience: run + plot + save in one call
# ════════════════════════════════════════════════════════════════

def run_and_plot_seed_variance(
    data,
    dataset_name,
    law_name="Chinchilla",
    mode="final",
    holdout_split="unified",
    n_seeds=30,
    n_restarts=100,
    base_seed=0,
    save_dir=None,
    min_coverage=None,
):
    """One-call: seed variance + plot + optionally save."""
    import matplotlib.pyplot as plt

    print(f"\n{'='*60}")
    print(f"  SEED VARIANCE: {dataset_name} / {law_name} / {mode}")
    print(f"  {n_seeds} seeds × {n_restarts} restarts per fit")
    print(f"{'='*60}")

    sv_df = run_seed_variance_convergence(
        data,
        law_name=law_name,
        mode=mode,
        n_seeds=n_seeds,
        n_restarts=n_restarts,
        base_seed=base_seed,
    )

    fig = plot_convergence_with_errorbars(
        sv_df,
        holdout_split=holdout_split,
        title=f"{dataset_name}, {law_name}, {mode.title()}",
        min_coverage=min_coverage,
    )

    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        tag = f"{dataset_name}_{law_name}_{mode}".replace(" ", "_")

        # Save plot
        for ext in ["pdf", "png"]:
            path = os.path.join(save_dir, f"seed_variance_{tag}.{ext}")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path}")

        # Save data
        parquet_path = os.path.join(save_dir, f"seed_variance_{tag}.parquet")
        sv_df.to_parquet(parquet_path, index=False)
        print(f"  Saved: {parquet_path}")

    plt.show()
    return sv_df