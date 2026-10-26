"""
Incremental coverage analysis for scaling laws.

Builds progressive subsets of training data by:
  - Colinear:     adding TPP levels from lowest → highest (bins scaled by mode)
  - Non-colinear: pinching inward from extreme D values (D_min+D_max paired per step)

Then fits all four scaling laws at each step and tracks R², RMSE, ΔR², etc.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import time

import numpy as np
import pandas as pd

from scaling_laws.data import (
    PreparedData, prepare_arrays, prepare_epoch_expanded,
)
from scaling_laws.laws import (
    LAWS, ScalingLaw, evaluate, evaluate_parametric_epoch,
)


# ═══════════════════════════════════════════════════════════════════════════
# TPP / D computation helpers
# ═══════════════════════════════════════════════════════════════════════════

# The 12 designed single-pass TPP values
BASE_TPP_BINS = [1.0, 1.5, 1.9, 2.0, 2.5, 2.7, 3.0, 3.3, 3.5, 4.0, 4.5, 5.0]

# Epoch multiplier per mode (matches prepare_arrays D-scaling logic)
MODE_MULTIPLIER = {
    "first":  1,
    "second": 2,
    "final":  3,
}

# Standard and epoch law modes
EPOCH_MODES = ["final", "first", "second"]


def effective_tpp_bins(mode: str, base_bins: Optional[List[float]] = None) -> List[float]:
    """Scale the designed TPP bins by the mode's epoch multiplier."""
    if base_bins is None:
        base_bins = BASE_TPP_BINS
    m = MODE_MULTIPLIER.get(mode, 3)
    return [b * m for b in base_bins]


def compute_tpp(run: Dict, mode: str = "final") -> float:
    """
    Effective tokens-per-parameter, matching prepare_arrays D-scaling.

    first:  D / N
    second: D * 2 / N
    final:  D * n_epochs / N
    """
    m = MODE_MULTIPLIER.get(mode, run["n_epochs"])
    return run["D"] * m / run["N"]


def snap_to_nearest(value: float, bins: List[float]) -> float:
    """Snap a value to the nearest bin."""
    return min(bins, key=lambda b: abs(b - value))


def assign_tpp_bins(
    train_data: List[Dict],
    mode: str = "final",
    base_tpp_bins: Optional[List[float]] = None,
) -> List[float]:
    """
    Compute effective TPP for each run and snap to the mode-scaled bins.
    """
    bins = effective_tpp_bins(mode, base_tpp_bins)
    raw = [compute_tpp(d, mode) for d in train_data]
    return [snap_to_nearest(t, bins) for t in raw]


# ═══════════════════════════════════════════════════════════════════════════
# Step builders
# ═══════════════════════════════════════════════════════════════════════════

def build_colinear_steps(
    train_data: List[Dict],
    mode: str = "final",
    base_tpp_bins: Optional[List[float]] = None,
) -> List[Dict]:
    """
    Incrementally add TPP levels from lowest → highest.

    Bins are scaled by mode (first=×1, second=×2, final=×3)
    so the levels match what prepare_arrays uses for fitting.

    Each step dict contains:
        step, n_levels, levels (sorted TPP values), data, n_runs, coverage_frac
    """
    binned = assign_tpp_bins(train_data, mode, base_tpp_bins)
    unique_tpps = sorted(set(binned))
    total = len(unique_tpps)

    steps, included = [], set()
    for tpp_val in unique_tpps:
        included.add(tpp_val)
        subset = [d for d, t in zip(train_data, binned) if t in included]
        steps.append({
            "step": len(steps) + 1,
            "n_levels": len(included),
            "levels": sorted(included),
            "data": subset,
            "n_runs": len(subset),
            "coverage_frac": len(included) / total,
        })

    return steps


def build_noncolinear_steps(
    train_data: List[Dict],
) -> List[Dict]:
    """
    Incrementally add D levels by pinching inward from the extremes,
    always adding a pair (bottom + top) per step.

    Step 1: D_min + D_max
    Step 2: D_2nd_min + D_2nd_max
    Step 3: D_3rd_min + D_3rd_max
    ...

    If there's an odd number of unique D values, the middle value
    is added as a solo in the final step.

    Each step dict contains:
        step, n_levels, levels (sorted D values), data, n_runs, coverage_frac
    """
    d_values = [d["D"] for d in train_data]
    unique_ds = sorted(set(d_values))
    total = len(unique_ds)

    # Build pairs: (bottom, top), (2nd bottom, 2nd top), ...
    pairs = []
    lo, hi = 0, len(unique_ds) - 1
    while lo <= hi:
        if lo == hi:
            pairs.append([unique_ds[lo]])  # odd middle element
        else:
            pairs.append([unique_ds[lo], unique_ds[hi]])
        lo += 1
        hi -= 1

    steps, included = [], set()
    for pair in pairs:
        for d_val in pair:
            included.add(d_val)
        subset = [d for d, dv in zip(train_data, d_values) if dv in included]
        steps.append({
            "step": len(steps) + 1,
            "n_levels": len(included),
            "levels": sorted(included),
            "data": subset,
            "n_runs": len(subset),
            "coverage_frac": len(included) / total,
        })

    return steps


# ═══════════════════════════════════════════════════════════════════════════
# Fitting at a single step
# ═══════════════════════════════════════════════════════════════════════════

def _fit_standard_law(
    law: ScalingLaw,
    train_data: List[Dict],
    holdouts: Dict[str, List[Dict]],
    mode: str,
    n_restarts: int,
) -> Dict:
    """Fit a standard (N, D) → L law at one step."""
    N_tr, D_tr, L_tr = prepare_arrays(train_data, mode)
    if len(L_tr) < 5:
        return {"law": law.name, "success": False}

    params, sse, success = law.fit_fn(N_tr, D_tr, L_tr, n_restarts=n_restarts)
    if params is None:
        return {"law": law.name, "success": False}

    row = {"law": law.name, "success": True, "params": list(params), "sse": sse}

    for ho_label, ho_data in holdouts.items():
        N_ho, D_ho, L_ho = prepare_arrays(ho_data, mode)
        metrics = evaluate(law.loss_fn, params, N_tr, D_tr, L_tr, N_ho, D_ho, L_ho)
        row["r2_train"] = metrics["r2_train"]
        row["rmse_train"] = metrics["rmse_train"]
        row[f"r2_{ho_label}"] = metrics["r2_holdout"]
        row[f"rmse_{ho_label}"] = metrics["rmse_holdout"]

    return row


def _fit_epoch_law(
    law: ScalingLaw,
    train_data: List[Dict],
    holdouts: Dict[str, List[Dict]],
    n_restarts: int,
) -> Dict:
    """Fit the Parametric-Epoch law (N, U_D, R) at one step."""
    N_tr, U_D_tr, R_tr, _ep, L_tr = prepare_epoch_expanded(train_data)
    if len(L_tr) < 5:
        return {"law": law.name, "success": False}

    params_dict, loss_val, success = law.fit_fn(
        N_tr, U_D_tr, R_tr, L_tr, n_restarts=n_restarts,
    )
    if params_dict is None:
        return {"law": law.name, "success": False}

    row = {"law": law.name, "success": True, "params": params_dict, "sse": loss_val}

    for ho_label, ho_data in holdouts.items():
        N_ho, U_D_ho, R_ho, _ep_ho, L_ho = prepare_epoch_expanded(ho_data)
        metrics = evaluate_parametric_epoch(
            params_dict, N_tr, U_D_tr, R_tr, L_tr,
            N_ho, U_D_ho, R_ho, L_ho,
        )
        row["r2_train"] = metrics["r2_train"]
        row["rmse_train"] = metrics["rmse_train"]
        row[f"r2_{ho_label}"] = metrics["r2_holdout"]
        row[f"rmse_{ho_label}"] = metrics["rmse_holdout"]

    return row


def fit_at_step(
    step_data: List[Dict],
    holdouts: Dict[str, List[Dict]],
    laws_to_fit: Optional[List[ScalingLaw]] = None,
    modes: Optional[List[str]] = None,
    n_restarts: int = 200,
) -> List[Dict]:
    """Fit all specified laws × modes on one subset; evaluate on holdouts."""
    if laws_to_fit is None:
        laws_to_fit = LAWS
    if modes is None:
        modes = EPOCH_MODES

    results = []
    for law in laws_to_fit:
        if law.is_epoch_law:
            row = _fit_epoch_law(law, step_data, holdouts, n_restarts)
            row["mode"] = "Epoch-Expanded"
            results.append(row)
        else:
            for mode in modes:
                row = _fit_standard_law(law, step_data, holdouts, mode, n_restarts)
                row["mode"] = mode.capitalize()
                results.append(row)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Full incremental run
# ═══════════════════════════════════════════════════════════════════════════

_R2_COLS = [
    "r2_train", "r2_unified", "r2_colinear", "r2_noncolinear",
]
_RMSE_COLS = [
    "rmse_train", "rmse_unified", "rmse_colinear", "rmse_noncolinear",
]
_METRIC_COLS = _R2_COLS + _RMSE_COLS + ["sse"]


def run_incremental_analysis(
    steps: List[Dict],
    holdouts: Dict[str, List[Dict]],
    design_label: str,
    modes: Optional[List[str]] = None,
    n_restarts: int = 200,
    laws_to_fit: Optional[List[ScalingLaw]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run all scaling-law fits at every incremental step.

    Returns a DataFrame with one row per (step, law, mode):
        design, step, n_levels, n_runs, coverage_frac, law, mode, success,
        r2_train, r2_unified, r2_colinear, r2_noncolinear,
        rmse_*, sse, params
    """
    all_rows = []
    total_steps = len(steps)

    for step_info in steps:
        step_num = step_info["step"]
        t0 = time.time()

        if verbose:
            print(
                f"  [{step_num}/{total_steps}] {design_label} | "
                f"{step_info['n_levels']} levels | "
                f"{step_info['n_runs']} runs | "
                f"coverage={step_info['coverage_frac']:.0%}",
                end="",
                flush=True,
            )

        fit_results = fit_at_step(
            step_info["data"],
            holdouts,
            laws_to_fit=laws_to_fit,
            modes=modes,
            n_restarts=n_restarts,
        )

        for res in fit_results:
            row = {
                "design": design_label,
                "step": step_num,
                "n_levels": step_info["n_levels"],
                "n_runs": step_info["n_runs"],
                "coverage_frac": step_info["coverage_frac"],
                "law": res["law"],
                "mode": res.get("mode", "Final"),
                "success": res.get("success", False),
                "params": res.get("params"),
            }
            for col in _METRIC_COLS:
                row[col] = res.get(col, np.nan)
            all_rows.append(row)

        if verbose:
            elapsed = time.time() - t0
            successful = [r for r in fit_results if r.get("success")]
            law_summary = "  ".join(
                f"{r['law'][:4]}/{r.get('mode','')[:3]}={r.get('r2_unified', float('nan')):.4f}"
                for r in successful[:6]
            )
            if len(successful) > 6:
                law_summary += f"  +{len(successful)-6} more"
            print(f"  ({elapsed:.0f}s) {law_summary}")

    return pd.DataFrame(all_rows)


# ═══════════════════════════════════════════════════════════════════════════
# Delta computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``delta_r2_*`` columns: each value minus the full-coverage
    (last-step) value for the same (design, law, mode).

    Also adds ``abs_delta_r2_*`` for the absolute difference.
    """
    df = df.copy()
    r2_cols = [c for c in df.columns if c.startswith("r2_")]

    for col in r2_cols:
        delta_col = f"delta_{col}"
        abs_col = f"abs_delta_{col}"
        df[delta_col] = np.nan
        df[abs_col] = np.nan

        for (design, law, mode), group in df.groupby(["design", "law", "mode"]):
            max_step = group["step"].max()
            full_vals = group.loc[group["step"] == max_step, col]
            if full_vals.empty or full_vals.isna().all():
                continue
            full_val = full_vals.values[0]
            mask = (
                (df["design"] == design)
                & (df["law"] == law)
                & (df["mode"] == mode)
            )
            df.loc[mask, delta_col] = df.loc[mask, col] - full_val
            df.loc[mask, abs_col] = (df.loc[mask, col] - full_val).abs()

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: build steps + run for a full PreparedData
# ═══════════════════════════════════════════════════════════════════════════

def run_full_incremental(
    data: PreparedData,
    n_restarts: int = 200,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    End-to-end: build CO and NC steps, fit all laws incrementally,
    compute deltas, and return a single combined DataFrame.

    Colinear: builds separate step sequences per mode (first/second/final)
    so TPP bins are correctly scaled (×1, ×2, ×3).
    Epoch law runs once with base TPP bins (mode-independent).

    Non-colinear: same D-based steps for all modes; fits all modes at each step.
    """
    holdouts = {
        "unified": data.unified_holdout,
        "colinear": data.colinear_holdout,
        "noncolinear": data.noncolinear_holdout,
    }

    frames = []

    # ── Colinear steps (per-mode, since TPP bins scale with mode) ──
    if data.colinear_train:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  COLINEAR incremental - {data.dataset_name}")
            print(f"{'='*60}")

        standard_laws = [l for l in LAWS if not l.is_epoch_law]
        epoch_laws = [l for l in LAWS if l.is_epoch_law]

        # Standard laws: one pass per mode with mode-scaled TPP bins
        for mode in EPOCH_MODES:
            co_steps = build_colinear_steps(data.colinear_train, mode=mode)
            bins = effective_tpp_bins(mode)
            if verbose:
                print(f"\n  Mode={mode} | {len(co_steps)} TPP levels: {bins}")

            df_co = run_incremental_analysis(
                co_steps, holdouts, "Colinear",
                modes=[mode],
                laws_to_fit=standard_laws,
                n_restarts=n_restarts, verbose=verbose,
            )
            df_co["dataset"] = data.dataset_name
            frames.append(df_co)

        # Epoch law: run once with base TPP bins (uses all epochs internally)
        if epoch_laws:
            if verbose:
                print("\n  Parametric-Epoch (epoch-expanded, mode-independent)")
            co_steps_epoch = build_colinear_steps(data.colinear_train, mode="first")
            bins = effective_tpp_bins("first")
            if verbose:
                print(f"  {len(co_steps_epoch)} TPP levels: {bins}")
            df_co_epoch = run_incremental_analysis(
                co_steps_epoch, holdouts, "Colinear",
                modes=["first"],  # placeholder; epoch law ignores mode
                laws_to_fit=epoch_laws,
                n_restarts=n_restarts, verbose=verbose,
            )
            df_co_epoch["dataset"] = data.dataset_name
            frames.append(df_co_epoch)

    # ── Non-colinear steps (D-based, paired extremes, same for all modes) ──
    if data.noncolinear_train:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  NON-COLINEAR incremental - {data.dataset_name}")
            print(f"{'='*60}")
        nc_steps = build_noncolinear_steps(data.noncolinear_train)
        if verbose:
            for s in nc_steps:
                print(f"    Step {s['step']}: {s['n_levels']} D levels, {s['n_runs']} runs, coverage={s['coverage_frac']:.0%}")

        # All modes + all laws at each step
        df_nc = run_incremental_analysis(
            nc_steps, holdouts, "Non-Colinear",
            modes=None,  # all modes
            n_restarts=n_restarts, verbose=verbose,
        )
        df_nc["dataset"] = data.dataset_name
        frames.append(df_nc)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = compute_deltas(combined)
    return combined