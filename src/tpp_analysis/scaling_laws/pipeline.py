"""
Pipeline: run all scaling-law fits for a single dataset.

For each combination of (law, design, epoch_mode) the pipeline:
  1. Fits on the training data.
  2. Evaluates on THREE holdout sets: unified, colinear-only, noncolinear-only.

Returns a flat list of result dicts that can be turned into DataFrames.
"""

from __future__ import annotations

from typing import Dict, List, Optional


from scaling_laws.data import PreparedData, load_and_prepare, prepare_arrays, prepare_epoch_expanded
from scaling_laws.laws import LAWS, evaluate, evaluate_parametric_epoch


# ── Single fit helper ─────────────────────────────────────────────────────

def _run_one_fit(
    law,
    design_label: str,
    mode: str,
    train_data: list,
    holdouts: Dict[str, list],
    n_restarts: int,
) -> Optional[Dict]:
    """Fit one (law, design, mode) combo; evaluate on each holdout variant."""
    N_tr, D_tr, L_tr = prepare_arrays(train_data, mode)
    if len(L_tr) < 5:
        return None

    params, sse, success = law.fit_fn(N_tr, D_tr, L_tr, n_restarts=n_restarts)
    if params is None:
        return None

    result = {
        "law": law.name,
        "design": design_label,
        "mode": mode.capitalize(),
        "params": params,
        "sse": sse,
    }

    # Evaluate on every holdout variant supplied
    for holdout_label, holdout_list in holdouts.items():
        N_ho, D_ho, L_ho = prepare_arrays(holdout_list, mode)
        metrics = evaluate(law.loss_fn, params, N_tr, D_tr, L_tr, N_ho, D_ho, L_ho)
        result["r2_train"] = metrics["r2_train"]
        result["rmse_train"] = metrics["rmse_train"]
        result[f"r2_{holdout_label}"] = metrics["r2_holdout"]
        result[f"rmse_{holdout_label}"] = metrics["rmse_holdout"]

    return result


# ── Epoch-law fit helper ──────────────────────────────────────────────────

def _run_one_epoch_fit(
    law,
    design_label: str,
    train_data: list,
    holdouts: Dict[str, list],
    n_restarts: int,
) -> Optional[Dict]:
    """Fit the Parametric-Epoch law (uses N, U_D, R arrays instead of N, D)."""
    N_tr, U_D_tr, R_tr, _ep_tr, L_tr = prepare_epoch_expanded(train_data)
    if len(L_tr) < 5:
        return None

    params_dict, loss_val, success = law.fit_fn(N_tr, U_D_tr, R_tr, L_tr, n_restarts=n_restarts)
    if params_dict is None:
        return None

    result = {
        "law": law.name,
        "design": design_label,
        "mode": "Epoch-Expanded",
        "params": params_dict,
        "sse": loss_val,
    }

    # Evaluate on every holdout variant
    for holdout_label, holdout_list in holdouts.items():
        N_ho, U_D_ho, R_ho, _ep_ho, L_ho = prepare_epoch_expanded(holdout_list)
        metrics = evaluate_parametric_epoch(
            params_dict, N_tr, U_D_tr, R_tr, L_tr, N_ho, U_D_ho, R_ho, L_ho,
        )
        result["r2_train"] = metrics["r2_train"]
        result["rmse_train"] = metrics["rmse_train"]
        result[f"r2_{holdout_label}"] = metrics["r2_holdout"]
        result[f"rmse_{holdout_label}"] = metrics["rmse_holdout"]

    return result


# ── Main pipeline ─────────────────────────────────────────────────────────

EPOCH_MODES = ["final", "first", "second"]


def run_dataset_fits(
    data: PreparedData,
    n_restarts: int = 300,
    multi_epoch: bool = True,
    verbose: bool = True,
) -> List[Dict]:
    """
    Run all fits for a single PreparedData object.

    Parameters
    ----------
    multi_epoch : if True (default), fits Final/First/All modes + Parametric-Epoch law.
                  if False, fits Final mode only and skips Parametric-Epoch.
                  Set to False for single-epoch datasets.

    Returns a list of result dicts (one per law × design × mode).
    Each dict contains R² on unified / colinear / noncolinear holdouts.
    """

    holdouts = {
        "unified": data.unified_holdout,
        "colinear": data.colinear_holdout,
        "noncolinear": data.noncolinear_holdout,
    }

    designs = {}
    if data.colinear_train:
        designs["Colinear"] = data.colinear_train
    if data.noncolinear_train:
        designs["Non-Colinear"] = data.noncolinear_train

    # Select modes based on multi_epoch flag
    active_modes = EPOCH_MODES if multi_epoch else ["final"]

    # Count total jobs: standard laws get N modes each, epoch law gets 1 (if multi_epoch)
    standard_laws = [l for l in LAWS if not l.is_epoch_law]
    epoch_laws    = [l for l in LAWS if l.is_epoch_law] if multi_epoch else []
    total = len(standard_laws) * len(designs) * len(active_modes) + len(epoch_laws) * len(designs)

    if verbose and not multi_epoch:
        print("  [multi_epoch=False] Running Final mode only, skipping epoch laws")

    results: List[Dict] = []
    done = 0

    # ── Standard laws (Kaplan, Chinchilla, Droppo-Elibol) ──
    for law in standard_laws:
        for design_label, train_data in designs.items():
            for mode in active_modes:
                done += 1
                tag = f"[{done}/{total}] {law.name} | {design_label} | {mode}"
                if verbose:
                    print(f"  Fitting {tag} ...", end=" ", flush=True)

                row = _run_one_fit(
                    law, design_label, mode, train_data, holdouts, n_restarts,
                )

                if row is not None:
                    row["dataset"] = data.dataset_name
                    results.append(row)
                    if verbose:
                        r2u = row.get("r2_unified", float("nan"))
                        r2c = row.get("r2_colinear", float("nan"))
                        r2n = row.get("r2_noncolinear", float("nan"))
                        print(f"R² uni={r2u:.4f}  co={r2c:.4f}  nc={r2n:.4f}")
                else:
                    if verbose:
                        print("SKIPPED (insufficient data)")

    # ── Epoch laws (Parametric-Epoch with D'/N' half-lives) ──
    for law in epoch_laws:
        for design_label, train_data in designs.items():
            done += 1
            tag = f"[{done}/{total}] {law.name} | {design_label} | epoch-expanded"
            if verbose:
                print(f"  Fitting {tag} ...", end=" ", flush=True)

            row = _run_one_epoch_fit(
                law, design_label, train_data, holdouts, n_restarts,
            )

            if row is not None:
                row["dataset"] = data.dataset_name
                results.append(row)
                if verbose:
                    r2u = row.get("r2_unified", float("nan"))
                    r2c = row.get("r2_colinear", float("nan"))
                    r2n = row.get("r2_noncolinear", float("nan"))
                    print(f"R² uni={r2u:.4f}  co={r2c:.4f}  nc={r2n:.4f}")
            else:
                if verbose:
                    print("SKIPPED (insufficient data)")

    return results


# ── Convenience: load + fit in one call ───────────────────────────────────

def run_full_analysis(
    filepath: str,
    dataset_name: str = "Dataset",
    n_restarts: int = 300,
    train_keep: float = 1.0,
    holdout_keep: float = 1.0,
    multi_epoch: bool = True,
    verbose: bool = True,
) -> List[Dict]:
    """
    End-to-end: parse CSV → organise → filter → fit all combos.

    Parameters
    ----------
    multi_epoch : if True (default), fits Final/First/All modes + Parametric-Epoch law.
                  if False, fits Final mode only and skips Parametric-Epoch.
                  Set to False for single-epoch datasets.

    Returns list of result dicts ready for `build_summary_df()`.
    """
    data = load_and_prepare(
        filepath, dataset_name,
        train_keep=train_keep,
        holdout_keep=holdout_keep,
        verbose=verbose,
    )
    return run_dataset_fits(data, n_restarts=n_restarts, multi_epoch=multi_epoch, verbose=verbose)
