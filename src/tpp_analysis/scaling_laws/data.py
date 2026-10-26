"""
Data organisation, filtering, and array preparation.

Handles the split into colinear / non-colinear, train / holdout,
loss-based filtering, and conversion to (N, D, L) numpy arrays
with three epoch modes: 'final', 'first', 'all'.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from scaling_laws.parsing import parse_epoch_results


# ── Organising ────────────────────────────────────────────────────────────

def _organise(runs: List[Dict]) -> Dict[str, List[Dict]]:
    """Split raw runs into four buckets."""
    buckets: Dict[str, List[Dict]] = {
        "colinear_train": [],
        "colinear_holdout": [],
        "noncolinear_train": [],
        "noncolinear_holdout": [],
    }
    for run in runs:
        entry = {
            "N": run["N"],
            "D": run["D"],
            "n_epochs": run["n_epochs"],
            "final_loss": run["epoch_losses"][-1],
            "first_loss": run["epoch_losses"][0],
            "all_losses": run["epoch_losses"],
        }
        if run["is_colinear"]:
            key = "colinear_holdout" if run["is_holdout"] else "colinear_train"
        else:
            key = "noncolinear_holdout" if run["is_holdout"] else "noncolinear_train"
        buckets[key].append(entry)
    return buckets


# ── Filtering ─────────────────────────────────────────────────────────────

def _filter_by_loss(
    data: List[Dict],
    keep_fraction: float = 1.0,
    loss_key: str = "final_loss",
) -> List[Dict]:
    """Keep only the *keep_fraction* runs with lowest loss."""
    if not data or keep_fraction >= 1.0:
        return data
    sorted_data = sorted(data, key=lambda x: x[loss_key])
    n_keep = max(1, int(len(sorted_data) * keep_fraction))
    return sorted_data[:n_keep]


# ── Array preparation ─────────────────────────────────────────────────────

def prepare_arrays(
    data: List[Dict], mode: str = "final"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a list of run dicts to (N, D, L) numpy arrays.

    Modes
    -----
    'final' : D = dataset_size × n_epochs  (total tokens seen)
    'first' : D = dataset_size              (single pass)
    'all'   : one point per epoch, D = dataset_size × epoch_index
    """
    if not data:
        return np.array([]), np.array([]), np.array([])

    if mode == "final":
        N = np.array([d["N"] for d in data])
        D = np.array([d["D"] * d["n_epochs"] for d in data])
        L = np.array([d["final_loss"] for d in data])

    elif mode == "first":
        N = np.array([d["N"] for d in data])
        D = np.array([d["D"] for d in data])
        L = np.array([d["all_losses"][0] for d in data])

    # In data.py, inside prepare_arrays()

    elif mode == "second":
        # Only keep runs that actually reached epoch 2
        valid_data = [d for d in data if len(d["all_losses"]) >= 2]
        N = np.array([d["N"] for d in valid_data])
        D = np.array([d["D"] * 2 for d in valid_data])
        L = np.array([d["all_losses"][1] for d in valid_data])

    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    return N, D, L


# ── Epoch-expanded arrays (for Parametric Epoch law) ─────────────────────

def prepare_epoch_expanded(
    data: List[Dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Expand runs → one point per epoch for the Parametric Epoch law.

    Returns (N, U_D, R, epoch, L) where:
      N    = model params (U_N)
      U_D  = unique tokens (dataset size, NOT scaled by epoch)
      R    = epoch - 1  (repetition count, same for data and params)
      epoch = epoch number (1, 2, 3, ...)
      L    = loss at that epoch
    """
    if not data:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, np.array([], dtype=np.int32), empty

    Ns, UDs, Rs, Es, Ls = [], [], [], [], []
    for d in data:
        for idx, loss in enumerate(d["all_losses"]):
            ep = idx + 1
            Ns.append(d["N"])
            UDs.append(d["D"])
            Rs.append(ep - 1)
            Es.append(ep)
            Ls.append(loss)

    return (
        np.array(Ns, dtype=np.float64),
        np.array(UDs, dtype=np.float64),
        np.array(Rs, dtype=np.float64),
        np.array(Es, dtype=np.int32),
        np.array(Ls, dtype=np.float64),
    )


# ── Public entry-point ────────────────────────────────────────────────────

@dataclass
class PreparedData:
    """Everything needed to run the scaling-law fits for one dataset."""

    dataset_name: str

    # Filtered train lists (for fitting)
    colinear_train: List[Dict] = field(default_factory=list)
    noncolinear_train: List[Dict] = field(default_factory=list)

    # Three holdout variants
    unified_holdout: List[Dict] = field(default_factory=list)      # NC + CO
    colinear_holdout: List[Dict] = field(default_factory=list)     # CO only
    noncolinear_holdout: List[Dict] = field(default_factory=list)  # NC only

    # Counts (for display)
    raw_counts: Dict[str, int] = field(default_factory=dict)


def load_and_prepare(
    filepath: str,
    dataset_name: str = "Dataset",
    train_keep: float = 1.0,
    holdout_keep: float = 1.0,
    verbose: bool = True,
) -> PreparedData:
    """
    Parse → organise → filter → return a `PreparedData` object.

    Parameters
    ----------
    filepath : path to the epoch-results CSV
    dataset_name : label used in tables / plots
    train_keep : fraction of training runs to keep (by lowest loss)
    holdout_keep : fraction of holdout runs to keep (by lowest loss)
    """
    raw_runs = parse_epoch_results(filepath)
    org = _organise(raw_runs)

    raw_counts = {k: len(v) for k, v in org.items()}

    co_train = _filter_by_loss(org["colinear_train"], train_keep)
    nc_train = _filter_by_loss(org["noncolinear_train"], train_keep)
    co_hold  = _filter_by_loss(org["colinear_holdout"], holdout_keep)
    nc_hold  = _filter_by_loss(org["noncolinear_holdout"], holdout_keep)

    data = PreparedData(
        dataset_name=dataset_name,
        colinear_train=co_train,
        noncolinear_train=nc_train,
        unified_holdout=nc_hold + co_hold,
        colinear_holdout=co_hold,
        noncolinear_holdout=nc_hold,
        raw_counts=raw_counts,
    )

    if verbose:
        print(f"{'=' * 60}")
        print(f"  {dataset_name}")
        print(f"{'=' * 60}")
        print(f"  Loaded {len(raw_runs)} total runs from {filepath}")
        for k, v in raw_counts.items():
            print(f"    {k:>25s}: {v}")
        print(f"  After filtering (train={train_keep}, holdout={holdout_keep}):")
        print(f"    colinear_train:       {len(co_train)}")
        print(f"    noncolinear_train:    {len(nc_train)}")
        print(f"    unified_holdout:      {len(data.unified_holdout)}")
        print(f"    colinear_holdout:     {len(co_hold)}")
        print(f"    noncolinear_holdout:  {len(nc_hold)}")
        print()

    return data
