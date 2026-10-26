#!/usr/bin/env python3
"""
Extract training losses from .npz metrics files and produce CSVs for scaling law fits.

Two modes:
  1. from_hf    - reads .npz metrics from the HuggingFace repo directory structure
                  (dataset/design/N_{n}/*.npz).
  2. from_local - reads .npz metrics from a flat training_metrics dump.
                  Filenames contain the model's actual parameter count.

Both use trailing-100-batch converged loss at each epoch boundary (not epoch averages).
Both produce one CSV per dataset in an extracted_losses/ folder.

Three dataset tiers:
  - MAIN_DATASETS:    the 5 corpora used for the main experiment.
  - SPECIAL_DATASETS: BF16 replication (Wikipedia at BF16 precision).
                      Enabled with --include-special.
  - PRELIM_DATASETS:  preliminary BigTPP single-law experiment.
                      Enabled with --include-prelim.
  These are independent flags, so you can pull just what you need.

Usage:
    # From a downloaded HF repo:
    python get_results.py from_hf /path/to/colinear_scaling_models -o extracted_losses

    # From the cluster training_metrics folder:
    python get_results.py from_local /path/to/training_metrics -o extracted_losses

    # Include BF16 replication:
    python get_results.py from_local /path/to/metrics -o extracted_losses --include-special

    # Include BigTPP preliminary:
    python get_results.py from_local /path/to/metrics -o extracted_losses --include-prelim

    # Both:
    python get_results.py from_local /path/to/metrics -o extracted_losses --include-special --include-prelim

Output CSV format (one per dataset):
    run_type,N,D,tpp,wd,num_epochs,epoch,train_loss,val_loss,cumulative_time_s
"""

import os
import re
import glob
import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

# ============================================================
# Datasets
# ============================================================
MAIN_DATASETS    = ["wikipedia", "cosmopedia", "redpajama", "pes2o", "c4"]
SPECIAL_DATASETS = ["wikipedia_bf16"]
PRELIM_DATASETS  = ["wikipedia_bigtpp"]

# Design directories on HF (just two after cleanup)
VALID_DESIGNS = {"colinear", "non_colinear"}


def selected_datasets(include_special: bool = False,
                       include_prelim: bool = False) -> List[str]:
    """Return the active dataset list given the flag combination."""
    out = list(MAIN_DATASETS)
    if include_special:
        out.extend(SPECIAL_DATASETS)
    if include_prelim:
        out.extend(PRELIM_DATASETS)
    return out


# ============================================================
# Shared data structures
# ============================================================

@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: Optional[float]
    cumulative_time_s: float
    cumulative_tokens: int

@dataclass
class RunResult:
    dataset: str
    run_type: str
    N: int
    D: int
    tpp: Optional[float]
    wd: float
    epochs: List[EpochRecord] = field(default_factory=list)


# ============================================================
# Shared .npz extraction utilities
# ============================================================

def extract_unique_tokens(data: dict) -> Optional[int]:
    """Extract D (unique tokens per epoch) from cumulative_tokens array."""
    cumulative_tokens = data.get("cumulative_tokens", None)
    epoch_train_losses = data.get("epoch_train_losses", None)
    if cumulative_tokens is None or epoch_train_losses is None:
        return None
    cumulative_tokens = np.asarray(cumulative_tokens)
    num_epochs = len(epoch_train_losses)
    if num_epochs == 0 or len(cumulative_tokens) == 0:
        return None
    batches_per_epoch = len(cumulative_tokens) // num_epochs
    if batches_per_epoch == 0:
        return None
    return int(cumulative_tokens[batches_per_epoch - 1])


def compute_epoch_end_losses(data: dict, num_epochs: int, tail: int = 100) -> Optional[List[float]]:
    """Compute trailing-average loss at each epoch boundary from per-batch losses.

    For each epoch, takes the mean of the last `tail` batch losses (or all
    batches in the epoch if fewer than `tail`). This gives the converged
    loss the model reached at the end of each epoch, free of the early-
    training spike that inflates epoch-average metrics.
    """
    batch_losses = data.get("batch_losses", None)
    if batch_losses is None:
        return None
    batch_losses = np.asarray(batch_losses)
    total_batches = len(batch_losses)
    if total_batches == 0 or num_epochs == 0:
        return None
    batches_per_epoch = total_batches // num_epochs
    if batches_per_epoch == 0:
        return None

    epoch_end = []
    for ep in range(num_epochs):
        end_idx = (ep + 1) * batches_per_epoch
        start_idx = max(ep * batches_per_epoch, end_idx - tail)
        segment = batch_losses[start_idx:end_idx]
        epoch_end.append(float(np.mean(segment)))
    return epoch_end


def extract_tpp_from_filename(filename: str) -> Optional[float]:
    """Extract TPP from filename if present."""
    m = re.search(r'_TPP([\d.]+)_', filename)
    return float(m.group(1)) if m else None


def extract_wd_from_filename(filename: str) -> float:
    """Extract weight decay from filename."""
    m = re.search(r'_wd([\d.]+)', filename)
    return float(m.group(1)) if m else 0.1


def extract_n_from_filename(filename: str) -> Optional[int]:
    """Extract exact N from the _m{N}_token or _m{N}_ pattern."""
    m = re.search(r'_m(\d+)_token\d+', filename)
    if m:
        return int(m.group(1))
    m = re.search(r'_m(\d+)_', filename)
    if m:
        return int(m.group(1))
    return None


def build_epoch_records(data: dict, use_final_loss: bool = True) -> Optional[List[EpochRecord]]:
    """Build epoch records from a loaded .npz file.

    Uses trailing-100-batch converged loss by default.
    Falls back to epoch_train_losses if batch_losses unavailable.
    """
    epoch_train_losses = list(data.get("epoch_train_losses", []))
    if not epoch_train_losses:
        return None

    num_epochs = len(epoch_train_losses)
    val_losses = list(data.get("val_losses", []))
    epoch_times = list(data.get("epoch_times", []))
    cumulative_tokens = list(data.get("cumulative_tokens", []))
    batches_per_epoch = len(cumulative_tokens) // num_epochs if num_epochs > 0 and cumulative_tokens else 0

    # Trailing-100-batch converged loss
    if use_final_loss:
        final_losses = compute_epoch_end_losses(data, num_epochs)
    else:
        final_losses = None
    train_losses = final_losses if final_losses is not None else epoch_train_losses

    epochs = []
    cum_time = 0.0
    for i in range(num_epochs):
        if i < len(epoch_times):
            cum_time += epoch_times[i]
        cum_tok = 0
        if batches_per_epoch > 0 and (i + 1) * batches_per_epoch <= len(cumulative_tokens):
            cum_tok = int(cumulative_tokens[(i + 1) * batches_per_epoch - 1])

        epochs.append(EpochRecord(
            epoch=i + 1,
            train_loss=float(train_losses[i]),
            val_loss=float(val_losses[i]) if i < len(val_losses) else None,
            cumulative_time_s=cum_time,
            cumulative_tokens=cum_tok,
        ))

    return epochs


# ============================================================
# CSV output - old block format compatible with scaling_laws.data.load_and_prepare
# ============================================================
# Format:
#   N={N}, D={D}, WD={wd} [{run_type}]
#   run_type={run_type},is_holdout={is_holdout},is_colinear={is_colinear}
#   Epoch,Train Loss,Val Loss,Cumulative Time (s)
#   1,5.1234,,10.7
#   2,4.8765,,20.2
#   <blank line>

def write_dataset_csv(output_dir: str, dataset: str, runs: List[RunResult]):
    """Write one CSV per dataset in the block format that load_and_prepare expects."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{dataset}.csv")

    runs_sorted = sorted(runs, key=lambda r: (r.run_type, r.wd, r.N, r.D))

    lines = []
    for run in runs_sorted:
        is_colinear = run.run_type in ("colinear", "colinear_holdout")
        is_holdout = run.run_type in ("holdout", "colinear_holdout")

        lines.append(f"N={run.N}, D={run.D}, WD={run.wd} [{run.run_type}]")
        lines.append(f"run_type={run.run_type},is_holdout={is_holdout},is_colinear={is_colinear}")
        lines.append("Epoch,Train Loss,Val Loss,Cumulative Time (s)")

        for ep in sorted(run.epochs, key=lambda e: e.epoch):
            val_str = f"{ep.val_loss:.4f}" if ep.val_loss is not None else ""
            lines.append(f"{ep.epoch},{ep.train_loss:.4f},{val_str},{ep.cumulative_time_s:.1f}")

        lines.append("")  # blank line between runs

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"  Wrote {path}: {len(runs_sorted)} runs")


# ============================================================
# MODE 1: from_hf - read .npz metrics from HF directory structure
# ============================================================

def detect_run_type_from_filename(filename: str) -> str:
    """Detect run type from .npz filename (used in from_hf mode).

    Since colinear/ dir contains both colinear and colinear_holdout files,
    and non_colinear/ contains both nxd_grid and holdout files, we must
    detect from the filename, not the directory.
    """
    fn_upper = filename.upper()
    if "COLINEAR_HOLDOUT" in fn_upper:
        return "colinear_holdout"
    if "HOLDOUT" in fn_upper and "COLINEAR" not in fn_upper:
        return "holdout"
    if "COLINEAR" in fn_upper:
        return "colinear"
    if "NXD_GRID" in fn_upper:
        return "nxd_grid"
    # Bare filename → nxd_grid
    return "nxd_grid"


def extract_from_hf(repo_dir: str, output_dir: str,
                     include_special: bool = False,
                     include_prelim: bool = False,
                     use_final_loss: bool = True):
    """Extract results from HuggingFace repo directory structure (.npz files)."""
    datasets_to_process = selected_datasets(include_special, include_prelim)

    for dataset in datasets_to_process:
        dataset_dir = os.path.join(repo_dir, dataset)
        if not os.path.isdir(dataset_dir):
            print(f"  {dataset}: directory not found, skipping")
            continue

        runs = []

        for design in sorted(os.listdir(dataset_dir)):
            design_dir = os.path.join(dataset_dir, design)
            if not os.path.isdir(design_dir):
                continue
            if design not in VALID_DESIGNS:
                print(f"  {dataset}/{design}: unknown design, skipping")
                continue

            for n_dirname in sorted(os.listdir(design_dir)):
                n_dir = os.path.join(design_dir, n_dirname)
                if not os.path.isdir(n_dir) or not n_dirname.startswith("N_"):
                    continue

                # N already correct on HF - no correction
                N = int(n_dirname.replace("N_", ""))

                npz_files = sorted(glob.glob(os.path.join(n_dir, "*_metrics.npz")))
                if not npz_files:
                    continue

                for filepath in npz_files:
                    filename = os.path.basename(filepath)
                    if "resumed" in filename or "psi" in filename or "intermediate" in filename or "REPRO_TEST" in filename:
                        continue

                    # Detect run_type from filename, not directory
                    file_run_type = detect_run_type_from_filename(filename)

                    try:
                        data = dict(np.load(filepath))
                    except Exception as e:
                        print(f"    WARNING: Failed to load {filename}: {e}")
                        continue

                    D = extract_unique_tokens(data)
                    if D is None:
                        continue

                    epochs = build_epoch_records(data, use_final_loss=use_final_loss)
                    if not epochs:
                        continue

                    tpp = extract_tpp_from_filename(filename)
                    wd = extract_wd_from_filename(filename)

                    runs.append(RunResult(
                        dataset=dataset, run_type=file_run_type, N=N, D=D,
                        tpp=tpp, wd=wd, epochs=epochs,
                    ))

        # Dedup
        if runs:
            deduped = {}
            for run in runs:
                key = (run.run_type, run.N, run.D, run.wd)
                if key not in deduped or len(run.epochs) > len(deduped[key].epochs):
                    deduped[key] = run
            runs = list(deduped.values())
            write_dataset_csv(output_dir, dataset, runs)
        else:
            print(f"  {dataset}: no runs found")

    print(f"\nDone. CSVs written to {output_dir}/")


# ============================================================
# MODE 2: from_local - read .npz metrics from flat directory
# ============================================================

SKIP_PATTERNS = ["resumed", "psi", "intermediate", "NxDxWDxLR", "NxDxWD_GRID",
                 "COSMO_LR", "WIKI_LR", "RP_LR", "PES2O_LR", "C4_LR",
                 "REPRO_TEST"]


def detect_dataset(filename: str) -> Optional[str]:
    """Detect dataset from a metrics filename.

    Order matters: BigTPP and BF16 markers are checked BEFORE the generic
    Wikipedia detector, since both special variants are also wikipedia
    runs and would otherwise be misclassified as the main `wikipedia`.
    """
    fn_upper = filename.upper()

    # Special variants - checked first so they take priority
    if "BIGTPP" in fn_upper:
        return "wikipedia_bigtpp"
    if "FP16WIKI" in fn_upper or "FP16_WIKI" in fn_upper or "BF16" in fn_upper:
        return "wikipedia_bf16"

    # Embedded dataset tag (most reliable)
    dataset_match = re.search(r'_(c4|pes2o|redpajama|wikipedia|cosmopedia)_m\d+_', filename.lower())
    if dataset_match:
        return dataset_match.group(1)

    # Leading prefix fallback
    if fn_upper.startswith("C4_"):
        return "c4"
    if fn_upper.startswith("PES2O_"):
        return "pes2o"
    if fn_upper.startswith("RP_") or fn_upper.startswith("REDPAJAMA_"):
        return "redpajama"
    if fn_upper.startswith("WIKI_"):
        return "wikipedia"

    # Default: cosmopedia
    return "cosmopedia"


def detect_run_type(filename: str) -> Tuple[str, bool, bool]:
    """Detect run type from metrics filename.
    Returns: (run_type, is_colinear, is_holdout)
    """
    fn_upper = filename.upper()

    if "COLINEAR_HOLDOUT" in fn_upper:
        return "colinear_holdout", True, True
    if "HOLDOUT" in fn_upper and "COLINEAR" not in fn_upper:
        return "holdout", False, True
    if "COLINEAR" in fn_upper:
        return "colinear", True, False
    if "NXD_GRID" in fn_upper:
        return "nxd_grid", False, False

    # Bare run → nxd_grid
    return "nxd_grid", False, False


def extract_from_local(metrics_dir: str, output_dir: str,
                       include_special: bool = False,
                       include_prelim: bool = False,
                       use_final_loss: bool = True):
    """Extract results from flat training_metrics directory (.npz files)."""

    pattern = os.path.join(metrics_dir, "*_metrics.npz")
    all_files = sorted(glob.glob(pattern))
    print(f"Found {len(all_files)} metrics files in {metrics_dir}")

    runs_by_dataset: Dict[str, List[RunResult]] = defaultdict(list)
    active = set(selected_datasets(include_special, include_prelim))

    for filepath in all_files:
        filename = os.path.basename(filepath)

        if any(pat in filename for pat in SKIP_PATTERNS):
            continue
        if "_initial.json" in filename:
            continue

        dataset = detect_dataset(filename)
        if dataset is None:
            continue
        if dataset not in active:
            # Either special/prelim flag not set, or unknown dataset
            continue

        run_type, is_colinear, is_holdout = detect_run_type(filename)

        raw_n = extract_n_from_filename(filename)
        if raw_n is None:
            continue
        # No N correction needed in this codebase (HF naming is canonical)
        N = raw_n

        tpp = extract_tpp_from_filename(filename)
        wd = extract_wd_from_filename(filename)

        try:
            data = dict(np.load(filepath))
        except Exception as e:
            print(f"  WARNING: Failed to load {filename}: {e}")
            continue

        D = extract_unique_tokens(data)
        if D is None:
            continue

        epochs = build_epoch_records(data, use_final_loss=use_final_loss)
        if not epochs:
            continue

        runs_by_dataset[dataset].append(RunResult(
            dataset=dataset, run_type=run_type, N=N, D=D,
            tpp=tpp, wd=wd, epochs=epochs,
        ))

    # Dedup per dataset
    for dataset in runs_by_dataset:
        runs = runs_by_dataset[dataset]
        deduped = {}
        for run in runs:
            key = (run.run_type, run.N, run.D, run.wd)
            if key not in deduped or len(run.epochs) > len(deduped[key].epochs):
                deduped[key] = run
        runs_by_dataset[dataset] = list(deduped.values())

    # Write CSVs in tier order: main, special, prelim
    for dataset in MAIN_DATASETS:
        runs = runs_by_dataset.get(dataset, [])
        if runs:
            write_dataset_csv(output_dir, dataset, runs)
        else:
            print(f"  {dataset}: no runs found")

    if include_special:
        for dataset in SPECIAL_DATASETS:
            runs = runs_by_dataset.get(dataset, [])
            if runs:
                write_dataset_csv(output_dir, dataset, runs)
            else:
                print(f"  {dataset}: no runs found")

    if include_prelim:
        for dataset in PRELIM_DATASETS:
            runs = runs_by_dataset.get(dataset, [])
            if runs:
                write_dataset_csv(output_dir, dataset, runs)
            else:
                print(f"  {dataset}: no runs found")

    # Summary
    print("\nSummary:")
    all_active = (MAIN_DATASETS
                  + (SPECIAL_DATASETS if include_special else [])
                  + (PRELIM_DATASETS if include_prelim else []))
    for dataset in all_active:
        runs = runs_by_dataset.get(dataset, [])
        if runs:
            total_epochs = sum(len(r.epochs) for r in runs)
            by_type = defaultdict(int)
            for r in runs:
                by_type[r.run_type] += 1
            type_str = ", ".join(f"{t}={c}" for t, c in sorted(by_type.items()))
            print(f"  {dataset}: {len(runs)} runs ({total_epochs} epochs) - {type_str}")

    print(f"\nDone. CSVs written to {output_dir}/")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract training losses for scaling law fits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From downloaded HF repo (reads .npz metrics):
  python get_results.py from_hf ./colinear_scaling_models -o extracted_losses

  # From the cluster training_metrics (flat dump):
  python get_results.py from_local /path/to/training_metrics -o extracted_losses

  # Include BF16 replication:
  python get_results.py from_local /path/to/metrics -o extracted_losses --include-special

  # Include BigTPP preliminary experiment:
  python get_results.py from_local /path/to/metrics -o extracted_losses --include-prelim
        """,
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Shared flag definitions to keep both subcommands in sync
    def _add_shared(p):
        p.add_argument("-o", "--output-dir", default="extracted_losses",
                       help="Output directory for CSVs (default: extracted_losses)")
        p.add_argument("--include-special", action="store_true",
                       help="Also extract BF16 replication (wikipedia_bf16)")
        p.add_argument("--include-prelim", action="store_true",
                       help="Also extract BigTPP preliminary experiment "
                            "(wikipedia_bigtpp)")
        p.add_argument("--no-final-loss", action="store_true",
                       help="Use epoch averages instead of trailing-100-batch "
                            "converged loss")

    # from_hf
    hf_parser = subparsers.add_parser(
        "from_hf",
        help="Extract from HuggingFace repo directory structure (.npz metrics)",
    )
    hf_parser.add_argument("repo_dir", help="Path to downloaded HF repo")
    _add_shared(hf_parser)

    # from_local
    local_parser = subparsers.add_parser(
        "from_local",
        help="Extract from flat training_metrics directory (.npz files)",
    )
    local_parser.add_argument("metrics_dir", help="Path to training_metrics directory")
    _add_shared(local_parser)

    args = parser.parse_args()

    print("=" * 70)
    print("Scaling Law Results Extractor")
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output_dir}")
    print("Active tiers: main"
          + (" + special(BF16)" if args.include_special else "")
          + (" + prelim(BigTPP)" if args.include_prelim else ""))
    print("=" * 70)

    if args.mode == "from_hf":
        extract_from_hf(
            repo_dir=args.repo_dir,
            output_dir=args.output_dir,
            include_special=args.include_special,
            include_prelim=args.include_prelim,
            use_final_loss=not args.no_final_loss,
        )
    elif args.mode == "from_local":
        extract_from_local(
            metrics_dir=args.metrics_dir,
            output_dir=args.output_dir,
            include_special=args.include_special,
            include_prelim=args.include_prelim,
            use_final_loss=not args.no_final_loss,
        )


if __name__ == "__main__":
    main()