#!/usr/bin/env python3
"""
Reproducibility verification: compare a freshly trained run against the
authoritative reference on HuggingFace.

The script can locate and download the matching reference file automatically
from `leibnitz-lab/colinear_scaling_models`, so there is no need to
download the entire repo (~100s of GB) just to verify one run. Pass
`--from-hf` and the script:

  1. Inspects the reproduced .pt / .npz to read its (dataset, design, N,
     token-count, run-type) identity from the filename.
  2. Lists the matching HF directory and finds the file that shares those
     four identifying tokens, regardless of surrounding-name boilerplate.
  3. Downloads ONLY that single file (cached under ~/.cache/tpp_verify).
  4. Runs the comparison.

Usage:
    # Auto-fetch reference (recommended - nothing to pre-download):
    python verify_reproducibility.py compare --from-hf \\
        --reproduced saved_checkpoints/REPRO_*_metrics.npz

    # Strict weight-level check, auto-fetched:
    python verify_reproducibility.py weights --from-hf \\
        --reproduced saved_checkpoints/REPRO_*_epoch3_*.pt

    # Pair two local files explicitly:
    python verify_reproducibility.py compare \\
        --original colinear_scaling_models/.../ref_metrics.npz \\
        --reproduced training_metrics/REPRO_..._metrics.npz

The authoritative reference is hosted at
https://huggingface.co/datasets/leibnitz-lab/colinear_scaling_models
and was produced on NVIDIA GH200 GPUs (driver ~550.x, CUDA 12.4) in early 2026.
"""

import os
import re
import sys
import glob
import argparse
import numpy as np
import torch


# Keep in sync with src/tpp_analysis/hf_dataset.py (tests/test_entry_points.py checks it).
HF_REPO_ID = "leibnitz-lab/colinear_scaling_models"
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/tpp_verify")

# Run-type tokens we recognize in filenames. Order matters - longer/more-specific
# patterns must precede shorter ones so we don't match HOLDOUT inside
# COLINEAR_HOLDOUT.
_RUN_TYPE_PATTERNS = [
    "COLINEAR_HOLDOUT",
    "NXD_GRID",
    "BIGTPP",
    "HOLDOUT",
    "COLINEAR",
]


def _extract_run_type(fn: str) -> str:
    """Return the run-type token (e.g. 'NXD_GRID') from a filename, or ''."""
    fn_upper = fn.upper()
    for pat in _RUN_TYPE_PATTERNS:
        if pat in fn_upper:
            return pat
    return ""


def _extract_identity(fn: str):
    """Pull (dataset, design, m, token, run_type) from any local or HF
    filename. m and token are the canonical identifiers - they're always
    present and always exact across naming conventions."""
    fn_lower = fn.lower()
    fn_upper = fn.upper()

    # Dataset (special variants take priority - both are wikipedia runs)
    if "BIGTPP" in fn_upper:
        dataset = "wikipedia_bigtpp"
    elif "BF16" in fn_upper or "FP16WIKI" in fn_upper:
        dataset = "wikipedia_bf16"
    else:
        m = re.search(r'_(c4|pes2o|redpajama|wikipedia|cosmopedia)_m\d+_', fn_lower)
        if m:
            dataset = m.group(1)
        else:
            # Last resort - leading dataset prefix on HF files
            m = re.match(r'^(c4|pes2o|redpajama|wikipedia|cosmopedia)_', fn_lower)
            dataset = m.group(1) if m else None

    # Design directory
    if "COLINEAR" in fn_upper:
        design = "colinear"
    else:
        design = "non_colinear"

    # Param count and token count
    m = re.search(r'_m(\d+)_token(\d+)', fn)
    if not m:
        return None
    n_params = int(m.group(1))
    n_tokens = int(m.group(2))

    return {
        "dataset":  dataset,
        "design":   design,
        "m":        n_params,
        "token":    n_tokens,
        "run_type": _extract_run_type(fn),
    }


def _resolve_ambiguous(repo_dir: str, target_fn: str) -> str:
    """List the HF directory and find the file that shares the same
    identity tuple (dataset, design, m{N}, token{D}, run_type) as the target.

    This handles the case where local filenames (e.g.
    `REPRO_NxD_GRID_cosmopedia_m10070160_token35000000_...`) have different
    leading structure than HF filenames (e.g.
    `cosmopedia_NxD_GRID_N10M_D35M_cosmopedia_m10070160_token35000000_...`).
    Both share the same `_m{N}_token{D}_` core, which we use as the anchor."""
    from huggingface_hub import HfApi
    api = HfApi()

    try:
        files = api.list_repo_files(HF_REPO_ID, repo_type="dataset")
    except Exception as e:
        raise RuntimeError(f"Could not list HF repo files: {e}")

    candidates = [f for f in files if f.startswith(repo_dir + "/")]
    if not candidates:
        raise FileNotFoundError(
            f"No reference files found at HF://{repo_dir}/. Check that the "
            f"reproduced run's identity (dataset, design, N) is correct.")

    # Exact match first (rare - local timestamps almost never line up with HF)
    target_path = f"{repo_dir}/{target_fn}"
    if target_path in candidates:
        return target_path

    target_id = _extract_identity(target_fn)
    if target_id is None:
        raise FileNotFoundError(
            f"Could not extract m{{N}}_token{{D}} from {target_fn} for matching")

    target_ext = os.path.splitext(target_fn)[1]

    matches = []
    for c in candidates:
        c_fn = os.path.basename(c)
        if not c_fn.endswith(target_ext):
            continue
        c_id = _extract_identity(c_fn)
        if c_id is None:
            continue
        # Anchor on m and token (canonical identifiers)
        if c_id["m"] != target_id["m"] or c_id["token"] != target_id["token"]:
            continue
        # Run-type must match if we know both - otherwise NxD_GRID and HOLDOUT
        # at the same (m, token) would be confused
        if target_id["run_type"] and c_id["run_type"]:
            if target_id["run_type"] != c_id["run_type"]:
                continue
        matches.append(c)

    if len(matches) == 1:
        print(f"  Identity-matched HF path: {matches[0]}")
        return matches[0]
    if len(matches) > 1:
        # Multiple references for the same (m, token, run_type) - pick the
        # most recent by sort order on the timestamp suffix
        chosen = sorted(matches)[-1]
        print(f"  {len(matches)} candidates; using most recent: {chosen}")
        return chosen

    raise FileNotFoundError(
        f"No HF reference matches m{target_id['m']}_token{target_id['token']} "
        f"(run_type={target_id['run_type'] or '?'}) in {repo_dir}/.\n"
        f"  Available: {[os.path.basename(c) for c in candidates[:5]]}"
        + (f" ... ({len(candidates)} total)" if len(candidates) > 5 else ""))


def _filename_to_repo_dir(filename: str) -> str:
    """Map a local filename to the HF directory containing its reference.

    HF tree structure:
        <dataset>/<design>/N_<n>/<filename>.pt
        training_metrics/<dataset>/<design>/N_<n>/<filename>_metrics.npz
    """
    fn = os.path.basename(filename)
    identity = _extract_identity(fn)
    if identity is None:
        raise ValueError(f"Could not extract identity from filename: {fn}")
    if identity["dataset"] is None:
        raise ValueError(f"Could not detect dataset from filename: {fn}")

    is_metrics = fn.endswith(".npz")
    prefix = "training_metrics/" if is_metrics else ""
    return (
        f"{prefix}{identity['dataset']}/{identity['design']}/N_{identity['m']}"
    )


def fetch_reference(reproduced_path: str,
                     cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    """Download the matching HF reference for a reproduced run.

    Returns the local path. Caches under `cache_dir` so re-runs are instant.
    """
    from huggingface_hub import hf_hub_download

    fn = os.path.basename(reproduced_path)
    print(f"  Resolving HF reference for: {fn}")

    repo_dir = _filename_to_repo_dir(reproduced_path)
    print(f"  HF directory: {repo_dir}/")

    # Strip local-only prefixes (REPRO_, SMOKE_TEST_) before stem matching  - 
    # the matcher uses identity tokens not stems, but cleaning the target
    # filename gives nicer error messages on miss.
    clean_fn = re.sub(r'^(REPRO_TEST_|REPRO_|SMOKE_TEST_)', '', fn)
    repo_path = _resolve_ambiguous(repo_dir, clean_fn)

    print(f"  Downloading from HF: {repo_path}")
    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename=repo_path,
        cache_dir=cache_dir,
    )
    print(f"  Cached at: {local_path}")
    return local_path


# ============================================================
# Loaders
# ============================================================

def load_checkpoint(path: str) -> dict:
    """Load a .pt checkpoint."""
    print(f"  Loading: {os.path.basename(path)}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt


def load_metrics(path: str) -> dict:
    """Load a .npz metrics file."""
    print(f"  Loading: {os.path.basename(path)}")
    data = dict(np.load(path, allow_pickle=True))
    return data


# ============================================================
# Comparators
# ============================================================

def compare_scalars(name: str, orig, repro, atol: float = 1e-6) -> bool:
    """Compare two scalar values."""
    if orig is None and repro is None:
        print(f"  {name}: both None ✓")
        return True
    if orig is None or repro is None:
        print(f"  {name}: MISMATCH - orig={orig}, repro={repro} ✗")
        return False

    orig_f = float(orig)
    repro_f = float(repro)
    diff = abs(orig_f - repro_f)
    match = diff <= atol
    rel_diff = diff / max(abs(orig_f), 1e-12) * 100

    if match:
        print(f"  {name}: {orig_f:.6f} == {repro_f:.6f} (diff={diff:.2e}) ✓")
    else:
        print(f"  {name}: {orig_f:.6f} != {repro_f:.6f} (diff={diff:.2e}, {rel_diff:.4f}%) ✗")
    return match


def compare_arrays(name: str, orig: np.ndarray, repro: np.ndarray,
                   atol: float = 1e-6) -> bool:
    """Compare two numpy arrays element-wise."""
    if len(orig) != len(repro):
        print(f"  {name}: LENGTH MISMATCH - orig={len(orig)}, repro={len(repro)} ✗")
        return False

    diffs = np.abs(orig.astype(float) - repro.astype(float))
    if len(diffs) == 0:
        print(f"  {name}: both empty (length 0) ✓")
        return True
    max_diff = diffs.max()
    mean_diff = diffs.mean()
    match = max_diff <= atol

    if match:
        print(f"  {name}: {len(orig)} values, max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e} ✓")
    else:
        worst_idx = np.argsort(diffs)[-3:][::-1]
        print(f"  {name}: {len(orig)} values, max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e} ✗")
        for idx in worst_idx:
            print(f"    [{idx}] orig={orig[idx]:.6f}, repro={repro[idx]:.6f}, diff={diffs[idx]:.2e}")
    return match


def compare_checkpoints(orig_path: str, repro_path: str, atol: float = 1e-6):
    """Compare two .pt checkpoints."""
    print(f"\n{'='*70}")
    print("CHECKPOINT COMPARISON")
    print(f"{'='*70}")

    orig = load_checkpoint(orig_path)
    repro = load_checkpoint(repro_path)

    results = []

    # Compare metadata
    print("\n--- Metadata ---")
    results.append(compare_scalars("epoch", orig.get("epoch"), repro.get("epoch"), atol=0))
    results.append(compare_scalars("total_tokens", orig.get("total_tokens"), repro.get("total_tokens"), atol=0))
    results.append(compare_scalars("train_loss", orig.get("train_loss"), repro.get("train_loss"), atol=atol))
    results.append(compare_scalars("val_loss", orig.get("val_loss"), repro.get("val_loss"), atol=atol))
    results.append(compare_scalars("cumulative_time", orig.get("cumulative_time"), repro.get("cumulative_time"), atol=float("inf")))
    results.append(compare_scalars("seed", orig.get("seed"), repro.get("seed"), atol=0))
    results.append(compare_scalars("weight_decay", orig.get("weight_decay"), repro.get("weight_decay"), atol=0))
    results.append(compare_scalars("batch_size", orig.get("batch_size"), repro.get("batch_size"), atol=0))

    # Compare model config
    orig_cfg = orig.get("model_config", {})
    repro_cfg = repro.get("model_config", {})
    if orig_cfg and repro_cfg:
        print("\n--- Model Config ---")
        for key in sorted(set(list(orig_cfg.keys()) + list(repro_cfg.keys()))):
            if key in ("total_params",):
                continue
            results.append(compare_scalars(
                f"config.{key}",
                orig_cfg.get(key),
                repro_cfg.get(key),
                atol=0,
            ))

    print_verdict(results)


def compare_metrics(orig_path: str, repro_path: str, atol: float = 1e-5):
    """Compare two .npz metrics files."""
    print(f"\n{'='*70}")
    print("METRICS COMPARISON")
    print(f"{'='*70}")

    orig = load_metrics(orig_path)
    repro = load_metrics(repro_path)

    results = []

    # Key arrays to compare
    key_arrays = [
        ("epoch_train_losses", atol),
        ("val_losses", atol),
        ("batch_losses", atol),
        ("cumulative_tokens", 0),     # Must be exact
        ("learning_rates", 1e-10),    # Should be exact (deterministic schedule)
    ]

    # Informational arrays (will differ due to hardware/timing)
    info_arrays = ["epoch_times", "batch_times"]

    for name, tol in key_arrays:
        if name in orig and name in repro:
            print(f"\n--- {name} ---")
            results.append(compare_arrays(name, orig[name], repro[name], atol=tol))
        elif name in orig or name in repro:
            print(f"\n--- {name}: missing in {'repro' if name in orig else 'orig'} ✗ ---")
            results.append(False)

    print("\n--- Informational (expected to differ) ---")
    for name in info_arrays:
        if name in orig and name in repro:
            compare_arrays(name, orig[name], repro[name], atol=float("inf"))

    print_verdict(results)


def compare_weights(orig_path: str, repro_path: str, atol: float = 1e-6):
    """Compare model weights between two checkpoints."""
    print(f"\n{'='*70}")
    print("WEIGHT COMPARISON")
    print(f"{'='*70}")

    orig = load_checkpoint(orig_path)
    repro = load_checkpoint(repro_path)

    orig_sd = orig.get("model_state_dict", {})
    repro_sd = repro.get("model_state_dict", {})

    if not orig_sd or not repro_sd:
        print("  ERROR: model_state_dict not found in one or both checkpoints")
        return

    orig_keys = set(orig_sd.keys())
    repro_keys = set(repro_sd.keys())

    if orig_keys != repro_keys:
        only_orig = orig_keys - repro_keys
        only_repro = repro_keys - orig_keys
        if only_orig:
            print(f"  Keys only in original: {only_orig}")
        if only_repro:
            print(f"  Keys only in reproduced: {only_repro}")
        return

    print(f"  {len(orig_keys)} parameter tensors")

    results = []
    total_params = 0
    total_exact = 0
    worst_diffs = []

    for key in sorted(orig_keys):
        orig_t = orig_sd[key].float()
        repro_t = repro_sd[key].float()

        if orig_t.shape != repro_t.shape:
            print(f"  {key}: SHAPE MISMATCH {orig_t.shape} vs {repro_t.shape} ✗")
            results.append(False)
            continue

        diff = (orig_t - repro_t).abs()
        max_diff = diff.max().item()
        n_params = orig_t.numel()
        n_exact = (diff == 0).sum().item()

        total_params += n_params
        total_exact += n_exact

        match = max_diff <= atol
        results.append(match)
        worst_diffs.append((max_diff, key))

        if not match:
            pct_exact = n_exact / n_params * 100
            print(f"  {key}: max_diff={max_diff:.2e}, exact={pct_exact:.1f}% ✗")

    pct_exact = total_exact / max(total_params, 1) * 100
    worst_diffs.sort(reverse=True)

    print("\n--- Summary ---")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Bitwise identical: {total_exact:,} ({pct_exact:.2f}%)")
    if worst_diffs:
        print("  Worst diffs:")
        for diff_val, key in worst_diffs[:5]:
            print(f"    {diff_val:.2e}  {key}")

    print_verdict(results)


def print_verdict(results):
    """Print final pass/fail verdict."""
    passed = sum(results)
    total = len(results)
    all_pass = all(results)

    print(f"\n{'='*70}")
    if all_pass:
        print(f"VERDICT: PASS - {passed}/{total} checks passed ✓")
    else:
        failed = total - passed
        print(f"VERDICT: FAIL - {failed}/{total} checks failed ✗")
    print(f"{'='*70}")


# ============================================================
# Glob expansion (so users can write `--reproduced path/REPRO_*.pt`)
# ============================================================

def _expand_glob(path: str, label: str) -> str:
    """Expand glob patterns and pick the most recent match if multiple."""
    if not any(c in path for c in "*?["):
        return path
    matches = sorted(glob.glob(path))
    if not matches:
        raise FileNotFoundError(f"No files matched {label}: {path}")
    if len(matches) > 1:
        print(f"  WARNING: {label} matched {len(matches)} files; "
              f"using most recent: {matches[-1]}")
    return matches[-1]


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify reproducibility by comparing training outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick start (no pre-download needed):
    python verify_reproducibility.py compare --from-hf \\
        --reproduced saved_checkpoints/REPRO_*_metrics.npz

Manually pair two local files:
    python verify_reproducibility.py compare \\
        --original path/to/ref.npz --reproduced path/to/repro.npz
        """,
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    def _add_common(p):
        p.add_argument("--reproduced", required=True,
                       help="Path to reproduced .pt or .npz (glob OK)")
        ref_group = p.add_mutually_exclusive_group(required=True)
        ref_group.add_argument("--original", default=None,
                               help="Path to a local reference file")
        ref_group.add_argument("--from-hf", action="store_true",
                               help=f"Auto-fetch matching reference from "
                                    f"{HF_REPO_ID}")
        p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                       help=f"Where HF references are cached "
                            f"(default: {DEFAULT_CACHE_DIR})")

    cmp_parser = subparsers.add_parser(
        "compare", help="Compare checkpoints or metrics")
    _add_common(cmp_parser)
    cmp_parser.add_argument("--atol", type=float, default=1e-5,
                            help="Absolute tolerance (default: 1e-5)")

    wt_parser = subparsers.add_parser(
        "weights", help="Compare model weights between checkpoints")
    _add_common(wt_parser)
    wt_parser.add_argument("--atol", type=float, default=1e-6,
                           help="Absolute tolerance (default: 1e-6)")

    args = parser.parse_args()

    repro_path = _expand_glob(args.reproduced, "--reproduced")

    if args.from_hf:
        try:
            orig_path = fetch_reference(repro_path, cache_dir=args.cache_dir)
        except Exception as e:
            print(f"\nFailed to fetch HF reference: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        orig_path = _expand_glob(args.original, "--original")

    if args.mode == "compare":
        if orig_path.endswith(".npz") and repro_path.endswith(".npz"):
            compare_metrics(orig_path, repro_path, atol=args.atol)
        elif orig_path.endswith(".pt") and repro_path.endswith(".pt"):
            compare_checkpoints(orig_path, repro_path, atol=args.atol)
        else:
            print("Both files must be the same type (.pt or .npz). "
                  f"Got orig={orig_path}, repro={repro_path}", file=sys.stderr)
            sys.exit(1)
    elif args.mode == "weights":
        if not (orig_path.endswith(".pt") and repro_path.endswith(".pt")):
            print(f"weights mode requires .pt files. "
                  f"Got orig={orig_path}, repro={repro_path}", file=sys.stderr)
            sys.exit(1)
        compare_weights(orig_path, repro_path, atol=args.atol)


if __name__ == "__main__":
    main()