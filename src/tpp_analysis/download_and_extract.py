#!/usr/bin/env python3
"""
Download training metrics from HuggingFace and extract loss CSVs for scaling law fits.

This is the entry point for reproducing the analysis. By default it downloads ONLY the training
metrics (~100s of MB), not the model checkpoints (~100s of GB). The metrics
contain per-batch losses needed to compute trailing-100-batch converged loss
at each epoch boundary - the values used in all paper tables and figures.

Three dataset tiers:
  - default:           the 5 main corpora.
  - --include-special: BF16 replication (wikipedia_bf16) for Appendix BF16 results.
  - --include-prelim:  BigTPP preliminary single-law experiment (wikipedia_bigtpp).
  These are independent flags, so you can pull just what you need.

Pre-computed results download (separate from raw metrics):
  --include-results: also pull the analysis output directories
                     (extracted_losses, full_coverage_results,
                      seed_variance_results, subset_bbox_seeds,
                      special_results, prelim_bigtpp_results)
                     into the project root, so figures and tables can be
                     checked WITHOUT re-running the fitting pipeline.
  --results-only:    pull JUST the results directories above. Skips raw
                     metrics download and the extract step entirely.
                     Use this for the fastest path to the paper's numbers.

The dataset is public: no HuggingFace account or token is needed. A token in
HF_TOKEN (or from ``hf auth login``) is used if present, for higher rate limits.

Run from src/tpp_analysis/ (or pass paths), inside the project environment:
    uv run python src/tpp_analysis/download_and_extract.py --results-only

Usage:
    # Download metrics and extract CSVs (default - fast, ~100s MB):
    python download_and_extract.py

    # Also download model checkpoints (~100s GB):
    python download_and_extract.py --include-checkpoints

    # Only specific datasets:
    python download_and_extract.py --datasets wikipedia cosmopedia

    # Skip download if you already have the repo:
    python download_and_extract.py --skip-download

    # Include BF16 replication:
    python download_and_extract.py --include-special

    # Include BigTPP preliminary experiment:
    python download_and_extract.py --include-prelim

    # Pull the pre-computed analysis results (use this to skip fitting):
    python download_and_extract.py --results-only

    # Or pull metrics AND results in one go:
    python download_and_extract.py --include-results

Output:
    extracted_losses/
        wikipedia.csv
        cosmopedia.csv
        redpajama.csv
        pes2o.csv
        c4.csv
      (+ wikipedia_bf16.csv     if --include-special)
      (+ wikipedia_bigtpp.csv   if --include-prelim)

    With --include-results or --results-only, also lands at project root:
        extracted_losses/        (pre-extracted CSVs, plus parquet)
        full_coverage_results/
        seed_variance_results/
        subset_bbox_seeds/
        special_results/
        prelim_bigtpp_results/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hf_dataset import REPO_ID  # noqa: E402
MAIN_DATASETS = ["wikipedia", "cosmopedia", "redpajama", "pes2o", "c4"]

# These are the analysis-output directories on HF. Pulled when --include-results or --results-only
# is set. Each lands at the project root with the same name.
RESULTS_DIRS = [
    "extracted_losses",
    "full_coverage_results",
    "seed_variance_results",
    "subset_bbox_seeds",
    "special_results",
    "prelim_bigtpp_results",
]


def download(repo_dir: str, datasets: list[str] | None = None,
             include_checkpoints: bool = False,
             include_special: bool = False,
             include_prelim: bool = False):
    """Download from HuggingFace.

    Tier flags layer additive partitions onto the download:
      - include_special → also pull wikipedia_bf16/training_metrics
      - include_prelim  → also pull wikipedia_bigtpp/training_metrics
    """
    from huggingface_hub import snapshot_download

    print(f"Downloading from {REPO_ID}...")
    print(f"  Destination: {repo_dir}")

    kwargs = dict(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=repo_dir,
    )

    patterns = ["README.md"]

    if include_checkpoints:
        # Download everything (or filtered datasets)
        if datasets:
            for ds in datasets:
                patterns.append(f"{ds}/**")
                patterns.append(f"training_metrics/{ds}/**")
            print(f"  Downloading checkpoints + metrics for: {datasets}")
        else:
            # No filter - download all
            patterns = None
            print("  Downloading full repo (checkpoints + metrics)")
    else:
        # Metrics only (default - much smaller)
        if datasets:
            for ds in datasets:
                patterns.append(f"training_metrics/{ds}/**")
            print(f"  Downloading metrics only for: {datasets}")
        else:
            patterns.append("training_metrics/**")
            print("  Downloading metrics only (all datasets)")

    # Layer in the optional tiers (only meaningful if patterns is not None,
    # i.e. we are NOT already downloading the entire repo).
    if patterns is not None:
        if include_special:
            patterns.append("wikipedia_bf16/**" if include_checkpoints
                            else "training_metrics/wikipedia_bf16/**")
            print("  + including wikipedia_bf16 (BF16 replication)")
        if include_prelim:
            patterns.append("wikipedia_bigtpp/**" if include_checkpoints
                            else "training_metrics/wikipedia_bigtpp/**")
            print("  + including wikipedia_bigtpp (BigTPP preliminary)")
        kwargs["allow_patterns"] = patterns

    snapshot_download(**kwargs)
    print("  Done.\n")


def download_results(dest_root: str = "."):
    """Download pre-computed analysis result directories from HF and place
    them at dest_root/<dirname>/, matching the project's expected layout.

    Each of RESULTS_DIRS lands at dest_root/<dirname>/, so notebooks and
    fitting scripts find them where they expect.
    """
    from huggingface_hub import snapshot_download
    import shutil

    print(f"Downloading pre-computed results from {REPO_ID}...")
    print(f"  Destination root: {os.path.abspath(dest_root)}")

    # Snapshot to a staging dir, then promote each results subfolder to the
    # project root. snapshot_download won't write outside its local_dir, so
    # this two-step is the cleanest way to avoid mixing results with the
    # raw-metrics layout in the same directory.
    staging = os.path.join(dest_root, ".hf_results_staging")
    os.makedirs(staging, exist_ok=True)

    patterns = ["README.md"] + [f"{d}/**" for d in RESULTS_DIRS]

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=staging,
        allow_patterns=patterns,
    )

    promoted = []
    skipped = []
    for d in RESULTS_DIRS:
        src = os.path.join(staging, d)
        dst = os.path.join(dest_root, d)
        if not os.path.isdir(src):
            skipped.append(d)
            continue
        if os.path.exists(dst):
            print(f"  WARN  {d}/ already exists locally; merging from HF "
                  f"(existing files will be overwritten where names collide)")
            # Walk and copy file-by-file so we don't blow away anything
            # that's only local (e.g. user's own intermediates).
            for root, _, files in os.walk(src):
                rel = os.path.relpath(root, src)
                target = os.path.join(dst, rel) if rel != "." else dst
                os.makedirs(target, exist_ok=True)
                for f in files:
                    shutil.copy2(os.path.join(root, f),
                                 os.path.join(target, f))
        else:
            shutil.move(src, dst)
        promoted.append(d)
        print(f"  ✓ {d}/")

    # Clean up staging
    try:
        shutil.rmtree(staging, ignore_errors=True)
    except Exception:
        pass

    print(f"\n  Promoted {len(promoted)} result directories into "
          f"{os.path.abspath(dest_root)}/")
    if skipped:
        print(f"  Skipped (not on HF): {', '.join(skipped)}")
    print()


def extract(repo_dir: str, output_dir: str,
            include_special: bool = False,
            include_prelim: bool = False):
    """Extract loss CSVs from downloaded metrics."""
    from get_results.get_results import extract_from_hf

    # Metrics are in repo_dir/training_metrics/{dataset}/{design}/N_{n}/*.npz
    metrics_dir = os.path.join(repo_dir, "training_metrics")

    if not os.path.isdir(metrics_dir):
        print(f"ERROR: {metrics_dir} not found.")
        print("Make sure you downloaded the repo first (remove --skip-download).")
        return

    print(f"Extracting losses from {metrics_dir}...")
    extract_from_hf(
        repo_dir=metrics_dir,
        output_dir=output_dir,
        include_special=include_special,
        include_prelim=include_prelim,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Download training metrics and extract loss CSVs for scaling law fits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: download metrics only, extract main 5-dataset CSVs
    python download_and_extract.py

    # Specific datasets only
    python download_and_extract.py --datasets wikipedia c4

    # Include model checkpoints (large download)
    python download_and_extract.py --include-checkpoints

    # Already downloaded, just re-extract
    python download_and_extract.py --skip-download

    # BF16 replication
    python download_and_extract.py --include-special

    # BigTPP preliminary experiment
    python download_and_extract.py --include-prelim

    # Pull the pre-computed analysis outputs (skip fitting yourself)
    python download_and_extract.py --results-only

    # Pull metrics AND pre-computed results in one go
    python download_and_extract.py --include-results
        """,
    )
    parser.add_argument("--repo-dir", default="colinear_scaling_models",
                       help="Where to download / find the repo (default: ./colinear_scaling_models)")
    parser.add_argument("-o", "--output-dir", default="extracted_losses",
                       help="Output directory for CSVs (default: extracted_losses)")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip download, use existing repo directory")
    parser.add_argument("--datasets", nargs="+", choices=MAIN_DATASETS, default=None,
                       help="Only download/extract specific main datasets "
                            "(does not affect --include-special / --include-prelim)")
    parser.add_argument("--include-checkpoints", action="store_true",
                       help="Also download model checkpoints (~100s GB). "
                            "Default downloads only metrics (~100s MB).")
    parser.add_argument("--include-special", action="store_true",
                       help="Also download/extract BF16 replication (wikipedia_bf16)")
    parser.add_argument("--include-prelim", action="store_true",
                       help="Also download/extract BigTPP preliminary "
                            "experiment (wikipedia_bigtpp)")
    parser.add_argument("--include-results", action="store_true",
                       help="Also download the pre-computed analysis result "
                            "directories (extracted_losses, full_coverage_results, "
                            "seed_variance_results, subset_bbox_seeds, "
                            "special_results, prelim_bigtpp_results) and place "
                            "them at the project root, to check tables and "
                            "figures without re-running the fits.")
    parser.add_argument("--results-only", action="store_true",
                       help="Download ONLY the pre-computed result directories "
                            "(implies --include-results) and skip metrics "
                            "download + extraction. Fastest path to paper's "
                            "numbers.")
    parser.add_argument("--results-dest", default=".",
                       help="Where to drop the result directories with "
                            "--include-results / --results-only "
                            "(default: project root)")
    args = parser.parse_args()

    if args.results_only:
        # results_only implies include_results and skips everything else
        args.include_results = True

    print("=" * 60)
    print("Scaling Law Reproducibility - Download & Extract")
    tiers = ["main"]
    if args.include_special: tiers.append("special(BF16)")
    if args.include_prelim:  tiers.append("prelim(BigTPP)")
    if args.include_results: tiers.append("results")
    print(f"Tiers: {' + '.join(tiers)}")
    if args.results_only:
        print("Mode: --results-only (skipping metrics + extract)")
    print("=" * 60)

    # --- Pre-computed results path -------------------------------------
    if args.include_results:
        download_results(dest_root=args.results_dest)

    if args.results_only:
        print("\nDone. Result directories are in place at "
              f"{os.path.abspath(args.results_dest)}/")
        print("You can now run notebooks / fitting verification without "
              "re-doing any of the analysis pipeline.")
        return

    # --- Standard metrics + extract path -------------------------------
    if not args.skip_download:
        download(
            args.repo_dir,
            datasets=args.datasets,
            include_checkpoints=args.include_checkpoints,
            include_special=args.include_special,
            include_prelim=args.include_prelim,
        )
    else:
        if not os.path.isdir(args.repo_dir):
            print(f"ERROR: {args.repo_dir} not found. Remove --skip-download to fetch it.")
            return
        print(f"Skipping download, using {args.repo_dir}\n")

    extract(
        args.repo_dir,
        args.output_dir,
        include_special=args.include_special,
        include_prelim=args.include_prelim,
    )

    print(f"\nCSVs are in {args.output_dir}/. These plug directly into the fitting scripts.")


if __name__ == "__main__":
    main()
