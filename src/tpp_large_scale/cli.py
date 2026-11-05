"""Command-line entry point.

Usage, from the repository root::

    uv run python large_scale/scripts/run_analysis.py                 # full analysis + figures
    uv run python large_scale/scripts/run_analysis.py --no-figures    # results only
    uv run python large_scale/scripts/run_analysis.py --check-only    # parity with the paper's code
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import AnalysisConfig
from .model.fitting import verify_upstream_parity
from .figures import FigureBuilder
from .model.grid import TrainingGrid
from .runner import AnalysisRunner
from .upstream import UPSTREAM_PACKAGE_DIR, UpstreamNotFound


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tpp-large-scale",
        description=(
            "Large-scale validation of TPP-coverage scaling-law conditioning, "
            "reusing the companion paper's own analysis code."
        ),
    )
    parser.add_argument(
        "--data", type=Path, default=None, help="grid metrics CSV (default: repo data file)"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=None, help="where to write results.json"
    )
    parser.add_argument("--figures-dir", type=Path, default=None, help="where to write PNGs")
    parser.add_argument("--no-figures", action="store_true", help="skip figure rendering")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "worker processes for the resampling experiments "
            "(default: all available cores minus 2; results do not depend on this)"
        ),
    )
    parser.add_argument(
        "--no-parity-check",
        action="store_true",
        help="skip the upstream-fitter parity assertion",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify the upstream code is importable and in parity, then exit",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    defaults = AnalysisConfig.__dataclass_fields__
    overrides: dict[str, Path] = {}
    if args.data is not None:
        overrides["metrics_csv"] = args.data
    if args.results_dir is not None:
        overrides["results_dir"] = args.results_dir
    if args.figures_dir is not None:
        overrides["figures_dir"] = args.figures_dir
    if not overrides:
        return AnalysisConfig()
    base = {k: v.default for k, v in defaults.items() if k not in ("seeds",)}
    base.update(overrides)
    return AnalysisConfig(**{k: v for k, v in base.items() if v is not None})


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _config_from_args(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    try:
        if args.check_only:
            grid = TrainingGrid.from_csv(config.metrics_csv)
            deltas = verify_upstream_parity(grid.arrays)
            print(f"upstream package: {UPSTREAM_PACKAGE_DIR}")
            print(f"SSE delta vs the paper's own fitters: {deltas}")
            print("PARITY OK")
            return 0

        runner = AnalysisRunner(config, workers=args.workers)
        results = runner.run(check_parity=not args.no_parity_check)

        if not args.no_figures:
            grid = TrainingGrid.from_csv(config.metrics_csv)
            written = FigureBuilder(config, grid).build_all(results)
            print(f"=== SAVED {len(written)} figures to {config.figures_dir} ===")
        return 0
    except UpstreamNotFound as error:
        print(f"error: {error}", file=sys.stderr)
        return 3
    except AssertionError as error:
        print(f"parity check failed: {error}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
