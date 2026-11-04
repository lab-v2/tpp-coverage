"""Orchestration: run the sections in order and persist the results."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .config import AnalysisConfig
from .model.fitting import verify_upstream_parity
from .model.grid import TrainingGrid
from .experiments.parallel import DeterministicExecutor, resolve_worker_count
from .experiments.sections import (
    AnalysisContext,
    AnalysisSection,
    CollinearRaySection,
    DiversityRegimeSection,
    FormalValidationSection,
    FullGridFitsSection,
    HoldoutSection,
    RobustnessSection,
)
from .upstream import UPSTREAM_PACKAGE_DIR, package_dir_label

RESULTS_FILENAME = "results.json"
ENVELOPE_FILENAME = "envelope.csv"


def write_json_atomically(path: Path, payload: dict[str, object]) -> None:
    """Write JSON via a temporary file and rename, so readers never see a partial file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    os.replace(temporary, path)


class AnalysisRunner:
    """Runs every section in dependency order and writes the results document.

    Sections are ordered, not independent: the diversity, ray, robustness and
    formal-validation sections all read the full-grid Chinchilla fit, so
    :class:`FullGridFitsSection` must run first. The order here is the contract.
    """

    def __init__(
        self, config: AnalysisConfig | None = None, workers: int | None = None
    ) -> None:
        self._config = config or AnalysisConfig()
        self._config.ensure_output_dirs()
        self._workers = workers

    @staticmethod
    def default_sections() -> list[AnalysisSection]:
        return [
            FullGridFitsSection(),
            DiversityRegimeSection(),
            CollinearRaySection(),
            HoldoutSection(),
            RobustnessSection(),
            FormalValidationSection(),
        ]

    def run(self, check_parity: bool = True) -> dict[str, object]:
        """Execute the analysis and return the full results document."""
        grid = TrainingGrid.from_csv(self._config.metrics_csv)
        self._print_grid_summary(grid)

        context = AnalysisContext(
            config=self._config,
            grid=grid,
            executor=DeterministicExecutor(self._workers),
        )
        print(
            f"parallelism: up to {resolve_worker_count(self._workers, 1 << 20)} worker "
            "processes, BLAS pinned to 1 thread each (results are workers-invariant)"
        )
        results: dict[str, object] = {"grid": grid.describe()}

        if check_parity:
            print("\n=== UPSTREAM PARITY ===")
            deltas = verify_upstream_parity(grid.arrays)
            print(f"  reusing {UPSTREAM_PACKAGE_DIR}")
            print(f"  SSE delta vs paper's own fitters: {deltas}")
            results["upstream_parity"] = {
                "package_dir": package_dir_label(),
                "sse_delta_by_law": deltas,
            }

        for section in self.default_sections():
            print(f"\n=== {section.title} ===")
            results[section.key] = section.run(context)
            # The win-rate significance check needs the holdout summary, and the
            # equal-n control is produced as a by-product of the ray section.
            if isinstance(section, HoldoutSection):
                context.holdout_summary = results[section.key]
            if isinstance(section, CollinearRaySection):
                results["equal_n_control"] = section.equal_n_control

        self._persist(results, grid)
        return results

    def _print_grid_summary(self, grid: TrainingGrid) -> None:
        summary = grid.describe()
        print("=== GRID STRUCTURE ===")
        print(
            f"finished runs: {summary['finished_runs']}  |  "
            f"envelope cells: {summary['envelope_cells']}"
        )
        print(f"N (non-embed) levels: {[f'{x:.3g}' for x in summary['N_nonembed_levels']]}")
        print(
            f"design-TPP range: [{summary['tpp_design_min']:.1f}, "
            f"{summary['tpp_design_max']:.1f}]  bands: {summary['tpp_bands']}"
        )
        print(f"k=D/N_nonembed range: [{summary['k_fit_min']:.1f}, {summary['k_fit_max']:.1f}]")
        print("cells per (N x TPP-band):")
        print(grid.envelope.groupby(["N", "band"]).size().unstack(fill_value=0))

    def _persist(self, results: dict[str, object], grid: TrainingGrid) -> None:
        results_path = self._config.results_dir / RESULTS_FILENAME
        envelope_path = self._config.results_dir / ENVELOPE_FILENAME
        write_json_atomically(results_path, results)
        grid.envelope.to_csv(envelope_path, index=False)
        print(f"\n=== SAVED {results_path.name}, {envelope_path.name} ===")
