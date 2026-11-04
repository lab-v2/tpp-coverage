"""Constants and run configuration for the large-scale validation.

Every magic number the analysis depends on lives here, so one file shows
what was fixed a priori. Seeds are explicit and plumbed through
:class:`AnalysisConfig`; two runs with the same config must agree exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .upstream import KAPPA_TARGETS as UPSTREAM_KAPPA_TARGETS
from .upstream import N_RESTARTS as UPSTREAM_N_RESTARTS

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Non-embedding parameter count per (n_layers, d_model), from the grid's
#: ``grid_4d.csv`` convention. The grid's own ``N_grid`` column is the *total*
#: (embedding-inclusive) budget and is kept separately as the design TPP base.
NONEMBED_PARAMS: dict[tuple[int, int], int] = {
    (2, 48): 56_640,
    (2, 128): 396_800,
    (6, 384): 10_647_552,
    (12, 768): 85_056_000,
    (18, 1280): 354_196_480,
}

#: Quarter-decade design-TPP bands. Cells are snapped to these in *log* space.
TPP_BANDS: tuple[float, ...] = (20.0, 37.0, 67.0, 120.0, 215.0)

#: The one band holding exactly one cell per model size: the clean single ray.
CLEAN_RAY_BAND = 67.0

#: A band needs this many cells before it can stand in as a CO training design.
MIN_CELLS_FOR_RAY = 4

#: Non-embedding N below this is embedding-dominated (>97% embeddings); the
#: "drop small N" robustness check refits above it.
SMALL_N_CUTOFF = 1e7

#: Target condition numbers at which the diversity criterion is evaluated.
KAPPA_TARGETS: tuple[float, ...] = tuple(float(k) for k in UPSTREAM_KAPPA_TARGETS)

#: Restart count used by the paper's fitters.
DEFAULT_RESTARTS: int = int(UPSTREAM_N_RESTARTS)

#: Restarts for single-ray and control fits. Lower than the full-grid count
#: because these designs have at most 10 points.
RAY_RESTARTS = 200

#: Restarts for the inner loops of resampling experiments, where thousands of
#: fits run and the global search is disabled for tractability.
TRIAL_RESTARTS = 60

#: Restarts for the smallest resampled designs (5-point subsets, two-band pairs).
SUBSET_RESTARTS = 40

#: Irreducible-loss value used by the "E fixed" robustness refit.
FIXED_IRREDUCIBLE_LOSS = 0.1

#: FLOPs per parameter per token (the standard 6ND accounting).
FLOPS_PER_PARAM_TOKEN = 6.0



class Regime(str, Enum):
    """Conditioning regime from the paper's diversity criterion."""

    ILL_CONDITIONED = "A"
    WELL_CONDITIONED = "B"


@dataclass(frozen=True)
class SeedBook:
    """Every seed the analysis consumes, in one auditable place."""

    primary_fit: int = 42
    holdout_driver: int = 0
    robustness_driver: int = 1
    two_band_fit: int = 7
    clean_ray_fit: int = 5

    def __post_init__(self) -> None:
        if self.primary_fit < 0:
            raise ValueError("seeds must be non-negative")


@dataclass(frozen=True)
class AnalysisConfig:
    """Paths and sample sizes for one end-to-end run."""

    repo_root: Path = REPO_ROOT
    metrics_csv: Path = field(default=REPO_ROOT / "data_final_metrics.csv")
    results_dir: Path = field(default=REPO_ROOT / "results")
    figures_dir: Path = field(default=REPO_ROOT / "figs")
    seeds: SeedBook = field(default_factory=SeedBook)

    holdout_trials: int = 200
    robustness_trials: int = 200
    random_subset_trials: int = 400
    random_subset_size: int = 5
    bootstrap_resamples: int = 1000
    eps_sweep_points: int = 40
    eps_sweep_max: float = 0.12
    eps_sweep_min: float = 1e-3
    rmin_sweep_points: int = 160

    def __post_init__(self) -> None:
        if not self.metrics_csv.is_file():
            raise FileNotFoundError(f"grid metrics CSV not found: {self.metrics_csv}")
        for positive in (self.holdout_trials, self.robustness_trials, self.random_subset_trials):
            if positive <= 0:
                raise ValueError("trial counts must be positive")

    def ensure_output_dirs(self) -> None:
        """Create the output directories if they do not exist."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
