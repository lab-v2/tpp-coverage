"""Shared fixtures. The real grid is small enough to use directly in tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from tpp_large_scale.config import REPO_ROOT, AnalysisConfig  # noqa: E402
from tpp_large_scale.model.grid import FitArrays, TrainingGrid  # noqa: E402


@pytest.fixture(scope="session")
def config() -> AnalysisConfig:
    return AnalysisConfig()


@pytest.fixture(scope="session")
def grid() -> TrainingGrid:
    return TrainingGrid.from_csv(REPO_ROOT / "data_final_metrics.csv")


@pytest.fixture(scope="session")
def envelope_arrays(grid: TrainingGrid) -> FitArrays:
    return grid.arrays


@pytest.fixture(scope="session")
def synthetic_chinchilla() -> FitArrays:
    """A noiseless Chinchilla surface with known parameters, on a 2D design.

    Ground truth: E=1.5, A=100, alpha=0.35, B=2000, beta=0.28.
    """
    n_values = np.logspace(5, 8.5, 7)
    ratios = np.array([20.0, 70.0, 200.0])
    n_grid, ratio_grid = np.meshgrid(n_values, ratios)
    n_flat = n_grid.ravel()
    d_flat = (n_grid * ratio_grid).ravel()
    loss = 1.5 + 100.0 / np.power(n_flat, 0.35) + 2000.0 / np.power(d_flat, 0.28)
    return FitArrays(n=n_flat, d=d_flat, loss=loss)


@pytest.fixture
def tiny_runs_frame() -> pd.DataFrame:
    """Minimal well-formed grid export covering two sizes and two TPP bands."""
    rows = []
    for n_layers, d_model, n_total in ((2, 48, 5_000_000), (6, 384, 50_000_000)):
        for design_tpp in (20.0, 120.0):
            for lr, loss in ((0.01, 5.0), (0.001, 5.4)):
                tokens = design_tpp * n_total
                rows.append(
                    {
                        "n_layers": n_layers,
                        "d_model": d_model,
                        "N_grid": n_total,
                        "D": tokens,
                        "tokens": tokens,
                        "final_val_loss": loss,
                        "has_final": True,
                        "LR": lr,
                    }
                )
    return pd.DataFrame(rows)
