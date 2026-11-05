"""Loading the 4D grid and reducing it to the compute-optimal envelope."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import MIN_CELLS_FOR_RAY, NONEMBED_PARAMS, TPP_BANDS

#: Columns the raw grid export must provide.
REQUIRED_COLUMNS = frozenset(
    {"n_layers", "d_model", "N_grid", "D", "tokens", "final_val_loss", "has_final"}
)


@dataclass(frozen=True)
class FitArrays:
    """The (N, D, L) triple a scaling-law fitter consumes."""

    n: np.ndarray
    d: np.ndarray
    loss: np.ndarray

    def __len__(self) -> int:
        return int(self.n.size)

    @property
    def ratios(self) -> np.ndarray:
        """Fit-space ratio k = D / N_nonembed."""
        return self.d / self.n


class LogBandAssigner:
    """Snap a design TPP to the nearest band in **log** space.

    Upstream's ``incremental_coverage.coverage.snap_to_nearest`` snaps on linear
    distance. Because the bands are quarter-decade spaced, linear snapping is
    biased toward the lower band near every midpoint (TPP 90 would snap to 67
    rather than 120), so this project uses the log-distance rule instead. The
    divergence is deliberate and documented in ``large_scale/UPSTREAM_NOTES.md``.
    """

    def __init__(self, bands: tuple[float, ...] = TPP_BANDS) -> None:
        if not bands:
            raise ValueError("at least one band is required")
        if any(b <= 0 for b in bands):
            raise ValueError("bands must be positive for log-space snapping")
        self._bands = tuple(sorted(bands))
        self._log_bands = np.log(np.asarray(self._bands, dtype=float))

    @property
    def bands(self) -> tuple[float, ...]:
        return self._bands

    def assign(self, tpp: float) -> float:
        if tpp <= 0:
            raise ValueError(f"design TPP must be positive, got {tpp}")
        return float(self._bands[int(np.argmin(np.abs(self._log_bands - np.log(tpp))))])

    def assign_all(self, tpp: np.ndarray) -> np.ndarray:
        return np.array([self.assign(float(t)) for t in tpp], dtype=float)


class TrainingGrid:
    """The 4D (N, D, LR, WD) grid and its compute-optimal envelope.

    The envelope takes the **minimum** loss per (N, D) cell across the
    learning-rate / weight-decay sweep. The median over hyperparameters mixes in
    mistuned and diverged runs and its fit degenerates; the min frontier is the
    standard reduction.
    """

    def __init__(self, runs: pd.DataFrame, assigner: LogBandAssigner | None = None) -> None:
        missing = REQUIRED_COLUMNS - set(runs.columns)
        if missing:
            raise ValueError(f"grid export is missing columns: {sorted(missing)}")
        self._assigner = assigner or LogBandAssigner()
        self._runs = self._annotate(runs)
        self._envelope = self._build_envelope(self._runs)

    @classmethod
    def from_csv(cls, path: Path, assigner: LogBandAssigner | None = None) -> "TrainingGrid":
        if not path.is_file():
            raise FileNotFoundError(f"grid metrics CSV not found: {path}")
        return cls(pd.read_csv(path), assigner)

    def _annotate(self, runs: pd.DataFrame) -> pd.DataFrame:
        finished = runs[runs["has_final"].astype(bool)].copy()
        if finished.empty:
            raise ValueError("no finished runs in the grid export")
        shapes = list(zip(finished["n_layers"].astype(int), finished["d_model"].astype(int)))
        unknown = sorted({s for s in shapes if s not in NONEMBED_PARAMS})
        if unknown:
            raise KeyError(
                f"no non-embedding parameter count for architectures {unknown}; "
                "extend NONEMBED_PARAMS in config.py"
            )
        finished["N"] = [NONEMBED_PARAMS[s] for s in shapes]
        finished["Dt"] = finished["tokens"].astype(float)
        finished["L"] = finished["final_val_loss"].astype(float)
        finished["Ntot"] = finished["N_grid"].astype(float)
        finished["TPPd"] = finished["D"].astype(float) / finished["Ntot"]
        return finished.reset_index(drop=True)

    def _build_envelope(self, runs: pd.DataFrame) -> pd.DataFrame:
        env = runs.loc[runs.groupby(["N", "D"])["L"].idxmin()].reset_index(drop=True)
        env = env.sort_values(["N", "Dt"]).reset_index(drop=True)
        env["band"] = self._assigner.assign_all(env["TPPd"].to_numpy(dtype=float))
        return env

    @property
    def runs(self) -> pd.DataFrame:
        """All finished runs, annotated."""
        return self._runs

    @property
    def envelope(self) -> pd.DataFrame:
        """One row per (N, D) cell: the best loss across the LR/WD sweep."""
        return self._envelope

    @property
    def arrays(self) -> FitArrays:
        return self.arrays_of(self._envelope)

    @staticmethod
    def arrays_of(frame: pd.DataFrame) -> FitArrays:
        return FitArrays(
            n=frame["N"].to_numpy(dtype=float),
            d=frame["Dt"].to_numpy(dtype=float),
            loss=frame["L"].to_numpy(dtype=float),
        )

    @property
    def model_sizes(self) -> list[float]:
        return sorted(float(x) for x in self._envelope["N"].unique())

    @property
    def populated_bands(self) -> list[float]:
        return sorted(float(b) for b in self._envelope["band"].unique())

    def band(self, value: float) -> pd.DataFrame:
        """Cells assigned to one design-TPP band."""
        return self._envelope[self._envelope["band"] == value]

    def off_band(self, value: float) -> pd.DataFrame:
        """Cells outside one design-TPP band."""
        return self._envelope[self._envelope["band"] != value]

    def ray_bands(self, min_cells: int = MIN_CELLS_FOR_RAY) -> dict[float, pd.DataFrame]:
        """Bands with enough cells to serve as a collinear (CO) training design."""
        return {
            b: sub
            for b in self.populated_bands
            if len(sub := self.band(b)) >= min_cells
        }

    def median_envelope(self) -> pd.DataFrame:
        """Envelope built from the median over (LR, WD) instead of the minimum.

        Used only as a robustness contrast: this reduction degenerates.
        """
        return (
            self._runs.groupby(["N", "D"])
            .agg(N=("N", "first"), Dt=("Dt", "first"), L=("L", "median"))
            .reset_index(drop=True)
        )

    def describe(self) -> dict[str, object]:
        tpp = self._envelope["TPPd"].to_numpy(dtype=float)
        ratios = self.arrays.ratios
        return {
            "finished_runs": int(len(self._runs)),
            "envelope_cells": int(len(self._envelope)),
            "N_nonembed_levels": self.model_sizes,
            "tpp_design_min": float(tpp.min()),
            "tpp_design_max": float(tpp.max()),
            "tpp_bands": self.populated_bands,
            "k_fit_min": float(ratios.min()),
            "k_fit_max": float(ratios.max()),
        }
