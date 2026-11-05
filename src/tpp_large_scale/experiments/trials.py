"""Self-contained work items for the resampling experiments, and their workers.

Each work item carries its own data and its own seed, so a worker is a pure
function of its argument with no shared or global state. Items are small (the
envelope has 29 cells), so embedding the arrays costs less than coordinating
shared memory and removes an entire class of race condition.

Workers live at module level because :class:`~tpp_large_scale.experiments.parallel.DeterministicExecutor`
must be able to pickle them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..model.conditioning import GaussNewtonConditioning
from ..model.fitting import CHINCHILLA, LawFitter
from ..model.grid import FitArrays
from ..upstream import evaluate

Vector = tuple[float, ...]


def _arrays(n: Vector, d: Vector, loss: Vector) -> FitArrays:
    return FitArrays(
        n=np.asarray(n, dtype=float),
        d=np.asarray(d, dtype=float),
        loss=np.asarray(loss, dtype=float),
    )


@dataclass(frozen=True)
class Design:
    """A training design, frozen so it can cross a process boundary."""

    n: Vector
    d: Vector
    loss: Vector

    @classmethod
    def from_frame(cls, frame) -> "Design":
        return cls(
            n=tuple(float(x) for x in frame["N"]),
            d=tuple(float(x) for x in frame["Dt"]),
            loss=tuple(float(x) for x in frame["L"]),
        )

    def to_arrays(self) -> FitArrays:
        return _arrays(self.n, self.d, self.loss)

    def __len__(self) -> int:
        return len(self.n)


@dataclass(frozen=True)
class ComparisonTrial:
    """One budget-matched CO-vs-NC contest on a shared holdout."""

    index: int
    collinear: Design
    non_collinear: Design
    holdout: Design
    collinear_seed: int
    non_collinear_seed: int
    restarts: int


@dataclass(frozen=True)
class ComparisonOutcome:
    """Scores of one contest. ``index`` restores plan order after a parallel map."""

    index: int
    rmse_collinear: float
    rmse_non_collinear: float
    r2_collinear: float
    r2_non_collinear: float


@dataclass(frozen=True)
class ConditioningTrial:
    """One randomly drawn design whose (A, B) conditioning we measure."""

    index: int
    design: Design
    seed: int
    restarts: int


@dataclass(frozen=True)
class ConditioningOutcome:
    index: int
    kappa_scale_block: float


def run_comparison(trial: ComparisonTrial) -> Optional[ComparisonOutcome]:
    """Fit both designs and score them on the shared holdout.

    Returns ``None`` when either fit fails to converge or produces a non-finite
    score, which the caller drops from the trial count.
    """
    fitter = LawFitter(restarts=trial.restarts, global_search=False)
    holdout = trial.holdout.to_arrays()
    scores = []
    for design, seed in (
        (trial.collinear, trial.collinear_seed),
        (trial.non_collinear, trial.non_collinear_seed),
    ):
        train = design.to_arrays()
        fit = fitter.fit(CHINCHILLA, train, seed=seed)
        if fit is None:
            return None
        scores.append(
            evaluate(
                CHINCHILLA.loss_fn,
                fit.params,
                train.n,
                train.d,
                train.loss,
                holdout.n,
                holdout.d,
                holdout.loss,
            )
        )
    collinear, non_collinear = scores
    if not (
        np.isfinite(collinear["rmse_holdout"]) and np.isfinite(non_collinear["rmse_holdout"])
    ):
        return None
    return ComparisonOutcome(
        index=trial.index,
        rmse_collinear=float(collinear["rmse_holdout"]),
        rmse_non_collinear=float(non_collinear["rmse_holdout"]),
        r2_collinear=float(collinear["r2_holdout"]),
        r2_non_collinear=float(non_collinear["r2_holdout"]),
    )


def run_conditioning(trial: ConditioningTrial) -> Optional[ConditioningOutcome]:
    """Fit one design and return its (A, B) block condition number."""
    arrays = trial.design.to_arrays()
    fit = LawFitter(restarts=trial.restarts, global_search=False).fit(
        CHINCHILLA, arrays, seed=trial.seed
    )
    if fit is None:
        return None
    kappa = GaussNewtonConditioning.from_fit(fit, arrays).kappa_scale_block()
    if not np.isfinite(kappa):
        return None
    return ConditioningOutcome(index=trial.index, kappa_scale_block=float(kappa))
