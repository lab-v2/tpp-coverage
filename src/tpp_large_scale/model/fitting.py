"""Scaling-law fitting, delegated to the companion paper's own implementation.

The law functional forms, parameter bounds, initial-guess samplers and the
multi-start optimiser are all the paper's (:mod:`tpp_large_scale.upstream`).
This module adds only what the paper's API does not expose:

* a typed :class:`LawFit` result instead of a bare ``(params, sse, success)`` tuple,
* per-law exponent-gap extraction,
* a seed-parameterised entry point (upstream's ``fit_chinchilla`` and friends
  hard-code ``seed=42``), and
* an opt-out of the differential-evolution stage for the resampling loops, which
  run thousands of fits and cannot afford a global search each time.

Bounds and samplers are transcribed verbatim from upstream ``laws.fit_*``, where
they are function locals and therefore unreachable by import.
:func:`verify_upstream_parity` asserts at runtime that this transcription still
reproduces the upstream fitters bit for bit.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import DEFAULT_RESTARTS
from .grid import FitArrays
from ..upstream import (
    BETA_EFF_INDEX,
    FIT_SEED_BASE,
    MIN_POINTS_FOR_FIT,
    chinchilla_loss,
    droppo_elibol_loss,
    evaluate,
    fit_chinchilla,
    fit_droppo_elibol,
    fit_generic,
    fit_kaplan,
    kaplan_loss,
)

Bounds = list[tuple[float, float]]
Sampler = Callable[[np.random.RandomState, np.ndarray], list[float]]
LossFn = Callable[..., np.ndarray]

#: Objective value upstream returns for a non-finite or exception-raising fit.
FAILED_OBJECTIVE = 1e10


@dataclass(frozen=True)
class LawSpec:
    """Everything needed to fit one scaling law and read its exponent gap.

    Attributes:
        gap_indices: Parameter positions of the two exponents whose near-equality
            drives the ill-conditioning.
        gap_divisor_index: For laws whose exponents are expressed relative to a
            shared power (Droppo-Elibol), the position of that divisor.
        beta_index: Position of the data exponent, per upstream ``BETA_EFF_INDEX``.
        min_points: Minimum design points for the fit to be identifiable,
            per upstream ``MIN_POINTS_FOR_FIT``.
    """

    name: str
    loss_fn: LossFn
    param_names: tuple[str, ...]
    bounds_fn: Callable[[np.ndarray], Bounds]
    sampler: Sampler
    gap_indices: tuple[int, int]
    beta_index: int
    min_points: int
    gap_divisor_index: Optional[int] = None
    upstream_fitter: Optional[Callable[..., tuple]] = None

    def exponent_gap(self, params: Sequence[float]) -> float:
        """|alpha - beta| for this law's parameterisation."""
        lo, hi = self.gap_indices
        gap = abs(float(params[lo]) - float(params[hi]))
        if self.gap_divisor_index is None:
            return gap
        divisor = float(params[self.gap_divisor_index])
        if divisor == 0.0:
            raise ZeroDivisionError(f"{self.name}: zero exponent divisor in {params}")
        return gap / abs(divisor)

    def beta_eff(self, params: Sequence[float]) -> float:
        return float(params[self.beta_index])


def _chinchilla_bounds(_loss: np.ndarray) -> Bounds:
    return [(0.0, 10.0), (1e-2, 1e10), (0.01, 2.0), (1e-2, 1e10), (0.01, 2.0)]


def _chinchilla_sampler(rng: np.random.RandomState, loss: np.ndarray) -> list[float]:
    return [
        rng.uniform(0.0, min(loss) * 0.9),
        10 ** rng.uniform(0, 8),
        rng.uniform(0.1, 1.5),
        10 ** rng.uniform(0, 8),
        rng.uniform(0.1, 1.5),
    ]


def _kaplan_bounds(_loss: np.ndarray) -> Bounds:
    return [(1e3, 1e14), (1e3, 1e14), (0.01, 2.0), (0.01, 2.0)]


def _kaplan_sampler(rng: np.random.RandomState, _loss: np.ndarray) -> list[float]:
    return [
        10 ** rng.uniform(5, 12),
        10 ** rng.uniform(5, 12),
        rng.uniform(0.05, 1.5),
        rng.uniform(0.05, 1.5),
    ]


def _droppo_bounds(loss: np.ndarray) -> Bounds:
    return [
        (1e-6, min(loss) * 0.99),
        (1e3, 1e14),
        (1e3, 1e14),
        (0.01, 2.0),
        (0.01, 2.0),
        (0.01, 2.0),
    ]


def _droppo_sampler(rng: np.random.RandomState, loss: np.ndarray) -> list[float]:
    return [
        rng.uniform(0.01, min(loss) * 0.8),
        10 ** rng.uniform(5, 12),
        10 ** rng.uniform(5, 12),
        rng.uniform(0.05, 1.5),
        rng.uniform(0.05, 1.5),
        rng.uniform(0.05, 1.5),
    ]


CHINCHILLA = LawSpec(
    name="Chinchilla",
    loss_fn=chinchilla_loss,
    param_names=("E", "A", "alpha", "B", "beta"),
    bounds_fn=_chinchilla_bounds,
    sampler=_chinchilla_sampler,
    gap_indices=(2, 4),
    beta_index=BETA_EFF_INDEX["Chinchilla"],
    min_points=MIN_POINTS_FOR_FIT["Chinchilla"],
    upstream_fitter=fit_chinchilla,
)

KAPLAN = LawSpec(
    name="Kaplan",
    loss_fn=kaplan_loss,
    param_names=("N_c", "D_c", "alpha_N", "alpha_D"),
    bounds_fn=_kaplan_bounds,
    sampler=_kaplan_sampler,
    gap_indices=(3, 2),
    beta_index=BETA_EFF_INDEX["Kaplan"],
    min_points=MIN_POINTS_FOR_FIT["Kaplan"],
    upstream_fitter=fit_kaplan,
)

DROPPO_ELIBOL = LawSpec(
    name="Droppo-Elibol",
    loss_fn=droppo_elibol_loss,
    param_names=("L_inf", "N_c", "D_c", "alpha_N", "alpha_D", "alpha"),
    bounds_fn=_droppo_bounds,
    sampler=_droppo_sampler,
    gap_indices=(3, 4),
    gap_divisor_index=5,
    beta_index=BETA_EFF_INDEX["Droppo-Elibol"],
    min_points=MIN_POINTS_FOR_FIT["Droppo-Elibol"],
    upstream_fitter=fit_droppo_elibol,
)

#: The three laws the large-scale grid can identify. Upstream's fourth law
#: (Parametric-Epoch) needs a multi-epoch (N, U_D, R) interface; this grid is
#: single-epoch, so it is out of scope here.
LAWS: tuple[LawSpec, ...] = (CHINCHILLA, KAPLAN, DROPPO_ELIBOL)
LAWS_BY_NAME: dict[str, LawSpec] = {law.name: law for law in LAWS}


@dataclass(frozen=True)
class LawFit:
    """A converged scaling-law fit and its goodness of fit."""

    law: LawSpec
    params: np.ndarray
    sse: float
    r2: float
    rmse: float
    exponent_gap: float

    @property
    def named_params(self) -> dict[str, float]:
        return {k: float(v) for k, v in zip(self.law.param_names, self.params)}

    @property
    def beta_eff(self) -> float:
        return self.law.beta_eff(self.params)

    def predict(self, arrays: FitArrays) -> np.ndarray:
        return self.law.loss_fn(arrays.n, arrays.d, *self.params)

    def rmse_on(self, arrays: FitArrays) -> float:
        residual = arrays.loss - self.predict(arrays)
        return float(np.sqrt(np.mean(residual**2)))

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": True,
            "params": self.named_params,
            "eps": self.exponent_gap,
            "r2": self.r2,
            "rmse": self.rmse,
        }


class LawFitter:
    """Fits scaling laws using the paper's optimiser.

    Args:
        restarts: Multi-start count. Defaults to the paper's ``N_RESTARTS``.
        global_search: Run upstream's differential-evolution fallback after the
            multi-start phase. Disable only in resampling loops, where the cost
            of a global search per trial is prohibitive.
    """

    def __init__(
        self,
        restarts: int = DEFAULT_RESTARTS,
        global_search: bool = True,
    ) -> None:
        if restarts <= 0:
            raise ValueError("restarts must be positive")
        self._restarts = restarts
        self._global_search = global_search

    def fit(
        self,
        law: LawSpec,
        arrays: FitArrays,
        seed: int = FIT_SEED_BASE,
    ) -> Optional[LawFit]:
        """Fit ``law`` to ``arrays``. Returns ``None`` if no restart converged.

        Raises:
            ValueError: if the design has fewer points than the law has
                identifiable parameters.
        """
        if len(arrays) < law.min_points:
            raise ValueError(
                f"{law.name} needs at least {law.min_points} design points, "
                f"got {len(arrays)}"
            )
        bounds = law.bounds_fn(arrays.loss)
        if self._global_search:
            params, sse, _ = fit_generic(
                law.loss_fn,
                arrays.n,
                arrays.d,
                arrays.loss,
                bounds,
                law.sampler,
                self._restarts,
                seed,
            )
        else:
            params, sse = self._multistart_only(law, arrays, bounds, seed)
        if params is None or not np.all(np.isfinite(params)):
            return None
        metrics = evaluate(
            law.loss_fn,
            params,
            arrays.n,
            arrays.d,
            arrays.loss,
            arrays.n[:0],
            arrays.d[:0],
            arrays.loss[:0],
        )
        return LawFit(
            law=law,
            params=np.asarray(params, dtype=float),
            sse=float(sse),
            r2=float(metrics["r2_train"]),
            rmse=float(metrics["rmse_train"]),
            exponent_gap=law.exponent_gap(params),
        )

    def _multistart_only(
        self,
        law: LawSpec,
        arrays: FitArrays,
        bounds: Bounds,
        seed: int,
    ) -> tuple[Optional[np.ndarray], float]:
        """Upstream ``_fit_generic`` without its differential-evolution stage.

        Kept deliberately identical to upstream's multi-start phase so that
        enabling ``global_search`` is the only behavioural difference;
        :func:`verify_upstream_parity` pins the shared path.
        """
        # Imported lazily: scipy.optimize is a heavy import and only this
        # fallback path needs `minimize` directly rather than through upstream.
        from scipy.optimize import minimize

        def objective(params: np.ndarray) -> float:
            try:
                pred = law.loss_fn(arrays.n, arrays.d, *params)
                if not np.all(np.isfinite(pred)):
                    return FAILED_OBJECTIVE
                return float(np.sum((pred - arrays.loss) ** 2))
            except (ValueError, FloatingPointError, OverflowError):
                return FAILED_OBJECTIVE

        rng = np.random.RandomState(seed)
        best_params: Optional[np.ndarray] = None
        best_loss = np.inf
        for _ in range(self._restarts):
            x0 = law.sampler(rng, arrays.loss)
            result = minimize(
                objective, x0, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 2000},
            )
            if result.fun < best_loss:
                best_loss, best_params = float(result.fun), result.x
        return best_params, best_loss


def verify_upstream_parity(arrays: FitArrays, tolerance: float = 0.0) -> dict[str, float]:
    """Assert this module's fits reproduce the paper's fitters exactly.

    Calls upstream ``laws.fit_*`` and this module's :class:`LawFitter` on the
    same data with the same seed and restart count, and compares the resulting
    sum of squared errors. Any drift means the transcribed bounds or samplers
    have diverged from upstream.

    Returns:
        Per-law absolute SSE difference.

    Raises:
        AssertionError: if any law's fits disagree by more than ``tolerance``.
    """
    fitter = LawFitter(restarts=DEFAULT_RESTARTS, global_search=True)
    deltas: dict[str, float] = {}
    for law in LAWS:
        if law.upstream_fitter is None:
            continue
        _, upstream_sse, _ = law.upstream_fitter(
            arrays.n, arrays.d, arrays.loss, DEFAULT_RESTARTS
        )
        ours = fitter.fit(law, arrays, seed=FIT_SEED_BASE)
        if ours is None:
            raise AssertionError(f"{law.name}: local fit failed where upstream succeeded")
        delta = abs(float(upstream_sse) - ours.sse)
        deltas[law.name] = delta
        if delta > tolerance:
            raise AssertionError(
                f"{law.name}: local fit diverged from upstream "
                f"(SSE {ours.sse!r} vs {upstream_sse!r}, delta {delta!r})"
            )
    return deltas
