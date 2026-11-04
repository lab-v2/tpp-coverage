"""Gauss-Newton conditioning of a scaling-law design.

This is the one part of the pipeline with **no upstream implementation**. The
paper defines the conditioning objects (the Jacobian block structure, kappa,
kappa_{A,B}, the CI-inflation ratio) analytically in its tables; a repo-wide
search of the paper's code package finds no Jacobian or condition-number code.
Everything here is therefore local, and written to match the paper's
definitions rather than to reproduce any upstream function.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .fitting import CHINCHILLA, LawFit, LawSpec
from .grid import FitArrays

#: Eigenvalues at or below this are treated as a numerically exact zero.
EIGENVALUE_FLOOR = 1e-300

#: Column positions of the two scale coefficients (A, B) in the Chinchilla
#: Jacobian, i.e. the block whose collinearity the paper's Prop. 1 concerns.
CHINCHILLA_SCALE_COLUMNS = (1, 3)


def _condition_number_on_range(matrix: np.ndarray) -> float:
    """Condition number restricted to the non-null subspace.

    Eigenvalues at or below the floor are dropped rather than clamped, so a
    rank-deficient design reports the conditioning of the directions it can
    actually resolve. Used for the *full* Jacobian, where an exactly zero
    eigenvalue means a parameter is absent from the model rather than
    collinear with another.
    """
    eigenvalues = np.linalg.eigvalsh(matrix)
    positive = eigenvalues[eigenvalues > EIGENVALUE_FLOOR]
    if positive.size == 0:
        return float("inf")
    return float(positive[-1] / positive[0])


def _condition_number_clamped(matrix: np.ndarray) -> float:
    """Condition number with the smallest eigenvalue clamped, not dropped.

    Used for two-column blocks, where a vanishing smallest eigenvalue is the
    quantity of interest: it means the two columns are collinear and the
    coefficients are unidentifiable. Dropping it would report a perfectly
    conditioned block for an exactly singular one, which is the opposite of
    the truth. Perfect collinearity yields ``inf``.
    """
    eigenvalues = np.linalg.eigvalsh(matrix)
    largest = float(eigenvalues[-1])
    smallest = float(eigenvalues[0])
    if smallest <= EIGENVALUE_FLOOR:
        return float("inf") if largest > 0 else float("nan")
    return largest / smallest


def chinchilla_jacobian(arrays: FitArrays, params: Sequence[float]) -> np.ndarray:
    """Analytic Jacobian of the Chinchilla law, columns (E, A, alpha, B, beta).

    L(N, D) = E + A N^-alpha + B D^-beta
    """
    _, coeff_a, alpha, coeff_b, beta = (float(p) for p in params)
    n, d = arrays.n, arrays.d
    return np.column_stack(
        [
            np.ones_like(n),
            np.power(n, -alpha),
            -coeff_a * np.power(n, -alpha) * np.log(n),
            np.power(d, -beta),
            -coeff_b * np.power(d, -beta) * np.log(d),
        ]
    )


def numerical_jacobian(
    law: LawSpec,
    arrays: FitArrays,
    params: Sequence[float],
    relative_step: float = 1e-5,
) -> np.ndarray:
    """Central-difference Jacobian, used to cross-check the analytic one."""
    values = np.asarray(params, dtype=float)
    columns = []
    for index in range(values.size):
        step = relative_step * max(abs(values[index]), 1e-8)
        forward, backward = values.copy(), values.copy()
        forward[index] += step
        backward[index] -= step
        columns.append(
            (
                law.loss_fn(arrays.n, arrays.d, *forward)
                - law.loss_fn(arrays.n, arrays.d, *backward)
            )
            / (2 * step)
        )
    return np.column_stack(columns)


@dataclass(frozen=True)
class ConditioningReport:
    """Condition numbers of one design at one parameter vector."""

    kappa_full: float
    kappa_full_scaled: float
    kappa_scale_block: float

    def to_dict(self) -> dict[str, float]:
        return {
            "kappa_full": self.kappa_full,
            "kappa_full_scaled": self.kappa_full_scaled,
            "kappa_AB": self.kappa_scale_block,
        }


class GaussNewtonConditioning:
    """Condition numbers of J^T J for a fitted design.

    Args:
        jacobian: Design Jacobian, one row per design point, one column per
            parameter.
        scale_columns: Positions of the two scale coefficients whose block
            condition number the paper's Prop. 1 bounds.
    """

    def __init__(
        self,
        jacobian: np.ndarray,
        scale_columns: tuple[int, int] = CHINCHILLA_SCALE_COLUMNS,
    ) -> None:
        if jacobian.ndim != 2:
            raise ValueError(f"jacobian must be 2-D, got shape {jacobian.shape}")
        if max(scale_columns) >= jacobian.shape[1]:
            raise IndexError(
                f"scale columns {scale_columns} out of range for "
                f"{jacobian.shape[1]}-column Jacobian"
            )
        self._jacobian = jacobian
        self._scale_columns = scale_columns

    @classmethod
    def from_fit(cls, fit: LawFit, arrays: FitArrays) -> "GaussNewtonConditioning":
        """Build from a Chinchilla fit using the analytic Jacobian.

        Raises:
            NotImplementedError: for laws without an analytic Jacobian here.
        """
        if fit.law is not CHINCHILLA:
            raise NotImplementedError(
                f"no analytic Jacobian for {fit.law.name}; use numerical_jacobian"
            )
        return cls(chinchilla_jacobian(arrays, fit.params))

    @property
    def jacobian(self) -> np.ndarray:
        return self._jacobian

    def kappa_full(self) -> float:
        """Condition number of the full J^T J."""
        return _condition_number_on_range(self._jacobian.T @ self._jacobian)

    def kappa_full_scaled(self) -> float:
        """Column-normalised condition number.

        Removes column-scale disparities so the number reflects collinearity
        rather than the arbitrary units of each parameter.
        """
        norms = np.linalg.norm(self._jacobian, axis=0)
        norms[norms == 0] = 1.0
        normalised = self._jacobian / norms
        return _condition_number_on_range(normalised.T @ normalised)

    def kappa_block(self, columns: tuple[int, int] | None = None) -> float:
        """Condition number of a two-column sub-block of J^T J."""
        selected = list(columns or self._scale_columns)
        block = self._jacobian[:, selected]
        return _condition_number_clamped(block.T @ block)

    def kappa_scale_block(self) -> float:
        """Condition number of the (A, B) scale-coefficient block."""
        return self.kappa_block(self._scale_columns)

    def variance_factor(self, column: int) -> float:
        """Diagonal of inv(J^T J) for one parameter: its variance scale factor.

        Falls back to the pseudo-inverse when J^T J is singular, and clips a
        negative diagonal (round-off on a near-singular inverse) to zero.
        """
        gram = self._jacobian.T @ self._jacobian
        try:
            covariance = np.linalg.inv(gram)
        except np.linalg.LinAlgError:
            covariance = np.linalg.pinv(gram)
        return float(np.clip(covariance[column, column], 0.0, None))

    def report(self) -> ConditioningReport:
        return ConditioningReport(
            kappa_full=self.kappa_full(),
            kappa_full_scaled=self.kappa_full_scaled(),
            kappa_scale_block=self.kappa_scale_block(),
        )
