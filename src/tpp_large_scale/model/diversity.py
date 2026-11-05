"""TPP-diversity criterion (V_K, tau_K, Regime A/B).

All three quantities come from the paper's own implementation in
``subset_analysis.subset_analysis_bbox_seeded`` (``compute_VK``,
``compute_tauK``, ``classify_regime``). This module only wraps them in typed
value objects and adds the two-ray inversion (the minimum spread R that clears
the threshold), which upstream does not provide.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..config import KAPPA_TARGETS, Regime
from ..upstream import classify_regime, compute_tauK, compute_VK


@dataclass(frozen=True)
class DiversityVerdict:
    """Whether a design clears the diversity threshold for one kappa target."""

    kappa_target: float
    v_k: float
    tau_k: float
    regime: Regime

    @property
    def well_conditioned(self) -> bool:
        return self.regime is Regime.WELL_CONDITIONED

    def to_dict(self) -> dict[str, object]:
        return {"tau_K": self.tau_k, "regime": self.regime.value}


class DiversityCriterion:
    """The paper's necessary-and-sufficient TPP-diversity test.

    Args:
        beta_eff: The fitted data exponent, which sets the k^-beta transform.
    """

    def __init__(self, beta_eff: float) -> None:
        if not np.isfinite(beta_eff):
            raise ValueError(f"beta_eff must be finite, got {beta_eff}")
        self._beta_eff = float(beta_eff)

    @property
    def beta_eff(self) -> float:
        return self._beta_eff

    def v_k(self, ratios: Sequence[float]) -> float:
        """Achieved TPP diversity V_K of a design."""
        if len(ratios) == 0:
            raise ValueError("cannot compute V_K of an empty design")
        return float(compute_VK(np.asarray(ratios, dtype=float), self._beta_eff))

    def tau_k(self, ratios: Sequence[float], kappa_target: float) -> float:
        """Threshold tau_K a design must clear for the given kappa target."""
        if kappa_target <= 0:
            raise ValueError(f"kappa target must be positive, got {kappa_target}")
        return float(compute_tauK(np.asarray(ratios, dtype=float), self._beta_eff, kappa_target))

    def verdict(self, ratios: Sequence[float], kappa_target: float) -> DiversityVerdict:
        v_k = self.v_k(ratios)
        tau_k = self.tau_k(ratios, kappa_target)
        return DiversityVerdict(
            kappa_target=kappa_target,
            v_k=v_k,
            tau_k=tau_k,
            regime=Regime(classify_regime(v_k, tau_k)),
        )

    def sweep_targets(
        self,
        ratios: Sequence[float],
        kappa_targets: Sequence[float] = KAPPA_TARGETS,
    ) -> dict[str, DiversityVerdict]:
        """Verdicts keyed by a ``"1e+04"``-style label, one per target."""
        return {f"{t:.0e}": self.verdict(ratios, t) for t in kappa_targets}

    def minimum_two_ray_spread(
        self,
        kappa_target: float,
        max_spread: float = 10**2.2,
        points: int = 160,
    ) -> float:
        """Smallest two-ray spread R = k2/k1 that reaches ``kappa_target``.

        The paper's Prop. 2 specialised to K=2: a design of two rays is
        well-conditioned exactly when their ratio spread is at least R_min.

        Returns:
            The minimum spread, or ``inf`` if no spread up to ``max_spread``
            clears the threshold.
        """
        if max_spread <= 1.0:
            raise ValueError("max_spread must exceed 1")
        spreads = np.logspace(0.0, np.log10(max_spread), points)
        for spread in spreads:
            if self.verdict([1.0, float(spread)], kappa_target).well_conditioned:
                return float(spread)
        return float("inf")

    def two_ray_curves(
        self,
        kappa_target: float,
        max_spread: float = 10**2.2,
        points: int = 160,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(spreads, V_2, tau_2) for plotting the two-ray threshold crossing."""
        spreads = np.logspace(0.0, np.log10(max_spread), points)
        v_values = np.array([self.v_k([1.0, float(r)]) for r in spreads])
        tau_values = np.array([self.tau_k([1.0, float(r)], kappa_target) for r in spreads])
        return spreads, v_values, tau_values
