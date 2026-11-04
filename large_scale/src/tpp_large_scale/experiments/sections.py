"""The analysis sections, one class per block of the report.

Each section is a :class:`AnalysisSection`: it takes the grid and the shared
context, runs one self-contained block of the analysis, and returns the JSON
fragment that lands under its key in ``results.json``. The runner owns
ordering and I/O; sections own only their own computation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..config import (
    CLEAN_RAY_BAND,
    DEFAULT_RESTARTS,
    FIXED_IRREDUCIBLE_LOSS,
    FLOPS_PER_PARAM_TOKEN,
    KAPPA_TARGETS,
    RAY_RESTARTS,
    SMALL_N_CUTOFF,
    SUBSET_RESTARTS,
    TRIAL_RESTARTS,
    AnalysisConfig,
)
from ..model.conditioning import GaussNewtonConditioning
from ..model.diversity import DiversityCriterion
from ..model.fitting import CHINCHILLA, LAWS, FitArrays, LawFit, LawFitter
from ..model.grid import TrainingGrid
from .parallel import DeterministicExecutor, drop_failures
from .trials import (
    ComparisonTrial,
    ConditioningTrial,
    Design,
    run_comparison,
    run_conditioning,
)
from ..upstream import fit_generic

#: A single-band design has one TPP ratio, so V_1 = 0 < tau_1 by construction.
SINGLE_BAND_DIVERSITY = 0.0

#: Confidence level for the normal-approximation win-rate interval.
Z_95 = 1.96


@dataclass
class AnalysisContext:
    """State the runner threads between sections."""

    config: AnalysisConfig
    grid: TrainingGrid
    executor: DeterministicExecutor = field(default_factory=DeterministicExecutor)
    chinchilla_fit: Optional[LawFit] = None
    #: Populated by the runner once the holdout section has run; the win-rate
    #: significance check in the robustness section reads it.
    holdout_summary: Optional[dict[str, object]] = None

    def require_chinchilla(self) -> LawFit:
        if self.chinchilla_fit is None:
            raise RuntimeError("FullGridSection must run before this section")
        return self.chinchilla_fit


class AnalysisSection(ABC):
    """One block of the analysis, producing one key of ``results.json``."""

    #: Key this section writes under in the results document.
    key: str
    #: Human-readable banner printed when the section runs.
    title: str

    @abstractmethod
    def run(self, context: AnalysisContext) -> dict[str, object]:
        """Compute this section and return its JSON fragment."""


class FullGridFitsSection(AnalysisSection):
    """Per-law fits on the full 2D envelope, plus Chinchilla conditioning."""

    key = "full_grid_fits"
    title = "FULL-GRID (NC) FITS"

    def run(self, context: AnalysisContext) -> dict[str, object]:
        fitter = LawFitter(restarts=DEFAULT_RESTARTS, global_search=True)
        arrays = context.grid.arrays
        results: dict[str, object] = {}
        for law in LAWS:
            fit = fitter.fit(law, arrays, seed=context.config.seeds.primary_fit)
            if fit is None:
                results[law.name] = {"ok": False}
                print(f"  {law.name}: FAILED")
                continue
            record = fit.to_dict()
            extra = ""
            if law is CHINCHILLA:
                context.chinchilla_fit = fit
                conditioning = GaussNewtonConditioning.from_fit(fit, arrays)
                record.update(conditioning.report().to_dict())
                record["ci_inflation_approx"] = 1.0 / max(fit.exponent_gap, 1e-9)
                extra = (
                    f"  kappa_AB={record['kappa_AB']:.3g}"
                    f"  kappa_full={record['kappa_full']:.3g}"
                )
            results[law.name] = record
            print(
                f"  {law.name}: eps={fit.exponent_gap:.4f}  R2={fit.r2:.4f}  "
                f"RMSE={fit.rmse:.4f}{extra}"
            )
        return results


class DiversityRegimeSection(AnalysisSection):
    """TPP diversity of the full envelope and its regime at each kappa target."""

    key = "full_grid_tpp"
    title = "FULL-GRID TPP DIVERSITY / REGIME"

    def run(self, context: AnalysisContext) -> dict[str, object]:
        fit = context.require_chinchilla()
        grid = context.grid
        criterion = DiversityCriterion(fit.beta_eff)
        ratios = grid.arrays.ratios
        distinct = sorted(set(np.round(ratios, 6)))

        verdicts = criterion.sweep_targets(ratios)
        for tag, verdict in verdicts.items():
            print(
                f"  kappa_target={tag}: V_K={verdict.v_k:.4g}  "
                f"tau_K={verdict.tau_k:.4g}  -> Regime {verdict.regime.value}"
            )

        band_ratios = np.asarray(grid.populated_bands, dtype=float)
        band_verdicts = criterion.sweep_targets(band_ratios)
        return {
            "beta_eff": fit.beta_eff,
            "K": len(distinct),
            "V_K": criterion.v_k(ratios),
            "k_values_head": [float(x) for x in distinct[:12]],
            "by_target": {t: v.to_dict() for t, v in verdicts.items()},
            "band_view": {
                "K_bands": len(band_ratios),
                "V_K_bands": criterion.v_k(band_ratios),
                "by_target": {t: v.to_dict() for t, v in band_verdicts.items()},
            },
        }


class CollinearRaySection(AnalysisSection):
    """Conditioning of each single-TPP ray, and the equal-n 2D control."""

    key = "co_rays"
    title = "CO (single-TPP ray) vs NC (2D grid)"

    def __init__(self) -> None:
        self._fitter = LawFitter(restarts=RAY_RESTARTS, global_search=True)
        self.equal_n_control: dict[str, object] = {}

    def _kappa_scale_block(self, frame: pd.DataFrame, seed: int) -> Optional[float]:
        arrays = TrainingGrid.arrays_of(frame)
        fit = self._fitter.fit(CHINCHILLA, arrays, seed=seed)
        if fit is None:
            return None
        return GaussNewtonConditioning.from_fit(fit, arrays).kappa_scale_block()

    def run(self, context: AnalysisContext) -> dict[str, object]:
        grid = context.grid
        seed = context.config.seeds.primary_fit
        results: dict[str, object] = {}
        for band, subset in grid.ray_bands().items():
            arrays = TrainingGrid.arrays_of(subset)
            record: dict[str, object] = {
                "n_points": int(len(subset)),
                "k_min": float(arrays.ratios.min()),
                "k_max": float(arrays.ratios.max()),
                # A single design band means K=1, so V_1 = 0 < tau_1: Regime A.
                "V_K_designband": SINGLE_BAND_DIVERSITY,
                "regime_designband": "A",
            }
            fit = self._fitter.fit(CHINCHILLA, arrays, seed=seed)
            if fit is not None:
                conditioning = GaussNewtonConditioning.from_fit(fit, arrays)
                record["kappa_AB"] = conditioning.kappa_scale_block()
                record["eps"] = fit.exponent_gap
            results[f"{band:.0f}"] = record
            print(
                f"  ray TPP~{band:.0f}: n={len(subset)}  "
                f"kappa_AB={record.get('kappa_AB', float('nan')):.3g}  Regime A (K=1)"
            )

        self.equal_n_control = self._run_equal_n_control(context)
        return results

    def _run_equal_n_control(self, context: AnalysisContext) -> dict[str, object]:
        """Contrast the collinear ray with an equal-size, 2D-spread sample.

        Isolates collinearity from sample size: both designs have the same
        number of points, so any conditioning gap is due to the geometry.
        """
        print("\n=== EQUAL-n (5-point) CONTROL: collinear ray vs 2D spread ===")
        grid = context.grid
        seed = context.config.seeds.primary_fit
        ray = grid.band(CLEAN_RAY_BAND)
        spread = grid.envelope.assign(kf=grid.arrays.ratios).sort_values("kf")
        size = min(len(ray), len(spread))
        picks = np.linspace(0, len(spread) - 1, size).round().astype(int)
        two_d = spread.iloc[picks]

        collinear = self._kappa_scale_block(ray, seed)
        non_collinear = self._kappa_scale_block(two_d, seed)
        print(f"  collinear single-TPP ray (n={len(ray)}): kappa_AB={collinear:.3g}")
        print(f"  2D-spread sample        (n={len(two_d)}): kappa_AB={non_collinear:.3g}")
        return {
            "n_collinear": int(len(ray)),
            "collinear_ray_kappa_AB": collinear,
            "n_2d": int(len(two_d)),
            "twod_spread_kappa_AB": non_collinear,
        }


class HoldoutSection(AnalysisSection):
    """Budget-matched CO-vs-NC holdout comparison (the paper's Thm. 1 test)."""

    key = "co_vs_nc_holdout"
    title = "CO-vs-NC HOLDOUT WIN RATE"

    #: Holdout size floor, so a small ray still gets a meaningful test set.
    MIN_HOLDOUT = 6

    def run(self, context: AnalysisContext) -> dict[str, object]:
        trials = self._plan(context)
        print(f"  planned {context.executor.describe(len(trials))}")
        outcomes = drop_failures(context.executor.map(run_comparison, trials))
        outcomes.sort(key=lambda o: o.index)
        return self._summarise(outcomes)

    def _plan(self, context: AnalysisContext) -> list[ComparisonTrial]:
        """Draw every random choice sequentially and freeze it into work items.

        This is the only consumer of the driver RNG, so the draw order is fixed
        regardless of how the trials are later executed.
        """
        grid = context.grid
        rng = np.random.RandomState(context.config.seeds.holdout_driver)
        rays = grid.ray_bands()
        envelope = grid.envelope
        trials: list[ComparisonTrial] = []

        for index in range(context.config.holdout_trials):
            band = rng.choice(list(rays.keys()))
            co_subset = rays[band]
            off_band = grid.off_band(band)
            if len(off_band) < 4:
                continue
            holdout_size = min(len(off_band), max(self.MIN_HOLDOUT, len(co_subset)))
            holdout_index = rng.choice(off_band.index.values, size=holdout_size, replace=False)
            excluded = set(co_subset.index.values) | set(holdout_index)
            pool = envelope.loc[[i for i in envelope.index if i not in excluded]]
            if len(pool) < len(co_subset):
                continue
            nc_subset = pool.sample(n=len(co_subset), random_state=int(rng.randint(1e9)))
            trials.append(
                ComparisonTrial(
                    index=index,
                    collinear=Design.from_frame(co_subset),
                    non_collinear=Design.from_frame(nc_subset),
                    holdout=Design.from_frame(envelope.loc[holdout_index]),
                    collinear_seed=int(rng.randint(1e9)),
                    non_collinear_seed=int(rng.randint(1e9)),
                    restarts=TRIAL_RESTARTS,
                )
            )
        return trials

    def _summarise(self, outcomes: list) -> dict[str, object]:
        rmse_co = [o.rmse_collinear for o in outcomes]
        rmse_nc = [o.rmse_non_collinear for o in outcomes]
        r2_co = [o.r2_collinear for o in outcomes]
        r2_nc = [o.r2_non_collinear for o in outcomes]
        nc_wins = sum(1 for o in outcomes if o.rmse_non_collinear < o.rmse_collinear)
        co_wins = sum(1 for o in outcomes if o.rmse_collinear < o.rmse_non_collinear)
        used = len(rmse_co)
        if used == 0:
            raise RuntimeError("no holdout trial produced a usable comparison")
        win_rate = nc_wins / used
        median_co, median_nc = float(np.median(rmse_co)), float(np.median(rmse_nc))
        median_r2_co, median_r2_nc = float(np.median(r2_co)), float(np.median(r2_nc))
        print(
            f"  trials used: {used}  NC wins: {nc_wins}  CO wins: {co_wins}  "
            f"NC win-rate: {win_rate:.3f}"
        )
        print(
            f"  mean RMSE  CO={np.mean(rmse_co):.4f}  NC={np.mean(rmse_nc):.4f}   "
            f"median  CO={median_co:.4f}  NC={median_nc:.4f}"
        )
        print(
            f"  median R2  CO={median_r2_co}  NC={median_r2_nc}  "
            "(Thm1 Regime A predicts R2_NC > R2_CO)"
        )
        return {
            "trials": used,
            "nc_wins": nc_wins,
            "co_wins": co_wins,
            "nc_win_rate": win_rate,
            "mean_rmse_co": float(np.mean(rmse_co)),
            "mean_rmse_nc": float(np.mean(rmse_nc)),
            "median_rmse_co": median_co,
            "median_rmse_nc": median_nc,
            "median_r2_co": median_r2_co,
            "median_r2_nc": median_r2_nc,
            "mean_r2_co": float(np.mean(r2_co)),
            "mean_r2_nc": float(np.mean(r2_nc)),
            "rmse_co_samples": rmse_co,
            "rmse_nc_samples": rmse_nc,
        }


class RobustnessSection(AnalysisSection):
    """Robustness: does the verdict survive design choices?

    Six independent challenges to the headline result: fixing the irreducible
    loss, dropping embedding-dominated model sizes, swapping the envelope
    reduction, a fair off-band extrapolation contest, the conditioning of
    randomly drawn 2D designs, and the significance of the win rate.
    """

    key = "robustness"
    title = "ROBUSTNESS CHECKS"

    def run(self, context: AnalysisContext) -> dict[str, object]:
        checks: dict[str, object] = {}
        checks["drop_small_N"] = self._drop_small_models(context)
        checks["median_envelope"] = self._median_envelope(context)
        checks["fair_co_vs_nc"] = self._fair_extrapolation(context)
        checks["chin_E_fixed_0p1"] = self._fixed_irreducible_loss(context)
        checks["random_2d_5pt_kappaAB"] = self._random_two_d_designs(context)
        checks["winrate_significance"] = self._winrate_significance(context)
        return {k: v for k, v in checks.items() if v is not None}

    def _fit_and_condition(
        self, arrays: FitArrays, seed: int, restarts: int = DEFAULT_RESTARTS
    ) -> Optional[tuple[LawFit, float]]:
        fit = LawFitter(restarts=restarts, global_search=True).fit(CHINCHILLA, arrays, seed=seed)
        if fit is None:
            return None
        kappa = GaussNewtonConditioning.from_fit(fit, arrays).kappa_scale_block()
        return fit, kappa

    def _drop_small_models(self, context: AnalysisContext) -> Optional[dict[str, object]]:
        """Refit above the embedding-dominated model sizes."""
        big = context.grid.envelope[context.grid.envelope["N"] >= SMALL_N_CUTOFF]
        outcome = self._fit_and_condition(
            TrainingGrid.arrays_of(big), context.config.seeds.primary_fit
        )
        if outcome is None:
            return None
        fit, kappa = outcome
        print(
            f"  drop 2 smallest N (n={len(big)}): eps={fit.exponent_gap:.4f}  kappa_AB={kappa:.3g}"
        )
        return {
            "n_cells": int(len(big)),
            "eps": fit.exponent_gap,
            "alpha": float(fit.params[2]),
            "beta": float(fit.params[4]),
            "kappa_AB": kappa,
        }

    def _median_envelope(self, context: AnalysisContext) -> Optional[dict[str, object]]:
        """Median-over-hyperparameters envelope instead of the min frontier."""
        frame = context.grid.median_envelope()
        arrays = FitArrays(
            n=frame["N"].to_numpy(dtype=float),
            d=frame["Dt"].to_numpy(dtype=float),
            loss=frame["L"].to_numpy(dtype=float),
        )
        outcome = self._fit_and_condition(arrays, context.config.seeds.primary_fit)
        if outcome is None:
            return None
        fit, kappa = outcome
        print(
            f"  median envelope: eps={fit.exponent_gap:.4f}  R2={fit.r2:.4f}  kappa_AB={kappa:.3g}"
        )
        return {
            "eps": fit.exponent_gap,
            "alpha": float(fit.params[2]),
            "beta": float(fit.params[4]),
            "r2": fit.r2,
            "kappa_AB": kappa,
        }

    def _fair_extrapolation(self, context: AnalysisContext) -> Optional[dict[str, object]]:
        """Both arms train off the held-out band and extrapolate onto it.

        Removes the objection that CO is disadvantaged by construction: neither
        design sees the target band.
        """
        grid = context.grid
        counts = grid.envelope["band"].value_counts().to_dict()
        eligible = [b for b in grid.populated_bands if counts.get(b, 0) >= 3]
        if not eligible:
            return None
        held_out = 20.0 if counts.get(20.0, 0) >= 3 else float(max(eligible))
        holdout = grid.band(held_out)
        co_subset = grid.band(CLEAN_RAY_BAND)
        nc_pool = grid.envelope[~grid.envelope["band"].isin([held_out, CLEAN_RAY_BAND])]
        if len(holdout) < 3 or len(co_subset) < 4 or nc_pool["band"].nunique() < 2:
            return None

        rng = np.random.RandomState(context.config.seeds.robustness_driver)
        holdout_design = Design.from_frame(holdout)
        co_design = Design.from_frame(co_subset)

        # Plan every trial first, so the driver RNG sequence is independent of
        # how many workers later execute them.
        trials = [
            ComparisonTrial(
                index=index,
                collinear=co_design,
                non_collinear=Design.from_frame(
                    nc_pool.sample(
                        n=min(len(co_subset), len(nc_pool)),
                        random_state=int(rng.randint(1e9)),
                    )
                ),
                holdout=holdout_design,
                collinear_seed=int(rng.randint(1e9)),
                non_collinear_seed=int(rng.randint(1e9)),
                restarts=TRIAL_RESTARTS,
            )
            for index in range(context.config.robustness_trials)
        ]
        outcomes = drop_failures(context.executor.map(run_comparison, trials))
        outcomes.sort(key=lambda o: o.index)
        wins = [1 if o.rmse_non_collinear < o.rmse_collinear else 0 for o in outcomes]
        if not wins:
            return None

        win_rate = float(np.mean(wins))
        bootstrap = [
            float(np.mean(rng.choice(wins, len(wins), replace=True)))
            for _ in range(context.config.bootstrap_resamples)
        ]
        low, high = float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))
        print(
            f"  fair CO-vs-NC (hold TPP~{held_out:.0f}): NC win-rate={win_rate:.3f}  "
            f"95%CI[{low:.3f},{high:.3f}]  n={len(wins)}"
        )
        return {
            "held_out_band": held_out,
            "co_band": CLEAN_RAY_BAND,
            "trials": len(wins),
            "nc_win_rate": win_rate,
            "ci95": [low, high],
        }

    def _fixed_irreducible_loss(self, context: AnalysisContext) -> Optional[dict[str, object]]:
        """Refit Chinchilla with E pinned, so E cannot absorb the exponent gap."""
        arrays = context.grid.arrays

        def loss_fn(n: np.ndarray, d: np.ndarray, a: float, alpha: float, b: float, beta: float):
            return CHINCHILLA.loss_fn(n, d, FIXED_IRREDUCIBLE_LOSS, a, alpha, b, beta)

        bounds = [(1e-2, 1e10), (0.01, 2.0), (1e-2, 1e10), (0.01, 2.0)]

        def sampler(rng: np.random.RandomState, _loss: np.ndarray) -> list[float]:
            return [
                10 ** rng.uniform(0, 8),
                rng.uniform(0.1, 1.5),
                10 ** rng.uniform(0, 8),
                rng.uniform(0.1, 1.5),
            ]

        params, _, _ = fit_generic(
            loss_fn,
            arrays.n,
            arrays.d,
            arrays.loss,
            bounds,
            sampler,
            RAY_RESTARTS,
            context.config.seeds.primary_fit,
        )
        if params is None:
            return None
        _, alpha, _, beta = (float(p) for p in params)
        print(f"  R1a E={FIXED_IRREDUCIBLE_LOSS} fixed: alpha={alpha:.3f} beta={beta:.3f} "
              f"eps={abs(alpha - beta):.3f}")
        return {"alpha": alpha, "beta": beta, "eps": abs(alpha - beta)}

    def _random_two_d_designs(self, context: AnalysisContext) -> Optional[dict[str, object]]:
        """Conditioning of randomly drawn genuine-2D 5-point designs.

        Guards against the reading that "2D fixes it": most random 2D designs
        remain ill-conditioned. Conditioning tracks TPP coverage, not
        two-dimensionality per se.
        """
        grid = context.grid
        rng = np.random.RandomState(context.config.seeds.robustness_driver)
        size = context.config.random_subset_size
        trials: list[ConditioningTrial] = []
        for index in range(context.config.random_subset_trials):
            subset = grid.envelope.iloc[rng.choice(len(grid.envelope), size, replace=False)]
            # The seed draw is deliberately inside the guard: a rejected subset
            # must not advance the RNG, or the planned sequence would shift.
            if subset["N"].nunique() < 2 or subset["band"].nunique() < 2:
                continue
            trials.append(
                ConditioningTrial(
                    index=index,
                    design=Design.from_frame(subset),
                    seed=int(rng.randint(1e9)),
                    restarts=SUBSET_RESTARTS,
                )
            )
        outcomes = drop_failures(context.executor.map(run_conditioning, trials))
        outcomes.sort(key=lambda o: o.index)
        kappas = [o.kappa_scale_block for o in outcomes]
        if not kappas:
            return None
        values = np.asarray(kappas)
        q25, q75 = float(np.percentile(values, 25)), float(np.percentile(values, 75))
        below = float(np.mean(values < KAPPA_TARGETS[0]))
        print(
            f"  R2 random 2D-{size}pt kappa_AB: median={np.median(values):.1f} "
            f"IQR=[{q25:.1f},{q75:.1f}] frac<1e4={below:.2f}"
        )
        return {
            "n": len(kappas),
            "median": float(np.median(values)),
            "q25": q25,
            "q75": q75,
            "frac_below_1e4": below,
        }

    def _winrate_significance(self, context: AnalysisContext) -> Optional[dict[str, object]]:
        """Is the NC win rate significantly above chance?"""
        holdout = context.holdout_summary
        if not holdout:
            return None
        wins, trials = int(holdout["nc_wins"]), int(holdout["trials"])
        p_value = float(stats.binomtest(wins, trials, 0.5, alternative="greater").pvalue)
        rate = wins / trials
        half_width = Z_95 * float(np.sqrt(rate * (1 - rate) / trials))
        print(
            f"  R3 win-rate {wins}/{trials}: binom p(>0.5)={p_value:.2e} "
            f"95%CI=[{rate - half_width:.3f},{rate + half_width:.3f}]"
        )
        return {
            "nc_wins": wins,
            "trials": trials,
            "p_binom_gt_half": p_value,
            "ci95": [rate - half_width, rate + half_width],
        }


class FormalValidationSection(AnalysisSection):
    """Direct checks of every formal statement in the companion paper.

    Prop. 1 and Cor. 1 are demonstrated by a *controlled exponent-gap sweep*
    rather than by reading the gap off the full-grid fit. This grid spans N over
    about 6200x, which by itself keeps the (A, B) block well-conditioned at
    fixed true exponents; plugging the full-grid exponents into the ray Jacobian
    therefore gives a misleadingly small, non-monotone kappa. The blow-up comes
    from the single-ray *fit* collapsing the two exponents, so the clean way to
    exhibit the Theta(eps^-2) rate is to hold alpha at its well-identified grid
    value and shrink the gap.
    """

    key = "formal_validation"
    title = "FORMAL-STATEMENT VALIDATION"

    #: Jacobian column of the scale coefficient A in the reduced (E, A, B) model.
    A_COLUMN = 1

    def run(self, context: AnalysisContext) -> dict[str, object]:
        results: dict[str, object] = {}
        results["prop1_cor1"] = self._exponent_gap_sweep(context)
        results["prop2_rmin"] = self._two_ray_threshold(context)
        results["isoflop"] = self._isoflop_invariance(context)
        return results

    def _exponent_gap_sweep(self, context: AnalysisContext) -> dict[str, object]:
        """Prop. 1 (kappa_AB ~ eps^-2) and Cor. 1 (CI ratio ~ eps^-1)."""
        grid = context.grid
        full_fit = context.require_chinchilla()
        ray = TrainingGrid.arrays_of(grid.band(CLEAN_RAY_BAND))
        alpha = float(full_fit.params[2])
        ray_ratio = float(np.median(ray.ratios))

        def scale_columns(gap: float) -> tuple[np.ndarray, np.ndarray]:
            beta = alpha - gap
            column_a = np.power(ray.n, -alpha)
            column_b = (ray_ratio**-beta) * np.power(ray.n, -beta)
            return column_a, column_b

        def kappa_at(gap: float) -> float:
            column_a, column_b = scale_columns(gap)
            return GaussNewtonConditioning(
                np.column_stack([column_a, column_b]), scale_columns=(0, 1)
            ).kappa_scale_block()

        def ci_ratio_at(gap: float) -> float:
            """CI(A) in the full model relative to CI(A) in the reduced model."""
            column_a, column_b = scale_columns(gap)
            ones = np.ones_like(ray.n)
            full = GaussNewtonConditioning(
                np.column_stack([ones, column_a, column_b]), scale_columns=(1, 2)
            )
            reduced = GaussNewtonConditioning(
                np.column_stack([ones, column_a]), scale_columns=(0, 1)
            )
            denominator = max(reduced.variance_factor(self.A_COLUMN), 1e-300)
            return float(np.sqrt(full.variance_factor(self.A_COLUMN) / denominator))

        gaps = np.logspace(
            np.log10(context.config.eps_sweep_min),
            np.log10(context.config.eps_sweep_max),
            context.config.eps_sweep_points,
        )
        kappas = np.array([kappa_at(g) for g in gaps])
        ci_ratios = np.array([ci_ratio_at(g) for g in gaps])
        kappa_slope = float(np.polyfit(np.log(gaps), np.log(kappas), 1)[0])
        ci_slope = float(np.polyfit(np.log(gaps), np.log(ci_ratios), 1)[0])

        # Reduced psi-model on the ray: identified, so its kappa stays O(1).
        reduced_jacobian = np.column_stack([np.ones_like(ray.n), np.power(ray.n, -alpha)])
        reduced_kappa = float(np.linalg.cond(reduced_jacobian.T @ reduced_jacobian))

        per_law = [
            {
                "law": law.name,
                "eps": gap,
                "eps_inv": 1.0 / gap,
                "eps_inv_sq": 1.0 / gap**2,
                "kappa_AB_model": kappa_at(gap),
                "ci_ratio_model": ci_ratio_at(gap),
            }
            for law, gap in self._fitted_gaps(context).items()
        ]

        print(
            f"  Prop1: kappa_AB ~ eps^{kappa_slope:.2f} (predict -2) ;  "
            f"Cor1: CI(A)/CI(psi) ~ eps^{ci_slope:.2f} (predict -1)"
        )
        print(
            f"  reduced-model kappa (ray)={reduced_kappa:.1f} O(1) [reduced_id] ; "
            "per-law model kappa_AB: "
            + ", ".join(
                f"{d['law'][:4]} eps={d['eps']:.3f}->{d['kappa_AB_model']:.0f}" for d in per_law
            )
        )
        return {
            "alpha_fixed": alpha,
            "ray_ratio": ray_ratio,
            "kappa_slope_loglog": kappa_slope,
            "ci_slope_loglog": ci_slope,
            "kappa_reduced_ray": reduced_kappa,
            "per_law": per_law,
            "eps_grid": [float(g) for g in gaps],
            "kappa_AB": [float(k) for k in kappas],
            "ci_ratio": [float(c) for c in ci_ratios],
        }

    @staticmethod
    def _fitted_gaps(context: AnalysisContext) -> dict:
        """Fitted exponent gap per law, for marking on the sweep curve.

        Only Chinchilla's blow-up is measured directly: the clean ray has five
        model sizes, too few to identify Kaplan (4 params, borderline) or
        Droppo-Elibol (6 params, impossible). All four laws share the same
        j2 = c j1 N^eps collinearity structure, so the shared leading-order
        eps^-2 curve is evaluated at each law's fitted gap instead.
        """
        gaps = {}
        fitter = LawFitter(restarts=DEFAULT_RESTARTS, global_search=True)
        for law in LAWS:
            fit = fitter.fit(law, context.grid.arrays, seed=context.config.seeds.primary_fit)
            if fit is not None and fit.exponent_gap > 0:
                gaps[law] = fit.exponent_gap
        return gaps

    def _two_ray_threshold(self, context: AnalysisContext) -> dict[str, object]:
        """Prop. 2: kappa <= target iff V_K >= tau_K; for K=2, R >= R_min."""
        grid = context.grid
        beta_eff = context.require_chinchilla().beta_eff
        criterion = DiversityCriterion(beta_eff)
        target = KAPPA_TARGETS[0]
        r_min = criterion.minimum_two_ray_spread(
            target, points=context.config.rmin_sweep_points
        )

        fitter = LawFitter(restarts=SUBSET_RESTARTS, global_search=False)
        bands = grid.populated_bands
        measured: list[dict[str, float]] = []
        for i, first in enumerate(bands):
            for second in bands[i + 1:]:
                left = grid.band(first).drop_duplicates("N")
                right = grid.band(second).drop_duplicates("N")
                if len(left) < 2 or len(right) < 2:
                    continue
                combined = pd.concat([left, right])
                arrays = TrainingGrid.arrays_of(combined)
                fit = fitter.fit(CHINCHILLA, arrays, seed=context.config.seeds.two_band_fit)
                if fit is None:
                    continue
                conditioning = GaussNewtonConditioning.from_fit(fit, arrays)
                measured.append(
                    {"R": float(second / first), "kappa_AB": conditioning.kappa_scale_block()}
                )

        print(
            f"  Prop2 R_min(kt={target:.0e},beta_eff={beta_eff:.3f})={r_min:.2f}; "
            f"{len(measured)} measured 2-band designs"
        )
        return {
            "beta_eff": beta_eff,
            "kappa_target": target,
            "R_min": r_min,
            "measured": sorted(measured, key=lambda d: d["R"]),
        }

    def _isoflop_invariance(self, context: AnalysisContext) -> dict[str, object]:
        """Def. 4: the induced isoFLOP reparameterisation leaves RMSE unchanged."""
        grid = context.grid
        ray_frame = grid.band(CLEAN_RAY_BAND)
        ray = TrainingGrid.arrays_of(ray_frame)
        holdout_frame = grid.envelope[~grid.envelope.index.isin(ray_frame.index)]
        holdout = TrainingGrid.arrays_of(holdout_frame)

        co_fit = LawFitter(restarts=RAY_RESTARTS, global_search=True).fit(
            CHINCHILLA, ray, seed=context.config.seeds.clean_ray_fit
        )
        if co_fit is None:
            raise RuntimeError("clean-ray Chinchilla fit failed; cannot build isoFLOP curves")
        nc_fit = context.require_chinchilla()

        # Reparameterise each holdout point through its own compute budget
        # C = 6 N D: substituting D = C / (6 N) must be an identity.
        compute = FLOPS_PER_PARAM_TOKEN * holdout.n * holdout.d
        reparameterised = FitArrays(
            n=holdout.n, d=compute / (FLOPS_PER_PARAM_TOKEN * holdout.n), loss=holdout.loss
        )
        invariance_error = float(
            np.max(np.abs(co_fit.predict(holdout) - co_fit.predict(reparameterised)))
        )
        co_rmse = co_fit.rmse_on(holdout)
        nc_rmse = nc_fit.rmse_on(holdout)
        print(
            f"  isoFLOP invariance max|err|={invariance_error:.2e} (Prop: ~0);  "
            f"off-ray holdout RMSE CO={co_rmse:.3f} NC={nc_rmse:.3f}"
        )
        return {
            "invariance_max_abs_err": invariance_error,
            "holdout_rmse_co": co_rmse,
            "holdout_rmse_nc": nc_rmse,
            "n_holdout": int(len(holdout)),
            "co_params": co_fit.named_params,
            "compute_budgets": [float(c) for c in compute],
        }
