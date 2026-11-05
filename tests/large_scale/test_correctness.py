"""Correctness of every computational method, against closed-form ground truth.

Where a quantity has an analytic value (a Jacobian of a known law, the condition
number of a constructed matrix, V_K of a design with one ratio) the test asserts
that value rather than a golden number captured from a previous run.
"""

from __future__ import annotations

import numpy as np
import pytest

from tpp_large_scale.model.conditioning import (
    GaussNewtonConditioning,
    chinchilla_jacobian,
    numerical_jacobian,
)
from tpp_large_scale.config import KAPPA_TARGETS, Regime
from tpp_large_scale.model.diversity import DiversityCriterion
from tpp_large_scale.model.fitting import (
    CHINCHILLA,
    DROPPO_ELIBOL,
    KAPLAN,
    LAWS,
    LawFitter,
    verify_upstream_parity,
)
from tpp_large_scale.model.grid import FitArrays, LogBandAssigner, TrainingGrid
from tpp_large_scale.upstream import chinchilla_loss, compute_tauK, compute_VK


class TestUpstreamReuse:
    """The point of the refactor: we compute what the paper's code computes."""

    def test_fits_match_the_papers_own_fitters(self, envelope_arrays) -> None:
        deltas = verify_upstream_parity(envelope_arrays)
        assert set(deltas) == {"Chinchilla", "Kaplan", "Droppo-Elibol"}
        assert all(d == 0.0 for d in deltas.values()), deltas

    def test_law_specs_wrap_the_upstream_loss_functions(self) -> None:
        assert CHINCHILLA.loss_fn is chinchilla_loss

    def test_diversity_delegates_to_upstream(self) -> None:
        ratios = [20.0, 67.0, 215.0]
        criterion = DiversityCriterion(beta_eff=0.3)
        assert criterion.v_k(ratios) == compute_VK(np.asarray(ratios), 0.3)
        assert criterion.tau_k(ratios, 1e4) == compute_tauK(np.asarray(ratios), 0.3, 1e4)


class TestLawSpecs:
    def test_chinchilla_gap_is_alpha_minus_beta(self) -> None:
        # params: E, A, alpha, B, beta
        assert CHINCHILLA.exponent_gap([1.0, 10.0, 0.35, 20.0, 0.28]) == pytest.approx(0.07)

    def test_kaplan_gap_is_alpha_d_minus_alpha_n(self) -> None:
        # params: N_c, D_c, alpha_N, alpha_D
        assert KAPLAN.exponent_gap([1e6, 1e6, 0.30, 0.34]) == pytest.approx(0.04)

    def test_droppo_gap_is_scaled_by_the_shared_exponent(self) -> None:
        # params: L_inf, N_c, D_c, alpha_N, alpha_D, alpha
        gap = DROPPO_ELIBOL.exponent_gap([0.5, 1e6, 1e6, 0.40, 0.30, 2.0])
        assert gap == pytest.approx(0.05)

    def test_zero_divisor_is_rejected_loudly(self) -> None:
        with pytest.raises(ZeroDivisionError):
            DROPPO_ELIBOL.exponent_gap([0.5, 1e6, 1e6, 0.4, 0.3, 0.0])

    def test_beta_index_selects_the_data_exponent(self) -> None:
        assert CHINCHILLA.beta_eff([1.0, 10.0, 0.35, 20.0, 0.28]) == pytest.approx(0.28)

    def test_every_law_has_a_distinct_name(self) -> None:
        assert len({law.name for law in LAWS}) == len(LAWS)


class TestFitting:
    def test_recovers_known_parameters_on_a_noiseless_surface(
        self, synthetic_chinchilla: FitArrays
    ) -> None:
        fit = LawFitter(restarts=200, global_search=True).fit(
            CHINCHILLA, synthetic_chinchilla, seed=0
        )
        assert fit is not None
        assert fit.r2 > 0.9999
        assert fit.exponent_gap == pytest.approx(0.07, abs=0.02)

    def test_rejects_designs_with_too_few_points(self) -> None:
        tiny = FitArrays(
            n=np.array([1e6, 1e7]), d=np.array([1e8, 1e9]), loss=np.array([3.0, 2.5])
        )
        with pytest.raises(ValueError, match="at least"):
            LawFitter(restarts=5).fit(CHINCHILLA, tiny, seed=0)

    def test_rejects_nonpositive_restarts(self) -> None:
        with pytest.raises(ValueError, match="restarts must be positive"):
            LawFitter(restarts=0)

    def test_rmse_on_matches_manual_computation(
        self, synthetic_chinchilla: FitArrays
    ) -> None:
        fit = LawFitter(restarts=30, global_search=False).fit(
            CHINCHILLA, synthetic_chinchilla, seed=3
        )
        assert fit is not None
        expected = float(
            np.sqrt(np.mean((synthetic_chinchilla.loss - fit.predict(synthetic_chinchilla)) ** 2))
        )
        assert fit.rmse_on(synthetic_chinchilla) == pytest.approx(expected)

    def test_named_params_line_up_with_param_names(
        self, synthetic_chinchilla: FitArrays
    ) -> None:
        fit = LawFitter(restarts=20, global_search=False).fit(
            CHINCHILLA, synthetic_chinchilla, seed=0
        )
        assert fit is not None
        assert tuple(fit.named_params) == CHINCHILLA.param_names


class TestJacobian:
    def test_analytic_matches_central_difference(self, envelope_arrays) -> None:
        params = [1.0, 100.0, 0.35, 2000.0, 0.28]
        analytic = chinchilla_jacobian(envelope_arrays, params)
        numeric = numerical_jacobian(CHINCHILLA, envelope_arrays, params)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-9)

    def test_column_order_is_E_A_alpha_B_beta(self) -> None:
        arrays = FitArrays(
            n=np.array([1e6, 1e7, 1e8]),
            d=np.array([1e9, 1e10, 1e11]),
            loss=np.zeros(3),
        )
        jacobian = chinchilla_jacobian(arrays, [1.0, 100.0, 0.35, 2000.0, 0.28])
        # dL/dE is identically 1.
        np.testing.assert_array_equal(jacobian[:, 0], np.ones(3))
        # dL/dA is N^-alpha.
        np.testing.assert_allclose(jacobian[:, 1], np.power(arrays.n, -0.35))
        # dL/dB is D^-beta.
        np.testing.assert_allclose(jacobian[:, 3], np.power(arrays.d, -0.28))


class TestConditioning:
    def test_identity_design_is_perfectly_conditioned(self) -> None:
        cond = GaussNewtonConditioning(np.eye(4)[:, :2], scale_columns=(0, 1))
        assert cond.kappa_scale_block() == pytest.approx(1.0)

    def test_known_condition_number(self) -> None:
        """Columns scaled by s give J^T J eigenvalues 1 and s^2."""
        jacobian = np.array([[1.0, 0.0], [0.0, 3.0]])
        cond = GaussNewtonConditioning(jacobian, scale_columns=(0, 1))
        assert cond.kappa_scale_block() == pytest.approx(9.0)

    def test_exactly_collinear_columns_are_infinitely_conditioned(self) -> None:
        column = np.array([1.0, 2.0, 3.0])
        jacobian = np.column_stack([column, column])
        assert GaussNewtonConditioning(jacobian, scale_columns=(0, 1)).kappa_scale_block() > 1e15

    def test_column_scaling_removes_unit_disparity(self) -> None:
        """Rescaling a column changes kappa_full but not the scaled variant."""
        base = np.column_stack([np.ones(5), np.linspace(1, 2, 5), np.linspace(3, 4, 5)])
        stretched = base.copy()
        stretched[:, 2] *= 1e6
        assert GaussNewtonConditioning(base, (1, 2)).kappa_full_scaled() == pytest.approx(
            GaussNewtonConditioning(stretched, (1, 2)).kappa_full_scaled()
        )
        assert GaussNewtonConditioning(base, (1, 2)).kappa_full() != pytest.approx(
            GaussNewtonConditioning(stretched, (1, 2)).kappa_full()
        )

    def test_variance_factor_matches_explicit_inverse(self) -> None:
        jacobian = np.column_stack([np.ones(6), np.linspace(1.0, 3.0, 6)])
        cond = GaussNewtonConditioning(jacobian, scale_columns=(0, 1))
        expected = np.linalg.inv(jacobian.T @ jacobian)[1, 1]
        assert cond.variance_factor(1) == pytest.approx(expected)

    def test_rejects_out_of_range_scale_columns(self) -> None:
        with pytest.raises(IndexError):
            GaussNewtonConditioning(np.ones((5, 2)), scale_columns=(0, 5))

    def test_rejects_one_dimensional_input(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            GaussNewtonConditioning(np.ones(5))

    def test_from_fit_refuses_laws_without_an_analytic_jacobian(
        self, synthetic_chinchilla
    ) -> None:
        fit = LawFitter(restarts=10, global_search=False).fit(
            KAPLAN, synthetic_chinchilla, seed=0
        )
        assert fit is not None
        with pytest.raises(NotImplementedError, match="no analytic Jacobian"):
            GaussNewtonConditioning.from_fit(fit, synthetic_chinchilla)


class TestDiversityCriterion:
    def test_single_ratio_has_zero_diversity(self) -> None:
        assert DiversityCriterion(0.3).v_k([50.0]) == pytest.approx(0.0)

    def test_single_ratio_is_regime_A(self) -> None:
        verdict = DiversityCriterion(0.3).verdict([50.0], 1e4)
        assert verdict.regime is Regime.ILL_CONDITIONED
        assert not verdict.well_conditioned

    def test_v_k_is_the_variance_of_k_to_the_minus_beta(self) -> None:
        ratios, beta = [20.0, 200.0], 0.25
        transformed = np.asarray(ratios) ** -beta
        expected = float(np.mean(transformed**2) - np.mean(transformed) ** 2)
        assert DiversityCriterion(beta).v_k(ratios) == pytest.approx(expected)

    def test_wider_spread_increases_diversity(self) -> None:
        criterion = DiversityCriterion(0.3)
        assert criterion.v_k([20.0, 40.0]) < criterion.v_k([20.0, 400.0])

    def test_tau_falls_as_the_target_loosens(self) -> None:
        criterion = DiversityCriterion(0.3)
        ratios = [20.0, 200.0]
        assert criterion.tau_k(ratios, 1e6) < criterion.tau_k(ratios, 1e4)

    def test_minimum_spread_is_the_first_clearing_spread(self) -> None:
        criterion = DiversityCriterion(0.0385)
        r_min = criterion.minimum_two_ray_spread(1e4)
        assert criterion.verdict([1.0, r_min], 1e4).well_conditioned
        assert not criterion.verdict([1.0, r_min * 0.9], 1e4).well_conditioned

    def test_empty_design_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty design"):
            DiversityCriterion(0.3).v_k([])

    def test_nonpositive_target_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            DiversityCriterion(0.3).tau_k([20.0, 200.0], 0.0)

    def test_sweep_covers_every_configured_target(self) -> None:
        verdicts = DiversityCriterion(0.3).sweep_targets([20.0, 200.0])
        assert len(verdicts) == len(KAPPA_TARGETS)


class TestBandAssignment:
    def test_snaps_in_log_space_not_linear_space(self) -> None:
        """TPP 90 is nearer 67 linearly but nearer 120 in log space."""
        assigner = LogBandAssigner((67.0, 120.0))
        assert abs(90 - 67) < abs(90 - 120)  # linear would pick 67
        assert assigner.assign(90.0) == 120.0

    def test_exact_band_values_map_to_themselves(self) -> None:
        assigner = LogBandAssigner()
        for band in assigner.bands:
            assert assigner.assign(band) == band

    def test_rejects_nonpositive_tpp(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            LogBandAssigner().assign(0.0)

    def test_rejects_nonpositive_bands(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            LogBandAssigner((0.0, 10.0))

    def test_rejects_empty_bands(self) -> None:
        with pytest.raises(ValueError, match="at least one band"):
            LogBandAssigner(())


class TestTrainingGrid:
    def test_envelope_takes_the_minimum_loss_per_cell(self, tiny_runs_frame) -> None:
        grid = TrainingGrid(tiny_runs_frame)
        # Each (N, D) cell has two LR values with losses 5.0 and 5.4.
        assert grid.envelope["L"].max() == pytest.approx(5.0)
        assert len(grid.envelope) == 4

    def test_missing_columns_are_reported(self, tiny_runs_frame) -> None:
        with pytest.raises(ValueError, match="missing columns"):
            TrainingGrid(tiny_runs_frame.drop(columns=["tokens"]))

    def test_unknown_architecture_is_reported(self, tiny_runs_frame) -> None:
        frame = tiny_runs_frame.copy()
        frame.loc[0, "d_model"] = 999
        with pytest.raises(KeyError, match="non-embedding parameter count"):
            TrainingGrid(frame)

    def test_all_unfinished_runs_is_an_error(self, tiny_runs_frame) -> None:
        frame = tiny_runs_frame.copy()
        frame["has_final"] = False
        with pytest.raises(ValueError, match="no finished runs"):
            TrainingGrid(frame)

    def test_real_grid_shape(self, grid: TrainingGrid) -> None:
        summary = grid.describe()
        assert summary["finished_runs"] == 461
        assert summary["envelope_cells"] == 29
        assert len(summary["N_nonembed_levels"]) == 5

    def test_ray_bands_require_enough_cells(self, grid: TrainingGrid) -> None:
        rays = grid.ray_bands()
        assert set(rays) == {37.0, 67.0, 120.0}
        assert all(len(sub) >= 4 for sub in rays.values())

    def test_clean_ray_has_one_cell_per_model_size(self, grid: TrainingGrid) -> None:
        ray = grid.band(67.0)
        assert len(ray) == len(grid.model_sizes) == 5

    def test_band_and_off_band_partition_the_envelope(self, grid: TrainingGrid) -> None:
        for band in grid.populated_bands:
            assert len(grid.band(band)) + len(grid.off_band(band)) == len(grid.envelope)

    def test_ratios_are_tokens_over_nonembedding_params(self, grid: TrainingGrid) -> None:
        arrays = grid.arrays
        np.testing.assert_allclose(arrays.ratios, arrays.d / arrays.n)

    def test_missing_csv_is_reported(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            TrainingGrid.from_csv(tmp_path / "nope.csv")
