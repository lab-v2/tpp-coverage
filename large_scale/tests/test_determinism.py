"""The determinism contract: results must not depend on worker count.

These are the tests that make the parallelisation safe to trust. They assert
the three properties the design rests on: workers do not change results, the
planner's RNG consumption is worker-independent, and every fit is bit-exactly
repeatable.
"""

from __future__ import annotations

import numpy as np
import pytest

from tpp_large_scale.config import TRIAL_RESTARTS, AnalysisConfig
from tpp_large_scale.model.fitting import CHINCHILLA, LawFitter
from tpp_large_scale.model.grid import TrainingGrid
from tpp_large_scale.experiments.parallel import (
    MIN_ITEMS_FOR_POOL,
    DeterministicExecutor,
    resolve_worker_count,
)
from tpp_large_scale.experiments.sections import (
    AnalysisContext,
    FullGridFitsSection,
    HoldoutSection,
    RobustnessSection,
)
from tpp_large_scale.experiments.trials import Design, run_comparison


def _square(value: int) -> int:
    """Module-level so it is picklable by the process pool."""
    return value * value


class TestExecutorOrdering:
    def test_preserves_input_order_regardless_of_workers(self) -> None:
        items = list(range(64))
        serial = DeterministicExecutor(workers=1).map(_square, items)
        parallel = DeterministicExecutor(workers=8).map(_square, items)
        assert serial == parallel == [i * i for i in items]

    def test_empty_input(self) -> None:
        assert DeterministicExecutor(workers=4).map(_square, []) == []

    def test_small_workloads_stay_serial(self) -> None:
        assert resolve_worker_count(64, MIN_ITEMS_FOR_POOL - 1) == 1

    def test_worker_count_clamped_to_item_count(self) -> None:
        assert resolve_worker_count(1000, 32) == 32

    def test_rejects_nonpositive_workers(self) -> None:
        with pytest.raises(ValueError, match="workers must be"):
            resolve_worker_count(0, 100)


class TestFitReproducibility:
    def test_same_seed_gives_bit_identical_fit(self, envelope_arrays) -> None:
        fitter = LawFitter(restarts=40, global_search=False)
        first = fitter.fit(CHINCHILLA, envelope_arrays, seed=123)
        second = fitter.fit(CHINCHILLA, envelope_arrays, seed=123)
        assert first is not None and second is not None
        np.testing.assert_array_equal(first.params, second.params)
        assert first.sse == second.sse

    def test_different_seeds_explore_differently(self, envelope_arrays) -> None:
        fitter = LawFitter(restarts=5, global_search=False)
        a = fitter.fit(CHINCHILLA, envelope_arrays, seed=1)
        b = fitter.fit(CHINCHILLA, envelope_arrays, seed=2)
        assert a is not None and b is not None
        assert a.sse != b.sse or not np.array_equal(a.params, b.params)

    def test_trial_worker_is_pure(self, grid: TrainingGrid) -> None:
        """Running the same work item twice must give the same outcome."""
        ray = Design.from_frame(grid.band(67.0))
        other = Design.from_frame(grid.band(120.0))
        from tpp_large_scale.experiments.trials import ComparisonTrial

        trial = ComparisonTrial(
            index=0,
            collinear=ray,
            non_collinear=other,
            holdout=Design.from_frame(grid.band(37.0)),
            collinear_seed=7,
            non_collinear_seed=11,
            restarts=TRIAL_RESTARTS,
        )
        first, second = run_comparison(trial), run_comparison(trial)
        assert first is not None and second is not None
        assert first == second


class TestPlanIsWorkerIndependent:
    def test_holdout_plan_identical_across_executors(self, grid, config) -> None:
        section = HoldoutSection()
        plans = []
        for workers in (1, 16):
            context = AnalysisContext(
                config=config, grid=grid, executor=DeterministicExecutor(workers)
            )
            plans.append(section._plan(context))
        assert plans[0] == plans[1]

    def test_holdout_plan_is_repeatable(self, grid, config) -> None:
        section = HoldoutSection()
        context = AnalysisContext(config=config, grid=grid)
        assert section._plan(context) == section._plan(context)

    def test_plan_seeds_are_all_distinct(self, grid, config) -> None:
        """A repeated seed would silently correlate trials."""
        trials = HoldoutSection()._plan(AnalysisContext(config=config, grid=grid))
        seeds = [t.collinear_seed for t in trials] + [t.non_collinear_seed for t in trials]
        assert len(set(seeds)) == len(seeds)


@pytest.fixture(scope="module")
def small_config(config: AnalysisConfig) -> AnalysisConfig:
    """A reduced workload for the worker-invariance tests.

    Worker-invariance is a structural property of the plan/execute/reduce
    split, so it does not need the full trial counts to be exercised. Keeping
    these tests to a workload that actually gets run is worth more than
    proving the same property on 400 trials instead of 40.
    """
    return AnalysisConfig(
        holdout_trials=40,
        robustness_trials=40,
        random_subset_trials=48,
        bootstrap_resamples=100,
    )


@pytest.mark.slow
class TestSectionsAreWorkerInvariant:
    """The headline guarantee, exercised end to end on the real grid."""

    @staticmethod
    def _context(grid, cfg, workers: int) -> AnalysisContext:
        return AnalysisContext(
            config=cfg, grid=grid, executor=DeterministicExecutor(workers)
        )

    def test_holdout_section_identical_serial_vs_parallel(self, grid, small_config) -> None:
        serial = HoldoutSection().run(self._context(grid, small_config, 1))
        parallel = HoldoutSection().run(self._context(grid, small_config, 16))
        assert serial == parallel

    def test_robustness_section_identical_serial_vs_parallel(self, grid, small_config) -> None:
        outputs = []
        for workers in (1, 16):
            context = self._context(grid, small_config, workers)
            FullGridFitsSection().run(context)
            context.holdout_summary = HoldoutSection().run(context)
            outputs.append(RobustnessSection().run(context))
        serial, parallel = outputs
        assert serial["fair_co_vs_nc"] == parallel["fair_co_vs_nc"]
        assert serial["random_2d_5pt_kappaAB"] == parallel["random_2d_5pt_kappaAB"]
        assert serial["winrate_significance"] == parallel["winrate_significance"]


class TestThreadPinning:
    def test_blas_capped_at_import(self) -> None:
        """Multi-threaded BLAS would make reductions order-dependent."""
        import os

        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            assert os.environ.get(var) == "1", f"{var} not pinned"
