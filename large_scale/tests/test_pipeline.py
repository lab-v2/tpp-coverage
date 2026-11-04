"""End-to-end behaviour: upstream bootstrap, persistence, CLI."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tpp_large_scale import upstream
from tpp_large_scale.cli import build_parser, main
from tpp_large_scale.config import AnalysisConfig, SeedBook
from tpp_large_scale.runner import write_json_atomically


class TestUpstreamBootstrap:
    def test_locates_the_papers_package(self) -> None:
        package = upstream.locate_upstream_package()
        assert package.is_dir()
        assert (package / "scaling_laws" / "laws.py").is_file()

    def test_defaults_to_this_repositorys_code_and_records_it_relative(self) -> None:
        assert upstream.package_dir_label() == "src/tpp_analysis"
        assert upstream.package_dir_label(upstream.Path("/elsewhere/src/tpp_analysis")) == \
            "/elsewhere/src/tpp_analysis"

    def test_missing_checkout_raises_actionable_error(self, monkeypatch) -> None:
        monkeypatch.setenv(upstream.CODE_REPO_ENV, "/nonexistent/path/xyz")
        with pytest.raises(upstream.UpstreamNotFound, match="TPP_CODE_REPO"):
            upstream.locate_upstream_package()

    def test_unused_upstream_module_stays_excluded(self) -> None:
        """full_coverage_ci is not part of this analysis; nothing re-exports it."""
        assert "full_coverage_ci" not in upstream.__all__

    def test_linear_snapper_stays_excluded(self) -> None:
        assert "snap_to_nearest" not in upstream.__all__

    def test_reexports_are_all_importable(self) -> None:
        for name in upstream.__all__:
            assert hasattr(upstream, name), name

    def test_split_bucketer_is_reused_from_the_package(self) -> None:
        assert "organise_runs" in upstream.__all__
        assert "PreparedData" in upstream.__all__


class TestConfig:
    def test_missing_data_file_is_rejected(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="grid metrics CSV"):
            AnalysisConfig(metrics_csv=tmp_path / "absent.csv")

    def test_nonpositive_trial_count_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="trial counts must be positive"):
            AnalysisConfig(holdout_trials=0)

    def test_seedbook_is_frozen(self) -> None:
        seeds = SeedBook()
        with pytest.raises(Exception):
            seeds.primary_fit = 5  # type: ignore[misc]

    def test_output_dirs_are_created(self, tmp_path, config) -> None:
        cfg = AnalysisConfig(
            results_dir=tmp_path / "r", figures_dir=tmp_path / "f",
            metrics_csv=config.metrics_csv,
        )
        cfg.ensure_output_dirs()
        assert cfg.results_dir.is_dir() and cfg.figures_dir.is_dir()


class TestAtomicWrite:
    def test_writes_and_leaves_no_temp_file(self, tmp_path) -> None:
        target = tmp_path / "out.json"
        write_json_atomically(target, {"a": 1})
        assert json.loads(target.read_text()) == {"a": 1}
        assert not list(tmp_path.glob("*.tmp"))

    def test_overwrites_existing(self, tmp_path) -> None:
        target = tmp_path / "out.json"
        write_json_atomically(target, {"a": 1})
        write_json_atomically(target, {"b": 2})
        assert json.loads(target.read_text()) == {"b": 2}


class TestCli:
    def test_parser_defaults(self) -> None:
        args = build_parser().parse_args([])
        assert args.workers is None
        assert not args.no_figures

    def test_workers_flag_parsed(self) -> None:
        assert build_parser().parse_args(["--workers", "32"]).workers == 32

    def test_run_script_answers_help(self) -> None:
        import subprocess
        import sys

        from tpp_large_scale.config import REPO_ROOT

        out = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "run_analysis.py"),
                              "--help"], capture_output=True, text=True, check=True).stdout
        assert "--check-only" in out and "--workers" in out

    def test_missing_data_file_exits_nonzero(self, tmp_path) -> None:
        assert main(["--data", str(tmp_path / "absent.csv")]) == 2

    @pytest.mark.slow
    def test_check_only_verifies_parity(self, capsys) -> None:
        assert main(["--check-only"]) == 0
        assert "PARITY OK" in capsys.readouterr().out


class TestResultsDocument:
    """Guard the shape the report depends on."""

    @pytest.fixture(scope="class")
    def results(self) -> dict:
        from tpp_large_scale.config import REPO_ROOT

        return json.loads((REPO_ROOT / "results" / "results.json").read_text())

    def test_top_level_keys(self, results) -> None:
        expected = {
            "grid", "upstream_parity", "full_grid_fits", "full_grid_tpp", "co_rays",
            "equal_n_control", "co_vs_nc_holdout", "robustness", "formal_validation",
        }
        assert expected <= set(results)

    def test_all_formal_validation_blocks_present(self, results) -> None:
        """A silently skipped block would leave the report quoting stale numbers."""
        assert set(results["formal_validation"]) == {"prop1_cor1", "prop2_rmin", "isoflop"}

    def test_all_robustness_checks_present(self, results) -> None:
        expected = {
            "drop_small_N", "median_envelope", "fair_co_vs_nc", "chin_E_fixed_0p1",
            "random_2d_5pt_kappaAB", "winrate_significance",
        }
        assert set(results["robustness"]) == expected

    def test_upstream_parity_is_exact(self, results) -> None:
        assert all(d == 0.0 for d in results["upstream_parity"]["sse_delta_by_law"].values())

    def test_formal_statements_match_their_predictions(self, results) -> None:
        prop1 = results["formal_validation"]["prop1_cor1"]
        assert prop1["kappa_slope_loglog"] == pytest.approx(-2.0, abs=0.1)
        assert prop1["ci_slope_loglog"] == pytest.approx(-1.0, abs=0.1)
        assert results["formal_validation"]["isoflop"]["invariance_max_abs_err"] == 0.0

    def test_regime_verdicts_are_wellformed(self, results) -> None:
        for verdict in results["full_grid_tpp"]["by_target"].values():
            assert verdict["regime"] in {"A", "B"}
            assert np.isfinite(verdict["tau_K"])

    def test_collinear_ray_is_far_worse_than_the_full_grid(self, results) -> None:
        """The paper's central empirical claim, as an executable assertion."""
        full = results["full_grid_fits"]["Chinchilla"]["kappa_AB"]
        ray = results["co_rays"]["67"]["kappa_AB"]
        assert ray / full > 1e4

    def test_equal_size_control_isolates_geometry_from_sample_size(self, results) -> None:
        control = results["equal_n_control"]
        assert control["n_collinear"] == control["n_2d"]
        assert control["collinear_ray_kappa_AB"] / control["twod_spread_kappa_AB"] > 1e4

    def test_non_collinear_wins_the_holdout(self, results) -> None:
        holdout = results["co_vs_nc_holdout"]
        assert holdout["nc_win_rate"] > 0.5
        assert holdout["median_rmse_nc"] < holdout["median_rmse_co"]
        assert holdout["median_r2_nc"] > holdout["median_r2_co"]

    def test_win_rate_is_significant(self, results) -> None:
        significance = results["robustness"]["winrate_significance"]
        assert significance["p_binom_gt_half"] < 0.05
        assert significance["ci95"][0] > 0.5


class TestFrontierSplit:
    """The train/holdout split is bucketed by the paper's own organise_runs."""

    def test_buckets_partition_the_envelope(self, grid) -> None:
        from tpp_large_scale.model.splits import frontier_split

        split = frontier_split(grid)
        counts = split.counts()
        assert sum(counts.values()) == len(grid.envelope) == 29
        assert len(split.train) + len(split.holdout) == 29

    def test_holdout_is_the_largest_model_frontier(self, grid) -> None:
        from tpp_large_scale.model.splits import frontier_split

        split = frontier_split(grid)
        assert (split.holdout["N"] == grid.envelope["N"].max()).all()
        assert split.train["N"].max() < grid.envelope["N"].max()

    def test_bucketing_matches_the_package(self, grid) -> None:
        """counts() comes straight from upstream organise_runs, not our code."""
        from tpp_large_scale.model.splits import frontier_split
        from tpp_large_scale.upstream import organise_runs

        split = frontier_split(grid)
        assert set(split.counts()) == {
            "colinear_train", "colinear_holdout",
            "noncolinear_train", "noncolinear_holdout",
        }
        assert organise_runs([]) == {
            "colinear_train": [], "colinear_holdout": [],
            "noncolinear_train": [], "noncolinear_holdout": [],
        }
