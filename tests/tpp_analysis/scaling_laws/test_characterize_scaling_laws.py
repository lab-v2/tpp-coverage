"""Characterization of `scaling_laws`: the fit pipeline, the LaTeX tables and every figure.

Inputs are the released loss tables and the released full-coverage fits (seed 0); expected values
are the base commit's outputs, frozen in `frozen/` (see `tests/characterize.py`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pathlib import Path

import pandas as pd
import pytest

from characterize import FigureCapture, check_frozen, released, released_fits

if TYPE_CHECKING:
    from scaling_laws.data import PreparedData

pytestmark = pytest.mark.network
FROZEN = Path(__file__).parent / "frozen"


@pytest.fixture(scope="module")
def losses() -> Path:
    return released("extracted_losses/*.csv") / "extracted_losses"


@pytest.fixture(scope="module")
def fits() -> list[dict[str, object]]:
    root = released("full_coverage_results/full_coverage_all_seeds.parquet")
    return released_fits(
        pd.read_parquet(root / "full_coverage_results/full_coverage_all_seeds.parquet")
    )


@pytest.fixture(scope="module")
def wiki(losses: Path) -> PreparedData:
    from scaling_laws.data import load_and_prepare

    return load_and_prepare(str(losses / "wikipedia.csv"), "Wikipedia", verbose=False)


@pytest.mark.slow
def test_run_full_analysis(losses: Path) -> None:
    """Every law, design and epoch mode fitted on the Wikipedia table (2 restarts each)."""
    from scaling_laws import run_full_analysis

    results = run_full_analysis(str(losses / "wikipedia.csv"), "Wikipedia", n_restarts=2)
    check_frozen(FROZEN / "run_full_analysis_wikipedia.json", results)


def test_single_epoch_fits_on_filtered_data(losses: Path) -> None:
    """Final mode only, on the lowest-loss half of the C4 training runs."""
    from scaling_laws import load_and_prepare, run_dataset_fits

    data = load_and_prepare(str(losses / "c4.csv"), "C4", train_keep=0.5, holdout_keep=0.8)
    results = run_dataset_fits(data, n_restarts=2, multi_epoch=False)
    sizes = {k: len(getattr(data, k)) for k in ("colinear_train", "noncolinear_train")}
    check_frozen(
        FROZEN / "single_epoch_fits_c4.json",
        {"raw_counts": data.raw_counts, "sizes": sizes, "results": results},
    )


def test_tables(fits: list[dict[str, object]]) -> None:
    from scaling_laws import tables

    check_frozen(
        FROZEN / "tables.json",
        {
            "summary": tables.build_summary_df(fits),
            "winners": tables.get_winners_summary(fits),
            "comparison": tables.display_comparison_tables(fits),
            "comparison_one": tables.display_comparison_tables(fits, dataset_filter="C4"),
            "comparison_none": tables.display_comparison_tables(fits, dataset_filter="None"),
            "aggregated": tables.display_aggregated_stats(fits),
            "aggregated_50": tables.display_aggregated_stats(fits, win_threshold=50.0),
        },
    )


def test_exploration_figures(losses: Path, wiki: PreparedData, tmp_path: Path) -> None:
    from scaling_laws import visualization as viz

    csv = str(losses / "wikipedia.csv")
    with FigureCapture() as cap:
        viz.plot_3d_overview(csv, title="Wikipedia", save_dir=str(tmp_path))
        viz.plot_overlay(csv, title="Wikipedia")
        viz.plot_distributions({"Wikipedia": wiki})
    assert (tmp_path / "colinear_3d.png").exists()
    check_frozen(FROZEN / "exploration_figures.json", cap.figures)


def test_surface_figures(fits: list[dict[str, object]], wiki: PreparedData) -> None:
    from scaling_laws import visualization as viz

    with FigureCapture() as cap:
        viz.plot_all_surfaces(fits, wiki, dataset_filter="Wikipedia", epoch_law_epochs=[1, 3])
    check_frozen(FROZEN / "surface_figures.json", cap.figures)


def test_isoflop_figures(fits: list[dict[str, object]], wiki: PreparedData) -> None:
    from scaling_laws import visualization as viz

    with FigureCapture() as cap:
        for holdout in ("unified", "colinear", "noncolinear"):
            viz.plot_isoflop_single(
                fits, wiki, holdout_type=holdout, dataset_filter="Wikipedia", mode_filter="Final"
            )
        viz.plot_isoflops([f for f in fits if f["dataset"] == "Wikipedia"], wiki)
    check_frozen(FROZEN / "isoflop_figures.json", cap.figures)


def test_dual_blanket_figures(fits: list[dict[str, object]], wiki: PreparedData) -> None:
    from scaling_laws import visualization as viz

    with FigureCapture() as cap:
        for law in ("Chinchilla", "Kaplan", "Droppo-Elibol"):
            viz.plot_dual_blanket_from_results(fits, wiki, law_name=law, dataset_filter="Wikipedia")
        viz.plot_dual_blanket_from_results(
            fits,
            wiki,
            mode_filter="First",
            dataset_filter="Wikipedia",
            holdout_keys=["NC Holdout"],
            show_gap_bars=False,
        )
    check_frozen(FROZEN / "dual_blanket_figures.json", cap.figures)
