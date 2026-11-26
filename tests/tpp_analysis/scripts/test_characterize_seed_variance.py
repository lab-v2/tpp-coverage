"""Characterization of `seed_variance.py`: the per-step seeded fits and the convergence plot.

The fits run on the released Wikipedia loss table (2 seeds of 1 restart per step); the plot also
draws the released seed-variance results. Expected values are the base commit's outputs, frozen
in `frozen/seed_variance/` (see `tests/characterize.py`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from characterize import FigureCapture, check_frozen, released

pytestmark = pytest.mark.network
FROZEN = Path(__file__).parent / "frozen" / "seed_variance"


@pytest.fixture(scope="module")
def wiki():
    from scaling_laws.data import load_and_prepare

    root = released("extracted_losses/wikipedia.csv")
    return load_and_prepare(
        str(root / "extracted_losses/wikipedia.csv"), "Wikipedia", verbose=False
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "law, mode", [("Kaplan", "final"), ("Chinchilla", "first"), ("Droppo-Elibol", "second")]
)
def test_convergence(wiki, law, mode):
    import seed_variance as sv

    df = sv.run_seed_variance_convergence(wiki, law_name=law, mode=mode, n_seeds=2, n_restarts=1)
    check_frozen(FROZEN / f"{law}_{mode}.json", df)


def test_run_and_plot(wiki, tmp_path):
    import seed_variance as sv

    root = released("seed_variance_results/seed_variance_C4_*.parquet")
    c4 = pd.read_parquet(root / "seed_variance_results/seed_variance_C4_Chinchilla_final.parquet")
    with FigureCapture() as cap:
        df = sv.run_and_plot_seed_variance(
            wiki,
            "Wikipedia",
            law_name="Kaplan",
            n_seeds=2,
            n_restarts=1,
            save_dir=str(tmp_path),
            min_coverage=0.2,
        )
        for split, band in (
            ("unified", "percentile"),
            ("noncolinear", "std"),
            ("colinear", "percentile"),
        ):
            sv.plot_convergence_with_errorbars(
                c4, holdout_split=split, band_type=band, y_min=None, title="C4"
            )
    saved = pd.read_parquet(tmp_path / "seed_variance_Wikipedia_Kaplan_final.parquet")
    pd.testing.assert_frame_equal(saved, df)
    assert (tmp_path / "seed_variance_Wikipedia_Kaplan_final.pdf").exists()
    check_frozen(FROZEN / "plots.json", cap.figures)
