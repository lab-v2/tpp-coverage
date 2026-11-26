"""Characterization of `full_coverage_ci.py`: the seeded full-coverage fits and every table.

The tables are built from the released per-seed fits (`full_coverage_all_seeds.parquet`); the fits
run on the released Wikipedia loss table with 2 seeds of 1 restart. Expected values are the base
commit's outputs, frozen in `frozen/full_coverage_ci/` (see
`tests/characterize.py`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from characterize import check_frozen, released

pytestmark = pytest.mark.network
FROZEN = Path(__file__).parent / "frozen" / "full_coverage_ci"


@pytest.fixture(scope="module")
def seeds() -> pd.DataFrame:
    root = released("full_coverage_results/full_coverage_all_seeds.parquet")
    return pd.read_parquet(root / "full_coverage_results/full_coverage_all_seeds.parquet")


def test_tables(seeds):
    import full_coverage_ci as fc

    summary = fc.build_summary_table(seeds)
    per_combo_wins, winrate = fc.build_winrate_table(seeds)
    check_frozen(
        FROZEN / "tables.json",
        {
            "summary": summary,
            "per_combo": fc.build_per_combo_table(seeds),
            "winrate_per_combo": per_combo_wins,
            "winrate": winrate,
            "param_ci": fc.build_param_ci_table(seeds),
            "param_ci_summary": fc.build_param_ci_summary(seeds),
            "summary_tex": fc.summary_to_latex(summary),
            "winrate_tex": fc.winrate_to_latex(winrate),
            "param_ci_tex": fc.param_ci_to_latex(seeds),
            "param_ci_full_tex": fc.param_ci_to_latex(seeds, per_combo=True),
        },
    )


def test_seeded_fits_and_cache(tmp_path):
    """Two seeds per (law, mode, design) on Wikipedia; a second call reads the cached parquets."""
    import full_coverage_ci as fc
    from scaling_laws.data import load_and_prepare

    root = released("extracted_losses/wikipedia.csv")
    data = {
        "Wikipedia": load_and_prepare(
            str(root / "extracted_losses/wikipedia.csv"), "Wikipedia", verbose=False
        )
    }
    kwargs = dict(
        laws=["Kaplan", "Chinchilla", "Droppo-Elibol"], n_seeds=2, n_restarts=1, base_seed=5
    )
    fresh = fc.run_all_full_coverage(data, modes=["final"], save_dir=str(tmp_path), **kwargs)
    cached = fc.run_all_full_coverage(data, modes=["final"], save_dir=str(tmp_path), **kwargs)
    pd.testing.assert_frame_equal(fresh, cached)
    assert len(list(tmp_path.glob("pci_fullcov_*.parquet"))) == 6
    check_frozen(FROZEN / "fits.json", fresh)
