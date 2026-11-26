"""Characterization of `run_analysis.py`'s paper tables, fast: built from the released fits.

`build_paper_tables` and `build_holdout_split_winrate` run on the released per-seed table
(`full_coverage_all_seeds.parquet`, the file the paper gate `test_paper_numbers.py` refits) and
must write the released tables byte for byte. The paper gate checks the same files after a full
refit and runs alone (`-m paper`); this check takes under a second and runs in the default suite.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pandas as pd
import pytest

from characterize import released

pytestmark = pytest.mark.network
SUB = "full_coverage_results"
PAPER_TABLES = [
    "summary_table.csv",
    "table_summary_full.tex",
    "table_winrate.tex",
    "winrate_per_combo.csv",
]
SPLIT_TABLES = [
    "table_winrate_by_holdout_split_appendix.tex",
    "winrate_by_holdout_split_appendix.csv",
]


@pytest.fixture(scope="module")
def release() -> Path:
    names = [*PAPER_TABLES, *SPLIT_TABLES, "full_coverage_all_seeds.parquet"]
    return released(*[f"{SUB}/{n}" for n in names]) / SUB


@pytest.fixture(scope="module")
def seeds(release: Path) -> pd.DataFrame:
    df = pd.read_parquet(release / "full_coverage_all_seeds.parquet")
    assert len(df) == 3000 and set(df["design"]) == {"CO", "NC"}
    return df


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def test_build_paper_tables_writes_the_released_tables(
    seeds: pd.DataFrame, release: Path, tmp_path: Path
) -> None:
    import run_analysis as ra

    summary, wins = _quiet(ra.build_paper_tables, seeds, str(tmp_path))
    for name in PAPER_TABLES + SPLIT_TABLES:
        assert (tmp_path / name).read_bytes() == (release / name).read_bytes(), name
    assert len(summary) == 8 and list(summary["Design"]) == ["CO", "NC"] * 4
    assert wins is not None and int(wins["N_paired"].sum()) > 0


def test_build_holdout_split_winrate_alone(
    seeds: pd.DataFrame, release: Path, tmp_path: Path
) -> None:
    import run_analysis as ra

    _quiet(ra.build_holdout_split_winrate, seeds, str(tmp_path))
    for name in SPLIT_TABLES:
        assert (tmp_path / name).read_bytes() == (release / name).read_bytes(), name

    # Without the per-dataset block: exactly the released table's first (aggregate) rows.
    _quiet(ra.build_holdout_split_winrate, seeds, str(tmp_path), "top", False)
    top = pd.read_csv(tmp_path / "winrate_by_holdout_split_top.csv")
    full = pd.read_csv(release / "winrate_by_holdout_split_appendix.csv")
    assert list(top["Holdout_Split"]) == ["unified", "colinear", "noncolinear"]
    pd.testing.assert_frame_equal(top, full.head(3))
