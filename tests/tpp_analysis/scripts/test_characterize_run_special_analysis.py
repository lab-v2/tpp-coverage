"""Characterization of `run_special_analysis.py`: the BF16 fits and the tables built from them.

The tables are built from the released BF16 fits (`special_results/full_coverage_bf16.parquet`);
the base commit writes them byte-identical to the released tables, so the test checks both. The
script also runs end to end on the released BF16 loss table (2 seeds of 1 restart). Expected
values are the base commit's outputs, frozen in `frozen/run_special_analysis/` (see
`tests/characterize.py`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from characterize import check_frozen, released
from conftest import ANALYSIS, pool_width, run

pytestmark = pytest.mark.network
FROZEN = Path(__file__).parent / "frozen" / "run_special_analysis"
TABLES = [
    "summary_bf16.csv",
    "table_summary_bf16.tex",
    "winrate_bf16.csv",
    "table_winrate_bf16.tex",
    "winrate_by_holdout_split_bf16.csv",
    "table_winrate_by_holdout_split_bf16.tex",
]


def test_tables(tmp_path):
    import run_special_analysis as rs

    root = released(
        "special_results/*.csv",
        "special_results/*.tex",
        "special_results/full_coverage_bf16.parquet",
    )
    special = root / "special_results"
    df = pd.read_parquet(special / "full_coverage_bf16.parquet")
    rs.build_tables(df, str(tmp_path), "bf16")
    rs.build_holdout_split_winrate(df, str(tmp_path), "bf16")
    written = {name: (tmp_path / name).read_text() for name in TABLES}
    assert written == {name: (special / name).read_text() for name in TABLES}
    check_frozen(FROZEN / "tables.json", written)


@pytest.mark.slow
def test_script(tmp_path):
    root = released("extracted_losses/wikipedia_bf16.csv")
    out = tmp_path / "special"
    args = [
        "--data-dir",
        root / "extracted_losses",
        "-o",
        out,
        "--n-seeds",
        "2",
        "--n-restarts",
        "1",
    ]
    run(["python", ANALYSIS / "run_special_analysis.py", *args, "--n-workers", str(pool_width(13))])
    df = pd.read_parquet(out / "full_coverage_bf16.parquet")
    df = df.sort_values(["law", "mode", "design", "seed"]).reset_index(drop=True)
    tables = {name: (out / name).read_text() for name in TABLES}
    # A rerun reads every combo from its cache and writes the same tables.
    run(["python", ANALYSIS / "run_special_analysis.py", *args, "--n-workers", "2"])
    assert {name: (out / name).read_text() for name in TABLES} == tables
    check_frozen(FROZEN / "script.json", {"fits": df, "tables": tables})
