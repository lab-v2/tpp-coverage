"""The paper's main analysis, refit from scratch, reproduces the released results.

The regression gate for any change to the analysis code: `run_analysis.py full_coverage` and
`seed_variance` refit every law on the released loss tables (30 seeds, 100 restarts), and
`generate_appendix_tables.py` writes the appendix tables. Against the released outputs on
HuggingFace:

- every per-combo fit (95 parquet files) is bit-identical;
- the two master files hold the same rows (their row order follows the order the fits finished
  in the released run, so rows are compared sorted);
- the main tables and CSVs are byte-identical;
- the appendix tables hold the same lines (their dataset blocks follow the master file's order).

About 6 minutes on 480 threads; it uses every thread of the affinity mask.
"""

from __future__ import annotations

import os
import shutil

import pandas as pd
import pytest

from conftest import ANALYSIS, run

pytestmark = [pytest.mark.network, pytest.mark.slow]

TABLES = ["summary_table.csv", "table_summary_full.tex", "table_winrate.tex",
          "table_winrate_by_holdout_split_appendix.tex", "winrate_by_holdout_split_appendix.csv",
          "winrate_per_combo.csv", "appendix_fitted_params.tex"]
SAME_LINES = ["appendix_full_metrics.tex"]
MASTERS = {"full_coverage_results/full_coverage_all_seeds.parquet": ["dataset", "law", "mode", "design", "seed"],
           "seed_variance_results/seed_variance_all.parquet": None}


@pytest.fixture(scope="module")
def refit(tmp_path_factory):
    """(released dir, refit dir): the released results, and a from-scratch refit of them."""
    root = tmp_path_factory.mktemp("paper")
    released = root / "released"
    run(["python", ANALYSIS / "download_and_extract.py", "--results-only",
         "--repo-dir", root / "hf", "--results-dest", released])
    losses = released / "extracted_losses"
    out = root / "refit"
    workers = str(len(os.sched_getaffinity(0)))
    for mode, sub in (("full_coverage", "full_coverage_results"), ("seed_variance", "seed_variance_results")):
        run(["python", ANALYSIS / "run_analysis.py", mode, "--data-dir", losses, "-o", out / sub,
             "--n-workers", workers], timeout=3600)
    run(["python", ANALYSIS / "generate_appendix_tables.py", out / "full_coverage_results"])
    yield released, out
    shutil.rmtree(root, ignore_errors=True)


def per_combo(released):
    """The 50 full-coverage and 45 seed-variance per-combo files of the release."""
    fc = (released / "full_coverage_results").glob("fullcov_*.parquet")
    sv = (released / "seed_variance_results").glob("seed_variance_*_*.parquet")
    return sorted(str(p.relative_to(released)) for p in [*fc, *sv])


def test_every_per_combo_fit_is_bit_identical(refit):
    released, out = refit
    names = per_combo(released)
    assert len(names) == 95, names
    differ = [n for n in names if not pd.read_parquet(out / n).equals(pd.read_parquet(released / n))]
    assert differ == []


@pytest.mark.parametrize("name", sorted(MASTERS))
def test_master_files_hold_the_same_rows(refit, name):
    released, out = refit
    new, old = pd.read_parquet(out / name), pd.read_parquet(released / name)
    assert sorted(new.columns) == sorted(old.columns)
    keys = MASTERS[name] or list(new.columns)
    old = old[new.columns]
    assert new.sort_values(keys).reset_index(drop=True).equals(old.sort_values(keys).reset_index(drop=True))


@pytest.mark.parametrize("name", TABLES)
def test_tables_are_byte_identical(refit, name):
    released, out = refit
    sub = "full_coverage_results"
    assert (out / sub / name).read_bytes() == (released / sub / name).read_bytes()


@pytest.mark.parametrize("name", SAME_LINES)
def test_appendix_tables_hold_the_same_lines(refit, name):
    released, out = refit
    sub = "full_coverage_results"
    assert sorted((out / sub / name).read_text().splitlines()) == sorted(
        (released / sub / name).read_text().splitlines())
