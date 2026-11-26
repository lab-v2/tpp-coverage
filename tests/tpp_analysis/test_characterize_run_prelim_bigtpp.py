"""Characterization of `run_prelim_bigtpp.py`: the BigTPP Chinchilla fits and their win-rate table.

The table is built from the released BigTPP fits (`prelim_bigtpp_results/prelim_bigtpp.parquet`)
and checked against the base commit's output and the released table (they are identical); the script also runs end to
end on the released BigTPP loss table (2 seeds of 1 restart). Expected values are frozen in
`frozen/` (see `tests/characterize.py`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from characterize import check_frozen, released
from conftest import ANALYSIS, pool_width, run

pytestmark = pytest.mark.network
FROZEN = Path(__file__).parent / "frozen"
TABLES = ["winrate_bigtpp_prelim.csv", "table_winrate_bigtpp_prelim.tex"]


def test_table(tmp_path):
    import run_prelim_bigtpp as rp

    prelim = released("prelim_bigtpp_results/*") / "prelim_bigtpp_results"
    rp.build_winrate_table(pd.read_parquet(prelim / "prelim_bigtpp.parquet"), str(tmp_path))
    written = {name: (tmp_path / name).read_text() for name in TABLES}
    assert written == {name: (prelim / name).read_text() for name in TABLES}
    check_frozen(FROZEN / "run_prelim_bigtpp_table.json", written)


def test_script(tmp_path):
    losses = released("extracted_losses/wikipedia_bigtpp.csv") / "extracted_losses"
    out = tmp_path / "prelim"
    args = ["--data-dir", losses, "-o", out, "--n-seeds", "2", "--n-restarts", "1"]
    run(["python", ANALYSIS / "run_prelim_bigtpp.py", *args, "--n-workers", str(pool_width(4))])
    fits = pd.read_parquet(next(out.glob("*.parquet")))
    fits = fits.sort_values(["design", "seed"]).reset_index(drop=True)
    tables = {name: (out / name).read_text() for name in TABLES}
    check_frozen(FROZEN / "run_prelim_bigtpp_script.json", {"fits": fits, "tables": tables})
