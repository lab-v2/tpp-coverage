"""Camera-ready A2: the public-checkpoint reanalysis reproduces its committed outputs.

The run tables come from the public HuggingFace dataset (no account needed); every fit is
recomputed from them and compared with the committed reference outputs under
camera_ready/public_checkpoints/results/.
"""

from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import A2, run

pytestmark = pytest.mark.network
RESULTS = A2 / "results"
SMALL_CELLS = ["olmo_ladder/dolma17-peteish", "gadre_overtrain/C4"]
KEY = ["cell", "window", "ray_bin", "n_rays", "seed", "law"]


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """A private work directory holding the HuggingFace run tables."""
    path = tmp_path_factory.mktemp("pc_work")
    env = {**os.environ, "TPP_PC_WORK": str(path)}
    run(["python", A2 / "get_runs.py"], cwd=A2, env=env)
    return path, env


def _assert_same(fresh: pd.DataFrame, committed: pd.DataFrame, keys: list[str]) -> None:
    fresh = fresh.sort_values(keys).reset_index(drop=True)
    committed = committed.sort_values(keys).reset_index(drop=True)
    assert len(fresh) == len(committed) > 0
    pd.testing.assert_frame_equal(fresh[committed.columns], committed, check_exact=False,
                                  rtol=1e-9, atol=0)


def test_run_tables_downloaded(work):
    path, _ = work
    expanded = pd.read_csv(path / "artifacts" / "runs_expanded.csv")
    assert set(expanded["corpus"]) == {"E", "X"}
    assert {"cell", "run_id", "N", "D", "k", "loss"} <= set(expanded.columns)
    np.testing.assert_allclose(expanded["k"], expanded["D"] / expanded["N"], rtol=1e-9)


def test_fits_reproduce_committed_rows(work):
    """Fit two cells from scratch: every row equals the committed 5-seed table."""
    path, env = work
    run(["python", A2 / "analysis.py", "--runs", "runs_expanded.csv", "--corpus", "E",
         "--cells", *SMALL_CELLS, "--screening", "screening_small.csv",
         "--out", "results_small.csv", "--workers", str(min(32, os.cpu_count() or 1))],
        cwd=A2, env=env)
    fresh = pd.read_csv(path / "results_small.csv")
    committed = pd.read_csv(RESULTS / "results_E.csv")
    _assert_same(fresh, committed[committed["cell"].isin(SMALL_CELLS)], KEY)
    screen = pd.read_csv(path / "screening_small.csv")
    ref = pd.read_csv(RESULTS / "screening_E.csv")
    _assert_same(screen, ref[ref["cell"].isin(SMALL_CELLS)], ["cell"])


def test_margins_reproduce(work):
    """margins.py rebuilds every E-corpus design and must match the committed margins."""
    path, env = work
    shutil.copy(RESULTS / "results_E.csv", path / "results_E.csv")
    run(["python", A2 / "margins.py", "--runs", "runs_expanded.csv", "--corpus", "E",
         "--results", "results_E.csv", "--out", "margins_E.csv"], cwd=A2, env=env)
    _assert_same(pd.read_csv(path / "margins_E.csv"), pd.read_csv(RESULTS / "margins_E.csv"),
                 KEY[:-1])


def test_seed_enumeration_reproduces_committed_rows(work):
    """The 512-seed enumeration, seeds 0-7, two cells: equal to the committed slice."""
    path, env = work
    run(["python", A2 / "enumerate_seeds.py", "--corpus", "E", "--runs", "runs_expanded.csv",
         "--seeds", "8", "--cells", *SMALL_CELLS, "--out", "seeds_small.csv",
         "--workers", str(min(32, os.cpu_count() or 1))], cwd=A2, env=env)
    fresh = pd.read_csv(path / "seeds_small.csv")
    committed = pd.read_csv(RESULTS / "results_E_seeds512_seeds0-7.csv.gz")
    _assert_same(fresh, committed[committed["cell"].isin(SMALL_CELLS)],
                 ["cell", "window", "ray_bin", "law", "seed"])


def test_five_seed_estimates_match_committed():
    """Point estimates on the committed 5-seed tables equal the beta_5 column of the estimates."""
    sys.path.insert(0, str(A2))
    import conditional as C
    from cluster_bootstrap import ClusterOLS

    res = pd.read_csv(RESULTS / "results_E.csv")
    mar = pd.read_csv(RESULTS / "margins_E.csv")
    df = res.merge(mar, on=KEY[:-1], how="left", validate="many_to_one")
    df = C.add_margins(df[df["n_rays"] == 1].copy(), C.BETA_PRIMARY)
    df["_one"] = "all"
    est = pd.read_csv(RESULTS / "estimates_seeds512.csv")
    for spec, fe in (("cell FE (PRIMARY)", "cell"), ("no fixed effect (pooled)", "_one")):
        beta = ClusterOLS(df, "gap_offray", ["Delta_NC"], fe_col=fe).point()[0]
        row = est[(est["corpus"] == "E") & (est["spec"] == spec) & (est["arm"] == "NC")]
        assert float(row["beta_5"].iloc[0]) == pytest.approx(beta, abs=1e-10)


@pytest.mark.parametrize("spec,stratum,pct,lo,hi", [
    ("cell FE (PRIMARY)", "all", 40, 24, 52),            # within a study
    ("no fixed effect (pooled)", "all", 12, 5, 18),      # across studies
    ("stratum 100M-1B", "100M-1B", 42, 20, 58),
    ("stratum >=1B", ">=1B", 28, -9, 41),
    ("b1 Delta x logN", "all", -14, -40, 18),            # size interaction
])
def test_headline_estimates(spec, stratum, pct, lo, hi):
    est = pd.read_csv(RESULTS / "estimates_seeds512.csv")
    row = est[(est["corpus"] == "E") & (est["spec"] == spec) & (est["arm"] == "NC")
              & (est["stratum"] == stratum)].iloc[0]
    assert (round(row["pct_512"]), round(row["pct_lo_512"]), round(row["pct_hi_512"])) == \
        (pct, lo, hi)


def test_datadecide_and_vk_ratio_headlines():
    est = pd.read_csv(RESULTS / "estimates_seeds512.csv")
    dd = est[(est["corpus"] == "X") & (est["spec"] == "suite = datadecide")].iloc[0]
    assert (round(dd["pct_512"]), round(dd["pct_lo_512"]), round(dd["pct_hi_512"])) == (19, 4, 35)
    vk = pd.read_csv(RESULTS / "vk_ratio.csv").set_index("stratum")["median_vk_ratio"]
    assert round(vk[">=1B"], 1) == 3.7 and round(vk["<1B"], 1) == 5.8


def test_regime_win_rates():
    dr = pd.read_csv(RESULTS / "doseresponse_seeds512.csv")
    dr = dr[(dr["corpus"] == "E") & (dr["arm"] == "NC")]
    sep = dr[dr["regime_cell"] == "CO=A,NC=B"].set_index("kappa")["win_512"]
    assert [round(100 * sep[k]) for k in (1e4, 1e5, 1e6)] == [59, 61, 76]
    same = dr[dr["regime_cell"].isin(["CO=A,NC=A", "CO=B,NC=B"]) & (dr["pairs_512"] > 10_000)]
    assert same["win_512"].between(0.43, 0.47).all()
