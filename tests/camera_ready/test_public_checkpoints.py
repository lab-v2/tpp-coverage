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

from conftest import A2, assert_same_rows, run

pytestmark = pytest.mark.network
RESULTS = A2 / "results"
SMALL_CELLS = ["olmo_ladder/dolma17-peteish", "gadre_overtrain/C4"]
KEY = ["cell", "window", "ray_bin", "n_rays", "seed", "law"]


@pytest.fixture(scope="module")
def work(pc_work):
    """A private work directory holding the HuggingFace run tables."""
    return pc_work


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
    assert_same_rows(fresh, committed[committed["cell"].isin(SMALL_CELLS)], KEY)
    screen = pd.read_csv(path / "screening_small.csv")
    ref = pd.read_csv(RESULTS / "screening_E.csv")
    assert_same_rows(screen, ref[ref["cell"].isin(SMALL_CELLS)], ["cell"])


def test_margins_reproduce(work):
    """margins.py rebuilds every E-corpus design and must match the committed margins."""
    path, env = work
    shutil.copy(RESULTS / "results_E.csv", path / "results_E.csv")
    run(["python", A2 / "margins.py", "--runs", "runs_expanded.csv", "--corpus", "E",
         "--results", "results_E.csv", "--out", "margins_E.csv"], cwd=A2, env=env)
    assert_same_rows(pd.read_csv(path / "margins_E.csv"), pd.read_csv(RESULTS / "margins_E.csv"),
                 KEY[:-1])


def test_seed_enumeration_reproduces_committed_rows(work):
    """The 512-seed enumeration, seeds 0-7, two cells: equal to the committed slice."""
    path, env = work
    run(["python", A2 / "enumerate_seeds.py", "--corpus", "E", "--runs", "runs_expanded.csv",
         "--seeds", "8", "--cells", *SMALL_CELLS, "--out", "seeds_small.csv",
         "--workers", str(min(32, os.cpu_count() or 1))], cwd=A2, env=env)
    fresh = pd.read_csv(path / "seeds_small.csv")
    committed = pd.read_csv(RESULTS / "results_E_seeds512_seeds0-7.csv.gz")
    assert_same_rows(fresh, committed[committed["cell"].isin(SMALL_CELLS)],
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


COMPANIONS = ["no fixed effect (pooled)", "no FE, suite-weighted (A4.4)",
              "cell FE, suite-weighted (A4.4)", "(cell,window) FE", "cell FE, complete case",
              "law = Chinchilla", "law = Kaplan", "C2 beta_eff = 0.1", "C2 beta_eff = 0.5",
              "NCM arm, cell FE", "NCM arm, (cell,window) FE", "NCM arm, no fixed effect"]


def _rows(corpus: str, section: str | None = None) -> pd.DataFrame:
    est = pd.read_csv(RESULTS / "estimates_seeds512.csv")
    est = est[est["corpus"] == corpus]
    return (est if section is None else est[est["section"] == section]).set_index("spec")


def _pct3(row: pd.Series) -> tuple[int, int, int]:
    return round(row["pct_512"]), round(row["pct_lo_512"]), round(row["pct_hi_512"])


def test_results_paragraph_five_releases():
    """The appendix's Results paragraph, the five fully trained releases."""
    e = _rows("E")
    suite = _rows("E", "2 per suite")
    assert [round(suite.loc[f"suite = {s}", "pct_512"]) for s in
            ("chinchilla_isoflop", "farseer", "gemstones_cooldown", "gadre_overtrain")] == \
        [32, 38, 168, -42]
    gadre = suite.loc["suite = gadre_overtrain"]
    assert gadre["lo_512"] == gadre["hi_512"] == gadre["beta_512"]  # three cells, one window each
    assert "NOT IDENTIFIED" in suite.loc["suite = olmo_ladder", "note"]
    # every companion of the results table excludes zero
    assert e.loc[COMPANIONS, "excl0_512"].all()
    # the secondary regressor, per tenfold larger shortfall of V_K below tau_K: 10**(-beta) - 1
    sec = e.loc["secondary regressor m_CO (pred < 0)"]
    assert [round(100 * (10 ** -sec[c] - 1)) for c in ("beta_512", "hi_512", "lo_512")] == [23, 6, 35]
    # the middle values of the beta_eff grid
    mid = [round(e.loc[f"C2 beta_eff = {b}", "pct_512"]) for b in (0.2, 0.28, 0.35)]
    assert (min(mid), max(mid)) == (38, 40)
    # refits that each drop one study: ten positive points, every cell-FE interval excludes 0,
    # every pooled one but the refit without the OLMo ladder
    cell, pooled = _rows("E", "2 LOSO cellFE"), _rows("E", "2 LOSO pooled")
    assert len(cell) == len(pooled) == 5
    assert (cell["beta_512"] > 0).all() and (pooled["beta_512"] > 0).all()
    assert cell["excl0_512"].all()
    assert pooled.index[~pooled["excl0_512"]].tolist() == ["drop olmo_ladder"]
    assert _pct3(pooled.loc["drop olmo_ladder"])[1:] == (-1, 20)


def test_results_paragraph_all_ten():
    """The appendix's Results paragraph, all ten releases."""
    x = _rows("X")
    assert _pct3(x.loc["cell FE (PRIMARY)"]) == (4, -10, 21)
    assert _pct3(x.loc["no fixed effect (pooled)"]) == (0, -11, 15)
    sec = x.loc["secondary regressor m_CO (pred < 0)"]
    assert [round(100 * (10 ** -sec[c] - 1)) for c in ("beta_512", "hi_512", "lo_512")] == [7, -5, 20]
    exclude = x.loc[COMPANIONS][x.loc[COMPANIONS, "excl0_512"].astype(bool)]
    assert sorted(exclude.index) == sorted(["no FE, suite-weighted (A4.4)", "NCM arm, cell FE",
                                            "NCM arm, (cell,window) FE"])
    assert [_pct3(exclude.loc[s]) for s in ("no FE, suite-weighted (A4.4)", "NCM arm, cell FE",
                                            "NCM arm, (cell,window) FE")] == \
        [(7, 3, 13), (15, 2, 29), (15, 4, 27)]
    mid = [round(x.loc[f"C2 beta_eff = {b}", "pct_512"]) for b in (0.2, 0.28, 0.35)]
    assert (min(mid), max(mid)) == (3, 5)
    suite = _rows("X", "2 per suite")
    assert _pct3(suite.loc["suite = datadecide"]) == (19, 4, 35)
    assert _pct3(suite.loc["suite = gemstones_hot"]) == (188, 54, 347)
    assert _pct3(suite.loc["suite = mad"]) == (-86, -95, -54)
    assert _pct3(suite.loc["suite = gpt3"]) == (18, -34, 180)
    assert round(suite.loc["suite = opt_trajectories", "pct_512"]) == -63
    assert suite.loc["suite = opt_trajectories", "n_windows"] == 1
    cell = _rows("X", "2 LOSO cellFE")
    assert cell.index[cell["excl0_512"].astype(bool)].tolist() == ["drop mad"]
    assert _pct3(cell.loc["drop mad"]) == (28, 16, 41)


def test_scale_paragraph_bands():
    x = _rows("X", "3 stratum")
    x = x[x["arm"] == "NC"]
    assert [round(x.loc[f"stratum {b}", "pct_512"]) for b in ("<100M", "100M-1B", ">=1B")] == \
        [32, -28, 23]
    assert _pct3(_rows("X", "3 interaction").query("arm == 'NC'").loc["b1 Delta x logN"]) == \
        (-39, -57, -18)
