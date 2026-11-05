"""The intervals of the public-checkpoint appendix: printed values, code paths, full reruns.

Three layers, all offline:

1. The committed outputs in camera_ready/public_checkpoints/results/ carry the values the
   paper's appendix prints. Every expected value is a literal copied from the appendix text;
   where two scripts compute the same quantity, their outputs agree.
2. Each script, run on the committed seeds 0-7 slice of the E-corpus enumeration, agrees with
   a direct computation from that slice written here from the definitions (band edges 100M
   and 1B on the window's geometric-mean size, Regime B iff m + log10(kappa/100) >= 0).
3. With the full 512-seed tables in the work directory (TPP_PC_WORK, default
   camera_ready/public_checkpoints/work), each script reproduces its committed output. The
   repository does not ship those tables: enumerate_seeds.py writes them from the
   HuggingFace run tables (about 16 min for E and 2 h for X on 480 cores). Without them
   these tests skip and name the file they need.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import A2, assert_same_rows, run

sys.path.insert(0, str(A2))
from paths import WORK  # noqa: E402

RESULTS = A2 / "results"
SCRIPTS = A2 / "intervals"
SLICE = RESULTS / "results_E_seeds512_seeds0-7.csv.gz"
FULL_E = WORK / "results_E_seeds512.csv"
FULL_X = WORK / "results_X_seeds512.csv.gz"
KAPPAS = (100, 10_000, 100_000, 1_000_000)
SHARED = {("CO=A,NC=A", 100), ("CO=B,NC=B", 10_000), ("CO=B,NC=B", 100_000),
          ("CO=B,NC=B", 1_000_000)}  # both designs in one regime; the 826-pair cell is set aside
WORKERS = str(min(16, os.cpu_count() or 1))
RTOL_ENDS = 1e-4  # percentile interpolation before or after 10 ** x (vk_ratio_seeds.py)


def pct(x: pd.Series) -> list[int]:
    return [round(100 * v) for v in x]


# ── 1. the committed outputs carry the printed values ──────────────────────────────────────

def test_median_ratio_intervals_as_printed():
    """5.8 [5.4, 6.9] below 1B, 3.7 [2.1, 5.6] at 1B and above, 0.64 [0.2, 1.3], 300,000 draws."""
    seeds = pd.read_csv(RESULTS / "vk_ratio_seeds.csv")
    pooled = seeds[seeds["stream"] == "pooled"].set_index("statistic")
    assert (pooled["n_boot"] == 300_000).all()
    assert (seeds[seeds["stream"] == "seed"].groupby("statistic").size() == 30).all()
    cols = ["point", "lo95", "hi95"]
    assert pooled.loc["<1B", cols].astype(float).round(1).tolist() == [5.8, 5.4, 6.9]
    assert pooled.loc[">=1B", cols].astype(float).round(1).tolist() == [3.7, 2.1, 5.6]
    assert round(pooled.loc["ratio", "point"], 2) == 0.64
    assert pooled.loc["ratio", ["lo95", "hi95"]].astype(float).round(1).tolist() == [0.2, 1.3]
    ref = pd.read_csv(RESULTS / "vk_ratio.csv").set_index("stratum")
    assert round(ref.loc[">=1B", "N_max"] / 1e9, 1) == 12.6  # "whose largest model is 12.6B"


def test_single_stream_intervals_are_the_released_scripts():
    """The released streams of vk_ratio_seeds.py are the 10,000-draw outputs of the two scripts.

    The same draws: vk_ratio_boot.py takes the percentiles of the median log ratio and
    exponentiates them, vk_ratio_seeds.py takes them of the ratio, so the interpolated ends
    agree to RTOL_ENDS rather than exactly.
    """
    seeds = pd.read_csv(RESULTS / "vk_ratio_seeds.csv")
    rel = seeds[seeds["stream"] == "released"].set_index("statistic")
    boot = pd.read_csv(RESULTS / "vk_ratio_boot.csv")
    e2 = boot[boot["scheme"] == "E2"].set_index("stratum")
    for stat in ("<1B", ">=1B"):
        np.testing.assert_allclose(rel.loc[stat, ["point", "lo95", "hi95"]].astype(float),
                                   e2.loc[stat, ["median_vk_ratio", "lo95", "hi95"]].astype(float),
                                   rtol=RTOL_ENDS)
        assert rel.loc[stat, "point"] == pytest.approx(e2.loc[stat, "median_vk_ratio"], rel=1e-12)
    rom = pd.read_csv(RESULTS / "ratio_of_medians_boot.csv").iloc[0]
    np.testing.assert_allclose(rel.loc["ratio", ["point", "lo95", "hi95"]].astype(float),
                               [rom["ratio_of_medians"], rom["lo95"], rom["hi95"]], rtol=1e-12)
    # the same medians as vk_ratio.py, an output written by another script
    ref = pd.read_csv(RESULTS / "vk_ratio.csv").set_index("stratum")
    np.testing.assert_allclose(e2["median_vk_ratio"], ref.loc[e2.index, "median_vk_ratio"],
                               rtol=1e-12)
    assert rom["ratio_of_medians"] == pytest.approx(
        ref.loc[">=1B", "median_vk_ratio"] / ref.loc["<1B", "median_vk_ratio"], rel=1e-12)


def test_windows_at_1b_and_above_as_printed():
    win = pd.read_csv(RESULTS / "vk_ratio_boot_ge1B_windows.csv")
    assert len(win) == 13 and int((win["median_vk_ratio"] < 1).sum()) == 3
    assert not win["cell"].str.startswith("olmo_ladder").any()  # its CO designs have V_K = 0
    ref = pd.read_csv(RESULTS / "vk_ratio.csv").set_index("stratum")
    assert win["n_pairs"].sum() == ref.loc[">=1B", "n_pairs"]


@pytest.fixture(scope="module")
def cells() -> pd.DataFrame:
    c = pd.read_csv(RESULTS / "regime_cells.csv")
    c["kappa"] = c["kappa"].astype(int)
    return c


def _family(cells: pd.DataFrame, family: str) -> pd.DataFrame:
    return cells[cells["family"] == family].set_index(["corpus", "kappa", "cell", "band"])


def test_regime_cell_verdicts_do_not_depend_on_the_seed(cells):
    assert (cells["replicates"] == 10).all()
    assert cells["share_excl0"].isin([0.0, 1.0]).all()


def test_joint_regime_test_as_printed(cells):
    j = _family(cells, "joint").xs("all", level="band")
    sep = [j.loc[(c, k, "CO=A,NC=B")] for c in ("E", "X") for k in KAPPAS[1:]]
    assert pct(pd.Series([r["win"] for r in sep])) == [59, 61, 76, 59, 56, 76]
    assert all(r["excludes0"] and r["median"] > 0 for r in sep)
    assert [round(r["median"], 3) for r in sep[:3]] == [0.074, 0.069, 0.151]  # not monotone
    # at the main-text kappa_target: no separated pair on the five, two windows on the ten
    assert j.loc[("E", 100, "CO=A,NC=B"), "n_pairs"] == 0
    assert j.loc[("X", 100, "CO=A,NC=B"), "n_windows"] == 2
    # the four large cells with both subsets in one regime
    large = [("CO=A,NC=A", 100), ("CO=B,NC=B", 10_000), ("CO=B,NC=B", 100_000),
             ("CO=B,NC=B", 1_000_000)]
    e = pd.DataFrame([j.loc[("E", k, c)] for c, k in large])
    assert (min(pct(e["win"])), max(pct(e["win"]))) == (44, 47)
    worse = [round(100 * (10 ** -m - 1)) for m in e["median"]]  # NC's off-ray error, higher by
    assert (min(worse), max(worse)) == (6, 11)
    assert e["excludes0"].tolist() == [False, True, False, True]
    assert e["hi"].iloc[3] < 0 and e["hi"].iloc[3] > -0.002  # excludes zero only just at 1e6
    assert 0 < e["hi"].iloc[2] < 0.002  # contains zero only just at 1e5
    x = pd.DataFrame([j.loc[("X", k, c)] for c, k in large])
    assert not x["excludes0"].any() and (min(pct(x["win"])), max(pct(x["win"]))) == (48, 50)
    # the small cell at 1e4 with both in Regime A: 826 pairs in 3 windows, all from the five
    small = [j.loc[(c, 10_000, "CO=A,NC=A")] for c in ("E", "X")]
    assert [(r["n_pairs"], r["n_windows"]) for r in small] == [(826, 3), (826, 3)]
    assert pct(pd.Series([r["win"] for r in small])) == [30, 30]
    assert all(r["excludes0"] for r in small)
    # the reversed cell
    rev = [j.loc[(c, 10_000, "CO=B,NC=A")] for c in ("E", "X")]
    assert pct(pd.Series([r["win"] for r in rev])) == [26, 19]


def test_collinear_only_regime_as_printed(cells):
    a = _family(cells, "coonly").xs(("CO=A", "all"), level=("cell", "band"))
    # every pair is in Regime A at the main-text kappa_target = 100
    est = pd.read_csv(RESULTS / "estimates_seeds512.csv")
    primary = est[(est["spec"] == "cell FE (PRIMARY)") & (est["arm"] == "NC")].set_index("corpus")
    for corpus in ("E", "X"):
        assert a.loc[(corpus, 100), "n_pairs"] == primary.loc[corpus, "n_pairs_512"]
    assert pct(a.loc["E"].loc[list(KAPPAS), "win"]) == [47, 58, 61, 76]
    assert pct(a.loc["X"].loc[list(KAPPAS), "win"]) == [50, 58, 56, 76]
    assert a.loc["E"].loc[list(KAPPAS), "excludes0"].tolist() == [False, False, True, True]
    assert a.loc["X"].loc[list(KAPPAS), "excludes0"].tolist() == [False, True, True, True]


def test_each_laws_own_exponent_as_printed(cells):
    """Kaplan pairs classified at beta_eff = 0.10: 54, 62, 67% (five) and 57, 54, 67% (ten)."""
    a = _family(cells, "table7").xs(("CO=A", "all"), level=("cell", "band"))
    big = list(KAPPAS[1:])
    assert pct(a.loc["E"].loc[big, "win"]) == [54, 62, 67]
    assert pct(a.loc["X"].loc[big, "win"]) == [57, 54, 67]
    assert a.loc["E"].loc[big, "excludes0"].tolist() == [False, True, True]
    assert a.loc["X"].loc[big, "excludes0"].tolist() == [True, False, True]


def _band(rows: pd.DataFrame, corpus: str, stratum: str, shared: bool) -> pd.DataFrame:
    keys = list(zip(rows["cell"], rows["kappa"]))
    pick = [(k in SHARED) if shared else (k[0] == "CO=A,NC=B") for k in keys]
    sub = rows[pick]
    return sub[(sub["corpus"] == corpus) & (sub["band"] == stratum)]


def test_regime_arms_by_band_as_printed(cells):
    rows = cells[cells["family"] == "band"]

    # below 100M: the five give 2 windows in one cell
    e_small = rows[(rows["corpus"] == "E") & (rows["band"] == "<100M")]
    assert e_small["n_windows"].max() == 2
    shared = _band(rows, "X", "<100M", shared=True)
    assert len(shared) == 4 and min(pct(shared["win"])) == 36 and max(pct(shared["win"])) == 43
    assert shared["excludes0"].all() and (shared["median"] < 0).all()
    sep = _band(rows, "X", "<100M", shared=False).set_index("kappa")
    assert round(100 * sep.loc[10_000, "win"]) == 59 and sep.loc[10_000, "excludes0"]

    # 100M to 1B
    sep, shared = (_band(rows, "E", "100M-1B", shared=s) for s in (False, True))
    assert (min(pct(sep["win"])), max(pct(sep["win"]))) == (47, 62)
    assert (min(pct(shared["win"])), max(pct(shared["win"]))) == (46, 48)
    assert not sep["excludes0"].any() and not shared["excludes0"].any()
    sep, shared = (_band(rows, "X", "100M-1B", shared=s) for s in (False, True))
    assert (min(pct(shared["win"])), max(pct(shared["win"]))) == (56, 57)
    assert shared["excludes0"].all() and (shared["median"] > 0).all()
    assert not sep["excludes0"].any()

    # 1B and above, both corpora: CO ahead at every target; the five's 1e2 only just contains 0
    shared = pd.concat([_band(rows, c, ">=1B", shared=True) for c in ("E", "X")])
    assert len(shared) == 8 and (min(pct(shared["win"])), max(pct(shared["win"]))) == (41, 44)
    assert (shared["median"] < 0).all()
    open_ = shared[~shared["excludes0"]]
    assert list(zip(open_["corpus"], open_["kappa"])) == [("E", 100)]
    assert 0 < open_["hi"].iloc[0] < 0.002
    sep = pd.concat([_band(rows, c, ">=1B", shared=False) for c in ("E", "X")])
    assert sep["n_windows"].max() == 3


@pytest.mark.parametrize("name,keys,family,cell_col", [
    ("regime_by_band_ci.csv", ["corpus", "kappa", "regime_cell", "stratum"], "band", "regime_cell"),
    ("co_only_regime.csv", ["corpus", "kappa", "co_regime"], "coonly", "co_regime"),
])
def test_2000_draw_tables_hold_the_same_pairs(cells, name, keys, family, cell_col):
    """The released 2,000-draw tables and regime_cells.csv describe the same pairs."""
    old = pd.read_csv(RESULTS / name)
    new = cells[(cells["family"] == family) & (cells["n_pairs"] > 0)].copy()
    if family == "coonly":
        new["co_regime"] = new["cell"].str[3]
        new = new.drop(columns="cell")
    else:
        new = new.rename(columns={"cell": "regime_cell", "band": "stratum"})
    m = old.merge(new, on=keys, suffixes=("_old", ""), validate="one_to_one")
    assert len(m) == len(old) == len(new)
    assert (m["n_pairs_old"] == m["n_pairs"]).all() and (m["n_windows_old"] == m["n_windows"]).all()
    np.testing.assert_allclose(m["win_old"], m["win"], rtol=1e-12)
    np.testing.assert_allclose(m["median_old"], m["median"], rtol=1e-12)
    assert cell_col in m


def test_slice_facts_the_appendix_prints():
    d = pd.read_csv(SLICE)
    olmo = d[d["cell"].str.startswith("olmo_ladder")]
    assert np.isneginf(olmo["CO_m_beta0.28"]).all()  # every collinear subset has V_K = 0
    assert (round(olmo["N_min"].min() / 1e6), round(olmo["N_max"].max() / 1e8) / 10) == (190, 13.2)
    assert d["N_max"].max() < 16e9  # Chinchilla's 16.2B run enters no window


# ── 2. each script on the committed slice, against a direct computation ────────────────────

@pytest.fixture(scope="module")
def slice_frame() -> pd.DataFrame:
    d = pd.read_csv(SLICE)
    d["delta"] = d["NC_m_beta0.28"] - d["CO_m_beta0.28"]
    d["band"] = np.where(d["N_gm"] < 1e8, "<100M", np.where(d["N_gm"] < 1e9, "100M-1B", ">=1B"))
    return d


def _run(script: str, tmp_path, *args: str) -> None:
    run(["python", SCRIPTS / script, "--results", SLICE, *args], cwd=A2,
        env={**os.environ, "TPP_PC_WORK": str(tmp_path)})


def test_vk_ratio_boot_on_slice(tmp_path, slice_frame):
    _run("vk_ratio_boot.py", tmp_path, "--n-boot", "200", "--workers", WORKERS,
         "--out", tmp_path / "a.csv", "--windows-out", tmp_path / "w.csv")
    out = pd.read_csv(tmp_path / "a.csv")
    fin = slice_frame[np.isfinite(slice_frame["delta"])]
    bands = {"<100M": fin["band"] == "<100M", "100M-1B": fin["band"] == "100M-1B",
             "<1B": fin["band"] != ">=1B", ">=1B": fin["band"] == ">=1B"}
    for stratum, mask in bands.items():
        rows = out[out["stratum"] == stratum]
        assert len(rows) == 3 and (rows["n_pairs"] == mask.sum()).all()
        np.testing.assert_allclose(rows["median_vk_ratio"], 10 ** fin.loc[mask, "delta"].median(),
                                   rtol=1e-12)
    win = pd.read_csv(tmp_path / "w.csv")
    top = fin[fin["band"] == ">=1B"].groupby(["cell", "window"])["delta"].median()
    np.testing.assert_allclose(win.set_index(["cell", "window"])["median_vk_ratio"],
                               10 ** top.loc[list(zip(win["cell"], win["window"]))].to_numpy(),
                               rtol=1e-12)

    # the output does not depend on how the tasks are spread over workers
    _run("vk_ratio_boot.py", tmp_path, "--n-boot", "200", "--workers", "1",
         "--out", tmp_path / "b.csv", "--windows-out", tmp_path / "w1.csv")
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()


def test_ratio_of_medians_on_slice(tmp_path, slice_frame):
    _run("ratio_of_medians_boot.py", tmp_path, "--n-boot", "200", "--out", tmp_path / "r.csv")
    row = pd.read_csv(tmp_path / "r.csv").iloc[0]
    fin = slice_frame[np.isfinite(slice_frame["delta"])]
    top = fin["band"] == ">=1B"
    expect = 10 ** (fin.loc[top, "delta"].median() - fin.loc[~top, "delta"].median())
    assert row["ratio_of_medians"] == pytest.approx(expect, rel=1e-12)
    assert row["n_pairs"] == len(fin) and row["lo95"] <= row["hi95"]


def _regime_letters(m: pd.Series, kappa: int) -> np.ndarray:
    return np.where(m.fillna(-np.inf) + np.log10(kappa / 100) >= 0, "B", "A")


def test_co_only_regime_on_slice(tmp_path, slice_frame):
    _run("co_only_regime.py", tmp_path, "--skip-x", "--workers", WORKERS,
         "--out", tmp_path / "c.csv")
    out = pd.read_csv(tmp_path / "c.csv").set_index(["kappa", "co_regime"])
    d = slice_frame
    a_counts = []
    for kappa in KAPPAS:
        co = _regime_letters(d["CO_m_beta0.28"], kappa)
        for regime in ("A", "B"):
            g = d.loc[co == regime, "gap_offray"]
            if g.empty:
                assert (kappa, regime) not in out.index
                continue
            row = out.loc[(kappa, regime)]
            assert row["n_pairs"] == len(g)
            assert row["win"] == pytest.approx((g > 0).mean(), abs=1e-15)
            assert row["median"] == pytest.approx(g.median(), abs=1e-15)
        a_counts.append(int((co == "A").sum()))
    # a larger kappa_target only moves designs from Regime A to B
    assert a_counts == sorted(a_counts, reverse=True) and a_counts[0] > a_counts[-1]


def test_regime_by_band_on_slice(tmp_path, slice_frame):
    _run("regime_by_band_ci.py", tmp_path, "--skip-x", "--workers", WORKERS,
         "--out", tmp_path / "b.csv")
    _run("co_only_regime.py", tmp_path, "--skip-x", "--workers", WORKERS,
         "--out", tmp_path / "c.csv")
    out = pd.read_csv(tmp_path / "b.csv")
    d = slice_frame
    for kappa in KAPPAS:
        cell = ("CO=" + pd.Series(_regime_letters(d["CO_m_beta0.28"], kappa), index=d.index) + ",NC="
                + _regime_letters(d["NC_m_beta0.28"], kappa))
        for (name, band), g in d.groupby([cell, d["band"]]):
            rows = out[(out["kappa"] == kappa) & (out["regime_cell"] == name)
                       & (out["stratum"] == band)]
            if name == "CO=B,NC=A":
                assert rows.empty  # the reversed cell is not part of the test
                continue
            assert len(rows) == 1 and rows["n_pairs"].iloc[0] == len(g)
            assert rows["win"].iloc[0] == pytest.approx((g["gap_offray"] > 0).mean(), abs=1e-15)
            assert rows["median"].iloc[0] == pytest.approx(g["gap_offray"].median(), abs=1e-15)
    # the CO=A pairs of the two scripts are the same pairs
    co = pd.read_csv(tmp_path / "c.csv")
    for kappa in KAPPAS:
        a = out[(out["kappa"] == kappa) & out["regime_cell"].str.startswith("CO=A,")]
        ref = co[(co["kappa"] == kappa) & (co["co_regime"] == "A")]["n_pairs"].sum()
        assert a["n_pairs"].sum() == ref


def test_vk_ratio_seeds_on_slice(tmp_path):
    """The released streams give back the two scripts' intervals; the pooled row pools seeds."""
    _run("vk_ratio_seeds.py", tmp_path, "--seeds", "2", "--n-boot", "200", "--workers", WORKERS,
         "--out", tmp_path / "s.csv")
    _run("vk_ratio_boot.py", tmp_path, "--n-boot", "200", "--workers", WORKERS,
         "--out", tmp_path / "a.csv", "--windows-out", tmp_path / "w.csv")
    _run("ratio_of_medians_boot.py", tmp_path, "--n-boot", "200", "--out", tmp_path / "r.csv")
    s = pd.read_csv(tmp_path / "s.csv")
    rel = s[s["stream"] == "released"].set_index("statistic")
    a = pd.read_csv(tmp_path / "a.csv")
    e2 = a[a["scheme"] == "E2"].set_index("stratum")
    for stat in ("<1B", ">=1B"):
        np.testing.assert_allclose(rel.loc[stat, ["point", "lo95", "hi95"]].astype(float),
                                   e2.loc[stat, ["median_vk_ratio", "lo95", "hi95"]].astype(float),
                                   rtol=RTOL_ENDS)
        assert rel.loc[stat, "point"] == pytest.approx(e2.loc[stat, "median_vk_ratio"], rel=1e-12)
    r = pd.read_csv(tmp_path / "r.csv").iloc[0]
    np.testing.assert_allclose(rel.loc["ratio", ["point", "lo95", "hi95"]].astype(float),
                               [r["ratio_of_medians"], r["lo95"], r["hi95"]], rtol=1e-12)
    pooled = s[s["stream"] == "pooled"].set_index("statistic")
    assert (pooled["n_boot"] == 400).all() and (s[s["stream"] == "seed"]["seed"].nunique() == 2)
    one = s[s["stream"] == "seed"].groupby("statistic")
    assert (pooled["lo95"] >= one["lo95"].min() - 1e-12).all()  # pooling cannot widen past both
    assert (pooled["hi95"] <= one["hi95"].max() + 1e-12).all()


def test_cluster_medians_resample_whole_windows():
    """A hand-built cell: every draw is the median of a resample of whole windows."""
    import estimate as EST

    vals = np.array([0.0, 0.0, 10.0, 10.0, 5.0])
    groups = [[np.array([0, 1]), np.array([2, 3])], [np.array([4])]]  # two cells
    d = EST.cluster_medians(vals, groups, np.random.default_rng(0), 500)
    # cell 1 draws {0,0}+{0,0}, {0,0}+{10,10} or {10,10}+{10,10}; cell 2 always adds the 5
    assert set(np.round(d, 9)) == {0.0, 5.0, 10.0}
    one_window = EST.cluster_medians(vals, [[np.array([0, 1, 2, 3, 4])]],
                                     np.random.default_rng(0), 50)
    assert (one_window == np.median(vals)).all()  # one window per cell: no spread


def test_regime_cells_on_slice(tmp_path, slice_frame):
    _run("regime_cells.py", tmp_path, "--skip-x", "--replicates", "1", "--workers", WORKERS,
         "--out", tmp_path / "c.csv")
    out = pd.read_csv(tmp_path / "c.csv")
    out["kappa"] = out["kappa"].astype(int)
    out = out.set_index(["family", "kappa", "cell", "band"])
    assert (out["replicates"] == 1).all()
    d = slice_frame
    kaplan = d["law"] == "Kaplan"
    own = np.where(kaplan, d["CO_m_beta0.1"], d["CO_m_beta0.28"])
    for kappa in KAPPAS:
        co = _regime_letters(d["CO_m_beta0.28"], kappa)
        nc = _regime_letters(d["NC_m_beta0.28"], kappa)
        checks = {("coonly", f"CO={r}", "all"): co == r for r in ("A", "B")}
        checks[("table7", "CO=A", "all")] = _regime_letters(pd.Series(own), kappa) == "A"
        for c in ("A", "B"):
            for n in ("A", "B"):
                checks[("joint", f"CO={c},NC={n}", "all")] = (co == c) & (nc == n)
        for (family, cell, band), mask in checks.items():
            g = d.loc[mask, "gap_offray"]
            if family == "coonly" and g.empty:
                assert (family, kappa, cell, band) not in out.index
                continue
            row = out.loc[(family, kappa, cell, band)]
            assert row["n_pairs"] == len(g), (family, kappa, cell)
            if len(g):
                assert row["win"] == pytest.approx((g > 0).mean(), abs=1e-15)
                assert row["median"] == pytest.approx(g.median(), abs=1e-15)
                if row["n_windows"] >= 2:
                    assert row["lo"] <= row["median"] <= row["hi"]
    # the released 2,000-draw loop, fed the released state, is the one median_ci.py extends
    sys.path.insert(0, str(SCRIPTS))
    import median_ci as M
    import estimate as EST

    jobs = [j for j in EST.dose_jobs(EST.prepare(SLICE), "E|512") if len(j[1]) > 0
            and sum(len(g) for g in j[2]) >= 2][:2]
    M.check_released(jobs)


# ── 3. with the full 512-seed tables, each script gives back its committed output ──────────

def _needs(*paths):
    missing = [str(p) for p in paths if not p.exists()]
    return pytest.mark.skipif(bool(missing), reason=f"needs {', '.join(missing)} "
                              "(written by enumerate_seeds.py; not shipped)")


@_needs(FULL_E)
def test_full_vk_ratio_boot_reproduces(tmp_path):
    run(["python", SCRIPTS / "vk_ratio_boot.py", "--results", FULL_E, "--out", tmp_path / "a.csv",
         "--windows-out", tmp_path / "w.csv", "--workers", "12"], cwd=A2)
    assert_same_rows(pd.read_csv(tmp_path / "a.csv"), pd.read_csv(RESULTS / "vk_ratio_boot.csv"),
                     ["stratum", "scheme"])
    assert_same_rows(pd.read_csv(tmp_path / "w.csv"),
                     pd.read_csv(RESULTS / "vk_ratio_boot_ge1B_windows.csv"), ["cell", "window"])


@_needs(FULL_E)
def test_full_ratio_of_medians_reproduces(tmp_path):
    run(["python", SCRIPTS / "ratio_of_medians_boot.py", "--results", FULL_E,
         "--out", tmp_path / "r.csv"], cwd=A2)
    assert_same_rows(pd.read_csv(tmp_path / "r.csv"),
                     pd.read_csv(RESULTS / "ratio_of_medians_boot.csv"), ["statistic"])


@pytest.mark.parametrize("script,keys", [
    ("regime_by_band_ci.py", ["corpus", "kappa", "regime_cell", "stratum"]),
    ("co_only_regime.py", ["corpus", "kappa", "co_regime"]),
])
@_needs(FULL_E, FULL_X)
def test_full_regime_tables_reproduce(tmp_path, script, keys):
    run(["python", SCRIPTS / script, "--results", FULL_E, "--results-x", FULL_X,
         "--out", tmp_path / "o.csv", "--workers", "48"], cwd=A2)
    name = script.replace(".py", ".csv")
    assert_same_rows(pd.read_csv(tmp_path / "o.csv"), pd.read_csv(RESULTS / name), keys)


@_needs(FULL_E)
def test_full_vk_ratio_seeds_reproduces(tmp_path):
    run(["python", SCRIPTS / "vk_ratio_seeds.py", "--results", FULL_E, "--out", tmp_path / "s.csv",
         "--workers", "93"], cwd=A2)
    assert_same_rows(pd.read_csv(tmp_path / "s.csv"), pd.read_csv(RESULTS / "vk_ratio_seeds.csv"),
                     ["statistic", "stream", "seed"])


@_needs(FULL_E, FULL_X)
def test_full_regime_cells_reproduce(tmp_path):
    """Replicate 0 of every cell; the other nine replicates only test verdict stability."""
    run(["python", SCRIPTS / "regime_cells.py", "--results", FULL_E, "--results-x", FULL_X,
         "--replicates", "1", "--out", tmp_path / "c.csv", "--workers", str(os.cpu_count() or 1)],
        cwd=A2, timeout=3600)
    cols = ["family", "corpus", "kappa", "cell", "band", "key", "n_pairs", "n_windows", "win",
            "median", "lo", "hi", "excludes0"]
    assert_same_rows(pd.read_csv(tmp_path / "c.csv")[cols],
                     pd.read_csv(RESULTS / "regime_cells.csv")[cols], cols[:5])
