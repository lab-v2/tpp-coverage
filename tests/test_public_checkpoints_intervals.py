"""The intervals of the public-checkpoint appendix: printed values, code paths, full reruns.

Three layers, all offline:

1. The committed outputs in camera_ready/public_checkpoints/results/ carry the values the
   paper's appendix prints. Every expected value is a literal copied from the appendix text.
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


def pct(x: pd.Series) -> list[int]:
    return [round(100 * v) for v in x]


# ── 1. the committed outputs carry the printed values ──────────────────────────────────────

def test_median_ratio_intervals_as_printed():
    boot = pd.read_csv(RESULTS / "vk_ratio_boot.csv")
    e2 = boot[boot["scheme"] == "E2"].set_index("stratum")
    assert (e2["n_boot"] == 10_000).all()
    cols = ["median_vk_ratio", "lo95", "hi95"]
    assert e2.loc["<1B", cols].astype(float).round(1).tolist() == [5.8, 5.4, 6.9]
    assert e2.loc[">=1B", cols].astype(float).round(1).tolist() == [3.7, 2.1, 5.5]
    # the same medians as vk_ratio.py, an output written by another script
    ref = pd.read_csv(RESULTS / "vk_ratio.csv").set_index("stratum")
    np.testing.assert_allclose(e2["median_vk_ratio"], ref.loc[e2.index, "median_vk_ratio"],
                               rtol=1e-12)
    assert round(ref.loc[">=1B", "N_max"] / 1e9, 1) == 12.6  # "whose largest model is 12.6B"


def test_windows_at_1b_and_above_as_printed():
    win = pd.read_csv(RESULTS / "vk_ratio_boot_ge1B_windows.csv")
    assert len(win) == 13 and int((win["median_vk_ratio"] < 1).sum()) == 3
    assert not win["cell"].str.startswith("olmo_ladder").any()  # its CO designs have V_K = 0
    ref = pd.read_csv(RESULTS / "vk_ratio.csv").set_index("stratum")
    assert win["n_pairs"].sum() == ref.loc[">=1B", "n_pairs"]


def test_ratio_of_medians_as_printed():
    row = pd.read_csv(RESULTS / "ratio_of_medians_boot.csv").iloc[0]
    assert row["n_boot"] == 10_000 and row["scheme"] == "E2"
    assert [round(row[c], 2) for c in ("ratio_of_medians", "lo95", "hi95")] == [0.64, 0.20, 1.32]
    ref = pd.read_csv(RESULTS / "vk_ratio.csv").set_index("stratum")["median_vk_ratio"]
    assert row["ratio_of_medians"] == pytest.approx(ref[">=1B"] / ref["<1B"], rel=1e-12)


def test_collinear_only_regime_as_printed():
    co = pd.read_csv(RESULTS / "co_only_regime.csv")
    a = co[co["co_regime"] == "A"].set_index(["corpus", "kappa"])
    # every pair is in Regime A at the main-text kappa_target = 100
    assert not ((co["kappa"] == 100) & (co["co_regime"] == "B")).any()
    est = pd.read_csv(RESULTS / "estimates_seeds512.csv")
    primary = est[(est["corpus"] == "E") & (est["spec"] == "cell FE (PRIMARY)")].iloc[0]
    assert a.loc[("E", 100), "n_pairs"] == primary["n_pairs_512"]
    assert pct(a.loc["E"].loc[list(KAPPAS), "win"]) == [47, 58, 61, 76]
    assert pct(a.loc["X"].loc[list(KAPPAS), "win"]) == [50, 58, 56, 76]
    assert a.loc["E"].loc[list(KAPPAS), "excludes_zero"].tolist() == [False, False, True, True]
    assert a.loc["X"].loc[list(KAPPAS), "excludes_zero"].tolist() == [False, True, True, True]


def _band(rows: pd.DataFrame, corpus: str, stratum: str, shared: bool) -> pd.DataFrame:
    keys = list(zip(rows["regime_cell"], rows["kappa"]))
    pick = [(k in SHARED) if shared else (k[0] == "CO=A,NC=B") for k in keys]
    sub = rows[pick]
    return sub[(sub["corpus"] == corpus) & (sub["stratum"] == stratum)]


def test_regime_arms_by_band_as_printed():
    rows = pd.read_csv(RESULTS / "regime_by_band_ci.csv")

    # below 100M: the five give 2 windows in one cell
    e_small = rows[(rows["corpus"] == "E") & (rows["stratum"] == "<100M")]
    assert e_small["n_windows"].max() == 2 and (e_small["n_cells"] == 1).all()
    shared = _band(rows, "X", "<100M", shared=True)
    assert len(shared) == 4 and min(pct(shared["win"])) == 36 and max(pct(shared["win"])) == 43
    assert shared["excludes_zero"].all() and (shared["median"] < 0).all()
    sep = _band(rows, "X", "<100M", shared=False).set_index("kappa")
    assert round(100 * sep.loc[10_000, "win"]) == 59 and sep.loc[10_000, "excludes_zero"]

    # 100M to 1B
    sep, shared = (_band(rows, "E", "100M-1B", shared=s) for s in (False, True))
    assert (min(pct(sep["win"])), max(pct(sep["win"]))) == (47, 62)
    assert (min(pct(shared["win"])), max(pct(shared["win"]))) == (46, 48)
    assert not sep["excludes_zero"].any() and not shared["excludes_zero"].any()
    sep, shared = (_band(rows, "X", "100M-1B", shared=s) for s in (False, True))
    assert (min(pct(shared["win"])), max(pct(shared["win"]))) == (56, 57)
    assert shared["excludes_zero"].all() and (shared["median"] > 0).all()
    assert not sep["excludes_zero"].any()

    # 1B and above, both corpora: CO ahead everywhere, only just at 1e2
    shared = pd.concat([_band(rows, c, ">=1B", shared=True) for c in ("E", "X")])
    assert len(shared) == 8 and (min(pct(shared["win"])), max(pct(shared["win"]))) == (41, 44)
    assert shared["excludes_zero"].all() and (shared["median"] < 0).all()
    for _, g in shared.groupby("corpus"):
        assert g.set_index("kappa")["hi_cl"].idxmax() == 100
    sep = pd.concat([_band(rows, c, ">=1B", shared=False) for c in ("E", "X")])
    assert sep["n_windows"].max() == 3


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
