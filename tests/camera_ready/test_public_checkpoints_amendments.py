"""The amendment and scale numbers of the public-checkpoint appendix.

Three layers, all offline, as in test_public_checkpoints_intervals.py:

1. The committed outputs of camera_ready/public_checkpoints/amendments/ carry the values the
   appendix prints; every expected value is a literal copied from the appendix text, and
   where two scripts compute the same quantity their outputs agree.
2. Each script runs on the committed seeds 0-7 slice of the E-corpus enumeration and agrees
   with a direct computation from that slice.
3. With the full 512-seed tables in the work directory, each script reproduces its committed
   output; without them these tests skip and name the file they need.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from conftest import A2, assert_same_rows, run

sys.path.insert(0, str(A2))
from paths import WORK  # noqa: E402

RESULTS = A2 / "results"
SCRIPTS = A2 / "amendments"
SLICE = RESULTS / "results_E_seeds512_seeds0-7.csv.gz"
FULL_E = WORK / "results_E_seeds512.csv"
FULL_X = WORK / "results_X_seeds512.csv.gz"
WINDOWS = [WORK / f"results_E_win{v}.csv" for v in ("V1", "V2", "V3", "V4", "V7")]


def rounded(row: pd.Series, cols=("pct", "pct_lo", "pct_hi")) -> list[int]:
    return [round(row[c]) for c in cols]


# ── 1. the committed outputs carry the printed values ──────────────────────────────────────

@pytest.fixture(scope="module")
def a5() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "amendment5_1b.csv").set_index(["corpus", "spec", "arm"])


def test_amendment5_as_printed(a5):
    pre = "pre-registered (W1, window cluster)"
    centroid = "A5.4c smooth linear, at the >=1B centroid"
    e = a5.loc["E"]
    assert rounded(e.loc[(pre, "NC")]) == [28, -9, 41]
    assert rounded(e.loc[(pre, "NCM")]) == [36, 2, 54]  # the second non-collinear subset
    v4 = e.loc[("A5.3 V4 (5,1), 128 seeds, moving block L=5", "NC")]
    assert rounded(v4) == [26, 2, 47]
    smooth = e.loc[(centroid, "NC")]
    assert rounded(smooth) == [29, 5, 50] and round(smooth["pct_one_sided_lo"]) == 9
    assert round(smooth["N_at"] / 1e9, 1) == 1.8
    # disjoint windows: two intervals contain zero, the one with 4 windows is degenerate
    disjoint = [e.loc[(f"A5.3 {v}, 128 seeds, window cluster", "NC")]
                for v in ("V2 (8,8)", "V3 (5,5)", "V7 (4,4)")]
    assert [int(r["n_windows"]) for r in disjoint] == [4, 6, 8]
    assert disjoint[0]["lo"] == pytest.approx(disjoint[0]["beta"], abs=1e-12)
    assert [bool(r["excludes0"]) for r in disjoint[1:]] == [False, False]
    # attenuation: the 95th percentile of the share lost, unpooled and smooth
    assert round(100 * e.loc[("A5.4d attenuation fraction, unpooled", "NC"), "pct_one_sided_lo"]) == 112
    assert round(100 * e.loc[("A5.4d attenuation fraction, smooth", "NC"), "pct_one_sided_lo"]) == 77
    x = a5.loc["X"]
    assert rounded(x.loc[(pre, "NCM")]) == [40, 17, 61]
    assert rounded(x.loc[(centroid, "NC")]) == [-41, -64, -10]
    assert round(x.loc[(centroid, "NC"), "N_at"] / 1e9, 1) == 1.9


def test_amendment5_agrees_with_the_estimates(a5):
    """The pre-registered row and the b1 rows are the estimates_seeds512.csv rows, recomputed."""
    est = pd.read_csv(RESULTS / "estimates_seeds512.csv")
    for corpus in ("E", "X"):
        band = est[(est["corpus"] == corpus) & (est["spec"] == "stratum >=1B")
                   & (est["arm"] == "NC")].iloc[0]
        row = a5.loc[(corpus, "pre-registered (W1, window cluster)", "NC")]
        np.testing.assert_allclose(row[["beta", "lo", "hi"]].astype(float),
                                   band[["beta_512", "lo_512", "hi_512"]].astype(float), rtol=1e-9)
        for arm in ("NC", "NCM"):
            b1 = est[(est["corpus"] == corpus) & (est["spec"] == "b1 Delta x logN")
                     & (est["arm"] == arm)].iloc[0]
            row = a5.loc[(corpus, "A5.4c smooth linear, interaction b1", arm)]
            assert row["beta"] == pytest.approx(b1["beta_512"], rel=1e-9)
    # the second subset's own size interaction on the ten, -25% [-43%, -8%]
    b1 = est[(est["corpus"] == "X") & (est["spec"] == "b1 Delta x logN") & (est["arm"] == "NCM")]
    assert [round(b1[c].iloc[0]) for c in ("pct_512", "pct_lo_512", "pct_hi_512")] == [-25, -43, -8]


def test_size_interaction_as_printed():
    ic = pd.read_csv(RESULTS / "interaction_check.csv").set_index(["corpus", "spec"])
    centre = ic.xs("b0, at the centre", level="spec")
    assert round(centre.loc["E", "N_centre"] / 1e7) * 10 == 630  # 630M on the five
    assert round(centre.loc["X", "N_centre"] / 1e6) == 152  # 152M on the ten
    assert round(centre.loc["E", "pct"]) == 38
    up = ic.loc[("E", "b0 + b1, one decade above the centre")]
    assert round(up["N_at"] / 1e8) / 10 == 6.3 and rounded(up) == [19, -19, 62]
    # b0 is the b0 Delta row of the estimates
    est = pd.read_csv(RESULTS / "estimates_seeds512.csv")
    b0 = est[(est["spec"] == "b0 Delta") & (est["arm"] == "NC")].set_index("corpus")["beta_512"]
    np.testing.assert_allclose(centre["beta"], b0.loc[centre.index], rtol=1e-9)


def test_three_study_estimate_as_printed():
    t = pd.read_csv(RESULTS / "three_study_pooled.csv").set_index(["seeds", "spec"])
    assert rounded(t.loc[(5, "pooled, no fixed effect")]) == [8, -1, 16]
    assert rounded(t.loc[(512, "pooled, no fixed effect")]) == [6, -4, 15]
    assert rounded(t.loc[(512, "cell FE (primary)")]) == [33, 17, 43]
    assert (t["n_cells"] == 5).all()  # Farseer, Chinchilla and Gadre et al.'s three corpora
    per = pd.read_csv(RESULTS / "per_study.csv").set_index("study")["runs_used"]
    assert per[["Farseer", "Chinchilla isoFLOP", "Gadre et al."]].sum() == 753
    assert per[["Gemstones annealed", "OLMo ladder"]].sum() == 231


# ── 2. each script on the committed slice, against a direct computation ────────────────────

def test_three_study_pooled_on_slice(tmp_path):
    run(["python", SCRIPTS / "three_study_pooled.py", "--results", SLICE,
         "--out", tmp_path / "t.csv"], cwd=A2)
    out = pd.read_csv(tmp_path / "t.csv").set_index(["seeds", "spec"])
    d = pd.read_csv(SLICE)
    d = d[d["cell"].str.split("/").str[0].isin(["farseer", "chinchilla_isoflop", "gadre_overtrain"])]
    for seeds in (5, 512):
        sub = d[d["seed"] < seeds].copy()
        m = sub["CO_m_beta0.28"].replace(-np.inf, np.nan)
        x = sub["NC_m_beta0.28"] - m.fillna(m.min())  # A3.1: the floor of this analysis set
        y = sub["gap_offray"]
        ok = np.isfinite(x) & np.isfinite(y)
        slope = np.polyfit(x[ok], y[ok], 1)[0]  # no fixed effect: one OLS line
        assert out.loc[(seeds, "pooled, no fixed effect"), "beta"] == pytest.approx(slope, rel=1e-9)
        assert out.loc[(seeds, "pooled, no fixed effect"), "n_pairs"] == ok.sum()


def test_interaction_check_on_slice(tmp_path):
    run(["python", SCRIPTS / "interaction_check.py", "--skip-x", "--results", SLICE,
         "--out", tmp_path / "i.csv"], cwd=A2)
    out = pd.read_csv(tmp_path / "i.csv").set_index("spec")
    d = pd.read_csv(SLICE)
    logn = np.log10(d["N_gm"])
    assert out.loc["b0, at the centre", "N_centre"] == pytest.approx(10 ** logn.mean(), rel=1e-12)
    assert out.loc["b0 + b1, one decade above the centre", "N_at"] == \
        pytest.approx(10 ** (logn.mean() + 1), rel=1e-12)
    assert (out["n_pairs"] == len(d)).all()


def test_amendment5_on_slice(tmp_path):
    run(["python", SCRIPTS / "amendment5_1b.py", "--skip-x", "--windows-dir", "none",
         "--results", SLICE, "--out", tmp_path / "a.csv"], cwd=A2, timeout=3600)
    out = pd.read_csv(tmp_path / "a.csv").set_index(["spec", "arm"])
    d = pd.read_csv(SLICE)
    top = d["N_gm"] >= 1e9
    assert out.loc[("pre-registered (W1, window cluster)", "NC"), "n_pairs"] == top.sum()
    assert out.loc[("pre-registered (W1, window cluster)", "NC"), "n_windows"] == \
        d[top].groupby(["cell", "window"]).ngroups
    logn = np.log10(d["N_gm"])
    assert out.loc[("A5.4c smooth linear, at the >=1B centroid", "NC"), "N_at"] == \
        pytest.approx(10 ** logn[top].mean(), rel=1e-12)
    # the slope at log10 N = 9 lies between the readings either side of it
    lin = [out.loc[(f"A5.4c smooth linear, at log10 N = {lg}", "NC"), "beta"] for lg in (9.0, 9.5, 10.0)]
    assert lin[1] == pytest.approx((lin[0] + lin[2]) / 2, rel=1e-9)  # linear in log N


# ── 3. with the full tables, each script gives back its committed output ───────────────────

def _needs(*paths):
    missing = [str(p) for p in paths if not p.exists()]
    return pytest.mark.skipif(bool(missing), reason=f"needs {', '.join(missing)} "
                              "(written by enumerate_seeds.py; not shipped)")


@pytest.mark.parametrize("script,out,keys,args", [
    ("three_study_pooled.py", "three_study_pooled.csv", ["seeds", "spec"], []),
    ("interaction_check.py", "interaction_check.csv", ["corpus", "spec"], []),
])
@_needs(FULL_E, FULL_X)
def test_full_small_scripts_reproduce(tmp_path, script, out, keys, args):
    run(["python", SCRIPTS / script, "--out", tmp_path / out, *args], cwd=A2)
    assert_same_rows(pd.read_csv(tmp_path / out), pd.read_csv(RESULTS / out), keys)


@_needs(FULL_E, FULL_X, *WINDOWS)
def test_full_amendment5_reproduces(tmp_path):
    run(["python", SCRIPTS / "amendment5_1b.py", "--out", tmp_path / "a.csv"], cwd=A2,
        timeout=3600)
    fresh, ref = pd.read_csv(tmp_path / "a.csv"), pd.read_csv(RESULTS / "amendment5_1b.csv")
    assert_same_rows(fresh, ref, ["corpus", "spec", "arm"])
