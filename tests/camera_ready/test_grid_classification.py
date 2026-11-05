"""Camera-ready A1: the 24-design Proposition-2 classification reproduces its committed table."""

from __future__ import annotations

import pandas as pd

from conftest import A1, run


def test_classification_reproduces_results_csv(tmp_path):
    out = tmp_path / "results.csv"
    proc = run(["python", A1 / "classify.py", "--out", out], cwd=A1)
    assert out.read_bytes() == (A1 / "results.csv").read_bytes()
    assert "included (corpus)   : 24" in proc.stdout


def test_headline_counts():
    df = pd.read_csv(A1 / "results.csv")
    corpus = df[~df["is_calibration"]]
    assert len(corpus) == 24
    verdicts = corpus["verdict_at_own_beta"].value_counts().to_dict()
    assert verdicts == {"pass": 11, "fail": 13}
    assert corpus["verdict_at_0.28"].value_counts().to_dict() == {"pass": 14, "fail": 10}
    assert int(((corpus["verdict_at_own_beta"] == "fail") & (corpus["K"] == 1)).sum()) == 3
    candidates = pd.read_csv(A1 / "candidates.csv")
    assert len(candidates) == 206


def test_rows_read_from_run_tables():
    """The designs of Amendments 4 and 5, as the paper's appendix prints them."""
    df = pd.read_csv(A1 / "results.csv").set_index("key")
    margin = df["V_K_own_beta"] / df["tau_K_own_beta"]
    printed = {"krajewski2024": (16, 0.59, "fail"), "shukor2025mixtures": (7, 0.61, "fail"),
               "aghajanyan2023": (21, 3.42, "pass"), "kaplan2020": (34, 2.07, "pass")}
    for key, (k, m, verdict) in printed.items():
        assert (df.loc[key, "K"], round(margin[key], 2), df.loc[key, "verdict_at_own_beta"]) == \
            (k, m, verdict), key
    # Kaplan's row is the 38 runs read from its figure: kappa = 100 tau_K / V_K = 48
    assert round(100 / margin["kaplan2020"]) == 48
    passes = df[(df["verdict_at_own_beta"] == "pass") & ~df["is_calibration"]]
    assert not (passes["k_list_source"] == "endpoints").any()
