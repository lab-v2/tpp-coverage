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
    assert verdicts == {"pass": 13, "fail": 11}
    assert int(((corpus["verdict_at_own_beta"] == "fail") & (corpus["K"] == 1)).sum()) == 3
    candidates = pd.read_csv(A1 / "candidates.csv")
    assert len(candidates) == 206
