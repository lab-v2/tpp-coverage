"""The released data downloads anonymously: no token, no login prompt."""

from __future__ import annotations

import os

import pytest

from conftest import ANALYSIS, run

pytestmark = pytest.mark.network


def test_no_forced_login():
    text = (ANALYSIS / "download_and_extract.py").read_text()
    assert "login(" not in text and "whoami" not in text


def test_results_download_without_token(tmp_path):
    """download_results() with an empty HuggingFace home and no token in the environment."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("HF_")}
    env.update(HF_HOME=str(tmp_path / "hf_home"), HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import download_and_extract as d; "
        "d.RESULTS_DIRS = ['prelim_bigtpp_results']; d.download_results(sys.argv[2])"
    )
    run(["python", "-c", code, ANALYSIS, tmp_path], env=env, timeout=600)
    table = tmp_path / "prelim_bigtpp_results" / "winrate_bigtpp_prelim.csv"
    assert table.exists() and table.stat().st_size > 0
