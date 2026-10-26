"""The job generator emits exactly the runs that were trained and released on HuggingFace."""

from __future__ import annotations

import re
from functools import lru_cache

import pytest

from conftest import REPO, run

pytestmark = pytest.mark.network
GEN = REPO / "src/tpp_analysis/generate_jobs/generate_jobs.py"


@lru_cache(maxsize=1)
def _released_names() -> dict[str, set[str]]:
    from huggingface_hub import HfApi

    from hf_dataset import REPO_ID

    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    out: dict[str, set[str]] = {}
    for f in files:
        m = re.match(r"training_metrics/([a-z0-9_]+)/[a-z_]+/N_\d+/(.+?)_[a-z0-9]+_m\d+_token", f)
        if m and f.endswith("_metrics.npz"):
            out.setdefault(m.group(1), set()).add(_canonical(m.group(2)))
    return out


def _canonical(name: str) -> str:
    # The Wikipedia and RedPajama runs were named TPP3, the other corpora TPP3.0: same run.
    return re.sub(r"_TPP([\d.]+)_", lambda m: f"_TPP{float(m.group(1)):g}_", name)


def _generated(*args: str) -> set[str]:
    out = run(["python", GEN, "--list", *args]).stdout.splitlines()
    return {_canonical(line.split()[-1]) for line in out}


@pytest.mark.parametrize("dataset", ["wikipedia", "cosmopedia", "redpajama", "pes2o"])
def test_main_design_matches_release(dataset):
    assert _generated("--dataset", dataset) == _released_names()[dataset]


def test_c4_release_is_the_design_minus_three_runs():
    generated, released = _generated("--dataset", "c4"), _released_names()["c4"]
    assert released <= generated
    assert len(generated - released) == 3  # three NC holdout runs were never released


def test_bigtpp_matches_release():
    assert _generated("--dataset", "wikipedia", "--design", "bigtpp") == \
        _released_names()["wikipedia_bigtpp"]


def test_bf16_release_is_a_subset_of_the_design():
    generated = _generated("--dataset", "wikipedia", "--precision", "bf16")
    released = _released_names()["wikipedia_bf16"]
    assert released <= generated and len(released) >= 300
