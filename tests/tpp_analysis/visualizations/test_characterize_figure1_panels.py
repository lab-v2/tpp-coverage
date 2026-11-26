"""Characterization of `visualizations/`: the Figure 1 panel script and the helpers it draws with.

`figure1_panels.py` runs in-process (as `run_appendix_figures.sh` runs it) on a data root that
links the released loss tables, full-coverage fits and seed-variance results, for combos that
cover each law and mode. Expected values are the base commit's figure data, frozen in `frozen/`
(see `tests/characterize.py`); saving the figures is recorded, not rendered.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from characterize import FigureCapture, check_frozen, released
from conftest import ANALYSIS

pytestmark = pytest.mark.network
FROZEN = Path(__file__).parent / "frozen"
COMBOS = [
    ("01", "Red Pajama", "Kaplan", "First"),
    ("04", "Wikipedia", "Kaplan", "Final"),
    ("06", "C4", "Chinchilla", "First"),
    ("07", "Wikipedia", "Droppo-Elibol", "First"),
    ("13", "Red Pajama", "Kaplan", "Second"),
]


@pytest.fixture(scope="module")
def data_root(tmp_path_factory) -> Path:
    """A TPP_DATA_ROOT whose inputs link the released files the script reads."""
    src = released(
        "extracted_losses/*.csv",
        "full_coverage_results/full_coverage_all_seeds.parquet",
        "seed_variance_results/*.parquet",
    )
    root = tmp_path_factory.mktemp("data_root")
    for sub in ("extracted_losses", "full_coverage_results", "seed_variance_results"):
        (root / sub).mkdir()
        for f in (src / sub).iterdir():
            (root / sub / f.name).symlink_to(f)
    return root


@pytest.mark.parametrize("prefix, dataset, law, mode", COMBOS)
def test_figure1_panels(data_root, monkeypatch, prefix, dataset, law, mode):
    from matplotlib.figure import Figure

    for key, value in (
        ("PREFIX", f"{prefix}_"),
        ("DATASET", dataset),
        ("LAW", law),
        ("MODE", mode),
    ):
        monkeypatch.setenv(f"FIG1_{key}", value)
    monkeypatch.setenv("TPP_DATA_ROOT", str(data_root))
    saved: list[str] = []
    monkeypatch.setattr(Figure, "savefig", lambda self, path, **kw: saved.append(Path(path).name))
    with FigureCapture() as cap:
        runpy.run_path(str(ANALYSIS / "visualizations/figure1_panels.py"), run_name="__main__")
    check_frozen(FROZEN / f"figure1_{prefix}.json", {"saved": saved, "figures": cap.figures})
