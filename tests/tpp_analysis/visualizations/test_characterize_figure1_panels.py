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

from characterize import FREEZE, FigureCapture, check_frozen, released
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
def data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
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
def test_figure1_panels(
    data_root: Path, monkeypatch: pytest.MonkeyPatch, prefix: str, dataset: str, law: str, mode: str
) -> None:
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


def test_figure1_poster_style_draws_the_paper_data(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FIG1_STYLE=poster changes only text sizes and gaps: every drawn series, label, limit and
    legend equals the paper render's frozen data (figure1_01.json), compared as arrays."""
    from matplotlib.figure import Figure

    for key, value in (
        ("PREFIX", "01_"),
        ("DATASET", "Red Pajama"),
        ("LAW", "Kaplan"),
        ("MODE", "First"),
    ):
        monkeypatch.setenv(f"FIG1_{key}", value)
    monkeypatch.setenv("FIG1_STYLE", "poster")
    monkeypatch.setenv("TPP_DATA_ROOT", str(data_root))
    saved: list[str] = []
    monkeypatch.setattr(Figure, "savefig", lambda self, path, **kw: saved.append(Path(path).name))
    with FigureCapture() as cap:
        runpy.run_path(str(ANALYSIS / "visualizations/figure1_panels.py"), run_name="__main__")
    if FREEZE:
        pytest.skip("the paper render is the reference; nothing to freeze here")
    observed = {"saved": saved, "figures": cap.figures}
    check_frozen(FROZEN / "figure1_01.json", observed)
    # The comparison can fail: another combo's frozen data is refused.
    with pytest.raises((AssertionError, pytest.fail.Exception)):
        check_frozen(FROZEN / "figure1_04.json", observed)
