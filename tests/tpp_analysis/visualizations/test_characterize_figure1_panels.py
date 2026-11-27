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


def legend_problems(fig, window_in=None, pad_cm: float = 0.2) -> list[str]:
    """R6 / M4-1: every legend frame lies inside its own axes and at least pad_cm inside the saved
    area (the whole canvas, or the crop window `window_in` in inches), so its frame is closed."""
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    dpi = fig.dpi
    if window_in is None:
        area = fig.bbox
    else:
        area = Bbox.from_extents(*(v * dpi for v in window_in.extents))
    pad = pad_cm / 2.54 * dpi
    out = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        b = leg.get_window_extent()
        if not (
            b.x0 >= ax.bbox.x0 and b.x1 <= ax.bbox.x1 and b.y0 >= ax.bbox.y0 and b.y1 <= ax.bbox.y1
        ):
            out.append(f"legend outside its axes: {b.extents} vs {ax.bbox.extents}")
        if not (
            b.x0 >= area.x0 + pad
            and b.x1 <= area.x1 - pad
            and b.y0 >= area.y0 + pad
            and b.y1 <= area.y1 - pad
        ):
            out.append(
                f"legend within {pad_cm} cm of the saved edge: {b.extents} vs {area.extents}"
            )
    return out


def test_poster_legends_inside(data_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    found: dict[str, list[str]] = {}

    def check(self, path, **kw):  # noqa: ANN001, ANN003, ANN202
        name = Path(path).name
        if name.endswith(".pdf") and ("surface" in name or "isoflop" in name):
            window = kw.get("bbox_inches")
            found[name] = legend_problems(self, window if window not in (None, "tight") else None)

    monkeypatch.setattr(Figure, "savefig", check)
    runpy.run_path(str(ANALYSIS / "visualizations/figure1_panels.py"), run_name="__main__")
    assert len(found) == 2 and all(v == [] for v in found.values()), found


def test_planted_legend_outside() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1], label="a")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))  # beside the axes
    assert legend_problems(fig)
    ax.legend(loc="upper left", borderaxespad=0)  # its frame on the axes edge: no 0.2 cm pad
    assert legend_problems(
        fig, window_in=ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    )
    ax.legend(loc="center")
    assert not legend_problems(fig)
    plt.close(fig)
