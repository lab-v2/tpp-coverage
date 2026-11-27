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
    """R6 / M4-1 / M5: every legend is at least pad_cm inside the saved area (the canvas, or the
    crop window `window_in` in inches), so its frame is closed, and clear of the plot by pad_cm: a
    2-D legend at least pad_cm inside its axes frame; a 3-D legend (whose axes rect is not a
    frame) over no other ink within pad_cm, on a render with the legend hidden."""
    import numpy as np
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    dpi = fig.dpi
    area = (
        fig.bbox if window_in is None else Bbox.from_extents(*(v * dpi for v in window_in.extents))
    )
    pad = pad_cm / 2.54 * dpi
    out = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        b = leg.get_window_extent()
        grown = Bbox.from_extents(b.x0 - pad, b.y0 - pad, b.x1 + pad, b.y1 + pad)
        if not (
            grown.x0 >= area.x0
            and grown.x1 <= area.x1
            and grown.y0 >= area.y0
            and grown.y1 <= area.y1
        ):
            out.append(
                f"legend within {pad_cm} cm of the saved edge: {b.extents} vs {area.extents}"
            )
        if ax.name != "3d":
            f = ax.bbox
            if not (
                grown.x0 >= f.x0 and grown.x1 <= f.x1 and grown.y0 >= f.y0 and grown.y1 <= f.y1
            ):
                out.append(
                    f"legend within {pad_cm} cm of its axes frame: {b.extents} vs {f.extents}"
                )
            continue
        leg.set_visible(False)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        leg.set_visible(True)
        h = rgba.shape[0]
        x0, x1 = int(max(grown.x0, 0)), int(min(grown.x1, rgba.shape[1]))
        y0, y1 = int(max(h - grown.y1, 0)), int(min(h - grown.y0, h))
        patch = rgba[y0:y1, x0:x1, :3].astype(int)
        face = (np.array(fig.get_facecolor()[:3]) * 255).astype(int)
        if (np.abs(patch - face).sum(axis=2) > 30).any():
            out.append(f"3-D legend within {pad_cm} cm of plot ink at {b.extents}")
    return out


FRAME_GAP_CM, FRAME_TOL = 0.30, 0.05
CBAR_GAP_CM, CBAR_TOL = 1.0, 0.2


def frame_problems(fig, window_in=None) -> list[str]:
    """Review M8: each poster legend's frame sits 0.30 +- 0.05 cm from the edges it is set in (the
    two spines of its 2-D axes corner; the window edges of the 3-D panel), and a colour bar sits
    1.0 +- 0.2 cm right of its axes."""
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    cm = fig.dpi / 2.54
    area = (
        fig.bbox
        if window_in is None
        else Bbox.from_extents(*(v * fig.dpi for v in window_in.extents))
    )
    out = []
    main = [a for a in fig.axes if a.get_legend() is not None]
    for ax in main:
        f = ax.get_legend().get_frame().get_window_extent()
        box = area if ax.name == "3d" else ax.bbox
        gaps = {
            "left": f.x0 - box.x0,
            "right": box.x1 - f.x1,
            "bottom": f.y0 - box.y0,
            "top": box.y1 - f.y1,
        }
        near = sorted(gaps.items(), key=lambda kv: kv[1])[:2]  # the corner it sits in
        out += [
            f"legend frame {side} gap {g / cm:.2f} cm, not {FRAME_GAP_CM} +- {FRAME_TOL}"
            for side, g in near
            if abs(g / cm - FRAME_GAP_CM) > FRAME_TOL
        ]
        for cb in fig.axes:
            if cb is ax or cb.get_legend() is not None or cb.name == "3d":
                continue
            gap = (cb.bbox.x0 - ax.bbox.x1) / cm
            if 0 < gap < 10 and abs(gap - CBAR_GAP_CM) > CBAR_TOL:
                out.append(
                    f"colour bar {gap:.2f} cm from its axes, not {CBAR_GAP_CM} +- {CBAR_TOL}"
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
            win = window if window not in (None, "tight") else None
            found[name] = legend_problems(self, win) + frame_problems(self, win)

    monkeypatch.setattr(Figure, "savefig", check)
    runpy.run_path(str(ANALYSIS / "visualizations/figure1_panels.py"), run_name="__main__")
    assert len(found) == 2 and all(v == [] for v in found.values()), found


def test_planted_frame_and_colour_bar() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 12))
    sc = ax.scatter([0, 1], [0, 1], c=[0, 1], label="a")
    ax.legend(loc="upper right", borderaxespad=0.1)  # the M8 Fig 1c corner: jammed into the spines
    fig.colorbar(sc, ax=ax, pad=0.15)  # detached colour bar
    found = frame_problems(fig)
    assert any("legend frame" in f for f in found) and any("colour bar" in f for f in found), found
    plt.close(fig)


def test_planted_legend_outside() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1], label="a")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))  # beside the axes
    assert legend_problems(fig)
    ax.legend(loc="upper left", borderaxespad=0)  # on the axes frame: no 0.2 cm clearance
    assert legend_problems(fig)
    ax.legend(loc="center")
    assert not legend_problems(fig)
    fig3 = plt.figure(figsize=(6, 5))
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.plot([0, 1], [0, 1], [0, 1], label="a")
    ax3.legend(loc="center")  # over the 3-D box
    assert any("3-D" in p for p in legend_problems(fig3))
    plt.close(fig)
    plt.close(fig3)
