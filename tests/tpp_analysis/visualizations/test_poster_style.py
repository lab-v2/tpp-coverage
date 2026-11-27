"""The opt-in poster style of `panel_style.py`: what it sets, and that nothing in the paper path uses it.

The paper's figures stay frozen by the characterization tests beside this file; here the poster
helpers are checked on their own, and each check is shown to refuse a planted defect.
"""

from __future__ import annotations

import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from conftest import ANALYSIS  # noqa: E402

sys.path.insert(0, str(ANALYSIS / "visualizations"))
import panel_style  # noqa: E402

VIS = ANALYSIS / "visualizations"


def rc_problems(rc: dict[str, object]) -> list[str]:
    problems = []
    if rc.get("pdf.fonttype") != 42:
        problems.append("pdf.fonttype is not 42 (TrueType)")
    if rc.get("font.family") != "Lato":
        problems.append("text font is not Lato")
    return problems


def test_poster_rcparams_embed_truetype_lato() -> None:
    assert rc_problems(panel_style.poster_rcparams()) == []


def test_planted_rcparams() -> None:
    rc = panel_style.poster_rcparams()
    assert rc_problems({**rc, "pdf.fonttype": 3})
    assert rc_problems({**rc, "font.family": "DejaVu Sans"})


def test_style_ticks_poster_sets_white_and_print_sizes() -> None:
    fig, ax = plt.subplots()
    panel_style.style_ticks_2d(ax)
    assert ax.get_facecolor()[:3] != (1.0, 1.0, 1.0)  # the paper's grey, before the poster pass
    panel_style.style_ticks_poster(ax)
    assert ax.get_facecolor()[:3] == (1.0, 1.0, 1.0)
    size = ax.xaxis.get_major_ticks()[0].label1.get_fontsize()
    assert size == panel_style.POSTER_TICK_LABEL_FS > panel_style.TICK_LABEL_FS
    plt.close(fig)


def test_paper_path_never_calls_the_poster_style() -> None:
    """Outside `if POSTER:` and the opt-in flag, the scripts do not touch the poster helpers."""
    src = (VIS / "figure1_panels.py").read_text()
    calls = [m.start() for m in re.finditer(r"style_ticks_poster\(|poster_rcparams\(", src)]
    assert calls
    for pos in calls:
        line_start = src.rfind("\n", 0, pos) + 1
        indent = pos - line_start
        guard = src.rfind("if POSTER:", 0, pos)
        assert guard != -1 and indent >= 4, src[line_start : pos + 30]
    assert "budget_colorbar: bool = False" in (VIS / "isoflop_panel.py").read_text()
