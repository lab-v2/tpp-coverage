"""A figure on the page sets its text at the page's body size, measured from the image.

Each figure's CSS width scales its PNG. The test measures the height of the x tick labels' digits
in the PNG itself: they are the first short run of dark rows below the plot. It then checks that
those digits, at that scale, are within 20% of the body text's digit height. Bulma's body text is
1rem (16 px), and a Noto Sans digit stands 0.714 em. An 896 px figure set its tick labels at 2.5
times the body's height (2026-11-26).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
PAGE = (REPO / "website" / "index.html").read_text()
BODY_PX = 16.0  # Bulma's body font size, 1rem
DIGIT_EM = 0.714  # Noto Sans digit (cap) height per em
BAND = (0.8, 1.25)


def css_width(page: str) -> float | None:
    """The px width that `.content figure img` sets, or None."""
    rule = re.search(r"\.content figure img \{([^}]*)\}", page)
    width = re.search(r"(?<![-\w])width:\s*(\d+(?:\.\d+)?)px", rule[1]) if rule else None
    return float(width[1]) if width else None


def tick_digit_height(png: Path) -> int:
    """Height in image pixels of the x tick labels: the first short dark run below the plot."""
    dark = np.asarray(Image.open(png).convert("L")) < 100
    h = dark.shape[0]
    rows = dark.any(axis=1)
    runs, start = [], None
    for y in range(h // 2, h):
        if rows[y] and start is None:
            start = y
        elif not rows[y] and start is not None:
            runs.append(y - start)
            start = None
    short = [r for r in runs if r < 0.05 * h]
    assert short, f"{png.name}: no tick-label row found"
    return short[0]


def problems(page: str, repo: Path = REPO) -> list[str]:
    out: list[str] = []
    width = css_width(page)
    srcs = re.findall(r'<figure><img src="([^"]+\.png)"', page)
    if srcs and width is None:
        return ["`.content figure img` sets no px width"]
    for src in srcs:
        png = repo / "website" / src
        scale = width / Image.open(png).width
        ratio = tick_digit_height(png) * scale / (DIGIT_EM * BODY_PX)
        if not BAND[0] <= ratio <= BAND[1]:
            out.append(f"{src}: tick labels at {ratio:.2f} x body text (want {BAND[0]}-{BAND[1]})")
    return out


def test_figure_text_is_body_size() -> None:
    assert problems(PAGE) == []


@pytest.mark.parametrize(
    "what,old,new",
    [
        ("figure too wide", "width: 480px", "width: 896px"),
        ("figure too narrow", "width: 480px", "width: 300px"),
        ("no width", "width: 480px; ", ""),
    ],
)
def test_planted_sizes_fail(what: str, old: str, new: str) -> None:
    assert old in PAGE, f"planted change {what!r} no longer applies"
    assert problems(PAGE.replace(old, new, 1)), what
