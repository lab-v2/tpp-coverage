"""Grid gates of the poster makeover (reviewer audit M0, gates G1-G4, G8, G10, G15), measured on the
built PDF with PyMuPDF. Each gate has a planted failure that must be caught."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

PDF = Path(__file__).resolve().parents[2] / "poster" / "tpp_neurips2026_poster.pdf"
CM = 2.54 / 72
EDGES = (3.0, 35.0, 37.0, 69.0, 71.0, 103.0, 105.0, 137.0)
AXIS = 70.0
TYPE_SCALE = (110.0, 90.0, 44.0, 30.0, 24.0)
BODY_TOP, BODY_BOTTOM = 15.2, 83.6
Rect = tuple[float, float, float, float]


def cards(pdf: Path = PDF) -> list[Rect]:
    """Filled rectangles inside the body zone at least 30 x 5 cm: the block cards (figure axes are narrower)."""
    page = fitz.open(pdf)[0]
    out = set()
    for d in page.get_drawings():
        r = d["rect"]
        x0, y0, x1, y1 = (round(v * CM, 2) for v in (r.x0, r.y0, r.x1, r.y1))
        if (
            d.get("fill")
            and x1 - x0 >= 30
            and y1 - y0 >= 5
            and y0 >= BODY_TOP - 0.1
            and y1 <= BODY_BOTTOM + 0.1
        ):
            if d["fill"] != (1.0, 1.0, 1.0):  # white figure canvases are not cards
                out.add((x0, y0, x1, y1))
    return sorted(out)


def rules(pdf: Path = PDF) -> list[Rect]:
    page = fitz.open(pdf)[0]
    return [
        tuple(round(v * CM, 2) for v in (d["rect"].x0, d["rect"].y0, d["rect"].x1, d["rect"].y1))
        for d in page.get_drawings()
        if d.get("fill") and d["rect"].width * CM > 100 and d["rect"].height * CM < 0.5
    ]


def edge_problems(cs: list[Rect], tol: float = 0.10) -> list[str]:  # G1
    return [
        f"card {c} x edge off grid"
        for c in cs
        for x in (c[0], c[2])
        if min(abs(x - e) for e in EDGES) > tol
    ]


def mirror_problems(cs: list[Rect], tol: float = 0.10) -> list[str]:  # G2 + G3
    spans = {(round(c[0], 1), round(c[2], 1)) for c in cs}
    mirrored = {(round(2 * AXIS - b, 1), round(2 * AXIS - a, 1)) for a, b in spans}
    bad = [
        s
        for s in spans
        if not any(abs(s[0] - m[0]) <= tol and abs(s[1] - m[1]) <= tol for m in mirrored)
    ]
    widths = {round(EDGES[i + 1] - EDGES[i], 2) for i in range(0, 8, 2)}
    gutters = {round(EDGES[i + 1] - EDGES[i], 2) for i in range(1, 7, 2)}
    out = [f"span {s} has no mirror about x = {AXIS}" for s in bad]
    if len(widths) != 1 or len(gutters) != 1 or EDGES[0] != 140 - EDGES[-1]:
        out.append("grid constants not equal")
    return out


def bottom_problems(cs: list[Rect], tol: float = 0.20) -> list[str]:  # G4
    tops = {c[1] for c in cs}
    out = [] if min(tops) >= BODY_TOP - tol else ["a card starts above the body"]
    last = {}
    for c in cs:
        key = (c[0], c[2])
        last[key] = max(last.get(key, 0.0), c[3])
    columns = {k: v for k, v in last.items() if k in {(3.0, 35.0), (37.0, 103.0), (105.0, 137.0)}}
    if len(columns) != 3:
        out.append(f"expected c1, centre and c4 columns, found {sorted(last)}")
    out += [
        f"column {k} ends at {v}, not {BODY_BOTTOM}"
        for k, v in columns.items()
        if abs(v - BODY_BOTTOM) > tol
    ]
    first = [c[1] for c in cs if abs(c[1] - BODY_TOP) <= tol]
    if len(first) < 4:
        out.append(f"only {len(first)} cards start at the body top")
    return out


def gap_problems(cs: list[Rect], tol: float = 0.10) -> list[str]:  # G15
    out = []
    for key in {(c[0], c[2]) for c in cs}:
        col = sorted(c for c in cs if (c[0], c[2]) == key)
        for a, b in zip(col, col[1:]):
            if abs(b[1] - a[3] - 1.0) > tol:
                out.append(f"gap {b[1] - a[3]:.2f} in column {key}")
    centre_top = [c for c in cs if c[0] in (37.0, 71.0) and c[2] - c[0] < 40]
    fig = [c for c in cs if (c[0], c[2]) == (37.0, 103.0)]
    out += [
        f"centre gap {f[1] - t[3]:.2f}"
        for t in centre_top
        for f in fig
        if abs(f[1] - t[3] - 1.0) > tol
    ]
    return out


def rule_problems(rs: list[Rect], tol: float = 0.10) -> list[str]:  # G8
    if len(rs) != 2:
        return [f"expected header and footer rules, found {rs}"]
    return [f"rule {r} not 3-137" for r in rs if abs(r[0] - 3.0) > tol or abs(r[2] - 137.0) > tol]


def size_problems(
    pdf: Path = PDF, tol: float = 0.5
) -> list[str]:  # G10, text above the figures' own fonts
    page = fitz.open(pdf)[0]
    sizes = set()
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans or any(s["font"].startswith("Cm") for s in spans):
                continue  # figure text (Computer Modern, matplotlib) has its own gate
            sizes |= {round(s["size"], 1) for s in spans}
    # Math scripts are excluded: Latin Modern Math sets them at 0.7 and 0.5 of their base size.
    allowed = [f * t for t in TYPE_SCALE for f in (1.0, 0.7, 0.5)]
    return [
        f"size {s} pt off the type scale"
        for s in sorted(sizes)
        if min(abs(s - a) for a in allowed) > tol
    ]


def test_cards_on_column_edges() -> None:
    assert edge_problems(cards()) == []


def test_layout_mirrors_about_the_axis() -> None:
    assert mirror_problems(cards()) == []


def test_columns_end_on_one_line() -> None:
    assert bottom_problems(cards()) == []


def test_card_gaps_are_one_cm() -> None:
    assert gap_problems(cards()) == []


def test_rules_span_the_grid() -> None:
    assert rule_problems(rules()) == []


def test_type_scale() -> None:
    assert size_problems() == []


@pytest.mark.parametrize(
    "check, planted",
    [
        (edge_problems, [(3.4, 15.2, 35.0, 30.0)]),
        (mirror_problems, [(3.0, 15.2, 35.0, 30.0), (100.0, 15.2, 137.0, 30.0)]),
        (
            bottom_problems,
            [(3.0, 15.2, 35.0, 80.0), (37.0, 15.2, 103.0, 83.6), (105.0, 15.2, 137.0, 83.6)],
        ),
        (gap_problems, [(3.0, 15.2, 35.0, 30.0), (3.0, 32.5, 35.0, 83.6)]),
        (rule_problems, [(1.5, 13.0, 138.5, 13.2), (3.0, 85.6, 137.0, 85.8)]),
    ],
)
def test_planted_grid_failures(check, planted) -> None:  # noqa: ANN001
    assert check(planted) != []


def test_planted_off_scale_size(tmp_path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=140 / CM, height=90 / CM)
    page.insert_text((100, 100), "off scale", fontsize=34.5)
    doc.save(tmp_path / "p.pdf")
    assert size_problems(tmp_path / "p.pdf") != []
