"""Layout gates of the poster makeover, measured on the built PDF with PyMuPDF: the grid (audit M0,
gates G1-G4, G8, G10, G15) and review M1's R1-R11. Each gate has a planted failure that must be caught."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

PDF = Path(__file__).resolve().parents[2] / "poster" / "tpp_neurips2026_poster.pdf"
CM = 2.54 / 72
EDGES = (3.0, 46.333, 48.333, 91.667, 93.667, 137.0)
AXIS = 70.0
TYPE_SCALE = (110.0, 90.0, 44.0, 30.0, 24.0)
BODY_TOP, BODY_BOTTOM = 15.2, 83.6
Rect = tuple[float, float, float, float]
CARD_FILLS = ((243 / 255, 245 / 255, 249 / 255), (0.9098, 0.9569, 1.0))


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
            # Only the two card colours (cardplain, cardalert): figure panes are not cards.
            if any(all(abs(f - g) < 0.01 for f, g in zip(d["fill"], col)) for col in CARD_FILLS):
                out.add((x0, y0, x1, y1))
    return sorted(out)


def rules(pdf: Path = PDF) -> list[Rect]:
    page = fitz.open(pdf)[0]
    return [
        tuple(round(v * CM, 2) for v in (d["rect"].x0, d["rect"].y0, d["rect"].x1, d["rect"].y1))
        for d in page.get_drawings()
        if d.get("fill") and d["rect"].width * CM > 100 and d["rect"].height * CM < 0.5
    ]


def edge_problems(cs: list[Rect], tol: float = 0.05) -> list[str]:  # G1
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
    widths = {round(EDGES[i + 1] - EDGES[i], 2) for i in range(0, 6, 2)}
    gutters = {round(EDGES[i + 1] - EDGES[i], 2) for i in range(1, 5, 2)}
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
    columns = {k: v for k, v in last.items() if k in {(3.0, 46.33), (48.33, 91.67), (93.67, 137.0)}}
    if len(columns) != 3:
        out.append(f"expected C1, C2 and C3 columns, found {sorted(last)}")
    out += [
        f"column {k} ends at {v}, not {BODY_BOTTOM}"
        for k, v in columns.items()
        if abs(v - BODY_BOTTOM) > tol
    ]
    first = [c[1] for c in cs if abs(c[1] - BODY_TOP) <= tol]
    if len(first) < 3:
        out.append(f"only {len(first)} cards start at the body top")
    return out


def gap_problems(cs: list[Rect], tol: float = 0.10) -> list[str]:  # G15
    out = []
    for key in {(c[0], c[2]) for c in cs}:
        col = sorted(c for c in cs if (c[0], c[2]) == key)
        for a, b in zip(col, col[1:]):
            if abs(b[1] - a[3] - 1.0) > tol:
                out.append(f"gap {b[1] - a[3]:.2f} in column {key}")
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
    # Math scripts are allowed: Latin Modern Math and matplotlib's mathtext set them at 0.7 and
    # 0.5 of their base size. Figure text (Computer Modern) is checked by R4 below.
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
        (edge_problems, [(3.4, 15.2, 46.33, 30.0)]),
        (mirror_problems, [(3.0, 15.2, 46.33, 30.0), (100.0, 15.2, 137.0, 30.0)]),
        (
            bottom_problems,
            [(3.0, 15.2, 35.0, 80.0), (37.0, 15.2, 103.0, 83.6), (105.0, 15.2, 137.0, 83.6)],
        ),
        (gap_problems, [(3.0, 15.2, 46.33, 30.0), (3.0, 32.5, 46.33, 83.6)]),
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


# --- review M1: R2-R11 ------------------------------------------------------------------------------

PAD, MAX_SLACK, EMPTY_CM = 0.80, 3.0, 3.0
TEXT_W = EDGES[1] - EDGES[0] - 2 * PAD
Line = tuple[float, float, float, float, float, str, str]  # x0, y0, x1, y1, main size, font, text


def text_lines(pdf: Path = PDF) -> list[Line]:
    """Text lines (PyMuPDF), their main span's size and font, and their words: xelatex sets no
    space glyphs, so the words come from get_text("words"), keyed by block and line number."""
    page = fitz.open(pdf)[0]
    words: dict[tuple[int, int], list[str]] = {}
    for w in page.get_text("words"):
        words.setdefault((w[5], w[6]), []).append(w[4])
    out = []
    for bi, b in enumerate(page.get_text("dict")["blocks"]):
        for li, ln in enumerate(b.get("lines", [])):
            spans = [s for s in ln["spans"] if s["text"].strip()]
            if not spans:
                continue
            main = max(spans, key=lambda s: len(s["text"].strip()))
            x0, y0, x1, y1 = (v * CM for v in ln["bbox"])
            text = " ".join(words.get((bi, li), [])) or "".join(s["text"] for s in spans).strip()
            out.append((x0, y0, x1, y1, main["size"], main["font"], text))
    return out


def inside(line: Line, card: Rect) -> bool:
    return card[0] <= line[0] < card[2] and card[1] <= line[1] < card[3]


def slack_problems(cs: list[Rect], lines: list[Line]) -> list[str]:  # R2
    out = []
    for c in cs:
        low = max((ln[3] for ln in lines if inside(ln, c)), default=c[1])
        if c[3] - low - PAD > MAX_SLACK:
            out.append(f"card {c}: {c[3] - low - PAD:.2f} cm empty under its content")
    return out


def empty_band_problems(img, cs: list[Rect], dpi: float) -> list[str]:  # R3 (redefined, M3)
    """(a) a band >= 3 cm tall empty across a card's full text width; (c) an empty rectangle >= 3 cm
    tall and >= 40% of the text width wide inside the text area, where the space beside a single
    centred line (a title, a display, a table) does not count: a row counts as empty on a side only
    when its ink on that row is not centred on the card. (b), the figure slots, is gated by
    figure_extent_problems."""
    import numpy as np

    per_cm = dpi / 2.54
    out = []
    for c in cs:
        x0, x1 = (int(round((v) * per_cm)) for v in (c[0] + PAD, c[2] - PAD))
        y0, y1 = (int(round(v * per_cm)) for v in (c[1] + PAD, c[3] - PAD))
        sub = img[y0:y1, x0:x1].astype(int)
        bg = img[int(round((c[1] + 0.3) * per_cm)), int(round((c[0] + 0.3) * per_cm))].astype(int)
        ink = np.abs(sub - bg).sum(axis=2) >= 24
        rows = ink.any(axis=1)
        need = EMPTY_CM * per_cm
        run = best = 0
        for r in rows:
            run = 0 if r else run + 1
            best = max(best, run)
        if best >= need:
            out.append(f"card {c}: empty band {best / per_cm:.1f} cm tall across the text width")
            continue
        if (
            abs(c[0] - EDGES[2]) < 0.01
        ):  # the figure card: its slots are gated by figure_extent_problems
            continue
        width = ink.shape[1]
        free = np.zeros(ink.shape, dtype=bool)  # empty cells that are not beside a centred line
        for i, row in enumerate(ink):
            xs = np.nonzero(row)[0]
            if len(xs) == 0:
                free[i] = True
                continue
            centred = abs((xs[0] + xs[-1]) / 2 - width / 2) < 0.5 * per_cm
            if not centred:
                free[i] = ~row
                free[i, : xs[0]] = False  # text starts at the left edge: nothing free before it
        # Columns of free cells: the tallest run per column, then the widest span of columns
        # whose runs all reach 3 cm over a common window.
        wide = int(0.4 * width)
        for start in range(0, ink.shape[0] - int(need) + 1, max(1, int(per_cm // 2))):
            window = free[start : start + int(need)].all(axis=0)
            run = longest = 0
            for v in window:
                run = run + 1 if v else 0
                longest = max(longest, run)
            if longest >= wide:
                out.append(f"card {c}: empty {longest / per_cm:.1f} x {EMPTY_CM} cm area")
                break
    return out


def figure_inks(img, card: Rect, dpi: float) -> list[tuple[float, float, float, float]]:
    """Ink boxes (cm) of the figures in the figure card, split at the widest empty band."""
    import numpy as np

    per_cm = dpi / 2.54
    x0, x1 = int(card[0] * per_cm) + 2, int(card[2] * per_cm) - 2
    top, bottom = (
        int((card[1] + 3.0) * per_cm),
        int((card[3] - 4.0) * per_cm),
    )  # below the title, above the caption
    bg = img[int((card[1] + 0.3) * per_cm), int((card[0] + 0.3) * per_cm)].astype(int)
    ink = np.abs(img[top:bottom, x0:x1].astype(int) - bg).sum(axis=2) >= 24
    rows = np.nonzero(ink.any(axis=1))[0]
    gaps = np.diff(rows)
    k = int(np.argmax(gaps))
    out = []
    for a, b in ((rows[0], rows[k]), (rows[k + 1], rows[-1])):
        cols = np.nonzero(ink[a : b + 1].any(axis=0))[0]
        out.append(
            (
                (x0 + cols[0]) / per_cm,
                (top + a) / per_cm,
                (x0 + cols[-1]) / per_cm,
                (top + b) / per_cm,
            )
        )
    return out


def figure_extent_problems(inks: list, tol: float = 0.5) -> list[str]:  # R3 (b), R7, M3-2, M3-3
    left, right = EDGES[2] + PAD, EDGES[3] - PAD
    out = [
        f"figure ink {f[0]:.2f}-{f[2]:.2f}, not the text width {left:.2f}-{right:.2f}"
        for f in inks
        if abs(f[0] - left) > tol or abs(f[2] - right) > tol
    ]
    if len(inks) == 2 and abs(inks[1][1] - inks[0][3] - 1.0) > 0.3:
        out.append(f"gap between the figures {inks[1][1] - inks[0][3]:.2f} cm, not 1.0 +- 0.3")
    return out


def render(pdf: Path, dpi: int) -> object:
    import numpy as np

    pix = fitz.open(pdf)[0].get_pixmap(dpi=dpi)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[
        :, :, :3
    ]


def all_size_problems(lines_spans: list[float], tol: float = 0.5) -> list[str]:  # R4
    allowed = [f * t for t in (109.6, 89.7, 43.8, 29.9, 23.9) for f in (1.0, 0.7, 0.5)]
    return [
        f"{s:.1f} pt" for s in sorted(set(lines_spans)) if min(abs(s - a) for a in allowed) > tol
    ]


def span_sizes(pdf: Path = PDF) -> list[float]:
    page = fitz.open(pdf)[0]
    return [
        round(s["size"], 1)
        for b in page.get_text("dict")["blocks"]
        for ln in b.get("lines", [])
        for s in ln["spans"]
        if s["text"].strip()
    ]


def short_line_problems(cs: list[Rect], lines: list[Line]) -> list[str]:  # R8
    """A body line starting at a card's text edge narrower than 15% of the text width must hold
    at least 2 words (the end of a paragraph), never a lone orphan word."""
    out = []
    for ln in lines:
        card = next((c for c in cs if inside(ln, c)), None)
        if card is None or ln[5].startswith("Cm") or abs(ln[4] - 29.9) > 0.5:
            continue
        if ln[0] - card[0] > PAD + 1.5:  # centred tables and displays are not running text
            continue
        if ln[2] - ln[0] < 0.15 * TEXT_W and len(ln[6].split()) < 2:
            out.append(f"orphan {ln[6]!r} at y {ln[1]:.2f}")
    return out


MARKER = ("\u2022", "1.", "2.", "3.")


def inset_problems(cs: list[Rect], lines: list[Line], tol: float = 0.10) -> list[str]:  # R9
    out = []
    for c in cs:
        mine = [ln for ln in lines if inside(ln, c) and not ln[5].startswith("Cm")]
        lists = any(ln[6].startswith(MARKER) for ln in mine)
        for ln in mine:
            off = ln[0] - c[0]
            if off > PAD + 2.0 or abs(off - PAD) <= tol:
                continue
            if lists and PAD < off <= PAD + 1.5:  # list markers and their hanging indent
                continue
            out.append(f"line {ln[6][:20]!r} starts {off:.2f} into its card")
    return out


def margin_problems(img, dpi: float) -> list[str]:  # R10
    import numpy as np

    per_cm = dpi / 2.54
    dark = img.astype(int).sum(axis=2) < 600
    out = []
    foot = np.nonzero(dark[int(85.9 * per_cm) :].any(axis=1))[0]
    lowest = 85.9 + foot[-1] / per_cm if len(foot) else 0.0
    if 90.0 - lowest < 1.0 - 0.10:
        out.append(f"footer ink {90.0 - lowest:.2f} cm from the trim")
    head = dark[: int(12.9 * per_cm), int(25 * per_cm) : int(115 * per_cm)]
    rows = np.nonzero(head.any(axis=1))[0]
    top, bottom = rows[0] / per_cm, rows[-1] / per_cm
    if abs(top - (13.0 - bottom)) > 0.30:
        out.append(f"header ink {top:.2f} below the trim, {13.0 - bottom:.2f} above the rule")
    return out


def title_wrap_problems(cs: list[Rect], lines: list[Line]) -> list[str]:  # R11
    out = []
    for c in cs:
        mine = sorted((ln for ln in lines if inside(ln, c)), key=lambda ln: ln[1])
        if len(mine) >= 2 and abs(mine[0][4] - 43.8) < 0.5 and abs(mine[1][4] - 43.8) < 0.5:
            if mine[1][1] - mine[0][1] < 2.2 and mine[1][0] > c[0] + PAD + 0.5:
                out.append(f"title wraps: {mine[0][6]!r} / {mine[1][6]!r}")
    return out


def test_card_slack_under_3_cm() -> None:
    assert slack_problems(cards(), text_lines()) == []


def test_no_empty_band_or_area_in_a_card() -> None:
    assert empty_band_problems(render(PDF, 20), cards(), 20) == []


def test_figures_span_the_text_width_one_cm_apart() -> None:
    fig = next(c for c in cards() if c[0] == 48.33)
    assert figure_extent_problems(figure_inks(render(PDF, 50), fig, 50)) == []


def test_all_text_on_the_type_scale() -> None:
    assert all_size_problems(span_sizes()) == []


def test_no_orphan_lines() -> None:
    assert short_line_problems(cards(), text_lines()) == []


def test_body_text_inset() -> None:
    assert inset_problems(cards(), text_lines()) == []


def test_header_and_footer_margins() -> None:
    assert margin_problems(render(PDF, 100), 100) == []


def test_no_title_wraps() -> None:
    assert title_wrap_problems(cards(), text_lines()) == []


def test_planted_r_gates() -> None:
    import numpy as np

    card = (3.0, 15.2, 46.33, 40.0)
    line = lambda x, y, w, text, size=29.9: (x, y, x + w, y + 1.0, size, "LMRoman10-Regular", text)  # noqa: E731
    assert slack_problems([card], [line(3.8, 16.0, 30, "only one line")])  # R2
    img = np.full((int(90 * 20 / 2.54), int(140 * 20 / 2.54), 3), 240, dtype=np.uint8)
    assert empty_band_problems(img, [card], 20)  # R3 (a): an empty card
    px = 20 / 2.54
    img[int(16 * px) : int(39 * px), int(3.8 * px) : int(20 * px)] = 0  # left side full height
    img[int(16 * px) : int(30 * px), int(25 * px) : int(45.5 * px)] = 0  # right side 9 cm shorter
    assert any("area" in p for p in empty_band_problems(img, [card], 20))  # R3 (c)
    assert figure_extent_problems(
        [(49.1, 18, 90.9, 46), (49.1, 49.5, 90.9, 78)]
    )  # R3 (b): 3.5 cm gap
    assert figure_extent_problems(
        [(53.5, 18, 86.3, 46), (49.1, 47, 90.9, 78)]
    )  # Fig 1a narrower (R7)
    assert not figure_extent_problems([(49.2, 18, 90.8, 46), (49.2, 47.1, 90.8, 78)])
    assert all_size_problems([29.9, 34.5])  # R4
    assert not all_size_problems([29.9, 20.9, 16.7])
    assert short_line_problems([card], [line(3.8, 20.0, 1.2, "N.")])  # R8
    assert not short_line_problems([card], [line(3.8, 20.0, 4.0, "of pairs.")])
    assert inset_problems([card], [line(4.15, 20.0, 20, "For K")])  # R9
    blank = np.full((int(90 * 100 / 2.54), int(140 * 100 / 2.54), 3), 255, dtype=np.uint8)
    blank[int(89.5 * 100 / 2.54), 500:600] = 0  # footer ink 0.5 cm from the trim
    blank[int(2 * 100 / 2.54) : int(12.8 * 100 / 2.54), 4000:4100] = 0  # header hugging the rule
    assert len(margin_problems(blank, 100)) == 2  # R10
    title = [
        line(5.0, 15.8, 30, "Kaplan law on RedPajama", 43.8),
        line(10.0, 17.6, 10, "1a, 1c", 43.8),
    ]
    assert title_wrap_problems([card], title)  # R11
