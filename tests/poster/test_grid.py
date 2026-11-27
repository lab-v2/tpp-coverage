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
        bold = len(mine) >= 2 and "Bold" in mine[1][5]  # a 44 pt display is not a title line
        if bold and abs(mine[0][4] - 43.8) < 0.5 and abs(mine[1][4] - 43.8) < 0.5:
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
    title = [(*t[:5], "LMRoman10-Bold", t[6]) for t in title]
    assert title_wrap_problems([card], title)  # R11


# --- review M3: card profiles (title-to-body gap, ink bottoms), footer baselines, legends ---------

TITLE_GAP, TITLE_GAP_TOL = 0.70, 0.15
BOTTOM_TOL = 0.15


def _profile(args: tuple) -> tuple[float, float, float]:
    """(title ink bottom, first body ink top, ink bottom) of one card (cm), from a raster."""
    import numpy as np

    img, card, dpi = args
    per_cm = dpi / 2.54
    a, b = int((card[0] + PAD) * per_cm), int((card[2] - PAD) * per_cm)
    top, bottom = int((card[1] + 0.3) * per_cm), int((card[3] - 0.3) * per_cm)
    sub = img[top:bottom, a:b].astype(int)
    bg = np.median(sub.reshape(-1, 3), axis=0)
    rows = np.nonzero((np.abs(sub - bg).sum(axis=2) > 40).any(axis=1))[0]
    breaks = np.nonzero(np.diff(rows) > int(0.3 * per_cm))[0]
    k = breaks[0] if len(breaks) else len(rows) - 1
    first_body = rows[k + 1] if len(breaks) else rows[-1]
    return ((top + rows[k]) / per_cm, (top + first_body) / per_cm, (top + rows[-1]) / per_cm)


def card_profiles(img, cs: list[Rect], dpi: float) -> list[tuple[float, float, float]]:
    """Profiles of all cards, one process per card (the cards are independent)."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    workers = min(len(cs), len(os.sched_getaffinity(0)))
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_profile, [(img, c, dpi) for c in cs]))


def title_gap_problems(cs: list[Rect], profiles: list) -> list[str]:  # M3-7
    return [
        f"card {c}: title to body {p[1] - p[0]:.2f} cm, not {TITLE_GAP} +- {TITLE_GAP_TOL}"
        for c, p in zip(cs, profiles)
        if abs(p[1] - p[0] - TITLE_GAP) > TITLE_GAP_TOL
    ]


def ink_bottom_problems(cs: list[Rect], profiles: list) -> list[str]:  # M4-2
    last: dict[float, float] = {}
    for c, p in zip(cs, profiles):
        if c[3] >= BODY_BOTTOM - 0.05:
            last[c[0]] = p[2]
    if len(last) != 3:
        return [f"expected 3 column bottoms, found {last}"]
    spread = max(last.values()) - min(last.values())
    return (
        []
        if spread <= BOTTOM_TOL
        else [f"column ink bottoms {sorted(last.values())} spread {spread:.2f}"]
    )


def footer_problems(lines: list[Line], tol: float = 0.10) -> list[str]:  # M3-8, M4-3
    """Every footer line inside one column (or 3-137), and the three blocks' first baselines equal."""
    foot = [ln for ln in lines if ln[1] > 85.8]
    spans = [(EDGES[i], EDGES[i + 1]) for i in range(0, 6, 2)] + [(EDGES[0], EDGES[-1])]
    out = [
        f"footer line {ln[6][:25]!r} {ln[0]:.2f}-{ln[2]:.2f} crosses a gutter"
        for ln in foot
        if not any(a - tol <= ln[0] and ln[2] <= b + tol for a, b in spans[:3])
    ]
    firsts = []
    for a, b in spans[:3]:
        mine = [ln for ln in foot if a - tol <= ln[0] and ln[2] <= b + tol]
        if mine:
            firsts.append(min(ln[3] for ln in mine))
    if len(firsts) != 3 or max(firsts) - min(firsts) > 0.05:
        out.append(f"footer first lines end at {[round(f, 2) for f in firsts]}")
    return out


def hero_problems(lines: list[Line], tol: float = 0.10) -> list[str]:  # M3-6
    x0 = EDGES[4] + PAD
    q = (EDGES[5] - EDGES[4] - 2 * PAD) / 4
    big = sorted(ln[0] for ln in lines if abs(ln[4] - 109.6) < 0.5)
    keys = sorted(
        ln[0] for ln in lines if abs(ln[4] - 43.8) < 0.5 and ln[0] > x0 - 0.1 and 20 < ln[1] < 30
    )
    keys = [k for i, k in enumerate(keys) if i == 0 or k - keys[i - 1] > 2.0]  # one per sub-column
    out = []
    if len(big) != 2 or abs(big[0] - x0) > tol or abs(big[1] - x0 - 2 * q) > tol:
        out.append(f"hero numbers at {big}")
    if len(keys) != 4 or any(abs(k - x0 - i * q) > tol for i, k in enumerate(keys)):
        out.append(f"key numbers at {keys}")
    return out


def test_title_to_body_gap() -> None:
    cs = cards()
    assert title_gap_problems(cs, card_profiles(render(PDF, 100), cs, 100)) == []


def test_column_ink_bottoms_equal() -> None:
    cs = cards()
    assert ink_bottom_problems(cs, card_profiles(render(PDF, 100), cs, 100)) == []


def test_footer_on_the_grid() -> None:
    assert footer_problems(text_lines()) == []


def test_hero_sub_grid() -> None:
    assert hero_problems(text_lines()) == []


def test_planted_m3_gates() -> None:
    card = (3.0, 15.2, 46.33, 40.0)
    assert title_gap_problems([card], [(17.3, 18.9, 39.0)])  # 1.6 cm title to body
    cs = [(3.0, 15.2, 46.33, 83.6), (48.33, 15.2, 91.67, 83.6), (93.67, 15.2, 137.0, 83.6)]
    assert ink_bottom_problems(cs, [(0, 0, 82.8), (0, 0, 82.8), (0, 0, 81.1)])
    ln = lambda x0, x1, y1, t: (x0, y1 - 0.9, x1, y1, 23.9, "LMRoman10-Regular", t)  # noqa: E731
    good = [ln(3.0, 20, 86.9, "a b"), ln(55, 85, 86.9, "c d"), ln(120, 137, 86.9, "e f")]
    assert not footer_problems(good)
    assert footer_problems([ln(3.0, 20, 87.4, "a b")] + good[1:])  # first baselines differ
    assert footer_problems(good + [ln(44.0, 96.0, 87.9, "refs across a gutter")])
    hero = lambda xs: [(x, 23.3, x + 5, 24.6, 43.8, "LMRoman10-Bold", "1") for x in xs]  # noqa: E731
    big = [
        (94.47, 18, 106, 21, 109.6, "LMRoman10-Bold", "97.3%"),
        (116.17, 18, 127, 21, 109.6, "LMRoman10-Bold", "98.7%"),
    ]
    assert hero_problems(big + hero([94.47, 104.9, 115.33, 125.77]))  # 98.7% 0.84 off the half line


# --- review M4: vertical rhythm inside cards ------------------------------------------------------

SEG_GAP_CM = 0.12  # ink runs closer than this belong to one segment (one line, one display)
DISPLAY_GAP, DISPLAY_TOL = 0.5, 0.15


def _segments(args: tuple) -> list[tuple[float, float]]:
    """Ink runs (top, bottom, cm) of one card's text area, top to bottom, from a raster."""
    import numpy as np

    img, card, dpi = args
    per_cm = dpi / 2.54
    a, b = int((card[0] + PAD) * per_cm), int((card[2] - PAD) * per_cm)
    top, bottom = int((card[1] + 0.3) * per_cm), int((card[3] - 0.3) * per_cm)
    sub = img[top:bottom, a:b].astype(int)
    bg = np.median(sub.reshape(-1, 3), axis=0)
    rows = np.nonzero((np.abs(sub - bg).sum(axis=2) > 40).any(axis=1))[0]
    out, start = [], rows[0]
    for i in range(1, len(rows)):
        if rows[i] - rows[i - 1] > SEG_GAP_CM * per_cm:
            out.append(((top + start) / per_cm, (top + rows[i - 1]) / per_cm))
            start = rows[i]
    out.append(((top + start) / per_cm, (top + rows[-1]) / per_cm))
    return out


def card_segments(img, cs: list[Rect], dpi: float) -> list[list[tuple[float, float]]]:
    import os
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(min(len(cs), len(os.sched_getaffinity(0)))) as pool:
        return list(pool.map(_segments, [(img, c, dpi) for c in cs]))


def table_spans(pdf: Path = PDF) -> list[Rect]:
    """Booktabs tables: (x0, top rule y, x1, bottom rule y) from the thin rules inside cards."""
    page = fitz.open(pdf)[0]
    bars = sorted(
        tuple(round(v * CM, 2) for v in (d["rect"].x0, d["rect"].y0, d["rect"].x1, d["rect"].y1))
        for d in page.get_drawings()
        if d["rect"].height * CM < 0.15
        and 5 < d["rect"].width * CM < 40
        and d["rect"].y0 * CM > BODY_TOP
        and not EDGES[2] <= d["rect"].x0 * CM < EDGES[3]  # C2 holds figures, not tables
    )
    out: list[Rect] = []
    for b in bars:
        if out and abs(out[-1][0] - b[0]) < 0.05 and b[1] - out[-1][3] < 6.0:
            out[-1] = (out[-1][0], out[-1][1], out[-1][2], b[1])
        else:
            out.append(b)
    return [t for t in out if t[3] > t[1]]


def display_rows(card: Rect, lines: list[Line], tables: list[Rect] = ()) -> list[float]:
    """Centre y of each display: a row of lines centred on the card, away from its text edge,
    that is neither the title (the first row) nor a table row (a table holds 3 or more rows of
    several lines in one x-range)."""
    mine = sorted((ln for ln in lines if inside(ln, card)), key=lambda ln: ln[1])
    rows: list[list[Line]] = []
    for ln in mine:
        if rows and ln[1] < max(r[3] for r in rows[-1]) - 0.3:
            rows[-1].append(ln)
        else:
            rows.append([ln])
    centre = (card[0] + card[2]) / 2
    out = []
    for row in rows[1:]:
        x0, x1 = min(r[0] for r in row), max(r[2] for r in row)
        y = (min(r[1] for r in row) + max(r[3] for r in row)) / 2
        if any(t[0] - 0.1 <= x0 and t[1] - 0.5 <= y <= t[3] + 0.5 for t in tables):
            continue
        if x0 > card[0] + PAD + 1.0 and abs((x0 + x1) / 2 - centre) < 1.0 and len(row) <= 6:
            if (
                not any(r[5].startswith("Cm") for r in row)
                and len(row) < 4
                or any("Math" in r[5] for r in row)
            ):
                out.append((min(r[1] for r in row) + max(r[3] for r in row)) / 2)
    return out


def display_gap_problems(
    cs: list[Rect], lines: list[Line], segs: list, tables: list[Rect] = ()
) -> list[str]:
    out = []
    for c, seg in zip(cs, segs):
        for y in display_rows(c, lines, tables):
            k = next((i for i, s in enumerate(seg) if s[0] - 0.2 <= y <= s[1] + 0.2), None)
            if k is None:
                continue
            for gap, side in (  # segment 0 is the title: its gap is the title-to-body gate's
                (seg[k][0] - seg[k - 1][1] if k > 1 else None, "above"),
                (seg[k + 1][0] - seg[k][1] if k + 1 < len(seg) else None, "below"),
            ):
                if gap is not None and abs(gap - DISPLAY_GAP) > DISPLAY_TOL and gap < 3.0:
                    out.append(f"card {c[:2]}: display at {y:.2f}: {gap:.2f} cm {side}")
    return out


def baseline_rows(pdf: Path = PDF, top: float = 85.8) -> list[list[tuple[float, float]]]:
    """Footer lines as (x0, main-span baseline), grouped into rows of nearby baselines."""
    page = fitz.open(pdf)[0]
    pts = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            spans = [s for s in ln["spans"] if s["text"].strip()]
            if spans:
                main = max(spans, key=lambda s: len(s["text"]))
                if main["origin"][1] * CM > top:
                    pts.append((main["origin"][0] * CM, main["origin"][1] * CM))
    rows: list[list[tuple[float, float]]] = []
    for p in sorted(pts, key=lambda p: p[1]):
        if rows and p[1] - rows[-1][0][1] < 0.4:
            rows[-1].append(p)
        else:
            rows.append([p])
    return rows


def footer_baseline_problems(rows: list) -> list[str]:  # M5: every footer line on shared baselines
    return [
        f"footer row at {r[0][1]:.2f}: baselines spread {max(p[1] for p in r) - min(p[1] for p in r):.2f}"
        for r in rows
        if max(p[1] for p in r) - min(p[1] for p in r) > 0.05
    ]


def hero_gap_problems(gaps: list[float], tol: float = 0.05) -> list[str]:  # M5: the two hero gaps
    big = sorted(gaps)[-2:]
    return [] if len(big) == 2 and big[1] - big[0] <= tol else [f"hero gaps {big}"]


def row_edge_problems(box: Rect, table: tuple[float, float], tol: float = 0.15) -> list[str]:  # M5
    """A box and the table beside it share their top and bottom."""
    return [
        f"box {box[1]:.2f}-{box[3]:.2f} vs table {table[0]:.2f}-{table[1]:.2f}"
        for _ in [0]
        if abs(box[1] - table[0]) > tol or abs(box[3] - table[1]) > tol
    ]


def kappa_row(pdf: Path = PDF) -> tuple[Rect, tuple[float, float]]:
    """The kappa box fill and its table's top rule to note-ink bottom (cm)."""
    import numpy as np

    page = fitz.open(pdf)[0]
    box = next(
        tuple(v * CM for v in (d["rect"].x0, d["rect"].y0, d["rect"].x1, d["rect"].y1))
        for d in page.get_drawings()
        if d.get("fill")
        and abs(d["fill"][2] - 1.0) < 0.01
        and 5 < d["rect"].width * CM < 20
        and d["rect"].x0 * CM < 46
    )
    t = next(t for t in table_spans(pdf) if t[0] < 46 and t[1] > BODY_TOP)
    img, per_cm = render(pdf, 100), 100 / 2.54
    sub = img[
        int(t[3] * per_cm) : int((box[3] + 1.0) * per_cm), int(t[0] * per_cm) : int(t[2] * per_cm)
    ].astype(int)
    bg = sub[-1, -1]
    rows = np.nonzero((np.abs(sub - bg).sum(axis=2) > 40).any(axis=1))[0]
    run = [
        r for i, r in enumerate(rows) if i == 0 or r - rows[i - 1] <= 3
    ]  # the note, under the rule
    return box, (t[1], t[3] + run[-1] / per_cm)


def test_display_gaps() -> None:
    cs = cards()
    segs = card_segments(render(PDF, 100), cs, 100)
    assert display_gap_problems(cs, text_lines(), segs, table_spans()) == []


def test_hero_gaps_equal() -> None:
    cs = cards()
    hero = next(c for c in cs if c[0] > 90 and c[1] < BODY_TOP + 0.1)
    seg = card_segments(render(PDF, 100), [hero], 100)[0]
    assert hero_gap_problems([b[0] - a[1] for a, b in zip(seg, seg[1:])]) == []


def test_footer_baselines_shared() -> None:
    assert footer_baseline_problems(baseline_rows()) == []


def test_kappa_box_and_table_share_edges() -> None:
    assert row_edge_problems(*kappa_row()) == []


def test_planted_m5_gates() -> None:
    assert footer_baseline_problems([[(3.0, 87.62), (121.0, 87.27)]])
    assert not footer_baseline_problems([[(3.0, 87.62), (121.0, 87.58)]])
    assert hero_gap_problems([0.3, 2.28, 2.10])
    assert row_edge_problems((9.7, 60.2, 24.8, 66.3), (60.2, 65.6))  # the M4 box ran 0.7 low
    assert not row_edge_problems((9.7, 60.2, 24.8, 66.3), (60.22, 66.2))
    seg = [
        (15.6, 16.8),
        (17.5, 18.4),
        (19.9, 21.0),
        (21.3, 22.0),
    ]  # a display 1.5 cm below its text
    card = (3.0, 15.2, 46.33, 23.0)
    disp = [
        (10.0, 15.6, 30.0, 16.8, 43.8, "LMRoman10-Bold", "Title"),
        (14.67, 19.9, 34.67, 21.0, 29.9, "LatinModernMath-Regular", "x"),
    ]
    assert display_gap_problems([card], disp, [seg])


# --- review M6: figure-slot voids, line pitch, orphans, row ends, equal gaps -----------------------

BLOCK_GAP_MIN = 0.4  # segment gaps above this separate paragraphs or blocks, not lines
EQUAL_GAP_TOL = 0.15


def largest_empty_rect(empty) -> tuple[int, int]:
    """(width, height) in pixels of the largest-area all-True axis-aligned rectangle (histogram method)."""
    import numpy as np

    h = np.zeros(empty.shape[1], dtype=int)
    best = (0, 0, 0)
    for row in empty:
        h = np.where(row, h + 1, 0)
        stack: list[tuple[int, int]] = []
        for x in range(len(h) + 1):
            cur = h[x] if x < len(h) else 0
            start = x
            while stack and stack[-1][1] >= cur:
                start, hh = stack.pop()
                if (x - start) * hh > best[0]:
                    best = ((x - start) * hh, x - start, hh)
            stack.append((start, cur))
    return best[1], best[2]


def slot_void_problems(img, card: Rect, inks: list, dpi: float) -> list[str]:  # R3 (b), M6
    """The largest empty rectangle inside the figure slots (the card's text width, from the first
    figure's ink top to the last one's ink bottom) must not be 3 x 3 cm or more."""
    import numpy as np

    per_cm = dpi / 2.54
    x0, x1 = int((card[0] + PAD) * per_cm), int((card[2] - PAD) * per_cm)
    y0, y1 = int(inks[0][1] * per_cm), int(inks[-1][3] * per_cm)
    sub = img[y0:y1, x0:x1].astype(int)
    bg = img[int((card[1] + 0.3) * per_cm), int((card[0] + 0.3) * per_cm)].astype(int)
    w, h = largest_empty_rect(np.abs(sub - bg).sum(axis=2) < 24)
    if w / per_cm >= EMPTY_CM and h / per_cm >= EMPTY_CM:
        return [f"figure slot void {w / per_cm:.1f} x {h / per_cm:.1f} cm"]
    return []


def body_baselines(pdf: Path = PDF) -> list[tuple[float, float, int, str]]:
    """Body lines (29.9 pt Latin Modern main span): (x0, baseline, word count, text, x1) in cm."""
    page = fitz.open(pdf)[0]
    words = page.get_text("words")
    out = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            spans = [s for s in ln["spans"] if s["text"].strip()]
            if not spans:
                continue
            main = max(spans, key=lambda s: len(s["text"]))
            if abs(main["size"] - 29.9) > 0.5 or not main["font"].startswith("LM"):
                continue
            bx = fitz.Rect(ln["bbox"])
            n = sum(
                1 for w in words if bx.contains(fitz.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2))
            )
            text = "".join(s["text"] for s in spans)
            out.append((bx.x0 * CM, main["origin"][1] * CM, n, text, bx.x1 * CM))
    return out


def pitch_problems(cs: list[Rect], lines: list, tol: float = 0.05) -> list[str]:  # M3-4
    """Consecutive body lines starting at one x in a card are 1.30 apart, +- 0.05. 1.55 is allowed
    only at a real break: the lower line starts a list item (a bullet or a number), or the upper
    line ends a sentence and is shorter than 95% of the text width. 2.0 or more is a display or a
    block between them."""
    out = []
    for c in cs:
        mine = sorted(
            (ln for ln in lines if c[0] <= ln[0] < c[2] and c[1] <= ln[1] < c[3]),
            key=lambda ln: ln[1],
        )
        for a, b in zip(mine, mine[1:]):
            d = b[1] - a[1]
            if abs(a[0] - b[0]) > 0.1 or d >= 2.0:
                continue
            text_w = c[2] - c[0] - 2 * PAD
            item = b[3].lstrip().startswith(("\u2022", "1.", "2.", "3.", "4."))
            ends = a[3].rstrip().endswith((".", ":", ";"))
            sentence_end = ends and len(a) > 4 and a[4] - a[0] < 0.95 * text_w
            allowed = [1.30] + ([1.55] if item or sentence_end else [])
            if min(abs(d - v) for v in allowed) > tol:
                out.append(f"pitch {d:.2f} at {b[1]:.2f}: {b[3][:25]!r}")
    return out


def orphan_problems(cs: list[Rect], lines: list) -> list[str]:  # M3-5
    """No one-word line ends a paragraph or bullet: a one-word line 1.30 under a line of its paragraph."""
    out = []
    for c in cs:
        mine = sorted(
            (ln for ln in lines if c[0] <= ln[0] < c[2] and c[1] <= ln[1] < c[3]),
            key=lambda ln: ln[1],
        )
        for a, b in zip(mine, mine[1:]):
            if b[2] == 1 and abs(b[1] - a[1] - 1.30) <= 0.1:
                out.append(f"one-word last line {b[3]!r} at {b[1]:.2f}")
    return out


def row_end_problems(
    cs: list[Rect], rows: list[tuple[float, float, float]], tol: float = 0.20
) -> list[str]:  # M3-9
    """Each side-by-side row or table (x0, x1, y) ends at both text edges or is centred on its card."""
    out = []
    for x0, x1, y in rows:
        c = next(c for c in cs if c[0] <= x0 < c[2] and c[1] <= y < c[3])
        left, right = c[0] + PAD, c[2] - PAD
        flush = abs(x0 - left) <= tol and abs(x1 - right) <= tol
        if not flush and abs((x0 + x1) / 2 - (c[0] + c[2]) / 2) > tol:
            out.append(f"row {x0:.2f}-{x1:.2f} at {y:.2f} neither flush nor centred")
    return out


def side_rows(pdf: Path = PDF) -> list[tuple[float, float, float]]:
    """The tables (booktabs extents) inside cards, the kappa row as box plus table."""
    box, _ = kappa_row(pdf)
    out = []
    for t in table_spans(pdf):
        if t[0] < 46 and abs(t[1] - box[1]) < 1.0:  # the kappa table: one row with its box
            out.append((min(box[0], t[0]), max(box[2], t[2]), t[1]))
        elif t[2] - t[0] < 40:
            out.append((t[0], t[2], t[1]))
    return out


def equal_gap_problems(cs: list[Rect], segs: list, tables: list[Rect]) -> list[str]:  # M6
    """Paragraph and block gaps inside a card (segment gaps over BLOCK_GAP_MIN, after the title,
    outside tables) are equal within EQUAL_GAP_TOL; in the figure card, figure to figure equals
    figure to caption. Declared exception: the hero card's two spare gaps (gated equal by
    hero_gap_problems)."""
    out = []
    for c, seg in zip(cs, segs):
        pairs = list(zip(seg[1:], seg[2:]))
        gaps = [
            b[0] - a[1]
            for a, b in pairs
            if b[0] - a[1] > BLOCK_GAP_MIN
            and not any(
                t[0] - 0.2 < c[2] and t[1] - 0.3 <= a[1] and b[0] <= t[3] + 1.2
                for t in tables
                if c[0] <= t[0] < c[2]
            )
        ]
        if c[0] > 90 and c[1] < BODY_TOP + 0.1:
            gaps = sorted(gaps)[:-2]
        if gaps and max(gaps) - min(gaps) > EQUAL_GAP_TOL:
            out.append(f"card {c[:2]}: block gaps {[round(g, 2) for g in gaps]}")
    return out


def test_figure_slot_void() -> None:
    fig = next(c for c in cards() if abs(c[0] - EDGES[2]) < 0.01)
    img = render(PDF, 20)
    assert slot_void_problems(img, fig, figure_inks(render(PDF, 50), fig, 50), 20) == []


def test_line_pitch() -> None:
    assert pitch_problems(cards(), body_baselines()) == []


def test_no_one_word_last_lines() -> None:
    assert orphan_problems(cards(), body_baselines()) == []


def test_rows_flush_or_centred() -> None:
    assert row_end_problems(cards(), side_rows()) == []


def test_equal_gaps_inside_cards() -> None:
    cs = cards()
    assert equal_gap_problems(cs, card_segments(render(PDF, 100), cs, 100), table_spans()) == []


def test_planted_m6_gates() -> None:
    import numpy as np

    card = (48.33, 15.2, 91.67, 83.6)
    px = 20 / 2.54
    img = np.full((int(90 * px), int(140 * px), 3), 243, dtype=np.uint8)
    img[int(18 * px) : int(48 * px), int(56 * px) : int(90.8 * px)] = (
        0  # Fig 1a, 6.6 cm short at the left
    )
    img[int(49 * px) : int(78 * px), int(49.2 * px) : int(90.8 * px)] = 0
    assert slot_void_problems(
        img, card, [(49.1, 18, 90.8, 48), (49.1, 49, 90.8, 78)], 20
    )  # the M6 void
    c1 = (3.0, 15.2, 46.33, 30.0)
    assert pitch_problems(
        [c1], [(3.8, 20.0, 5, "a", 45.0), (3.8, 21.83, 5, "b", 45.0)]
    )  # Theorem 1, M2
    assert not pitch_problems([c1], [(3.8, 20.0, 5, "a", 45.0), (3.8, 21.30, 5, "b", 45.0)])
    # A forced \\[0.25cm] inside a paragraph: the upper line is full and ends mid-sentence.
    full = (3.8, 20.0, 9, "runs on to the next", 45.4)
    assert pitch_problems([c1], [full, (3.8, 21.55, 5, "line here.", 20.0)])
    assert not pitch_problems(
        [c1], [(3.8, 20.0, 4, "ends here.", 20.0), (3.8, 21.55, 5, "Next", 30.0)]
    )
    assert orphan_problems([c1], [(3.8, 20.0, 8, "full line"), (3.8, 21.30, 1, "margin.")])
    assert not orphan_problems(
        [c1], [(3.8, 20.0, 8, "full line"), (3.8, 21.30, 2, "sufficient margin.")]
    )
    assert row_end_problems([c1], [(3.8, 42.74, 25.0)])  # the M2 kappa row: flush left, 2.8 short
    assert not row_end_problems([c1], [(9.75, 39.58, 25.0)])  # centred on the card
    segs = [[(15.6, 16.8), (17.5, 18.4), (18.96, 19.9), (21.0, 22.0)]]  # gaps 0.56, then 1.10
    assert equal_gap_problems([c1], segs, [])


# --- review M7: glyph-level overlaps -------------------------------------------------------------

GLYPH_OVERLAP_CM = 0.02
# Glyphs of different lines must also keep this much ink clearance (the M7 V_K subscript sat 0.03 cm
# over "= 0": touching reads as overlapping at poster distance).
GLYPH_CLEARANCE_CM = 0.10
POSTER_FONT = ("LMRoman", "LatinModernMath")


def _font_bounds(doc, page) -> dict:
    """Per embedded poster font: a function glyph id -> (xmin, ymin, xmax, ymax) in em units."""
    import io

    from fontTools.cffLib import CFFFontSet

    out = {}
    for xref, ext, _t, base, _n, _e in page.get_fonts():
        name = base.split("+", 1)[-1].replace("-Identity-H", "")
        if not name.startswith(POSTER_FONT) or ext != "cid":
            continue
        cff = CFFFontSet()
        cff.decompile(io.BytesIO(doc.extract_font(xref)[3]), None)
        top = cff[cff.fontNames[0]]
        cs, order = top.CharStrings, top.charset
        scale = top.FontMatrix[0] if hasattr(top, "FontMatrix") else 0.001

        def bounds(gid, cs=cs, order=order, scale=scale):
            # Identity-H: the PDF's glyph number is the CID; the CFF names CID-keyed glyphs cidNNNNN.
            name = f"cid{gid:05d}"
            if name not in cs:
                name = order[gid] if gid < len(order) else None
            if name is None or name not in cs:
                return None
            b = cs[name].calcBounds(cs)
            return None if b is None else tuple(v * scale for v in b)

        out[name] = bounds
    return out


def glyph_boxes(pdf: Path = PDF) -> list[tuple[int, Rect, str]]:
    """Tight ink boxes (cm) of every poster-font glyph, tagged with its PyMuPDF text line."""
    doc = fitz.open(pdf)
    page = doc[0]
    bounds = _font_bounds(doc, page)
    gid = {}
    for span in page.get_texttrace():
        for c in span["chars"]:
            gid[(round(c[2][0], 2), round(c[2][1], 2))] = c[1]
    out, line_no = [], 0
    for b in page.get_text("rawdict")["blocks"]:
        for ln in b.get("lines", []):
            line_no += 1
            for s in ln["spans"]:
                f = bounds.get(s["font"].split("+", 1)[-1])
                if f is None:
                    continue
                for ch in s["chars"]:
                    g = gid.get((round(ch["origin"][0], 2), round(ch["origin"][1], 2)))
                    bb = f(g) if g is not None else None
                    if bb is None:
                        continue
                    ox, oy, sz = ch["origin"][0], ch["origin"][1], s["size"]
                    box = (ox + bb[0] * sz, oy - bb[3] * sz, ox + bb[2] * sz, oy - bb[1] * sz)
                    out.append((line_no, tuple(v * CM for v in box), ch["c"]))
    return out


def glyph_overlap_problems(glyphs: list) -> list[str]:  # owner rule: zero overlaps (M8)
    """Glyphs of different text lines whose ink boxes intersect by more than GLYPH_OVERLAP_CM in both
    axes, or that sit above one another closer than GLYPH_CLEARANCE_CM."""
    gs = sorted(glyphs, key=lambda g: g[1][0])
    out = []
    for i, (la, a, ca) in enumerate(gs):
        for lb, b, cb in gs[i + 1 :]:
            if b[0] >= a[2] - GLYPH_OVERLAP_CM:
                break
            if la == lb:
                continue
            ix = min(a[2], b[2]) - max(a[0], b[0])
            iy = min(a[3], b[3]) - max(a[1], b[1])
            if ix > GLYPH_OVERLAP_CM and iy > GLYPH_OVERLAP_CM:
                out.append(f"{ca!r} and {cb!r} overlap at {a[0]:.2f}, {a[1]:.2f}")
            elif ix > GLYPH_OVERLAP_CM and -GLYPH_CLEARANCE_CM < iy <= GLYPH_OVERLAP_CM:
                out.append(
                    f"{ca!r} and {cb!r} touch ({-iy:.2f} cm apart) at {a[0]:.2f}, {a[1]:.2f}"
                )
    return out


def test_no_glyph_overlaps() -> None:
    glyphs = glyph_boxes()
    assert len(glyphs) > 2000, len(glyphs)
    assert glyph_overlap_problems(glyphs) == []


def test_planted_m7_overlap() -> None:
    """The M7 build's threshold card (commit f9d8f79; its glyph boxes frozen in fixtures/, taken
    with glyph_boxes), whose V_K label sat on "= 0" in the next line, must fail."""
    import json

    raw = json.loads(
        (Path(__file__).parent / "fixtures" / "m7_threshold_card_glyphs.json").read_text()
    )
    found = glyph_overlap_problems([(ln, tuple(box), c) for ln, box, c in raw])
    assert any("\U0001d43e" in f for f in found), found  # the math-italic K of V_K
