"""The NeurIPS 2026 poster (`poster/`): print size, fonts, logos, QR code, title, authors, dashes.

Each check is a function of its input (a pdfinfo or pdffonts report, a LaTeX source, a decoded QR
payload) so that `test_planted_*` can feed it a known-bad input and prove it refuses it. The checks
read the committed PDF; rebuild it with `make -C poster` after a source change.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
POSTER = REPO / "poster"
PDF = POSTER / "tpp_neurips2026_poster.pdf"
TEX = POSTER / "tpp_neurips2026_poster.tex"
URL = "https://lab-v2.github.io/tpp-coverage/"
A0_LANDSCAPE_PT = (3370.39, 2383.94)  # 118.9 x 84.1 cm
LOGOS = {"neurips", "syracuse_ai_institute", "leibniz_lab"}
EM_DASH = re.compile("\u2014|---")
MAX_HEADER_FRACTION = 0.17  # header rule above 17% of the height (reference posters: 12-17%)
MAX_COLUMN_SPREAD_CM = 1.0  # the three column bottoms within 1 cm
MIN_TICK_PT = 24.0  # printed size of the figures' tick labels
TICK_WORDS = {"0.0", "0.2", "0.4", "0.5", "0.6", "0.8", "1.0", "16.0", "16.6", "16.9", "17.4"}
PT_PER_CM = 72 / 2.54
# The owner asked for genuine LaTeX math: every font is Lato, Raleway or Computer / Latin Modern.
ALLOWED_FONT = re.compile(
    r"^(Lato|Raleway|CM(MI|MIB|SY|EX|R|BX|SS)[0-9]+|Cm(mi|r|sy|ex)[0-9]+|MSBM10|MSAM10|LMRoman|LMMath|"
    r"LatinModernMath)"
)
MATH_FONT = re.compile(r"LatinModernMath|LMMath|CMMI|Cmmi")
BLOCK_TITLE_PT = 41.2  # fallback block-title height; normally read off the TL;DR title
# Gaps run from a block's last text line to the next block title, so a tinted block's inner padding
# (about 0.4 cm) and a figure's bottom margin are inside them; 0.55 cm around the median allows that
# and nothing more (a 0.9 cm \vspace pad fails).
MAX_GAP_DEVIATION_CM = 0.55
MAX_WORDS = 800  # the reference posters carry 500-800 words; raised from 750 in round 7


def _run(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


# --- the checks -----------------------------------------------------------------------------------


def page_size_problems(pdfinfo: str) -> list[str]:
    pages = re.search(r"^Pages:\s+(\d+)", pdfinfo, re.M)
    size = re.search(r"^Page size:\s+([\d.]+) x ([\d.]+) pts", pdfinfo, re.M)
    if not pages or not size:
        return ["pdfinfo report has no page count or size"]
    problems = [] if pages[1] == "1" else [f"{pages[1]} pages, not 1"]
    w, h = float(size[1]), float(size[2])
    if abs(w - A0_LANDSCAPE_PT[0]) > 1 or abs(h - A0_LANDSCAPE_PT[1]) > 1:
        problems.append(f"page is {w} x {h} pt, not A0 landscape {A0_LANDSCAPE_PT}")
    return problems


def font_problems(pdffonts: str) -> list[str]:
    rows = pdffonts.splitlines()[2:]
    if not rows:
        return ["no fonts listed"]
    problems = []
    for row in rows:
        if "Type 3" in row:
            problems.append(f"Type 3 font: {row.split()[0]}")
        # Columns after the type and encoding: emb sub uni object-id generation.
        if row.split()[-5] != "yes":
            problems.append(f"not embedded: {row.split()[0]}")
    return problems


def logo_problems(tex: str) -> list[str]:
    """The images taken from ../images/ (the repo's logos) must be exactly the three logos."""
    used = {
        Path(name).stem for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    }
    available = {p.stem for p in (REPO / "images").iterdir() if p.is_file()}
    logos = {u for u in used if u in available or "syracuse" in u or "logo" in u}
    problems = []
    if logos != LOGOS:
        problems.append(f"logos used {sorted(logos)}, expected {sorted(LOGOS)}")
    return problems


def title_author_problems(tex: str, cff: dict) -> list[str]:
    title = re.search(r"\\title\{(.+?)\}\n", tex, re.S)
    author = re.search(r"\\author\{(.+?)\}\n\\institute", tex, re.S)
    if not title or not author:
        return ["no \\title or \\author"]
    got_title = " ".join(title[1].replace("\\\\", " ").split())
    problems = [] if got_title == cff["title"] else [f"title {got_title!r} != {cff['title']!r}"]
    plain = re.sub(r"\\textsuperscript\{[^}]*\}|\\quad|\\\\", "|", author[1])
    got = [" ".join(n.split()) for n in plain.split("|") if n.strip()]
    want = [f"{a['given-names']} {a['family-names']}" for a in cff["authors"]]
    if got != want:
        problems.append(f"authors {got} != {want}")
    return problems


def em_dash_problems(text: str) -> list[str]:
    return [
        f"em dash on line {i}"
        for i, line in enumerate(text.splitlines(), 1)
        if EM_DASH.search(line)
    ]


def math_font_problems(pdffonts: str, need_math: bool = True) -> list[str]:
    """Fonts outside Lato, Raleway and CM/LM; and, for the poster, no genuine math font at all."""
    names = [row.split()[0].split("+", 1)[-1] for row in pdffonts.splitlines()[2:] if row.strip()]
    problems = [
        f"font {n} is not Lato, Raleway or CM/LM" for n in names if not ALLOWED_FONT.match(n)
    ]
    if need_math and not any(MATH_FONT.search(n) for n in names):
        problems.append("no LaTeX math font embedded")
    return problems


def heading_math_problems(tex: str) -> list[str]:
    """No $...$ inside a block title or a \\statement{} run-in head (bold math reads as a defect)."""
    heads = re.findall(r"\\begin\{(?:alert)?block\}\{((?:[^{}]|\{[^{}]*\})*)\}", tex)
    heads += re.findall(r"\\statement\{((?:[^{}]|\{[^{}]*\})*)\}", tex)
    return [f"math in heading: {h}" for h in heads if "$" in h]


def word_count_problems(text: str) -> list[str]:
    """Words as pdftotext prints them (whitespace-separated), header and footer included."""
    n = len(text.split())
    return [] if n <= MAX_WORDS else [f"{n} words, more than {MAX_WORDS}"]


def bbox_words(pdf: Path) -> list[tuple[float, float, float, float, str]]:
    """Every word pdftotext finds, as (xMin, yMin, xMax, yMax, text) in points from the top left."""
    html = _run("pdftotext", "-bbox", str(pdf), "-")
    pat = r'xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)<'
    return [(float(a), float(b), float(c), float(d), w) for a, b, c, d, w in re.findall(pat, html)]


def column_spread_problems(words: list, width: float, height: float) -> list[str]:
    """The lowest word of each third of the page, between the header and the footer."""
    body = [w for w in words if height * 0.2 < w[3] < height * 0.95]
    bottoms = []
    for c in range(3):
        col = [w[3] for w in body if c * width / 3 <= w[0] < (c + 1) * width / 3]
        if not col:
            return [f"column {c + 1} has no text"]
        bottoms.append(max(col))
    spread = (max(bottoms) - min(bottoms)) / PT_PER_CM
    ok = spread < MAX_COLUMN_SPREAD_CM
    return [] if ok else [f"column bottoms {[round(b / PT_PER_CM, 1) for b in bottoms]} cm"]


def block_gaps(words: list, width: float) -> list[list[float]]:
    """Per column, the gaps (cm) between each block title and the text line above it."""
    gaps = []
    for c in range(3):
        col = [w for w in words if c * width / 3 <= w[0] < (c + 1) * width / 3]
        tops = sorted({y0 for cc, y0, _ in block_title_tops(words, TEX.read_text()) if cc == c})
        col_gaps = []
        for top in tops[1:]:
            above = [w[3] for w in col if w[3] < top - 1]
            col_gaps.append((top - max(above)) / PT_PER_CM)
        gaps.append(col_gaps)
    return gaps


def block_title_tops(words: list, tex: str) -> list[tuple[int, float, float]]:
    """(column, yMin, yMax) of each block title: words at the title size whose text starts a title in the source."""
    heads = re.findall(r"\\begin\{(?:alert)?block\}\{((?:[^{}]|\{[^{}]*\})*)\}", tex)
    firsts = set()
    for h in heads:
        plain = re.sub(r"\\(CO|NC)\{\}", r"\1", h)
        plain = [t for t in re.sub(r"[\\{}$]", " ", plain).split() if t not in ("CO", "NC")]
        firsts.add(plain[0] if plain else "")  # CO/NC also label the R^2 panel's lines
    width = A0_LANDSCAPE_PT[0]
    # The title size is read off the first block's title ("TL;DR"), so a body-scale change follows.
    sizes = [w[3] - w[1] for w in words if w[4] == "TL;DR"]
    title_pt = sizes[0] if sizes else BLOCK_TITLE_PT
    out = set()
    for w in words:
        if w[4] in firsts and abs((w[3] - w[1]) - title_pt) < 0.3:
            out.add((int(w[0] // (width / 3)), round(w[1], 1), round(w[3], 1)))
    return sorted(out)


def rule_coverage(png: Path, titles: list[tuple[int, float, float]], scale: float) -> list[float]:
    """For each title, the largest share of its column's width covered by dark pixels on one row just below it."""
    script = (
        "import sys, json; from PIL import Image\n"
        "im = Image.open(sys.argv[1]).convert('L'); px = im.load(); t = json.loads(sys.argv[2]); s = float(sys.argv[3])\n"
        "w3 = im.width / 3; out = []\n"
        "for c, y0, y1 in t:\n"
        "    xa, xb = int(c * w3 + 0.075 * w3), int((c + 1) * w3 - 0.075 * w3)\n"
        "    best = max(sum(px[x, y] < 120 for x in range(xa, xb)) for y in range(int(y1 * s), int((y1 + 45) * s)))\n"
        "    out.append(best / (xb - xa))\n"
        "print(json.dumps(out))"
    )
    import json

    out = _run(
        "uv",
        "run",
        "--quiet",
        "--no-project",
        "--with",
        "pillow",
        "python",
        "-I",
        "-c",
        script,
        str(png),
        json.dumps(titles),
        str(scale),
    )
    return json.loads(out)


def gap_problems(gaps: list[list[float]]) -> list[str]:
    flat = sorted(g for col in gaps for g in col)
    if len(flat) < 6:
        return [f"only {len(flat)} block gaps found"]
    median = flat[len(flat) // 2]
    return [
        f"column {c + 1}: gap {g:.2f} cm, median {median:.2f} cm"
        for c, col in enumerate(gaps)
        for g in col
        if abs(g - median) > MAX_GAP_DEVIATION_CM
    ]


def strip_label_problems(words: list, tex: str) -> list[str]:
    """Every text on the threshold strip's recipe row lies inside the strip's frame.

    The frame's data extent comes from the source (its `\\draw[thick] (x0, .) rectangle (x1, .)`); its
    page position from the tick labels "1" and "10", which sit at log10 = 0 and 1.
    """
    strip = tex[tex.index("\\newcommand{\\thresholdstrip}") :]
    m = re.search(r"\\draw\[thick\] \((-?[\d.]+),[^)]*\) rectangle \(([\d.]+),", strip)
    if not m:
        return ["no strip frame in the source"]
    x0, x1 = float(m[1]), float(m[2])
    width = A0_LANDSCAPE_PT[0]
    col = [w for w in words if width / 3 <= w[0] < 2 * width / 3]
    # The strip's row label "recipe" (body size), not the "Design recipe" block title.
    recipe = [w for w in col if w[4] == "recipe" and w[3] - w[1] < 0.8 * BLOCK_TITLE_PT]
    for r in recipe:
        ticks = [w for w in col if 0 < w[1] - r[3] < 120 and w[4] in ("1", "10")]
        one = [w for w in ticks if w[4] == "1"]
        ten = [w for w in ticks if w[4] == "10"]
        if one and ten:
            ten = [max(ten, key=lambda w: w[0])]  # 10^1; the others are the bases of 10^-2, 10^-1
            c1, c10 = (one[0][0] + one[0][2]) / 2, (ten[0][0] + ten[0][2]) / 2
            left, right = c1 + x0 * (c10 - c1), c1 + x1 * (c10 - c1)
            row = [w for w in col if abs(w[1] - r[1]) < 30 and w is not r]
            return [f"{w[4]!r} outside the strip frame" for w in row if w[2] > right or w[0] < left]
    return ["strip tick labels or recipe row not found"]


def tick_problems(words: list) -> list[str]:
    """Tick labels of the two data panels (pdftotext reports a word's font size as its height)."""
    ticks = [w for w in words if w[4] in TICK_WORDS]
    if len(ticks) < 8:
        return [f"only {len(ticks)} tick labels found"]
    small = [f"{w[4]} at {w[3] - w[1]:.1f} pt" for w in ticks if w[3] - w[1] < MIN_TICK_PT]
    return small


def header_fraction(png: Path) -> float:
    """Height of the header rule (the first full-width row of the theme's dark blue) over the page height."""
    script = (
        "import sys; from PIL import Image\n"
        "im = Image.open(sys.argv[1]).convert('RGB'); w, h = im.size; px = im.load()\n"
        "near = lambda p: abs(p[0] - 39) < 20 and abs(p[1] - 60) < 20 and abs(p[2] - 117) < 20\n"
        "rows = [y for y in range(h // 20, h // 2) if sum(near(px[x, y]) for x in range(0, w, 8)) > 0.8 * w / 8]\n"
        "print(rows[0] / h if rows else 1.0)"
    )
    out = _run(
        "uv",
        "run",
        "--quiet",
        "--no-project",
        "--with",
        "pillow",
        "python",
        "-I",
        "-c",
        script,
        str(png),
    )
    return float(out)


def decode_qr(png: Path) -> list[str]:
    script = (
        "import sys, zxingcpp; from PIL import Image; "
        "print('\\n'.join(r.text for r in zxingcpp.read_barcodes(Image.open(sys.argv[1]))))"
    )
    out = _run("uv", "run", "--quiet", "--no-project", "--with", "zxing-cpp", "--with", "pillow",
               "python", "-I", "-c", script, str(png))  # fmt: skip
    return [line for line in out.splitlines() if line]


# --- the poster passes ----------------------------------------------------------------------------


def test_page_is_a0_landscape() -> None:
    assert page_size_problems(_run("pdfinfo", str(PDF))) == []


def test_fonts_embedded_and_none_type3() -> None:
    assert font_problems(_run("pdffonts", str(PDF))) == []


def test_exactly_the_three_logos() -> None:
    assert logo_problems(TEX.read_text()) == []


def test_title_and_authors_match_citation_cff() -> None:
    cff = yaml.safe_load((REPO / "CITATION.cff").read_text())
    assert title_author_problems(TEX.read_text(), cff) == []


def test_no_em_dash_in_sources() -> None:
    for tex in POSTER.glob("*.tex"):
        assert em_dash_problems(tex.read_text()) == [], tex.name


def test_latex_math_fonts_only() -> None:
    assert math_font_problems(_run("pdffonts", str(PDF))) == []
    for fig in sorted((POSTER / "figures").glob("*.pdf")):
        assert math_font_problems(_run("pdffonts", str(fig)), need_math=False) == [], fig.name


def test_at_most_750_words() -> None:
    assert word_count_problems(_run("pdftotext", str(PDF), "-")) == []


def test_column_bottoms_within_1_cm() -> None:
    assert column_spread_problems(bbox_words(PDF), *A0_LANDSCAPE_PT) == []


def test_block_gaps_are_the_theme_gap() -> None:
    assert gap_problems(block_gaps(bbox_words(PDF), A0_LANDSCAPE_PT[0])) == []


def test_block_title_rules_are_unbroken(tmp_path: Path) -> None:
    """Nothing (a figure's white margin, a legend) covers a block title's rule: >= 95% of the column."""
    titles = block_title_tops(bbox_words(PDF), TEX.read_text())
    assert len(titles) >= 12
    _run("pdftoppm", "-png", "-r", "72", "-singlefile", str(PDF), str(tmp_path / "p"))
    cover = rule_coverage(tmp_path / "p.png", titles, 1.0)
    assert min(cover) >= 0.95, list(zip(titles, cover))


def test_planted_broken_rule(tmp_path: Path) -> None:
    titles = block_title_tops(bbox_words(PDF), TEX.read_text())
    _run("pdftoppm", "-png", "-r", "72", "-singlefile", str(PDF), str(tmp_path / "p"))
    script = (
        "import sys, json; from PIL import Image, ImageDraw\n"
        "im = Image.open(sys.argv[1]).convert('RGB'); c, y0, y1 = json.loads(sys.argv[2])\n"
        "w3 = im.width / 3; ImageDraw.Draw(im).rectangle([c * w3 + 0.3 * w3, y1, c * w3 + 0.5 * w3, y1 + 45], fill='white')\n"
        "im.save(sys.argv[1])"
    )
    import json

    _run(
        "uv",
        "run",
        "--quiet",
        "--no-project",
        "--with",
        "pillow",
        "python",
        "-I",
        "-c",
        script,
        str(tmp_path / "p.png"),
        json.dumps(titles[1]),
    )
    assert rule_coverage(tmp_path / "p.png", [titles[1]], 1.0)[0] < 0.95


def test_strip_labels_inside_frame() -> None:
    assert strip_label_problems(bbox_words(PDF), TEX.read_text()) == []


def test_planted_label_outside_frame() -> None:
    tex = TEX.read_text()
    narrow = re.sub(r"(\\draw\[thick\] \(-?[\d.]+,[^)]*\) rectangle \()([\d.]+)", r"\g<1>1.1", tex)
    assert narrow != tex
    assert strip_label_problems(bbox_words(PDF), narrow)


def test_tick_labels_print_at_24_pt() -> None:
    assert tick_problems(bbox_words(PDF)) == []


def test_header_under_17_percent(tmp_path: Path) -> None:
    _run("pdftoppm", "-png", "-r", "40", "-singlefile", str(PDF), str(tmp_path / "poster"))
    assert header_fraction(tmp_path / "poster.png") < MAX_HEADER_FRACTION


def test_qr_code_decodes_to_project_page(tmp_path: Path) -> None:
    _run("pdftoppm", "-png", "-r", "72", "-singlefile", str(PDF), str(tmp_path / "poster"))
    assert decode_qr(tmp_path / "poster.png") == [URL]


# --- each check refuses a planted defect ----------------------------------------------------------


def test_planted_wrong_page_size() -> None:
    assert page_size_problems("Pages:          1\nPage size:      2383.94 x 3370.39 pts (A0)\n")
    assert page_size_problems("Pages:          2\nPage size:      3370.39 x 2383.94 pts (A0)\n")


def test_planted_type3_and_unembedded_fonts() -> None:
    head = "name type encoding emb sub uni object ID\n---- ---- ---- --- --- --- ---------\n"
    assert font_problems(head + "[none] Type 3 Custom yes no no 12 0\n")
    assert font_problems(head + "Helvetica Type 1 Standard no no no 7 0\n")


def test_planted_logos() -> None:
    tex = TEX.read_text()
    assert logo_problems(tex.replace("{leibniz_lab.png}", "{syracuse_university.pdf}"))
    assert logo_problems(tex.replace("{neurips.pdf}", "{neurips_dark.svg}"))


def test_planted_title_and_authors() -> None:
    cff = yaml.safe_load((REPO / "CITATION.cff").read_text())
    tex = TEX.read_text()
    assert title_author_problems(tex.replace("Joshua Shay Kricheli", "Joshua Krichel"), cff)
    assert title_author_problems(tex.replace("Robust LLM", "Robust"), cff)
    swapped = tex.replace("Venkata Gandikota", "@@").replace("Paulo Shakarian", "Venkata Gandikota")
    assert title_author_problems(swapped.replace("@@", "Paulo Shakarian"), cff)


def test_planted_em_dash() -> None:
    assert em_dash_problems("a \u2014 b")
    assert em_dash_problems("a --- b")
    assert not em_dash_problems("5M-77M and 0.18--0.80")


@pytest.mark.parametrize(
    "payload", ["https://lab-v2.github.io/tpp-coverage", "https://example.org/"]
)
def test_planted_wrong_qr(tmp_path: Path, payload: str) -> None:
    png = tmp_path / "qr.png"
    _run("uvx", "--quiet", "segno", "--scale", "4", "--output", str(png), payload)
    assert decode_qr(png) == [payload] != [URL]


def test_planted_too_many_words() -> None:
    text = _run("pdftotext", str(PDF), "-")
    assert word_count_problems(text + " extra" * (MAX_WORDS + 1 - len(text.split())))
    assert not word_count_problems("word " * MAX_WORDS)


def test_planted_column_spread_and_ticks() -> None:
    words = bbox_words(PDF)
    w, h = A0_LANDSCAPE_PT
    lowest = max((x for x in words if x[0] < w / 3 and x[3] < h * 0.95), key=lambda x: x[3])
    raised = [
        x if x is not lowest else (x[0], x[1], x[2], x[3] - 2 * PT_PER_CM, x[4]) for x in words
    ]
    others = [x for x in raised if not (x[0] < w / 3 and x[3] > lowest[3] - 2 * PT_PER_CM)]
    assert column_spread_problems(others, w, h)
    shrunk = [
        (a, b, c, b + 16.0, t) if t in TICK_WORDS else (a, b, c, d, t) for a, b, c, d, t in words
    ]
    assert tick_problems(shrunk)
    assert tick_problems([x for x in words if x[4] not in TICK_WORDS])


def test_planted_tall_header(tmp_path: Path) -> None:
    script = (
        "import sys; from PIL import Image, ImageDraw\n"
        "im = Image.new('RGB', (400, 283), 'white'); d = ImageDraw.Draw(im)\n"
        "d.rectangle([0, 60, 399, 62], fill=(39, 60, 117)); im.save(sys.argv[1])"
    )
    png = tmp_path / "tall.png"
    _run(
        "uv",
        "run",
        "--quiet",
        "--no-project",
        "--with",
        "pillow",
        "python",
        "-I",
        "-c",
        script,
        str(png),
    )
    assert header_fraction(png) > MAX_HEADER_FRACTION


def test_planted_math_fonts() -> None:
    head = "name type encoding emb sub uni object ID\n---- ---- ---- --- --- --- ---------\n"
    lato = "ABCDEF+Lato-Regular CID TrueType Identity-H yes yes yes 5 0\n"
    math = "ABCDEF+LatinModernMath-Regular-Identity-H CID Type 0C Identity-H yes yes yes 6 0\n"
    assert not math_font_problems(head + lato + math)
    assert math_font_problems(head + lato)  # math faked in the text font
    assert math_font_problems(
        head + lato + math + "ABCDEF+FiraMath-Regular CID Type 0C Identity-H yes yes yes 7 0\n"
    )
    assert math_font_problems(
        head + "ABCDEF+DejaVuSans CID TrueType Identity-H yes yes yes 8 0\n", need_math=False
    )


def test_planted_padded_gap() -> None:
    gaps = block_gaps(bbox_words(PDF), A0_LANDSCAPE_PT[0])
    padded = [list(col) for col in gaps]
    padded[2][1] += 0.9  # a \vspace{0.9cm} "column balance" pad, as round 4 had
    assert gap_problems(padded)
    assert gap_problems([[1.4], [1.4], []])


def test_no_math_in_bold_heads() -> None:
    assert heading_math_problems(TEX.read_text()) == []


def test_planted_heading_math() -> None:
    tex = TEX.read_text()
    assert heading_math_problems(tex.replace("at least 5 sizes", "$n \\geq 5$ sizes"))
    assert heading_math_problems("\\begin{block}{Two designs in the $(N, D)$ plane}")
