"""Collision check inside a figure PDF, run with PyMuPDF (`uv run --with pymupdf`).

Prints one JSON list of problems for the PDF in argv[1]:
* two words from different lines of text that touch or overlap (gap under MIN_GAP_PT);
* a word that a drawn line segment, rectangle edge or image crosses (by more than
  MIN_GAP_PT inside its box), e.g. a tick label running into a colorbar; only the part of a
  path inside its clip rectangle and not under a later opaque fill (a legend face) counts,
  and neither the page background nor an unstroked white fill is a line.
* an axis label more than MAX_LABEL_GAP tick-label heights from its tick row.
Boxes are words; superscripts and subscripts share their label's text line, so a label's pieces
never count against each other.
"""

import json
import re
import sys

import pymupdf

MIN_GAP_PT = 1.0


def _crosses(p, q, box, clip=None, covers=()) -> bool:
    for i in range(41):
        t = i / 40
        x, y = p.x + (q.x - p.x) * t, p.y + (q.y - p.y) * t
        if clip is not None and not (clip.x0 <= x <= clip.x1 and clip.y0 <= y <= clip.y1):
            continue  # this part of the path is clipped away and never printed
        if any(c.x0 <= x <= c.x1 and c.y0 <= y <= c.y1 for c in covers):
            continue  # an opaque fill drawn later (a legend's face) hides it
        if box[0] < x < box[2] and box[1] < y < box[3]:
            return True
    return False


def _segments(page):
    """Drawn line segments with the clip rectangle in force for each (None when unclipped).

    Matplotlib draws a curve whole and clips it to the axes, so a segment past the axes edge
    is not on the page; the page background (a rectangle covering the page) is no line either.
    """
    out, clips, opaque = [], [], []  # clips: (level, rect) stack; opaque: (order, rect)
    area = page.rect.width * page.rect.height
    drawings = page.get_drawings(extended=True)
    for k, d in enumerate(drawings):
        if (
            d["type"] in ("f", "fs")
            and d.get("fill_opacity", 1) == 1
            and d["rect"].width * d["rect"].height < 0.95 * area
        ):
            opaque.append((k, d["rect"]))
    for k, d in enumerate(drawings):
        level = d.get("level", 0)
        while clips and clips[-1][0] >= level:
            clips.pop()
        if d["type"] == "clip":
            clips.append((level, d["scissor"]))
            continue
        if d["type"] == "group":
            continue
        if d["rect"].width * d["rect"].height >= 0.95 * area:
            continue
        if d.get("color") is None and d.get("fill") == (1.0, 1.0, 1.0):
            continue  # an unstroked white fill (an axes background) has no visible edge
        clip = None
        for _, r in clips:
            clip = r if clip is None else clip & r
        covers = tuple(r for j, r in opaque if j > k)
        for it in d["items"]:
            if it[0] == "l":
                out.append((it[1], it[2], clip, covers))
            elif it[0] == "re":
                r = it[1]
                out += [
                    (a, b, clip, covers)
                    for a, b in ((r.tl, r.tr), (r.tr, r.br), (r.br, r.bl), (r.bl, r.tl))
                ]
    return out


NUMERIC = re.compile(r"^[0-9.,-]+$")
MAX_LABEL_GAP = 2.0  # an axis label sits within this many tick-label heights of its tick row


def floating_labels(page) -> list[str]:
    """Axis labels farther than MAX_LABEL_GAP tick heights from the nearest tick label in line with them.

    Tick labels are numeric words; an axis label is a text line with letters, outside every stroked
    frame (a legend's entries are not axis labels), that has tick labels beside it or above or below.
    """
    words = page.get_text("words")
    ticks = [w for w in words if NUMERIC.match(w[4])]
    if not ticks:
        return []
    h = sorted(w[3] - w[1] for w in ticks)[len(ticks) // 2]
    area = page.rect.width * page.rect.height
    frames = [
        d["rect"]
        for d in page.get_drawings()
        if d.get("color") is not None and d["rect"].width * d["rect"].height < 0.5 * area
    ]
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(sp["text"] for sp in line["spans"]).strip()
            if not re.search(r"[A-Za-z]", text):
                continue
            b = pymupdf.Rect(line["bbox"])
            if any(f.contains(b) for f in frames):
                continue
            gaps = []
            for t in ticks:
                tb = pymupdf.Rect(t[:4])
                if min(b.x1, tb.x1) > max(b.x0, tb.x0):  # stacked: label above or below the row
                    gaps.append(max(tb.y0 - b.y1, b.y0 - tb.y1))
                elif min(b.y1, tb.y1) > max(b.y0, tb.y0):  # side by side
                    gaps.append(max(tb.x0 - b.x1, b.x0 - tb.x1))
            if gaps and min(gaps) > MAX_LABEL_GAP * h:
                out.append(f"label {text!r} floats {min(gaps) / h:.1f} tick heights from its ticks")
    return out


def problems(path: str) -> list[str]:
    page = pymupdf.open(path)[0]
    # Words, keyed by their text line: a label's superscripts and subscripts share its line, and
    # word boxes (not line boxes) keep a slanted row of tick labels from spanning a wide envelope.
    words = [(tuple(w[:4]), w[4], (w[5], w[6])) for w in page.get_text("words")]
    out = []
    for i, (a, ta, la) in enumerate(words):
        for b, tb, lb in words[i + 1 :]:
            if la == lb:
                continue
            gx = max(a[0], b[0]) - min(a[2], b[2])
            gy = max(a[1], b[1]) - min(a[3], b[3])
            if gx < MIN_GAP_PT and gy < MIN_GAP_PT:
                out.append(f"text {ta!r} touches text {tb!r}")
    segments = _segments(page)
    images = [page.get_image_bbox(im) for im in page.get_images(full=True)]
    g = MIN_GAP_PT
    for box, text, _ in words:
        inner = (box[0] + g, box[1] + g, box[2] - g, box[3] - g)
        if any(_crosses(p, q, inner, c, cv) for p, q, c, cv in segments):
            out.append(f"text {text!r} crossed by a drawn line")
        for ib in images:
            if min(inner[2], ib.x1) > max(inner[0], ib.x0) and min(inner[3], ib.y1) > max(
                inner[1], ib.y0
            ):
                out.append(f"text {text!r} on an image")
    out += floating_labels(page)
    return sorted(set(out))


if __name__ == "__main__":
    print(json.dumps(problems(sys.argv[1])))
