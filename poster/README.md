# poster

The NeurIPS 2026 poster, 140 x 90 cm landscape (the NeurIPS maximum),
built with the [gemini](https://github.com/anishathalye/gemini) beamerposter theme (vendored in
`theme/` with its MIT licence). Every text, title and headings included, is set in Latin Modern Roman and
the mathematics in Latin Modern Math, as LaTeX sets them; every font is embedded (no Type 3).

`make` converts the SVG logos in `../images/` to PDF, draws the QR code of the project page, and
builds with `latexmk -xelatex` into `build/`; only `tpp_neurips2026_poster.pdf` is kept next to the
source. Every `make` (and `make png`) re-renders the virtual poster (`build/tpp_neurips2026_poster_virtual.png`, 4480 x
2880 px) and the thumbnail (`build/tpp_neurips2026_poster_thumbnail.png`, 320 x 206 px) within the
NeurIPS limits, and records the PDF's sha256 beside the PNG so `tests/poster/` fails on a PNG rendered from an older PDF. Needs `latexmk`, `xelatex`, poppler's `pdftoppm` and `uv` (for `cairosvg` and `segno`).

The poster shows only the paper's own figures: `figures/` holds Figure 1 (a), the loss surface, and
Figure 1 (c), the induced isoFLOP curves (Kaplan law on RedPajama, first epoch), copied byte for byte
from the paper's `src/updated_figures/` (rendered by `src/tpp_analysis/visualizations/figure1_panels.py` in Computer Modern, legends outside the axes); their captions are the paper's. The page has two zones: a top zone of three text columns (column 1 the TL;DR, the designs and the law; column 2 Gauss-Newton conditioning, Proposition 2 with its formulas beside the epsilon table, and Proposition 1; column 3 Theorem 1, the Design recipe and the published designs), whose widths (0.67 / 1.36 / 0.98 of `\colwidth`, separators 0.02425 of the paper width) make them end on one line, and an evidence row across the full width: the headline numbers with the Takeaways and Limitations under them, then the two figures (42 and 32.5 cm wide). The QR code of the project page
sits in the header, beside the NeurIPS logo. `tests/poster/` checks the page size, the fonts (Latin
Modern or Computer Modern only), the three logos, the QR code, the title and authors against
`CITATION.cff`, the absence of em dashes, at most 800 words, a header under 17% of the height, the
top-zone column bottoms within 1 cm, block gaps within a 0.8 cm band, the gap between the zones at most 3.5 cm with no `\vspace`, no overfull `\vbox` (the committed `build_stamp.txt` holds the PDF sha256 and overfull count that `make` wrote), no math in bold heads, block title rules
unbroken, logo groups centred on the title block within 0.5 cm, the figures being the paper's files (by
SHA-256), set at least 0.75 times their natural width (rendered at paper print size), every figure word printing at least 24 pt, and no TikZ figure, one body text size (captions at
least 26 pt), no overlapping text, and (with `TPP_PAPER_FIGURES=<paper src/updated_figures>`) the copies equal the paper's; the virtual PNG must be a render of the current PDF.

## Page size: A0 against 140 x 90 cm

The binding constraint is the page height: all the text sits in the top zone, and the evidence row
takes what is left. Measured with xelatex (body 28.8 pt, scale 1.16):

| Page | Layout | Figures (surface / isoFLOP) | Result |
|---|---|---|---|
| A0, 118.9 x 84.1 cm | band across columns 2-3 (round 15) | 30 / 24.5 cm | fits |
| A0 | evidence row, equal columns | 36 / 29 cm | overfull by 266.5 pt |
| A0 | evidence row, equal columns | 27 / 21 cm | overfull by 48.7 pt |
| 140 x 90 cm | evidence row, equal columns | 36 / 29 cm | overfull by 98.6 pt |
| 140 x 90 cm | evidence row, equal columns | 33 / 26 cm | overfull by 14.8 pt; columns end at 51.9 / 54.2 / 47.0 cm |
| 140 x 90 cm | evidence row, columns 0.80 / 1.50 / 0.70 | 34 / 26.5 cm | overfull by 11.0 pt |
| 140 x 90 cm | evidence row, columns 0.80 / 1.50 / 0.70 | **33.5 / 26 cm** | **fits; columns end at 52.3 / 52.7 / 53.0 cm** (round 17) |
| 140 x 90 cm | the same, body 30 pt (scale 1.21) | 33.5 / 26 cm | overfull by 67.2 pt (not used) |
| 140 x 90 cm | Takeaways in the banner cell, columns 0.67 / 1.36 / 0.98 (round 18) | **42 / 32.5 cm** | **fits; columns end at 46.2 / 46.3 / 46.7 cm** (used) |
| 140 x 90 cm | the same, body 30 pt (scale 1.21) | 42 / 32.5 cm | overfull by 85.9 pt (not used) |

At 42 / 32.5 cm the surface's 3-D box prints about 19.2 cm wide and the isoFLOP axes about 16.4 cm
(axes share of the natural canvas width: 399 / 873 and 442 / 874 pt).
