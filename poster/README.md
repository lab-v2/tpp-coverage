# poster

The NeurIPS 2026 poster, A0 landscape (118.9 x 84.1 cm, inside the conference limit of 140 x 90 cm),
built with the [gemini](https://github.com/anishathalye/gemini) beamerposter theme (vendored in
`theme/` with its MIT licence). Every text, title and headings included, is set in Latin Modern Roman and
the mathematics in Latin Modern Math, as LaTeX sets them; every font is embedded (no Type 3).

`make` converts the SVG logos in `../images/` to PDF, draws the QR code of the project page, and
builds with `latexmk -xelatex` into `build/`; only `tpp_neurips2026_poster.pdf` is kept next to the
source. `make png` renders the virtual poster (`build/tpp_neurips2026_poster_virtual.png`, 4073 x
2880 px) and the thumbnail (`build/tpp_neurips2026_poster_thumbnail.png`, 320 x 226 px) within the
NeurIPS limits. Needs `latexmk`, `xelatex`, poppler's `pdftoppm` and `uv` (for `cairosvg` and `segno`).

The poster shows only the paper's own figures: `figures/` holds Figure 1 (a), the loss surface, and
Figure 1 (c), the induced isoFLOP curves (Kaplan law on RedPajama, first epoch), copied byte for byte
from the paper's `src/updated_figures/` (rendered by `src/tpp_analysis/visualizations/figure1_panels.py` in Computer Modern, legends outside the axes); their captions are the paper's. The two figures run in a band across columns 2 and 3. The QR code of the project page
sits in the header, beside the NeurIPS logo. `tests/poster/` checks the page size, the fonts (Latin
Modern or Computer Modern only), the three logos, the QR code, the title and authors against
`CITATION.cff`, the absence of em dashes, at most 800 words, a header under 17% of the height, column
bottoms within 1 cm, block gaps within a 0.8 cm band, no math in bold heads, block title rules
unbroken, logo groups centred on the title block within 0.5 cm, the figures being the paper's files (by
SHA-256), set at least 0.98 times their natural width (rendered at paper print size), every figure word printing at least 21 pt, and no TikZ figure, one body text size (captions at
least 26 pt), no overlapping text, and (with `TPP_PAPER_FIGURES=<paper src/updated_figures>`) the copies equal the paper's.
