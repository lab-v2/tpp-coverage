# poster

The NeurIPS 2026 poster, A0 landscape (118.9 x 84.1 cm, inside the conference limit of 140 x 90 cm),
built with the [gemini](https://github.com/anishathalye/gemini) beamerposter theme (vendored in
`theme/` with its MIT licence; one local change loads Raleway by file name). Text is Lato with
Raleway headings, mathematics is Computer Modern, and every font is embedded (no Type 3).

`make` converts the SVG logos in `../images/` to PDF, draws the QR code of the project page, and
builds with `latexmk -xelatex` into `build/`; only `tpp_neurips2026_poster.pdf` is kept next to the
source. `make png` renders the virtual poster (`build/tpp_neurips2026_poster_virtual.png`, 4073 x
2880 px) and the thumbnail (`build/tpp_neurips2026_poster_thumbnail.png`, 320 x 226 px) within the
NeurIPS limits. Needs `latexmk`, `xelatex`, poppler's `pdftoppm` and `uv` (for `cairosvg` and `segno`).

`figures/` holds the two paper figures the poster shows, panels (b) and (c) of the paper's Figure 1
(Kaplan law on RedPajama, first epoch), with their text drawn as outlines. `tests/poster/` checks the
page size, the fonts, the three logos, the QR code, the title and authors against `CITATION.cff`,
and the absence of em dashes.
