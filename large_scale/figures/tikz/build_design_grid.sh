#!/usr/bin/env bash
# Compile the standalone TikZ design grid and export it to figures/design_grid.png.
# Tries pdftoppm (poppler), then magick/convert (ImageMagick), then gs (Ghostscript).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
DPI="${DPI:-300}"
OUT="$HERE/../design_grid.png"

# 1. LaTeX -> PDF (standalone auto-crops to content)
latexmk -pdf -interaction=nonstopmode -halt-on-error design_grid.tex >/dev/null

# 2. PDF -> PNG
if command -v pdftoppm >/dev/null 2>&1; then
  pdftoppm -png -r "$DPI" -singlefile design_grid.pdf "${OUT%.png}"
elif command -v magick >/dev/null 2>&1; then
  magick -density "$DPI" design_grid.pdf -background white -alpha remove -quality 100 "$OUT"
elif command -v convert >/dev/null 2>&1; then
  convert -density "$DPI" design_grid.pdf -background white -alpha remove -quality 100 "$OUT"
elif command -v gs >/dev/null 2>&1; then
  gs -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -r"$DPI" -sOutputFile="$OUT" design_grid.pdf
else
  echo "error: need one of pdftoppm, magick, convert, or gs to rasterize the PDF" >&2
  exit 1
fi

# 3. Clean latexmk aux files (keep the PDF)
latexmk -c design_grid.tex >/dev/null 2>&1 || true

echo "wrote $OUT"
