# Build the NeurIPS 2026 poster (140 x 90 cm landscape, the NeurIPS maximum): SVG logos from ../images to PDF, the QR code of the
# project page, then latexmk with xelatex into build/; only the final PDF is copied next to the source.
# Every PDF build re-renders the PNGs (`make png` alone does too): the virtual poster (2880 px high) and the thumbnail (320 px wide) into build/.
URL := https://lab-v2.github.io/tpp-coverage/
LOGOS := build/assets/neurips.pdf build/assets/syracuse_ai_institute.pdf
NAME := tpp_neurips2026_poster
# Absolute search paths: xdvipdfmx runs inside build/, so a relative image path would not resolve;
# the generated logos and QR code sit in build/assets/, out of reach of xelatex's own build/ lookup.
export TEXINPUTS := $(CURDIR)/theme//:$(CURDIR)/build/assets//:$(CURDIR)/figures//:$(CURDIR)/../images//:

$(NAME).pdf: $(NAME).tex $(LOGOS) build/assets/qr_project_page.pdf $(wildcard theme/*.sty) $(wildcard figures/*.pdf)
	latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build $<
	cp build/$@ $@
	# Committed stamp: the PDF's sha256 and its overfull \vbox count, so a checkout without TeX
	# can verify the committed PDF (tests/poster reads it).
	printf 'pdf_sha256 %s\noverfull_vbox %s\n' "$$(sha256sum $@ | cut -d' ' -f1)" "$$(grep -c '^Overfull \\vbox' build/$(NAME).log || true)" > build_stamp.txt
	rm -f build/$(NAME)_virtual.png build/$(NAME)_thumbnail.png && $(MAKE) --no-print-directory -o $(NAME).pdf png

build/assets/%.pdf: ../images/%.svg
	mkdir -p build/assets && uvx --quiet cairosvg $< -o $@

# segno draws the QR code as SVG and cairosvg converts it: xdvipdfmx cannot read segno's own PDF.
build/assets/qr_project_page.pdf:
	mkdir -p build/assets && uvx --quiet segno --scale 10 --border 1 --output build/assets/qr_project_page.svg "$(URL)"
	uvx --quiet cairosvg build/assets/qr_project_page.svg -o $@

# NeurIPS limits: virtual poster landscape, at most 5120 x 2880 px and 10 MB; thumbnail at most
# 320 x 256 px and 5 MB. 140 x 90 cm is 1.556:1, so the virtual poster is bound by its 2880 px
# height (4480 px wide) and the thumbnail by its 320 px width (206 px high).
png: build/$(NAME)_virtual.png build/$(NAME)_thumbnail.png
# The PDF's sha256 at render time sits beside the PNG; tests/poster checks it against the current PDF.
# Each PNG's bytes and pixel size go into the committed build_stamp.txt for the NeurIPS limits gate.
build/$(NAME)_virtual.png: $(NAME).pdf
	pdftoppm -png -singlefile -scale-to-x -1 -scale-to-y 2880 $< build/$(NAME)_virtual
	sha256sum $< | cut -d' ' -f1 > build/$(NAME)_virtual.png.pdf-sha256
	printf '%s %s %s\n' virtual_png "$$(stat -c %s $@)" "$$(file -b $@ | grep -o '[0-9]* x [0-9]*' | tr -d ' ')" >> build_stamp.txt
build/$(NAME)_thumbnail.png: $(NAME).pdf
	pdftoppm -png -singlefile -scale-to-x 320 -scale-to-y -1 $< build/$(NAME)_thumbnail
	printf '%s %s %s\n' thumbnail_png "$$(stat -c %s $@)" "$$(file -b $@ | grep -o '[0-9]* x [0-9]*' | tr -d ' ')" >> build_stamp.txt
.PHONY: png

clean:
	rm -rf build
.PHONY: clean
