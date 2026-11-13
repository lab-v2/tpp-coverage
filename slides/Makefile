# Build the five-minute NeurIPS 2026 talk: SVG logos from ../images to PDF, the QR code of the project page,
# then latexmk into build/; only the final PDF is copied next to the source.
URL := https://lab-v2.github.io/tpp-coverage/
LOGOS := build/neurips.pdf build/syracuse_university.pdf build/syracuse_ai_institute.pdf

tpp_neurips2026_slides.pdf: tpp_neurips2026_slides.tex $(LOGOS) build/qr_project_page.pdf
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build $<
	cp build/$@ $@

build/%.pdf: ../images/%.svg
	mkdir -p build && uvx --quiet cairosvg $< -o $@

build/qr_project_page.pdf:
	mkdir -p build && uvx --quiet segno --scale 10 --border 1 --output $@ "$(URL)"

clean:
	rm -rf build
.PHONY: clean
