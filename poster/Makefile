# Build the NeurIPS 2026 poster (A0 landscape): SVG logos from ../images to PDF, the QR code of the
# project page, then latexmk with xelatex into build/; only the final PDF is copied next to the source.
# `make png` renders the virtual poster (2880 px high) and the thumbnail (320 px wide) into build/.
URL := https://lab-v2.github.io/tpp-coverage/
LOGOS := build/assets/neurips.pdf build/assets/syracuse_ai_institute.pdf
NAME := tpp_neurips2026_poster
# Absolute search paths: xdvipdfmx runs inside build/, so a relative image path would not resolve;
# the generated logos and QR code sit in build/assets/, out of reach of xelatex's own build/ lookup.
export TEXINPUTS := $(CURDIR)/theme//:$(CURDIR)/build/assets//:$(CURDIR)/figures//:$(CURDIR)/../images//:

$(NAME).pdf: $(NAME).tex $(LOGOS) build/assets/qr_project_page.pdf $(wildcard theme/*.sty) $(wildcard figures/*.pdf)
	latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build $<
	cp build/$@ $@

build/assets/%.pdf: ../images/%.svg
	mkdir -p build/assets && uvx --quiet cairosvg $< -o $@

# segno draws the QR code as SVG and cairosvg converts it: xdvipdfmx cannot read segno's own PDF.
build/assets/qr_project_page.pdf:
	mkdir -p build/assets && uvx --quiet segno --scale 10 --border 1 --output build/assets/qr_project_page.svg "$(URL)"
	uvx --quiet cairosvg build/assets/qr_project_page.svg -o $@

# NeurIPS limits: virtual poster landscape, at most 5120 x 2880 px and 10 MB; thumbnail at most
# 320 x 256 px and 5 MB. A0 landscape is 1.414:1, so the virtual poster is bound by its 2880 px
# height (4073 px wide) and the thumbnail by its 320 px width (226 px high).
png: build/$(NAME)_virtual.png build/$(NAME)_thumbnail.png
build/$(NAME)_virtual.png: $(NAME).pdf
	pdftoppm -png -singlefile -scale-to-x -1 -scale-to-y 2880 $< build/$(NAME)_virtual
build/$(NAME)_thumbnail.png: $(NAME).pdf
	pdftoppm -png -singlefile -scale-to-x 320 -scale-to-y -1 $< build/$(NAME)_thumbnail
.PHONY: png

# `make figures TPP_DATA_ROOT=<root>` redraws the two data panels in the poster style from the released
# losses, fits and seed-variance results under <root> (extracted_losses/, full_coverage_results/,
# seed_variance_results/; `download_and_extract.py --results-only` fetches them), with the paper's
# own plotting code (`FIG1_STYLE=poster`), and copies them into figures/.
FIG1 := ../src/tpp_analysis/visualizations/figure1_panels.py
POSTER_PANELS := poster_convergence_Red_Pajama_Kaplan_First.pdf poster_isoflop_Red_Pajama_Kaplan_First.pdf
figures:
	@test -n "$(TPP_DATA_ROOT)" || { echo "figures: set TPP_DATA_ROOT=<root>"; exit 2; }
	cd .. && TPP_DATA_ROOT="$(TPP_DATA_ROOT)" FIG1_STYLE=poster FIG1_PREFIX=poster_ \
		FIG1_DATASET="Red Pajama" FIG1_LAW=Kaplan FIG1_MODE=First uv run --locked python $(FIG1:../%=%)
	for f in $(POSTER_PANELS); do cp "$(TPP_DATA_ROOT)/full_coverage_results/figures/$$f" figures/; done
.PHONY: figures

clean:
	rm -rf build
.PHONY: clean
