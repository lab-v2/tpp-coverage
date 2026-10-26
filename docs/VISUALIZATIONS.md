# Figures

[README](../README.md) · [Analysis](ANALYSIS.md)

All figure code is in `src/tpp_analysis/visualizations/`. It reads the analysis outputs at the
repository root (`extracted_losses/`, `full_coverage_results/`, `seed_variance_results/`,
`special_results/`), so run the [analysis](ANALYSIS.md) or `download_and_extract.py
--results-only` first. Set `TPP_DATA_ROOT` to read them from somewhere else.

| Script or notebook | Produces |
|---|---|
| `figure1_panels.py` | the three panels of Figure 1 (surface, convergence, isoFLOP) for one combination |
| `run_appendix_figures.sh` | Figure 1 and the 16 appendix three-panel figures (calls `figure1_panels.py`) |
| `visuals.ipynb` | the main-text figures and the top-ranked three-panel figures |
| `special_viz_notebook_bf16_v2.ipynb` | the BF16 appendix figures |
| `../subset_analysis/aggregate_seed_runs.py` | the kappa_target win-rate sweeps |
| `../utility/plot_nlayers_distribution.py` | the n-layers distribution |

```bash
uv run bash src/tpp_analysis/visualizations/run_appendix_figures.sh
FIG1_DATASET=Wikipedia FIG1_LAW=Chinchilla FIG1_MODE=First FIG1_PREFIX=custom_ \
    uv run python src/tpp_analysis/visualizations/figure1_panels.py
```

Figures go to `full_coverage_results/figures/` as PDF and PNG.

The notebooks need the `notebooks` extra; they resolve the repository root from their own
folder, so start Jupyter there:

```bash
uv sync --extra notebooks
cd src/tpp_analysis/visualizations && uv run jupyter lab
```

Shared plotting code: `isoflop_panel.py` (isoFLOP curves and the residual panel),
`panel_style.py` (sizes, fonts, legends), `surface_helpers.py` (3D surfaces and holdout styles).

The n-layers figure needs the architecture audit of every released epoch-3 checkpoint
(download them first, see [DOWNLOADING](DOWNLOADING.md)):

```bash
uv run python src/tpp_analysis/utility/audit_architectures.py colinear_scaling_models/ \
    --only-epoch 3 --out architecture_audit_e3.csv
uv run python src/tpp_analysis/utility/plot_nlayers_distribution.py --mode runs \
    --audit-csv architecture_audit_e3.csv --out fig_nlayers_runs
```
