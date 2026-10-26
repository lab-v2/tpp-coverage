# Tokens-per-Parameter Coverage Is Critical for Robust LLM Scaling Law Extrapolation

Code for the NeurIPS 2026 paper by Joshua Shay Kricheli\*, Alexander Lawrence Reid\*,
Soumajyoti Sarkar, Venkata Gandikota and Paulo Shakarian (\* equal contribution).

Scaling laws are fitted on small models and used to predict large ones. The paper shows that
whether such a fit extrapolates depends on the tokens-per-parameter (TPP) ratios D/N that the
training runs cover. A collinear (CO) design, which grows N and D together along a few TPP
rays, leaves the law's parameters badly conditioned and extrapolates poorly; a non-collinear
(NC) design with the same number of runs spread over TPP extrapolates well. We test this on
about 1,900 trained models over five corpora and four scaling laws (Kaplan, Chinchilla,
Droppo-Elibol, and a repeated-data law called Parametric-Epoch in the code), and give a
criterion, computable before any training, that tells which designs are well conditioned.

Everything the analysis needs is public on HuggingFace and needs no account:

- [`TPPIsCriticalFor/colinear_scaling_models`](https://huggingface.co/datasets/TPPIsCriticalFor/colinear_scaling_models):
  training metrics, checkpoints and pre-computed analysis outputs of every trained model;
- [`TPPIsCritical/public_checkpoint_reanalysis`](https://huggingface.co/datasets/TPPIsCritical/public_checkpoint_reanalysis):
  normalised run tables of the ten published studies re-analysed in the camera-ready version.

## Install

[uv](https://docs.astral.sh/uv/) is the only tool needed. One environment (Python 3.10,
pinned in `uv.lock`) runs the analysis, the training and the tests.

```bash
git clone <this repository> tpp-coverage && cd tpp-coverage
uv sync                 # add --extra notebooks for Jupyter
uv run pytest -n auto   # a few minutes; downloads ~2 MB from HuggingFace
```

## Quick start: reproduce the analysis from public data (no GPU)

```bash
# Pre-computed results and loss tables
uv run python src/tpp_analysis/download_and_extract.py --results-only

# Main tables (win rates, R2/RMSE summary) and seed-variance data
uv run python src/tpp_analysis/run_analysis.py all --data-dir extracted_losses/ --n-workers "$(nproc)"
uv run python src/tpp_analysis/generate_appendix_tables.py full_coverage_results/

# BF16 replication and the BigTPP preliminary
uv run python src/tpp_analysis/run_special_analysis.py --data-dir extracted_losses/ --n-workers "$(nproc)"
uv run python src/tpp_analysis/generate_appendix_tables.py special_results/ --special
uv run python src/tpp_analysis/run_prelim_bigtpp.py --data-dir extracted_losses/ --n-workers "$(nproc)"

# Figure 1 and the 16 appendix triptychs
uv run bash src/tpp_analysis/visualizations/run_appendix_triptychs.sh
```

`run_analysis.py` and `run_special_analysis.py` render from the downloaded per-combination
parquets. Add `--clear-cache` to refit every law from the loss tables instead; the refit
reproduces the released parquets bit for bit (the 30-seed main fit takes about 20 minutes on
50 cores).

## Full pipeline

| Step | Command | Guide |
|---|---|---|
| 1. Download and tokenize the five corpora | `src/tpp_analysis/dataset_downloaders/download_{wiki,cosmopedia,redpj,peso,c4}.py` | [TRAINING](docs/TRAINING.md#1-corpora) |
| 2. Write the SLURM jobs of the trained (N, D) grid | `src/tpp_analysis/generate_jobs/generate_jobs.py --dataset <name>` | [TRAINING](docs/TRAINING.md#2-the-trained-grid) |
| 3. Train (4 GPUs per run) | `src/training_models/training/dist_trainer_main.py` under `torchrun` | [TRAINING](docs/TRAINING.md#3-training-one-run) |
| 4. Extract per-epoch losses | `src/tpp_analysis/get_results/get_results.py from_local <metrics dir>` | [DOWNLOADING](docs/DOWNLOADING.md) |
| 5. Fit the four laws, write the tables | `run_analysis.py`, `run_special_analysis.py`, `run_prelim_bigtpp.py`, `generate_appendix_tables.py` | [ANALYSIS](docs/ANALYSIS.md) |
| 6. Subset enumeration (Regime-A table) | `dispatch_seeds.sh`, `aggregate_seed_runs.py`, `per_law_winrate_ci.py` | [ANALYSIS](docs/ANALYSIS.md#regime-a-subset-enumeration) |
| 7. Figures | `run_appendix_triptychs.sh`, notebooks in `visualizations/` | [VISUALIZATIONS](docs/VISUALIZATIONS.md) |

Steps 1-3 need GPUs (about 3,600 GPU-hours for every run). Step 4 can start from the released
metrics instead: `download_and_extract.py` without `--results-only` downloads them (~0.5 GB)
and extracts the loss tables.

## Paper item to command

Script paths are under `src/tpp_analysis/` unless noted; outputs land at the repository root.

| Paper item | Command | Output |
|---|---|---|
| Main win-rate table (97.3% NC wins) and R2/RMSE summary | `run_analysis.py full_coverage` | `full_coverage_results/table_winrate.tex`, `table_summary_full.tex` |
| Appendix per-dataset metrics, fitted parameters, win rate by holdout split | `run_analysis.py full_coverage`, then `generate_appendix_tables.py full_coverage_results/` | `appendix_full_metrics.tex`, `appendix_fitted_params.tex`, `table_winrate_by_holdout_split_appendix.tex` |
| BF16 tables (98.7%) and BF16 appendix tables | `run_special_analysis.py`, then `generate_appendix_tables.py special_results/ --special` | `special_results/*.tex` |
| BigTPP preliminary (63.3%) | `run_prelim_bigtpp.py` | `prelim_bigtpp_results/table_winrate_bigtpp_prelim.tex` |
| Regime-A subset table | `dispatch_seeds.sh 0 21`, `subset_analysis/aggregate_seed_runs.py`, `subset_analysis/per_law_winrate_ci.py --mode first` | `subset_bbox_seeds/aggregated/per_law_ci.csv`, `per_dataset_ci.csv` |
| Figure 1 (surface, convergence, isoFLOP) and appendix triptychs | `run_analysis.py seed_variance`, then `visualizations/run_appendix_triptychs.sh` | `full_coverage_results/figures/` |
| BF16 figures | `visualizations/special_viz_notebook_bf16_v2.ipynb` | `special_results/` |
| Model-architecture table, n-layers figure | `utility/audit_architectures.py` over the checkpoints, then `utility/plot_nlayers_distribution.py --mode runs` | CSV, PDF |
| n-gram KL table | `ngram_analysis/run_all_ngrams.sh` (needs the tokenized corpora) | `ngram_results/` |
| Published-design classification (24 designs: 13 Regime B, 11 Regime A) | `camera_ready/grid_classification/classify.py` | `camera_ready/grid_classification/results.csv` |
| Public-checkpoint reanalysis (40% extra error per tenfold shortfall; V_K ratio 3.7 at >= 1B) | `camera_ready/public_checkpoints/` | [its README](camera_ready/public_checkpoints/README.md) |

The theorems, the severity, eigenvalue-expansion and R_min tables, and the compute and storage
totals are analytic or arithmetic; no script computes them.

## Structure

- `src/` - the code that trained the models and analysed them (details below)
- `camera_ready/` - the two analyses added for the camera-ready version
- `docs/` - guides for downloading, training, analysis, figures and the n-gram check
- `tests/` - the pytest suite (`uv run pytest -n auto`)

```
src/
  training_models/              Llama-style models, DDP trainer, checkpoints
    training/dist_trainer_main.py       training entry point
    training/verify_reproducibility.py  compare a retrained run with the released one
  tpp_analysis/                 everything after training
    dataset_downloaders/        download and tokenize the five corpora
    generate_jobs/              SLURM scripts for the trained (N, D) designs
    get_results/                per-epoch losses from training metrics
    download_and_extract.py     public metrics and results from HuggingFace
    scaling_laws/               law forms, fitting, holdout splits, tables
    run_analysis.py             main fits (full coverage, seed variance)
    run_special_analysis.py     BF16 replication
    run_prelim_bigtpp.py        BigTPP preliminary
    generate_appendix_tables.py appendix tables from the fitted parquets
    subset_analysis/            Regime-A subset enumeration and its intervals
    incremental_coverage/       coverage-fraction analysis used by the figures
    ngram_analysis/             corpus n-gram KL divergence
    visualizations/             figure scripts and notebooks
    utility/                    architecture audit and n-layers plot
camera_ready/
  grid_classification/          Proposition-2 classification of 24 published designs
  public_checkpoints/           reanalysis of ten public training-loss releases
docs/                           DOWNLOADING, TRAINING, ANALYSIS, VISUALIZATIONS, NGRAM_KL
tests/                          pytest suite: uv run pytest -n auto
```

## Known behaviours

These produced the paper's numbers and are kept as they are.

- **Padding is scored.** The trainer's cross-entropy has no `ignore_index`, so label positions
  that are padding count toward the loss. Reported losses are means over all 256 positions of
  each sequence, padding included, and sit below the loss on real tokens.
- **BF16 is a full cast.** `--use-bf16` calls `model.bfloat16()`: weights, activations and
  optimizer state are BF16 and only the logits are cast to FP32 for the loss. There is no
  autocast and no FP32 master copy, although the paper calls this mixed precision.
- **Four GPUs per run.** The global batch is 96 sequences per GPU, and the learning-rate
  schedule follows the per-GPU step count, so another GPU count trains a different model.
- **One training seed.** Every model is trained with seed 42. The seeds of the analysis are
  fitting restarts and subset draws.
- **Released runs.** C4 has 321 of its 324 runs (three NC holdout runs are missing). Epoch-1
  checkpoints exist for only part of the runs; epoch-2 and epoch-3 checkpoints for all.
- **RedPajama split.** The RedPajama downloader shuffles shards, not individual documents.
- **Regime-A intervals.** `per_law_winrate_ci.py` reports a seed bootstrap (5,000 resamples),
  and its kappa_target grid is a fixed `np.logspace`.
- **`seed_variance_all.parquet`** has no dataset/law/mode columns; use the per-combination
  parquets beside it.

## Citation

```bibtex
@inproceedings{kricheli2026tpp,
  title     = {Tokens-per-Parameter Coverage Is Critical for Robust {LLM} Scaling Law Extrapolation},
  author    = {Kricheli, Joshua Shay and Reid, Alexander Lawrence and Sarkar, Soumajyoti and
               Gandikota, Venkata and Shakarian, Paulo},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  note      = {TODO: volume, pages, arXiv identifier and URL once published}
}
```

## Authors and license

Joshua Shay Kricheli and Alexander Lawrence Reid contributed equally; Leibnitz Lab, Syracuse
University. The code is released under version 2 of the GNU General Public License (`LICENSE`;
copyright notice in `COPYRIGHT`). The datasets on HuggingFace carry their own licenses.
