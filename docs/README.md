# Guides

[README](../README.md)

Step-by-step guides for the paper's main pipeline. Each experiment folder documents itself in its
own README.

| Guide | What it covers |
|---|---|
| [DOWNLOADING](DOWNLOADING.md) | Downloading the released metrics, checkpoints and loss tables |
| [TRAINING](TRAINING.md) | Tokenizing the corpora, writing the grid's jobs and training a run |
| [ANALYSIS](ANALYSIS.md) | Fitting the laws and writing the tables |
| [VISUALIZATIONS](VISUALIZATIONS.md) | The figures |
| [NGRAM_KL](NGRAM_KL.md) | The n-gram KL divergence between train and validation splits |
| [KNOWN_BEHAVIOURS](KNOWN_BEHAVIOURS.md) | What changes a reproduced number: padding, BF16, GPU count, seed, released runs, model sizes, run names, cached fits, drivers |

## Full pipeline

| Step | Command | Guide |
|---|---|---|
| 1. Download and tokenize the corpora | `src/tpp_analysis/dataset_downloaders/download_{wiki,cosmopedia,redpj,peso,c4}.py` | [TRAINING](TRAINING.md#1-corpora) |
| 2. Write the jobs of the trained grid | `src/tpp_analysis/generate_jobs/generate_jobs.py --dataset <DATASET>` | [TRAINING](TRAINING.md#2-the-trained-grid) |
| 3. Train (4 GPUs per run) | `src/training_models/training/dist_trainer_main.py` under `torchrun` | [TRAINING](TRAINING.md#3-training-one-run) |
| 4. Extract per-epoch losses | `src/tpp_analysis/get_results/get_results.py from_local <METRICS_DIR>` | [DOWNLOADING](DOWNLOADING.md) |
| 5. Fit the laws, write the tables | `run_analysis.py`, `run_special_analysis.py`, `run_prelim_bigtpp.py`, `generate_appendix_tables.py` | [ANALYSIS](ANALYSIS.md) |
| 6. Figures | `run_appendix_figures.sh`, notebooks in `visualizations/` | [VISUALIZATIONS](VISUALIZATIONS.md) |

Each table and figure of the paper maps to its command in
[Paper item to command](ANALYSIS.md#paper-item-to-command).

Steps 1-3 need GPUs (about 3,600 GPU-hours in total). Step 4 can start from the released
metrics: `download_and_extract.py` without `--results-only` fetches them (~0.5 GB).

## Experiment folders

The experiment folders (`camera_ready/*`, `large_scale/`, `cluster/`) share one layout: a
README saying what the folder holds and how to run it, the scripts, and the committed outputs
in a `results` folder. The grid classification keeps its `results.csv` beside its scripts,
where its pre-registered protocol names it. Their tests sit in [`tests/<folder>/`](../tests/README.md).
