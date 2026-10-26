# Downloading the released data

[README](../README.md) · [Training](TRAINING.md) · [Analysis](ANALYSIS.md)

Every trained model's metrics, checkpoints and the pre-computed analysis are public at
[`TPPIsCriticalFor/colinear_scaling_models`](https://huggingface.co/datasets/TPPIsCriticalFor/colinear_scaling_models).
No account or token is needed; a token in `HF_TOKEN` is used if present (higher rate limits).

## Three ways in

```bash
# A. Pre-computed results only (fastest; tables and figures render from these)
uv run python src/tpp_analysis/download_and_extract.py --results-only

# B. Training metrics (~0.5 GB), then extract per-epoch losses into extracted_losses/
uv run python src/tpp_analysis/download_and_extract.py
uv run python src/tpp_analysis/download_and_extract.py --include-special --include-prelim  # + BF16, BigTPP

# C. Everything, checkpoints included (~1.6 TB)
uv run hf download TPPIsCriticalFor/colinear_scaling_models --repo-type dataset \
    --local-dir colinear_scaling_models
uv run python src/tpp_analysis/download_and_extract.py --skip-download --repo-dir colinear_scaling_models
```

Other flags: `--datasets wikipedia c4` (subset of corpora), `--include-checkpoints` (adds the
`.pt` files to B), `--include-results` (B plus A in one call). Run
`download_and_extract.py --help` for the full list.

## Layout on HuggingFace

```
training_metrics/<dataset>/{colinear,non_colinear}/N_<n>/*_metrics.npz   per-batch losses
<dataset>/{colinear,non_colinear}/N_<n>/*_epoch{1,2,3}_*.pt            checkpoints
extracted_losses/  full_coverage_results/  seed_variance_results/
special_results/  prelim_bigtpp_results/  subset_bbox_seeds/              analysis outputs
```

`<dataset>` is `wikipedia`, `cosmopedia`, `redpajama`, `pes2o`, `c4`, `wikipedia_bf16` (BF16
replication) or `wikipedia_bigtpp` (BigTPP preliminary). Holdout runs sit beside the training
runs and carry `HOLDOUT` in their names.

## Extracting losses from your own runs

```bash
uv run python src/tpp_analysis/get_results/get_results.py from_local path/to/training_metrics -o extracted_losses
```

The loss of a run at the end of an epoch is the mean of its last 100 batch losses. Keep each
design (main, BF16, BigTPP) in its own metrics directory: runs are matched to a corpus by
name, so a BigTPP Wikipedia run would otherwise join the Wikipedia table.

## Loss table format

`extracted_losses/<dataset>.csv` holds one block per run:

```
N=5035656, D=5012972, WD=0.1 [colinear]
run_type=colinear,is_holdout=False,is_colinear=True
Epoch,Train Loss,Val Loss,Cumulative Time (s)
1,7.1234,,12.5
...
```

`N` is the realized parameter count, `D` the tokens per epoch, and `run_type` one of
`nxd_grid` (NC train), `colinear` (CO train), `holdout` (NC holdout) or `colinear_holdout`.
