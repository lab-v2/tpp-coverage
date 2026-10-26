# Training the models from scratch

[README](../README.md) · [Downloading](DOWNLOADING.md) · [Analysis](ANALYSIS.md)

You do not need this to reproduce the paper's tables and figures: the metrics of every run
are released (see [DOWNLOADING](DOWNLOADING.md)). Retraining everything takes about 3,600
GPU-hours on 4-GPU nodes, ~0.5 TB for the tokenized corpora and ~1.6 TB for checkpoints.

## 0. Environment

`uv sync` installs the pinned stack, including PyTorch 2.5.1 built for CUDA 12.4 on x86_64,
and `transformers` 5.7.0 (later 5.x releases refuse PyTorch < 2.6).

On aarch64 GPU nodes (for example NVIDIA GH200), PyPI has no CUDA build of PyTorch 2.5, so use
the cluster's PyTorch and install the remaining pins from the lock file:

```bash
module load <your cluster's CUDA and PyTorch 2.5 modules>
uv venv .venv-train --python "$(which python)" --system-site-packages
uv export --frozen --no-dev --no-emit-package torch --no-hashes -o train-requirements.txt
VIRTUAL_ENV=.venv-train uv pip install -r train-requirements.txt
source .venv-train/bin/activate
python -c "import torch; assert torch.cuda.is_available() and torch.distributed.is_nccl_available()"
```

The released runs used PyTorch 2.5 with CUDA 12.4 on GH200 GPUs (driver ~550). Training is
bitwise reproducible on a fixed driver; across drivers or GPU models expect per-epoch losses
to differ by up to ~1e-2.

## 1. Corpora

Each downloader fetches a corpus, tokenizes it with `cl100k_base`, splits it into disjoint
train and validation sets with a seed-42 permutation, and writes `{train,val}_tokens.bin` and
`{train,val}_lengths.bin` plus a `split_config.json`. Output goes to `$DATA_DIR/<dataset>/`,
or to `corpora/<dataset>/` in the repository when `DATA_DIR` is unset; the trainer reads from
the same place.

```bash
uv run python src/tpp_analysis/dataset_downloaders/download_wiki.py         # wikimedia/wikipedia 20231101.en
uv run python src/tpp_analysis/dataset_downloaders/download_cosmopedia.py   # HuggingFaceTB/cosmopedia web_samples_v2
uv run python src/tpp_analysis/dataset_downloaders/download_redpj.py        # RedPajama-Data-V2, 2023-06 en head_middle
uv run python src/tpp_analysis/dataset_downloaders/download_peso.py         # allenai/peS2o v2
uv run python src/tpp_analysis/dataset_downloaders/download_c4.py           # allenai/c4 en
```

`allenai/peS2o` is a loading-script dataset, which is why `datasets` is pinned below 4.0.

## 2. The trained grid

`generate_jobs.py` writes one SLURM script per group of runs and a `submit_all.sh`:

```bash
uv run python src/tpp_analysis/generate_jobs/generate_jobs.py --dataset wikipedia --list   # print the runs
uv run python src/tpp_analysis/generate_jobs/generate_jobs.py --dataset c4 --partition gpu --account myproj
bash slurm_scripts/c4_main_fp32/submit_all.sh
```

| Design flag | Runs | What |
|---|---|---|
| `--design main` (default) | 324 per corpus | the five-corpus experiment |
| `--design main --precision bf16 --dataset wikipedia` | 324 | the BF16 replication (adds `--use-bf16`) |
| `--design bigtpp --dataset wikipedia` | 156 | the single-epoch BigTPP preliminary |

The main design, per corpus:

- **Sizes.** Training: 5, 10, 14, 19, 22, 28, 32, 35, 40 and 45M target parameters. Holdout only:
  54.57, 60.88, 64.04 and 70.36M.
- **NC train**, 10 x 12 = 120 runs: every training size at D = 10, 35, 60, 80, 100, 130, 150,
  180, 200, 225, 250 and 275M tokens.
- **CO train**, 10 x 12 = 120 runs: every training size at D = k N for k = 1, 1.5, 1.9, 2,
  2.5, 2.7, 3, 3.3, 3.5, 4, 4.5 and 5.
- **NC holdout**, 42 runs in an L shape: every training size at D = 300M; the 54.57M size at all
  17 budgets (the 12 above and 300, 325, 350, 375, 400M); the three largest sizes at the five
  holdout budgets.
- **CO holdout**, 42 runs, the same L shape on k: every training size at k = 6; the 54.57M size
  at all 17 ratios (the 12 above and 6, 6.2, 6.5, 6.7, 7); the three largest at the five
  holdout ratios.

Holdout runs train on the validation split (`--use-train-as-val`), so the fitted laws never
see their data. The test `tests/test_release_match.py` checks that the generated run names
equal the released ones for every corpus.

Fixed hyperparameters: 3 epochs (1 for BigTPP), 96 sequences of 256 tokens per GPU on 4 GPUs,
AdamW with weight decay 0.1, 3% warmup then cosine decay, gradient clipping at 1.0, seed 42.
The peak learning rate is the only per-corpus setting:

| Corpus | `--dataset` | Peak learning rate |
|---|---|---|
| Wikipedia | `wikipedia` | 3e-4 |
| Cosmopedia | `cosmopedia` | 1e-3 |
| RedPajama | `redpajama` | 3e-4 |
| peS2o | `pes2o` | 1e-3 |
| C4 | `c4` | 1e-3 |

(The C4 rate is confirmed by the learning-rate schedules recorded in the released metrics.)

Cluster settings come from flags or environment variables, never from the code:
`--partition`/`SLURM_PARTITION`, `--account`/`SLURM_ACCOUNT`, `--exclude`/`SLURM_EXCLUDE`,
`--time`/`SLURM_TIME`, `--work-dir`/`WORK_DIR` (checkout on the cluster, default this
repository), `--data-dir`/`DATA_DIR` (default `<work dir>/corpora`) and `--venv`/`VENV`
(default `<work dir>/.venv`).

## 3. Training one run

```bash
uv run torchrun --nproc_per_node=4 --standalone src/training_models/training/dist_trainer_main.py \
    --model-size 10000000 --tokens 35000000 --learning-rate 3e-4 --weight-decay 0.1 \
    --epochs 3 --batch-size 96 --max-seq-len 256 --dataset wikipedia --architecture llama \
    --skip-val --output-prefix WIKI_NxD_GRID_N10M_D35M
```

Add `--use-train-as-val` (instead of `--skip-val`) for holdout runs and `--use-bf16` for BF16.
Checkpoints go to `saved_checkpoints/`, per-batch metrics to `training_metrics/`, and a second
copy of both to `$SHARED_DIR` (else `$DATA_DIR`, else `shared_backup/`).

`--model-size` is a target: the trainer picks the nearest valid `LlamaForCausalLM` shape, so
realized sizes differ slightly:

| Target | Realized N | (layers, d_model, d_ff, heads, head dim) |
|---|---|---|
| 5,000,000 | 5,035,656 | (24, 24, 96, 12, 2) |
| 10,000,000 | 10,070,160 | (12, 48, 192, 1, 48) |
| 14,000,000 | 14,411,456 | (24, 64, 256, 32, 2) |
| 19,000,000 | 19,400,928 | (1, 96, 384, 1, 96) |
| 22,000,000 | 22,796,832 | (24, 96, 384, 24, 4) |
| 28,000,000 | 28,819,840 | (12, 128, 512, 4, 32) |
| 32,000,000 | 32,498,720 | (1, 160, 640, 16, 10) |
| 35,000,000 | 35,368,160 | (8, 160, 640, 4, 40) |
| 40,000,000 | 40,277,184 | (3, 192, 768, 12, 16) |
| 45,000,000 | 47,949,888 | (16, 192, 768, 1, 192) |
| 54,566,912 | 55,538,432 | (4, 256, 1024, 32, 8) |
| 60,884,992 | 63,931,136 | (12, 256, 1024, 32, 8) |
| 64,044,032 | 68,127,488 | (16, 256, 1024, 32, 8) |
| 70,362,112 | 76,520,192 | (24, 256, 1024, 1, 256) |

Run names matter: `get_results.py` classifies a run by the `NxD_GRID`, `COLINEAR`, `HOLDOUT`
and `COLINEAR_HOLDOUT` tokens in its prefix and finds its corpus from the `_<dataset>_m<N>_`
tag the trainer appends. `generate_jobs.py` uses the released names (`WIKI_`, `cosmopedia_`,
`RP_`, `PES2O_`, `C4_`, with `BF16` or `BIGTPP_` in front for those designs).

See the README's "Known behaviours" for how the loss treats padding and what `--use-bf16` does.

## 4. Checking a retrained run against the released one

```bash
uv run python src/training_models/training/verify_reproducibility.py compare --from-hf \
    --reproduced training_metrics/WIKI_NxD_GRID_N10M_D35M_*_metrics.npz
uv run python src/training_models/training/verify_reproducibility.py weights --from-hf \
    --reproduced saved_checkpoints/WIKI_NxD_GRID_N10M_D35M_*_epoch3_*.pt
```

`--from-hf` downloads only the matching released file (cached in `~/.cache/tpp_verify/`).
Pass `--atol 1e-2` on hardware other than the reference. Then extract the losses
([DOWNLOADING](DOWNLOADING.md#extracting-losses-from-your-own-runs)) and run the
[analysis](ANALYSIS.md).
