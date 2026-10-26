# N-gram KL divergence between train and validation splits

[README](../README.md) · [Training](TRAINING.md)

Validates that train/val splits are statistically equivalent by computing symmetric KL divergence and cosine similarity across n-gram distributions (n = 1-5).

This analysis is independent of the scaling-law fitting pipeline. It requires only the pre-tokenized `.bin` files produced by the dataset downloaders.

---

## Prerequisites

The tokenized corpora, one folder per dataset, as the downloaders write them
(see [TRAINING](TRAINING.md#1-corpora)):

```bash
export DATA_DIR=/path/to/corpora     # holds wikipedia/, cosmopedia/, redpajama/, pes2o/, c4/
ls $DATA_DIR/wikipedia/train_tokens.bin $DATA_DIR/wikipedia/val_lengths.bin
```

Run the commands below inside the environment (`source .venv/bin/activate`, or prefix them
with `uv run`).

---

## Quick Start

```bash
export DATA_DIR=/path/to/corpora
bash src/tpp_analysis/ngram_analysis/run_all_ngrams.sh
```

This runs all 15 jobs (5 datasets × 3 n-gram orders), skipping any that already have results, then generates a LaTeX table.

---

## Usage

### Run everything

```bash
export DATA_DIR=/path/to/corpora
bash src/tpp_analysis/ngram_analysis/run_all_ngrams.sh
```

### Run all n-grams for one dataset

```bash
python src/tpp_analysis/ngram_analysis/ngram_kl_parallel.py \
    --mode run-dataset \
    --dataset wikipedia \
    --base-dir $DATA_DIR \
    --output-dir ngram_results/ \
    --workers 8
```

### Run a single (dataset, n) job

```bash
python src/tpp_analysis/ngram_analysis/ngram_kl_parallel.py \
    --mode single \
    --dataset wikipedia \
    --ngram 3 \
    --base-dir $DATA_DIR \
    --output-dir ngram_results/ \
    --workers 8
```

### Generate LaTeX table from existing results

```bash
python src/tpp_analysis/ngram_analysis/ngram_kl_parallel.py \
    --mode aggregate \
    --output-dir ngram_results/
```

### Quick test (capped token budget)

```bash
python src/tpp_analysis/ngram_analysis/ngram_kl_parallel.py \
    --mode single \
    --dataset wikipedia \
    --ngram 1 \
    --base-dir $DATA_DIR \
    --token-budget 50000000
```

---

## Output

Results are written to `ngram_results/` (one JSON per job, plus a LaTeX table):

```
ngram_results/
  wikipedia_ngram_1.json
  wikipedia_ngram_2.json
  ...
  c4_ngram_5.json
  ngram_kl_table.tex          # LaTeX table for the appendix
```

Each JSON contains:

| Field | Description |
|-------|-------------|
| `sym_kl` | Symmetric KL divergence between train and val n-gram distributions |
| `cosine` | Cosine similarity between train and val n-gram frequency vectors |
| `kl_train_val` | KL(train \|\| val) |
| `kl_val_train` | KL(val \|\| train) |
| `unique_types_train` | Number of distinct n-gram types in train |
| `unique_types_val` | Number of distinct n-gram types in val |
| `total_train` | Total n-gram count in train |
| `total_val` | Total n-gram count in val |

---

## Datasets

All five datasets use shuffled-permutation splits targeting ~1.3B tokens per side:

| Dataset | Tokenizer | Tokens/split |
|---------|-----------|-------------|
| Wikipedia | cl100k_base | ~1.8B |
| Cosmopedia | cl100k_base | ~600M |
| Red Pajama | cl100k_base | ~1.2B |
| peS2o | cl100k_base | ~1.2B |
| C4 | cl100k_base | ~1.7B |

---

## Parallelism

The script uses three levels of parallelism:

1. **Dataset level** - each dataset is an independent subprocess
2. **N-gram order level** - each (dataset, n) pair is an independent subprocess
3. **Token counting level** - within each job, counting is parallelized across chunks

`--max-parallel` controls how many (dataset, n) subprocesses run concurrently (default: 8). `--workers` controls the within-job chunk parallelism (default: CPU count / 4).

---

## Binary File Format

The dataset downloaders produce two files per split:

| File | Dtype | Description |
|------|-------|-------------|
| `*_tokens.bin` | uint32 | Padded token sequences (max_seq_len = 256) |
| `*_lengths.bin` | uint16 | Original sequence lengths (for masking padding) |

The n-gram script memory-maps these files and strips padding using the lengths array before counting.

## Storage notes

After the downloaders finish writing `*_tokens.bin` and `*_lengths.bin` to
`$DATA_DIR/<dataset>/`, the HuggingFace cache (default
`~/.cache/huggingface/datasets/`) and per-dataset `shard_cache/` directories
are no longer needed and can be deleted to reclaim disk space.

**Before deleting, verify all 5 datasets produced their bin files:**

```bash
for d in wikipedia redpajama cosmopedia pes2o c4; do
    ls $DATA_DIR/$d/{train,val}_tokens.bin || echo "MISSING: $d"
done
```

If every dataset prints four file paths and no `MISSING` lines appear,
it's safe to clear the caches:

```bash
rm -rf ~/.cache/huggingface/datasets
rm -rf $DATA_DIR/{c4,redpajama}/shard_cache
```

**peS2o is the outlier.** Unlike the other four datasets - which stream
shards on demand and tokenize on the fly - `allenai/peS2o` ships as a
HuggingFace loading script that requires the full `v2` corpus to be
downloaded and converted to Arrow format before random-access tokenization
can begin. On our hardware this resulted in a ~300 GB HuggingFace cache
footprint and roughly 15 minutes of "Generating train split: ..." progress
*after* the download bar finishes and *before* the first tokenization log
line appears. If disk space is tight, run the peS2o downloader last and
clear `~/.cache/huggingface/datasets` immediately afterward. 

Note also that `datasets` is pinned to `<4.0` in `pyproject.toml` because
peS2o v2's loading script is no longer supported in datasets 4.0+. This
pin preserves exact data parity with the trained models.
