#!/bin/bash
# N-gram KL divergence analysis (requires pre-tokenized datasets from the downloaders)
#
# Validates that train/val splits are statistically equivalent by computing
# symmetric KL divergence and cosine similarity across n-gram distributions.
#
# Expected layout under $DATA_DIR:
#   $DATA_DIR/{wikipedia,redpajama,cosmopedia,pes2o,c4}/{train,val}_{tokens,lengths}.bin
#
# Runs all 15 jobs (5 datasets × 3 n-gram orders) concurrently, throttled by
# --max-parallel. Each job is an independent Python subprocess. Skipping of
# already-completed jobs is handled inside ngram_kl_parallel.py.
#
# Usage:
#   export DATA_DIR=/path/to/your/data
#   bash src/tpp_analysis/ngram_analysis/run_all_ngrams.sh
#
# Tuning:
#   MAX_PARALLEL  Concurrent (dataset, n) subprocesses (default: 5)
#   WORKERS       Within-job chunk-counting workers   (default: 8)

BASE_DIR="${DATA_DIR:?Set DATA_DIR to the directory containing tokenized .bin files}"
OUT_DIR="ngram_results"
SCRIPT="src/tpp_analysis/ngram_analysis/ngram_kl_parallel.py"

MAX_PARALLEL="${MAX_PARALLEL:-5}"
WORKERS="${WORKERS:-8}"

mkdir -p "$OUT_DIR"

echo "========== START: $(date) =========="
echo "  base-dir:     $BASE_DIR"
echo "  output-dir:   $OUT_DIR"
echo "  max-parallel: $MAX_PARALLEL  (concurrent (dataset, n) subprocesses)"
echo "  workers:      $WORKERS  (within-job chunk parallelism)"
echo "==========================================="

python "$SCRIPT" \
    --mode run-all \
    --ngrams 1 2 3 \
    --base-dir "$BASE_DIR" \
    --output-dir "$OUT_DIR" \
    --workers "$WORKERS" \
    --max-parallel "$MAX_PARALLEL"

echo
echo "Completed: $(ls "$OUT_DIR"/*.json 2>/dev/null | wc -l) / 15"
echo "========== ALL DONE at $(date) =========="

python "$SCRIPT" --mode aggregate --output-dir "$OUT_DIR"