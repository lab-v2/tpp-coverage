#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  dispatch_seeds.sh - run a range of seeds sequentially on one box
# ═══════════════════════════════════════════════════════════════════
#
# Usage:
#   ./dispatch_seeds.sh <start_seed> <end_seed_inclusive> [out_base_dir] [n_workers]
#
# Examples:
#   # Full paper run: seeds 0..21, one worker per core
#   ./dispatch_seeds.sh 0 21 subset_bbox_seeds "$(nproc)"
#
#   # Quick smoke test - just seed 0
#   ./dispatch_seeds.sh 0 0 subset_bbox_seeds "$(nproc)"
#
# Seeds run sequentially; each uses all requested workers. Output goes to
#   <out_base_dir>/seed_<N>/subset_bbox_<dataset>.csv
# and run logs to
#   <out_base_dir>/seed_<N>/run.log
#
# If a seed's directory already contains CSVs, the script will pick up
# where it left off (existing subset_masks are cached).
#
# Requires: extracted_losses/{wikipedia,cosmopedia,redpajama,pes2o,c4}.csv
# from get_results.py.

set -e

START_SEED=${1:?usage: dispatch_seeds.sh <start> <end> [out_dir] [workers]}
END_SEED=${2:?usage: dispatch_seeds.sh <start> <end> [out_dir] [workers]}
OUT_BASE=${3:-"subset_bbox_seeds"}
N_WORKERS=${4:-$(nproc)}

# Resolve script dir so this works from any cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBSET_SCRIPT="$SCRIPT_DIR/subset_analysis/subset_analysis_bbox_seeded.py"

if [ ! -f "$SUBSET_SCRIPT" ]; then
    echo "ERROR: $SUBSET_SCRIPT not found"
    exit 1
fi

if [ ! -d "extracted_losses" ]; then
    echo "ERROR: extracted_losses/ not found in $(pwd)"
    echo "Run get_results.py first, or cd to the directory containing it."
    exit 1
fi

echo "════════════════════════════════════════════════════════════"
echo "  DISPATCH: seeds $START_SEED..$END_SEED"
echo "  Out base: $OUT_BASE"
echo "  Workers per seed: $N_WORKERS"
echo "  Script:   $SUBSET_SCRIPT"
echo "  Host: $(hostname)  Started: $(date)"
echo "════════════════════════════════════════════════════════════"

mkdir -p "$OUT_BASE"

for SEED in $(seq $START_SEED $END_SEED); do
    SEED_DIR="$OUT_BASE/seed_$SEED"
    mkdir -p "$SEED_DIR"

    echo ""
    echo "──────────────────────────────────────────────────────"
    echo "  SEED $SEED  -  $(date)"
    echo "──────────────────────────────────────────────────────"

    PYTHONUNBUFFERED=1 python "$SUBSET_SCRIPT" \
        --seed "$SEED" \
        --out-dir "$OUT_BASE" \
        --n-workers "$N_WORKERS" \
        --data-dir extracted_losses/ \
        2>&1 | tee "$SEED_DIR/run.log"

    echo "  seed $SEED complete  -  $(date)"
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  All seeds $START_SEED..$END_SEED done  -  $(date)"
echo "════════════════════════════════════════════════════════════"