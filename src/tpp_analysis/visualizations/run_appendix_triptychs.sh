#!/usr/bin/env bash
# Regenerate the 16 standard appendix triptychs with fixed (rank, ds, law, mode)
# tuples that match the appendix figure file names of the paper (figures_dump.tex).
# Prefix numbering is locked here (no rank-shuffling between runs).

set -euo pipefail

SCRIPT="$(dirname "$(realpath "$0")")/figure1_panels.py"

run() {
    local prefix="$1" ds="$2" law="$3" mode="$4"
    echo "=== $prefix $ds / $law / $mode ==="
    FIG1_PREFIX="${prefix}_" \
    FIG1_DATASET="$ds" \
    FIG1_LAW="$law" \
    FIG1_MODE="$mode" \
    python "$SCRIPT"
}

run 01 'Red Pajama'  'Kaplan'        'First'
run 02 'Wikipedia'   'Kaplan'        'First'
run 03 'Wikipedia'   'Kaplan'        'Second'
run 04 'Wikipedia'   'Kaplan'        'Final'
run 05 'Pes2o'       'Kaplan'        'First'
run 06 'C4'          'Chinchilla'    'First'
run 07 'Wikipedia'   'Droppo-Elibol' 'First'
run 08 'Red Pajama'  'Droppo-Elibol' 'First'
run 09 'Red Pajama'  'Chinchilla'    'First'
run 10 'Cosmopedia'  'Droppo-Elibol' 'First'
run 11 'Wikipedia'   'Chinchilla'    'First'
run 12 'Red Pajama'  'Kaplan'        'Final'
run 13 'Red Pajama'  'Kaplan'        'Second'
run 14 'C4'          'Droppo-Elibol' 'First'
run 15 'Pes2o'       'Kaplan'        'Final'
run 17 'Pes2o'       'Droppo-Elibol' 'First'

echo
echo "All 16 triptychs rendered."
