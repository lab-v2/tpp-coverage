#!/bin/bash
# One slot = one node. Claims tasks atomically (mkdir), runs them one after another, verifies
# the output, marks them done. Re-runnable: done tasks are skipped, so a resubmitted job resumes
# the queue where it stopped.
#
#   bash cluster/queue/slot_worker.sh <slot id>
#
# Protocol settings (overridable): GPUS_PER_SLOT=4, LEARNING_RATE=3e-4, WEIGHT_DECAY=0.1,
# EPOCHS=1, BATCH_SIZE=96 (per GPU), MAX_SEQ_LEN=256, DATASET=wikipedia, ARCH=llama.
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
queue_require_env || exit 1
set -u
SLOT_ID=${1:-0}
GPUS_PER_SLOT=${GPUS_PER_SLOT:-4}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-96}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-256}
DATASET=${DATASET:-wikipedia}
ARCH=${ARCH:-llama}
B=$QUEUE_DIR
mkdir -p "$B"/{claims,done,failed,rundirs} "$SHARED_DIR"

echo "[slot $SLOT_ID] $(hostname) starting $(date)"
while true; do
    picked=""
    for tf in "$B"/tasks/*.task; do
        name=$(basename "$tf" .task)
        [ -e "$B/done/$name" ] && continue
        [ -e "$B/failed/$name" ] && continue
        if mkdir "$B/claims/$name" 2>/dev/null; then
            picked=$name; break
        fi
    done
    [ -z "$picked" ] && { echo "[slot $SLOT_ID] queue drained $(date)"; break; }

    # Parse KEY=VALUE; NOT `source`: seed cells carry SPLIT_FLAG="--skip-val --seed 43",
    # which `source` reads as an assignment prefix plus a command, leaving SPLIT_FLAG unset
    # (fatal under set -u).
    PREFIX= ; MODEL_SIZE= ; TOKENS= ; SPLIT_FLAG= ; EST_NH=
    while IFS='=' read -r k v; do
        v="${v%\"}"; v="${v#\"}"      # strip optional surrounding quotes
        case $k in
            PREFIX) PREFIX=$v ;;
            MODEL_SIZE) MODEL_SIZE=$v ;;
            TOKENS) TOKENS=$v ;;
            SPLIT_FLAG) SPLIT_FLAG=$v ;;
            EST_NH) EST_NH=$v ;;
        esac
    done < "$B/tasks/$picked.task"
    if [ -z "$PREFIX" ] || [ -z "$MODEL_SIZE" ] || [ -z "$TOKENS" ] || [ -z "$SPLIT_FLAG" ]; then
        echo "task parse FAILED $(date)" > "$B/failed/$picked"
        echo "[slot $SLOT_ID] PARSE-FAIL $picked"; continue
    fi
    echo "[slot $SLOT_ID] START $picked (est ${EST_NH}h) $(date)"
    echo "$(hostname) $(date +%s)" > "$B/claims/$picked/host"

    RUNDIR=$B/rundirs/$picked
    mkdir -p "$RUNDIR" && cd "$RUNDIR" || exit 1
    # shellcheck disable=SC2086  # SPLIT_FLAG holds one or more flags
    "$VENV/bin/python" -m torch.distributed.run --nproc_per_node="$GPUS_PER_SLOT" --standalone \
        "$WORK_DIR/src/training_models/training/dist_trainer_main.py" \
        --model-size "$MODEL_SIZE" \
        --tokens "$TOKENS" \
        --learning-rate "$LEARNING_RATE" \
        --weight-decay "$WEIGHT_DECAY" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --max-seq-len "$MAX_SEQ_LEN" \
        --dataset "$DATASET" \
        --architecture "$ARCH" \
        $SPLIT_FLAG \
        --shared-dir "$SHARED_DIR" \
        --output-prefix "$PREFIX" \
        > "$RUNDIR/train.log" 2>&1
    rc=$?

    # done = trainer exit 0 AND final metrics visible in the shared dir
    if [ $rc -eq 0 ] && ls "$SHARED_DIR"/training_metrics/"${PREFIX}"_*_metrics.npz >/dev/null 2>&1; then
        touch "$B/done/$picked"
        rm -rf "$RUNDIR/saved_checkpoints"          # keep only the shared copy
        echo "[slot $SLOT_ID] DONE  $picked $(date)"
    else
        echo "rc=$rc $(hostname) $(date)" > "$B/failed/$picked"
        echo "[slot $SLOT_ID] FAIL  $picked rc=$rc - see $RUNDIR/train.log"
    fi
done
