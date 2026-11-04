#!/bin/bash
#SBATCH --job-name=tpp-queue
#SBATCH --time=34:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Each node of the job is one independent training slot (the protocol trains every model on
# the 4 GPUs of one node); every slot consumes the shared task queue, and the job ends when the
# queue drains. Cluster settings go on the command line, never in this file:
#
#   cd <repository root>; mkdir -p slurm_logs
#   QUEUE_DIR=/path/to/run sbatch --partition=<p> --account=<a> [--exclude=<nodes>] \
#       [--nodes=N] cluster/queue/run_queue.sbatch
#
# Several jobs (for example under two allocations) may share one queue: claims are atomic.
. "${SLURM_SUBMIT_DIR:-.}/cluster/queue/env.sh"
queue_require_env || exit 1
B=$QUEUE_DIR
mkdir -p "$B/logs"
STALE_SECONDS=${STALE_SECONDS:-1800}

echo "job $SLURM_JOB_ID on: $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"

# Stale-claim cleanup, safe when several jobs share the queue: a claim without a done marker is
# not necessarily dead. Release a claim only when its run is provably not alive: no heartbeat
# update for STALE_SECONDS, or no heartbeat at all and the claim itself older than that.
now=$(date +%s)
for c in "$B"/claims/*/; do
    [ -d "$c" ] || continue
    name=$(basename "$c")
    [ -e "$B/done/$name" ] && continue
    pfx=$(grep '^PREFIX=' "$B/tasks/$name.task" 2>/dev/null | cut -d= -f2)
    hb=""
    [ -n "$pfx" ] && hb=$(ls "$SHARED_DIR"/heartbeats/"${pfx}"_*.json 2>/dev/null | head -1)
    if [ -n "$hb" ]; then
        age=$(( now - $(stat -c %Y "$hb") ))
        [ "$age" -lt "$STALE_SECONDS" ] && { echo "keeping LIVE claim (hb ${age}s): $name"; continue; }
    else
        cage=$(( now - $(stat -c %Y "$c") ))
        [ "$cage" -lt "$STALE_SECONDS" ] && { echo "keeping RECENT claim (${cage}s, no hb): $name"; continue; }
    fi
    echo "releasing stale claim: $name"
    rm -rf "$c" "$B/failed/$name"
done

# SLOT_BASE offsets the slot ids so two jobs on one queue write distinct slot logs.
i=${SLOT_BASE:-0}
for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
    srun --nodes=1 --ntasks=1 -w "$node" --exact \
         --cpus-per-task="$SLURM_CPUS_PER_TASK" --gpus-per-node="${SLURM_GPUS_PER_NODE:-4}" --mem=0 \
         bash "$WORK_DIR/cluster/queue/slot_worker.sh" "$i" \
         > "$B/logs/slot${i}_${SLURM_JOB_ID}.log" 2>&1 &
    i=$((i+1))
done
wait
echo "all slots finished $(date)"
echo "done: $(ls "$B/done" 2>/dev/null | wc -l) / $(ls "$B"/tasks/*.task | wc -l)   failed: $(ls "$B/failed" 2>/dev/null | wc -l)"
