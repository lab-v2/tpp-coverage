#!/bin/bash
#SBATCH --job-name=tpp-bench
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# End-to-end check before a sweep: one real training run at protocol settings
# (N = 153,028,096, requested by its legacy count 140,644,352; D = 40M; 1 epoch; Wikipedia).
# It exercises CUDA torch and NCCL, data loading, the heartbeat, metrics and the checkpoint.
#
#   cd <repository root>; mkdir -p slurm_logs
#   sbatch --partition=<p> --account=<a> cluster/queue/bench_one_run.sbatch
. "${SLURM_SUBMIT_DIR:-.}/cluster/queue/env.sh"
set -x
BENCH_DIR=${BENCH_DIR:-$QUEUE_DIR/bench}
mkdir -p "$BENCH_DIR/rundir" && cd "$BENCH_DIR/rundir" || exit 1

echo "node=$(hostname)  T_START=$(date +%s)"
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'nccl', torch.cuda.nccl.version())"

time "$VENV/bin/python" -m torch.distributed.run --nproc_per_node=4 --standalone \
    "$WORK_DIR/src/training_models/training/dist_trainer_main.py" \
    --model-size 140644352 \
    --tokens 40000000 \
    --learning-rate 3e-4 \
    --weight-decay 0.1 \
    --epochs 1 \
    --batch-size 96 \
    --max-seq-len 256 \
    --dataset wikipedia \
    --architecture llama \
    --skip-val \
    --shared-dir "$BENCH_DIR" \
    --output-prefix BENCH_N153.03M_D40M

echo "T_END=$(date +%s)"
echo "=== artifacts ==="
ls -la "$BENCH_DIR/heartbeats/" "$BENCH_DIR/training_metrics/" 2>&1 | head -20
cat "$BENCH_DIR"/heartbeats/*.json 2>/dev/null
