# Cluster pipeline: the WIKI300M scale stratum

The paper's main experiments train models of 5M to 77M parameters. This pipeline trained a larger
Wikipedia stratum on the same protocol, to measure whether the gap between non-collinear (NC) and
collinear (CO) designs holds as model size grows: training sizes from 22.8M to 203.4M
parameters, holdout-only sizes from 229.5M to 380.6M, 1 epoch, learning rate 3e-4, weight decay
0.1, 96 sequences of 256 tokens per GPU on 4 GPUs, cosine schedule with 3% warmup. The published
design of the paper (`src/tpp_analysis/generate_jobs/generate_jobs.py`) is untouched; this
stratum has its own generator.

| Folder | What |
|---|---|
| [`queue/`](queue/README.md) | A SLURM task queue: an sbatch template whose nodes are independent training slots, the slot worker, a self-healing watchdog, a health check and a monitor |
| [`wiki300m/`](wiki300m/README.md) | The stratum's task generator and the 468-cell queue that was trained |
| [`audit/`](audit/README.md) | Re-analyses of the stratum's runs (CO versus NC per size stratum, holdout splits, bootstraps, fitter checks) and their result tables |

Every script takes its locations and cluster settings from environment variables or the `sbatch`
command line.

## Running a sweep

```bash
export QUEUE_DIR=/path/to/run DATA_DIR=/path/to/corpora VENV=/path/to/venv
uv run python cluster/wiki300m/generate_tasks.py "$QUEUE_DIR"     # the first 100 cells; cluster/wiki300m/tasks/ is the 468 trained
mkdir -p slurm_logs
sbatch --partition=<partition> --account=<account> cluster/queue/run_queue.sbatch
JOBID=<job id> SBATCH_ARGS="--partition=<partition> --account=<account>" \
    setsid nohup bash cluster/queue/watchdog.sh >> "$QUEUE_DIR/logs/watchdog.log" 2>&1 < /dev/null &
python cluster/queue/monitor.py --watch 60
```

Run from the repository root. A resubmitted job resumes the queue where the last one stopped:
the queue's state is its `done/`, `claims/` and `failed/` marker directories.

## Protocol invariants

- Wikipedia only, the pre-tokenized corpus chunked at 256 tokens.
- Training cells use `--skip-val` (the train split); holdout cells `--use-train-as-val` (the
  validation split), as in the paper's generator.
- One model per node of 4 GPUs: the GPU count sets the global batch and the step count the
  learning-rate schedule follows, so another count trains a different model
  ([Known behaviours](../docs/KNOWN_BEHAVIOURS.md#gpu-count-and-the-learning-rate-schedule)).
  On GPUs with less memory, `--grad-accum-steps` keeps the per-step batch.
- For sizes the paper did not train, `--model-size` takes the legacy parameter count, which is
  what the size search matches on ([Known behaviours](../docs/KNOWN_BEHAVIOURS.md#model-sizes));
  `generate_tasks.py` checks that each one maps back to the intended parameter count.
