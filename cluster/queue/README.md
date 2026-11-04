# Task queue

A file-based queue of training runs on SLURM. Each `.task` file in `$QUEUE_DIR/tasks/` is one run
(`PREFIX`, `MODEL_SIZE`, `TOKENS`, `SPLIT_FLAG`, `EST_NH`); a slot claims a task by creating
`claims/<task>/` (atomic `mkdir`), and marks it `done/` or `failed/`.

| File | What |
|---|---|
| `env.sh` | the locations every script sources (`QUEUE_DIR`, `WORK_DIR`, `DATA_DIR`, `VENV`, `SHARED_DIR`), with defaults inside the repository, and `queue_require_env` |
| `run_queue.sbatch` | the job: releases provably dead claims, then runs one `slot_worker.sh` per node |
| `slot_worker.sh` | claims tasks, trains each with `torchrun` at protocol settings, checks that the metrics landed |
| `watchdog.sh` | retries failed cells (capped), resubmits a job that died, reports stalled heartbeats and loss problems |
| `health_check.sh` | a one-shot snapshot with ALERT lines |
| `monitor.py` | queue counts, live heartbeats and loss sanity; used by the two scripts above |
| `bench_one_run.sbatch` | one real run at protocol settings, to check a cluster before a sweep |
