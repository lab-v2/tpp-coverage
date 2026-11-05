#!/bin/bash
# Locations shared by every script of the queue: source it, never repeat a path.
#
#   . "$(dirname "${BASH_SOURCE[0]}")/env.sh"
#
# Every value is overridable from the environment; the defaults are inside this repository.
#
#   QUEUE_DIR   the run directory: tasks/, claims/, done/, failed/, retries/, rundirs/, logs/
#               and shared/ (the trainer's second copy of metrics, checkpoints, heartbeats)
#   WORK_DIR    the checkout the trainer runs from          (default: this repository)
#   DATA_DIR    tokenized corpora, one folder per dataset  (default: $WORK_DIR/corpora)
#   VENV        the Python environment with torch           (default: $WORK_DIR/.venv)
#   SHARED_DIR  the trainer's shared copy                    (default: $QUEUE_DIR/shared)

_queue_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORK_DIR=${WORK_DIR:-$(cd "$_queue_here/../.." && pwd)}
export QUEUE_DIR=${QUEUE_DIR:-$WORK_DIR/runs/queue}
export DATA_DIR=${DATA_DIR:-$WORK_DIR/corpora}
export VENV=${VENV:-$WORK_DIR/.venv}
export SHARED_DIR=${SHARED_DIR:-$QUEUE_DIR/shared}
export PYTHONPATH=$WORK_DIR/src${PYTHONPATH:+:$PYTHONPATH}

# Fail loudly: a missing path is a visible error here, not a job that starts and dies inside
# torch.distributed with a FileNotFoundError forty lines deep.
queue_require_env() {
    local missing=0
    if [ ! -d "$DATA_DIR" ]; then
        echo "env.sh: DATA_DIR does not exist: $DATA_DIR" >&2
        missing=1
    fi
    if [ ! -x "$VENV/bin/python" ]; then
        echo "env.sh: no python in VENV: $VENV" >&2
        missing=1
    fi
    if [ ! -d "$QUEUE_DIR/tasks" ]; then
        echo "env.sh: no task queue at $QUEUE_DIR/tasks" >&2
        missing=1
    fi
    [ "$missing" -eq 0 ] || {
        echo "env.sh: set QUEUE_DIR, DATA_DIR or VENV, or stage the data first." >&2
        return 1
    }
    return 0
}
