#!/bin/bash
# One-shot health snapshot of the queue sweep. Exits 0; prints ALERT lines for anything that
# needs action. watchdog.sh fixes the mechanical cases (retry, resubmit); this is for a person,
# or a session polling at intervals, to reason about what is flagged (a systematic bug, a bad
# learning rate, a wrong architecture).
#
#   JOBID=<job id> bash cluster/queue/health_check.sh   # JOBID defaults to your tpp-queue job
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
JOBID=${JOBID:-$(squeue -h -u "$USER" -n tpp-queue -o %i 2>/dev/null | head -1)}

if [ -n "$JOBID" ]; then
    st=$(squeue -h -j "$JOBID" -o "%T %M %R" 2>/dev/null)
    [ -n "$st" ] && echo "job $JOBID: $st" || echo "ALERT: job $JOBID not in queue"
else
    echo "ALERT: no tpp-queue job in the queue"
fi
n_wd=$(pgrep -cf "watchdog.sh" 2>/dev/null || echo 0)
[ "$n_wd" -eq 1 ] || echo "ALERT: watchdog count=$n_wd (want exactly 1; two double-resubmit)"

"$VENV/bin/python" "$WORK_DIR/cluster/queue/monitor.py" --health
