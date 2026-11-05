#!/bin/bash
# Queue watchdog: poll the sweep, and self-heal what is safe to self-heal.
#
#   FAILED TASK   -> release the failed/ and claims/ markers so a live slot re-runs it, at most
#                    MAX_RETRY times per cell, so a permanently broken cell cannot burn the
#                    whole allocation.
#   JOB DIED      -> resubmit the sbatch if cells remain; the queue resumes from its markers,
#                    so this loses at most the in-flight cells. At most MAX_RESUB times.
#   STALE BEAT    -> reported, not killed: the slot is blocked inside torchrun, and killing it
#                    from outside risks a half-written checkpoint. A heartbeat stale for three
#                    times STALE_MIN (a job that hit its walltime mid-cell) releases the claim,
#                    under the same retry cap.
#   LOSS          -> NaN/inf, or a loss still near ln(vocab) past 15% of the run: reported only,
#                    the cause is upstream (learning rate, architecture, data).
#
# Lines starting ALERT/ACTION are the event stream. Run exactly one, detached (from the
# repository root; a plain `nohup &` from a tool shell can be reaped, setsid is not):
#
#   JOBID=<job id> SBATCH_ARGS="--partition=<p> --account=<a>" \
#       setsid nohup bash cluster/queue/watchdog.sh >> "$QUEUE_DIR/logs/watchdog.log" 2>&1 < /dev/null &
#
# A second job sharing the queue is watched too when JOBID2 (and SBATCH_ARGS2) are set; it is
# resubmitted while more than four cells remain.
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
set -u
B=$QUEUE_DIR
SBATCH=${SBATCH:-$WORK_DIR/cluster/queue/run_queue.sbatch}
SBATCH_ARGS=${SBATCH_ARGS:-}
SBATCH_ARGS2=${SBATCH_ARGS2:-$SBATCH_ARGS}
INTERVAL=${INTERVAL:-300}
STALE_MIN=${STALE_MIN:-25}
MAX_RETRY=${MAX_RETRY:-2}
MAX_RESUB=${MAX_RESUB:-3}
JOBID=${JOBID:?set JOBID to the queue job to watch}
JOBID2=${JOBID2:-}
resubs=0
resubs2=0
mkdir -p "$B/logs" "$B/retries"

log() { echo "[$(date '+%F %T %Z')] $*"; }

log "watchdog start: job=$JOBID${JOBID2:+ companion=$JOBID2} interval=${INTERVAL}s stale=${STALE_MIN}m"

while true; do
    ntask=$(ls "$B"/tasks/*.task 2>/dev/null | wc -l)
    ndone=$(ls "$B/done" 2>/dev/null | wc -l)
    nfail=$(ls "$B/failed" 2>/dev/null | wc -l)
    nclaim=$(ls "$B/claims" 2>/dev/null | wc -l)
    remaining=$(( ntask - ndone ))

    # ---------- all work accounted for ----------
    if [ "$remaining" -le 0 ]; then
        log "ALERT SWEEP COMPLETE: done=$ndone failed=$nfail of $ntask"
        exit 0
    fi

    # ---------- failed cells: release for retry ----------
    if [ "$nfail" -gt 0 ]; then
        for f in "$B"/failed/*; do
            [ -e "$f" ] || continue
            name=$(basename "$f")
            rc=$(head -1 "$B/failed/$name" 2>/dev/null)
            n=$(cat "$B/retries/$name" 2>/dev/null || echo 0)
            if [ "$n" -lt "$MAX_RETRY" ]; then
                echo $((n+1)) > "$B/retries/$name"
                rm -f "$B/failed/$name"
                rm -rf "$B/claims/$name"
                log "ACTION RETRY ($((n+1))/$MAX_RETRY) $name - was: $rc"
            else
                log "ALERT GIVING UP on $name after $MAX_RETRY retries - was: $rc"
                # the failed/ marker stays, so it is not retried again
            fi
        done
    fi

    # ---------- job alive? ----------
    if ! squeue -h -j "$JOBID" -t R,PD 2>/dev/null | grep -q .; then
        state=$(sacct -j "$JOBID" -X -n --format=State 2>/dev/null | head -1 | tr -d ' ')
        log "ALERT JOB $JOBID GONE (state=$state) with $remaining cells left"
        if [ "$resubs" -lt "$MAX_RESUB" ]; then
            # shellcheck disable=SC2086  # SBATCH_ARGS holds several flags
            new=$(cd "$WORK_DIR" && sbatch --parsable $SBATCH_ARGS "$SBATCH" 2>&1)
            if [[ "$new" =~ ^[0-9]+$ ]]; then
                resubs=$((resubs+1))
                JOBID=$new
                log "ACTION RESUBMITTED as job $JOBID ($resubs/$MAX_RESUB)"
            else
                log "ALERT RESUBMIT FAILED: $new"
            fi
        else
            log "ALERT resubmit cap $MAX_RESUB reached - needs a human"
            exit 1
        fi
    fi

    # ---------- companion job alive? (keeps its slots until the queue nearly drains) ----------
    if [ -n "$JOBID2" ] && [ "$remaining" -gt 4 ] && \
            ! squeue -h -j "$JOBID2" -t R,PD 2>/dev/null | grep -q .; then
        if [ "$resubs2" -lt "$MAX_RESUB" ]; then
            # shellcheck disable=SC2086
            new2=$(cd "$WORK_DIR" && sbatch --parsable $SBATCH_ARGS2 "$SBATCH" 2>&1)
            if [[ "$new2" =~ ^[0-9]+$ ]]; then
                resubs2=$((resubs2+1)); JOBID2=$new2
                log "ACTION RESUBMITTED companion as job $JOBID2 ($resubs2/$MAX_RESUB)"
            fi
        fi
    fi

    # ---------- stalled runs ----------
    # Iterate over IN-FLIGHT cells (claimed and not done), not over heartbeat files: a finished
    # run's heartbeat freezes at its last step and would otherwise raise a permanent stale
    # alert, one per completed cell, until the noise buries a real stall.
    now=$(date +%s)
    for cdir in "$B"/claims/*; do
        [ -e "$cdir" ] || continue
        c=$(basename "$cdir")
        [ -e "$B/done/$c" ] && continue
        [ -e "$B/failed/$c" ] && continue
        pfx=$(grep '^PREFIX=' "$B/tasks/$c.task" 2>/dev/null | cut -d= -f2)
        [ -n "$pfx" ] || continue
        hb=$(ls "$SHARED_DIR"/heartbeats/"$pfx"_*.json 2>/dev/null | head -1)
        if [ -z "$hb" ]; then
            cage=$(( now - $(stat -c %Y "$B/claims/$c" 2>/dev/null || echo "$now") ))
            [ "$cage" -gt $((STALE_MIN*60)) ] && \
                log "ALERT CLAIMED ${cage}s AGO, STILL NO HEARTBEAT: $c"
        else
            hage=$(( now - $(stat -c %Y "$hb") ))
            if [ "$hage" -gt $((STALE_MIN*60)) ]; then
                log "ALERT STALE HEARTBEAT ${hage}s (in-flight): $c"
                if [ "$hage" -gt $((STALE_MIN*180)) ]; then
                    n=$(cat "$B/retries/$c" 2>/dev/null || echo 0)
                    if [ "$n" -lt "$MAX_RETRY" ]; then
                        echo $((n+1)) > "$B/retries/$c"
                        rm -rf "$B/claims/$c" "$SHARED_DIR"/heartbeats/"$pfx"_*.json
                        log "ACTION RELEASED dead claim ($((n+1))/$MAX_RETRY): $c"
                    fi
                fi
            fi
        fi
    done

    # prune heartbeats of completed cells so monitors show only live runs
    for dmark in "$B"/done/*; do
        [ -e "$dmark" ] || continue
        c=$(basename "$dmark")
        pfx=$(grep '^PREFIX=' "$B/tasks/$c.task" 2>/dev/null | cut -d= -f2)
        [ -n "$pfx" ] && rm -f "$SHARED_DIR"/heartbeats/"$pfx"_*.json
    done

    # ---------- loss sanity ----------
    "$VENV/bin/python" "$WORK_DIR/cluster/queue/monitor.py" --loss-alerts 2>/dev/null

    inflight=$(( nclaim - ndone - nfail ))
    log "ok: done=$ndone/$ntask in_flight=$inflight failed=$nfail remaining=$remaining"
    sleep "$INTERVAL"
done
