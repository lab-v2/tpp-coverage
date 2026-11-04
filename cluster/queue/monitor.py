#!/usr/bin/env python3
"""Status of a task-queue sweep: queue counts, per-run heartbeats, loss sanity.

    python cluster/queue/monitor.py               # one snapshot
    python cluster/queue/monitor.py --watch 60    # refresh every 60 s
    python cluster/queue/monitor.py --health      # ALERT lines for anything needing action
    python cluster/queue/monitor.py --loss-alerts # only the loss-sanity ALERT lines

The queue directory is ``$QUEUE_DIR`` and the trainer's shared copy ``$SHARED_DIR`` (default
``$QUEUE_DIR/shared``), as set by ``env.sh``; ``--queue-dir`` and ``--shared-dir`` override them.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

#: A heartbeat older than this belongs to a finished run, not a live slot.
LIVE_SECONDS = 3600
#: A live run whose heartbeat is older than this is reported as stalled.
STALLED_SECONDS = 1500
#: A loss still above this past ``LEARNING_PCT`` percent of the run never started learning
#: (ln of the cl100k vocabulary is about 11.5).
NOT_LEARNING_LOSS = 11.0
LEARNING_PCT = 15.0


@dataclass(frozen=True)
class QueueState:
    tasks: list[str]
    done: set[str]
    failed: set[str]
    claimed: set[str]
    est_node_hours: dict[str, float]

    @property
    def running(self) -> set[str]:
        return self.claimed - self.done - self.failed

    @property
    def pending(self) -> int:
        return len(self.tasks) - len(self.done) - len(self.running) - len(self.failed)


def task_fields(path: Path) -> dict[str, str]:
    """KEY=VALUE lines of a .task file, surrounding quotes removed."""
    out = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out[key] = value.strip('"')
    return out


def queue_state(queue_dir: Path) -> QueueState:
    tasks = sorted(p.stem for p in (queue_dir / "tasks").glob("*.task"))
    names = lambda sub: {p.name for p in (queue_dir / sub).glob("*")}  # noqa: E731
    est = {}
    for name in tasks:
        value = task_fields(queue_dir / "tasks" / f"{name}.task").get("EST_NH")
        if value:
            est[name] = float(value)
    return QueueState(tasks, names("done"), names("failed"), names("claims"), est)


def heartbeats(shared_dir: Path) -> list[dict]:
    """Every readable heartbeat, in file-name order.

    Sorted because `glob` returns the directory's own order, which differs between
    file systems (tmpfs lists newest first, ext4 by name hash).
    """
    out = []
    for path in sorted(glob.glob(str(shared_dir / "heartbeats" / "*.json"))):
        try:
            out.append(json.loads(Path(path).read_text()))
        except (OSError, ValueError):
            continue
    return out


def loss_alert(hb: dict) -> str | None:
    """The loss-sanity alert for one heartbeat, or None."""
    loss, pct = hb.get("loss", float("nan")), hb.get("pct", 0.0)
    if math.isnan(loss) or math.isinf(loss):
        return "ALERT LOSS NaN/inf"
    if pct > LEARNING_PCT and loss > NOT_LEARNING_LOSS:
        return "ALERT LOSS NOT DESCENDING"
    return None


def done_prefixes(queue_dir: Path, state: QueueState) -> set[str]:
    out = set()
    for name in state.done:
        task = queue_dir / "tasks" / f"{name}.task"
        if task.is_file():
            prefix = task_fields(task).get("PREFIX")
            if prefix:
                out.add(prefix)
    return out


def health_lines(queue_dir: Path, shared_dir: Path, now: float | None = None) -> list[str]:
    """Per-live-run status lines, each prefixed with its ALERT tags."""
    now = time.time() if now is None else now
    state = queue_state(queue_dir)
    finished = done_prefixes(queue_dir, state)
    lines = [f"queue: done={len(state.done)}/{len(state.tasks)} claimed={len(state.claimed)} "
             f"failed={len(state.failed)}"]
    lines += [f"ALERT failed: {name}" for name in sorted(state.failed)]
    live = 0
    for hb in sorted(heartbeats(shared_dir), key=lambda h: h.get("prefix", "")):
        prefix = hb.get("prefix", "")
        age = int(now - hb.get("time", 0))
        if any(prefix.startswith(p) for p in finished) or age > LIVE_SECONDS:
            continue  # a finished run: its heartbeat is frozen, not stalled
        live += 1
        tags = ["ALERT STALLED"] if age > STALLED_SECONDS else []
        alert = loss_alert(hb)
        if alert:
            tags.append(alert)
        tag = " ".join(tags) + " " if tags else ""
        lines.append(f"{tag}{prefix[:52]}: {hb.get('pct', 0):.1f}% loss={hb.get('loss', 0):.3f} "
                     f"lr={hb.get('lr', 0):.1e} eta={hb.get('eta_h', 0):.2f}h hb_age={age}s")
    if live == 0 and state.running:
        lines.append(f"ALERT: no live heartbeats (<{LIVE_SECONDS // 60} min) - no slot is training")
    return lines


def loss_alert_lines(shared_dir: Path, now: float | None = None) -> list[str]:
    now = time.time() if now is None else now
    out = []
    for hb in heartbeats(shared_dir):
        if now - hb.get("time", 0) > LIVE_SECONDS:
            continue
        alert = loss_alert(hb)
        if alert:
            out.append(f"{alert} at {hb.get('pct', 0):.1f}%: {hb.get('prefix', '?')[:60]}")
    return out


def snapshot(queue_dir: Path, shared_dir: Path) -> None:
    state = queue_state(queue_dir)
    nh_all = sum(state.est_node_hours.values())
    nh_done = sum(state.est_node_hours.get(n, 0.0) for n in state.done)
    print(f"=== {queue_dir.name}  {time.strftime('%F %T')} ===")
    print(f"tasks: {len(state.tasks)}  done: {len(state.done)}  running: {len(state.running)}  "
          f"failed: {len(state.failed)}  pending: {state.pending}")
    print(f"node-h: {nh_done:.1f} / {nh_all:.1f} ({100 * nh_done / max(nh_all, 1e-9):.0f}%)")
    if state.failed:
        print("FAILED: " + ", ".join(sorted(state.failed)))
    live = [h for h in heartbeats(shared_dir) if time.time() - h.get("time", 0) < 300]
    if live:
        print(f"\n{'run':58s} {'node':>7} {'step':>13} {'pct':>6} "
              f"{'loss':>8} {'lr':>9} {'eta_h':>6} {'age_s':>6}")
        for h in sorted(live, key=lambda x: x.get("prefix", "")):
            print(f"{h.get('prefix', '?')[:58]:58s} {h.get('host', '?').split('.')[0]:>7} "
                  f"{h.get('step_in_epoch', 0):>6}/{h.get('total_steps', 0):<6} "
                  f"{h.get('pct', 0):>5.1f}% {h.get('loss', 0):>8.4f} "
                  f"{h.get('lr', 0):>9.2e} {h.get('eta_h', 0):>6.2f} "
                  f"{time.time() - h.get('time', 0):>6.0f}")
    else:
        print("\n(no live heartbeats in the last 5 min)")
    user = os.environ.get("USER")
    if user:
        try:
            q = subprocess.run(["squeue", "-u", user, "-o", "%.10i %.9j %.2t %.10M %R"],
                               capture_output=True, text=True, timeout=15)
            print("\n" + q.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--queue-dir", type=Path, default=os.environ.get("QUEUE_DIR"))
    p.add_argument("--shared-dir", type=Path, default=os.environ.get("SHARED_DIR"))
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--watch", type=int, metavar="SECONDS")
    mode.add_argument("--health", action="store_true")
    mode.add_argument("--loss-alerts", action="store_true")
    args = p.parse_args(argv)
    if args.queue_dir is None:
        p.error("set QUEUE_DIR (source cluster/queue/env.sh) or pass --queue-dir")
    shared = args.shared_dir or args.queue_dir / "shared"
    if args.loss_alerts:
        lines = loss_alert_lines(shared)
        if lines:
            print("\n".join(lines))
    elif args.health:
        print("\n".join(health_lines(args.queue_dir, shared)))
    elif args.watch:
        while True:
            print("\033[2J\033[H", end="")
            snapshot(args.queue_dir, shared)
            time.sleep(args.watch)
    else:
        snapshot(args.queue_dir, shared)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
