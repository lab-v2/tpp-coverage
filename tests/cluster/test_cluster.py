"""The cluster pipeline: queue status logic, the slot worker's task loop, the WIKI300M task
generator, and the audit tables against their recorded results."""

from __future__ import annotations

import importlib.util
import json
import os
import py_compile
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import REPO, run

CLUSTER = REPO / "cluster"
QUEUE = CLUSTER / "queue"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


monitor = _load("queue_monitor", QUEUE / "monitor.py")


def _queue(tmp_path: Path) -> Path:
    q = tmp_path / "queue"
    for sub in ("tasks", "done", "failed", "claims", "shared/heartbeats"):
        (q / sub).mkdir(parents=True)
    for i, (prefix, split, est) in enumerate([("RUN_A", "--skip-val", 2.0),
                                              ("RUN_B", '"--skip-val --seed 43"', 1.0),
                                              ("RUN_C", "--use-train-as-val", 0.5)]):
        (q / "tasks" / f"{i:03d}_{prefix}.task").write_text(
            f"PREFIX={prefix}\nMODEL_SIZE=1000\nTOKENS=2000\nSPLIT_FLAG={split}\nEST_NH={est}\n")
    return q


def _beat(q: Path, prefix: str, loss: float, pct: float, age: float) -> None:
    hb = {"prefix": prefix, "loss": loss, "pct": pct, "time": time.time() - age, "lr": 1e-4}
    (q / "shared" / "heartbeats" / f"{prefix}_x.json").write_text(json.dumps(hb))


def test_task_fields_strip_quotes(tmp_path):
    q = _queue(tmp_path)
    assert monitor.task_fields(q / "tasks" / "001_RUN_B.task")["SPLIT_FLAG"] == "--skip-val --seed 43"


def test_queue_state_counts(tmp_path):
    q = _queue(tmp_path)
    (q / "done" / "000_RUN_A").touch()
    (q / "claims" / "000_RUN_A").mkdir()
    (q / "claims" / "001_RUN_B").mkdir()
    (q / "failed" / "002_RUN_C").write_text("rc=1\n")
    (q / "claims" / "002_RUN_C").mkdir()
    s = monitor.queue_state(q)
    assert len(s.tasks) == 3 and s.running == {"001_RUN_B"} and s.pending == 0
    assert s.est_node_hours == {"000_RUN_A": 2.0, "001_RUN_B": 1.0, "002_RUN_C": 0.5}


@pytest.mark.parametrize("loss,pct,expected", [
    (float("nan"), 50.0, "ALERT LOSS NaN/inf"), (float("inf"), 1.0, "ALERT LOSS NaN/inf"),
    (11.4, 40.0, "ALERT LOSS NOT DESCENDING"), (11.4, 5.0, None), (4.2, 90.0, None)])
def test_loss_alert(loss, pct, expected):
    assert monitor.loss_alert({"loss": loss, "pct": pct}) == expected


def test_health_lines(tmp_path):
    q = _queue(tmp_path)
    (q / "done" / "000_RUN_A").touch()
    (q / "claims" / "001_RUN_B").mkdir()
    _beat(q, "RUN_A", float("nan"), 99.0, 10)    # finished: frozen heartbeat, never flagged
    _beat(q, "RUN_B", 11.6, 30.0, 1600)          # live, stalled and not learning
    _beat(q, "RUN_C", 3.0, 50.0, 7200)           # older than an hour: not a live slot
    lines = monitor.health_lines(q, q / "shared")
    assert lines[0] == "queue: done=1/3 claimed=1 failed=0"
    flagged = [line for line in lines if line.startswith("ALERT")]
    assert len(flagged) == 1 and flagged[0].startswith(
        "ALERT STALLED ALERT LOSS NOT DESCENDING RUN_B:")
    assert monitor.loss_alert_lines(q / "shared") == [
        "ALERT LOSS NaN/inf at 99.0%: RUN_A", "ALERT LOSS NOT DESCENDING at 30.0%: RUN_B"]


def test_monitor_cli_requires_a_queue(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("QUEUE_DIR", "SHARED_DIR")}
    proc = subprocess.run([sys.executable, QUEUE / "monitor.py"], env=env, capture_output=True,
                          text=True)
    assert proc.returncode == 2 and "QUEUE_DIR" in proc.stderr


STUB_PYTHON = """#!/bin/bash
# Stands in for the training interpreter: records its arguments, writes the metrics file the
# slot worker checks for, and fails for the task named in FAIL_PREFIX.
printf '%s\\n' "$*" >> "$STUB_LOG"
prefix=""; prev=""
for a in "$@"; do [ "$prev" = "--output-prefix" ] && prefix=$a; prev=$a; done
[ "$prefix" = "$FAIL_PREFIX" ] && exit 3
mkdir -p "$SHARED_DIR/training_metrics"
touch "$SHARED_DIR/training_metrics/${prefix}_m1_token2_metrics.npz"
"""


def test_slot_worker_runs_the_queue(tmp_path):
    q = _queue(tmp_path)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text(STUB_PYTHON)
    (venv / "bin" / "python").chmod(0o755)
    (tmp_path / "corpora").mkdir()
    log = tmp_path / "calls.log"
    env = dict(os.environ, QUEUE_DIR=str(q), VENV=str(venv), DATA_DIR=str(tmp_path / "corpora"),
               STUB_LOG=str(log), FAIL_PREFIX="RUN_C")
    env.pop("SHARED_DIR", None)
    proc = subprocess.run(["bash", QUEUE / "slot_worker.sh", "7"], env=env, capture_output=True,
                          text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert sorted(p.name for p in (q / "done").iterdir()) == ["000_RUN_A", "001_RUN_B"]
    assert [p.name for p in (q / "failed").iterdir()] == ["002_RUN_C"]
    calls = log.read_text().splitlines()
    assert len(calls) == 3
    # The quoted two-flag split of a seed cell reaches the trainer as two flags.
    assert "--skip-val --seed 43 --shared-dir" in calls[1]
    assert "--use-train-as-val" in calls[2] and "--learning-rate 3e-4" in calls[0]
    # A rerun skips done and failed cells: nothing is trained twice.
    subprocess.run(["bash", QUEUE / "slot_worker.sh", "7"], env=env, check=True,
                   capture_output=True, timeout=60)
    assert len(log.read_text().splitlines()) == 3


def test_slot_worker_refuses_a_missing_environment(tmp_path):
    env = dict(os.environ, QUEUE_DIR=str(tmp_path / "absent"), VENV=str(tmp_path / "novenv"),
               DATA_DIR=str(tmp_path / "nodata"))
    proc = subprocess.run(["bash", QUEUE / "slot_worker.sh"], env=env, capture_output=True,
                          text=True, timeout=60)
    assert proc.returncode == 1
    assert "DATA_DIR does not exist" in proc.stderr and "no task queue" in proc.stderr


@pytest.mark.parametrize("script", sorted(CLUSTER.rglob("*.sh")) + sorted(CLUSTER.rglob("*.sbatch")),
                         ids=lambda p: str(p.relative_to(CLUSTER)))
def test_shell_scripts_parse(script):
    subprocess.run(["bash", "-n", str(script)], check=True)
    if shutil.which("shellcheck"):
        subprocess.run(["shellcheck", "-S", "warning", "-s", "bash", str(script)], check=True)


@pytest.mark.parametrize("script", sorted((CLUSTER / "audit").rglob("*.py")) +
                         sorted((CLUSTER / "wiki300m").glob("*.py")),
                         ids=lambda p: str(p.relative_to(CLUSTER)))
def test_python_scripts_compile(script, tmp_path):
    py_compile.compile(str(script), cfile=str(tmp_path / "x.pyc"), doraise=True)


def test_generator_reproduces_the_first_100_committed_tasks(tmp_path):
    run(["python", CLUSTER / "wiki300m" / "generate_tasks.py", tmp_path], timeout=600)
    committed = CLUSTER / "wiki300m"
    assert (tmp_path / "manifest.tsv").read_text() == (committed / "manifest.tsv").read_text()
    generated = sorted((tmp_path / "tasks").glob("*.task"))
    assert len(generated) == 100
    for path in generated:
        assert path.read_text() == (committed / "tasks" / path.name).read_text(), path.name
    assert len(list((committed / "tasks").glob("*.task"))) == 468


def test_per_law_table_matches_the_recorded_strata_run():
    # The off-ray NC win rates bbox_strata.py printed when it wrote bbox_strata.csv:
    # Kaplan 24.2% / 94.3%, Chinchilla 97.8% / 92.5%, Droppo-Elibol 94.6% / 62.8% (low / high).
    out = run(["python", CLUSTER / "audit" / "tables" / "ci_per_law.py"], timeout=600).stdout
    rows = {line.split()[0]: line.split() for line in out.splitlines()
            if line.split() and line.split()[0] in ("Kaplan", "Chinchilla", "Droppo-Elibol")}
    assert rows["Kaplan"][1] == "24.2%" and rows["Kaplan"][4] == "94.3%"
    assert rows["Chinchilla"][1] == "97.8%" and rows["Chinchilla"][4] == "92.5%"
    assert rows["Droppo-Elibol"][1] == "94.6%" and rows["Droppo-Elibol"][4] == "62.8%"
