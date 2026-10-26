"""Shared paths and helpers for the test suite."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
ANALYSIS = SRC / "tpp_analysis"
A1 = REPO / "camera_ready" / "grid_classification"
A2 = REPO / "camera_ready" / "public_checkpoints"

for p in (ANALYSIS, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def run(args: list[str], cwd: Path = REPO, env: dict | None = None,
        timeout: int = 1800) -> subprocess.CompletedProcess:
    """Run a command with this environment's interpreter; fail the test on a non-zero exit."""
    cmd = [sys.executable if a == "python" else str(a) for a in args]
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        pytest.fail(f"{' '.join(map(str, cmd))} exited {proc.returncode}\n"
                    f"--- stdout ---\n{proc.stdout[-4000:]}\n--- stderr ---\n{proc.stderr[-4000:]}")
    return proc
