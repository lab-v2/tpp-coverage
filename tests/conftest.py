"""Shared paths and helpers for the test suite."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
ANALYSIS = SRC / "tpp_analysis"
A1 = REPO / "camera_ready" / "grid_classification"
A2 = REPO / "camera_ready" / "public_checkpoints"
# Generated or environment directories that no tree scan reads.
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "work"}

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


@pytest.fixture(scope="session")
def pc_work(tmp_path_factory) -> tuple[Path, dict]:
    """A private reanalysis work directory holding the HuggingFace run tables, and its env."""
    path = tmp_path_factory.mktemp("pc_work")
    env = {**os.environ, "TPP_PC_WORK": str(path)}
    run(["python", A2 / "get_runs.py"], cwd=A2, env=env)
    return path, env


def assert_same_rows(fresh: pd.DataFrame, committed: pd.DataFrame, keys: list[str]) -> None:
    """Equal row sets after sorting by `keys`: the committed columns, to 1e-9 relative."""
    fresh = fresh.sort_values(keys).reset_index(drop=True)
    committed = committed.sort_values(keys).reset_index(drop=True)
    assert len(fresh) == len(committed) > 0
    pd.testing.assert_frame_equal(fresh[committed.columns], committed, check_exact=False,
                                  rtol=1e-9, atol=0)


def tree_files(skip_names: set[str]) -> list[Path]:
    """Every file of the working tree outside SKIP_DIRS, except those named in `skip_names`.

    A walk rather than `git ls-files`, so the suite also runs on an exported tree.
    """
    return [path for path in REPO.rglob("*")
            if path.is_file() and not (set(path.relative_to(REPO).parts) & SKIP_DIRS)
            and path.name not in skip_names]


def tree_text(path: Path) -> str | None:
    """The file's text, or None for a compressed or binary file."""
    data = path.read_bytes()
    if path.suffix == ".gz" or b"\0" in data[:4096]:
        return None
    return data.decode("utf-8", errors="replace")
