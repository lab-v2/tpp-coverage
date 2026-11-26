"""The paper refit is collected only when asked for by name, so no broader run starts it."""

from __future__ import annotations

import subprocess
import sys

import pytest

from conftest import REPO


@pytest.mark.parametrize(
    "args,expected",
    [
        ([], 0),
        (["-m", "slow"], 0),
        (["-m", "network"], 0),
        (["-m", "slow and network"], 0),
        (["-m", "paper"], 11),
    ],
)
def test_paper_tests_collected_only_by_name(args, expected):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "0",
            "--collect-only",
            "-q",
            *args,
            "tests/tpp_analysis/test_paper_numbers.py",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode in (0, 5), proc.stdout + proc.stderr  # 5: nothing collected
    out = proc.stdout
    assert sum("test_paper_numbers.py::" in line for line in out.splitlines()) == expected, out
