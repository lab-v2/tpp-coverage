"""Line and branch coverage of the suite plus the paper gate, stamped with the commit measured.

    uv run --locked python tests/measure_coverage.py <out-dir>

Writes `<out-dir>/coverage-<short sha>-with-gate.json` (coverage's JSON report plus a top-level
`commit`, the full sha), from the `[tool.coverage.run]` settings in pyproject.toml. The suite runs
first, in parallel; the paper gate then runs alone (`-n 0`, it takes every thread), appended.
Refuses to run on a dirty tree, since the numbers would belong to no commit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def main() -> int:
    out = Path(sys.argv[1]).resolve()
    if git("status", "--porcelain", "--untracked-files=no"):
        print("measure_coverage: the tree has uncommitted changes; commit or stash them first")
        return 2
    full, short = git("rev-parse", "HEAD"), git("rev-parse", "--short", "HEAD")
    out.mkdir(parents=True, exist_ok=True)
    data = out / f".coverage-{short}"
    report = out / f"coverage-{short}-with-gate.json"
    env_args = ["--cov", f"--cov-report=json:{report}", "-q", "--no-header", "-p", "no:cacheprovider"]
    common = [sys.executable, "-m", "pytest", *env_args]
    env = {**os.environ, "COVERAGE_FILE": str(data)}
    for step in (["-n", "auto"], ["-n", "0", "-m", "paper", "--cov-append"]):
        if subprocess.run([*common, *step], cwd=REPO, env=env).returncode != 0:
            print(f"measure_coverage: pytest {' '.join(step)} failed")
            return 1
    rep = json.loads(report.read_text())
    rep["commit"] = full
    report.write_text(json.dumps(rep))
    print(f"measure_coverage: {report} ({rep['totals']['percent_covered_display']}% of lines and branches)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
