"""tests/measure_coverage.py refuses a dirty tree, and pyproject asks coverage for what it needs."""

from __future__ import annotations

import shutil
import subprocess
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:  # the project pins 3.10, where pytest brings tomli
    import tomli as tomllib

from conftest import REPO


def test_toml_reader_is_the_one_this_python_has() -> None:
    """A version check, not an ImportError fallback: a missing tomli fails loudly on 3.10."""
    assert tomllib.__name__ == ("tomllib" if sys.version_info >= (3, 11) else "tomli")


def test_coverage_settings_follow_pools_subprocesses_and_branches():
    cfg = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["coverage"]["run"]
    assert cfg["branch"] and cfg["parallel"]
    assert "multiprocessing" in cfg["concurrency"] and "subprocess" in cfg["patch"]
    assert {"src", "camera_ready", "cluster", "large_scale"} <= set(cfg["source"])


def test_a_dirty_tree_is_refused(tmp_path):
    clone = tmp_path / "repo"
    subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{REPO}", str(clone)], check=True)
    shutil.copy(REPO / "tests" / "measure_coverage.py", clone / "tests" / "measure_coverage.py")
    (clone / "README.md").write_text("dirty\n")
    p = subprocess.run(
        [sys.executable, str(clone / "tests" / "measure_coverage.py"), str(tmp_path / "out")],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 2 and "uncommitted" in p.stdout, p.stdout + p.stderr
    assert not (tmp_path / "out").exists()
