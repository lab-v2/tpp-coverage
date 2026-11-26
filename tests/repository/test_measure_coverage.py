"""tests/measure_coverage.py refuses a dirty tree, and pyproject asks coverage for what it needs."""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # the project pins 3.10, where pytest brings tomli
    import tomli as tomllib

from conftest import REPO


def _toml_import_block() -> str:
    """The source of this file's `if sys.version_info ...` TOML import, as it stands."""
    src = Path(__file__).read_text()
    node = next(
        n for n in ast.parse(src).body if isinstance(n, ast.If) and "tomllib" in ast.unparse(n)
    )
    return ast.get_source_segment(src, node)


def _run_import(tomli_missing: bool) -> subprocess.CompletedProcess:
    hide = "sys.modules['tomli'] = None\n" if tomli_missing else ""
    code = f"import sys\n{hide}{_toml_import_block()}\nprint(tomllib.__name__)"
    return subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)


def test_toml_reader_is_the_one_this_python_has() -> None:
    p = _run_import(tomli_missing=False)
    want = "tomllib" if sys.version_info >= (3, 11) else "tomli"
    assert p.returncode == 0 and p.stdout.strip() == want, p.stdout + p.stderr


def test_a_missing_tomli_fails_loudly_on_310() -> None:
    """A version check, not an ImportError fallback: without tomli, 3.10 cannot import at all."""
    p = _run_import(tomli_missing=True)
    if sys.version_info >= (3, 11):
        assert p.returncode == 0 and p.stdout.strip() == "tomllib", p.stdout + p.stderr
    else:
        assert p.returncode != 0 and "ModuleNotFoundError" in p.stderr, p.stdout + p.stderr


def test_coverage_settings_follow_pools_subprocesses_and_branches() -> None:
    cfg = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["coverage"]["run"]
    assert cfg["branch"] and cfg["parallel"]
    assert "multiprocessing" in cfg["concurrency"] and "subprocess" in cfg["patch"]
    assert {"src", "camera_ready", "cluster", "large_scale"} <= set(cfg["source"])


def test_a_dirty_tree_is_refused(tmp_path: Path) -> None:
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
