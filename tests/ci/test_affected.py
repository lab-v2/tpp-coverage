"""The test selector a pull request runs (`tests/affected.py`): what it reads from a trace, what it
chooses, and that it falls back to the whole suite rather than to nothing.

Each case builds a small git repository with a map of its own, so the verdicts do not depend on
this repository's history; the traced case runs a real test file under strace.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import affected


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "data").mkdir()
    (root / "src" / "a.py").write_text("A = 1\n")
    (root / "src" / "b.py").write_text("B = 1\n")
    (root / "data" / "table.csv").write_text("x\n1\n")
    (root / "README.md").write_text("readme\n")
    (root / "tests" / "test_a.py").write_text("def test_a():\n    pass\n")
    (root / "tests" / "test_b.py").write_text("def test_b():\n    pass\n")
    (root / "tests" / "conftest.py").write_text("")
    git(root, "init", "-q", "-b", "main")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    monkeypatch.setattr(affected, "REPO", root)
    return root


def unit(
    files: list[str], dirs: list[str] | None = None, probes: list[str] | None = None
) -> dict[str, object]:
    return {"exit": 0, "seconds": 0.0, "files": files, "probes": probes or [], "dirs": dirs or []}


def write_map(repo: Path, units: dict[str, dict[str, object]], **over: object) -> Path:
    data = {
        "schema": affected.SCHEMA,
        "commit": git(repo, "rev-parse", "HEAD"),
        "complete": True,
        "paper_files": [],
    }
    data.update(over)
    data["units"] = units
    path = repo.parent / "map.json"
    path.write_text(json.dumps(data))
    return path


def standard(repo: Path, **over: object) -> Path:
    return write_map(
        repo,
        {
            "tests/test_a.py": unit(["src/a.py", "tests/test_a.py"]),
            "tests/test_b.py": unit(["src/b.py", "tests/test_b.py", "README.md"], ["data"]),
            affected.PAPER: unit(["src/b.py", "data/table.csv"]),
        },
        **over,
    )


def commit(repo: Path, edits: dict[str, str | None]) -> None:
    for rel, text in edits.items():
        path = repo / rel
        if text is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "change")


def test_a_change_to_one_source_runs_only_its_readers(repo: Path) -> None:
    m = standard(repo)
    commit(repo, {"src/a.py": "A = 2\n"})
    mode, tests, paper, _ = affected.choose(m)
    assert (mode, tests, paper) == ("some", ["tests/test_a.py"], False)


def test_a_change_to_the_papers_data_runs_the_paper_gate(repo: Path) -> None:
    m = standard(repo)
    commit(repo, {"data/table.csv": "x\n2\n"})
    mode, tests, paper, _ = affected.choose(m)
    assert (mode, tests, paper) == ("some", [], True)


def test_a_doc_a_test_reads_selects_that_test(repo: Path) -> None:
    m = standard(repo)
    commit(repo, {"README.md": "changed\n"})
    assert affected.choose(m)[:3] == ("some", ["tests/test_b.py"], False)


def test_a_new_file_in_a_listed_directory_selects_the_lister(repo: Path) -> None:
    m = standard(repo)
    commit(repo, {"data/new.csv": "y\n"})
    mode, tests, paper, _ = affected.choose(m)
    assert (mode, tests) == ("some", ["tests/test_b.py"])
    assert paper is False  # the paper read table.csv, it never listed data/


def test_a_changed_test_file_and_a_new_one_run(repo: Path) -> None:
    m = standard(repo)
    commit(repo, {"tests/test_a.py": "def test_a():\n    assert 1\n", "tests/test_new.py": "x=1\n"})
    assert affected.choose(m)[:2] == ("some", ["tests/test_a.py", "tests/test_new.py"])


def test_reading_the_git_index_means_every_change(repo: Path) -> None:
    m = write_map(
        repo,
        {
            "tests/test_a.py": unit([affected.GIT]),
            "tests/test_b.py": unit([]),
            affected.PAPER: unit([]),
        },
    )
    commit(repo, {"src/b.py": "B = 3\n"})
    assert affected.choose(m)[:3] == ("some", ["tests/test_a.py"], False)


def test_a_directory_a_test_probed_for_and_the_diff_creates_selects_it(repo: Path) -> None:
    """The test stat'ed docs/drafts (absent then); the diff adds a file inside it."""
    m = write_map(
        repo,
        {
            "tests/test_a.py": unit([], probes=["docs/drafts"]),
            "tests/test_b.py": unit([]),
            affected.PAPER: unit([]),
        },
    )
    commit(repo, {"docs/drafts/x.md": "draft\n"})
    assert affected.choose(m)[:2] == ("some", ["tests/test_a.py"])


def test_a_test_newly_marked_paper_runs_the_paper_gate(repo: Path) -> None:
    m = standard(repo)
    commit(
        repo,
        {"tests/test_b.py": "import pytest\n@pytest.mark.paper\ndef test_b():\n    pass\n"},
    )
    mode, tests, paper, _ = affected.choose(m)
    assert (mode, tests, paper) == ("some", ["tests/test_b.py"], True)


def test_a_map_newer_than_the_base_runs_everything(repo: Path) -> None:
    base = git(repo, "rev-parse", "HEAD")
    commit(repo, {"src/b.py": "B = 9\n"})
    m = standard(repo)  # built on a commit the pull request's base does not contain
    assert affected.choose(m, base)[0] == "full"


def test_a_file_no_test_reads_runs_nothing(repo: Path) -> None:
    m = standard(repo)
    commit(repo, {"notes.txt": "unread\n"})
    assert affected.choose(m)[:3] == ("none", [], False)


def test_no_change_since_the_map_runs_nothing(repo: Path) -> None:
    assert affected.choose(standard(repo))[0] == "none"


@pytest.mark.parametrize(
    "path",
    [
        "tests/conftest.py",
        "pyproject.toml",
        "uv.lock",
        ".pytest.toml",
        ".github/workflows/ci.yml",
        "tests/affected.py",
    ],
)
def test_a_shared_input_runs_everything(repo: Path, path: str) -> None:
    m = standard(repo)
    commit(repo, {path: "changed\n"})
    assert affected.choose(m)[:3] == ("full", [], True)


def test_a_missing_or_broken_map_runs_everything(repo: Path) -> None:
    assert affected.choose(repo.parent / "absent.json")[0] == "full"
    broken = repo.parent / "broken.json"
    broken.write_text("{not json")
    assert affected.choose(broken)[0] == "full"
    assert affected.choose(standard(repo, schema=affected.SCHEMA + 1))[0] == "full"


def test_an_incomplete_map_runs_everything(repo: Path) -> None:
    m = standard(repo, complete=False)
    commit(repo, {"src/a.py": "A = 2\n"})
    assert affected.choose(m)[0] == "full"


def test_a_map_from_a_commit_off_this_history_runs_everything(repo: Path) -> None:
    m = standard(repo, commit="0" * 40)
    assert affected.choose(m)[0] == "full"


def test_changes_on_main_after_the_map_are_selected_too(repo: Path) -> None:
    m = standard(repo)
    commit(repo, {"src/b.py": "B = 2\n"})  # landed on main after the map was built
    commit(repo, {"src/a.py": "A = 2\n"})  # the pull request
    assert affected.choose(m)[:3] == ("some", ["tests/test_a.py", "tests/test_b.py"], True)


def test_the_parser_reads_split_calls_relative_paths_and_listings(repo: Path) -> None:
    r = str(repo)
    log = repo.parent / "t.strace"
    log.write_text(
        "\n".join(
            [
                f'11 openat(AT_FDCWD<{r}>, "src/a.py", O_RDONLY|O_CLOEXEC) = 3<{r}/src/a.py>',
                f'11 openat(AT_FDCWD<{r}>, "data", O_RDONLY|O_DIRECTORY <unfinished ...>',
                f'12 newfstatat(AT_FDCWD<{r}>, "/etc/passwd", {{st_mode=0}}, 0) = 0',
                f"11 <... openat resumed>) = 4<{r}/data>",
                f'12 execve("{r}/.venv/bin/python", ["python"], 0x0 /* 1 var */) = 0',
                f'12 openat(AT_FDCWD<{r}>, ".git/HEAD", O_RDONLY) = 3<{r}/.git/HEAD>',
                f'13 newfstatat(AT_FDCWD<{r}/src>, "absent.py", 0x1, 0) = -1 ENOENT',
                # No fd beside the path: joined to the process's directory, which a fork inherits
                # and a chdir moves.
                '11 chdir("data") = 0',
                "11 clone3({flags=CLONE_VM, exit_signal=SIGCHLD}, 88) = 14",
                '14 access("table.csv", R_OK) = 0',
                '14 execve("../src/b.py", ["b.py"], 0x0 /* 1 var */) = 0',
                f'15 newfstatat(AT_FDCWD<{r}>, "caf\\303\\251.md", 0x1, 0) = 0',
                "13 +++ exited with 0 +++",
            ]
        )
    )
    files, probes, dirs = affected.parse_trace(log)
    assert files == {"src/a.py", affected.GIT, "src/b.py"}  # opened, or executed
    assert probes >= {"src/absent.py", "data/table.csv", "caf\u00e9.md", "src/a.py"}
    assert affected.GIT not in probes and not any(p.startswith(".venv") for p in probes)
    assert dirs == {"data"}


def test_a_url_or_command_string_never_enters_the_map(repo: Path) -> None:
    """A test cloning from `file:///<repo>` names the host's own path; the map is public."""
    r = str(repo)
    log = repo.parent / "t.strace"
    log.write_text(
        f'11 execve("/usr/bin/git", ["git", "clone", "file://{r}/.git"], 0x0 /* 1 var */) = 0\n'
        f'11 newfstatat(AT_FDCWD<{r}>, "file:{r}/x", 0x1, 0) = -1 ENOENT\n'
        f"11 newfstatat(AT_FDCWD<{r}>, \"git-upload-pack '{r}/t'\", 0x1, 0) = -1 ENOENT\n"
        f'11 openat(AT_FDCWD<{r}>, "src/a.py", O_RDONLY) = 3<{r}/src/a.py>\n'
    )
    files, probes, dirs = affected.parse_trace(log)
    assert files == {"src/a.py"}
    assert not [p for p in probes | files | dirs if r.lstrip("/") in p or ":" in p], probes


def test_pytest_probing_the_git_directory_is_not_a_git_read(repo: Path) -> None:
    """pytest stats `.git` and `.git/pyvenv.cfg` looking for virtualenvs; git never ran."""
    r = str(repo)
    log = repo.parent / "t.strace"
    log.write_text(
        f'11 newfstatat(AT_FDCWD<{r}>, "{r}/.git/pyvenv.cfg", 0x1, 0) = -1 ENOENT\n'
        f'11 newfstatat(AT_FDCWD<{r}>, "{r}/.git", {{st_mode=S_IFDIR}}, 0) = 0\n'
        f'11 openat(AT_FDCWD<{r}>, ".git", O_RDONLY|O_DIRECTORY) = 3<{r}/.git>\n'
        f'11 openat(AT_FDCWD<{r}>, "src/a.py", O_RDONLY) = 3<{r}/src/a.py>\n'
    )
    files, probes, _ = affected.parse_trace(log)
    assert files == {"src/a.py"} and affected.GIT not in probes


def test_a_vfork_child_logged_before_its_parent_returns_inherits_its_directory(
    repo: Path,
) -> None:
    log = repo.parent / "t.strace"
    log.write_text(
        '11 chdir("data") = 0\n'
        "11 vfork( <unfinished ...>\n"
        '20 access("table.csv", R_OK) = 0\n'
        "11 <... vfork resumed>) = 20\n"
    )
    assert "data/table.csv" in affected.parse_trace(log)[1]


def test_the_artifact_download_drops_the_token_when_it_leaves_the_api_host() -> None:
    import urllib.request

    handler = affected._NoAuthElsewhere()
    req = urllib.request.Request("https://api.github.com/x", headers={"Authorization": "Bearer t"})
    away = handler.redirect_request(req, None, 302, "Found", {}, "https://blob.example.net/y")
    home = handler.redirect_request(req, None, 302, "Found", {}, "https://api.github.com/z")
    assert not away.has_header("Authorization")
    assert home.has_header("Authorization")


def test_a_stat_of_a_sibling_is_not_a_read_and_bytecode_is_its_source(repo: Path) -> None:
    """pytest stats every entry of a test's folder; only an open makes a file an input."""
    r = str(repo)
    log = repo.parent / "t.strace"
    log.write_text(
        f'11 newfstatat(AT_FDCWD<{r}>, "{r}/README.md", {{st_mode=S_IFREG}}, 0) = 0\n'
        f'11 openat(AT_FDCWD<{r}>, "src/__pycache__/a.cpython-310.pyc", O_RDONLY) = '
        f"3<{r}/src/__pycache__/a.cpython-310.pyc>\n"
    )
    files, probes, _ = affected.parse_trace(log)
    assert files == {"src/a.py"} and "README.md" in probes


def test_a_stat_only_file_changing_in_place_selects_nothing_but_removing_it_does(
    repo: Path,
) -> None:
    m = write_map(
        repo,
        {
            "tests/test_a.py": unit([], probes=["README.md"]),
            "tests/test_b.py": unit([]),
            affected.PAPER: unit([]),
        },
    )
    commit(repo, {"README.md": "edited\n"})
    assert affected.choose(m)[0] == "none"
    commit(repo, {"README.md": None})
    assert affected.choose(m)[:2] == ("some", ["tests/test_a.py"])


def test_a_path_through_a_symlink_to_the_repository_counts(repo: Path) -> None:
    link = repo.parent / "link"
    link.symlink_to(repo)
    log = repo.parent / "t.strace"
    log.write_text(f'11 newfstatat(AT_FDCWD<{link}>, "src/new.py", 0x1, 0) = -1 ENOENT\n')
    assert affected.parse_trace(log)[1] == {"src/new.py"}


def traced() -> bool:
    """This process is under a tracer already (affected.py build), and ptrace does not nest."""
    with open("/proc/self/status") as fh:
        return any(ln.split()[1:] != ["0"] for ln in fh if ln.startswith("TracerPid:"))


@pytest.mark.skipif(shutil.which("strace") is None, reason="needs strace")
@pytest.mark.skipif(traced(), reason="already traced: strace cannot trace inside strace")
def test_a_traced_unit_sees_what_its_subprocesses_read(repo: Path, tmp_path: Path) -> None:
    """The data file is read only by a grandchild process, which strace -f must follow."""
    (repo / "tests" / "test_reads.py").write_text(
        "import subprocess, sys\n"
        "def test_reads():\n"
        "    code = \"import subprocess,sys; subprocess.run([sys.executable, '-c', "
        '\'open(\\"data/table.csv\\").read()\'], check=True)"\n'
        "    subprocess.run([sys.executable, '-c', code], check=True)\n"
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    rec = affected.run_unit("tests/test_reads.py", scratch, [])
    assert rec["exit"] == 0
    assert "data/table.csv" in rec["files"] and "src/b.py" not in rec["files"]


def test_the_two_parts_join_only_when_whole_and_from_one_commit() -> None:
    suite = {
        "schema": affected.SCHEMA,
        "commit": "a",
        "complete": True,
        "part": "suite",
        "paper_files": [],
        "units": {"tests/test_a.py": unit([])},
    }
    paper = {
        "schema": affected.SCHEMA,
        "commit": "a",
        "complete": True,
        "part": affected.PAPER,
        "paper_files": ["tests/test_p.py"],
        "units": {affected.PAPER: unit([])},
    }
    joined = affected.join(suite, paper)
    assert joined["complete"] is True and set(joined["units"]) == {"tests/test_a.py", "paper"}
    assert joined["paper_files"] == ["tests/test_p.py"]
    assert affected.join(suite, {**paper, "commit": "b"})["complete"] is False
    assert affected.join(suite, {**paper, "complete": False})["complete"] is False
    assert affected.join(suite, suite)["complete"] is False  # two suite parts are not a map
