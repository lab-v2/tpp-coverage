"""Which test files a change can affect: a traced map built on main, and the selector a PR runs.

    uv run --locked python tests/affected.py build <map.json>
    uv run --locked python tests/affected.py fetch <map.json>
    uv run --locked python tests/affected.py select <map.json> [--base <sha>] [--github-output <f>]

`build` is the full suite: every test file runs in its own pytest process, all at once, and then
the paper gate (`-m paper`, its own files only) runs alone. Each process tree runs under
`strace -f`, which records every path it opens, stats, executes or lists, in the test itself, its
pool workers and every subprocess (scripts, git). A test file's inputs are the paths inside the
repository that its tree touched; a read inside the git directory means "every tracked file". The
map is written with the commit it was built on, and is marked incomplete when a unit failed (a
failed test may stop before it reads everything it depends on) or when a unit's trace does not
show it reading its own file and `tests/conftest.py` (a trace that lost lines).

`fetch` downloads the map of the newest successful push run of this workflow on main at an
ancestor of the pull request's base: two artifacts, the suite's part (CI's `tests` job builds it
with `--part suite`) and the paper gate's (the `paper` job, `--part paper`), joined into one.
Artifacts, because a pull request cannot write them, unlike a cache entry in its own scope.

`select` diffs the map's commit against HEAD (so a map a few commits old is still exact) and
chooses:

* `full`: the map is missing, unreadable, incomplete or from another schema, its commit is not
  an ancestor of the pull request's base, or a shared input changed (a `conftest.py`, the
  project or pytest configuration, the lock file, a workflow, or this file). The selector never
  fails open to running nothing.
* `some`: the test files whose recorded inputs the diff changes, or whose recorded paths it
  creates or removes (a directory a test probed for); the test files the diff changes; every
  file that reads the git directory; every file that listed a directory whose entries the diff
  adds or removes. `paper=true` when the paper gate's inputs are touched or a changed test file
  holds a paper test.
* `none`: no test reads anything the diff touches.
"""

from __future__ import annotations

import argparse
import functools
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[1]
SCHEMA = 3
PAPER = "paper"
#: The artifacts main's push run keeps: the suite's part (the `tests` job) and the paper's.
ARTIFACTS = {"suite": "affected-map", PAPER: "affected-map-paper"}
#: A change to any of these can change what every test does, so it runs the whole suite.
SHARED = (
    re.compile(r"(^|/)conftest\.py$"),
    re.compile(
        r"^(pyproject\.toml|uv\.lock|\.python-version|setup\.cfg|tox\.ini"
        r"|pytest\.ini|\.pytest\.ini|pytest\.toml|\.pytest\.toml)$"
    ),
    re.compile(r"^\.github/workflows/"),
    re.compile(r"^tests/affected\.py$"),
)
#: Paths no change in a pull request can reach (environments, caches).
IGNORED_PARTS = {".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
#: A read anywhere in the git directory (the index, objects, refs): it depends on every file.
GIT = "<git>"
TEST_FILE = re.compile(r"(^|/)(test_[^/]*|[^/]*_test)\.py$")

# A traced line: `openat(AT_FDCWD</cwd>, "rel", ...) = 3</abs>`. strace -y decodes every fd
# argument (and AT_FDCWD) and every returned fd to the path behind it.
_FD_ARG = re.compile(r'(?:AT_FDCWD|\d+)<((?:[^>\\]|\\.)*)>, "((?:[^"\\]|\\.)*)"')
_CWD = re.compile(r"AT_FDCWD<((?:[^>\\]|\\.)*)>")
_STR_ARG = re.compile(r'[(,] ?"((?:[^"\\]|\\.)*)"')
_RET_FD = re.compile(r"= \d+<((?:[^>\\]|\\.)*)>$")
_DIR_OPEN = re.compile(r"\bopenat2?\(.*O_DIRECTORY.*= \d+<((?:[^>\\]|\\.)*)>$")
_FORK = re.compile(r"^(?:clone3?|vfork|fork)\(.*\) = (\d+)")
_CHDIR = re.compile(r'^chdir\("((?:[^"\\]|\\.)*)"\) = 0')
_EXEC_OK = re.compile(r"^execve(at)?\(.*\) = 0$")
_PYC = re.compile(r"^(?P<dir>(?:.*/)?)__pycache__/(?P<name>[^/.]+)\.[^/]*\.pyc$")
_ESC = re.compile(rb"\\(x[0-9a-fA-F]{2}|[0-7]{1,3}|.)")
_SIMPLE = {b"n": b"\n", b"t": b"\t", b"r": b"\r", b"v": b"\v", b"f": b"\f"}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _unescape(s: str) -> str:
    """strace's C escapes (octal or hex bytes, \\n, \\") back to the UTF-8 path."""

    def one(m: re.Match[bytes]) -> bytes:
        e = m.group(1)
        if e[:1] == b"x":
            return bytes([int(e[1:], 16)])
        if e[:1].isdigit():
            return bytes([int(e, 8)])
        return _SIMPLE.get(e, e)

    return _ESC.sub(one, s.encode("utf-8", "surrogateescape")).decode("utf-8", "surrogateescape")


def _roots() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(the repository root's spellings, the git directories), all resolved."""
    return _roots_of(str(REPO))


@functools.lru_cache(maxsize=None)
def _roots_of(repo: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    root = {repo, os.path.realpath(repo)}
    gits = {
        os.path.realpath(os.path.join(repo, git("rev-parse", "--git-dir"))),
        os.path.realpath(os.path.join(repo, git("rev-parse", "--git-common-dir"))),
    }
    return tuple(sorted(root)), tuple(sorted(gits))


@functools.lru_cache(maxsize=None)
def _resolved(path: str) -> str:
    """The path with its directory's symlinks resolved; the last part is kept as named."""
    p = os.path.normpath(path)
    return os.path.join(os.path.realpath(os.path.dirname(p)), os.path.basename(p))


def _inside(path: str) -> str | None:
    """`path` relative to the repository, GIT inside the git directory, None elsewhere."""
    roots, gits = _roots()
    for p in {os.path.normpath(path), _resolved(path)}:
        if any(p == g or p.startswith(g + os.sep) for g in gits):
            return GIT
        for root in roots:
            if p == root:
                return "."
            if p.startswith(root + os.sep):
                rel = os.path.relpath(p, root)
                pyc = _PYC.match(rel)
                if pyc:  # an import that read the cached bytecode depends on the source
                    rel = f"{pyc['dir']}{pyc['name']}.py"
                parts = PurePosixPath(rel).parts
                if parts[0] == ".git":
                    return GIT
                return None if IGNORED_PARTS & set(parts) else rel
    return None


def _joined(lines: list[str]) -> list[tuple[str, str, str | None]]:
    """(pid, call, forking parent) per call: strace -f splits an interrupted call in two.

    A child's first calls can be logged before its parent's fork returns (vfork), so each call
    also names the newest process with a fork still unfinished, the child's likely parent.
    """
    pending: dict[str, str] = {}
    forking: list[str] = []
    out: list[tuple[str, str, str | None]] = []
    for line in lines:
        pid, _, rest = line.partition(" ")
        rest = rest.lstrip()
        hint = forking[-1] if forking else None
        if rest.endswith("<unfinished ...>"):
            pending[pid] = rest[: -len("<unfinished ...>")]
            if re.match(r"(clone3?|vfork|fork)\(", rest):
                forking.append(pid)
        elif rest.startswith("<... ") and "resumed>" in rest:
            if pid in forking:
                forking.remove(pid)
            out.append((pid, pending.pop(pid, "") + rest.split("resumed>", 1)[1], hint))
        else:
            out.append((pid, rest, hint))
    return out + [(pid, call, None) for pid, call in pending.items()]


def parse_trace(trace: Path, start_cwd: str | None = None) -> tuple[set[str], set[str], set[str]]:
    """(files read, paths probed, directories listed) inside the repository, from one log.

    Read: a file a process opened (its returned fd) or executed; a cached `.pyc` counts as its
    source. Probed: every path any call named, found or not (stat, access, a failed open): its
    contents do not matter, only whether it exists. A relative path with no fd beside it is
    joined to the calling process's working directory: the last AT_FDCWD it showed, a chdir, or
    its parent's at fork; the first process starts in `start_cwd` (the repository).
    """
    files: set[str] = set()
    probes: set[str] = set()
    dirs: set[str] = set()
    cwd: dict[str, str] = {}
    first = True
    for pid, line, parent in _joined(trace.read_text(errors="surrogateescape").splitlines()):
        if first:
            cwd[pid] = start_cwd or str(REPO)
            first = False
        elif pid not in cwd and parent in cwd:
            cwd[pid] = cwd[parent]
        if line.startswith(("+++", "---")):
            continue
        here = _CWD.search(line)
        if here:
            cwd[pid] = _unescape(here.group(1))
        fork = _FORK.search(line)
        if fork:
            cwd[fork.group(1)] = cwd.get(pid, str(REPO))
            continue
        named = [os.path.join(_unescape(d), _unescape(rel)) for d, rel in _FD_ARG.findall(line)]
        if not named:
            base = cwd.get(pid, str(REPO))
            named = [os.path.join(base, _unescape(x)) for x in _STR_ARG.findall(line)]
        ch = _CHDIR.search(line)
        if ch:
            cwd[pid] = os.path.normpath(
                os.path.join(cwd.get(pid, str(REPO)), _unescape(ch.group(1)))
            )
        probes.update(r for f in named if (r := _inside(f)) not in (None, ".", GIT))
        listed = _DIR_OPEN.search(line)
        ret = _RET_FD.search(line)
        if listed:
            if (rel := _inside(_unescape(listed.group(1)))) not in (None, GIT):
                dirs.add(rel)
        elif ret:
            # The git directory counts only when a file in it was opened (git reading its index
            # or objects); pytest stats `.git` and `.git/pyvenv.cfg` looking for virtualenvs.
            if (rel := _inside(_unescape(ret.group(1)))) not in (None, "."):
                files.add(rel)
        elif _EXEC_OK.search(line) and named:
            if (rel := _inside(named[0])) not in (None, ".", GIT):
                files.add(rel)
    return files, probes, dirs


def test_files() -> list[str]:
    tests = REPO / "tests"
    return sorted(str(p.relative_to(REPO)) for p in tests.rglob("*.py") if TEST_FILE.search(p.name))


def paper_test_files() -> list[str]:
    """The test files that hold a `paper` test, from pytest's own collection at the work tree."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-n", "0", "-m", PAPER, "--collect-only", "-q"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, 5):
        raise RuntimeError(f"collecting the paper tests failed:\n{proc.stdout[-3000:]}")
    return sorted({ln.split("::", 1)[0] for ln in proc.stdout.splitlines() if "::" in ln})


def run_unit(unit: str, scratch: Path, paper_files: list[str]) -> dict[str, object]:
    """Run one unit under strace; its exit code, wall time and traced inputs."""
    out = scratch / (re.sub(r"[^\w.-]", "_", unit) + ".strace")
    # The paper unit runs only the files that hold paper tests: collecting the whole suite would
    # import every test module, and their imports are not the paper's inputs.
    target = ["-n", "0", "-m", PAPER, *paper_files] if unit == PAPER else ["-n", "0", unit]
    cmd = [
        shutil.which("strace") or "strace",
        "-f",  # one log, not -ff: a file per process ran out of descriptors under `uv sync`
        "--seccomp-bpf",  # stop only at the traced calls, not at every system call
        "-qq",
        "-y",
        "-s",
        "4096",
        "-e",
        "trace=%file,%process",
        "-o",
        str(out),
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
        *target,
    ]
    start = time.monotonic()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    wall = time.monotonic() - start
    files, probes, dirs = parse_trace(out)
    out.unlink()
    # strace's own errors (a log it could not write) mean the trace lost lines.
    traced_ok = not re.search(r"^(\S*/)?strace: ", proc.stderr, re.M)
    expected = set(paper_files) if unit == PAPER else {unit, "tests/conftest.py"}
    sane = traced_ok and expected <= files
    code = proc.returncode
    if code not in (0, 5) or not sane:
        print(
            f"--- {unit} exited {code}"
            f"{'' if sane else ' (infrastructure: the trace is incomplete, not a test failure)'}\n"
            f"{proc.stdout[-6000:]}\n{proc.stderr[-3000:]}"
        )
    return {
        "exit": code if sane else -1,
        "seconds": round(wall, 1),
        "files": sorted(files),
        "probes": sorted(probes - files),
        "dirs": sorted(dirs),
    }


def build(path: Path, part: str = "all") -> int:
    """The full suite, one traced process per test file, then the paper gate alone.

    `part` is `suite` (the test files only: CI's `tests` job), `paper` (the paper gate only: CI's
    `paper` job, which runs beside it) or `all`; `fetch` joins the two parts again.
    """
    if shutil.which("strace") is None:
        print("affected: strace is not installed; the map cannot be built")
        return 2
    commit = git("rev-parse", "HEAD")
    units = [] if part == PAPER else test_files()
    paper_files = paper_test_files()
    width = max(1, min(len(units), len(os.sched_getaffinity(0))))
    with tempfile.TemporaryDirectory(prefix="affected-") as tmp:
        scratch = Path(tmp)
        # Thread pool: each unit is its own process tree; the threads only wait on them.
        with ThreadPoolExecutor(width) as pool:
            done = dict(zip(units, pool.map(lambda u: run_unit(u, scratch, paper_files), units)))
        if part != "suite":
            done[PAPER] = run_unit(PAPER, scratch, paper_files)  # alone: it takes every thread
    failed = [u for u, r in done.items() if r["exit"] not in (0, 5)]
    data = {
        "schema": SCHEMA,
        "commit": commit,
        "complete": not failed,
        "paper_files": paper_files,
        "part": part,
        "units": done,
    }
    path.write_text(json.dumps(data, indent=1))
    slow = sorted(done.items(), key=lambda kv: -float(kv[1]["seconds"]))[:3]
    print(
        f"affected: {len(done)} units traced at {commit[:12]}; slowest "
        + ", ".join(f"{u} {r['seconds']} s" for u, r in slow)
    )
    if failed:
        print(f"affected: {len(failed)} unit(s) failed: {', '.join(failed)}")
        return 1
    return 0


def join(suite: dict[str, object], paper: dict[str, object]) -> dict[str, object]:
    """One map from the two parts; incomplete unless both are whole and from one commit."""
    units = {**suite["units"], **paper["units"]}  # type: ignore[dict-item]
    whole = (
        suite.get("part") == "suite"
        and paper.get("part") == PAPER
        and suite["commit"] == paper["commit"]
        and suite["schema"] == paper["schema"]
        and suite["complete"] is True
        and paper["complete"] is True
    )
    return {
        **suite,
        "complete": whole,
        "part": "all",
        "paper_files": paper["paper_files"],
        "units": units,
    }


class _NoAuthElsewhere(urllib.request.HTTPRedirectHandler):
    """Follow a redirect without the token when it leaves the API's host (artifact storage
    answers 401 to a GitHub token, and must not receive it)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if (
            new is not None
            and urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc
        ):
            new.headers.pop("Authorization", None)
            new.unredirected_hdrs.pop("Authorization", None)
        return new


def fetch(path: Path, base: str = "HEAD") -> int:
    """The map of main's newest successful push run at an ancestor of `base`; none is fine."""
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    repo = os.environ["GITHUB_REPOSITORY"]
    headers = {
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json",
    }
    opener = urllib.request.build_opener(_NoAuthElsewhere)

    def get(url: str) -> bytes:
        with opener.open(urllib.request.Request(url, headers=headers), timeout=60) as r:
            return r.read()

    try:
        runs = json.loads(
            get(
                f"{api}/repos/{repo}/actions/workflows/ci.yml/runs"
                "?branch=main&event=push&status=success&per_page=30"
            )
        )["workflow_runs"]
        for run in runs:  # newest first
            if not _ancestor(run["head_sha"], base):
                continue
            arts = json.loads(get(f"{api}/repos/{repo}/actions/runs/{run['id']}/artifacts"))
            live = {a["name"]: a for a in arts["artifacts"] if not a["expired"]}
            if not all(name in live for name in ARTIFACTS.values()):
                continue
            parts = {}
            for part, name in ARTIFACTS.items():
                with zipfile.ZipFile(io.BytesIO(get(live[name]["archive_download_url"]))) as z:
                    (member,) = z.namelist()
                    parts[part] = json.loads(z.read(member))
            path.write_text(json.dumps(join(parts["suite"], parts[PAPER]), indent=1))
            print(f"affected: map of run {run['id']} ({run['head_sha'][:12]})")
            return 0
    except (OSError, KeyError, ValueError) as exc:  # no map: select falls back to everything
        print(f"affected: no map fetched ({type(exc).__name__}: {exc})"[:300])
        return 0
    print("affected: no successful push run on main with a map at an ancestor of the base")
    return 0


def _tree(commit: str) -> set[str]:
    """Every file and directory path in `commit`."""
    paths = {p for p in git("ls-tree", "-r", "-z", "--name-only", commit).split("\0") if p}
    return paths | {str(p) for f in paths for p in PurePosixPath(f).parents if str(p) != "."}


def _ancestor(commit: str, of: str) -> bool:
    proc = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", commit, of])
    return proc.returncode == 0


def choose(map_path: Path, base: str = "HEAD") -> tuple[str, list[str], bool, list[str]]:
    """(mode, test files, paper, reasons) for HEAD against the map."""
    try:
        data = json.loads(map_path.read_text())
        commit, units, paper_files = data["commit"], data["units"], data["paper_files"]
        assert data["schema"] == SCHEMA and data["complete"] is True and PAPER in units
    except (OSError, ValueError, KeyError, TypeError, AssertionError):
        return "full", [], True, [f"no usable map at {map_path.name}"]
    if not (_ancestor(commit, base) and _ancestor(commit, "HEAD")):
        return "full", [], True, [f"map commit {commit[:12]} is not an ancestor of {base}"]
    diff = git("diff", "--name-only", "-z", "--no-renames", commit, "HEAD")
    changed = [c for c in diff.split("\0") if c]
    if not changed:
        return "none", [], False, [f"nothing changed since {commit[:12]}"]
    shared = [c for c in changed if any(rx.search(c) for rx in SHARED)]
    if shared:
        return "full", [], True, [f"shared input changed: {', '.join(shared)}"]
    before, after = _tree(commit), _tree("HEAD")
    appeared = before ^ after  # files and directories the diff creates or removes
    relisted = {str(PurePosixPath(p).parent) for p in appeared}
    touched = set(changed) | appeared
    picked: dict[str, str] = {}
    for unit, rec in units.items():
        files, dirs = set(rec["files"]), set(rec["dirs"])
        probes = set(rec["probes"]) | files | dirs
        why = (
            (unit in touched and "its own file changed")
            or (GIT in files and "it reads the git directory")
            or (hit := sorted(files & set(changed)))
            and f"reads {hit[0]}"
            or (gone := sorted(probes & appeared))
            and f"probed for {gone[0]}, which the change creates or removes"
            or (lst := sorted(dirs & relisted))
            and f"lists {lst[0]}/"
        )
        if why:
            picked[unit] = why
    for c in changed:  # a test file the map has not seen yet
        if TEST_FILE.search(c) and c.startswith("tests/") and c not in units and c in after:
            picked[c] = "new test file"
    changed_tests = {c for c in changed if TEST_FILE.search(c) and c in after}
    if changed_tests and PAPER not in picked:
        try:
            now = set(paper_test_files())  # a test the diff marked `paper`, or moved
        except RuntimeError as exc:
            return "full", [], True, [f"the head's tests do not collect: {str(exc)[:2000]}"]
        if now != set(paper_files) or now & changed_tests:
            picked[PAPER] = "a changed test file holds paper tests"
    paper = PAPER in picked
    tests = sorted(u for u in picked if u != PAPER and u in after)
    reasons = [f"{u}: {w}" for u, w in sorted(picked.items())]
    return "some" if picked else "none", tests, paper, reasons or ["no test reads a changed path"]


def select(map_path: Path, base: str, github_output: Path | None) -> int:
    mode, tests, paper, reasons = choose(map_path, base)
    print(f"affected: mode={mode} tests={len(tests)} paper={str(paper).lower()}")
    for r in reasons:
        print(f"  {r}")
    if github_output is not None:
        with github_output.open("a") as fh:
            fh.write(f"mode={mode}\ntests={' '.join(tests)}\npaper={str(paper).lower()}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="run the full suite traced and write the map")
    b.add_argument("map", type=Path)
    b.add_argument("--part", choices=["all", "suite", PAPER], default="all")
    f = sub.add_parser("fetch", help="download the map of main's newest successful push run")
    f.add_argument("map", type=Path)
    f.add_argument("--base", default="HEAD", help="the pull request's base commit")
    s = sub.add_parser("select", help="choose the tests HEAD can affect")
    s.add_argument("map", type=Path)
    s.add_argument("--base", default="HEAD", help="the pull request's base commit")
    s.add_argument("--github-output", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        return build(args.map, args.part)
    if args.cmd == "fetch":
        return fetch(args.map, args.base)
    return select(args.map, args.base, args.github_output)


if __name__ == "__main__":
    sys.exit(main())
