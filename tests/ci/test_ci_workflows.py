"""The CI workflows: own runners only, the required check names, and a masked public log.

Every job asks for the project's own runners with the `primary` preference, and its
first step, before checkout prints anything, masks the runner's home, account,
machine name and work-directory root in the log. The mask step is run here on a
made-up runner layout and its masks are applied to log lines in the forms
actions/checkout prints them, plain and regex-escaped.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess

from typing import Any

import pytest
import yaml

from conftest import REPO

WORKFLOWS = REPO / ".github" / "workflows"
OWN_POOL = ["self-hosted", "linux", "primary"]
#: The status-check contexts branch protection requires (job `name:`s).
REQUIRED_CHECKS = {"tests", "ruff", "privacy", "pr-authorization", "Own runners guard"}
#: Jobs that are not status checks: the Pages deploy of the project page (pages.yml).
DEPLOY_JOBS = {"deploy"}


def load(name: str) -> dict[str, Any]:
    return yaml.safe_load((WORKFLOWS / name).read_text())


def triggers(doc: dict[str, Any]) -> dict[str, Any]:
    return doc.get("on", doc.get(True))  # YAML 1.1 reads a bare `on` as True


ALL = {p.name: load(p.name) for p in sorted(WORKFLOWS.glob("*.yml"))}
JOBS = [(f, jid, job) for f, doc in ALL.items() for jid, job in doc["jobs"].items()]


def test_the_workflows_are_found() -> None:
    assert set(ALL) == {"ci.yml", "pr-authorization.yml", "own-runners-guard.yml", "pages.yml"}


@pytest.mark.parametrize(("wf", "jid", "job"), JOBS, ids=[f"{f}:{j}" for f, j, _ in JOBS])
def test_every_job_runs_on_the_own_pool_with_primary(wf: str, jid: str, job: dict[str, Any]) -> None:
    assert job["runs-on"] == OWN_POOL
    assert "container" not in job and "services" not in job


def test_required_check_names_exist() -> None:
    assert {job["name"] for _, _, job in JOBS} == REQUIRED_CHECKS | DEPLOY_JOBS


def test_ci_runs_on_pull_requests_and_pushes_to_main() -> None:
    on = triggers(ALL["ci.yml"])
    assert on["pull_request"]["types"] == ["opened", "synchronize", "reopened", "ready_for_review"]
    assert on["push"]["branches"] == ["main"]


def test_pr_authorization_reruns_on_reviews() -> None:
    on = triggers(ALL["pr-authorization.yml"])
    assert on["pull_request"]["types"] == ["opened", "synchronize", "reopened", "ready_for_review"]
    assert on["pull_request_review"]["types"] == ["submitted", "dismissed"]
    assert ALL["pr-authorization.yml"]["permissions"] == {"contents": "read", "pull-requests": "read"}


def test_the_guard_listens_to_every_other_workflow() -> None:
    names = {doc["name"] for f, doc in ALL.items() if f != "own-runners-guard.yml"}
    assert set(triggers(ALL["own-runners-guard.yml"])["workflow_run"]["workflows"]) == names


def test_ci_runs_the_full_suite_in_parallel_ruff_and_the_privacy_scan() -> None:
    runs = {jid: " ".join(s.get("run", "") for s in job["steps"]) for jid, job in ALL["ci.yml"]["jobs"].items()}
    assert re.search(r"uv run .*pytest -n auto", runs["tests"])
    assert "tests/" not in runs["tests"].split("pytest", 1)[1]  # the whole suite, no subset
    assert re.search(r"ruff check( --output-format=github)? \.$", runs["ruff"])
    assert privacy_targets() and all(t.endswith("test_privacy.py") for t in privacy_targets())


def privacy_command() -> str:
    (step,) = [s["run"] for s in ALL["ci.yml"]["jobs"]["privacy"]["steps"] if "pytest" in s.get("run", "")]
    return step


def privacy_targets() -> list[str]:
    """The test paths the privacy job hands pytest."""
    return [a for a in shlex.split(privacy_command().split("pytest", 1)[1]) if not a.startswith("-")]


def test_the_privacy_job_targets_exist() -> None:
    assert [t for t in privacy_targets() if not (REPO / t).is_file()] == []


def test_the_privacy_job_installs_the_project_dependencies() -> None:
    """A job that fails on a missing package installs it; it never strips the import instead.

    Owner rule: a package that makes a job fail is installed, never worked around. The privacy
    job once ran with only the dev group and failed on conftest's pandas import.
    """
    assert "--only-group" not in privacy_command()


def test_the_privacy_job_passes_in_a_fresh_environment(tmp_path) -> None:
    """The privacy job's own command, in a fresh environment, as the runner does it."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PYTEST_", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"))}
    env["UV_PROJECT_ENVIRONMENT"] = str(tmp_path / "venv")
    proc = subprocess.run(["bash", "-ec", privacy_command()], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]


MASK_STEPS = [job["steps"][0] for _, _, job in JOBS]


def test_every_job_masks_first_with_one_identical_step() -> None:
    assert all(s.get("name", "").startswith("Mask this runner") for s in MASK_STEPS)
    assert len({s["run"] for s in MASK_STEPS}) == 1
    for _, _, job in JOBS:
        firsts = [i for i, s in enumerate(job["steps"]) if "checkout" in s.get("uses", "")]
        assert firsts and firsts[0] > 0


def masks_for(home: str, workspace: str) -> list[str]:
    script = MASK_STEPS[0]["run"]
    env = {"HOME": home, "GITHUB_WORKSPACE": workspace, "PATH": "/usr/bin:/bin"}
    out = subprocess.run(["bash", "-e", "-c", script], env=env, capture_output=True, text=True, check=True)
    return [line.removeprefix("::add-mask::") for line in out.stdout.splitlines()]


def apply(masks: list[str], text: str) -> str:
    for m in sorted(set(masks), key=len, reverse=True):
        text = text.replace(m, "***")
    return text


def test_mask_step_hides_the_runner_layout_in_checkout_lines() -> None:
    home, root = "/users/alice", "/srv/alice/own-runners/jit/runner-jit-repo-1-2"
    ws = f"{root}/_work/repo/repo"
    masks = masks_for(home, ws)
    esc = root.replace("/", "\\/").replace("-", "\\-")
    log = "\n".join([
        f"Working directory is '{ws}'",
        f"[command]{home}/.local/bin/git version",
        f"Copying '{home}/.gitconfig' to '{root}/_work/_temp/x/.gitconfig'",
        f"git config --local --unset includeif.gitdir:{ws}/.git.path {esc}\\/_work\\/_temp\\/c\\.config",
        "elsewhere: /srv/alice/cache/uv",
    ])
    masked = apply(masks, log)
    assert "alice" not in masked and "/srv" not in masked and "own-runners" not in masked
    assert "own\\-runners" not in masked
    assert "Working directory is '***/_work/repo/repo'" in masked


def test_mask_step_masks_the_kernel_host_name_and_skips_short_values() -> None:
    masks = masks_for("/users/alice", "/srv/alice/r/_work/repo/repo")
    kernel = open("/proc/sys/kernel/hostname").read().strip()
    if len(kernel) >= 4:
        assert kernel in masks
    assert all(len(m) >= 4 for m in masks)  # never `/` or another short, common string


def test_a_mask_that_misses_is_caught() -> None:
    # Negative control: without the escaped forms the escaped line still leaks.
    root = "/srv/alice/own-runners/jit/r"
    plain_only = [m for m in masks_for("/users/alice", f"{root}/_work/x/x") if "\\" not in m]
    esc = root.replace("/", "\\/").replace("-", "\\-")
    assert "own\\-runners" in apply(plain_only, f"{esc}\\/_work")
