"""Every entry point imports and answers --help; the job generator emits the trained design."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from conftest import A1, A2, REPO, run

ENTRY_ROOTS = [REPO / "src", A1, A2]


def _entry_points() -> list[Path]:
    found = []
    for root in ENTRY_ROOTS:
        for path in sorted(root.rglob("*.py")):
            if re.search(r'^if __name__ == "__main__":', path.read_text(), re.M):
                found.append(path)
    return found


ENTRY_POINTS = _entry_points()


def test_entry_points_are_discovered():
    # A guard against the discovery silently finding nothing.
    names = {p.name for p in ENTRY_POINTS}
    assert {"run_analysis.py", "dist_trainer_main.py", "download_and_extract.py",
            "generate_jobs.py", "classify.py", "estimate.py"} <= names
    assert len(ENTRY_POINTS) >= 25


@pytest.mark.parametrize("script", ENTRY_POINTS, ids=lambda p: str(p.relative_to(REPO)))
def test_help(script: Path):
    out = run(["python", script, "--help"], cwd=script.parent, timeout=300)
    assert "usage:" in out.stdout.lower()


@pytest.mark.parametrize("script", sorted(REPO.joinpath("src").rglob("*.sh")),
                         ids=lambda p: p.name)
def test_shell_scripts_parse(script: Path):
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_one_hf_dataset_id():
    from hf_dataset import REPO_ID

    verify = (REPO / "src/training_models/training/verify_reproducibility.py").read_text()
    assert f'HF_REPO_ID = "{REPO_ID}"' in verify


# ── the job generator ──────────────────────────────────────────────────────

GEN = REPO / "src/tpp_analysis/generate_jobs/generate_jobs.py"


def _listed(*args: str) -> list[str]:
    return run(["python", GEN, "--list", *args]).stdout.splitlines()


@pytest.mark.parametrize("design,counts", [
    ("main", {"nxd": 120, "colinear": 120, "holdout": 42, "colinear_holdout": 42}),
    ("bigtpp", {"nxd": 60, "colinear": 60, "holdout": 18, "colinear_holdout": 18}),
])
def test_generator_run_counts(design, counts):
    kinds = [line.split()[0] for line in _listed("--dataset", "wikipedia", "--design", design)]
    assert {k: kinds.count(k) for k in set(kinds)} == counts


def test_generator_scripts(tmp_path):
    env_free = {"PATH": "/usr/bin:/bin"}
    run(["python", GEN, "--dataset", "c4", "--out-dir", tmp_path, "--runs-per-job", "7"],
        env=env_free)
    scripts = sorted((tmp_path / "c4_main_fp32").glob("c4_main_fp32_*.sh"))
    assert len(scripts) == 47  # ceil(324 / 7)
    text = "\n".join(s.read_text() for s in scripts)
    assert text.count("torchrun") == 324
    assert text.count("--use-train-as-val") == 84  # every holdout run trains on the val split
    assert set(re.findall(r"--learning-rate (\S+)", text)) == {"0.001"}  # C4 was trained at 1e-3
    assert "src/training_models/training/dist_trainer_main.py" in text
    # No cluster-specific defaults: partition/account/exclude only when asked for.
    assert "#SBATCH --partition" not in text and "#SBATCH --exclude" not in text


def test_generator_bf16_flag(tmp_path):
    run(["python", GEN, "--dataset", "wikipedia", "--precision", "bf16", "--out-dir", tmp_path])
    text = "\n".join(p.read_text() for p in (tmp_path / "wikipedia_main_bf16").glob("*_0*.sh"))
    assert text.count("--use-bf16") == text.count("torchrun") > 0
    assert "--output-prefix BF16WIKI_" in text


def test_trainer_accepts_generated_flags():
    flags = run(["python", REPO / "src/training_models/training/dist_trainer_main.py",
                 "--help"]).stdout
    for flag in ("--use-bf16", "--use-train-as-val", "--skip-val", "--output-prefix",
                 "--model-size", "--tokens", "--learning-rate", "--weight-decay"):
        assert flag in flags
