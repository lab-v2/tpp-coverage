"""Characterization of the trainer: tiny seeded runs of `dist_trainer_main.py` on a GPU.

The trainer hard-codes CUDA (`torch.device("cuda:<rank>")`) and the nccl backend, so these tests
are marked `gpu` and skipped where torch sees no GPU (tests/conftest.py). Each run trains a
one-layer Llama (d_model 16) for two epochs on the first 64 rows of the real pre-tokenized
Wikipedia corpus (`$DATA_DIR/wikipedia/{train,val}_{tokens,lengths}.bin`, the layout the
downloaders write; skipped when absent), with the code's own deterministic flags.

What is compared with the base commit's values (`frozen/`, see `tests/characterize.py`): every
checkpoint's `model_state_dict` and optimizer tensors, its scalar fields and scheduler states, the
loss, learning-rate and token arrays of the metrics files, the metadata JSON, and the output of
`verify_reproducibility.py` on the run's files. Files are found by glob and read; their bytes and
names are never compared (both carry timestamps). Timing fields and the host name are left out.
The frozen values hold for the GPU and torch build they were taken on (GH200, torch 2.5.1+cu124).
"""

from __future__ import annotations

import json
import os
import re
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from characterize import check_frozen
from conftest import SRC, run

pytestmark = pytest.mark.gpu
FROZEN = Path(__file__).parent / "frozen"
TRAINING = SRC / "training_models" / "training"
ROWS = 64  # corpus rows copied per split
SEQ = 256  # the corpus' row width
# Run-time measurements and the machine's name: never part of the behaviour.
VOLATILE = {"timestamp", "time", "cumulative_time", "cumulative_time_seconds", "sec_per_step"}
VOLATILE |= {"eta_h", "host", "backup_timestamp", "batch_times", "epoch_times"}

BASE_ARGS = [
    "--dataset=wikipedia",
    "--shape=16,1,2,64",
    "--tokens=5000",
    "--epochs=2",
    "--batch-size=8",
    f"--max-seq-len={SEQ}",
    "--learning-rate=1e-3",
]
RUNS = {
    # The protocol path: AdamW, warmup then cosine, validation every epoch.
    "cosine": ["--val-tokens=8192", "--seed=42"],
    # Every other branch of the loop: Lion, warmup-stable-decay, two micro-batches per step,
    # padding scored, BF16 weights, no validation, one checkpoint kept, a custom prefix.
    "variants": [
        "--optimizer=lion",
        "--schedule=wsd",
        "--grad-accum-steps=2",
        "--score-padding",
        "--use-bf16",
        "--skip-val",
        "--keep-checkpoints=1",
        "--output-prefix=VAR",
        "--seed=7",
    ],
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _train(cwd: Path, data_dir: Path, args: list[str]) -> str:
    """One single-rank run of the trainer in `cwd`; its stdout."""
    cwd.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "DATA_DIR": str(data_dir),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(_free_port()),
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
    }
    cmd = ["python", TRAINING / "dist_trainer_main.py", *BASE_ARGS, *args]
    cmd.append(f"--shared-dir={cwd / 'shared'}")
    return run(cmd, cwd=cwd, env=env, timeout=900).stdout


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Path:
    """A DATA_DIR holding the first ROWS rows of each split of the real Wikipedia corpus."""
    src = Path(os.environ.get("DATA_DIR", "/nonexistent")) / "wikipedia"
    if not (src / "train_tokens.bin").exists():
        pytest.skip("needs the pre-tokenized Wikipedia corpus under $DATA_DIR/wikipedia")
    root = tmp_path_factory.mktemp("data")
    (root / "wikipedia").mkdir()
    for split in ("train", "val"):
        lengths = np.memmap(src / f"{split}_lengths.bin", dtype=np.uint16, mode="r")[:ROWS]
        tokens = np.memmap(src / f"{split}_tokens.bin", dtype=np.uint32, mode="r")[: ROWS * SEQ]
        lengths.tofile(root / "wikipedia" / f"{split}_lengths.bin")
        tokens.tofile(root / "wikipedia" / f"{split}_tokens.bin")
    return root


@pytest.fixture(scope="module")
def runs(corpus, tmp_path_factory) -> dict[str, Path]:
    """The two independent runs side by side, then a resume from the cosine run's epoch 1."""
    root = tmp_path_factory.mktemp("runs")
    # Two processes at once on one GPU: each is a few seconds of a 3M-parameter model.
    with ThreadPoolExecutor(len(RUNS)) as pool:
        futures = [pool.submit(_train, root / n, corpus, a) for n, a in RUNS.items()]
        for f in futures:
            f.result()
    epoch1 = next(p for p in _checkpoints(root / "cosine") if _load(p)["epoch"] == 1)
    _train(root / "resume", corpus, [*RUNS["cosine"], f"--resume={epoch1}"])
    return {name: root / name for name in [*RUNS, "resume"]}


def _load(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _checkpoints(cwd: Path, where: str = "saved_checkpoints") -> list[Path]:
    """The run's checkpoints, ordered by the epoch stored inside them."""
    found = (cwd / where).glob("*.pt")
    return sorted(found, key=lambda p: _load(p)["epoch"])


def _tensor(t) -> dict:
    return {"shape": list(t.shape), "dtype": str(t.dtype), "values": t.float().numpy()}


def _unstamp(value):
    """Strings with their time stamps (`20261126_021914`, a resumed run's prefix) replaced."""
    if isinstance(value, str):
        return re.sub(r"\d{8}_\d{6}", "<stamp>", value)
    if isinstance(value, dict):
        return {k: _unstamp(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unstamp(v) for v in value]
    return value


def _drop_volatile(d: dict) -> dict:
    return {k: _unstamp(v) for k, v in d.items() if k not in VOLATILE}


def _checkpoint_view(path: Path) -> dict:
    ckpt = _load(path)
    opt = ckpt["optimizer_state_dict"]
    view = {k: v for k, v in _drop_volatile(ckpt).items() if not k.endswith("state_dict")}
    view["weights"] = {k: _tensor(v) for k, v in ckpt["model_state_dict"].items()}
    view["optimizer"] = {
        "param_groups": opt["param_groups"],
        "state": {
            str(i): {k: _tensor(v) for k, v in s.items()} for i, s in sorted(opt["state"].items())
        },
    }
    for k in ("warmup_scheduler_state_dict", "cosine_scheduler_state_dict"):
        view[k] = ckpt[k]
    return view


def _npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return _drop_volatile({k: data[k].tolist() for k in sorted(data.files)})


def _json(path: Path) -> dict:
    return _drop_volatile(json.loads(path.read_text()))


def _one(cwd: Path, pattern: str) -> Path:
    found = sorted(cwd.glob(pattern))
    assert len(found) == 1, f"{pattern}: {found}"
    return found[0]


def _names(paths: list[Path]) -> list[str]:
    """File names with their time stamps and wall-clock hours replaced by fixed tokens."""
    out = []
    for p in paths:
        out.append(re.sub(r"_time[\d.]+h", "_time<h>", _unstamp(p.name)))
    return sorted(out)


def test_corpus_slice_is_the_frozen_one(corpus):
    """The slice the runs train on: a different corpus fails here, not in the tensors."""
    view = {}
    for split in ("train", "val"):
        lengths = np.fromfile(corpus / "wikipedia" / f"{split}_lengths.bin", dtype=np.uint16)
        tokens = np.fromfile(corpus / "wikipedia" / f"{split}_tokens.bin", dtype=np.uint32)
        view[split] = {"lengths": lengths.astype(float), "tokens": tokens.astype(float)}
    check_frozen(FROZEN / "corpus_slice.json", view)


@pytest.mark.parametrize("name", [*RUNS, "resume"])
def test_checkpoints(runs, name):
    cwd = runs[name]
    local = _checkpoints(cwd)
    shared = _checkpoints(cwd, "shared/saved_checkpoints")
    # The shared copy holds the same tensors as the local one.
    assert len(local) == len(shared) > 0
    for a, b in zip(local, shared):
        sa, sb = _load(a)["model_state_dict"], _load(b)["model_state_dict"]
        assert sa.keys() == sb.keys()
        assert all(sa[k].equal(sb[k]) for k in sa)
    view = {"names": _names(local), "checkpoints": [_checkpoint_view(p) for p in local]}
    check_frozen(FROZEN / f"checkpoints_{name}.json", view)


@pytest.mark.parametrize("name", [*RUNS, "resume"])
def test_metrics(runs, name):
    cwd = runs[name]
    local = _one(cwd, "training_metrics/*_metrics.npz")
    shared = _one(cwd, "shared/training_metrics/*_metrics.npz")
    assert _npz(local) == _npz(shared)
    names = [p for p in (cwd / "training_metrics").iterdir()]
    names += [p for p in (cwd / "shared").rglob("*") if p.is_file()]
    view = {
        "names": _names(names),
        "metrics": _npz(local),
        "intermediate": _npz(_one(cwd, "training_metrics/*_intermediate.npz")),
        "metadata": _json(_one(cwd, "training_metrics/*_metadata.json")),
        "shared_metadata": _json(_one(cwd, "shared/training_metrics/*_metadata.json")),
        "initial": _json(_one(cwd, "training_metrics/*_initial.json")),
        # Written when the global step is a multiple of 10; the resumed run starts at step 4.
        "heartbeats": [_json(p) for p in sorted(cwd.glob("shared/heartbeats/*.json"))],
    }
    check_frozen(FROZEN / f"metrics_{name}.json", view)


def _verify(*args) -> list[str]:
    """`verify_reproducibility.py` output without file names, timings and their lines."""
    out = run(["python", TRAINING / "verify_reproducibility.py", *map(str, args)]).stdout
    skip = ("Loading:", "cumulative_time", "epoch_times", "batch_times")
    return [ln for ln in out.splitlines() if not any(s in ln for s in skip)]


def test_verify_reproducibility_on_the_run(runs):
    cwd = runs["cosine"]
    first, last = _checkpoints(cwd)
    shared_last = _checkpoints(cwd, "shared/saved_checkpoints")[-1]
    metrics = _one(cwd, "training_metrics/*_metrics.npz")
    shared_metrics = _one(cwd, "shared/training_metrics/*_metrics.npz")
    view = {
        "weights_same": _verify("weights", "--original", last, "--reproduced", shared_last),
        "weights_epochs": _verify("weights", "--original", first, "--reproduced", last),
        "checkpoint_epochs": _verify("compare", "--original", first, "--reproduced", last),
        "metrics_same": _verify("compare", "--original", metrics, "--reproduced", shared_metrics),
        "metrics_runs": _verify(
            "compare",
            "--original",
            metrics,
            "--reproduced",
            _one(runs["resume"], "training_metrics/*_metrics.npz"),
        ),
    }
    assert "VERDICT: PASS" in "\n".join(view["weights_same"])
    check_frozen(FROZEN / "verify_reproducibility.json", view)
