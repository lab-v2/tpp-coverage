"""Characterization of the trainer's CPU pieces: seeding, checkpoint and metrics files, resume.

Everything here runs without a GPU: the seeding helpers, the save, load, validate and clean-up
methods of `DistributedTrainer` on a seeded model in a temporary working directory, and the
resume helpers of `dist_trainer_main.py`. The training loop itself needs CUDA and is
characterised by `trainer_run/` on a GPU host. Files are found by glob, read, and compared without
their time stamps; expected values are the base commit's, frozen in `frozen/`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import argparse
import importlib
import json
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from characterize import check_frozen
from conftest import SRC

if TYPE_CHECKING:
    from collections.abc import Iterator

sys.path.insert(0, str(SRC / "training_models" / "training"))
T = importlib.import_module("training_models.training.DistributedTrainer")
M = importlib.import_module("dist_trainer_main")
from training_models.models import ModelConfig, create_model  # noqa: E402

FROZEN = Path(__file__).parent / "frozen"
VOLATILE = {"timestamp", "time", "backup_timestamp", "host"}
PREFIX = "c4_m1234_token100_lr0-001_distributed"


def stamped(s: str) -> str:
    """Mask wall-clock values: run-name timestamps and the trainer's "took 0.01s" timings."""
    return re.sub(r"took \d+\.\d+s", "took <secs>", re.sub(r"\d{8}_\d{6}", "<stamp>", s))


@pytest.fixture
def restore_flags() -> Iterator[None]:
    """set_seed turns on global determinism flags; give the worker its own back afterwards."""
    flags = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic)
    bench = torch.backends.cudnn.benchmark
    yield
    torch.use_deterministic_algorithms(flags[0])
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = flags[1], bench


def test_seeding(restore_flags: None) -> None:
    T.set_seed(11)
    view = {
        "draws": [random.random(), float(np.random.rand()), torch.rand(3).tolist()],
        "flags": [
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
        ],
        "generator": torch.rand(3, generator=T.create_deterministic_generator(5)).tolist(),
    }
    torch.manual_seed(9)
    T.seed_worker(0)
    view["worker"] = [random.random(), float(np.random.rand())]
    check_frozen(FROZEN / "seeding.json", view)


def tiny_model() -> tuple[ModelConfig, torch.nn.Module]:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=50, max_seq_len=8, n_layers=1, n_heads=2, d_model=8, d_ff=32)
    return cfg, create_model(cfg)


def saved_state(
    model: torch.nn.Module, cfg: ModelConfig
) -> tuple[
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    torch.optim.lr_scheduler.LRScheduler,
]:
    """An optimizer and both schedulers after one step of the model, as the loop holds them."""
    opt = T.make_optimizer("adamw", model.parameters(), 1e-3, 0.3)
    ws = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=2)
    cs = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5, eta_min=1e-4)
    ids = torch.arange(16).reshape(2, 8) % 50
    model(ids).float().pow(2).mean().backward()
    opt.step()
    ws.step()
    return opt, ws, cs


def read_tree(root: Path) -> dict[str, object]:
    """Every file under `root` by stamp-free relative name: JSON and NPZ contents, checkpoints'
    scalar fields and tensors."""
    out = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        key = stamped(str(p.relative_to(root)))
        if p.suffix == ".json":
            out[key] = {k: v for k, v in json.loads(p.read_text()).items() if k not in VOLATILE}
        elif p.suffix == ".npz":
            data = np.load(p, allow_pickle=True)
            out[key] = {k: data[k].tolist() for k in data.files if k not in VOLATILE}
        elif p.suffix == ".pt":
            ck = torch.load(p, map_location="cpu", weights_only=False)
            view = {k: v for k, v in ck.items() if k not in VOLATILE and "state_dict" not in k}
            view["weights"] = {k: v.numpy() for k, v in ck["model_state_dict"].items()}
            view["optimizer"] = ck["optimizer_state_dict"]["param_groups"]
            view["schedulers"] = [
                ck["warmup_scheduler_state_dict"],
                ck["cosine_scheduler_state_dict"],
            ]
            out[key] = view
    return {k: (stamped(v) if isinstance(v, str) else v) for k, v in out.items()}


def test_checkpoint_and_metrics_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg, model = tiny_model()
    opt, ws, cs = saved_state(model, cfg)
    trainer = T.DistributedTrainer()
    shared = str(tmp_path / "shared")
    common = dict(prefix=PREFIX, config=cfg, weight_decay=0.3, batch_size=4, seed=3)
    for epoch in range(3):  # three epochs, two kept
        args = dict(model=SimpleNamespace(module=model), optimizer=opt, ws=ws, cs=cs, epoch=epoch)
        args.update(t_loss=4.0 - epoch, v_loss=None if epoch else 4.5, tokens=100 * (epoch + 1))
        args.update(current_step=2 * (epoch + 1), cumulative_time=7.0 * (epoch + 1))
        trainer._save_checkpoint(**args, **common, keep_last_n=2)
        trainer._save_checkpoint_shared(**args, **common, shared_dir=shared, keep_last_n=2)
    metrics = {
        "batch_losses": [4.1, 3.9, 3.7],
        "learning_rates": [1e-3, 5e-4, 1e-4],
        "steps": [0, 1, 2],
        "tokens_per_batch": [30, 40, 50],
        "cumulative_tokens": [30, 70, 120],
        "epoch_train_losses": [3.9],
        "val_losses": [],
    }
    trainer._save_initial_metadata(PREFIX, cfg, 1e-3, 3, 4, 0.3, seed=3, shared_dir=shared)
    trainer._save_intermediate_metrics(PREFIX, metrics, global_step=3)
    trainer._save_metrics(PREFIX, metrics, cfg, 1e-3, 3, 120, 0.3, 21.0, 4, 3)
    trainer._save_metrics_shared(PREFIX, metrics, cfg, 1e-3, 3, 120, 0.3, 21.0, 4, shared, 3)
    trainer._write_heartbeat(shared, PREFIX, {"loss": 3.7, "time": 1.0})
    trainer._write_heartbeat("", PREFIX, {"loss": 0.0})  # no shared dir: nothing written
    out = capsys.readouterr().out.replace(str(tmp_path), "<tmp>")
    log = [stamped(line) for line in out.splitlines()]
    check_frozen(FROZEN / "saved_files.json", {"files": read_tree(tmp_path), "log": log})

    # What the resume path reads back from those files.
    last = max((tmp_path / "saved_checkpoints").glob("*.pt"), key=lambda p: p.stat().st_mtime)
    view = {
        "epoch_tokens": list(trainer.set_current_epoch_initial_tokens(str(last), rank=0)),
        "valid": trainer._validate_checkpoint(str(last)),
        # None today: the checkpoint names carry `_wd<decay>`, which the pattern does not allow,
        # so a resume never finds the run's metrics by itself (pass --resume-metrics).
        "metrics_found": M.find_metrics_for_checkpoint(str(last)),
    }
    check_frozen(FROZEN / "resume_reads.json", view)


def test_checkpoint_validation_and_failed_saves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    trainer = T.DistributedTrainer()
    empty = tmp_path / "empty.pt"
    empty.write_bytes(b"")
    junk = tmp_path / "junk.pt"
    junk.write_bytes(b"junk")
    cfg, model = tiny_model()
    # No optimizer: building the checkpoint fails and the save prints instead of raising.
    bad = dict(model=SimpleNamespace(module=model), optimizer=None, ws=None, cs=None, epoch=0)
    bad.update(t_loss=1.0, v_loss=None, tokens=1, prefix=PREFIX, config=cfg, weight_decay=0.0)
    trainer._save_checkpoint(**bad)
    trainer._save_checkpoint_shared(**bad, shared_dir=str(tmp_path / "shared"))
    trainer._save_metrics_shared(PREFIX, {}, None, 1e-3, 1, 0, 0.0, shared_dir=str(tmp_path))
    trainer._save_intermediate_metrics(PREFIX, {"steps": [1]}, 1)  # no batch_losses key
    trainer._save_initial_metadata(PREFIX, None, 1e-3, 1, 1, 0.0)
    view = {
        "empty": trainer._validate_checkpoint(str(empty)),
        "junk": trainer._validate_checkpoint(str(junk)),
        "missing": trainer._validate_checkpoint(str(tmp_path / "absent.pt")),
        "log": [
            ln.split(":")[0]
            for ln in capsys.readouterr().out.replace(str(tmp_path), "<tmp>").splitlines()
        ],
        "leftovers": sorted(stamped(str(p.relative_to(tmp_path))) for p in tmp_path.rglob("*")),
    }
    check_frozen(FROZEN / "failed_saves.json", view)


def test_find_and_load_metrics(tmp_path: Path) -> None:
    d = tmp_path / "training_metrics"
    d.mkdir()
    ckpt = f"{PREFIX}_epoch2_bs4_seq8_time0.01h_completedAt20260101_000000.pt"
    view = {"none": M.find_metrics_for_checkpoint(ckpt, str(d))}
    view["bad_name"] = M.find_metrics_for_checkpoint("model.pt", str(d))
    np.savez(d / f"{PREFIX}_bs4_seq8_time0.01h_20260101_000000_metrics.npz", batch_losses=[1.0])
    view["timestamped"] = Path(M.find_metrics_for_checkpoint(ckpt, str(d))).name
    np.savez(d / f"{PREFIX}_intermediate.npz", batch_losses=[2.0, 1.5], steps=[0, 1])
    found = M.find_metrics_for_checkpoint(ckpt, str(d))
    view["intermediate"] = Path(found).name
    view["loaded"] = M.load_metrics_from_npz(found)
    with pytest.raises(FileNotFoundError, match="Metrics file not found"):
        M.load_metrics_from_npz(str(d / "absent.npz"))
    check_frozen(FROZEN / "find_metrics.json", view)


def test_resume_distributed_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "training_metrics").mkdir()
    np.savez(tmp_path / "training_metrics" / f"{PREFIX}_intermediate.npz", steps=[0, 1, 2])
    ckpt = f"{PREFIX}_epoch1.pt"
    common = dict(model_size=1234, learning_rate=1e-3, batch_size=4)
    cases = {
        "fresh": argparse.Namespace(resume=None, resume_metrics=None, **common),
        "auto": argparse.Namespace(resume=ckpt, resume_metrics=None, **common),
        "given": argparse.Namespace(resume=ckpt, resume_metrics="absent.npz", **common),
        "rank1": argparse.Namespace(resume=ckpt, resume_metrics=None, **common),
    }
    view = {}
    for name, args in cases.items():
        path, metrics_path, prior = M.resume_distributed_training(args, rank=int(name == "rank1"))
        view[name] = [path, metrics_path, prior]
    check_frozen(FROZEN / "resume.json", view)


def test_trainer_entry_without_a_process_group(restore_flags: None) -> None:
    """The trainer's own checks before CUDA: names, accumulation, the process group."""
    trainer = T.DistributedTrainer(seed=5)
    assert trainer.load_dataset("unknown", 8, True) == (None, None)
    kwargs = dict(epochs=1, lr=1e-3, batch_size=4, tokens=10, dataset="c4", max_seq_len=8)
    kwargs.update(log_interval=1, architecture="llama", val_tokens_limit=8, shared_dir="s")
    kwargs.update(skip_val=True, keep_checkpoints=1, weight_decay=0.0)
    kwargs.update(resume_checkpoint_path=None, resume_metrics_path=None, prior_metrics=None)
    cfg, _ = tiny_model()
    with pytest.raises(AssertionError, match="not divisible by grad_accum_steps 3"):
        trainer.train_model(cfg, grad_accum_steps=3, **kwargs)
    with pytest.raises(RuntimeError, match="Distributed training not initialized"):
        trainer.train_model(cfg, grad_accum_steps=2, score_padding=True, **kwargs)
    view = {
        "mask_padding": trainer.mask_padding,
        "scheme": trainer.get_gpu_scheme(),
        "seed": trainer.seed,
        "required_gpus_without_cuda": trainer.get_required_gpus(),
        "defaults": [M.DEFAULT_BATCH_SIZE, M.DEFAULT_MAX_SEQ_LEN, M.DEFAULT_WEIGHT_DECAY],
        "warmup": [T.DEFAULT_WARMUP_RATIO, T.DEFAULT_WARMUP_START_FACTOR, T.DEFAULT_SEED],
    }
    check_frozen(FROZEN / "entry.json", view)
