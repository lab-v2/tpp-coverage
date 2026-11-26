"""Characterization of `src/training_models/distributed/checkpoint_utils.py` on the CPU.

`load_model_safely` reads a checkpoint the trainer's own way (`model_state_dict` and
`model_config` keys) written from a seeded model built by the factory. `setup.py` is left out: it
calls `torch.cuda.set_device` and is covered by nothing on a CPU host. Expected values are the
base commit's outputs, frozen in `frozen/` (see `tests/characterize.py`).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import tiktoken
import torch

from characterize import check_frozen
from training_models.distributed import checkpoint_utils as C
from training_models.models import ModelConfig, create_model

FROZEN = Path(__file__).parent / "frozen"
VOCAB = tiktoken.get_encoding("cl100k_base").n_vocab


def test_extract_model_params_from_filename():
    good = [
        "wikipedia_m3213008_psi12_lr0.001_epoch3.pt",
        "c4_m5035656_psi200_lr1e-3_x_epoch10.pt",
        "wikipedia_m3_psi1_lr0.1_epoch3.pth",  # the epoch pattern also takes .pth
    ]
    bad = [
        "wikipedia_psi12_lr0.001_epoch3.pt",
        "wikipedia_m3_lr0.001_epoch3.pt",
        "wikipedia_m3_psi1_epoch3.pt",
        "wikipedia_m3_psi1_lr0.1_epoch3",
    ]
    errors = []
    for name in bad:
        with pytest.raises(ValueError) as e:
            C.extract_model_params_from_filename(name)
        errors.append(str(e.value))
    view = {"good": [list(C.extract_model_params_from_filename(n)) for n in good], "bad": errors}
    check_frozen(FROZEN / "filenames.json", view)


@pytest.mark.parametrize("architecture", ["llama", "custom"])
def test_load_model_safely(tmp_path, architecture):
    cfg = ModelConfig(VOCAB, 16, n_layers=1, n_heads=2, d_model=8, d_ff=32, dropout=0.0)
    torch.manual_seed(3)
    model = create_model(cfg, architecture=architecture)
    state = model.state_dict()
    if architecture == "custom":  # a DDP-and-wrapper prefix, as old checkpoints carry
        state = {
            f"model.model.{k}" if i % 2 else f"model.{k}": v
            for i, (k, v) in enumerate(state.items())
        }
    path = tmp_path / "ckpt.pt"
    torch.save(
        {"model_state_dict": state, "model_config": dataclasses.asdict(cfg), "epoch": 2}, path
    )
    loaded, ckpt = C.load_model_safely(str(path), torch.device("cpu"), architecture=architecture)
    for k, v in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v), k
    view = {
        "type": type(loaded).__name__,
        "config": dataclasses.asdict(loaded.config),
        "epoch": ckpt["epoch"],
    }
    check_frozen(FROZEN / f"load_{architecture}.json", view)


def test_load_model_safely_errors(tmp_path):
    cases = {
        "unreadable": b"not a checkpoint",
        "no_config": {"model_state_dict": {"w": torch.zeros(1)}},
        "missing_field": {"model_config": {"d_model": 8, "n_heads": 2, "n_layers": 1}},
        "no_state": {"model_config": {"d_model": 8, "n_heads": 2, "n_layers": 1, "max_seq_len": 4}},
    }
    # The loader rebuilds d_ff as 4 * d_model, so a model with any other width cannot load.
    cfg = ModelConfig(VOCAB, 16, n_layers=1, n_heads=2, d_model=8, d_ff=24)
    state = create_model(cfg).state_dict()
    cases["other_d_ff"] = {"model_state_dict": state, "model_config": dataclasses.asdict(cfg)}
    errors = {}
    for name, content in cases.items():
        path = tmp_path / f"{name}.pt"
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            torch.save(content, path)
        with pytest.raises(RuntimeError) as e:
            C.load_model_safely(str(path), torch.device("cpu"))
        errors[name] = str(e.value).split(":")[0]
    check_frozen(FROZEN / "load_errors.json", errors)
