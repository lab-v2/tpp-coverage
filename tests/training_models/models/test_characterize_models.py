"""Characterization of `src/training_models/models/`: configs, the factory and both architectures.

Each model is built seeded on the CPU and run on real text (the repository's LICENSE through the
trainer's tokenizer, cl100k_base); its weights, logits and parameter counts are compared with the
base commit's values in `frozen/` (see `tests/characterize.py`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dataclasses
from pathlib import Path

import pytest
import tiktoken
import torch

from characterize import check_frozen
from conftest import REPO
from training_models.core.utils import PUBLISHED_ARCHITECTURES
from training_models.models import ModelConfig, create_model
from training_models.models import factory

if TYPE_CHECKING:
    import numpy as np

FROZEN = Path(__file__).parent / "frozen"
ENC = tiktoken.get_encoding("cl100k_base")
SEQ = 24


def license_batch(rows: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """`rows` rows of SEQ LICENSE tokens, the last row's final third masked as padding."""
    ids = ENC.encode((REPO / "LICENSE").read_text())[: rows * SEQ]
    ids = torch.tensor(ids, dtype=torch.long).reshape(rows, SEQ)
    mask = torch.ones_like(ids)
    mask[-1, 2 * SEQ // 3 :] = 0
    return ids, mask


def tiny(**kw: object) -> ModelConfig:
    spec = dict(vocab_size=ENC.n_vocab, max_seq_len=SEQ, n_layers=2, n_heads=2, d_model=8, d_ff=32)
    return ModelConfig(**{**spec, **kw})


def weights(model: torch.nn.Module) -> dict[str, np.ndarray]:
    return {k: v.detach().numpy() for k, v in model.state_dict().items()}


def test_param_counts_of_the_published_architectures() -> None:
    view = {}
    for target, spec in sorted(PUBLISHED_ARCHITECTURES.items()):
        cfg = ModelConfig(vocab_size=ENC.n_vocab, max_seq_len=2048, **spec)
        view[str(target)] = [cfg.total_params, cfg.legacy_param_count, cfg.dropout]
    view["fields"] = [f.name for f in dataclasses.fields(ModelConfig)]
    check_frozen(FROZEN / "param_counts.json", view)


@pytest.mark.parametrize("architecture", ["llama", "custom"])
def test_seeded_model_weights_and_logits(architecture: str) -> None:
    torch.manual_seed(0)
    model = create_model(tiny(), architecture=architecture.upper()).eval()
    ids, mask = license_batch()
    with torch.no_grad():
        view = {
            "type": type(model).__name__,
            "weights": weights(model),
            "logits_masked": model(ids, attention_mask=mask).numpy(),
            "logits_unmasked": model(ids).numpy(),
        }
    if architecture == "llama":
        view["total_params"] = model.total_params
        view["actual_params"] = model.get_actual_param_count()
        llama = factory.model_config_to_llama_config(tiny())
        keys = ["vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers"]
        keys += ["num_attention_heads", "max_position_embeddings", "rms_norm_eps", "use_cache"]
        keys += ["pad_token_id", "num_key_value_heads", "hidden_act"]
        view["llama_config"] = {k: getattr(llama, k) for k in keys}
    check_frozen(FROZEN / f"model_{architecture}.json", view)


def test_custom_model_trains_one_step() -> None:
    """One AdamW step of the custom transformer in train mode (dropout on), seeded."""
    torch.manual_seed(1)
    model = create_model(tiny(dropout=0.1), architecture="custom").train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    ids, mask = license_batch()
    logits = model(ids[:, :-1], attention_mask=mask[:, :-1])
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), ids[:, 1:].ravel()
    )
    loss.backward()
    opt.step()
    check_frozen(FROZEN / "custom_step.json", {"loss": loss.item(), "weights": weights(model)})


def test_unknown_architecture_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown architecture: gpt"):
        create_model(tiny(), architecture="gpt")


def test_fp8_compatibility_and_replacement_stats() -> None:
    """Which shapes FP8 accepts, and the replacement count on a model none of whose layers fit
    (so Transformer Engine, a GPU-only package, is never imported)."""
    grid = [[factory._is_fp8_compatible(i, o) for o in (8, 12, 16, 24)] for i in (8, 16, 32, 48)]
    model = create_model(tiny(d_model=12, d_ff=36, n_heads=2), architecture="custom")
    stats = factory.replace_linear_with_fp8(model)
    check_frozen(FROZEN / "fp8.json", {"grid": grid, "stats": stats})
