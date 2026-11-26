"""Characterization of `src/training_models/core/`: the config search and the pre-tokenized data.

The pre-tokenized files are written the way the downloaders write them (rows of `SEQ` cl100k_base
tokens as uint32, padded with 0, and their lengths as uint16) from real text, the repository's
LICENSE. Expected values are the base commit's, frozen in `frozen/` (see `tests/characterize.py`).
`count_total_tokens` is left out: it starts a fixed pool of 10 processes, which inside an xdist
worker oversubscribes the host.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import tiktoken
import torch

from characterize import check_frozen
from conftest import REPO
from training_models.core import PretokenizedDataset, PretokenizedLoader, generate_model_configs
from training_models.core.PretokenizedLoader import _resolve_data_root

FROZEN = Path(__file__).parent / "frozen"
ENC = tiktoken.get_encoding("cl100k_base")
SEQ = 32
CORPORA = ["cosmopedia", "wikipedia", "pes2o", "redpajama", "c4", "fineweb"]


def configs(cfgs) -> list:
    return [dataclasses.asdict(c) if c is not None else None for c in cfgs]


def test_generate_model_configs():
    v = ENC.n_vocab
    view = {
        # Published targets, round and true counts: the exact released architectures.
        "published": configs(
            generate_model_configs(3, v, 2048, target_params=[5_000_000, 10_070_160, 76_520_192])
        ),
        # Off-list targets fall through to the nearest candidates of the search.
        "searched": configs(
            generate_model_configs(3, v, 256, target_params=[3_300_000, 4_000_000, 9_999_999])
        ),
        "mixed": configs(generate_model_configs(2, v, 256, target_params=[5_000_000, 3_300_000])),
        # No targets: log-spaced over the bounds.
        "logspaced": configs(generate_model_configs(6, v, 512, 1_000_000, 30_000_000)),
        "too_many": len(generate_model_configs(10_000, v, 512, 3_000_000, 4_000_000)),
        "none_fit": configs(generate_model_configs(3, v, 512, 10, 20)),
    }
    check_frozen(FROZEN / "model_configs.json", view)


def write_split(root: Path, name: str, split: str) -> None:
    """LICENSE paragraphs as one pre-tokenized split under `root/name`."""
    paragraphs = [p for p in (REPO / "LICENSE").read_text().split("\n\n") if p.strip()]
    rows = [
        ENC.encode(p)[:SEQ] for p in (paragraphs[:20] if split == "train" else paragraphs[20:30])
    ]
    tokens = np.zeros((len(rows), SEQ), dtype=np.uint32)
    for i, r in enumerate(rows):
        tokens[i, : len(r)] = r
    (root / name).mkdir(parents=True, exist_ok=True)
    tokens.tofile(root / name / f"{split}_tokens.bin")
    np.array([len(r) for r in rows], dtype=np.uint16).tofile(root / name / f"{split}_lengths.bin")


@pytest.fixture
def data_dir(tmp_path, monkeypatch) -> Path:
    for name in CORPORA:
        write_split(tmp_path, name, "train")
        write_split(tmp_path, name, "val")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    return tmp_path


def test_dataset_items(data_dir):
    ds = PretokenizedDataset(
        data_dir / "c4" / "train_tokens.bin", data_dir / "c4" / "train_lengths.bin", SEQ
    )
    items = [ds[i] for i in range(len(ds))]
    view = {
        "len": len(ds),
        "input_ids": torch.stack([x["input_ids"] for x in items]).numpy(),
        "attention_mask": torch.stack([x["attention_mask"] for x in items]).numpy(),
        "dtypes": [str(items[0]["input_ids"].dtype), str(items[0]["attention_mask"].dtype)],
        "select": [len(ds.select(range(3))), type(ds.select([0])).__name__],
    }
    check_frozen(FROZEN / "dataset_items.json", view)


def test_dataset_rejects_bad_files(data_dir):
    tokens, lengths = data_dir / "c4" / "train_tokens.bin", data_dir / "c4" / "train_lengths.bin"
    with pytest.raises(FileNotFoundError, match="Tokens file not found"):
        PretokenizedDataset(data_dir / "missing.bin", lengths, SEQ)
    with pytest.raises(FileNotFoundError, match="Lengths file not found"):
        PretokenizedDataset(tokens, data_dir / "missing.bin", SEQ)
    with pytest.raises(ValueError, match="Token file size mismatch"):
        PretokenizedDataset(tokens, lengths, SEQ + 1)


def test_every_loader_reads_its_corpus(data_dir):
    view = {}
    for name in dir(PretokenizedLoader):
        if not name.startswith("load_"):
            continue
        fn = getattr(PretokenizedLoader, name)
        if name.endswith(("_train", "_val")):
            out = [fn(SEQ)]
        else:
            out = list(fn(SEQ, False)) + [x for x in fn(SEQ, True)]
        view[name] = [
            None
            if d is None
            else [
                Path(d.tokens_data.filename).parent.name,
                Path(d.tokens_data.filename).name,
                len(d),
            ]
            for d in out
        ]
    check_frozen(FROZEN / "loaders.json", view)


def test_data_root(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert _resolve_data_root("c4") == str(tmp_path / "c4")
    monkeypatch.delenv("DATA_DIR")
    assert _resolve_data_root("c4") == str(REPO / "corpora" / "c4")
    with pytest.raises(FileNotFoundError, match="Preprocessed files not found"):
        PretokenizedLoader.load_c4_dataset_pretokenized_train(SEQ)
