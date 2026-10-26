import os

from training_models.core.pretokenized_dataset import PretokenizedDataset
from training_models.utils import print_info, print_success


def _resolve_data_root(dataset_name: str) -> str:
    """
    Resolve the directory holding ``{split}_tokens.bin`` for a dataset.

    Priority:
      1. ``$DATA_DIR/<dataset_name>`` when the DATA_DIR env var is set
      2. ``<repo>/corpora/<dataset_name>`` (where the downloaders write by default)
    """
    data_dir = os.environ.get("DATA_DIR")
    if data_dir:
        return os.path.join(data_dir, dataset_name)
    # This file lives in src/training_models/core/, so the repository root is three levels up.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    return os.path.join(repo_root, "corpora", dataset_name)


def _load_split(dataset_name: str, split: str, max_seq_len: int) -> PretokenizedDataset:
    """Generic loader for any (dataset, split) pair."""
    cache_root = _resolve_data_root(dataset_name)
    tokens_path = os.path.join(cache_root, f"{split}_tokens.bin")
    lengths_path = os.path.join(cache_root, f"{split}_lengths.bin")

    print_info(f"Loading pre-tokenized {dataset_name} {split} dataset (mmap)...")
    print_info(f"  Path: {cache_root}")

    if not os.path.exists(tokens_path) or not os.path.exists(lengths_path):
        raise FileNotFoundError(
            f"Preprocessed files not found at {cache_root}/\n"
            f"  Expected: {tokens_path}\n"
            f"            {lengths_path}\n"
            f"Please run the dataset downloader first, or set DATA_DIR env var."
        )

    dataset = PretokenizedDataset(tokens_path, lengths_path, max_seq_len)
    print_success(f"✓ Pre-tokenized {dataset_name} {split} ready: {len(dataset):,} examples")
    return dataset


class PretokenizedLoader:

    # ════════════════════════════════════════════════════════════
    # Cosmopedia
    # ════════════════════════════════════════════════════════════

    @staticmethod
    def load_cosmo_dataset_pretokenized(
        max_seq_len: int, skip_val: bool
    ) -> PretokenizedDataset:
        train_dataset = _load_split("cosmopedia", "train", max_seq_len)
        val_dataset = None if skip_val else _load_split("cosmopedia", "val", max_seq_len)
        return train_dataset, val_dataset

    @staticmethod
    def load_cosmo_dataset_pretokenized_train(max_seq_len: int) -> PretokenizedDataset:
        return _load_split("cosmopedia", "train", max_seq_len)

    @staticmethod
    def load_cosmo_dataset_pretokenized_val(max_seq_len: int) -> PretokenizedDataset:
        return _load_split("cosmopedia", "val", max_seq_len)

    # ════════════════════════════════════════════════════════════
    # Wikipedia
    # ════════════════════════════════════════════════════════════

    @staticmethod
    def load_wiki_dataset_pretokenized(
        max_seq_len: int, skip_val: bool
    ) -> PretokenizedDataset:
        train_dataset = _load_split("wikipedia", "train", max_seq_len)
        val_dataset = None if skip_val else _load_split("wikipedia", "val", max_seq_len)
        return train_dataset, val_dataset

    @staticmethod
    def load_wiki_dataset_pretokenized_train(max_seq_len: int) -> PretokenizedDataset:
        return _load_split("wikipedia", "train", max_seq_len)

    @staticmethod
    def load_wiki_dataset_pretokenized_val(max_seq_len: int) -> PretokenizedDataset:
        return _load_split("wikipedia", "val", max_seq_len)

    # ════════════════════════════════════════════════════════════
    # peS2o
    # ════════════════════════════════════════════════════════════

    @staticmethod
    def load_pes2o_dataset_pretokenized(
        max_seq_len: int, skip_val: bool
    ) -> PretokenizedDataset:
        train_dataset = _load_split("pes2o", "train", max_seq_len)
        val_dataset = None if skip_val else _load_split("pes2o", "val", max_seq_len)
        return train_dataset, val_dataset

    @staticmethod
    def load_pes2o_dataset_pretokenized_train(max_seq_len: int) -> PretokenizedDataset:
        return _load_split("pes2o", "train", max_seq_len)

    @staticmethod
    def load_pes2o_dataset_pretokenized_val(max_seq_len: int) -> PretokenizedDataset:
        return _load_split("pes2o", "val", max_seq_len)

    # ════════════════════════════════════════════════════════════
    # RedPajama
    # ════════════════════════════════════════════════════════════

    @staticmethod
    def load_redpajama_dataset_pretokenized(
        max_seq_len: int, skip_val: bool
    ) -> PretokenizedDataset:
        train_dataset = _load_split("redpajama", "train", max_seq_len)
        val_dataset = None if skip_val else _load_split("redpajama", "val", max_seq_len)
        return train_dataset, val_dataset

    @staticmethod
    def load_redpajama_dataset_pretokenized_train(max_seq_len: int) -> PretokenizedDataset:
        return _load_split("redpajama", "train", max_seq_len)

    @staticmethod
    def load_redpajama_dataset_pretokenized_val(max_seq_len: int) -> PretokenizedDataset:
        return _load_split("redpajama", "val", max_seq_len)

    # ════════════════════════════════════════════════════════════
    # C4
    # ════════════════════════════════════════════════════════════

    @staticmethod
    def load_c4_dataset_pretokenized(
        max_seq_len: int, skip_val: bool
    ) -> PretokenizedDataset:
        train_dataset = _load_split("c4", "train", max_seq_len)
        val_dataset = None if skip_val else _load_split("c4", "val", max_seq_len)
        return train_dataset, val_dataset

    @staticmethod
    def load_c4_dataset_pretokenized_train(max_seq_len: int) -> PretokenizedDataset:
        return _load_split("c4", "train", max_seq_len)

    @staticmethod
    def load_c4_dataset_pretokenized_val(max_seq_len: int) -> PretokenizedDataset:
        return _load_split("c4", "val", max_seq_len)