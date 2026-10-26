#!/usr/bin/env python3
"""
Contains the PretokenizedDataset class for loading pre-processed binary data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os


class PretokenizedDataset(Dataset):
    """
    A PyTorch Dataset for loading pre-tokenized data from binary files.

    This dataset uses numpy.memmap for zero-overhead, lazy loading
    of token and length data.
    """

    def __init__(self, tokens_path, lengths_path, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len

        if not os.path.exists(tokens_path):
            raise FileNotFoundError(f"Tokens file not found: {tokens_path}")
        if not os.path.exists(lengths_path):
            raise FileNotFoundError(f"Lengths file not found: {lengths_path}")

        # Memory-map the tokens file (uint32)
        self.tokens_data = np.memmap(tokens_path, dtype=np.uint32, mode="r")

        # Memory-map the lengths file (uint16)
        self.lengths_data = np.memmap(lengths_path, dtype=np.uint16, mode="r")

        # Calculate number of samples
        self.num_samples = len(self.lengths_data)

        # Validate file sizes
        expected_token_count = self.num_samples * self.max_seq_len
        if len(self.tokens_data) != expected_token_count:
            raise ValueError(
                f"Token file size mismatch. "
                f"Expected {expected_token_count} tokens, but file has {len(self.tokens_data)}. "
                f"Lengths file has {self.num_samples} samples."
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get the original length
        original_length = int(self.lengths_data[idx])

        # Get the tokens
        start_idx = idx * self.max_seq_len
        end_idx = start_idx + self.max_seq_len
        tokens_np = self.tokens_data[start_idx:end_idx]

        # Convert to torch tensor
        # .astype(np.int64) is crucial as CrossEntropyLoss expects Long tensors
        tokens = torch.from_numpy(tokens_np.astype(np.int64))

        # Create attention mask
        # 1 for the first 'original_length' tokens, 0 for padding
        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
        attention_mask[:original_length] = 1

        return {"input_ids": tokens, "attention_mask": attention_mask}

    def select(self, indices):
        """
        Returns a Subset wrapper that uses the given indices.
        This mimics the .select() method of your LazyJSONLDataset.
        """
        return torch.utils.data.Subset(self, indices)
