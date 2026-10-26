#!/usr/bin/env python3
"""
Checkpoint utilities for distributed training.

Extracted from the working evaluate_holdout_distributed.py implementation.
"""

import re
import torch
import torch.nn as nn
import tiktoken

# Import utils and models
from training_models.utils import print_info
from training_models.models import ModelConfig, create_model


def extract_model_params_from_filename(filename: str) -> tuple[int, int, float, int]:
    """Extract model parameters from checkpoint filename."""
    # Extract model size (m)
    m_match = re.search(r"_m(\d+)_", filename)
    if not m_match:
        raise ValueError(f"Could not extract model size from filename: {filename}")
    m = int(m_match.group(1))

    # Extract psi
    psi_match = re.search(r"_psi(\d+)_", filename)
    if not psi_match:
        raise ValueError(f"Could not extract psi from filename: {filename}")
    psi = int(psi_match.group(1))

    # Extract learning rate
    lr_match = re.search(r"_lr([\d.e-]+)_", filename)
    if not lr_match:
        raise ValueError(f"Could not extract learning rate from filename: {filename}")
    lr_str = lr_match.group(1)
    lr = float(lr_str.replace("e-", "e-"))

    # Extract epoch
    epoch_match = re.search(r"_epoch(\d+)\.pt", filename)
    if not epoch_match:
        raise ValueError(f"Could not extract epoch from filename: {filename}")
    epoch = int(epoch_match.group(1))

    return m, psi, lr, epoch


def load_model_safely(
    checkpoint_path: str, device: torch.device, architecture: str = "llama"
) -> tuple[nn.Module, dict]:
    """Load model with proper error handling and memory management, handling checkpoint key mismatch."""
    print_info(f"Loading checkpoint: {checkpoint_path} (Architecture: {architecture})")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    # Extract model config
    model_config_dict = checkpoint.get("model_config", {})
    if not model_config_dict:
        raise RuntimeError("No model config found in checkpoint")

    required_fields = ["d_model", "n_heads", "n_layers", "max_seq_len"]
    for field in required_fields:
        if field not in model_config_dict:
            raise RuntimeError(f"Missing required field '{field}' in checkpoint")

    # Create tokenizer and model config
    tokenizer = tiktoken.get_encoding("cl100k_base")
    model_config = ModelConfig(
        vocab_size=tokenizer.n_vocab,
        max_seq_len=model_config_dict["max_seq_len"],
        n_layers=model_config_dict["n_layers"],
        n_heads=model_config_dict["n_heads"],
        d_model=model_config_dict["d_model"],
        d_ff=model_config_dict["d_model"] * 4,
        dropout=model_config_dict.get("dropout", 0.1),
    )

    # Create model
    model = create_model(model_config, architecture=architecture)

    state_dict_raw = checkpoint.get("model_state_dict", {})
    if not state_dict_raw:
        raise RuntimeError("Key 'model_state_dict' not found in checkpoint.")

    state_dict_to_load = {}

    # Handle key mapping based on architecture
    if architecture == "custom":
        # Logic for SimpleTransformer (legacy handling)
        prefix_to_strip = "model.model."

        for k, v in state_dict_raw.items():
            if k.startswith(prefix_to_strip):
                # Strip the known problematic prefix
                new_key = k[len(prefix_to_strip) :]
            elif k.startswith("model.") and not k.startswith(prefix_to_strip):
                # Handle possible single 'model.' wrapper
                new_key = k[len("model.") :]
            else:
                new_key = k

            state_dict_to_load[new_key] = v
    elif architecture == "llama":
        # For Llama use state dict as is
        # The LlamaModelWrapper structure (model.model...) usually matches the checkpoint
        state_dict_to_load = state_dict_raw

    # Load state dict
    try:
        # Load the cleaned state dict, keeping strict=False just in case
        model.load_state_dict(state_dict_to_load, strict=False)
    except Exception as e:
        # This will catch any remaining fundamental structural mismatches
        raise RuntimeError(f"Failed to load model state dict after key stripping: {e}")

    # Move to device
    model = model.to(device)

    return model, checkpoint
