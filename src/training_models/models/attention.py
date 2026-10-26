"""
Multi-head attention component for transformer models.
"""

import torch
import torch.nn as nn

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from training_models.models import ModelConfig


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.d_model // config.n_heads

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)),
        )

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        # Project to Q, K, V
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.config.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.config.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.config.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        # Apply causal mask (prevent attending to future tokens)
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(
            causal_mask == 0, -1e9
        )  # Use large negative instead of -inf

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        # Softmax and apply to values
        attn_weights = torch.softmax(scores, dim=-1)

        # Replace any NaN in attention weights with uniform distribution
        if torch.isnan(attn_weights).any():
            attn_weights = torch.nan_to_num(attn_weights, nan=1.0 / seq_len)
        attn_out = torch.matmul(attn_weights, v)

        # Concatenate heads and project
        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )
        return self.out_proj(attn_out)
