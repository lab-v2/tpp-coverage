"""
Transformer model components.
"""

import sys
import os

import torch
import torch.nn as nn

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from training_models.models import ModelConfig, MultiHeadAttention, FeedForward


class TransformerLayer(nn.Module):
    """Single transformer layer."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Multi-head attention
        self.attention = MultiHeadAttention(config)
        self.ln1 = nn.LayerNorm(config.d_model)

        # Feed-forward network
        self.ff = FeedForward(config)
        self.ln2 = nn.LayerNorm(config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out = self.attention(x, attention_mask)
        x = self.ln1(x + self.dropout(attn_out))

        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout(ff_out))

        return x


class SimpleTransformer(nn.Module):
    """Simple decoder-only transformer."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.n_layers)]
        )

        # Final layer norm and output projection
        self.ln_final = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights with proper scaling."""
        if isinstance(module, nn.Linear):
            # Use normal initialization with small std
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Use normal initialization for embeddings
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create position ids
        position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(position_ids)
        x = token_embeds + pos_embeds

        # Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Final layer norm and projection
        x = self.ln_final(x)
        logits = self.output_proj(x)

        return logits
