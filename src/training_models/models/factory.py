"""
Model factory for creating different architectures.
Supports Llama (default, as used in Cosmo-1B) and SimpleTransformer (custom).
"""

import sys
import os

import torch
import torch.nn as nn

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from training_models.models import ModelConfig, SimpleTransformer
from transformers import LlamaForCausalLM, LlamaConfig
import torch.nn.functional as F


class LlamaModelWrapper(nn.Module):
    """Wrapper for LlamaForCausalLM to match SimpleTransformer interface."""

    def __init__(self, llama_model: nn.Module, model_config: ModelConfig):
        super().__init__()
        self.model = llama_model
        self.config = model_config

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass compatible with SimpleTransformer interface."""
        # Llama expects attention_mask in a specific format
        if attention_mask is not None:
            # Convert attention mask to the format Llama expects
            # Llama uses causal mask internally, but we can pass attention_mask
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.model(input_ids=input_ids)

        # Return logits (shape: [batch_size, seq_len, vocab_size])
        return outputs.logits

    @property
    def total_params(self) -> int:
        """Return total parameters from wrapped config."""
        return self.config.total_params

    def get_actual_param_count(self) -> int:
        """Get actual parameter count from the Llama model."""
        return sum(p.numel() for p in self.model.parameters())


def model_config_to_llama_config(model_config: ModelConfig) -> LlamaConfig:
    """Convert ModelConfig to LlamaConfig."""
    # Map ModelConfig to LlamaConfig
    # Llama uses different naming conventions:
    # - d_model -> hidden_size
    # - n_layers -> num_hidden_layers
    # - n_heads -> num_attention_heads
    # - d_ff -> intermediate_size
    # - max_seq_len -> max_position_embeddings

    llama_config = LlamaConfig(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.d_model,
        intermediate_size=model_config.d_ff,
        num_hidden_layers=model_config.n_layers,
        num_attention_heads=model_config.n_heads,
        max_position_embeddings=model_config.max_seq_len,
        rms_norm_eps=1e-6,
        use_cache=False,  # Disable cache for training
        pad_token_id=None,
    )

    return llama_config


class PaddedFP8Linear(nn.Module):
    """Wrapper around te.Linear that auto-pads inputs for FP8 dimension constraints.

    FP8 Tensor Core GEMMs require:
      - 2D input with height (batch*seq) divisible by 8
      - 2D input with width (in_features) divisible by 16

    The width constraint is checked at construction time (and we skip layers that
    don't meet it). The height constraint depends on runtime batch/seq sizes, so
    this wrapper transparently pads the batch dimension to the next multiple of 8,
    runs the FP8 GEMM, then strips the padding from the output.

    This ensures FP8 works with arbitrary batch sizes and sequence lengths, which
    is critical for scaling law experiments where model dimensions vary widely.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        import transformer_engine.pytorch as te

        self.te_linear = te.Linear(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original shape - input could be 2D [N, C] or 3D [B, S, C]
        orig_shape = x.shape

        # Flatten to 2D: [batch*seq, features]
        x_2d = x.reshape(-1, x.shape[-1])
        h = x_2d.shape[0]

        # Pad height to next multiple of 8 for FP8 compliance
        remainder = h % 16
        if remainder != 0:
            pad_size = 16 - remainder
            x_2d = F.pad(x_2d, (0, 0, 0, pad_size))  # pad along dim 0
        else:
            pad_size = 0

        # Run FP8 GEMM via Transformer Engine
        # Inside fp8_autocast context: uses FP8 Tensor Cores
        # Outside fp8_autocast context: falls back to regular precision
        out_2d = self.te_linear(x_2d)

        # Strip padding rows
        if pad_size > 0:
            out_2d = out_2d[:h]

        # Reshape back to original batch dimensions
        out_shape = orig_shape[:-1] + (self.out_features,)
        return out_2d.reshape(out_shape)


def _is_fp8_compatible(in_features: int, out_features: int) -> bool:
    """Check if layer dimensions are compatible with FP8 requirements.

    Weight matrix constraints for FP8 GEMMs:
      - in_features (width) must be divisible by 16
      - out_features (height) must be divisible by 8
    """
    return in_features % 16 == 0 and out_features % 8 == 0


def replace_linear_with_fp8(module: nn.Module, name_prefix: str = "") -> dict:
    """Recursively replace nn.Linear with PaddedFP8Linear for FP8 support.

    Only replaces layers where weight dimensions meet FP8 requirements.
    Layers with incompatible dimensions (e.g., lm_head with vocab_size=100277)
    are left as regular nn.Linear and will run in BF16/FP32.

    This matches the standard TE training pattern:
      - Linear layers (Q/K/V projections, FFN) → FP8 GEMMs
      - LayerNorm / RMSNorm → higher precision (untouched)
      - Softmax, activations → higher precision (untouched)
      - Embedding, lm_head → higher precision if dims are incompatible

    Returns:
        dict with counts of replaced and skipped layers
    """
    stats = {"replaced": 0, "skipped": 0, "skipped_layers": []}

    for name, child in module.named_children():
        full_name = f"{name_prefix}.{name}" if name_prefix else name

        if isinstance(child, nn.Linear):
            in_f = child.in_features
            out_f = child.out_features

            if _is_fp8_compatible(in_f, out_f):
                fp8_linear = PaddedFP8Linear(in_f, out_f, bias=child.bias is not None)
                # Copy weights from original linear to te.Linear inside wrapper
                # te.Linear stores weight differently, so we use the state_dict approach
                with torch.no_grad():
                    fp8_linear.te_linear.weight.copy_(child.weight)
                    if child.bias is not None:
                        fp8_linear.te_linear.bias.copy_(child.bias)
                setattr(module, name, fp8_linear)
                stats["replaced"] += 1
            else:
                stats["skipped"] += 1
                stats["skipped_layers"].append(f"{full_name} (in={in_f}, out={out_f})")
        else:
            child_stats = replace_linear_with_fp8(child, full_name)
            stats["replaced"] += child_stats["replaced"]
            stats["skipped"] += child_stats["skipped"]
            stats["skipped_layers"].extend(child_stats["skipped_layers"])

    return stats


def create_model(
    model_config: ModelConfig,
    architecture: str = "llama",
    use_fp8: bool = False,
) -> nn.Module:
    """
    Create a model instance based on architecture choice.

    Args:
        model_config: Model configuration
        architecture: Architecture type, either "llama" (default) or "custom"
        use_fp8: Enable FP8 training via Transformer Engine

    Returns:
        Model instance compatible with training interface
    """
    architecture = architecture.lower()

    if architecture == "llama":
        llama_config = model_config_to_llama_config(model_config=model_config)
        llama_model = LlamaForCausalLM(config=llama_config)
        return LlamaModelWrapper(llama_model=llama_model, model_config=model_config)

    elif architecture == "custom":
        return SimpleTransformer(config=model_config)

    else:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Supported architectures: 'llama', 'custom'"
        )
