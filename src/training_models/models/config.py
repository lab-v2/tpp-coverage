"""
Model configuration for transformer models.
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a transformer model."""
    vocab_size: int
    max_seq_len: int
    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int
    dropout: float = 0.1

    @property
    def total_params(self) -> int:
        """Correct parameter count for LlamaForCausalLM (RoPE-aware, SwiGLU FFN).
        Used for metadata, filenames, and (N, D) data points feeding scaling-law
        fits. This matches what state_dict.numel() actually sums to."""
        embed_params = self.vocab_size * self.d_model
        attn_params = 4 * self.d_model * self.d_model
        ff_params = self.d_model * self.d_ff * 3
        ln_params = 2 * self.d_model
        layer_params = attn_params + ff_params + ln_params
        transformer_params = self.n_layers * layer_params
        final_params = self.d_model + self.vocab_size * self.d_model
        return embed_params + transformer_params + final_params

    @property
    def legacy_param_count(self) -> int:
        """Pre-Llama-fix parameter count formula. Used for ARCHITECTURE SEARCH
        ONLY in generate_model_configs() so --model-size N selects the same
        architecture the published HuggingFace references were trained with.
        Never reported externally."""
        embed_params = self.vocab_size * self.d_model
        pos_embed_params = self.max_seq_len * self.d_model
        attn_params = 4 * self.d_model * self.d_model + 4 * self.d_model
        ff_params = self.d_model * self.d_ff * 2 + self.d_ff + self.d_model
        ln_params = 4 * self.d_model
        layer_params = attn_params + ff_params + ln_params
        transformer_params = self.n_layers * layer_params
        final_params = 2 * self.d_model + self.vocab_size * self.d_model
        return embed_params + pos_embed_params + transformer_params + final_params