"""
Models package for the LLM scaling experiments.

This package contains all neural network model components including:
- ModelConfig: Configuration dataclass for transformer models
- SimpleTransformer: Main transformer model
- TransformerLayer: Individual transformer layer
- MultiHeadAttention: Multi-head self-attention mechanism
- FeedForward: Position-wise feed-forward network
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from training_models.models.config import ModelConfig
from training_models.models.attention import MultiHeadAttention
from training_models.models.feedforward import FeedForward
from training_models.models.transformer import SimpleTransformer, TransformerLayer
from training_models.models.factory import create_model

__all__ = [
    "ModelConfig",
    "SimpleTransformer",
    "TransformerLayer",
    "MultiHeadAttention",
    "FeedForward",
    "create_model",
]
