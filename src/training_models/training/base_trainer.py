#!/usr/bin/env python3
"""
Base trainer interface for LLM Scaling Analysis.

Defines the common interface for all training execution schemes.
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import Any

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from training_models.models import ModelConfig



class BaseTrainer(ABC):
    """Base class for all training execution schemes."""

    @abstractmethod
    def train_model(
        self,
        model_config: ModelConfig,
        train_data: list[str],
        val_data: list[str],
        **kwargs,
    ) -> dict[str, Any]:
        """
        Train a model and return results.

        Args:
            model_config: Model configuration
            train_data: Training dataset
            val_data: Validation dataset
            **kwargs: Additional training parameters

        Returns:
            Dictionary containing training results
        """
        pass

    @abstractmethod
    def get_gpu_scheme(self) -> str:
        """
        Return the GPU scheme name.

        Returns:
            String identifying the GPU execution scheme
        """
        pass

    def get_required_gpus(self) -> int:
        """
        Return the number of GPUs required for this scheme.

        Returns:
            Number of GPUs required
        """
        return 1  # Default to single GPU
