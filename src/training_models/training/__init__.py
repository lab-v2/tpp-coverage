"""
Training execution schemes for LLM Scaling Analysis.

This package contains different GPU execution schemes for training:
- Distributed training (same model across GPUs)
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from training_models.training.base_trainer import BaseTrainer
from training_models.training.DistributedTrainer import DistributedTrainer

__all__ = ["BaseTrainer", "DistributedTrainer"]
