#!/usr/bin/env python3

import sys
import os

# ============================================================================
# DETERMINISM: Set environment variables BEFORE importing torch
# This must be at the very top of the script
# ============================================================================
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"  # Ensure Python hash seed is deterministic

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tiktoken.core import Encoding
import numpy as np
import json
from datetime import datetime
import tiktoken
import time
import glob
import re
import random  # Added for deterministic seeding

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


from training_models.training.base_trainer import BaseTrainer
from training_models.models import ModelConfig, create_model
from training_models.core import (
    PretokenizedDataset,
    PretokenizedLoader,
    count_total_tokens,
)
from training_models.utils import print_info, print_success, print_section
from training_models.distributed.checkpoint_utils import load_model_safely

from dist_trainer_main import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_KEEP_CHECKPOINTS,
    DEFAULT_WEIGHT_DECAY,
    SHARED_NVME_DIR,
)

DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_WARMUP_START_FACTOR = 0.01
DEFAULT_WARMUP_END_FACTOR = 1.0
DEFAULT_COSINE_ANNEALING_MIN_RATIO = 0.1

# Default seed for reproducibility
DEFAULT_SEED = 42


# ============================================================================
# DETERMINISM UTILITIES
# ============================================================================


def set_seed(seed: int = DEFAULT_SEED, rank: int = 0) -> None:
    """
    Set seeds for all random number generators to ensure reproducibility.

    This function sets seeds for:
    - Python's random module
    - NumPy's random generator
    - PyTorch CPU random generator
    - PyTorch CUDA random generator (all GPUs)
    - CuDNN deterministic flags

    Args:
        seed: Base seed value (default: 42)
        rank: DDP rank for rank-specific seeding if needed
    """
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch CPU seed
    torch.manual_seed(seed)

    # Set PyTorch CUDA seed for all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # CuDNN deterministic settings
    # This forces CuDNN to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    # Disable benchmark mode which can introduce non-determinism
    torch.backends.cudnn.benchmark = False

    # Enable PyTorch's deterministic algorithms globally
    # This will raise an error if a non-deterministic operation is attempted
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """
    Worker initialization function for DataLoader to ensure deterministic data loading.

    Each DataLoader worker needs its own seed derived from the initial seed.
    Without this, workers may use system time or other entropy sources,
    leading to non-deterministic data ordering across runs.

    Args:
        worker_id: The worker ID assigned by DataLoader
    """
    # Get the seed from PyTorch's initial seed (set by generator)
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_deterministic_generator(seed: int = DEFAULT_SEED) -> torch.Generator:
    """
    Create a PyTorch Generator with a fixed seed for DataLoader.

    This generator is passed to the DataLoader to ensure deterministic
    shuffling and batching operations.

    Args:
        seed: Seed value for the generator

    Returns:
        A seeded torch.Generator instance
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


class DistributedTrainer(BaseTrainer):
    """Distributed training implementation using PyTorch DDP with full features."""

    def __init__(self, seed: int = DEFAULT_SEED):
        """
        Initialize the DistributedTrainer.

        Args:
            seed: Random seed for reproducibility (default: 42)
        """
        super().__init__()
        self.seed = seed

    def get_gpu_scheme(self) -> str:
        return "distributed"

    def get_required_gpus(self) -> int:
        return torch.cuda.device_count() if torch.cuda.is_available() else 2

    def set_current_epoch_initial_tokens(
        self, resume_checkpoint: str, rank: int
    ) -> tuple[int, int]:
        """Set the current epoch and initial tokens based on the resume checkpoint."""

        ckpt = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        start_epoch = ckpt["epoch"]
        initial_tokens = ckpt["total_tokens"]
        if rank == 0:
            print_info(f"Resuming from epoch {start_epoch}, tokens {initial_tokens:,}")

        return start_epoch, initial_tokens

    def load_dataset(
        self, dataset_name: str, max_seq_len: int, skip_val: bool
    ) -> tuple[list[str] | None, list[str] | None]:
        """Load dataset based on name."""
        train_dataset = None
        val_dataset = None
        dataset_name = dataset_name.lower()

        if dataset_name == "cosmopedia":
            train_dataset, val_dataset = (
                PretokenizedLoader.load_cosmo_dataset_pretokenized(
                    max_seq_len, skip_val
                )
            )
        elif dataset_name == "wikipedia":
            train_dataset, val_dataset = (
                PretokenizedLoader.load_wiki_dataset_pretokenized(max_seq_len, skip_val)
            )
        elif dataset_name == "pes2o":
            train_dataset, val_dataset = (
                PretokenizedLoader.load_pes2o_dataset_pretokenized(
                    max_seq_len, skip_val
                )
            )
        elif dataset_name == "redpajama":
            train_dataset, val_dataset = (
                PretokenizedLoader.load_redpajama_dataset_pretokenized(
                    max_seq_len, skip_val
                )
            )
        elif dataset_name == "c4":
            train_dataset, val_dataset = (
                PretokenizedLoader.load_c4_dataset_pretokenized(
                    max_seq_len, skip_val
                )
            )

        return train_dataset, val_dataset

    def train_model(
        self,
        model_config: ModelConfig,
        train_data: list[str] | None = None,
        checkpoint_prefix: str | None = None,
        **kwargs,
    ) -> dict:
        """Train models using distributed training across multiple GPUs."""
        epochs = kwargs["epochs"]
        lr = kwargs["lr"]
        batch_size = kwargs["batch_size"]
        # psi = kwargs.get('psi', None)
        tokens = kwargs["tokens"]
        dataset_name = kwargs["dataset"]
        max_seq_len = kwargs["max_seq_len"]
        log_interval = kwargs["log_interval"]
        architecture = kwargs["architecture"]
        val_tokens_limit = kwargs["val_tokens_limit"]
        shared_dir = kwargs["shared_dir"]
        skip_val = kwargs["skip_val"]
        keep_checkpoints = kwargs["keep_checkpoints"]
        weight_decay = kwargs["weight_decay"]
        use_bf16 = kwargs.get("use_bf16", False)

        # Get seed from kwargs or use default
        seed = kwargs.get("seed", self.seed)

        val_data = None

        # Resume parameters
        resume_checkpoint_path = kwargs["resume_checkpoint_path"]
        _resume_metrics_path = kwargs["resume_metrics_path"]
        prior_metrics = kwargs["prior_metrics"]

        # Custom output prefix (optional)
        output_prefix = kwargs.get("output_prefix", None)

        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # ====================================================================
        # DETERMINISM: Set seeds before any random operations
        # This must happen before model initialization, data loading, etc.
        # ====================================================================
        set_seed(seed=seed, rank=rank)

        if rank == 0:
            print_info(f"Determinism enabled with seed={seed}")
            print_info(f"  - Precision: {'BF16' if use_bf16 else 'FP32'}")
            print_info(
                f"  - CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}"
            )
            print_info(
                f"  - torch.backends.cudnn.deterministic={torch.backends.cudnn.deterministic}"
            )
            print_info(
                f"  - torch.backends.cudnn.benchmark={torch.backends.cudnn.benchmark}"
            )


        if not dist.is_initialized():
            raise RuntimeError(
                "Distributed training not initialized. Must be launched with torchrun."
            )

        dataset_name = dataset_name.lower()
        architecture = architecture.lower()

        # Determine start_epoch and initial_tokens from checkpoint

        if resume_checkpoint_path is not None:
            start_epoch, initial_tokens = self.set_current_epoch_initial_tokens(
                resume_checkpoint_path, rank
            )
        else:
            start_epoch, initial_tokens = 0, 0

        if rank == 0:
            print_section("Running Distributed Training")
            print_info("Training setup:")
            print_info(
                f"  - Trainer: {self.get_gpu_scheme()} (world_size={world_size})"
            )
            print_info(f"  - Epochs: {start_epoch} -> {epochs}")
            print_info(f"  - Batch size: {batch_size} per GPU")
            print_info(f"  - Learning rate: {lr}")
            print_info(f"  - Dataset: {dataset_name}")
            print_info(f"  - Skip Validation: {skip_val}")
            if not skip_val:
                print_info(f"  - Val Token Limit: {val_tokens_limit/1e6:.1f}M")
            print_info(f"  - Model params: {model_config.total_params}")
            print_info(f"  - Shared NVMe dir: {shared_dir}")
            print_info(f"  - Keep checkpoints: {keep_checkpoints}")
            print_info(f"  - Seed: {seed}")
            if resume_checkpoint_path:
                print_info(f"  - Resuming from: {resume_checkpoint_path}")
            if output_prefix:
                print_info(f"  - Output prefix: {output_prefix}")

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(rank)

        tokenizer = tiktoken.get_encoding("cl100k_base")

        if train_data is None:
            train_data, val_data = self.load_dataset(
                dataset_name=dataset_name, max_seq_len=max_seq_len, skip_val=skip_val
            )

        if train_data is not None and not skip_val:
            _, val_data = self.load_dataset(
                dataset_name=dataset_name, max_seq_len=max_seq_len, skip_val=skip_val
            )

        if train_data is None:
            raise ValueError("Training data could not be loaded.")

        if val_data is None and not skip_val:
            raise ValueError("Val Data could not be loaded, val data requested")

        if tokens is not None and rank == 0:

            if isinstance(
                train_data, PretokenizedDataset
            ):  # cosmopedia; everything needs to be pretokenized here on out
                total_tokens = int(train_data.lengths_data.sum())

            elif isinstance(train_data, Subset) and isinstance(
                train_data.dataset, PretokenizedDataset
            ):
                # It's a Subset of PretokenizedDataset
                indices = train_data.indices
                total_tokens = int(train_data.dataset.lengths_data[indices].sum())
            else:
                total_tokens = count_total_tokens(
                    texts=train_data,
                    tokenizer=tokenizer,
                    max_seq_len=model_config.max_seq_len,
                )

            required_tokens = int(tokens)
            if total_tokens < required_tokens:
                raise ValueError(
                    f"Insufficient train tokens: {total_tokens:,.0f} < {required_tokens:,}"
                )

        if checkpoint_prefix is None:
            lr_str = f"{lr:.10f}".rstrip("0").rstrip(".").replace(".", "-")
            token_str = f"_token{int(tokens)}"
            base_prefix = f"{dataset_name}_m{model_config.total_params}{token_str}_lr{lr_str}_distributed"

            # Prepend custom output_prefix if provided
            if output_prefix:
                checkpoint_prefix = f"{output_prefix}_{base_prefix}"
            else:
                checkpoint_prefix = base_prefix

        # If resuming, add suffix to avoid overwriting source checkpoint
        if resume_checkpoint_path is not None:
            resume_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_prefix = f"{checkpoint_prefix}_resumed_{resume_timestamp}"
            if rank == 0:
                print_info(f"  - New checkpoint prefix: {checkpoint_prefix}")

        model = create_model(
            model_config=model_config, architecture=architecture, use_fp8=False
        )

        # Load model weights if resuming (before .bfloat16() so dtypes match checkpoint)
        if resume_checkpoint_path is not None:
            loaded_model, checkpoint = load_model_safely(resume_checkpoint_path, device)
            model.load_state_dict(loaded_model.state_dict())
            del loaded_model
            if rank == 0:
                print_success(f"Loaded model weights from epoch {checkpoint['epoch']}")

        # BF16: convert all parameters and buffers to bfloat16
        if use_bf16:
            model.bfloat16()
            if rank == 0:
                print_info("Model converted to BF16 via .bfloat16()")

        model = model.to(device)
        model = DDP(model, device_ids=[rank])

        # ====================================================================
        # DETERMINISM: Create generator for DataLoader
        # ====================================================================
        dataloader_generator = create_deterministic_generator(seed)

        val_loader = None

        if not skip_val:
            val_dataset = val_data
            val_sampler = DistributedSampler(
                dataset=val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=seed,  # Explicit seed for sampler
            )
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=0,
                pin_memory=True,
                worker_init_fn=seed_worker,  # Seed workers for determinism
                generator=dataloader_generator,  # Deterministic generator
            )

        train_sampler = DistributedSampler(
            dataset=train_data,
            num_replicas=world_size,
            rank=rank,
            seed=seed,  # Explicit seed for sampler determinism
        )
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True,
            worker_init_fn=seed_worker,  # Seed workers for determinism
            generator=dataloader_generator,  # Deterministic generator
        )

        final_train_loss, epoch_losses, metrics = self._train_model_distributed(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            checkpoint_prefix=checkpoint_prefix,
            model_config=model_config,
            rank=rank,
            world_size=world_size,
            log_interval=log_interval,
            val_tokens_limit=val_tokens_limit,
            start_epoch=start_epoch,
            resume_checkpoint=resume_checkpoint_path,
            initial_tokens=initial_tokens,
            prior_metrics=prior_metrics,
            batch_size=batch_size,
            shared_dir=shared_dir,
            keep_checkpoints=keep_checkpoints,
            weight_decay=weight_decay,
            seed=seed,  # Pass seed for metadata
            use_bf16=use_bf16,
        )

        dist.barrier()

        final_val_loss = None
        if rank == 0 and metrics.get("val_losses"):
            final_val_loss = metrics["val_losses"][-1]

        return {
            "final_train_loss": final_train_loss,
            "epoch_losses": epoch_losses,
            "final_val_loss": final_val_loss,
            "model_size": model_config.total_params,
            "gpu_scheme": self.get_gpu_scheme(),
            "seed": seed,
            "precision": "bf16" if use_bf16 else "fp32",
        }

    def _estimate_average_tokens_per_example(
        self,
        data: Dataset,
        tokenizer: Encoding,
        max_seq_len: int,
        sample_size: int = 10000,
    ):
        """Estimate average tokens per example for datasets with length metadata.
        Input: data: PretokenizedDataset
        Output: average tokens per example (float)
        """

        if isinstance(data, PretokenizedDataset):
            # limit = min(len(data), sample_size)
            # lengths_sample = data.lengths_data[:limit]
            return float(data.lengths_data.mean())

        else:
            underlying_data = data.dataset
            indices = (
                data.indices if hasattr(data, "indices") else list(range(len(data)))
            )
            limit = min(len(indices), sample_size)
            lengths_sample = underlying_data.lengths_data[indices[:limit]]
            return float(lengths_sample.mean())

    def _save_initial_metadata(
        self,
        checkpoint_prefix,
        model_config,
        lr,
        epochs,
        batch_size,
        weight_decay,
        seed=DEFAULT_SEED,
        shared_dir=SHARED_NVME_DIR,
    ):
        """Save initial training config metadata before training starts."""
        try:
            metadata = {
                "checkpoint_prefix": checkpoint_prefix,
                "timestamp": datetime.now().isoformat(),
                "status": "started",
                # Model architecture
                "model_size": model_config.total_params,
                "d_model": model_config.d_model,
                "n_layers": model_config.n_layers,
                "n_heads": model_config.n_heads,
                "d_ff": model_config.d_ff,
                "vocab_size": model_config.vocab_size,
                "max_seq_len": model_config.max_seq_len,
                # Training config
                "learning_rate": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                # Determinism config
                "seed": seed,
                "deterministic": True,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            }

            # Save locally
            os.makedirs("training_metrics", exist_ok=True)
            with open(f"training_metrics/{checkpoint_prefix}_initial.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Save to shared NVMe
            shared_metrics_dir = os.path.join(shared_dir, "training_metrics")
            os.makedirs(shared_metrics_dir, exist_ok=True)
            shared_path = os.path.join(
                shared_metrics_dir, f"{checkpoint_prefix}_initial.json"
            )
            tmp_path = shared_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(metadata, f, indent=2)
            os.replace(tmp_path, shared_path)

            print(f"[Rank 0] Initial metadata saved: {checkpoint_prefix}_initial.json")
        except Exception as e:
            print(f"Initial metadata save failed: {e}")

    def _save_intermediate_metrics(
        self, checkpoint_prefix: str, metrics: dict, global_step: int
    ):
        """Save intermediate metrics backup (overwrites same file for crash recovery)."""
        try:
            os.makedirs("training_metrics", exist_ok=True)
            backup_path = f"training_metrics/{checkpoint_prefix}_intermediate.npz"

            data_to_save = {k: np.array(v) for k, v in metrics.items() if v}
            data_to_save["last_step"] = np.array(global_step)
            data_to_save["backup_timestamp"] = np.array(
                datetime.now().isoformat(), dtype=object
            )

            # np.savez auto-appends .npz, so use a base name without extension
            tmp_base = f"training_metrics/{checkpoint_prefix}_intermediate_tmp"
            np.savez(tmp_base, **data_to_save)  # Creates tmp_base + '.npz'
            os.replace(tmp_base + ".npz", backup_path)

            print(
                f"[Rank 0] Intermediate metrics saved at step {global_step} ({len(metrics['batch_losses'])} batches)"
            )
        except Exception as e:
            print(f"[Rank 0] Intermediate metrics save failed: {e}")

    def _build_metadata(
        self,
        checkpoint_prefix,
        metrics,
        model_config,
        lr,
        epochs,
        total_tokens,
        cumulative_time,
        batch_size,
        weight_decay,
        seed=DEFAULT_SEED,
    ):
        """Build metadata dictionary for JSON export."""
        train_losses = metrics.get("epoch_train_losses", [])
        val_losses = metrics.get("val_losses", [])
        tokens_per_batch = metrics.get("tokens_per_batch", [])

        return {
            "checkpoint_prefix": checkpoint_prefix,
            "timestamp": datetime.now().isoformat(),
            # Model architecture
            "model_size": model_config.total_params,
            "d_model": model_config.d_model,
            "n_layers": model_config.n_layers,
            "n_heads": model_config.n_heads,
            "d_ff": model_config.d_ff,
            "vocab_size": model_config.vocab_size,
            "max_seq_len": model_config.max_seq_len,
            # Training config
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "global_batch_tokens": (
                int(np.mean(tokens_per_batch)) if tokens_per_batch else None
            ),
            # Determinism config
            "seed": seed,
            "deterministic": True,
            # Training progress
            "total_steps": len(metrics.get("batch_losses", [])),
            "total_tokens": int(total_tokens),
            "cumulative_time_seconds": cumulative_time,
            # Train loss stats
            "final_train_loss": float(train_losses[-1]) if train_losses else None,
            "mean_train_loss": float(np.mean(train_losses)) if train_losses else None,
            "std_train_loss": float(np.std(train_losses)) if train_losses else None,
            # Val loss stats
            "final_val_loss": float(val_losses[-1]) if val_losses else None,
            "best_val_loss": float(np.min(val_losses)) if val_losses else None,
            "mean_val_loss": float(np.mean(val_losses)) if val_losses else None,
            "val_epochs_count": len(val_losses),
        }

    def _train_model_distributed(
        self,
        model: DDP,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        device: torch.device,
        epochs: int,
        lr: float,
        checkpoint_prefix: str,
        model_config: ModelConfig,
        rank: int,
        world_size: int,
        log_interval: int,
        val_tokens_limit: int,
        start_epoch: int = 0,
        resume_checkpoint: str | None = None,
        initial_tokens: int = 0,
        prior_metrics: dict | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        shared_dir: str = SHARED_NVME_DIR,
        keep_checkpoints: int = DEFAULT_KEEP_CHECKPOINTS,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        seed: int = DEFAULT_SEED,
        use_bf16: bool = False,
    ) -> tuple[float, list[float], dict]:
        # make sure weights and optimizer states are being resumed
        num_batches = len(train_loader)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        total_steps = epochs * num_batches
        warmup_steps = max(1, int(total_steps * DEFAULT_WARMUP_RATIO))

        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=DEFAULT_WARMUP_START_FACTOR,
            end_factor=DEFAULT_WARMUP_END_FACTOR,
            total_iters=warmup_steps,
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=lr * DEFAULT_COSINE_ANNEALING_MIN_RATIO,
        )

        # Load optimizer/scheduler state if resuming
        if resume_checkpoint is not None:
            checkpoint = torch.load(
                resume_checkpoint, map_location="cpu", weights_only=False
            )
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if rank == 0:
                    print_info("Restored optimizer state")
            if "warmup_scheduler_state_dict" in checkpoint:
                warmup_scheduler.load_state_dict(
                    checkpoint["warmup_scheduler_state_dict"]
                )
            if "cosine_scheduler_state_dict" in checkpoint:
                cosine_scheduler.load_state_dict(
                    checkpoint["cosine_scheduler_state_dict"]
                )
        if use_bf16:
            if rank == 0:
                print_info("BF16 training enabled (no GradScaler needed)")

        if rank == 0:
            if prior_metrics is not None:
                metrics = {
                    "batch_losses": list(prior_metrics.get("batch_losses", [])),
                    "learning_rates": list(prior_metrics.get("learning_rates", [])),
                    "batch_times": list(prior_metrics.get("batch_times", [])),
                    "epoch_times": list(prior_metrics.get("epoch_times", [])),
                    "steps": list(prior_metrics.get("steps", [])),
                    "tokens_per_batch": list(prior_metrics.get("tokens_per_batch", [])),
                    "cumulative_tokens": list(
                        prior_metrics.get("cumulative_tokens", [])
                    ),
                    "epoch_train_losses": list(
                        prior_metrics.get("epoch_train_losses", [])
                    ),
                    "val_losses": list(prior_metrics.get("val_losses", [])),
                }
                print_info(
                    f"Initialized metrics with {len(metrics['batch_losses'])} prior batch records"
                )
            else:
                metrics = {
                    "batch_losses": [],
                    "learning_rates": [],
                    "batch_times": [],
                    "epoch_times": [],
                    "steps": [],
                    "tokens_per_batch": [],
                    "cumulative_tokens": [],
                    "epoch_train_losses": [],
                    "val_losses": [],
                }
        else:
            metrics = {}

        # Calculate starting step from prior metrics or estimate
        if prior_metrics and prior_metrics.get("steps"):
            global_step = int(prior_metrics["steps"][-1]) + 1
        else:
            global_step = start_epoch * num_batches

        current_step = global_step

        total_tokens_seen = initial_tokens // world_size if initial_tokens else 0
        cumulative_train_time = (
            sum(prior_metrics.get("epoch_times", [])) if prior_metrics else 0.0
        )

        if rank == 0:
            print_info(
                f"Starting from global_step={global_step}, tokens_seen={total_tokens_seen * world_size:,}"
            )
            print("\n" * 2)
        if rank == 0 and checkpoint_prefix:
            self._save_initial_metadata(
                checkpoint_prefix=checkpoint_prefix,
                model_config=model_config,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                weight_decay=weight_decay,
                seed=seed,
                shared_dir=shared_dir,
            )

        for epoch in range(start_epoch, epochs):
            model.train()
            # ================================================================
            # DETERMINISM: set_epoch() is CRITICAL for DDP reproducibility
            # This ensures shuffling is deterministic yet different each epoch
            # The shuffle order is determined by: seed + epoch
            # ================================================================
            train_loader.sampler.set_epoch(epoch)
            total_loss = 0
            num_batches = 0
            epoch_start = time.time()

            desc = f"Train e{epoch+1}/{epochs} [R{rank}]"
            train_pbar = tqdm(train_loader, desc=desc, leave=False, disable=(rank != 0))

            for batch in train_pbar:
                t0 = time.time()
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)

                batch_tokens = attention_mask.sum().item()
                total_tokens_seen += batch_tokens

                labels = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1].contiguous()
                attention_mask = attention_mask[:, :-1].contiguous()

                optimizer.zero_grad()

                # ================================================================
                # BF16: model is already .bfloat16(), no autocast or scaler needed.
                # BF16 has the same exponent range as FP32 (8 exponent bits) so
                # gradients don't underflow - no GradScaler required.
                # Logits are cast to FP32 for the loss to keep log-softmax
                # numerically stable.
                # ================================================================
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(
                    outputs.float().view(-1, outputs.size(-1)), labels.view(-1)
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if current_step < warmup_steps:
                    warmup_scheduler.step()
                    current_lr = warmup_scheduler.get_last_lr()[0]
                else:
                    if current_step == warmup_steps:
                        cosine_scheduler.base_lrs = [
                            g["lr"] for g in optimizer.param_groups
                        ]
                    cosine_scheduler.step()
                    current_lr = cosine_scheduler.get_last_lr()[0]
                current_step += 1

                batch_loss = loss.item()
                total_loss += batch_loss
                num_batches += 1

                if rank == 0:
                    metrics["batch_losses"].append(batch_loss)
                    metrics["learning_rates"].append(current_lr)
                    metrics["steps"].append(global_step)
                    metrics["batch_times"].append(time.time() - t0)
                    metrics["tokens_per_batch"].append(batch_tokens * world_size)
                    metrics["cumulative_tokens"].append(total_tokens_seen * world_size)

                    train_pbar.set_postfix(
                        {
                            "loss": f"{batch_loss:.4f}",
                            "lr": f"{current_lr:.2e}",
                            "tok": f"{(total_tokens_seen * world_size)/1e6:.1f}M",
                        }
                    )

                    # if writer and global_step % log_interval == 0:
                    #    writer.add_scalar('Loss/Train_Batch', batch_loss, global_step)
                    #    writer.add_scalar('LR', current_lr, global_step)
                    #    writer.add_scalar('Tokens/Cumulative_M', (total_tokens_seen * world_size) / 1e6, global_step)

                global_step += 1
            epoch_time = time.time() - epoch_start
            cumulative_train_time += epoch_time

            avg_train_loss = total_loss / max(num_batches, 1)
            loss_tensor = torch.tensor(avg_train_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            global_train_loss = loss_tensor.item() / world_size

            if rank == 0:
                metrics["epoch_train_losses"].append(global_train_loss)
                metrics["epoch_times"].append(epoch_time)
                # if writer:
                #    writer.add_scalar('Loss/Train_Epoch', global_train_loss, epoch)

            global_val_loss = None
            if val_loader is not None:
                global_val_loss = self._run_validation(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    criterion=criterion,
                    val_tokens_limit=val_tokens_limit,
                    rank=rank,
                    world_size=world_size,
                    epoch=epoch,
                    total_epochs=epochs,
                    max_seq_len=model_config.max_seq_len,
                    use_bf16=use_bf16,
                )
                if rank == 0:
                    metrics["val_losses"].append(global_val_loss)
                    # if writer:
                    #    writer.add_scalar('Loss/Val_Epoch', global_val_loss, epoch)

            if rank == 0:
                val_str = (
                    f"Val Loss: {global_val_loss:.4f}"
                    if global_val_loss is not None
                    else "Val: skipped"
                )
                print(
                    f"[Epoch {epoch+1}] Train Loss: {global_train_loss:.4f} | {val_str} | "
                    f"Tokens: {(total_tokens_seen*world_size)/1e6:.1f}M"
                )

                if checkpoint_prefix:
                    # Save to local directory
                    print("[Rank 0] Saving local checkpoint...")
                    self._save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        ws=warmup_scheduler,
                        cs=cosine_scheduler,
                        epoch=epoch,
                        t_loss=global_train_loss,
                        v_loss=global_val_loss,
                        tokens=total_tokens_seen * world_size,
                        prefix=checkpoint_prefix,
                        config=model_config,
                        current_step=current_step,
                        cumulative_time=cumulative_train_time,
                        batch_size=batch_size,
                        keep_last_n=keep_checkpoints,
                        weight_decay=weight_decay,
                        seed=seed,
                    )
                    print("[Rank 0] Saving shared NVMe checkpoint...")
                    # Save to shared NVMe directory
                    self._save_checkpoint_shared(
                        model=model,
                        optimizer=optimizer,
                        ws=warmup_scheduler,
                        cs=cosine_scheduler,
                        epoch=epoch,
                        t_loss=global_train_loss,
                        v_loss=global_val_loss,
                        tokens=total_tokens_seen * world_size,
                        prefix=checkpoint_prefix,
                        config=model_config,
                        current_step=current_step,
                        cumulative_time=cumulative_train_time,
                        batch_size=batch_size,
                        shared_dir=shared_dir,
                        keep_last_n=keep_checkpoints,
                        weight_decay=weight_decay,
                        seed=seed,
                    )
                    print("[Rank 0] Saving intermediate metrics...")
                    self._save_intermediate_metrics(
                        checkpoint_prefix, metrics, global_step
                    )
                    print("[Rank 0] All saves complete, hitting barrier...")
            dist.barrier()
        # if writer: writer.close()

        if checkpoint_prefix and rank == 0:
            # Save final metrics to local directory
            self._save_metrics(
                checkpoint_prefix=checkpoint_prefix,
                metrics=metrics,
                model_config=model_config,
                lr=lr,
                epochs=epochs,
                total_tokens=total_tokens_seen * world_size,
                cumulative_time=cumulative_train_time,
                batch_size=batch_size,
                weight_decay=weight_decay,
                seed=seed,
            )
            # Save final metrics to shared NVMe directory
            self._save_metrics_shared(
                checkpoint_prefix=checkpoint_prefix,
                metrics=metrics,
                model_config=model_config,
                lr=lr,
                epochs=epochs,
                total_tokens=total_tokens_seen * world_size,
                cumulative_time=cumulative_train_time,
                batch_size=batch_size,
                shared_dir=shared_dir,
                weight_decay=weight_decay,
                seed=seed,
            )

        if rank == 0:
            last_train_loss = float(metrics["epoch_train_losses"][-1])
            epoch_train_losses = list(metrics["epoch_train_losses"])
            metrics_dict = dict(metrics)
        else:
            last_train_loss = float("inf")
            epoch_train_losses = []
            metrics_dict = {}

        return last_train_loss, epoch_train_losses, metrics_dict

    def _run_validation(
        self,
        model: DDP,
        val_loader: DataLoader,
        device: torch.device,
        criterion: nn.Module,
        val_tokens_limit: int,
        rank: int,
        world_size: int,
        epoch: int,
        total_epochs: int,
        max_seq_len: int,
        use_bf16: bool = False,
    ):
        """Runs validation with a token limit across DDP."""
        model.eval()
        val_loss_sum = 0
        val_batches = 0
        local_tokens_seen = 0
        token_limit = val_tokens_limit
        per_gpu_token_limit = token_limit / world_size
        batch_size = val_loader.batch_size
        estimated_tokens_per_batch = batch_size * max_seq_len
        target_batches = int(per_gpu_token_limit / estimated_tokens_per_batch)
        total_batches = min(target_batches, len(val_loader))
        desc = f"Val e{epoch+1}/{total_epochs} [R{rank}]"
        val_pbar = tqdm(
            val_loader, desc=desc, total=total_batches, leave=False, disable=(rank != 0)
        )

        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                batch_tokens = attention_mask.sum().item()
                local_tokens_seen += batch_tokens

                labels = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1].contiguous()
                attention_mask = attention_mask[:, :-1].contiguous()

                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(
                    outputs.float().view(-1, outputs.size(-1)), labels.view(-1)
                )

                val_loss_sum += loss.item()
                val_batches += 1

                val_pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

                if local_tokens_seen >= per_gpu_token_limit:
                    break

        stats = torch.tensor(
            [val_loss_sum, val_batches], dtype=torch.float32, device=device
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        total_loss_sum = stats[0].item()
        total_batches_processed = stats[1].item()

        return total_loss_sum / max(total_batches_processed, 1)

    def _validate_checkpoint(self, path):
        try:
            # 1. Check if file exists and has size > 0
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                print(f"Checkpoint {path} is empty or missing!")
                return False

            # 2. Lightweight map_location='cpu' with mmap (avoids full read)
            # We only check if the file header is readable, not every weight
            try:
                # excessive I/O fix: just load the structure, not the data
                torch.load(path, map_location="cpu", mmap=True)
            except Exception as e:
                print(f"Header check failed for {path}: {e}")
                return False

            return True
        except Exception as e:
            print(f"Validation error: {e}")
            return False

    def _cleanup_old_checkpoints(
        self, directory: str, prefix: str, keep_last_n: int = 2
    ):
        """Remove old checkpoints, keeping only the last N for this run prefix."""
        pattern = os.path.join(directory, f"{prefix}_epoch*.pt")
        checkpoints = glob.glob(pattern)

        if len(checkpoints) <= keep_last_n:
            return

        # Sort by epoch number
        def extract_epoch(path):
            match = re.search(r"_epoch(\d+)", os.path.basename(path))
            return int(match.group(1)) if match else 0

        checkpoints_sorted = sorted(checkpoints, key=extract_epoch)
        to_delete = checkpoints_sorted[:-keep_last_n]

        for old_ckpt in to_delete:
            try:
                os.remove(old_ckpt)
                print(f"[Rank 0] Removed old checkpoint: {os.path.basename(old_ckpt)}")
            except Exception as e:
                print(f"[Rank 0] Warning: couldn't delete {old_ckpt}: {e}")

    def _save_checkpoint(
        self,
        model,
        optimizer,
        ws,
        cs,
        epoch,
        t_loss,
        v_loss,
        tokens,
        prefix,
        config,
        weight_decay,
        current_step=None,
        cumulative_time=0.0,
        batch_size=128,
        keep_last_n: int = 2,
        seed: int = DEFAULT_SEED,
    ):
        """Save checkpoint to local directory with atomic write and cleanup."""
        tmp_path = None
        try:
            os.makedirs("saved_checkpoints", exist_ok=True)
            time_hours = cumulative_time / 3600
            filename = f"{prefix}_epoch{epoch+1}_bs{batch_size}_seq{config.max_seq_len}_wd{weight_decay}_time{time_hours:.2f}h_completedAt{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            final_path = os.path.join("saved_checkpoints", filename)
            tmp_path = final_path + ".tmp"

            checkpoint_data = {
                "epoch": epoch + 1,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "warmup_scheduler_state_dict": ws.state_dict(),
                "cosine_scheduler_state_dict": cs.state_dict(),
                "train_loss": t_loss,
                "val_loss": v_loss,
                "total_tokens": tokens,
                "current_step": current_step,
                "model_config": config.__dict__,
                "cumulative_time": cumulative_time,
                "batch_size": batch_size,
                "timestamp": datetime.now().isoformat(),
                "weight_decay": weight_decay,
                "seed": seed,  # Save seed for reproducibility verification
            }

            # Atomic write: save to temp, validate, then rename
            torch.save(checkpoint_data, tmp_path)

            if not self._validate_checkpoint(tmp_path):
                print(
                    "[Rank 0] ERROR: Checkpoint validation failed, not replacing old checkpoint"
                )
                os.remove(tmp_path)
                return

            os.replace(tmp_path, final_path)
            print(f"[Rank 0] Checkpoint saved (local): {final_path}")

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints("saved_checkpoints", prefix, keep_last_n)

        except Exception as e:
            print(f"[Rank 0] Local checkpoint save failed: {e}")
            # Clean up temp file if it exists
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except BaseException:
                    pass

    def _save_checkpoint_shared(
        self,
        model,
        optimizer,
        ws,
        cs,
        epoch,
        t_loss,
        v_loss,
        tokens,
        prefix,
        config,
        weight_decay,
        current_step=None,
        cumulative_time=0.0,
        batch_size=128,
        shared_dir: str = SHARED_NVME_DIR,
        keep_last_n: int = 2,
        seed: int = DEFAULT_SEED,
    ):
        """Save checkpoint to shared NVMe directory with atomic write and cleanup."""
        tmp_path = None
        try:
            time_hours = cumulative_time / 3600
            filename = f"{prefix}_epoch{epoch+1}_bs{batch_size}_seq{config.max_seq_len}_wd{weight_decay}_time{time_hours:.2f}h_completedAt{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"

            shared_ckpt_dir = os.path.join(shared_dir, "saved_checkpoints")
            os.makedirs(shared_ckpt_dir, exist_ok=True)
            final_path = os.path.join(shared_ckpt_dir, filename)
            tmp_path = final_path + ".tmp"

            checkpoint_data = {
                "epoch": epoch + 1,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "warmup_scheduler_state_dict": ws.state_dict(),
                "cosine_scheduler_state_dict": cs.state_dict(),
                "train_loss": t_loss,
                "val_loss": v_loss,
                "total_tokens": tokens,
                "current_step": current_step,
                "model_config": config.__dict__,
                "cumulative_time": cumulative_time,
                "batch_size": batch_size,
                "timestamp": datetime.now().isoformat(),
                "weight_decay": weight_decay,
                "seed": seed,  # Save seed for reproducibility verification
            }

            print(f"[Rank 0] debug: Starting shared save to {tmp_path}...", flush=True)
            _t0 = time.time()

            # Atomic write: save to temp, validate, then rename
            torch.save(checkpoint_data, tmp_path)

            t1 = time.time()
            if self._validate_checkpoint(tmp_path):
                print(
                    f"[Rank 0] debug: Validation took {time.time() - t1:.2f}s",
                    flush=True,
                )
            else:
                print("[Rank 0] debug: Validation FAILED", flush=True)

            t2 = time.time()
            os.replace(tmp_path, final_path)
            print(f"[Rank 0] debug: Rename took {time.time() - t2:.2f}s", flush=True)

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints(shared_ckpt_dir, prefix, keep_last_n)

        except Exception as e:
            print(f"[Rank 0] Shared checkpoint save failed: {e}")
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except BaseException:
                    pass

    def _save_metrics(
        self,
        checkpoint_prefix,
        metrics,
        model_config,
        lr,
        epochs,
        total_tokens,
        weight_decay,
        cumulative_time=0.0,
        batch_size=128,
        seed=DEFAULT_SEED,
    ):
        """Save final training metrics to local timestamped NPZ and JSON."""
        os.makedirs("training_metrics", exist_ok=True)

        data_to_save = {k: np.array(v) for k, v in metrics.items()}

        time_hours = cumulative_time / 3600
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{checkpoint_prefix}_bs{batch_size}_seq{model_config.max_seq_len}_wd{weight_decay}_time{time_hours:.2f}h_{timestamp}"

        np.savez(f"training_metrics/{filename_base}_metrics.npz", **data_to_save)

        metadata = self._build_metadata(
            checkpoint_prefix,
            metrics,
            model_config,
            lr,
            epochs,
            total_tokens,
            cumulative_time,
            batch_size,
            weight_decay,
            seed,
        )

        with open(f"training_metrics/{filename_base}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(
            f"[Rank 0] Final metrics saved (local): {len(metrics['batch_losses'])} batch records"
        )

    def _save_metrics_shared(
        self,
        checkpoint_prefix,
        metrics,
        model_config,
        lr,
        epochs,
        total_tokens,
        weight_decay,
        cumulative_time=0.0,
        batch_size=128,
        shared_dir: str = SHARED_NVME_DIR,
        seed=DEFAULT_SEED,
    ):
        """Save final training metrics to shared NVMe directory only."""
        try:
            time_hours = cumulative_time / 3600
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"{checkpoint_prefix}_bs{batch_size}_seq{model_config.max_seq_len}_wd{weight_decay}_time{time_hours:.2f}h_{timestamp}"

            shared_metrics_dir = os.path.join(shared_dir, "training_metrics")
            os.makedirs(shared_metrics_dir, exist_ok=True)

            data_to_save = {k: np.array(v) for k, v in metrics.items()}

            metadata = self._build_metadata(
                checkpoint_prefix,
                metrics,
                model_config,
                lr,
                epochs,
                total_tokens,
                cumulative_time,
                batch_size,
                weight_decay,
                seed,
            )

            # Final paths
            npz_path = os.path.join(shared_metrics_dir, f"{filename_base}_metrics.npz")
            json_path = os.path.join(
                shared_metrics_dir, f"{filename_base}_metadata.json"
            )

            # np.savez auto-appends .npz, so use base name without extension
            tmp_npz_base = os.path.join(
                shared_metrics_dir, f"{filename_base}_metrics_tmp"
            )
            np.savez(tmp_npz_base, **data_to_save)
            os.replace(tmp_npz_base + ".npz", npz_path)

            tmp_json = json_path + ".tmp"
            with open(tmp_json, "w") as f:
                json.dump(metadata, f, indent=2)
            os.replace(tmp_json, json_path)

            print(f"[Rank 0] Metrics saved (shared NVMe): {npz_path}")

        except Exception as e:
            print(f"Shared metrics save failed: {e}")
