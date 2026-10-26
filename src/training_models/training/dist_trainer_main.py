import argparse
import sys
import os
from typing import Optional
import torch
import torch.distributed as dist
import numpy as np
import tiktoken
import glob
import re
from torch.utils.data import Subset

# Make ``training_models`` importable when this file is run by path (src/ on sys.path).
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


from training_models.training import DistributedTrainer
from training_models.core import generate_model_configs, PretokenizedDataset
from training_models.utils import print_info, print_success, print_warning, print_section

# Optional shared directory for backup saves (falls back to DATA_DIR, then the cwd)
SHARED_NVME_DIR = os.environ.get("SHARED_DIR", os.environ.get("DATA_DIR", "."))


DEFAULT_BATCH_SIZE = 128
DEFAULT_DATASET_NAME = "cosmopedia"
DEFAULT_ARCHITECTURE = "llama"
DEFAULT_MAX_SEQ_LEN = 2048
DEFAULT_VAL_TOKENS_LIMIT = 10_000_000
DEFAULT_LOG_INTERVAL = 10
DEFAULT_WEIGHT_DECAY = 0.3
DEFAULT_KEEP_CHECKPOINTS = 2
DEFAULT_TOKENS = 100000
DEFAULT_SKIP_VAL = False
DEFAULT_KEEP_CHECKPOINTS = 2
DEFAULT_WEIGHT_DECAY = 0.3


def find_metrics_for_checkpoint(
    checkpoint_path: str, metrics_dir: str = "training_metrics"
) -> str | None:
    """Find the metrics NPZ file corresponding to a checkpoint."""
    checkpoint_name = os.path.basename(checkpoint_path)

    # Extract prefix (everything before _epochN...)
    match = re.match(
        r"(.+)_epoch\d+(?:_bs\d+_seq\d+_time[\d.]+h)?(?:_completedAt\d{8}_\d{6})?\.pt",
        checkpoint_name,
    )
    if not match:
        return None

    prefix = match.group(1)

    # Look for intermediate metrics first (most recent state)
    intermediate_path = os.path.join(metrics_dir, f"{prefix}_intermediate.npz")
    if os.path.exists(intermediate_path):
        return intermediate_path

    # Look for timestamped metrics files, get the most recent
    pattern = os.path.join(metrics_dir, f"{prefix}_*_metrics.npz")
    matches = glob.glob(pattern)
    if matches:
        return max(matches, key=os.path.getmtime)

    return None


def load_metrics_from_npz(npz_path: str) -> dict:
    """Load training metrics from NPZ file."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Metrics file not found: {npz_path}")

    print_info(f"Loading metrics from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    metrics = {
        "batch_losses": data["batch_losses"].tolist() if "batch_losses" in data else [],
        "learning_rates": (
            data["learning_rates"].tolist() if "learning_rates" in data else []
        ),
        "batch_times": data["batch_times"].tolist() if "batch_times" in data else [],
        "epoch_times": data["epoch_times"].tolist() if "epoch_times" in data else [],
        "steps": data["steps"].tolist() if "steps" in data else [],
        "tokens_per_batch": (
            data["tokens_per_batch"].tolist() if "tokens_per_batch" in data else []
        ),
        "cumulative_tokens": (
            data["cumulative_tokens"].tolist() if "cumulative_tokens" in data else []
        ),
        "epoch_train_losses": (
            data["epoch_train_losses"].tolist() if "epoch_train_losses" in data else []
        ),
        "val_losses": data["val_losses"].tolist() if "val_losses" in data else [],
    }

    print_info(
        f"  Loaded {len(metrics['batch_losses'])} batch records, {len(metrics['epoch_train_losses'])} epochs"
    )
    return metrics


def resume_distributed_training(
    args: argparse.Namespace, rank: int
) -> tuple[Optional[str], Optional[str], Optional[dict]]:
    resume_checkpoint_path = None
    resume_metrics_path = None
    prior_metrics = None
    if args.resume:
        resume_checkpoint_path = args.resume
        if rank == 0:
            print_section("Resuming from Checkpoint")
            print_info(f"Checkpoint: {args.resume}")
            print_info(
                f"Using CLI args: model_size={args.model_size}, lr={args.learning_rate}, batch_size={args.batch_size}"
            )
        # Load metrics NPZ
        if args.resume_metrics:
            resume_metrics_path = args.resume_metrics
        else:
            resume_metrics_path = find_metrics_for_checkpoint(args.resume)
        if resume_metrics_path and rank == 0:
            try:
                prior_metrics = load_metrics_from_npz(resume_metrics_path)
            except Exception as e:
                print_warning(f"Could not load metrics: {e}")
                prior_metrics = None
        elif rank == 0:
            print_warning("No metrics file found, starting metrics fresh")
    return resume_checkpoint_path, resume_metrics_path, prior_metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Distributed LLM Training")
    parser.add_argument("--model-size", type=int)
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Custom prefix for checkpoint and metrics files",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_NAME,
        choices=[
            "cosmopedia",
            "wikipedia",
            "pes2o",
            "redpajama",
            "c4"
        ],
    )
    parser.add_argument("--architecture", type=str, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--val-tokens", type=int, help="Tokens for validation")
    parser.add_argument(
        "--skip-val", action="store_true", help="Skip validation during training"
    )
    parser.add_argument(
        "--use-train-as-val",
        action="store_true",
        help="Use validation set as training data (for holdout training)",
    )
    parser.add_argument(
        "--shared-dir",
        type=str,
        default=SHARED_NVME_DIR,
        help="Shared NVMe directory for backup saves",
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=DEFAULT_KEEP_CHECKPOINTS,
        help="Number of checkpoints to keep per run",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="Weight decay for AdamW optimizer",
    )

    # Resume from checkpoint
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    parser.add_argument(
        "--resume-metrics",
        type=str,
        default=None,
        help="Path to metrics NPZ file (auto-detected from checkpoint if not specified)",
    )
    parser.add_argument(
        "--use-bf16",
        action="store_true",
        default=False,
        help="Enable BF16 training (model.bfloat16(), no GradScaler needed)",
    )

    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    # Handle checkpoint resumption - only load weights/optimizer/metrics, not config
    resume_checkpoint_path = None
    resume_metrics_path = None
    prior_metrics = None

    resume_checkpoint_path, resume_metrics_path, prior_metrics = None, None, None

    if args.resume:
        resume_checkpoint_path = args.resume

        if rank == 0:
            print_section("Resuming from Checkpoint")
            print_info(f"Checkpoint: {args.resume}")
            print_info(
                f"Using CLI args: model_size={args.model_size}, lr={args.learning_rate}, batch_size={args.batch_size}"
            )

        # Load metrics NPZ
        if args.resume_metrics:
            resume_metrics_path = args.resume_metrics
        else:
            resume_metrics_path = find_metrics_for_checkpoint(args.resume)

        if resume_metrics_path and rank == 0:
            try:
                prior_metrics = load_metrics_from_npz(resume_metrics_path)
            except Exception as e:
                print_warning(f"Could not load metrics: {e}")
                prior_metrics = None
        elif rank == 0:
            print_warning("No metrics file found, starting metrics fresh")

    # print output on first rank only
    if rank == 0:
        print_section("Distributed LLM Training")
        print_info(f"Dataset: {args.dataset}")
        print_info(f"Architecture: {args.architecture}")
        print_info(f"Tokens: {args.tokens}")
        print_info(f"Precision: {'BF16' if args.use_bf16 else 'FP32'}")
        print_info(f"Skip Validation: {args.skip_val}")
        print_info(f"Use Train as Val (holdout): {args.use_train_as_val}")
        if not args.skip_val and args.val_tokens is not None:
            print_info(f"Val Token Limit: {args.val_tokens/1e6:.1f}M")
        print_info(f"Model sizes: {args.model_size}")
        print_info(f"Learning rates: {args.learning_rate}")
        print_info(f"Batch size: {args.batch_size}")
        print_info(f"Epochs: {args.epochs}")
        print_info(f"Shared NVMe dir: {args.shared_dir}")
        print_info(f"Keep checkpoints: {args.keep_checkpoints}")
        if args.output_prefix:
            print_info(f"Output prefix: {args.output_prefix}")

    train_data = None
    dataset_name = args.dataset.lower()
    tokenizer = tiktoken.get_encoding("cl100k_base")

    model_size = args.model_size
    lr = args.learning_rate

    vocab_size = tokenizer.n_vocab

    config = generate_model_configs(
        num_configs=1,
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        min_params=100,
        max_params=1_000_000_000,
        target_params=[model_size],
    )[0]

    trainer = DistributedTrainer()

    train_data, val_data = trainer.load_dataset(
        dataset_name=dataset_name, max_seq_len=args.max_seq_len, skip_val=args.skip_val
    )

    # If --use-train-as-val is set, swap: use val_data as our training set
    if args.use_train_as_val:
        if val_data is None:
            raise ValueError(
                "Cannot use --use-train-as-val: validation set is not available. "
                "Make sure --skip-val is not set when loading the dataset."
            )
        if rank == 0:
            print_info("Using validation set as training data (holdout mode)")
        train_data = val_data
        val_data = None  # No validation when training on holdout

    if train_data is None:
        raise ValueError(f"Failed to load dataset: {dataset_name}")

    if dataset_name in ["cosmopedia", "wikipedia", "pes2o", "redpajama", "c4"]:
        total_tokens_target = int(args.tokens)

        # Use actual cumulative token counts instead of average estimate
        est_examples = 0
        if rank == 0:
            if isinstance(train_data, PretokenizedDataset):
                lengths = train_data.lengths_data
            elif isinstance(train_data, Subset) and isinstance(
                train_data.dataset, PretokenizedDataset
            ):
                lengths = train_data.dataset.lengths_data[train_data.indices]
            else:
                lengths = None

            if lengths is not None:
                cumsum = np.cumsum(lengths)
                est_examples = (
                    int(np.searchsorted(cumsum, total_tokens_target, side="right")) + 1
                )
                est_examples = min(est_examples, len(train_data))
                actual_tokens = int(cumsum[est_examples - 1])
                print_info(
                    f"Subset: {est_examples:,} examples → {actual_tokens:,} tokens (target: {total_tokens_target:,})"
                )
            else:
                # Fallback to average estimate for non-pretokenized datasets
                avg_tokens = trainer._estimate_average_tokens_per_example(
                    data=train_data, tokenizer=tokenizer, max_seq_len=args.max_seq_len
                )
                est_examples = max(
                    1, int(total_tokens_target // max(int(avg_tokens), 1))
                )
                est_examples = min(est_examples, len(train_data))

        if world_size > 1:
            t = torch.tensor(est_examples, dtype=torch.long, device=f"cuda:{rank}")
            dist.broadcast(t, src=0)
            est_examples = t.item()

        if hasattr(train_data, "select"):
            train_subset = train_data.select(range(est_examples))
        else:
            train_subset = Subset(train_data, range(est_examples))

    # use the distributed trainer to train the model

    results = trainer.train_model(
        model_config=config,
        train_data=train_subset,
        epochs=args.epochs,
        lr=lr,
        batch_size=args.batch_size,
        tokens=args.tokens,
        dataset=args.dataset,
        max_seq_len=args.max_seq_len,
        log_interval=10,
        architecture=args.architecture,
        val_tokens_limit=args.val_tokens,
        skip_val=args.skip_val
        or args.use_train_as_val,  # Skip val if using holdout mode
        resume_checkpoint_path=resume_checkpoint_path,
        resume_metrics_path=resume_metrics_path,
        prior_metrics=prior_metrics,
        shared_dir=args.shared_dir,
        keep_checkpoints=args.keep_checkpoints,
        weight_decay=args.weight_decay,
        output_prefix=args.output_prefix,  # Pass the custom prefix
        use_bf16=args.use_bf16,
    )

    if rank == 0:
        val_str = (
            f"Val: {results['final_val_loss']:.4f}"
            if results["final_val_loss"] is not None
            else "Val: skipped"
        )
        print_success(
            f"Training Complete. Final Train: {results['final_train_loss']:.4f}, {val_str}"
        )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
