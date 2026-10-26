import numpy as np
import tiktoken
from tqdm import tqdm
from multiprocessing import Pool
from training_models.models import ModelConfig

# ============================================================================
# Published architectures: target → exact (d_model, n_layers, n_heads, d_ff)
# ----------------------------------------------------------------------------
# These are the exact 14 architectures used to train the published HuggingFace
# reference checkpoints. The mapping accepts either form of target:
#
#   - Round target (what the original training scripts passed):
#       e.g. --model-size 10000000
#   - True parameter count (what HF filenames use as `m{N}`):
#       e.g. --model-size 10070160
#
# Both resolve to the same architecture. The chosen architecture's true
# parameter count (computed from ModelConfig.total_params) becomes the N
# labeled in checkpoint filenames and reported in (N, D) data.
#
# Verified by direct inspection of all 1,933 final-epoch checkpoints; see
# architecture_audit_e3.csv. Within each entry:
#   - The architecture is identical across all datasets, designs, and seeds
#     at this N (zero variance).
#   - true_total_params (from ModelConfig.total_params) matches state_dict
#     numel() exactly for every published checkpoint at this size.
# ============================================================================

# Round target → architecture spec
_ROUND_TARGET_TO_ARCHITECTURE: dict[int, dict] = {
    5_000_000:  dict(d_model=24,  n_layers=24, n_heads=12, d_ff=96),
    10_000_000: dict(d_model=48,  n_layers=12, n_heads=1,  d_ff=192),
    14_000_000: dict(d_model=64,  n_layers=24, n_heads=32, d_ff=256),
    19_000_000: dict(d_model=96,  n_layers=1,  n_heads=1,  d_ff=384),
    22_000_000: dict(d_model=96,  n_layers=24, n_heads=24, d_ff=384),
    28_000_000: dict(d_model=128, n_layers=12, n_heads=4,  d_ff=512),
    32_000_000: dict(d_model=160, n_layers=1,  n_heads=16, d_ff=640),
    35_000_000: dict(d_model=160, n_layers=8,  n_heads=4,  d_ff=640),
    40_000_000: dict(d_model=192, n_layers=3,  n_heads=12, d_ff=768),
    45_000_000: dict(d_model=192, n_layers=16, n_heads=1,  d_ff=768),
    # The four largest (held-out evaluation set) targets are the legacy buggy
    # values that were hardcoded into the original training scripts; they map
    # onto the correct true-count architectures listed below.
    54_566_912: dict(d_model=256, n_layers=4,  n_heads=32, d_ff=1024),
    60_884_992: dict(d_model=256, n_layers=12, n_heads=32, d_ff=1024),
    64_044_032: dict(d_model=256, n_layers=16, n_heads=32, d_ff=1024),
    70_362_112: dict(d_model=256, n_layers=24, n_heads=1,  d_ff=1024),
}

# True parameter count → architecture spec (HF filename `m{N}` → architecture)
_TRUE_N_TO_ARCHITECTURE: dict[int, dict] = {
    5_035_656:  dict(d_model=24,  n_layers=24, n_heads=12, d_ff=96),
    10_070_160: dict(d_model=48,  n_layers=12, n_heads=1,  d_ff=192),
    14_411_456: dict(d_model=64,  n_layers=24, n_heads=32, d_ff=256),
    19_400_928: dict(d_model=96,  n_layers=1,  n_heads=1,  d_ff=384),
    22_796_832: dict(d_model=96,  n_layers=24, n_heads=24, d_ff=384),
    28_819_840: dict(d_model=128, n_layers=12, n_heads=4,  d_ff=512),
    32_498_720: dict(d_model=160, n_layers=1,  n_heads=16, d_ff=640),
    35_368_160: dict(d_model=160, n_layers=8,  n_heads=4,  d_ff=640),
    40_277_184: dict(d_model=192, n_layers=3,  n_heads=12, d_ff=768),
    47_949_888: dict(d_model=192, n_layers=16, n_heads=1,  d_ff=768),
    55_538_432: dict(d_model=256, n_layers=4,  n_heads=32, d_ff=1024),
    63_931_136: dict(d_model=256, n_layers=12, n_heads=32, d_ff=1024),
    68_127_488: dict(d_model=256, n_layers=16, n_heads=32, d_ff=1024),
    76_520_192: dict(d_model=256, n_layers=24, n_heads=1,  d_ff=1024),
}

# Combined lookup: accepts either a round target or a true parameter count.
PUBLISHED_ARCHITECTURES: dict[int, dict] = {
    **_ROUND_TARGET_TO_ARCHITECTURE,
    **_TRUE_N_TO_ARCHITECTURE,
}


def _count_tokens_batch(args: tuple[list[str], str, int]) -> int:
    """Helper function to count tokens in a batch of texts.

    Args:
        args: Tuple of (texts, encoding_name, max_seq_len)

    Returns:
        Total token count for the batch
    """
    texts, encoding_name, max_seq_len = args
    tokenizer = tiktoken.get_encoding(encoding_name)
    total_tokens = 0
    for text in texts:
        tokens = tokenizer.encode(text)
        tokens_count = min(len(tokens), max_seq_len)
        total_tokens += tokens_count
    return total_tokens


def count_total_tokens(
    texts: list[str], tokenizer: tiktoken.Encoding, max_seq_len: int
) -> int:
    """Count total tokens in a dataset applying max_seq_len truncation per text.

    This function counts tokens exactly as they would be used in training,
    matching the logic in PretokenizedDataset.__getitem__ where tokens are
    truncated at max_seq_len per text. Uses multiprocessing to parallelize
    token counting across 10 worker processes.

    Args:
        texts: List of text strings to tokenize
        tokenizer: Tiktoken encoding to use for tokenization
        max_seq_len: Maximum sequence length to truncate tokens at

    Returns:
        Total number of tokens across all texts (with truncation applied per text)
    """
    num_processes = 10
    encoding_name = tokenizer.name

    # Split texts into batches for parallel processing
    batch_size = max(1, len(texts) // (num_processes * 4))  # 4 batches per process
    batches = []
    for i in range(0, len(texts), batch_size):
        batches.append(texts[i : i + batch_size])

    # Prepare arguments for each batch
    batch_args = [(batch, encoding_name, max_seq_len) for batch in batches]

    # Process batches in parallel with progress bar
    total_tokens = 0
    with Pool(processes=num_processes) as pool:
        results = []
        for result in tqdm(
            pool.imap(_count_tokens_batch, batch_args),
            total=len(batch_args),
            desc="Counting tokens",
            unit="batch",
        ):
            results.append(result)
        total_tokens = sum(results)

    return total_tokens


def generate_model_configs(
    num_configs: int,
    vocab_size: int,
    max_seq_len: int = 512,
    min_params: int = 100,
    max_params: int = 100_000_000,
    target_params: list[int] | None = None,
) -> list[ModelConfig]:
    """..."""

    # ---- Hot path: published architectures ----
    # When the caller asks for one of the 14 published targets, return the
    # exact architecture used to train the corresponding HF reference. This
    # guarantees bit-perfect architecture reproduction. ModelConfig.total_params
    # then reports the correct true count, which propagates to the checkpoint
    # filename's m{N} label and to (N, D) scaling-law data.
    if target_params is not None and len(target_params) > 0:
        selected = []
        for t in target_params:
            if t in PUBLISHED_ARCHITECTURES:
                spec = PUBLISHED_ARCHITECTURES[t]
                selected.append(ModelConfig(
                    vocab_size=vocab_size,
                    max_seq_len=max_seq_len,
                    **spec,
                ))
            else:
                selected.append(None)
        if all(c is not None for c in selected):
            return selected[:num_configs]
        # else fall through to search

    # ---- Original search code, as a backup.
    layer_counts = [1, 2, 3, 4, 6, 8, 12, 16, 24]
    d_model_options = [
        16,
        24,
        32,
        48,
        64,
        96,
        128,
        160,
        192,
        256,
        384,
        512,
        768,
        1024,
        1536,
        2048,
    ]
    n_heads_options = [1, 2, 4, 6, 8, 12, 16, 24, 32]

    candidates: list[ModelConfig] = []

    for n_layers in layer_counts:
        for d_model in d_model_options:
            # Only allow head counts that evenly divide d_model
            valid_heads = [h for h in n_heads_options if d_model % h == 0]
            if not valid_heads:
                continue
            for n_heads in valid_heads:
                head_dim = d_model // n_heads
                if head_dim % 2 != 0:  # Must be even for rotate_half
                    continue
                d_ff = d_model * 4
                cfg = ModelConfig(
                    vocab_size=vocab_size,
                    max_seq_len=max_seq_len,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    d_model=d_model,
                    d_ff=d_ff,
                )
                total = cfg.legacy_param_count
                if min_params <= total <= max_params:
                    candidates.append(cfg)

    # Sort by parameter count and sample evenly in log-space between requested bounds
    candidates.sort(key=lambda c: c.legacy_param_count)
    if not candidates:
        return []

    # If explicit target parameter counts are provided, pick nearest candidates
    if target_params is not None and len(target_params) > 0:
        used: set[int] = set()
        selected: list[ModelConfig] = []
        candidate_params = np.array(
            [c.legacy_param_count for c in candidates], dtype=np.float64
        )
        for t in target_params:
            diffs = np.abs(candidate_params - float(t))
            order = np.argsort(diffs, kind="stable")
            for idx in order:
                if int(idx) not in used:
                    used.add(int(idx))
                    selected.append(candidates[int(idx)])
                    break
        return selected[:num_configs]

    # Otherwise, clamp bounds and sample evenly in log-space
    feasible_min = candidates[0].legacy_param_count
    feasible_max = candidates[-1].legacy_param_count
    lo = max(min_params, feasible_min)
    hi = min(max_params, feasible_max)

    targets = np.logspace(np.log10(lo), np.log10(hi), num_configs)
    used = set()
    selected = []
    logs = np.log10([c.legacy_param_count for c in candidates])

    for t in targets:
        lt = np.log10(t)
        diffs = np.abs(logs - lt)
        order = np.argsort(diffs, kind="stable")
        for idx in order:
            if int(idx) not in used:
                used.add(int(idx))
                selected.append(candidates[int(idx)])
                break

    # Ensure exact count (in rare cases of duplicates)
    if len(selected) < num_configs:
        for i in range(len(candidates)):
            if i not in used:
                selected.append(candidates[i])
                if len(selected) == num_configs:
                    break

    return selected
