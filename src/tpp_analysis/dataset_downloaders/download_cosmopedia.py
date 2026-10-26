#!/usr/bin/env python3
"""
Download + pretokenize Cosmopedia train and val splits using SHUFFLED indices.

APPROACH (shuffled, distribution-matched):
  1. Generate a random permutation of all example indices (with fixed seed).
  2. Walk through shuffled indices, tokenize + chunk each example into
     ceil(tokens / max_seq_len) sequences. Write train bins until we
     hit the target token count.
  3. Continue walking through the SAME shuffled permutation (no overlap
     by construction) and write val bins until we hit the target token count.
  4. Save split_config.json with all metadata for reproducibility.

WHY SHUFFLED:
  The Cosmopedia HuggingFace dump may have ordering artifacts (e.g. by
  source, topic, or generation batch). Sequential train/val splits from
  the beginning/end risk systematic distribution shift that biases
  scaling law fits. Shuffling eliminates this. Train and val are
  contiguous slices of a single permutation - zero overlap by construction.

Dataset: HuggingFaceTB/cosmopedia, subset web_samples_v2
Tokenizer: cl100k_base
Chunking: examples split into max_seq_len-token sequences (no truncation)

Usage:
    python cosmopedia_train_val.py
    python cosmopedia_train_val.py --target-tokens 600000000 --max-seq-len 256 --seed 42
"""

import os
import json
import numpy as np
import tiktoken
import argparse
from datasets import load_dataset
from tqdm import tqdm
import time

DEFAULT_SEED = 42


def chunk_and_write(tokens, max_seq_len, pad_token_id, token_file, length_file):
    """
    Split a token list into ceil(len/max_seq_len) chunks.
    Write each chunk (padded if needed) to binary files.
    Returns (real_tokens_written, chunks_written).
    """
    num_tokens = len(tokens)
    if num_tokens == 0:
        return 0, 0

    real_tokens = 0
    chunks = 0

    for start in range(0, num_tokens, max_seq_len):
        chunk = tokens[start : start + max_seq_len]
        chunk_len = len(chunk)

        if chunk_len < max_seq_len:
            chunk = chunk + [pad_token_id] * (max_seq_len - chunk_len)

        length_file.write(np.array([chunk_len], dtype=np.uint16).tobytes())
        token_file.write(np.array(chunk, dtype=np.uint32).tobytes())

        real_tokens += chunk_len
        chunks += 1

    return real_tokens, chunks


def process_split(
    dataset,
    shuffled_indices,
    start_pos,
    target_tokens,
    max_seq_len,
    pad_token_id,
    tokenizer,
    tokens_bin_path,
    lengths_bin_path,
    split_name,
):
    """
    Walk through shuffled_indices starting at start_pos, tokenize + chunk
    examples, write bins until target_tokens reached.

    Returns:
        (end_pos, example_indices_used, total_tokens, total_chunks)
    """
    total_tokens = 0
    total_chunks = 0
    examples_processed = 0
    example_indices_used = []
    pos = start_pos

    print(f"\n{'='*70}")
    print(f"{split_name.upper()}: Walking shuffled indices from position {start_pos:,}")
    print(f"  Target: {target_tokens:,} tokens ({target_tokens/1e9:.3f}B)")
    print(f"{'='*70}")

    t0 = time.time()

    with (
        open(tokens_bin_path, "wb") as tf,
        open(lengths_bin_path, "wb") as lf,
    ):
        pbar = tqdm(desc=split_name.capitalize(), unit=" examples")

        while pos < len(shuffled_indices) and total_tokens < target_tokens:
            # Get the actual example index from the shuffled permutation
            example_idx = int(shuffled_indices[pos])

            # Fetch and tokenize
            text = dataset[example_idx]["text"]
            tokens = tokenizer.encode(text, allowed_special="all")

            # Chunk and write
            real, chunks = chunk_and_write(tokens, max_seq_len, pad_token_id, tf, lf)

            total_tokens += real
            total_chunks += chunks
            examples_processed += 1
            example_indices_used.append(example_idx)
            pos += 1

            pbar.update(1)

            if examples_processed % 5000 == 0:
                elapsed = time.time() - t0
                pbar.set_postfix(
                    {
                        "tokens": f"{total_tokens/1e9:.3f}B",
                        "chunks": f"{total_chunks:,}",
                        "ex/s": f"{examples_processed/elapsed:.0f}",
                    }
                )

        pbar.close()

    elapsed = time.time() - t0

    print(f"\n✓ {split_name.capitalize()} complete ({elapsed/60:.1f} min):")
    print(f"  Shuffled positions: [{start_pos:,}, {pos:,})")
    print(f"  Examples:  {examples_processed:,}")
    print(f"  Chunks:    {total_chunks:,}")
    print(f"  Tokens:    {total_tokens:,} ({total_tokens/1e9:.3f}B)")
    print(f"  Avg tok/example: {total_tokens/examples_processed:.1f}")
    print(f"  Files: {tokens_bin_path}")
    print(f"         {lengths_bin_path}")

    return pos, example_indices_used, total_tokens, total_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Download + pretokenize Cosmopedia train and val (SHUFFLED approach)"
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=600_000_000,
        help="Target tokens for EACH of train and val (default: 600M)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=256,
        help="Chunk size / max sequence length (default: 256)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for shuffling (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="web_samples_v2",
        help="Cosmopedia subset (default: web_samples_v2)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Output directory (default: $DATA_DIR/cosmopedia, else <repo>/corpora/cosmopedia)",
    )
    args = parser.parse_args()

    # ── Setup ──
    if args.data_dir is not None:
        cache_dir = args.data_dir
    else:
        data_dir = os.environ.get("DATA_DIR")
        if data_dir:
            cache_dir = os.path.join(data_dir, "cosmopedia")
        else:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "corpora", "cosmopedia")

    os.makedirs(cache_dir, exist_ok=True)

    print("Cosmopedia Train + Val Pipeline (SHUFFLED)")
    print(f"{'='*60}")
    print(f"Subset:         {args.subset}")
    print(f"Target/split:   {args.target_tokens:,} ({args.target_tokens/1e9:.2f}B)")
    print(f"max_seq_len:    {args.max_seq_len}")
    print(f"Seed:           {args.seed}")
    print(f"Output dir:     {cache_dir}")
    print("Approach:       SHUFFLED chunking (random permutation → contiguous slices)")
    print("                Train = shuffled[0:T], Val = shuffled[T:T+V], zero overlap")

    # ── Tokenizer ──
    print("\nLoading tokenizer 'cl100k_base'...")
    tokenizer = tiktoken.get_encoding("cl100k_base")
    pad_token_id = tokenizer.eot_token
    print(f"Vocab size: {tokenizer.n_vocab}. Pad token: {pad_token_id}")

    # ── Load dataset (random access, NOT streaming) ──
    print("\nLoading dataset (random access mode)...")
    dataset = load_dataset(
        "HuggingFaceTB/cosmopedia",
        args.subset,
        split="train",
    )
    total_examples = len(dataset)
    print(f"Loaded {total_examples:,} examples")

    # ── Generate shuffled permutation ──
    print(f"\nGenerating shuffled permutation (seed={args.seed})...")
    rng = np.random.default_rng(args.seed)
    shuffled_indices = rng.permutation(total_examples)
    print(f"Permutation ready: {len(shuffled_indices):,} indices")
    print(f"First 10 shuffled indices: {shuffled_indices[:10].tolist()}")

    # ── File paths ──
    train_tokens_path = os.path.join(cache_dir, "train_tokens.bin")
    train_lengths_path = os.path.join(cache_dir, "train_lengths.bin")
    val_tokens_path = os.path.join(cache_dir, "val_tokens.bin")
    val_lengths_path = os.path.join(cache_dir, "val_lengths.bin")

    # ── TRAIN: walk shuffled indices from position 0 ──
    train_end_pos, train_example_indices, train_tokens, train_chunks = process_split(
        dataset=dataset,
        shuffled_indices=shuffled_indices,
        start_pos=0,
        target_tokens=args.target_tokens,
        max_seq_len=args.max_seq_len,
        pad_token_id=pad_token_id,
        tokenizer=tokenizer,
        tokens_bin_path=train_tokens_path,
        lengths_bin_path=train_lengths_path,
        split_name="train",
    )

    # ── VAL: continue from where train stopped ──
    val_end_pos, val_example_indices, val_tokens, val_chunks = process_split(
        dataset=dataset,
        shuffled_indices=shuffled_indices,
        start_pos=train_end_pos,
        target_tokens=args.target_tokens,
        max_seq_len=args.max_seq_len,
        pad_token_id=pad_token_id,
        tokenizer=tokenizer,
        tokens_bin_path=val_tokens_path,
        lengths_bin_path=val_lengths_path,
        split_name="val",
    )

    # ── Overlap verification ──
    print(f"\n{'='*70}")
    print("OVERLAP VERIFICATION")
    print(f"{'='*70}")
    train_set = set(train_example_indices)
    val_set = set(val_example_indices)
    overlap = train_set & val_set
    print(f"  Train examples: {len(train_set):,}")
    print(f"  Val examples:   {len(val_set):,}")
    print(f"  Overlap:        {len(overlap)}")
    assert len(overlap) == 0, (
        f"OVERLAP DETECTED: {len(overlap)} examples appear in both splits! "
        f"This should never happen with contiguous slices of a permutation."
    )
    print("  ✓ Zero overlap confirmed")

    # ── Save shuffled indices for reproducibility ──
    indices_path = os.path.join(cache_dir, "shuffled_indices.npy")
    np.save(indices_path, shuffled_indices)
    print(f"\n  Saved full permutation: {indices_path}")

    train_indices_path = os.path.join(cache_dir, "train_example_indices.npy")
    val_indices_path = os.path.join(cache_dir, "val_example_indices.npy")
    np.save(train_indices_path, np.array(train_example_indices, dtype=np.int64))
    np.save(val_indices_path, np.array(val_example_indices, dtype=np.int64))
    print(f"  Saved train indices: {train_indices_path}")
    print(f"  Saved val indices:   {val_indices_path}")

    # ── Save split config ──
    avg_tok_train = (
        train_tokens / len(train_example_indices) if train_example_indices else 0
    )
    avg_tok_val = val_tokens / len(val_example_indices) if val_example_indices else 0
    unused_examples = (
        total_examples - len(train_example_indices) - len(val_example_indices)
    )

    split_config = {
        "dataset": f"HuggingFaceTB/cosmopedia/{args.subset}",
        "total_examples": total_examples,
        "max_seq_len": args.max_seq_len,
        "approach": "shuffled_chunking",
        "seed": args.seed,
        "overlap": len(overlap),
        "train": {
            "shuffled_positions": f"[0, {train_end_pos})",
            "examples": len(train_example_indices),
            "chunks": train_chunks,
            "tokens": train_tokens,
            "avg_tokens_per_example": avg_tok_train,
            "avg_chunks_per_example": (
                train_chunks / len(train_example_indices)
                if train_example_indices
                else 0
            ),
        },
        "val": {
            "shuffled_positions": f"[{train_end_pos}, {val_end_pos})",
            "examples": len(val_example_indices),
            "chunks": val_chunks,
            "tokens": val_tokens,
            "avg_tokens_per_example": avg_tok_val,
            "avg_chunks_per_example": (
                val_chunks / len(val_example_indices) if val_example_indices else 0
            ),
        },
        "unused": {
            "shuffled_positions": f"[{val_end_pos}, {total_examples})",
            "examples": unused_examples,
        },
        "distribution_check": {
            "train_mean_tok_per_example": avg_tok_train,
            "val_mean_tok_per_example": avg_tok_val,
            "delta_mean": avg_tok_val - avg_tok_train,
            "delta_pct": (
                (avg_tok_val - avg_tok_train) / avg_tok_train * 100
                if avg_tok_train > 0
                else 0
            ),
        },
    }

    config_path = os.path.join(cache_dir, "split_config.json")
    with open(config_path, "w") as f:
        json.dump(split_config, f, indent=2)

    # ── Summary ──
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")
    print(f"  Seed:     {args.seed}")
    print(
        f"  Train:    {len(train_example_indices):,} examples, "
        f"{train_chunks:,} chunks, "
        f"{train_tokens:,} tokens ({train_tokens/1e9:.3f}B)"
    )
    print(
        f"  Val:      {len(val_example_indices):,} examples, "
        f"{val_chunks:,} chunks, "
        f"{val_tokens:,} tokens ({val_tokens/1e9:.3f}B)"
    )
    print(f"  Unused:   {unused_examples:,} examples")
    print(f"  Overlap:  {len(overlap)} ✓")
    print(
        f"  Δ mean:   {avg_tok_val - avg_tok_train:+.1f} tok/example "
        f"({(avg_tok_val - avg_tok_train) / avg_tok_train * 100:+.1f}%)"
    )

    print("\n📁 Output files:")
    print(f"  {train_tokens_path}")
    print(f"  {train_lengths_path}")
    print(f"  {val_tokens_path}")
    print(f"  {val_lengths_path}")
    print(f"  {indices_path}")
    print(f"  {train_indices_path}")
    print(f"  {val_indices_path}")
    print(f"  {config_path}")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
