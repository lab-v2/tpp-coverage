#!/usr/bin/env python3
"""
Download + pretokenize peS2o v2 train and val splits using SHUFFLED indices.

APPROACH (shuffled, distribution-matched):
  1. Load peS2o v2 from HuggingFace (random access).
  2. Generate a random permutation of all document indices (with fixed seed).
  3. Walk through shuffled indices, tokenize + chunk each document into
     ceil(tokens / max_seq_len) sequences. Write train bins until we
     hit the target token count.
  4. Continue walking through the SAME shuffled permutation (no overlap
     by construction) and write val bins until we hit the target token count.
  5. Save split_config.json with all metadata for reproducibility.

WHY SHUFFLED:
  peS2o contains two sub-corpora: s2ag (short abstracts, ~100-200 tokens)
  and s2orc (full papers, ~2000+ tokens). These are grouped in the dataset
  ordering. A sequential left/right split produces Δ mean = +5818 tokens
  and length KL = 8.24 (catastrophic). Shuffling reduces this to Δ = +14.6
  and KL = 0.0015 - a 5613x improvement.

OVERLAP GUARANTEE:
  Train gets shuffled_indices[0 : T] and val gets shuffled_indices[T : T+V].
  These are disjoint contiguous slices of a permutation - zero overlap by
  construction. An explicit set-intersection check is run and asserted.

Dataset: allenai/peS2o, version v2 (~38.8M documents)
Tokenizer: cl100k_base
Chunking: documents split into max_seq_len-token sequences (no truncation)

Usage:
    python pes2o_train_val.py
    python pes2o_train_val.py --target-tokens 600000000 --max-seq-len 256 --seed 42
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
DEFAULT_VERSION = "v2"


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
    documents, write bins until target_tokens reached.

    Returns:
        (end_pos, doc_indices_used, total_tokens, total_chunks)
    """
    total_tokens = 0
    total_chunks = 0
    docs_processed = 0
    doc_indices_used = []
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
        pbar = tqdm(desc=split_name.capitalize(), unit=" docs")

        while pos < len(shuffled_indices) and total_tokens < target_tokens:
            doc_idx = int(shuffled_indices[pos])

            text = dataset[doc_idx]["text"]
            tokens = tokenizer.encode(text, allowed_special="all")

            if len(tokens) == 0:
                pos += 1
                continue

            real, chunks = chunk_and_write(tokens, max_seq_len, pad_token_id, tf, lf)

            total_tokens += real
            total_chunks += chunks
            docs_processed += 1
            doc_indices_used.append(doc_idx)
            pos += 1

            pbar.update(1)

            if docs_processed % 5000 == 0:
                elapsed = time.time() - t0
                pbar.set_postfix(
                    {
                        "tokens": f"{total_tokens/1e9:.3f}B",
                        "chunks": f"{total_chunks:,}",
                        "doc/s": f"{docs_processed/elapsed:.0f}",
                    }
                )

        pbar.close()

    elapsed = time.time() - t0

    print(f"\n✓ {split_name.capitalize()} complete ({elapsed/60:.1f} min):")
    print(f"  Shuffled positions: [{start_pos:,}, {pos:,})")
    print(f"  Documents: {docs_processed:,}")
    print(f"  Chunks:    {total_chunks:,}")
    print(f"  Tokens:    {total_tokens:,} ({total_tokens/1e9:.3f}B)")
    print(f"  Avg tok/doc: {total_tokens/docs_processed:.1f}")
    print(f"  Files: {tokens_bin_path}")
    print(f"         {lengths_bin_path}")

    return pos, doc_indices_used, total_tokens, total_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Download + pretokenize peS2o v2 train and val (SHUFFLED approach)"
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
        "--version",
        type=str,
        default=DEFAULT_VERSION,
        help=f"peS2o version (default: {DEFAULT_VERSION})",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Output directory (default: $DATA_DIR/pes2o, else <repo>/corpora/pes2o)",
    )
    args = parser.parse_args()

    # ── Setup ──
    if args.data_dir is not None:
        cache_dir = args.data_dir
    else:
        data_dir = os.environ.get("DATA_DIR")
        if data_dir:
            cache_dir = os.path.join(data_dir, "pes2o")
        else:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "corpora", "pes2o")

    os.makedirs(cache_dir, exist_ok=True)

    print("peS2o Train + Val Pipeline (SHUFFLED)")
    print(f"{'='*60}")
    print(f"Version:        {args.version}")
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

    # ── Load dataset ──
    print(f"\nLoading peS2o {args.version} dataset (random access mode)...")
    dataset = load_dataset(
        "allenai/peS2o", args.version, split="train", trust_remote_code=True
    )
    total_docs = len(dataset)
    print(f"Loaded {total_docs:,} documents")

    # Quick source distribution check
    from collections import Counter

    source_sample = Counter(
        dataset[i]["source"] for i in range(min(10_000, total_docs))
    )
    print(f"Source distribution (first 10k): {dict(source_sample)}")

    # ── Generate shuffled permutation ──
    print(f"\nGenerating shuffled permutation (seed={args.seed})...")
    rng = np.random.default_rng(args.seed)
    shuffled_indices = rng.permutation(total_docs)
    print(f"Permutation ready: {len(shuffled_indices):,} indices")
    print(f"First 10 shuffled indices: {shuffled_indices[:10].tolist()}")

    # ── File paths ──
    train_tokens_path = os.path.join(cache_dir, "train_tokens.bin")
    train_lengths_path = os.path.join(cache_dir, "train_lengths.bin")
    val_tokens_path = os.path.join(cache_dir, "val_tokens.bin")
    val_lengths_path = os.path.join(cache_dir, "val_lengths.bin")

    # ── TRAIN: walk shuffled indices from position 0 ──
    train_end_pos, train_doc_indices, train_tokens, train_chunks = process_split(
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
    val_end_pos, val_doc_indices, val_tokens, val_chunks = process_split(
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
    train_set = set(train_doc_indices)
    val_set = set(val_doc_indices)
    overlap = train_set & val_set
    print(f"  Train documents: {len(train_set):,}")
    print(f"  Val documents:   {len(val_set):,}")
    print(f"  Overlap:         {len(overlap)}")
    assert len(overlap) == 0, (
        f"OVERLAP DETECTED: {len(overlap)} documents in both splits! "
        f"This should never happen with contiguous slices of a permutation."
    )
    print("  ✓ Zero overlap confirmed")

    # ── Save indices for reproducibility ──
    indices_path = os.path.join(cache_dir, "shuffled_indices.npy")
    np.save(indices_path, shuffled_indices)
    print(f"\n  Saved full permutation: {indices_path}")

    train_indices_path = os.path.join(cache_dir, "train_article_indices.npy")
    val_indices_path = os.path.join(cache_dir, "val_article_indices.npy")
    np.save(train_indices_path, np.array(train_doc_indices, dtype=np.int64))
    np.save(val_indices_path, np.array(val_doc_indices, dtype=np.int64))
    print(f"  Saved train indices: {train_indices_path}")
    print(f"  Saved val indices:   {val_indices_path}")

    # ── Save split config ──
    avg_tok_train = train_tokens / len(train_doc_indices) if train_doc_indices else 0
    avg_tok_val = val_tokens / len(val_doc_indices) if val_doc_indices else 0
    unused_docs = total_docs - len(train_doc_indices) - len(val_doc_indices)

    split_config = {
        "dataset": f"allenai/peS2o/{args.version}",
        "total_documents": total_docs,
        "max_seq_len": args.max_seq_len,
        "approach": "shuffled_chunking",
        "seed": args.seed,
        "overlap": len(overlap),
        "train": {
            "shuffled_positions": f"[0, {train_end_pos})",
            "documents": len(train_doc_indices),
            "chunks": train_chunks,
            "tokens": train_tokens,
            "avg_tokens_per_doc": avg_tok_train,
            "avg_chunks_per_doc": (
                train_chunks / len(train_doc_indices) if train_doc_indices else 0
            ),
        },
        "val": {
            "shuffled_positions": f"[{train_end_pos}, {val_end_pos})",
            "documents": len(val_doc_indices),
            "chunks": val_chunks,
            "tokens": val_tokens,
            "avg_tokens_per_doc": avg_tok_val,
            "avg_chunks_per_doc": (
                val_chunks / len(val_doc_indices) if val_doc_indices else 0
            ),
        },
        "unused": {
            "shuffled_positions": f"[{val_end_pos}, {total_docs})",
            "documents": unused_docs,
        },
        "distribution_check": {
            "train_mean_tok_per_doc": avg_tok_train,
            "val_mean_tok_per_doc": avg_tok_val,
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
    print(f"  Dataset:  peS2o {args.version}")
    print(f"  Seed:     {args.seed}")
    print(
        f"  Train:    {len(train_doc_indices):,} docs, "
        f"{train_chunks:,} chunks, "
        f"{train_tokens:,} tokens ({train_tokens/1e9:.3f}B)"
    )
    print(
        f"  Val:      {len(val_doc_indices):,} docs, "
        f"{val_chunks:,} chunks, "
        f"{val_tokens:,} tokens ({val_tokens/1e9:.3f}B)"
    )
    print(f"  Unused:   {unused_docs:,} docs")
    print(f"  Overlap:  {len(overlap)} ✓")
    print(
        f"  Δ mean:   {avg_tok_val - avg_tok_train:+.1f} tok/doc "
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
