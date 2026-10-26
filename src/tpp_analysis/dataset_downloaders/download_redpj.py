#!/usr/bin/env python3
"""
Download + pretokenize RedPajama-V2 train and val splits using SHUFFLED indices.

APPROACH (shuffled, distribution-matched):
  1. Fetch shard manifest for a single CC snapshot (default: 2023-06, en, head_middle).
  2. Shuffle shard download order (avoids WARC-file domain clustering bias).
  3. Download shards, extract documents into memory.
  4. Generate a random permutation of all document indices (with fixed seed).
  5. Walk through shuffled indices, tokenize + chunk each document into
     ceil(tokens / max_seq_len) sequences. Write train bins until we
     hit the target token count.
  6. Continue walking through the SAME shuffled permutation (no overlap
     by construction) and write val bins until we hit the target token count.
  7. Save split_config.json with all metadata for reproducibility.

WHY SHUFFLED:
  RedPajama shards are organized by WARC file origin. Documents in early
  shards cluster by domain (news, forums, etc.) differently from later shards.
  Within a single snapshot, unigram KL between sequential first-N / last-N
  splits is 5.2x worse than a shuffled split. Shuffling both shard order
  and document order eliminates this content bias.

OVERLAP GUARANTEE:
  Train gets shuffled_doc_indices[0 : T] and val gets shuffled_doc_indices[T : T+V].
  These are disjoint contiguous slices of a permutation - zero overlap by
  construction. An explicit set-intersection check is run and asserted.

Dataset: togethercomputer/RedPajama-Data-V2
  Snapshot: 2023-06 (single, stable)
  Language: en
  Partition: head_middle (high-quality, Wikipedia-like perplexity)

Tokenizer: cl100k_base
Chunking: documents split into max_seq_len-token sequences (no truncation)

Usage:
    python redpajama_train_val.py
    python redpajama_train_val.py --target-tokens 600000000 --max-seq-len 256 --seed 42
    python redpajama_train_val.py --snapshot 2023-06 --max-shards 20
"""

import os
import json
import gzip
import numpy as np
import tiktoken
import argparse
import requests
from tqdm import tqdm
import time

DEFAULT_SEED = 42
DEFAULT_SNAPSHOT = "2023-06"
DEFAULT_LANG = "en"
DEFAULT_PARTITION = "head_middle"
BASE_URL = "https://data.together.xyz/redpajama-data-v2/v1.0.0"

# Rough estimate for progress bars - actual count determined after download
EST_TOKENS_PER_DOC = 300


# ============================================================
# SHARD DOWNLOAD + EXTRACTION
# ============================================================


def fetch_shard_manifest(snapshot, lang, partition, base_url):
    """Fetch the list of shard paths from the RedPajama manifest."""
    listings_url = f"{base_url}/listings/{lang}-{snapshot}-{partition}.txt"
    print(f"Fetching manifest: {listings_url}")
    resp = requests.get(listings_url, timeout=30)
    resp.raise_for_status()
    shard_paths = [line.strip() for line in resp.text.splitlines() if line.strip()]
    print(f"Found {len(shard_paths)} shards in manifest")
    return shard_paths


def download_shard(shard_path, base_url, cache_dir):
    """
    Download a single shard .json.gz file. Returns local path.
    Writes to .tmp first, renames on success. Verifies file size.
    """
    if not shard_path.endswith(".json.gz"):
        shard_path_dl = shard_path + ".json.gz"
    else:
        shard_path_dl = shard_path

    safe_name = shard_path_dl.replace("/", "_")
    local_path = os.path.join(cache_dir, safe_name)

    if os.path.exists(local_path):
        # Verify existing file isn't truncated
        if os.path.getsize(local_path) > 1024:
            return local_path
        else:
            os.remove(local_path)

    url = f"{base_url}/documents/{shard_path_dl}"
    tmp_path = local_path + ".tmp"

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        expected_size = int(r.headers.get("content-length", 0))
        with (
            open(tmp_path, "wb") as f,
            tqdm(
                total=expected_size,
                unit="B",
                unit_scale=True,
                desc=f"  {safe_name[:50]}",
                leave=False,
            ) as bar,
        ):
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                bar.update(len(chunk))

    # Verify download
    actual_size = os.path.getsize(tmp_path)
    if expected_size > 0 and actual_size < expected_size * 0.95:
        os.remove(tmp_path)
        raise IOError(
            f"Incomplete download: got {actual_size:,} of {expected_size:,} bytes"
        )

    os.rename(tmp_path, local_path)
    return local_path


def read_shard_documents(local_path):
    """
    Read all documents from a .json.gz shard. Returns list of text strings.
    Handles truncated gzip files gracefully.
    """
    docs = []
    try:
        with gzip.open(local_path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    text = obj.get("raw_content", "")
                    if text and len(text) > 50:  # skip tiny docs
                        docs.append(text)
                except json.JSONDecodeError:
                    continue
    except EOFError:
        print(
            f"  ⚠ Truncated gzip: {os.path.basename(local_path)}, "
            f"recovered {len(docs):,} docs"
        )
    except Exception as e:
        print(
            f"  ⚠ Error reading {os.path.basename(local_path)}: {e}, "
            f"recovered {len(docs):,} docs"
        )
    return docs


def download_all_shards(
    shard_paths, base_url, cache_dir, seed, total_token_budget, max_shards=None
):
    """
    Download shards in shuffled order, extract documents.
    Stops when estimated tokens exceed budget (with 20% buffer).
    Returns list of all document texts.
    """
    # Shuffle shard order to sample across full snapshot domain distribution
    shard_rng = np.random.default_rng(seed)
    shard_order = list(range(len(shard_paths)))
    shard_rng.shuffle(shard_order)

    print(f"\nShard download order (first 10): {shard_order[:10]}")
    print(f"Downloading until ~{total_token_budget * 1.2 / 1e9:.2f}B estimated tokens")

    all_documents = []
    shards_downloaded = 0

    for shard_idx in tqdm(shard_order, desc="Downloading shards"):
        shard_path = shard_paths[shard_idx]

        try:
            local_path = download_shard(shard_path, base_url, cache_dir)
            docs = read_shard_documents(local_path)
        except Exception as e:
            print(f"  ⚠ Shard {shard_idx} failed: {e}. Skipping.")
            continue

        all_documents.extend(docs)
        shards_downloaded += 1

        est_tokens = len(all_documents) * EST_TOKENS_PER_DOC

        if shards_downloaded % 5 == 0 or shards_downloaded <= 3:
            print(
                f"  Shard {shards_downloaded}: +{len(docs):,} docs "
                f"(total: {len(all_documents):,}, "
                f"est ~{est_tokens / 1e9:.2f}B tokens)"
            )

        # Stop if we have enough (with 20% buffer)
        if est_tokens >= total_token_budget * 1.2:
            print(f"\n  Enough documents. Stopping at {shards_downloaded} shards.")
            break

        if max_shards and shards_downloaded >= max_shards:
            print(f"\n  Hit --max-shards={max_shards}. Stopping.")
            break

    print(
        f"\n✓ Downloaded {shards_downloaded} shards, "
        f"{len(all_documents):,} documents"
    )
    return all_documents, shards_downloaded


# ============================================================
# CHUNKING + BINARY WRITING (same as Wikipedia pipeline)
# ============================================================


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
    all_documents,
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

            text = all_documents[doc_idx]
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


# ============================================================
# MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Download + pretokenize RedPajama-V2 train and val (SHUFFLED approach)"
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
        "--snapshot",
        type=str,
        default=DEFAULT_SNAPSHOT,
        help=f"CommonCrawl snapshot (default: {DEFAULT_SNAPSHOT})",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=DEFAULT_LANG,
        help=f"Language (default: {DEFAULT_LANG})",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default=DEFAULT_PARTITION,
        help=f"Quality partition (default: {DEFAULT_PARTITION})",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Max shards to download (default: None = as many as needed)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Output directory (default: $DATA_DIR/redpajama, else <repo>/corpora/redpajama)",
    )
    args = parser.parse_args()

    # ── Setup ──
    if args.data_dir is not None:
        cache_dir = args.data_dir
    else:
        data_dir = os.environ.get("DATA_DIR")
        if data_dir:
            cache_dir = os.path.join(data_dir, "redpajama")
        else:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "corpora", "redpajama")

    shard_cache = os.path.join(cache_dir, "shard_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(shard_cache, exist_ok=True)

    total_token_budget = args.target_tokens * 2  # train + val

    print("RedPajama-V2 Train + Val Pipeline (SHUFFLED)")
    print(f"{'='*60}")
    print(f"Snapshot:       {args.snapshot}")
    print(f"Language:       {args.lang}")
    print(f"Partition:      {args.partition}")
    print(f"Target/split:   {args.target_tokens:,} ({args.target_tokens/1e9:.2f}B)")
    print(f"Total budget:   {total_token_budget:,} ({total_token_budget/1e9:.2f}B)")
    print(f"max_seq_len:    {args.max_seq_len}")
    print(f"Seed:           {args.seed}")
    print(f"Output dir:     {cache_dir}")
    print(f"Shard cache:    {shard_cache}")
    print("Approach:       SHUFFLED shards + SHUFFLED docs → contiguous slices")

    # ── Tokenizer ──
    print("\nLoading tokenizer 'cl100k_base'...")
    tokenizer = tiktoken.get_encoding("cl100k_base")
    pad_token_id = tokenizer.eot_token
    print(f"Vocab size: {tokenizer.n_vocab}. Pad token: {pad_token_id}")

    # ── Fetch manifest ──
    shard_paths = fetch_shard_manifest(
        args.snapshot, args.lang, args.partition, BASE_URL
    )

    # ── Download shards (shuffled order) ──
    all_documents, shards_downloaded = download_all_shards(
        shard_paths=shard_paths,
        base_url=BASE_URL,
        cache_dir=shard_cache,
        seed=args.seed,
        total_token_budget=total_token_budget,
        max_shards=args.max_shards,
    )

    n_docs = len(all_documents)
    print(f"\nTotal documents in pool: {n_docs:,}")
    print(f"Estimated tokens: ~{n_docs * EST_TOKENS_PER_DOC / 1e9:.2f}B")

    if n_docs * EST_TOKENS_PER_DOC < total_token_budget * 0.8:
        print(
            f"⚠ May not have enough documents for {total_token_budget:,} tokens. "
            f"Consider increasing --max-shards."
        )

    # ── Generate shuffled permutation of documents ──
    print(f"\nGenerating shuffled document permutation (seed={args.seed})...")
    rng = np.random.default_rng(args.seed)
    shuffled_indices = rng.permutation(n_docs)
    print(f"Permutation ready: {len(shuffled_indices):,} indices")
    print(f"First 10 shuffled doc indices: {shuffled_indices[:10].tolist()}")

    # ── File paths ──
    train_tokens_path = os.path.join(cache_dir, "train_tokens.bin")
    train_lengths_path = os.path.join(cache_dir, "train_lengths.bin")
    val_tokens_path = os.path.join(cache_dir, "val_tokens.bin")
    val_lengths_path = os.path.join(cache_dir, "val_lengths.bin")

    # ── TRAIN ──
    train_end_pos, train_doc_indices, train_tokens, train_chunks = process_split(
        all_documents=all_documents,
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
        all_documents=all_documents,
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

    train_indices_path = os.path.join(cache_dir, "train_article_indices.npy")
    val_indices_path = os.path.join(cache_dir, "val_article_indices.npy")
    np.save(train_indices_path, np.array(train_doc_indices, dtype=np.int64))
    np.save(val_indices_path, np.array(val_doc_indices, dtype=np.int64))

    print(f"\n  Saved permutation:    {indices_path}")
    print(f"  Saved train indices:  {train_indices_path}")
    print(f"  Saved val indices:    {val_indices_path}")

    # ── Save split config ──
    avg_tok_train = train_tokens / len(train_doc_indices) if train_doc_indices else 0
    avg_tok_val = val_tokens / len(val_doc_indices) if val_doc_indices else 0
    unused_docs = n_docs - len(train_doc_indices) - len(val_doc_indices)

    split_config = {
        "dataset": "togethercomputer/RedPajama-Data-V2",
        "snapshot": args.snapshot,
        "lang": args.lang,
        "partition": args.partition,
        "total_documents": n_docs,
        "shards_downloaded": shards_downloaded,
        "total_shards_available": len(shard_paths),
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
            "shuffled_positions": f"[{val_end_pos}, {n_docs})",
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
    print(
        f"  Dataset:  RedPajama-V2 / {args.snapshot} / {args.lang} / {args.partition}"
    )
    print(f"  Seed:     {args.seed}")
    print(f"  Shards:   {shards_downloaded} / {len(shard_paths)} (shuffled order)")
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
