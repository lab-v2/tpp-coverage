#!/usr/bin/env python3
"""
Parallel n-gram KL divergence comparison (n=1,2,3,4,5) for train vs val splits.

PARALLELISM ARCHITECTURE (3 levels):
  Level 1: Each DATASET is its own independent subprocess      (5 processes)
  Level 2: Each N-GRAM ORDER is its own independent subprocess (5 per dataset = 25 total)
  Level 3: Within each n-gram job, token counting is parallelized across chunks (multiprocessing)

  So running --mode run-all spawns up to 25 independent processes, each of which
  internally uses --workers threads for the actual counting. On godol, you can
  launch all 25 at once or throttle with --max-parallel.

Datasets & split strategies:
  - Wikipedia:   shuffled permutation, 1.3B tokens each side
  - Red Pajama:  shuffled permutation (shard + doc level), 1.3B tokens each side
  - peS2o:       shuffled permutation, 1.3B tokens each side
  - C4:          shuffled permutation, 1.3B tokens each side
  - Cosmopedia:  left/right sequential split (synthetic data, no ordering bias), 1.3B tokens each side

On-disk layout (produced by the dataset_downloaders):
  $DATA_DIR/
    wikipedia/{train,val}_{tokens,lengths}.bin
    redpajama/{train,val}_{tokens,lengths}.bin
    pes2o/{train,val}_{tokens,lengths}.bin
    cosmopedia/{train,val}_{tokens,lengths}.bin
    c4/{train,val}_{tokens,lengths}.bin

Binary format (from our pretokenization pipeline):
  - *_tokens.bin:  uint32, padded sequences of max_seq_len
  - *_lengths.bin: uint16, original sequence lengths (for masking padding)

Usage:
  # Run ALL 15 jobs (5 datasets x 5 n-grams) in parallel:
  python ngram_kl_parallel.py --mode run-all --base-dir $DATA_DIR --workers 4

  # Run all 5 n-gram orders for ONE dataset:
  python ngram_kl_parallel.py --mode run-dataset --dataset wikipedia --base-dir $DATA_DIR

  # Run a SINGLE (dataset, n) job (the atomic unit):
  python ngram_kl_parallel.py --mode single --dataset wikipedia --ngram 3 --base-dir $DATA_DIR

  # Aggregate all per-job JSON files into a LaTeX table:
  python ngram_kl_parallel.py --mode aggregate --output-dir ./ngram_results

  # Quick test with token budget cap:
  python ngram_kl_parallel.py --mode single --dataset wikipedia --ngram 1 --token-budget 50000000
"""

import os
import sys
import json
import argparse
import time
import numpy as np
from collections import Counter
from multiprocessing import Pool, cpu_count
from scipy.special import rel_entr


# ============================================================================
# Config
# ============================================================================

VOCAB_SIZE = 100277  # cl100k_base
MAX_SEQ_LEN = 256
MAX_NGRAM = 5
LAPLACE_ALPHA = 1e-8

# Each dataset lives in its own subdirectory of $DATA_DIR with plain filenames.
DATASET_PATHS = {
    "wikipedia": {
        "subdir": "wikipedia",
        "train_tokens": "train_tokens.bin",
        "train_lengths": "train_lengths.bin",
        "val_tokens":    "val_tokens.bin",
        "val_lengths":   "val_lengths.bin",
        "split_method":  "shuffled",
    },
    "redpajama": {
        "subdir": "redpajama",
        "train_tokens": "train_tokens.bin",
        "train_lengths": "train_lengths.bin",
        "val_tokens":    "val_tokens.bin",
        "val_lengths":   "val_lengths.bin",
        "split_method":  "shuffled",
    },
    "pes2o": {
        "subdir": "pes2o",
        "train_tokens": "train_tokens.bin",
        "train_lengths": "train_lengths.bin",
        "val_tokens":    "val_tokens.bin",
        "val_lengths":   "val_lengths.bin",
        "split_method":  "shuffled",
    },
    "cosmopedia": {
        "subdir": "cosmopedia",
        "train_tokens": "train_tokens.bin",
        "train_lengths": "train_lengths.bin",
        "val_tokens":    "val_tokens.bin",
        "val_lengths":   "val_lengths.bin",
        "split_method":  "shuffled",
    },
    "c4": {
        "subdir": "c4",
        "train_tokens": "train_tokens.bin",
        "train_lengths": "train_lengths.bin",
        "val_tokens":    "val_tokens.bin",
        "val_lengths":   "val_lengths.bin",
        "split_method":  "shuffled",
    },
}


# ============================================================================
# Loading
# ============================================================================

def load_real_tokens(tokens_path, lengths_path, max_seq_len, token_budget=None):
    """
    Memory-map the binary files and return a flat array of real (non-padded) token IDs.
    """
    tokens_mmap = np.memmap(tokens_path, dtype=np.uint32, mode='r')
    lengths_mmap = np.memmap(lengths_path, dtype=np.uint16, mode='r')

    num_sequences = len(lengths_mmap)
    total_padded = len(tokens_mmap)
    assert total_padded == num_sequences * max_seq_len, (
        f"Shape mismatch: {total_padded} tokens vs {num_sequences} seqs * {max_seq_len}"
    )

    tokens_2d = tokens_mmap.reshape(num_sequences, max_seq_len)

    all_tokens = []
    running_count = 0

    for i in range(num_sequences):
        real_len = int(lengths_mmap[i])
        if real_len == 0:
            continue
        seq_tokens = tokens_2d[i, :real_len]
        all_tokens.append(seq_tokens)
        running_count += real_len
        if token_budget is not None and running_count >= token_budget:
            break

    return np.concatenate(all_tokens)


# ============================================================================
# Chunked n-gram counting (multiprocessing within a single job)
# ============================================================================

def count_ngrams_chunk(args):
    """
    Worker: count n-grams for a single chunk of token IDs.
    Args: (chunk_bytes, n) -- bytes for pickling efficiency.
    """
    chunk_bytes, n = args
    tokens = np.frombuffer(chunk_bytes, dtype=np.uint32)

    if len(tokens) < n:
        return {}

    if n == 1:
        unique, counts = np.unique(tokens, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    else:
        counter = Counter()
        token_list = tokens.tolist()
        ngram_iter = zip(*(token_list[i:] for i in range(n)))
        counter.update(ngram_iter)
        return dict(counter)


def parallel_ngram_count(tokens, n, num_workers):
    """
    Split tokens into chunks and count n-grams in parallel.
    Chunks overlap by (n-1) tokens at boundaries so no n-grams are missed.
    """
    total_len = len(tokens)
    if total_len < n:
        return Counter()

    chunk_size = max(1_000_000, total_len // num_workers)
    overlap = n - 1

    chunks = []
    start = 0
    while start < total_len:
        end = min(start + chunk_size, total_len)
        chunk_end = min(end + overlap, total_len)
        chunk = tokens[start:chunk_end]
        chunks.append((chunk.tobytes(), n))
        start = end

    with Pool(processes=min(num_workers, len(chunks))) as pool:
        results = pool.map(count_ngrams_chunk, chunks)

    merged = Counter()
    for r in results:
        for k, v in r.items():
            merged[k] += v

    return merged


# ============================================================================
# KL and cosine
# ============================================================================

def compute_kl_cosine(counter_a, counter_b):
    all_keys = set(counter_a.keys()) | set(counter_b.keys())
    num_types = len(all_keys)

    if num_types == 0:
        return {"sym_kl": 0.0, "kl_ab": 0.0, "kl_ba": 0.0, "cosine": 1.0,
                "types_a": 0, "types_b": 0, "types_union": 0,
                "total_a": 0, "total_b": 0}

    total_a = sum(counter_a.values())
    total_b = sum(counter_b.values())

    freq_a = np.array([counter_a.get(k, 0) for k in all_keys], dtype=np.float64)
    freq_b = np.array([counter_b.get(k, 0) for k in all_keys], dtype=np.float64)

    p = (freq_a + LAPLACE_ALPHA) / (total_a + LAPLACE_ALPHA * num_types)
    q = (freq_b + LAPLACE_ALPHA) / (total_b + LAPLACE_ALPHA * num_types)

    kl_pq = float(np.sum(rel_entr(p, q)))
    kl_qp = float(np.sum(rel_entr(q, p)))
    sym_kl = (kl_pq + kl_qp) / 2.0

    norm_a = np.linalg.norm(freq_a)
    norm_b = np.linalg.norm(freq_b)
    cosine = float(np.dot(freq_a, freq_b) / (norm_a * norm_b)) if norm_a and norm_b else 0.0

    return {
        "sym_kl": sym_kl, "kl_ab": kl_pq, "kl_ba": kl_qp, "cosine": cosine,
        "types_a": len(counter_a), "types_b": len(counter_b),
        "types_union": num_types, "total_a": total_a, "total_b": total_b,
    }


# ============================================================================
# ATOMIC UNIT: single (dataset, n) job
# ============================================================================

def run_single_ngram(dataset_name, n, base_dir, output_dir, num_workers, token_budget=None):
    """
    Run n-gram KL for one dataset and one n-gram order.
    This is the atomic parallelizable unit.
    Writes: {output_dir}/{dataset_name}_ngram_{n}.json
    """
    cfg = DATASET_PATHS[dataset_name]
    print(f"\n[{dataset_name} | n={n}] Starting ({num_workers} workers)")

    ds_dir = os.path.join(base_dir, cfg["subdir"])
    train_tok_path = os.path.join(ds_dir, cfg["train_tokens"])
    train_len_path = os.path.join(ds_dir, cfg["train_lengths"])
    val_tok_path   = os.path.join(ds_dir, cfg["val_tokens"])
    val_len_path   = os.path.join(ds_dir, cfg["val_lengths"])

    for p in [train_tok_path, train_len_path, val_tok_path, val_len_path]:
        if not os.path.exists(p):
            print(f"  ERROR: File not found: {p}")
            return None

    # Load tokens
    t0 = time.time()
    train_tokens = load_real_tokens(train_tok_path, train_len_path, MAX_SEQ_LEN, token_budget)
    print(f"  [{dataset_name}|n={n}] Train: {len(train_tokens):,} tokens ({time.time()-t0:.1f}s)")

    t0 = time.time()
    val_tokens = load_real_tokens(val_tok_path, val_len_path, MAX_SEQ_LEN, token_budget)
    print(f"  [{dataset_name}|n={n}] Val: {len(val_tokens):,} tokens ({time.time()-t0:.1f}s)")

    # Count n-grams
    t0 = time.time()
    train_counts = parallel_ngram_count(train_tokens, n, num_workers)
    print(f"  [{dataset_name}|n={n}] Train {n}-grams: {len(train_counts):,} types ({time.time()-t0:.1f}s)")

    t0 = time.time()
    val_counts = parallel_ngram_count(val_tokens, n, num_workers)
    print(f"  [{dataset_name}|n={n}] Val {n}-grams: {len(val_counts):,} types ({time.time()-t0:.1f}s)")

    # Compute metrics
    t0 = time.time()
    metrics = compute_kl_cosine(train_counts, val_counts)
    print(f"  [{dataset_name}|n={n}] symKL={metrics['sym_kl']:.6f}  "
          f"cosine={metrics['cosine']:.6f} ({time.time()-t0:.1f}s)")

    result = {
        "dataset": dataset_name,
        "n": n,
        "split_method": cfg["split_method"],
        "train_tokens": int(len(train_tokens)),
        "val_tokens": int(len(val_tokens)),
        "sym_kl": metrics["sym_kl"],
        "kl_train_val": metrics["kl_ab"],
        "kl_val_train": metrics["kl_ba"],
        "cosine": metrics["cosine"],
        "unique_types_train": metrics["types_a"],
        "unique_types_val": metrics["types_b"],
        "unique_types_union": metrics["types_union"],
        "total_train": metrics["total_a"],
        "total_val": metrics["total_b"],
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{dataset_name}_ngram_{n}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [{dataset_name}|n={n}] Written: {out_path}")

    return result


# ============================================================================
# Launch modes
# ============================================================================

def launch_subprocesses(jobs, max_parallel, extra_args):
    """
    Launch a list of (dataset, n) jobs as independent subprocesses.
    Throttle concurrency to max_parallel.
    """
    import subprocess
    from collections import deque

    active = deque()
    pending = list(jobs)
    completed = []

    while pending or active:
        # Check for completions
        still_active = deque()
        for ds, n, p in active:
            ret = p.poll()
            if ret is not None:
                status = "OK" if ret == 0 else f"FAILED({ret})"
                print(f"  Finished: {ds} n={n} -> {status}")
                completed.append((ds, n, ret))
            else:
                still_active.append((ds, n, p))
        active = still_active

        # Launch up to max_parallel
        while pending and len(active) < max_parallel:
            ds, n = pending.pop(0)
            cmd = [
                sys.executable, __file__,
                "--mode", "single",
                "--dataset", ds,
                "--ngram", str(n),
            ] + extra_args
            print(f"  Launching: {ds} n={n}")
            p = subprocess.Popen(cmd)
            active.append((ds, n, p))

        if active:
            time.sleep(1.0)

    return completed


def run_all(args):
    """Launch jobs for all datasets x selected n-grams."""
    ngrams = args.ngrams or list(range(1, MAX_NGRAM + 1))
    datasets = args.datasets or list(DATASET_PATHS.keys())
    jobs = [(ds, n) for ds in datasets for n in ngrams]
    print(f"Launching {len(jobs)} jobs (datasets={datasets}, ngrams={ngrams}, max {args.max_parallel} concurrent)...")

    extra = ["--base-dir", args.base_dir, "--output-dir", args.output_dir,
             "--workers", str(args.workers)]
    if args.token_budget:
        extra += ["--token-budget", str(args.token_budget)]

    results = launch_subprocesses(jobs, args.max_parallel, extra)

    failed = [r for r in results if r[2] != 0]
    if failed:
        print(f"\n{len(failed)} jobs FAILED:")
        for ds, n, rc in failed:
            print(f"  {ds} n={n} (exit {rc})")
    else:
        print(f"\nAll {len(results)} jobs completed successfully!")
    print(f"\nRun aggregation:  python {__file__} --mode aggregate --output-dir {args.output_dir}")


def run_dataset(args):
    """Launch selected n-gram jobs for a single dataset."""
    ngrams = args.ngrams or list(range(1, MAX_NGRAM + 1))
    jobs = [(args.dataset, n) for n in ngrams]
    print(f"Launching {len(jobs)} n-gram jobs for {args.dataset} (ngrams={ngrams})...")

    extra = ["--base-dir", args.base_dir, "--output-dir", args.output_dir,
             "--workers", str(args.workers)]
    if args.token_budget:
        extra += ["--token-budget", str(args.token_budget)]

    launch_subprocesses(jobs, args.max_parallel, extra)


# ============================================================================
# Aggregate -> LaTeX
# ============================================================================

def aggregate_results(output_dir, table_max_n=3):
    datasets_display = {
        "wikipedia": "Wikipedia", "redpajama": "Red Pajama",
        "cosmopedia": "Cosmopedia", "pes2o": "peS2o", "c4": "C4",
    }
    split_display = {"shuffled": "Shuffled", "left_right": "Left/Right"}

    all_results = {}
    for ds_name in DATASET_PATHS:
        all_results[ds_name] = {}
        for n in range(1, table_max_n + 1):
            path = os.path.join(output_dir, f"{ds_name}_ngram_{n}.json")
            if os.path.exists(path):
                with open(path) as f:
                    all_results[ds_name][n] = json.load(f)
            else:
                print(f"  WARNING: Missing {path}")

    # Console summary
    print(f"\n{'='*100}")
    print("  N-GRAM KL DIVERGENCE SUMMARY (train vs val)")
    print(f"{'='*100}")
    header = (f"{'Dataset':<14} {'Split':<12} {'n':>3}  {'symKL':>12} {'cosine':>10} "
              f"{'types(union)':>14} {'train_tok':>14} {'val_tok':>14}")
    print(header)
    print("-" * 100)

    for ds_name in ["wikipedia", "redpajama", "cosmopedia", "pes2o", "c4"]:
        for n in range(1, table_max_n + 1):
            r = all_results[ds_name].get(n)
            if r is None:
                continue
            split = split_display.get(r.get("split_method", ""), "?")
            print(f"{datasets_display[ds_name]:<14} {split:<12} {n:>3}  "
                  f"{r['sym_kl']:>12.6f} {r['cosine']:>10.6f} "
                  f"{r['unique_types_union']:>14,} "
                  f"{r['total_train']:>14,} {r['total_val']:>14,}")
        print("-" * 100)

    # LaTeX table
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Symmetric KL divergence and cosine similarity between train and validation "
                 r"$n$-gram distributions ($n=1,\ldots," + str(table_max_n) + r"$). Wikipedia, Red Pajama, peS2o, and C4 use "
                 r"shuffled-permutation splits; Cosmopedia uses a left/right sequential split "
                 r"(synthetic data with no inherent ordering bias). All splits target 1.3B tokens per side.}")
    latex.append(r"\label{tab:ngram_kl}")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{ll" + "r" * table_max_n + "}")
    latex.append(r"\toprule")

    ngram_headers = " & ".join([f"$n={n}$" for n in range(1, table_max_n + 1)])
    latex.append(r"\textbf{Dataset} & \textbf{Metric} & " + ngram_headers + r" \\")
    latex.append(r"\midrule")

    for i, ds_name in enumerate(["wikipedia", "redpajama", "cosmopedia", "pes2o", "c4"]):
        display = datasets_display[ds_name]
        r0 = all_results[ds_name].get(1, {})
        split = split_display.get(r0.get("split_method", ""), "?")

        kl_vals = []
        cos_vals = []
        for n in range(1, table_max_n + 1):
            r = all_results[ds_name].get(n)
            kl_vals.append(f"{r['sym_kl']:.4f}" if r else "--")
            cos_vals.append(f"{r['cosine']:.6f}" if r else "--")

        latex.append(
            f"\\multirow{{2}}{{*}}{{{display} ({split})}} & Sym.~KL & "
            + " & ".join(kl_vals) + r" \\"
        )
        latex.append(" & Cosine & " + " & ".join(cos_vals) + r" \\")

        if i < 4:
            latex.append(r"\midrule")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    latex_str = "\n".join(latex)

    tex_path = os.path.join(output_dir, "ngram_kl_table.tex")
    with open(tex_path, "w") as f:
        f.write(latex_str)
    print(f"\nLaTeX table written to {tex_path}")
    print("\n" + latex_str)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Parallel n-gram KL divergence (each dataset x n-gram = its own process)"
    )
    parser.add_argument("--mode", required=True,
                        choices=["single", "run-dataset", "run-all", "aggregate"],
                        help="single: one (dataset,n) job; run-dataset: all n for one dataset; "
                             "run-all: all 25 jobs; aggregate: build LaTeX")
    parser.add_argument("--dataset", choices=list(DATASET_PATHS.keys()),
                        help="Dataset name (for single / run-dataset)")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_PATHS.keys()),
                        default=None,
                        help="Datasets to run (for run-all). "
                             "E.g. --datasets wikipedia c4. Default: all.")
    parser.add_argument("--ngram", type=int, choices=range(1, MAX_NGRAM + 1),
                        help="N-gram order (for single mode)")
    parser.add_argument("--ngrams", type=int, nargs="+",
                        choices=range(1, MAX_NGRAM + 1), default=None,
                        help="N-gram orders to run (for run-all / run-dataset). "
                             "E.g. --ngrams 1 2 3. Default: all 1-5.")
    parser.add_argument("--base-dir", default=os.environ.get("DATA_DIR", "."),
                        help="Root dir containing dataset subdirectories "
                             "(defaults to $DATA_DIR if set, else current dir)")
    parser.add_argument("--output-dir", default="./ngram_results",
                        help="Where to write per-job JSON results and LaTeX table")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() // 4),
                        help="Workers for within-job parallel counting")
    parser.add_argument("--max-parallel", type=int, default=8,
                        help="Max concurrent subprocesses for run-all / run-dataset")
    parser.add_argument("--token-budget", type=int, default=None,
                        help="Cap tokens loaded per split (for testing)")
    args = parser.parse_args()

    if args.mode == "single":
        if not args.dataset or not args.ngram:
            parser.error("--dataset and --ngram required for --mode single")
        run_single_ngram(
            args.dataset, args.ngram, args.base_dir, args.output_dir,
            args.workers, args.token_budget,
        )

    elif args.mode == "run-dataset":
        if not args.dataset:
            parser.error("--dataset required for --mode run-dataset")
        run_dataset(args)

    elif args.mode == "run-all":
        run_all(args)

    elif args.mode == "aggregate":
        aggregate_results(args.output_dir)


if __name__ == "__main__":
    main()