#!/usr/bin/env python3
"""
Distributed training setup utilities.

Extracted from the working evaluate_holdout_distributed.py implementation.
"""

import os
import time
import torch
import torch.distributed as dist
from datetime import timedelta

# Import utils for logging
from training_models.utils import print_info, print_warning


def setup_distributed(rank: int, world_size: int) -> bool:
    """Initialize distributed environment with robust error handling."""
    try:
        log_dir = "debug_logs"
        os.makedirs(log_dir, exist_ok=True)
        with open(f"{log_dir}/distributed_setup_{rank}.log", "w") as f:
            f.write(f"Process {rank}: Starting distributed setup\n")
            f.flush()

        # Set environment variables for distributed training (from web search fixes)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29501"  # Use different port to avoid conflicts

        # NCCL configuration for better stability (from web search)
        os.environ["NCCL_DEBUG"] = "INFO"  # Enable detailed NCCL logging
        os.environ["NCCL_TIMEOUT"] = "3600"  # 60 minute timeout (increased)
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"  # Use loopback interface
        os.environ["NCCL_SOCKET_NTHREADS"] = "8"  # Optimize socket threads
        os.environ["NCCL_NSOCKS_PERTHREAD"] = "8"  # Sockets per thread
        os.environ["NCCL_MIN_NCHANNELS"] = "4"  # Minimum channels
        os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"  # Enable all NCCL debug info

        # Disable problematic features (from web search)
        os.environ["NCCL_IB_DISABLE"] = "1"  # Disable Infiniband support
        os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable peer-to-peer communication

        # Set Gloo to use shared memory instead of TCP sockets
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"  # Use loopback interface
        os.environ["GLOO_SOCKET_FAMILY"] = "AF_INET"  # Use IPv4
        os.environ["GLOO_USE_SHARED_MEMORY"] = "1"  # Force shared memory usage

        with open(f"{log_dir}/distributed_setup_{rank}.log", "a") as f:
            f.write(f"Process {rank}: Environment variables set\n")
            f.flush()

        # Set CUDA device before initializing process group
        torch.cuda.set_device(rank)

        with open(f"{log_dir}/distributed_setup_{rank}.log", "a") as f:
            f.write(f"Process {rank}: CUDA device set to {rank}\n")
            f.flush()

        # Use file-based synchronization to avoid race conditions
        sync_file = f"{log_dir}/distributed_sync"

        # Wait for all processes to be ready
        with open(f"{log_dir}/distributed_setup_{rank}.log", "a") as f:
            f.write(f"Process {rank}: Waiting for all processes to be ready...\n")
            f.flush()

        # Create a ready file for this process
        with open(f"{sync_file}_ready_{rank}", "w") as f:
            f.write(f"Process {rank} ready\n")
            f.flush()

        # Wait for all processes to be ready (with timeout)
        max_wait = 60  # 60 seconds
        wait_time = 0
        while wait_time < max_wait:
            ready_files = [
                f
                for f in os.listdir(log_dir)
                if f.startswith("distributed_sync_ready_")
            ]
            if len(ready_files) >= world_size:
                break
            time.sleep(1)
            wait_time += 1

        with open(f"{log_dir}/distributed_setup_{rank}.log", "a") as f:
            f.write(
                f"Process {rank}: All processes ready, starting Gloo initialization...\n"
            )
            f.flush()

        # Use file-based initialization instead of TCP (more reliable for single-node)
        init_file = f"{log_dir}/dist_init_file"
        init_method = f"file://{init_file}"

        with open(f"{log_dir}/distributed_setup_{rank}.log", "a") as f:
            f.write(f"Process {rank}: Using file-based init_method: {init_method}\n")
            f.flush()

        # Try NCCL with file-based initialization
        try:
            with open(f"{log_dir}/distributed_setup_{rank}.log", "a") as f:
                f.write(f"Process {rank}: Attempting NCCL with file-based init...\n")
                f.flush()

            dist.init_process_group(
                backend="nccl",
                init_method=init_method,
                rank=rank,
                world_size=world_size,
                timeout=timedelta(minutes=30),  # 30 minute timeout
            )
            backend_used = "nccl"

            with open(f"{log_dir}/distributed_setup_{rank}.log", "a") as f:
                f.write(
                    f"Process {rank}: NCCL initialized successfully with file-based method\n"
                )
                f.flush()
        except Exception as nccl_error:
            with open(f"{log_dir}/distributed_setup_{rank}.log", "a") as f:
                f.write(
                    f"Process {rank}: NCCL failed ({nccl_error}), trying Gloo with file-based init...\n"
                )
                f.flush()

            # Fallback to Gloo with file-based initialization
            dist.init_process_group(
                backend="gloo",
                init_method=init_method,
                rank=rank,
                world_size=world_size,
                timeout=timedelta(minutes=30),  # 30 minute timeout
            )
            backend_used = "gloo"

            with open(f"{log_dir}/distributed_setup_{rank}.log", "a") as f:
                f.write(
                    f"Process {rank}: Gloo initialized successfully with file-based method\n"
                )
                f.flush()

        with open(f"{log_dir}/distributed_setup_{rank}.log", "a") as f:
            f.write(
                f"Process {rank}: {backend_used.upper()} backend initialized successfully\n"
            )
            f.flush()
        print_info(
            f"Process {rank}: {backend_used.upper()} backend initialized successfully"
        )

        # Synchronize all processes
        with open(f"{log_dir}/distributed_setup_{rank}.log", "a") as f:
            f.write(f"Process {rank}: Calling barrier...\n")
            f.flush()
        dist.barrier()

        with open(f"{log_dir}/distributed_setup_{rank}.log", "a") as f:
            f.write(f"Process {rank}: Barrier passed\n")
            f.flush()

        # Verify CUDA is working
        torch.cuda.synchronize()

        with open(f"{log_dir}/distributed_setup_{rank}.log", "a") as f:
            f.write(f"Process {rank}: CUDA synchronized\n")
            f.flush()

        print_info(f"Process {rank}/{world_size} initialized on GPU {rank}")
        return True

    except Exception as e:
        with open(f"{log_dir}/distributed_setup_{rank}.log", "a") as f:
            f.write(f"Process {rank}: ERROR - {e}\n")
            f.flush()
        print_warning(f"Failed to initialize distributed training on rank {rank}: {e}")
        return False


def cleanup_distributed():
    """Clean up distributed environment."""
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print_warning(f"Error during cleanup: {e}")
