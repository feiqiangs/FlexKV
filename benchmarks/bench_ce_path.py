#!/usr/bin/env python3
"""
Benchmark: FlexKV CE path (transfer_kv_all_layer_mla_ce) performance test.

This script directly calls flexkv's sglang_transfer.cc CE function to measure
the D2H bandwidth of the production code path on P800.

Compares:
  1. transfer_kv_all_layer_mla_ce (the actual FlexKV CE path)
  2. Simple contiguous tensor.copy_() (baseline)
  3. Per-layer tensor.copy_() (78 separate copies)

Usage:
    python bench_ce_path.py [--num-tokens 24064] [--warmup 3] [--repeat 5]
"""

import argparse
import os
import sys
import time
from typing import List, Tuple

import torch

# ============================================================================
# Constants matching production parameters (GLM5 on P800)
# ============================================================================
DEFAULT_NUM_LAYERS = 78
DEFAULT_KV_DIM = 576       # num_head=1, head_size=576 for MLA
DEFAULT_TOKENS_PER_BLOCK = 64
DEFAULT_NUM_TOKENS = 24064
DEFAULT_DTYPE = torch.bfloat16
DEFAULT_WARMUP = 3
DEFAULT_REPEAT = 5


def cuda_host_register(tensor: torch.Tensor) -> None:
    """Register CPU tensor as pinned memory."""
    ptr = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()
    err = torch.cuda.cudart().cudaHostRegister(ptr, size, 0)
    if isinstance(err, tuple):
        err = err[0]
    if err != 0:
        raise RuntimeError(f"cudaHostRegister failed: err={err}")


def prepare_sglang_layout(
    num_layers: int,
    num_tokens: int,
    kv_dim: int,
    dtype: torch.dtype,
    use_shm: bool = True,
):
    """
    Prepare GPU and CPU buffers matching FlexKV's sglang CE path layout.

    GPU: num_layers separate tensors, each [max_tokens, 1, kv_dim]
    CPU: num_layers separate tensors (views of a shared flat buffer)

    Returns:
        gpu_layers: list of per-layer GPU tensors
        cpu_layers: list of per-layer CPU tensors
        src_layers_ptrs: int64 tensor of GPU data_ptr() values
        dst_layers_ptrs: int64 tensor of CPU data_ptr() values
        src_indices: contiguous token indices on GPU side
        dst_indices: contiguous token indices on CPU side
    """
    # item_size = 1 * kv_dim * dtype.itemsize (for MLA: num_head=1)
    item_size = 1 * kv_dim * dtype.itemsize

    # GPU: per-layer tensors
    gpu_max_tokens = num_tokens + 128  # some extra space
    gpu_layers = []
    for _ in range(num_layers):
        t = torch.randn(gpu_max_tokens, 1, kv_dim, dtype=dtype, device="cuda:0")
        gpu_layers.append(t)

    # CPU: flat shared buffer, then create per-layer views
    cpu_max_tokens = num_tokens + 128
    cpu_flat = torch.empty(num_layers * cpu_max_tokens * 1 * kv_dim, dtype=dtype, device="cpu")
    if use_shm:
        cpu_flat.share_memory_()
    cuda_host_register(cpu_flat)

    cpu_layers = []
    layer_size = cpu_max_tokens * 1 * kv_dim
    for i in range(num_layers):
        t = cpu_flat[i * layer_size:(i + 1) * layer_size].reshape(cpu_max_tokens, 1, kv_dim)
        cpu_layers.append(t)

    # Build pointer tables (uint64 tensors of data_ptr values)
    # For MLA (kv_dim=1 in the CE function sense), src_layers has shape [num_layers]
    # C++ side expects UInt64 (uintptr_t)
    import numpy as np
    src_ptrs_np = np.array([t.data_ptr() for t in gpu_layers], dtype=np.uint64)
    dst_ptrs_np = np.array([t.data_ptr() for t in cpu_layers], dtype=np.uint64)
    src_ptrs = torch.from_numpy(src_ptrs_np)
    dst_ptrs = torch.from_numpy(dst_ptrs_np)

    # Indices: contiguous range (Path 0 in sglang_transfer.cc)
    src_offset = 64  # start from offset to be realistic
    src_indices = torch.arange(src_offset, src_offset + num_tokens, dtype=torch.int64)
    dst_indices = torch.arange(0, num_tokens, dtype=torch.int64)

    return gpu_layers, cpu_layers, cpu_flat, src_ptrs, dst_ptrs, src_indices, dst_indices, item_size


def bench_ce_path(
    transfer_fn,
    src_ptrs: torch.Tensor,
    dst_ptrs: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    num_layers: int,
    num_tokens: int,
    kv_dim_ce: int,  # kv_dim parameter for CE function (1 for MLA)
    warmup: int,
    repeat: int,
) -> Tuple[float, float]:
    """
    Benchmark transfer_kv_all_layer_mla_ce.

    Returns: (avg_ms, bandwidth_gbps)
    """
    total_bytes = num_tokens * item_size * num_layers * kv_dim_ce

    # Get current CUDA stream handle
    stream = torch.cuda.current_stream()
    stream_handle = stream.cuda_stream

    def run():
        transfer_fn(
            src_ptrs,       # src_layers (GPU ptrs)
            dst_ptrs,       # dst_layers (CPU ptrs)
            src_indices,    # src_indices
            dst_indices,    # dst_indices
            item_size,      # item_size
            num_layers,     # num_layers
            kv_dim_ce,      # kv_dim
            stream_handle,  # stream_handle
            0,              # device_index
            "bench",        # role
            0,              # gpu_id
            0,              # tp_id
            False,          # layerwise
        )
        torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        run()

    # Measure
    times_ms = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)

    avg_ms = sum(times_ms) / len(times_ms)
    total_gb = total_bytes / (1024**3)
    bandwidth = total_gb / (avg_ms / 1000) if avg_ms > 0 else 0.0
    return avg_ms, bandwidth


def bench_contiguous_copy(
    gpu_layers: List[torch.Tensor],
    cpu_flat: torch.Tensor,
    num_layers: int,
    num_tokens: int,
    kv_dim: int,
    dtype: torch.dtype,
    warmup: int,
    repeat: int,
) -> Tuple[float, float]:
    """
    Benchmark single contiguous tensor.copy_() (best case baseline).
    """
    item_size = 1 * kv_dim * dtype.itemsize
    total_bytes = num_tokens * item_size * num_layers

    # Create one big contiguous GPU tensor
    gpu_flat = torch.cat([t[:num_tokens].reshape(-1) for t in gpu_layers])
    cpu_target = cpu_flat[:gpu_flat.numel()]

    def run():
        cpu_target.copy_(gpu_flat, non_blocking=False)

    # Warmup
    for _ in range(warmup):
        run()

    # Measure
    times_ms = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)

    avg_ms = sum(times_ms) / len(times_ms)
    total_gb = total_bytes / (1024**3)
    bandwidth = total_gb / (avg_ms / 1000) if avg_ms > 0 else 0.0
    return avg_ms, bandwidth


def bench_per_layer_copy(
    gpu_layers: List[torch.Tensor],
    cpu_layers: List[torch.Tensor],
    num_layers: int,
    num_tokens: int,
    kv_dim: int,
    dtype: torch.dtype,
    warmup: int,
    repeat: int,
) -> Tuple[float, float]:
    """
    Benchmark per-layer tensor.copy_() (78 separate copies, non_blocking).
    """
    item_size = 1 * kv_dim * dtype.itemsize
    total_bytes = num_tokens * item_size * num_layers

    def run():
        for i in range(num_layers):
            cpu_layers[i][:num_tokens].copy_(gpu_layers[i][:num_tokens], non_blocking=True)
        torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        run()

    # Measure
    times_ms = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)

    avg_ms = sum(times_ms) / len(times_ms)
    total_gb = total_bytes / (1024**3)
    bandwidth = total_gb / (avg_ms / 1000) if avg_ms > 0 else 0.0
    return avg_ms, bandwidth


def bench_manual_cudamemcpy_per_layer(
    gpu_layers: List[torch.Tensor],
    cpu_layers: List[torch.Tensor],
    num_layers: int,
    num_tokens: int,
    kv_dim: int,
    dtype: torch.dtype,
    warmup: int,
    repeat: int,
    src_offset: int = 64,
) -> Tuple[float, float]:
    """
    Benchmark manual cudaMemcpyAsync per layer (simulating CE Path 0).
    Uses torch.cuda.cudart().cudaMemcpyAsync directly.
    """
    item_size = 1 * kv_dim * dtype.itemsize
    total_bytes = num_tokens * item_size * num_layers
    copy_size = num_tokens * item_size

    stream = torch.cuda.current_stream()
    stream_handle = stream.cuda_stream

    cudart = torch.cuda.cudart()

    def run():
        for i in range(num_layers):
            src_ptr = gpu_layers[i].data_ptr() + src_offset * item_size
            dst_ptr = cpu_layers[i].data_ptr()
            # cudaMemcpyDeviceToHost = 2
            cudart.cudaMemcpyAsync(dst_ptr, src_ptr, copy_size, 2, stream_handle)
        torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        run()

    # Measure
    times_ms = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)

    avg_ms = sum(times_ms) / len(times_ms)
    total_gb = total_bytes / (1024**3)
    bandwidth = total_gb / (avg_ms / 1000) if avg_ms > 0 else 0.0
    return avg_ms, bandwidth


def main():
    parser = argparse.ArgumentParser(description="Benchmark FlexKV CE path performance")
    parser.add_argument("--num-tokens", type=int, default=DEFAULT_NUM_TOKENS)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--no-shm", action="store_true", help="Don't use shared memory for CPU buffer")
    args = parser.parse_args()

    kv_dim = DEFAULT_KV_DIM
    dtype = DEFAULT_DTYPE
    item_size = 1 * kv_dim * dtype.itemsize  # 1152 bytes for MLA
    total_bytes = args.num_tokens * item_size * args.num_layers
    total_gb = total_bytes / (1024**3)
    num_blocks = (args.num_tokens + DEFAULT_TOKENS_PER_BLOCK - 1) // DEFAULT_TOKENS_PER_BLOCK

    print("=" * 80)
    print("  FlexKV CE Path Performance Benchmark")
    print("=" * 80)
    print(f"  Parameters:")
    print(f"    num_layers={args.num_layers}, kv_dim={kv_dim}, dtype={dtype}")
    print(f"    num_tokens={args.num_tokens}")
    print(f"    item_size={item_size} bytes ({item_size/1024:.1f} KB per token)")
    print(f"    per_layer_copy_size={args.num_tokens * item_size / (1024**2):.2f} MB")
    print(f"    total_data={total_gb:.3f} GB")
    print(f"    CE Path 0: {args.num_layers} cudaMemcpyAsync calls (1 per layer)")
    print(f"    warmup={args.warmup}, repeat={args.repeat}")
    print(f"    use_shared_memory={not args.no_shm}")
    print(f"    CUDA device: {torch.cuda.get_device_name(0)}")
    print()

    # Try import flexkv CE function
    has_flexkv_ce = False
    transfer_fn = None
    try:
        sys.path.insert(0, "/workspace/flexkv_dev")
        from flexkv.c_ext import transfer_kv_all_layer_mla_ce
        transfer_fn = transfer_kv_all_layer_mla_ce
        has_flexkv_ce = True
        print("  flexkv.c_ext.transfer_kv_all_layer_mla_ce: AVAILABLE")
    except ImportError as e:
        print(f"  flexkv.c_ext.transfer_kv_all_layer_mla_ce: NOT AVAILABLE ({e})")
    print()

    # Prepare buffers
    print("  Allocating buffers...")
    gpu_layers, cpu_layers, cpu_flat, src_ptrs, dst_ptrs, src_indices, dst_indices, item_sz = \
        prepare_sglang_layout(args.num_layers, args.num_tokens, kv_dim, dtype, use_shm=not args.no_shm)
    print(f"  GPU: {args.num_layers} tensors × [{args.num_tokens + 128}, 1, {kv_dim}]")
    print(f"  CPU: {'shared_memory' if not args.no_shm else 'heap'} + cudaHostRegister")
    print(f"       total={cpu_flat.numel() * cpu_flat.element_size() / (1024**3):.3f} GB")
    print(f"       is_pinned={cpu_flat.is_pinned()}")
    print()

    results = []

    # Test 1: FlexKV CE path (transfer_kv_all_layer_mla_ce)
    if has_flexkv_ce:
        print("=" * 60)
        print("  [1] FlexKV CE path (transfer_kv_all_layer_mla_ce, Path 0)")
        print("      78 cudaMemcpyAsync calls, each ~27.7 MB")
        try:
            avg_ms, bw = bench_ce_path(
                transfer_fn, src_ptrs, dst_ptrs, src_indices, dst_indices,
                item_sz, args.num_layers, args.num_tokens, 1,  # kv_dim_ce=1 for MLA
                args.warmup, args.repeat,
            )
            print(f"      avg_ms={avg_ms:.2f}, bandwidth={bw:.2f} GB/s")
            results.append(("FlexKV CE (transfer_kv_all_layer_mla_ce)", avg_ms, bw))
        except Exception as e:
            print(f"      FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append(("FlexKV CE (FAILED)", 0, 0))

    # Test 2: Manual cudaMemcpyAsync per layer (Python loop)
    # NOTE: Skipped on P800 because torch._C._cudart doesn't expose cudaMemcpyAsync
    print()
    print("=" * 60)
    print("  [2] Manual cudaMemcpyAsync per layer — SKIPPED (not available on P800)")
    print()

    # Test 3: Per-layer tensor.copy_() (78 copies, non_blocking)
    print()
    print("=" * 60)
    print(f"  [3] Per-layer tensor.copy_() ({args.num_layers} copies, non_blocking)")
    avg_ms, bw = bench_per_layer_copy(
        gpu_layers, cpu_layers, args.num_layers, args.num_tokens,
        kv_dim, dtype, args.warmup, args.repeat,
    )
    print(f"      avg_ms={avg_ms:.2f}, bandwidth={bw:.2f} GB/s")
    results.append((f"Per-layer tensor.copy_() ×{args.num_layers}", avg_ms, bw))

    # Test 4: Single contiguous copy (best case)
    print()
    print("=" * 60)
    print("  [4] Single contiguous tensor.copy_() (1 big copy, best case)")
    avg_ms, bw = bench_contiguous_copy(
        gpu_layers, cpu_flat, args.num_layers, args.num_tokens,
        kv_dim, dtype, args.warmup, args.repeat,
    )
    print(f"      avg_ms={avg_ms:.2f}, bandwidth={bw:.2f} GB/s")
    results.append(("Contiguous copy (1 big memcpy)", avg_ms, bw))

    # Summary
    print()
    print("=" * 80)
    print("  SUMMARY (D2H: GPU → CPU)")
    print("=" * 80)
    print(f"  {'Method':<50} {'ms':<10} {'GB/s':<10} {'Efficiency'}")
    print(f"  {'-'*50} {'-'*10} {'-'*10} {'-'*12}")

    best_bw = max(r[2] for r in results) if results else 1.0
    for name, ms, bw in results:
        ms_str = f"{ms:.2f}" if ms > 0 else "N/A"
        bw_str = f"{bw:.2f}" if bw > 0 else "N/A"
        eff_str = f"{bw/best_bw*100:.1f}%" if bw > 0 and best_bw > 0 else "N/A"
        print(f"  {name:<50} {ms_str:<10} {bw_str:<10} {eff_str}")

    # Diagnosis
    print()
    print("#" * 80)
    print("  DIAGNOSIS")
    print("#" * 80)
    if results:
        ce_bw = next((r[2] for r in results if "FlexKV CE" in r[0] and r[2] > 0), 0)
        best = max(r[2] for r in results)
        if ce_bw > 0:
            ratio = ce_bw / best
            if ratio > 0.8:
                print(f"  CE path efficiency: {ratio*100:.1f}% — GOOD, no significant overhead")
                print(f"  The CE path is NOT the bottleneck.")
                print(f"  If production shows lower bandwidth, the issue is elsewhere")
                print(f"  (e.g., contention with prefill compute, scheduling delays).")
            elif ratio > 0.5:
                print(f"  CE path efficiency: {ratio*100:.1f}% — MODERATE overhead")
                print(f"  Some overhead from 78 cudaMemcpyAsync calls, but not catastrophic.")
            else:
                print(f"  CE path efficiency: {ratio*100:.1f}% — SIGNIFICANT overhead!")
                print(f"  The CE path has major performance issues.")
    print()


if __name__ == "__main__":
    main()
