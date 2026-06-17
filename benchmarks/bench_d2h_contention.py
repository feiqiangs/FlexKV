#!/usr/bin/env python3
"""
Proof: D2H bandwidth degradation under GPU compute contention.

This script proves that D2H transfer bandwidth drops significantly when
GPU compute kernels (simulating prefill attention/MLP) are running concurrently.

Experiment design:
  1. D2H alone (no compute) — baseline bandwidth
  2. D2H + light compute (small matmul on separate stream)
  3. D2H + heavy compute (large matmul simulating prefill, on separate stream)
  4. D2H + sustained compute (continuous matmul loop)

If PCIe bandwidth contention is the root cause, we expect:
  - Test 1: ~18 GB/s (matches our standalone benchmark)
  - Test 2-4: significantly lower (matching production's ~5.5 GB/s or ~1 GB/s)

Usage:
    python bench_d2h_contention.py [--num-tokens 24064] [--warmup 2] [--repeat 5]
"""

import argparse
import os
import sys
import time
import threading
from typing import Tuple

import torch

# ============================================================================
# Constants matching production (GLM5 on P800)
# ============================================================================
NUM_LAYERS = 78
KV_DIM = 576
DTYPE = torch.bfloat16
ITEM_SIZE = 1 * KV_DIM * 2  # 1152 bytes per token (bf16)


def cuda_host_register(tensor: torch.Tensor) -> None:
    """Register CPU tensor as pinned memory."""
    ptr = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()
    err = torch.cuda.cudart().cudaHostRegister(ptr, size, 0)
    if isinstance(err, tuple):
        err = err[0]
    if err != 0:
        raise RuntimeError(f"cudaHostRegister failed: err={err}")


def prepare_d2h_buffers(num_tokens: int):
    """Prepare GPU and CPU buffers for D2H transfer."""
    gpu_layers = []
    for _ in range(NUM_LAYERS):
        t = torch.randn(num_tokens + 128, 1, KV_DIM, dtype=DTYPE, device="cuda:0")
        gpu_layers.append(t)

    cpu_flat = torch.empty(NUM_LAYERS * (num_tokens + 128) * KV_DIM, dtype=DTYPE, device="cpu")
    cpu_flat.share_memory_()
    cuda_host_register(cpu_flat)

    cpu_layers = []
    layer_size = (num_tokens + 128) * KV_DIM
    for i in range(NUM_LAYERS):
        t = cpu_flat[i * layer_size:(i + 1) * layer_size].reshape(num_tokens + 128, 1, KV_DIM)
        cpu_layers.append(t)

    return gpu_layers, cpu_layers, cpu_flat


def do_d2h_transfer(gpu_layers, cpu_layers, num_tokens, stream=None):
    """Perform D2H transfer on given stream (or default stream)."""
    if stream is not None:
        with torch.cuda.stream(stream):
            for i in range(NUM_LAYERS):
                cpu_layers[i][:num_tokens].copy_(gpu_layers[i][:num_tokens], non_blocking=True)
        stream.synchronize()
    else:
        for i in range(NUM_LAYERS):
            cpu_layers[i][:num_tokens].copy_(gpu_layers[i][:num_tokens], non_blocking=True)
        torch.cuda.synchronize()


def bench_d2h_only(gpu_layers, cpu_layers, num_tokens, warmup, repeat):
    """Benchmark D2H transfer alone (no compute contention)."""
    total_bytes = num_tokens * ITEM_SIZE * NUM_LAYERS

    # Warmup
    for _ in range(warmup):
        do_d2h_transfer(gpu_layers, cpu_layers, num_tokens)

    # Measure
    times_ms = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        do_d2h_transfer(gpu_layers, cpu_layers, num_tokens)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)

    avg_ms = sum(times_ms) / len(times_ms)
    bw = (total_bytes / (1024**3)) / (avg_ms / 1000)
    return avg_ms, bw, times_ms


def bench_d2h_with_compute(gpu_layers, cpu_layers, num_tokens, warmup, repeat,
                           matmul_m, matmul_n, matmul_k, compute_iters=10):
    """
    Benchmark D2H transfer while GPU compute is running on a separate stream.

    The compute stream runs repeated matmuls to simulate prefill attention/MLP.
    """
    total_bytes = num_tokens * ITEM_SIZE * NUM_LAYERS

    # Prepare compute workload
    A = torch.randn(matmul_m, matmul_k, dtype=DTYPE, device="cuda:0")
    B = torch.randn(matmul_k, matmul_n, dtype=DTYPE, device="cuda:0")

    compute_stream = torch.cuda.Stream()
    d2h_stream = torch.cuda.Stream()

    def launch_compute():
        """Launch sustained compute on compute_stream."""
        with torch.cuda.stream(compute_stream):
            for _ in range(compute_iters):
                torch.mm(A, B)

    # Warmup
    for _ in range(warmup):
        launch_compute()
        do_d2h_transfer(gpu_layers, cpu_layers, num_tokens, stream=d2h_stream)
        compute_stream.synchronize()

    # Measure: launch compute first, then immediately start D2H
    times_ms = []
    for _ in range(repeat):
        torch.cuda.synchronize()

        # Launch compute (will keep GPU busy)
        launch_compute()

        # Immediately start D2H on separate stream
        t0 = time.perf_counter()
        do_d2h_transfer(gpu_layers, cpu_layers, num_tokens, stream=d2h_stream)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)

        # Wait for compute to finish
        compute_stream.synchronize()

    avg_ms = sum(times_ms) / len(times_ms)
    bw = (total_bytes / (1024**3)) / (avg_ms / 1000)
    return avg_ms, bw, times_ms


def bench_d2h_with_sustained_compute(gpu_layers, cpu_layers, num_tokens, warmup, repeat,
                                     matmul_m, matmul_n, matmul_k):
    """
    Benchmark D2H while a background thread continuously runs compute.
    This most closely simulates the production scenario where prefill
    is always running.
    """
    total_bytes = num_tokens * ITEM_SIZE * NUM_LAYERS

    A = torch.randn(matmul_m, matmul_k, dtype=DTYPE, device="cuda:0")
    B = torch.randn(matmul_k, matmul_n, dtype=DTYPE, device="cuda:0")

    compute_stream = torch.cuda.Stream()
    d2h_stream = torch.cuda.Stream()

    stop_compute = threading.Event()

    def compute_loop():
        """Continuously run matmuls until stopped."""
        with torch.cuda.stream(compute_stream):
            while not stop_compute.is_set():
                for _ in range(5):
                    torch.mm(A, B)
                # Small sleep to avoid Python GIL starvation
                # but keep GPU busy
                compute_stream.synchronize()

    # Start background compute
    stop_compute.clear()
    compute_thread = threading.Thread(target=compute_loop, daemon=True)
    compute_thread.start()

    # Give compute a moment to start
    time.sleep(0.5)

    # Warmup
    for _ in range(warmup):
        do_d2h_transfer(gpu_layers, cpu_layers, num_tokens, stream=d2h_stream)

    # Measure
    times_ms = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        do_d2h_transfer(gpu_layers, cpu_layers, num_tokens, stream=d2h_stream)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)

    # Stop compute
    stop_compute.set()
    compute_thread.join(timeout=5)

    avg_ms = sum(times_ms) / len(times_ms)
    bw = (total_bytes / (1024**3)) / (avg_ms / 1000)
    return avg_ms, bw, times_ms


def bench_d2h_default_stream_with_compute(gpu_layers, cpu_layers, num_tokens, warmup, repeat,
                                          matmul_m, matmul_n, matmul_k, compute_iters=10):
    """
    Benchmark D2H on DEFAULT stream while compute runs on a SEPARATE stream.
    This simulates the use_stream=False scenario in FlexKV where D2H
    uses the default stream but compute is on another stream.
    """
    total_bytes = num_tokens * ITEM_SIZE * NUM_LAYERS

    A = torch.randn(matmul_m, matmul_k, dtype=DTYPE, device="cuda:0")
    B = torch.randn(matmul_k, matmul_n, dtype=DTYPE, device="cuda:0")

    compute_stream = torch.cuda.Stream()

    def launch_compute():
        with torch.cuda.stream(compute_stream):
            for _ in range(compute_iters):
                torch.mm(A, B)

    # Warmup
    for _ in range(warmup):
        launch_compute()
        do_d2h_transfer(gpu_layers, cpu_layers, num_tokens, stream=None)
        compute_stream.synchronize()

    # Measure
    times_ms = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        launch_compute()

        t0 = time.perf_counter()
        do_d2h_transfer(gpu_layers, cpu_layers, num_tokens, stream=None)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)

        compute_stream.synchronize()

    avg_ms = sum(times_ms) / len(times_ms)
    bw = (total_bytes / (1024**3)) / (avg_ms / 1000)
    return avg_ms, bw, times_ms


def main():
    parser = argparse.ArgumentParser(description="Prove D2H bandwidth contention with GPU compute")
    parser.add_argument("--num-tokens", type=int, default=24064)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    total_bytes = args.num_tokens * ITEM_SIZE * NUM_LAYERS
    total_gb = total_bytes / (1024**3)

    print("=" * 80)
    print("  PROOF: D2H Bandwidth Degradation Under GPU Compute Contention")
    print("=" * 80)
    print(f"  Parameters:")
    print(f"    num_tokens={args.num_tokens}, num_layers={NUM_LAYERS}")
    print(f"    total_d2h_data={total_gb:.3f} GB")
    print(f"    CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"    warmup={args.warmup}, repeat={args.repeat}")
    print()

    # Prepare buffers
    print("  Preparing buffers...")
    gpu_layers, cpu_layers, cpu_flat = prepare_d2h_buffers(args.num_tokens)
    print(f"  Done. CPU pinned={cpu_flat.is_pinned()}")
    print()

    results = []

    # ========================================================================
    # Test 1: D2H alone (baseline)
    # ========================================================================
    print("=" * 70)
    print("  [1] D2H ALONE (no compute, baseline)")
    print("=" * 70)
    avg_ms, bw, times = bench_d2h_only(
        gpu_layers, cpu_layers, args.num_tokens, args.warmup, args.repeat)
    print(f"      avg_ms={avg_ms:.2f}, bandwidth={bw:.2f} GB/s")
    print(f"      individual: {[f'{t:.1f}' for t in times]}")
    results.append(("D2H alone (baseline)", avg_ms, bw))

    # ========================================================================
    # Test 2: D2H + light compute (small matmul)
    # ========================================================================
    print()
    print("=" * 70)
    print("  [2] D2H + LIGHT COMPUTE (matmul 1024×1024, 10 iters)")
    print("      Simulates light background work")
    print("=" * 70)
    avg_ms, bw, times = bench_d2h_with_compute(
        gpu_layers, cpu_layers, args.num_tokens, args.warmup, args.repeat,
        matmul_m=1024, matmul_n=1024, matmul_k=1024, compute_iters=10)
    print(f"      avg_ms={avg_ms:.2f}, bandwidth={bw:.2f} GB/s")
    print(f"      individual: {[f'{t:.1f}' for t in times]}")
    results.append(("D2H + light compute (1024³)", avg_ms, bw))

    # ========================================================================
    # Test 3: D2H + medium compute (prefill-like matmul)
    # ========================================================================
    print()
    print("=" * 70)
    print("  [3] D2H + MEDIUM COMPUTE (matmul 4096×4096, 10 iters)")
    print("      Simulates medium prefill workload")
    print("=" * 70)
    avg_ms, bw, times = bench_d2h_with_compute(
        gpu_layers, cpu_layers, args.num_tokens, args.warmup, args.repeat,
        matmul_m=4096, matmul_n=4096, matmul_k=4096, compute_iters=10)
    print(f"      avg_ms={avg_ms:.2f}, bandwidth={bw:.2f} GB/s")
    print(f"      individual: {[f'{t:.1f}' for t in times]}")
    results.append(("D2H + medium compute (4096³)", avg_ms, bw))

    # ========================================================================
    # Test 4: D2H + heavy compute (large matmul, simulating full prefill)
    # ========================================================================
    print()
    print("=" * 70)
    print("  [4] D2H + HEAVY COMPUTE (matmul 8192×8192, 20 iters)")
    print("      Simulates heavy prefill (attention + MLP)")
    print("=" * 70)
    avg_ms, bw, times = bench_d2h_with_compute(
        gpu_layers, cpu_layers, args.num_tokens, args.warmup, args.repeat,
        matmul_m=8192, matmul_n=8192, matmul_k=8192, compute_iters=20)
    print(f"      avg_ms={avg_ms:.2f}, bandwidth={bw:.2f} GB/s")
    print(f"      individual: {[f'{t:.1f}' for t in times]}")
    results.append(("D2H + heavy compute (8192³×20)", avg_ms, bw))

    # ========================================================================
    # Test 5: D2H + very heavy compute (even larger)
    # ========================================================================
    print()
    print("=" * 70)
    print("  [5] D2H + VERY HEAVY COMPUTE (matmul 16384×16384, 10 iters)")
    print("      Simulates full-scale prefill with large batch")
    print("=" * 70)
    avg_ms, bw, times = bench_d2h_with_compute(
        gpu_layers, cpu_layers, args.num_tokens, args.warmup, args.repeat,
        matmul_m=16384, matmul_n=16384, matmul_k=4096, compute_iters=10)
    print(f"      avg_ms={avg_ms:.2f}, bandwidth={bw:.2f} GB/s")
    print(f"      individual: {[f'{t:.1f}' for t in times]}")
    results.append(("D2H + very heavy compute (16384²)", avg_ms, bw))

    # ========================================================================
    # Test 6: D2H on default stream + compute on separate stream
    # ========================================================================
    print()
    print("=" * 70)
    print("  [6] D2H (default stream) + COMPUTE (separate stream)")
    print("      Simulates FlexKV use_stream=False scenario")
    print("=" * 70)
    avg_ms, bw, times = bench_d2h_default_stream_with_compute(
        gpu_layers, cpu_layers, args.num_tokens, args.warmup, args.repeat,
        matmul_m=8192, matmul_n=8192, matmul_k=8192, compute_iters=20)
    print(f"      avg_ms={avg_ms:.2f}, bandwidth={bw:.2f} GB/s")
    print(f"      individual: {[f'{t:.1f}' for t in times]}")
    results.append(("D2H(default) + compute(separate)", avg_ms, bw))

    # ========================================================================
    # Test 7: Sustained compute background
    # ========================================================================
    print()
    print("=" * 70)
    print("  [7] D2H + SUSTAINED BACKGROUND COMPUTE (continuous matmul)")
    print("      Most realistic: compute never stops, D2H must compete")
    print("=" * 70)
    avg_ms, bw, times = bench_d2h_with_sustained_compute(
        gpu_layers, cpu_layers, args.num_tokens, args.warmup, args.repeat,
        matmul_m=8192, matmul_n=8192, matmul_k=8192)
    print(f"      avg_ms={avg_ms:.2f}, bandwidth={bw:.2f} GB/s")
    print(f"      individual: {[f'{t:.1f}' for t in times]}")
    results.append(("D2H + sustained compute (bg thread)", avg_ms, bw))

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print()
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    baseline_bw = results[0][2]
    print(f"  {'Scenario':<50} {'ms':<10} {'GB/s':<10} {'vs baseline'}")
    print(f"  {'-'*50} {'-'*10} {'-'*10} {'-'*12}")
    for name, ms, bw in results:
        ratio = bw / baseline_bw if baseline_bw > 0 else 0
        print(f"  {name:<50} {ms:.2f}{'':<4} {bw:.2f}{'':<4} {ratio*100:.1f}%")

    print()
    print("=" * 80)
    print("  CONCLUSION")
    print("=" * 80)
    degraded = [(name, bw) for name, ms, bw in results[1:] if bw < baseline_bw * 0.7]
    if degraded:
        worst_name, worst_bw = min(degraded, key=lambda x: x[1])
        print(f"""
  ✓ PROVEN: D2H bandwidth degrades under GPU compute contention.

  Baseline (D2H alone):     {baseline_bw:.2f} GB/s
  Worst (with compute):     {worst_bw:.2f} GB/s ({worst_name})
  Degradation:              {(1 - worst_bw/baseline_bw)*100:.1f}%

  This explains why production FlexKV D2H shows only ~5.5 GB/s:
  - Standalone D2H: ~18 GB/s (no contention)
  - With prefill compute: bandwidth drops due to PCIe/memory bus contention
  - The DMA engine and compute units share the same PCIe/HBM bandwidth

  Root cause: On P800, GPU compute and DMA (cudaMemcpyAsync) compete for
  the same PCIe bandwidth. When prefill attention/MLP kernels are running,
  they consume memory bandwidth that would otherwise be available for D2H.
""")
    else:
        print(f"""
  ✗ NOT PROVEN: D2H bandwidth did NOT degrade significantly under compute.
  
  Baseline: {baseline_bw:.2f} GB/s
  All tests maintained >70% of baseline bandwidth.
  
  The production slowdown may be caused by other factors:
  - Software scheduling delays
  - Lock contention
  - Python GIL
  - Other system-level interference
""")


if __name__ == "__main__":
    main()
