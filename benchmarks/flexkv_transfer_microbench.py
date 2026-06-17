#!/usr/bin/env python3
"""
FlexKV transfer.cu microbenchmark.

Run inside a pod with flexkv_env activated. It directly calls flexkv.c_ext.transfer_kv_blocks
so it can trigger transfer.cu path0/path1/path2 without going through SGLang scheduler.

Examples:
  python3 flexkv_transfer_microbench.py --direction d2h --pattern path1 --segments 2 --iters 20
  XSGL_TRANSFER_MERGED=1 python3 flexkv_transfer_microbench.py --direction d2h --pattern path1 --segments 2 --iters 20
  FLEXKV_TRANSFER_FORCE_PATH=2 python3 flexkv_transfer_microbench.py --direction d2h --pattern scatter --iters 20
"""
import argparse
import os
import re
import statistics
import time
from typing import List, Tuple

import torch

try:
    from flexkv.c_ext import transfer_kv_blocks
except Exception as e:
    raise SystemExit(f"Failed to import flexkv.c_ext.transfer_kv_blocks: {e}")


def make_ids(pattern: str, n: int, segments: int, gap: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (gpu_ids_cpu, cpu_ids_cpu) int64 host tensors."""
    if pattern == "path0":
        gpu = list(range(n))
        cpu = list(range(n))
    elif pattern == "path1":
        # multiple simultaneously contiguous segments: [0..run), [gap..gap+run), ...
        segments = max(1, min(segments, n))
        base = n // segments
        rem = n % segments
        gpu, cpu = [], []
        g0 = 0
        c0 = 0
        for s in range(segments):
            run = base + (1 if s < rem else 0)
            gpu.extend(range(g0, g0 + run))
            cpu.extend(range(c0, c0 + run))
            g0 += run + gap
            c0 += run + gap
    elif pattern == "scatter":
        # src contiguous, dst non-contiguous -> path2 D2H staging+CPU scatter
        gpu = list(range(n))
        cpu = [i * (gap + 1) for i in range(n)]
    elif pattern == "gather":
        # src non-contiguous, dst contiguous -> path2 GPU gather + direct D2H
        gpu = [i * (gap + 1) for i in range(n)]
        cpu = list(range(n))
    elif pattern == "both_sparse":
        gpu = [i * (gap + 1) for i in range(n)]
        cpu = [i * (gap + 2) for i in range(n)]
    else:
        raise ValueError(f"unknown pattern={pattern}")
    return torch.tensor(gpu, dtype=torch.int64), torch.tensor(cpu, dtype=torch.int64)


def q(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    idx = min(len(xs) - 1, max(0, int(len(xs) * p)))
    return xs[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["d2h", "h2d"], default="d2h")
    ap.add_argument("--pattern", choices=["path0", "path1", "scatter", "gather", "both_sparse"], default="path1")
    ap.add_argument("--num-layers", type=int, default=78)
    ap.add_argument("--num-blocks", type=int, default=512, help="blocks transferred per op")
    ap.add_argument("--segments", type=int, default=2, help="for pattern=path1")
    ap.add_argument("--gap", type=int, default=8)
    ap.add_argument("--chunk-size-bytes", type=int, default=16384)
    ap.add_argument("--gpu-blocks", type=int, default=None)
    ap.add_argument("--cpu-blocks", type=int, default=None)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--transfer-num-cta", type=int, default=4)
    ap.add_argument("--force-path", type=int, choices=[0, 1, 2], default=None)
    ap.add_argument("--merged", action="store_true", help="set XSGL_TRANSFER_MERGED=1 before import/use")
    ap.add_argument("--threshold", type=int, default=None, help="set XSGL_TRANSFER_SEGMENT_THRESHOLD")
    ap.add_argument("--no-ce", action="store_true", help="use custom CUDA kernel instead of CE path")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    if args.force_path is not None:
        os.environ["FLEXKV_TRANSFER_FORCE_PATH"] = str(args.force_path)
    if args.merged:
        os.environ["XSGL_TRANSFER_MERGED"] = "1"
    if args.threshold is not None:
        os.environ["XSGL_TRANSFER_SEGMENT_THRESHOLD"] = str(args.threshold)

    assert args.chunk_size_bytes % 8 == 0
    elems_per_block = args.chunk_size_bytes // 8
    gpu_ids_cpu, cpu_ids_cpu = make_ids(args.pattern, args.num_blocks, args.segments, args.gap)
    max_gpu = int(gpu_ids_cpu.max().item()) if gpu_ids_cpu.numel() else 0
    max_cpu = int(cpu_ids_cpu.max().item()) if cpu_ids_cpu.numel() else 0
    gpu_blocks = args.gpu_blocks or (max_gpu + 1)
    cpu_blocks = args.cpu_blocks or (max_cpu + 1)
    if gpu_blocks <= max_gpu or cpu_blocks <= max_cpu:
        raise SystemExit(f"Need gpu_blocks>{max_gpu}, cpu_blocks>{max_cpu}; got {gpu_blocks}, {cpu_blocks}")

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")

    # SGLang layout: a list of per-layer tensors. For MLA kv_dim=1, only num_layers ptrs are used.
    gpu_layers: List[torch.Tensor] = [
        torch.empty((gpu_blocks, elems_per_block), dtype=torch.int64, device=device)
        for _ in range(args.num_layers)
    ]
    cpu_tensor = torch.empty((args.num_layers, 1, cpu_blocks, elems_per_block), dtype=torch.int64, device="cpu")

    # Touch memory to fault-in. Do not spend too much time filling huge tensors.
    for t in gpu_layers[: min(2, len(gpu_layers))]:
        t.zero_()
    cpu_tensor.view(-1)[: min(cpu_tensor.numel(), 1024)].zero_()
    torch.cuda.synchronize()

    ptrs = torch.tensor([t.data_ptr() for t in gpu_layers], dtype=torch.int64)
    gpu_ids = gpu_ids_cpu.contiguous()
    cpu_ids = cpu_ids_cpu.contiguous()

    gpu_block_stride = args.chunk_size_bytes
    gpu_kv_stride = 0
    gpu_layer_stride = 0
    cpu_block_stride = args.chunk_size_bytes
    cpu_kv_stride = cpu_blocks * args.chunk_size_bytes
    cpu_layer_stride = cpu_kv_stride
    is_h2d = args.direction == "h2d"
    use_ce = not args.no_ce

    print("=== FlexKV transfer microbench ===")
    print(f"direction={args.direction} pattern={args.pattern} num_blocks={args.num_blocks} segments={args.segments} gap={args.gap}")
    print(f"layers={args.num_layers} chunk={args.chunk_size_bytes}B gpu_blocks={gpu_blocks} cpu_blocks={cpu_blocks}")
    print(f"env XSGL_TRANSFER_SEGMENT_THRESHOLD={os.getenv('XSGL_TRANSFER_SEGMENT_THRESHOLD')} XSGL_TRANSFER_MERGED={os.getenv('XSGL_TRANSFER_MERGED')} FLEXKV_TRANSFER_FORCE_PATH={os.getenv('FLEXKV_TRANSFER_FORCE_PATH')}")
    print(f"per_op_bytes={args.num_layers * args.num_blocks * args.chunk_size_bytes / (1024**3):.3f} GiB")

    def one_iter():
        t0 = time.perf_counter()
        transfer_kv_blocks(
            gpu_ids,
            ptrs,
            gpu_kv_stride,
            gpu_block_stride,
            gpu_layer_stride,
            cpu_ids,
            cpu_tensor,
            cpu_kv_stride,
            cpu_layer_stride,
            cpu_block_stride,
            args.chunk_size_bytes,
            0,
            args.num_layers,
            args.transfer_num_cta,
            is_h2d,
            use_ce,
            True,   # is_mla
            2,      # gpu_block_type=SGLANG
            True,   # sync
        )
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000

    for _ in range(args.warmup):
        one_iter()

    times = []
    for i in range(args.iters):
        ms = one_iter()
        times.append(ms)
        print(f"iter={i} time_ms={ms:.3f}")

    total_gib = args.num_layers * args.num_blocks * args.chunk_size_bytes / (1024**3)
    print("=== summary ===")
    print(f"count={len(times)} min={min(times):.3f}ms p50={q(times,0.5):.3f}ms p90={q(times,0.9):.3f}ms max={max(times):.3f}ms mean={statistics.mean(times):.3f}ms")
    print(f"effective_bw_GiB_s mean={total_gib/(statistics.mean(times)/1000):.3f} p50={total_gib/(q(times,0.5)/1000):.3f}")


if __name__ == "__main__":
    main()
