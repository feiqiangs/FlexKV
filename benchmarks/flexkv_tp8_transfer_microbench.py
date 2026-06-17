#!/usr/bin/env python3
"""
FlexKV TPTransferThreadGroup 8-GPU microbenchmark.

Purpose:
  Exercise the same C++ path as FlexKV's real main-KV worker:
    TPTransferThreadGroup.tp_group_transfer(...)
  This covers 8 GPU worker threads, per-GPU streams, path0/path1/path2 selection,
  and the [INNER]/[PATH1] timing instrumentation in C++.

Run inside one P800 pod with all 8 XPUs free:
  source /root/miniconda/etc/profile.d/conda.sh && conda activate flexkv_env
  python3 /workspace/zittozhang/flexkv_tp8_transfer_microbench.py --pattern path1 --segments 2

Important:
  For D2H MLA, TPTransferThreadGroup shards one logical CPU chunk across 8 GPUs.
  Therefore this benchmark allocates each GPU block with total_chunk_bytes and
  transfers shard_bytes per GPU using the same enable_sharded_d2h branch.
"""
import argparse
import os
import statistics
import time
from typing import List, Tuple

import torch

try:
    from flexkv.c_ext import TPTransferThreadGroup
except Exception as e:
    raise SystemExit(f"Failed to import TPTransferThreadGroup: {e}")


def make_ids(pattern: str, n: int, segments: int, gap: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if pattern == "path0":
        gpu = list(range(n)); cpu = list(range(n))
    elif pattern == "path1":
        segments = max(1, min(segments, n))
        base = n // segments
        rem = n % segments
        gpu, cpu = [], []
        g0 = 0; c0 = 0
        for s in range(segments):
            run = base + (1 if s < rem else 0)
            gpu.extend(range(g0, g0 + run))
            cpu.extend(range(c0, c0 + run))
            g0 += run + gap
            c0 += run + gap
    elif pattern == "scatter":
        gpu = list(range(n))
        cpu = [i * (gap + 1) for i in range(n)]
    elif pattern == "gather":
        gpu = [i * (gap + 1) for i in range(n)]
        cpu = list(range(n))
    elif pattern == "both_sparse":
        gpu = [i * (gap + 1) for i in range(n)]
        cpu = [i * (gap + 2) for i in range(n)]
    else:
        raise ValueError(pattern)
    return torch.tensor(gpu, dtype=torch.int64), torch.tensor(cpu, dtype=torch.int64)


def quantile(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    return xs[min(len(xs)-1, max(0, int(len(xs) * p)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["d2h"], default="d2h", help="Only D2H is currently modeled exactly like FlexKV MLA sharded path")
    ap.add_argument("--pattern", choices=["path0", "path1", "scatter", "gather", "both_sparse"], default="path1")
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=78)
    ap.add_argument("--num-blocks", type=int, default=512)
    ap.add_argument("--segments", type=int, default=2)
    ap.add_argument("--gap", type=int, default=8)
    ap.add_argument("--shard-bytes", type=int, default=16384, help="bytes transferred per GPU per block; total CPU block bytes = shard_bytes*num_gpus")
    ap.add_argument("--gpu-blocks", type=int, default=None)
    ap.add_argument("--cpu-blocks", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--transfer-num-cta", type=int, default=4)
    ap.add_argument("--threshold", type=int, default=None)
    ap.add_argument("--merged", action="store_true")
    ap.add_argument("--force-path", type=int, choices=[0,1,2], default=None)
    ap.add_argument("--no-pin-cpu", action="store_true")
    ap.add_argument("--touch-all", action="store_true", help="zero all tensors up front; slower but avoids first-touch effects")
    args = ap.parse_args()

    if args.threshold is not None:
        os.environ["XSGL_TRANSFER_SEGMENT_THRESHOLD"] = str(args.threshold)
    if args.merged:
        os.environ["XSGL_TRANSFER_MERGED"] = "1"
    if args.force_path is not None:
        os.environ["FLEXKV_TRANSFER_FORCE_PATH"] = str(args.force_path)

    assert args.shard_bytes % 8 == 0
    total_chunk_bytes = args.shard_bytes * args.num_gpus
    elems_per_gpu_block = total_chunk_bytes // 8
    elems_per_cpu_block = total_chunk_bytes // 8

    gpu_ids_cpu, cpu_ids_cpu = make_ids(args.pattern, args.num_blocks, args.segments, args.gap)
    max_gpu = int(gpu_ids_cpu.max()) if gpu_ids_cpu.numel() else 0
    max_cpu = int(cpu_ids_cpu.max()) if cpu_ids_cpu.numel() else 0
    gpu_blocks = args.gpu_blocks or (max_gpu + 1)
    cpu_blocks = args.cpu_blocks or (max_cpu + 1)

    print("=== FlexKV TP8 transfer microbench ===")
    print(f"pattern={args.pattern} num_gpus={args.num_gpus} blocks={args.num_blocks} segments={args.segments} gap={args.gap}")
    print(f"layers={args.num_layers} shard_bytes={args.shard_bytes} total_chunk_bytes={total_chunk_bytes}")
    print(f"gpu_blocks={gpu_blocks} cpu_blocks={cpu_blocks} pin_cpu={not args.no_pin_cpu}")
    print(f"env XSGL_TRANSFER_SEGMENT_THRESHOLD={os.getenv('XSGL_TRANSFER_SEGMENT_THRESHOLD')} XSGL_TRANSFER_MERGED={os.getenv('XSGL_TRANSFER_MERGED')} FLEXKV_TRANSFER_FORCE_PATH={os.getenv('FLEXKV_TRANSFER_FORCE_PATH')}")

    # Create per-GPU SGLang style ptr table: num_tensors_per_gpu = num_layers*2.
    # kv_idx=0 is used for MLA, kv_idx=1 is allocated for backend_type=SGLANG detection.
    all_gpu_tensors: List[List[torch.Tensor]] = []
    gpu_ptrs_flat: List[int] = []
    gpu_device_ids: List[int] = []
    for gid in range(args.num_gpus):
        torch.cuda.set_device(gid)
        dev = torch.device(f"cuda:{gid}")
        tensors = []
        for _ in range(args.num_layers * 2):
            t = torch.empty((gpu_blocks, elems_per_gpu_block), dtype=torch.int64, device=dev)
            tensors.append(t)
            gpu_ptrs_flat.append(t.data_ptr())
        all_gpu_tensors.append(tensors)
        gpu_device_ids.append(gid)

    cpu_tensor = torch.empty(
        (args.num_layers, 1, cpu_blocks, elems_per_cpu_block),
        dtype=torch.int64,
        device="cpu",
        pin_memory=(not args.no_pin_cpu),
    )

    if args.touch_all:
        print("touching all gpu/cpu tensors ...")
        for tensors in all_gpu_tensors:
            for t in tensors[: args.num_layers]:
                t.zero_()
        cpu_tensor.zero_()
        for gid in range(args.num_gpus):
            torch.cuda.set_device(gid)
            torch.cuda.synchronize()
    else:
        # light touch to initialize contexts
        for tensors in all_gpu_tensors:
            tensors[0].view(-1)[:1024].zero_()
        cpu_tensor.view(-1)[:1024].zero_()
        for gid in range(args.num_gpus):
            torch.cuda.set_device(gid)
            torch.cuda.synchronize()

    group = TPTransferThreadGroup(
        args.num_gpus,
        gpu_ptrs_flat,
        args.num_layers * 2,                 # SGLANG backend
        cpu_tensor.data_ptr(),
        args.num_layers,
        [0 for _ in range(args.num_gpus)],   # gpu_kv_strides unused in SGLANG ptr_at
        [total_chunk_bytes for _ in range(args.num_gpus)],
        [0 for _ in range(args.num_gpus)],
        [total_chunk_bytes for _ in range(args.num_gpus)],
        gpu_device_ids,
    )

    gpu_ids = gpu_ids_cpu.contiguous()
    cpu_ids = cpu_ids_cpu.contiguous()
    cpu_kv_stride = cpu_blocks * total_chunk_bytes
    cpu_layer_stride = cpu_kv_stride
    cpu_block_stride = total_chunk_bytes
    cpu_tp_stride = args.shard_bytes

    bytes_per_op = args.num_layers * args.num_blocks * args.shard_bytes * args.num_gpus
    print(f"bytes_per_op={bytes_per_op/(1024**3):.3f} GiB")

    def one():
        t0 = time.perf_counter()
        group.tp_group_transfer(
            gpu_ids,
            cpu_ids,
            cpu_kv_stride,
            cpu_layer_stride,
            cpu_block_stride,
            cpu_tp_stride,
            args.transfer_num_cta,
            False,  # is_host_to_device: D2H
            True,   # use_ce_transfer
            0,
            args.num_layers,
            True,   # is_mla -> sharded D2H branch
        )
        # group waits for futures, and transfer.cu syncs when requested internally; sync all for timing safety.
        for gid in range(args.num_gpus):
            torch.cuda.set_device(gid)
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000

    for _ in range(args.warmup):
        one()
    times = []
    for i in range(args.iters):
        ms = one()
        times.append(ms)
        print(f"iter={i} time_ms={ms:.3f}")

    mean = statistics.mean(times)
    print("=== summary ===")
    print(f"count={len(times)} min={min(times):.3f}ms p50={quantile(times,0.5):.3f}ms p90={quantile(times,0.9):.3f}ms max={max(times):.3f}ms mean={mean:.3f}ms")
    print(f"effective_bw_GiB_s mean={bytes_per_op/(1024**3)/(mean/1000):.3f} p50={bytes_per_op/(1024**3)/(quantile(times,0.5)/1000):.3f}")


if __name__ == "__main__":
    main()
