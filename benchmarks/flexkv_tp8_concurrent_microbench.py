#!/usr/bin/env python3
"""
Concurrent FlexKV TPTransferThreadGroup 8-GPU microbenchmark.

This script stresses TPTransferThreadGroup with multiple concurrent callers.
It has two modes:
  - thread: one shared TPTransferThreadGroup, N Python threads call tp_group_transfer.
            Best model for testing queue/stream contention inside one worker.
            Note: if pybind keeps GIL, this may serialize at Python boundary.
  - process: N processes, each creates its own TPTransferThreadGroup and uses 8 GPUs.
             Best model for service-like multi-process contention, but uses more memory.

Run inside a pod with 8 idle P800 cards:
  source /root/miniconda/etc/profile.d/conda.sh && conda activate flexkv_env
  python3 flexkv_tp8_concurrent_microbench.py --mode thread --concurrency 4 --pattern path1 --segments 2

Useful envs:
  XSGL_TRANSFER_MERGED=1
  XSGL_TRANSFER_SEGMENT_THRESHOLD=8
  FLEXKV_INNER_TIMING=1
  FLEXKV_PATH_TIMING=1
"""
import argparse
import multiprocessing as mp
import os
import queue
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(len(xs) * p)))]


@dataclass
class BenchConfig:
    pattern: str
    num_gpus: int
    num_layers: int
    num_blocks: int
    segments: int
    gap: int
    shard_bytes: int
    gpu_blocks: int | None
    cpu_blocks: int | None
    warmup: int
    iters: int
    transfer_num_cta: int
    no_pin_cpu: bool
    touch_all: bool


def build_group(cfg: BenchConfig):
    assert cfg.shard_bytes % 8 == 0
    total_chunk_bytes = cfg.shard_bytes * cfg.num_gpus
    elems_per_gpu_block = total_chunk_bytes // 8
    elems_per_cpu_block = total_chunk_bytes // 8

    gpu_ids_cpu, cpu_ids_cpu = make_ids(cfg.pattern, cfg.num_blocks, cfg.segments, cfg.gap)
    max_gpu = int(gpu_ids_cpu.max()) if gpu_ids_cpu.numel() else 0
    max_cpu = int(cpu_ids_cpu.max()) if cpu_ids_cpu.numel() else 0
    gpu_blocks = cfg.gpu_blocks or (max_gpu + 1)
    cpu_blocks = cfg.cpu_blocks or (max_cpu + 1)

    all_gpu_tensors: List[List[torch.Tensor]] = []
    gpu_ptrs_flat: List[int] = []
    gpu_device_ids: List[int] = []

    for gid in range(cfg.num_gpus):
        torch.cuda.set_device(gid)
        dev = torch.device(f"cuda:{gid}")
        tensors = []
        # SGLANG backend requires num_tensors_per_gpu == num_layers * 2.
        # kv_idx=0 is used for MLA; kv_idx=1 is not used. Allocate tiny dummy tensors for kv_idx=1.
        for layer in range(cfg.num_layers):
            t = torch.empty((gpu_blocks, elems_per_gpu_block), dtype=torch.int64, device=dev)
            tensors.append(t)
            gpu_ptrs_flat.append(t.data_ptr())
        for layer in range(cfg.num_layers):
            t = torch.empty((1, elems_per_gpu_block), dtype=torch.int64, device=dev)
            tensors.append(t)
            gpu_ptrs_flat.append(t.data_ptr())
        all_gpu_tensors.append(tensors)
        gpu_device_ids.append(gid)

    cpu_tensor = torch.empty(
        (cfg.num_layers, 1, cpu_blocks, elems_per_cpu_block),
        dtype=torch.int64,
        device="cpu",
        pin_memory=(not cfg.no_pin_cpu),
    )

    if cfg.touch_all:
        for tensors in all_gpu_tensors:
            for t in tensors[: cfg.num_layers]:
                t.zero_()
        cpu_tensor.zero_()
    else:
        for tensors in all_gpu_tensors:
            tensors[0].view(-1)[:1024].zero_()
        cpu_tensor.view(-1)[:1024].zero_()
    for gid in range(cfg.num_gpus):
        torch.cuda.set_device(gid)
        torch.cuda.synchronize()

    group = TPTransferThreadGroup(
        cfg.num_gpus,
        gpu_ptrs_flat,
        cfg.num_layers * 2,
        cpu_tensor.data_ptr(),
        cfg.num_layers,
        [0 for _ in range(cfg.num_gpus)],
        [total_chunk_bytes for _ in range(cfg.num_gpus)],
        [0 for _ in range(cfg.num_gpus)],
        [total_chunk_bytes for _ in range(cfg.num_gpus)],
        gpu_device_ids,
    )

    cpu_kv_stride = cpu_blocks * total_chunk_bytes
    cpu_layer_stride = cpu_kv_stride
    cpu_block_stride = total_chunk_bytes
    cpu_tp_stride = cfg.shard_bytes
    bytes_per_op = cfg.num_layers * cfg.num_blocks * cfg.shard_bytes * cfg.num_gpus

    return group, gpu_ids_cpu.contiguous(), cpu_ids_cpu.contiguous(), cpu_kv_stride, cpu_layer_stride, cpu_block_stride, cpu_tp_stride, bytes_per_op


def sync_all(num_gpus: int):
    for gid in range(num_gpus):
        torch.cuda.set_device(gid)
        torch.cuda.synchronize()


def run_one(group, gpu_ids, cpu_ids, cfg: BenchConfig, cpu_kv_stride, cpu_layer_stride, cpu_block_stride, cpu_tp_stride):
    t0 = time.perf_counter()
    group.tp_group_transfer(
        gpu_ids,
        cpu_ids,
        cpu_kv_stride,
        cpu_layer_stride,
        cpu_block_stride,
        cpu_tp_stride,
        cfg.transfer_num_cta,
        False,  # D2H only
        True,   # CE
        0,
        cfg.num_layers,
        True,   # MLA
    )
    sync_all(cfg.num_gpus)
    return (time.perf_counter() - t0) * 1000


def worker_thread(worker_id, group, gpu_ids, cpu_ids, cfg: BenchConfig, strides, start_barrier):
    cpu_kv_stride, cpu_layer_stride, cpu_block_stride, cpu_tp_stride = strides
    start_barrier.wait()
    times = []
    for i in range(cfg.iters):
        ms = run_one(group, gpu_ids, cpu_ids, cfg, cpu_kv_stride, cpu_layer_stride, cpu_block_stride, cpu_tp_stride)
        times.append(ms)
        print(f"worker={worker_id} iter={i} time_ms={ms:.3f}", flush=True)
    return times


def run_thread_mode(cfg: BenchConfig, concurrency: int):
    group, gpu_ids, cpu_ids, cpu_kv_stride, cpu_layer_stride, cpu_block_stride, cpu_tp_stride, bytes_per_op = build_group(cfg)
    strides = (cpu_kv_stride, cpu_layer_stride, cpu_block_stride, cpu_tp_stride)
    for _ in range(cfg.warmup):
        run_one(group, gpu_ids, cpu_ids, cfg, *strides)

    print("=== concurrent thread run ===", flush=True)
    start_barrier = threading.Barrier(concurrency)
    t0 = time.perf_counter()
    all_times = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(worker_thread, wid, group, gpu_ids, cpu_ids, cfg, strides, start_barrier) for wid in range(concurrency)]
        for fut in as_completed(futs):
            all_times.extend(fut.result())
    wall_s = time.perf_counter() - t0
    return all_times, wall_s, bytes_per_op


def process_entry(rank: int, cfg: BenchConfig, q):
    try:
        # Keep CUDA contexts independent per process.
        group, gpu_ids, cpu_ids, cpu_kv_stride, cpu_layer_stride, cpu_block_stride, cpu_tp_stride, bytes_per_op = build_group(cfg)
        for _ in range(cfg.warmup):
            run_one(group, gpu_ids, cpu_ids, cfg, cpu_kv_stride, cpu_layer_stride, cpu_block_stride, cpu_tp_stride)
        times = []
        for i in range(cfg.iters):
            ms = run_one(group, gpu_ids, cpu_ids, cfg, cpu_kv_stride, cpu_layer_stride, cpu_block_stride, cpu_tp_stride)
            times.append(ms)
            print(f"process={rank} iter={i} time_ms={ms:.3f}", flush=True)
        q.put((rank, times, bytes_per_op, None))
    except Exception as e:
        q.put((rank, [], 0, repr(e)))


def run_process_mode(cfg: BenchConfig, concurrency: int):
    q = mp.Queue()
    ps = [mp.Process(target=process_entry, args=(i, cfg, q)) for i in range(concurrency)]
    t0 = time.perf_counter()
    for p in ps:
        p.start()
    results = [q.get() for _ in ps]
    for p in ps:
        p.join()
    wall_s = time.perf_counter() - t0
    all_times = []
    bytes_per_op = 0
    for rank, times, b, err in sorted(results):
        if err:
            print(f"process={rank} ERROR {err}")
        all_times.extend(times)
        bytes_per_op = max(bytes_per_op, b)
    return all_times, wall_s, bytes_per_op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["thread", "process"], default="thread")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--pattern", choices=["path0", "path1", "scatter", "gather", "both_sparse"], default="path1")
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=78)
    ap.add_argument("--num-blocks", type=int, default=512)
    ap.add_argument("--segments", type=int, default=2)
    ap.add_argument("--gap", type=int, default=8)
    ap.add_argument("--shard-bytes", type=int, default=16384)
    ap.add_argument("--gpu-blocks", type=int, default=None)
    ap.add_argument("--cpu-blocks", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10, help="iters per worker")
    ap.add_argument("--transfer-num-cta", type=int, default=4)
    ap.add_argument("--threshold", type=int, default=None)
    ap.add_argument("--merged", action="store_true")
    ap.add_argument("--force-path", type=int, choices=[0,1,2], default=None)
    ap.add_argument("--no-pin-cpu", action="store_true")
    ap.add_argument("--touch-all", action="store_true")
    args = ap.parse_args()

    if args.threshold is not None:
        os.environ["XSGL_TRANSFER_SEGMENT_THRESHOLD"] = str(args.threshold)
    if args.merged:
        os.environ["XSGL_TRANSFER_MERGED"] = "1"
    if args.force_path is not None:
        os.environ["FLEXKV_TRANSFER_FORCE_PATH"] = str(args.force_path)

    cfg = BenchConfig(
        pattern=args.pattern,
        num_gpus=args.num_gpus,
        num_layers=args.num_layers,
        num_blocks=args.num_blocks,
        segments=args.segments,
        gap=args.gap,
        shard_bytes=args.shard_bytes,
        gpu_blocks=args.gpu_blocks,
        cpu_blocks=args.cpu_blocks,
        warmup=args.warmup,
        iters=args.iters,
        transfer_num_cta=args.transfer_num_cta,
        no_pin_cpu=args.no_pin_cpu,
        touch_all=args.touch_all,
    )

    print("=== FlexKV concurrent TP8 transfer microbench ===")
    print(f"mode={args.mode} concurrency={args.concurrency} pattern={args.pattern} iters_per_worker={args.iters}")
    print(f"gpus={args.num_gpus} layers={args.num_layers} blocks={args.num_blocks} segments={args.segments} shard_bytes={args.shard_bytes}")
    print(f"env XSGL_TRANSFER_SEGMENT_THRESHOLD={os.getenv('XSGL_TRANSFER_SEGMENT_THRESHOLD')} XSGL_TRANSFER_MERGED={os.getenv('XSGL_TRANSFER_MERGED')} FLEXKV_TRANSFER_FORCE_PATH={os.getenv('FLEXKV_TRANSFER_FORCE_PATH')}")

    if args.mode == "thread":
        times, wall_s, bytes_per_op = run_thread_mode(cfg, args.concurrency)
    else:
        times, wall_s, bytes_per_op = run_process_mode(cfg, args.concurrency)

    total_ops = len(times)
    total_gib = total_ops * bytes_per_op / (1024**3)
    print("=== summary ===")
    print(f"ops={total_ops} wall={wall_s:.3f}s ops_per_s={total_ops/wall_s:.3f} total_GiB={total_gib:.3f} aggregate_GiB_s={total_gib/wall_s:.3f}")
    if times:
        print(f"latency_ms min={min(times):.3f} p50={quantile(times,0.5):.3f} p90={quantile(times,0.9):.3f} p99={quantile(times,0.99):.3f} max={max(times):.3f} mean={statistics.mean(times):.3f}")


if __name__ == "__main__":
    main()
