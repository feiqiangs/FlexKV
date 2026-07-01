"""
End-to-end correctness benchmark for FlexKV distributed KVCache over the
**mooncake-store** backend (direct mode, non-server_client_mode).

Goal
----
Prove that two separate machines really SHARE KVCache through the mooncake-store
distributed pool:

    Node A  (--mode put)  : writes KV blocks for deterministic sequences into
                            the shared mooncake-store, then stays alive (its
                            RDMA segment holds the data).
    Node B  (--mode get)  : starts with a COLD local cache and fetches the SAME
                            sequences. Because B never PUT them locally, any hit
                            can ONLY come from the shared store written by A.

Verification / observability (why this proves sharing)
------------------------------------------------------
1. Determinism: both nodes use the same ``--seed`` -> identical token_ids ->
   identical content-hash block keys (``build_key``). No coordination channel
   other than the store itself.
2. Direct store probe: a query-only ``MooncakeStoreClient`` calls
   ``batch_exists`` on the exact keys ``build_key`` produces. On Node B this
   shows the keys physically live in the shared store (written by A) BEFORE we
   even run the FlexKV GET path.
3. Cold-cache hit: Node B's local CPU/GPU cache is empty (fresh process), yet
   the FlexKV GET reports a high cache-hit ratio -> the KV was served remotely.
4. Negative control: Node B also GETs a DIFFERENT random sequence that nobody
   ever PUT -> expected ~0% hit. This rules out spurious / always-hit behavior.

mooncake-store needs NEITHER redis NOR etcd for this path:
    metadata_server = "P2PHANDSHAKE"  +  one running ``mooncake_master``.

Usage
-----
    # 0. On the master node, start the mooncake master (once):
    #    mooncake_master --enable_http_metadata_server=true \\
    #                    --http_metadata_server_port=8080 \\
    #                    --eviction_high_watermark_ratio=0.95

    # 1. Node A (PUT) — keeps running so its segment stays mounted:
    python benchmarks/dist_benchmark/benchmark_dist_direct_mooncake_store.py \\
        --config benchmarks/dist_benchmark/example_dist_direct_mooncake_store_config.yml \\
        --mode put --seed 20240701 \\
        --master-addr 10.206.0.9:50051 --local-hostname 10.206.0.9 --device-name mlx5_0

    # 2. Node B (GET) — same seed, cold cache:
    python benchmarks/dist_benchmark/benchmark_dist_direct_mooncake_store.py \\
        --config benchmarks/dist_benchmark/example_dist_direct_mooncake_store_config.yml \\
        --mode get --seed 20240701 \\
        --master-addr 10.206.0.9:50051 --local-hostname 10.206.0.13 --device-name mlx5_0

Exit code is non-zero if verification fails (CI-friendly).
"""
import argparse
import atexit
import json
import os
import signal
import sys
import tempfile
import time
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np
import torch

# Add this directory to path (mirrors benchmark_dist_direct.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flexkv.server.client import KVTPClient
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.config import (
    ModelConfig, CacheConfig, UserConfig, RankInfo,
    IndexerCacheConfig,
    update_default_config_from_user_config,
    GLOBAL_CONFIG_FROM_ENV,
)
from flexkv.common.block import SequenceMeta
from flexkv.common.debug import flexkv_logger
from flexkv.kvmanager import KVManager
from flexkv.kvtask import KVResponseStatus

flexkv_logger.set_level("INFO")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_mooncake_store_config(config_path: str, cli_overrides: dict):
    """Load config and enable the mooncake-store backend.

    Auto-generates a mooncake_store.json (consumed by MooncakeStoreConfig.from_file)
    and points FLEXKV_MOONCAKE_STORE_CONFIG_PATH at it.
    """
    import yaml

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    print(f"[INFO] Loaded config: {config}")

    # ---- mooncake-store MUST be on; kv_sharing (redis) MUST be off ----
    os.environ["FLEXKV_USE_MOONCAKE_STORE_BACKEND"] = "1"
    os.environ.pop("FLEXKV_SERVER_CLIENT_MODE", None)
    GLOBAL_CONFIG_FROM_ENV.server_client_mode = False

    model_config = ModelConfig()
    user_config = UserConfig()

    model_config.num_layers = config["num_layers"]
    model_config.num_kv_heads = config["num_kv_heads"]
    model_config.head_size = config["head_size"]
    model_config.dtype = eval(f"torch.{config['dtype']}")
    model_config.use_mla = config["use_mla"]
    model_config.tp_size = config["tp_size"]
    model_config.dp_size = config["dp_size"]

    # ---- build mooncake_store.json (fields required by from_file) ----
    # MUST happen BEFORE CacheConfig() so its __post_init__ sees the env var.
    master_addr = cli_overrides.get("master_addr") or config["mooncake_master_addr"]
    local_hostname = cli_overrides.get("local_hostname") or config["mooncake_local_hostname"]
    device_name = cli_overrides.get("device_name")
    if device_name is None:
        device_name = config.get("mooncake_device_name", "")
    seg_gb = cli_overrides.get("global_segment_size_gb")
    if seg_gb is None:
        seg_gb = config.get("mooncake_global_segment_size_gb", 4)

    mooncake_store_json = {
        "master_addr": master_addr,
        "metadata_server": config.get("mooncake_metadata_server", "P2PHANDSHAKE"),
        "protocol": config.get("mooncake_protocol", "rdma"),
        "device_name": device_name,
        "local_hostname": local_hostname,
        "global_segment_size": int(seg_gb),  # GB; from_file multiplies by 1024**3
        "enable_ssd_offload": False,
        "ssd_offload_path": None,
        "master_metrics_port": config.get("mooncake_master_metrics_port", 9003),
    }
    fd, store_cfg_path = tempfile.mkstemp(suffix=".json", prefix="mooncake_store_")
    with os.fdopen(fd, "w") as f:
        json.dump(mooncake_store_json, f, indent=2)
    os.environ["FLEXKV_MOONCAKE_STORE_CONFIG_PATH"] = store_cfg_path
    user_config.use_mooncake_store_backend = True
    user_config.mooncake_store_config_path = store_cfg_path
    print(f"[INFO] mooncake_store.json -> {store_cfg_path}")
    print(f"[INFO] mooncake_store config: {json.dumps(mooncake_store_json, indent=2)}")

    # Now safe to construct CacheConfig (FLEXKV_MOONCAKE_STORE_CONFIG_PATH is set).
    cache_config = CacheConfig()
    cache_config.tokens_per_block = config["tokens_per_block"]
    if "cpu_cache_gb" in config:
        user_config.cpu_cache_gb = config["cpu_cache_gb"]

    # ---- optional: sparse-attention indexer side-car pool (DSA/NSA, e.g. GLM5) ----
    # Enabling this makes enable_pool_specs() include PoolKind.INDEXER, allocates a
    # separate indexer CPU pool, and (with the mooncake backend) creates a pure-client
    # INDEXER MooncakeStoreTransferWorker.  MUST be set BEFORE
    # update_default_config_from_user_config() so the indexer block-size / cpu-pool
    # sizing is taken into account.
    if config.get("indexer"):
        idx = config["indexer"]
        idx_dtype = eval(f"torch.{idx.get('dtype', 'uint8')}")
        cache_config.indexer = IndexerCacheConfig(
            head_size=int(idx["head_size"]),
            num_kv_heads=int(idx.get("num_kv_heads", 1)),
            dtype=idx_dtype,
        )
        print(f"[INFO] indexer pool ENABLED: head_size={cache_config.indexer.head_size} "
              f"num_kv_heads={cache_config.indexer.num_kv_heads} dtype={cache_config.indexer.dtype}")

    update_default_config_from_user_config(
        RankInfo(model_config=model_config), cache_config, user_config
    )
    # belt-and-suspenders: ensure the flags survive
    cache_config.use_mooncake_store_backend = True
    cache_config.mooncake_store_config_path = store_cfg_path

    assert not cache_config.enable_kv_sharing, (
        "enable_kv_sharing must be OFF with mooncake-store (mutually exclusive). "
        "Do NOT set enable_p2p_cpu/ssd/3rd_remote in the config."
    )
    assert cache_config.use_mooncake_store_backend, "mooncake-store backend not enabled!"

    return model_config, cache_config, store_cfg_path


# ---------------------------------------------------------------------------
# GPU registration helpers (same shape as benchmark_dist_direct.py)
# ---------------------------------------------------------------------------
def run_tp_client(dp_client_id, tp_rank, gpu_register_port, model_config, cache_config, num_gpu_blocks):
    device_id = tp_rank + dp_client_id * model_config.tp_size
    tp_client = KVTPClient(gpu_register_port, dp_client_id, device_id)
    gpu_kv_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=model_config.num_layers,
        num_block=num_gpu_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        is_mla=model_config.use_mla,
    )
    gpu_blocks_for_tp = []
    for _ in range(model_config.num_layers):
        gpu_blocks_for_tp.append(
            torch.empty(size=tuple(gpu_kv_layout.kv_shape[1:]), dtype=model_config.dtype).cuda(device_id)
        )

    # ---- optional indexer side-car buffers (1:1 with main KV blocks) ----
    indexer_buffers = None
    indexer_layout = None
    idx_cfg = getattr(cache_config, "indexer", None)
    if idx_cfg is not None:
        indexer_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=model_config.num_layers,
            num_block=num_gpu_blocks,          # 1:1 with main KV blocks
            tokens_per_block=1,                # each page is one indivisible block
            num_head=idx_cfg.num_kv_heads,     # MLA-style: 1
            head_size=idx_cfg.head_size,       # packed per-page indexer width
            is_mla=True,
        )
        indexer_buffers = []
        for _ in range(model_config.num_layers):
            indexer_buffers.append(
                torch.empty(size=tuple(indexer_layout.kv_shape[1:]), dtype=idx_cfg.dtype).cuda(device_id)
            )

    tp_client.register_to_server(
        gpu_blocks_for_tp, gpu_kv_layout,
        indexer_buffers=indexer_buffers,
        indexer_layout=indexer_layout,
    )
    while True:
        time.sleep(1)


def shutdown_tp_clients(tp_client_processes):
    for tp_process in tp_client_processes:
        if tp_process.is_alive():
            tp_process.terminate()
            tp_process.join(timeout=5)
            if tp_process.is_alive():
                tp_process.kill()
                tp_process.join(timeout=2)


# ---------------------------------------------------------------------------
# Sequence generation & mooncake key observability
# ---------------------------------------------------------------------------
def gen_sequences(seed, batch_size, seq_len):
    """Deterministic token sequences: identical on both nodes for the same seed."""
    rng = np.random.default_rng(seed)
    seqs, slots = [], []
    for i in range(batch_size):
        seqs.append(rng.integers(0, 100000, size=(seq_len,), dtype=np.int64))
        slots.append(np.arange(i * seq_len, (i + 1) * seq_len, dtype=np.int64))
    return seqs, slots


def expected_mooncake_keys(token_ids, cache_config, kind=None):
    """Compute the exact mooncake-store keys FlexKV would use for these blocks.

    Mirrors build_key(...) with the PP/layer-range params on cache_config, so the
    probe below queries the very same namespace the transfer worker writes to.
    ``kind`` selects the pool (KV / INDEXER); defaults to the main KV pool.
    """
    from flexkv.external.mooncake_store_keys import build_key, PoolKind

    if kind is None:
        kind = PoolKind.KV
    sm = SequenceMeta(np.asarray(token_ids, dtype=np.int64), cache_config.tokens_per_block)
    hashes = [str(sm.block_hashes[i]) for i in range(sm.num_blocks)]
    keys = [
        build_key(
            h, kind,
            pp_rank=int(getattr(cache_config, "mooncake_store_pp_rank", 0) or 0),
            pp_size=int(getattr(cache_config, "mooncake_store_pp_size", 1) or 1),
            node_layer_start=int(getattr(cache_config, "mooncake_store_node_layer_start", 0) or 0),
            node_layer_end=int(getattr(cache_config, "mooncake_store_node_layer_end", 0) or 0),
            total_layers=int(getattr(cache_config, "mooncake_store_total_layers", 0) or 0),
        )
        for h in hashes
    ]
    return hashes, keys


def probe_all_pools(seqs, cache_config, tag):
    """Probe every active pool (KV + optional indexer) for all sequences.

    Returns (total_exist, total). This is what proves *indexer distributed
    sharing*: both ``<hash>_FlexKV`` and ``<hash>_FlexKV_indexer`` keys must be
    present in the shared store.
    """
    total_exist = total = 0
    for spec in cache_config.enable_pool_specs():
        for seq in seqs:
            _, keys = expected_mooncake_keys(seq, cache_config, spec.kind)
            e, t = probe_store(keys, f"{tag}:{spec.kind.value}")
            total_exist += e
            total += t
    return total_exist, total


def probe_store(keys, tag):
    """Directly query the shared store for `keys` via a query-only client.

    Returns (num_existing_prefix, total). Prints a few sample keys for eyeballing.
    This is the ground-truth observability that a key physically exists in the
    distributed pool (independent of the FlexKV GET path).
    """
    from flexkv.external.mooncake_store_utils import MooncakeStoreClient, MooncakeStoreConfig

    cfg = MooncakeStoreConfig.from_file(
        type("C", (), {"mooncake_store_config_path": os.environ["FLEXKV_MOONCAKE_STORE_CONFIG_PATH"]})()
    )
    client = MooncakeStoreClient(cfg, query_only=True)
    per_key = client.batch_exists_impl(keys)
    num_exist = sum(1 for r in per_key if r == 1)
    print(f"  [store-probe:{tag}] {num_exist}/{len(keys)} keys exist in the shared store")
    for k in keys[:3]:
        print(f"      sample key: {k}")
    return num_exist, len(keys)


# ---------------------------------------------------------------------------
# PUT / GET phases
# ---------------------------------------------------------------------------
def do_put(kvmanager, model_config, cache_config, seqs, slots):
    print("\n" + "=" * 64)
    print("  NODE A — PUT phase (writing to shared mooncake-store)")
    print("=" * 64)
    put_ids = []
    t0 = time.time()
    for seq, slot in zip(seqs, slots):
        put_ids.append(kvmanager.put_async(torch.from_numpy(seq), torch.from_numpy(slot), token_mask=None))
    result = kvmanager.wait(put_ids, completely=True)
    dt = time.time() - t0

    put_tokens = sum(
        r.return_mask.sum().item() for r in result.values()
        if r.status == KVResponseStatus.SUCCESS and r.return_mask is not None
    )
    gb = put_tokens * model_config.token_size_in_bytes / (1024 ** 3)
    print(f"  PUT: {put_tokens} tokens, {gb:.3f} GB, {dt*1000:.1f} ms, "
          f"{gb/dt if dt>0 else 0:.2f} GB/s")

    # Observability: confirm keys now exist in the shared store (all pools).
    total_exist, total = probe_all_pools(seqs, cache_config, "after-PUT")
    ok = (total_exist == total and total > 0)
    npools = len(cache_config.enable_pool_specs())
    print(f"  [verify] keys present in store after PUT: {total_exist}/{total} "
          f"across {npools} pool(s) -> {'OK' if ok else 'FAIL'}")
    return ok


def _get_hit_ratio(kvmanager, seqs, slots):
    get_ids = []
    for seq in seqs:
        task_id, _ = kvmanager.get_match(torch.from_numpy(seq), token_mask=None)
        get_ids.append(task_id)
    kvmanager.launch(get_ids, [s for s in slots], as_batch=True, layerwise_transfer=False)
    result = kvmanager.wait(get_ids)
    cached = sum(
        r.return_mask.sum().item() for r in result.values()
        if r.status == KVResponseStatus.SUCCESS and r.return_mask is not None
    )
    total = sum(len(s) for s in seqs)
    return cached, total


def do_get(kvmanager, model_config, cache_config, seqs, slots, neg_seqs, neg_slots, threshold):
    print("\n" + "=" * 64)
    print("  NODE B — GET phase (COLD local cache; hits must come from remote)")
    print("=" * 64)

    # (1) Pre-GET store probe: keys written by A should already be visible here
    #     (main KV + indexer pools).
    pre_exist, pre_total = probe_all_pools(seqs, cache_config, "pre-GET")
    npools = len(cache_config.enable_pool_specs())
    print(f"  [verify] keys visible from Node B BEFORE get: {pre_exist}/{pre_total} "
          f"across {npools} pool(s)")

    # (2) Functional GET — cold local cache, so any hit is served from the store.
    t0 = time.time()
    cached, total = _get_hit_ratio(kvmanager, seqs, slots)
    dt = time.time() - t0
    hit_ratio = cached / total if total else 0.0
    gb = cached * model_config.token_size_in_bytes / (1024 ** 3)
    print(f"  GET(shared): {cached}/{total} tokens, hit_ratio={hit_ratio*100:.2f}%, "
          f"{dt*1000:.1f} ms, {gb/dt if dt>0 else 0:.2f} GB/s")

    # (3) Negative control — a never-PUT random sequence must NOT hit.
    neg_cached, neg_total = _get_hit_ratio(kvmanager, neg_seqs, neg_slots)
    neg_ratio = neg_cached / neg_total if neg_total else 0.0
    print(f"  GET(negative-control, never PUT): {neg_cached}/{neg_total} tokens, "
          f"hit_ratio={neg_ratio*100:.2f}% (expected ~0%)")

    # ---- Verdict ----
    shared_ok = hit_ratio >= threshold
    store_ok = (pre_exist == pre_total and pre_total > 0)
    neg_ok = neg_ratio <= 0.01
    print("\n" + "-" * 64)
    print("  VERIFICATION")
    print(f"    store keys visible cross-node : {store_ok}  ({pre_exist}/{pre_total})")
    print(f"    cold-cache shared hit >= {threshold*100:.0f}% : {shared_ok}  ({hit_ratio*100:.2f}%)")
    print(f"    negative control ~0%          : {neg_ok}  ({neg_ratio*100:.2f}%)")
    verdict = shared_ok and store_ok and neg_ok
    print(f"    ==> CROSS-NODE SHARING {'CONFIRMED ✅' if verdict else 'FAILED ❌'}")
    print("-" * 64)
    return verdict


# ---------------------------------------------------------------------------
# Bring-up (shared with reference benchmark)
# ---------------------------------------------------------------------------
def bring_up(model_config, cache_config, num_gpu_blocks):
    kvmanager = KVManager(model_config, cache_config)
    kvmanager.start()
    assert not kvmanager.server_client_mode, "Expected direct mode."

    tp_procs = []

    def _cleanup():
        shutdown_tp_clients(tp_procs)
        try:
            kvmanager.shutdown()
        except Exception:
            pass
    atexit.register(_cleanup)

    def _sig(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        _cleanup()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    # IMPORTANT: use spawn (not fork) for tp_client processes. The main process
    # has already initialised a CUDA context (via the mooncake query-only client
    # created inside KVManager), so a forked child would fail torch cuda init
    # with cudaErrorInitializationError. FlexKV's own workers already use spawn.
    spawn_ctx = mp.get_context("spawn")
    for tp_rank in range(model_config.tp_size):
        p = spawn_ctx.Process(target=run_tp_client,
                              args=(0, tp_rank, kvmanager.gpu_register_port,
                                    model_config, cache_config, num_gpu_blocks),
                              daemon=True)
        p.start()
        tp_procs.append(p)

    print("\nWaiting for FlexKV to be ready...")
    t0 = time.time()
    while not kvmanager.is_ready():
        time.sleep(1)
        if time.time() - t0 > 180:
            raise TimeoutError("Timeout waiting for FlexKV ready (180s)")
    print(f"FlexKV ready ({time.time()-t0:.1f}s)")
    return kvmanager, tp_procs, _cleanup


def main(args):
    cli_overrides = {
        "master_addr": args.master_addr,
        "local_hostname": args.local_hostname,
        "device_name": args.device_name,
        "global_segment_size_gb": args.global_segment_size_gb,
    }
    model_config, cache_config, store_cfg_path = load_mooncake_store_config(args.config, cli_overrides)

    # pad seq_len to tokens_per_block
    seq_len = ((args.sequence_length - 1) // cache_config.tokens_per_block + 1) * cache_config.tokens_per_block
    num_gpu_blocks = int(seq_len * args.batch_size / cache_config.tokens_per_block * 1.5) + 64

    if model_config.tp_size * model_config.dp_size > torch.cuda.device_count():
        raise ValueError("tp_size*dp_size exceeds available GPUs")

    print("=" * 64)
    print(f"  mooncake-store E2E  |  mode={args.mode}  seed={args.seed}")
    print(f"  master_addr={cli_overrides['master_addr']}  local_hostname={cli_overrides['local_hostname']}"
          f"  device={cli_overrides['device_name']}")
    print(f"  use_mooncake_store_backend={cache_config.use_mooncake_store_backend}"
          f"  enable_kv_sharing={cache_config.enable_kv_sharing}")
    print(f"  seq_len={seq_len} batch={args.batch_size} num_gpu_blocks={num_gpu_blocks}")
    print("=" * 64)

    kvmanager, tp_procs, _cleanup = bring_up(model_config, cache_config, num_gpu_blocks)

    exit_code = 0
    try:
        seqs, slots = gen_sequences(args.seed, args.batch_size, seq_len)

        if args.mode == "put":
            ok = do_put(kvmanager, model_config, cache_config, seqs, slots)
            exit_code = 0 if ok else 2
            print("\n" + "-" * 64)
            print("PUT done. Node A MUST stay alive so its RDMA segment keeps holding the")
            print("data for Node B to GET. Press Enter (or Ctrl+C) to shut down...")
            print("-" * 64)
            try:
                input()
            except EOFError:
                while True:
                    time.sleep(1)

        elif args.mode == "get":
            # negative control: a sequence generated with a different seed (never PUT)
            neg_seqs, neg_slots = gen_sequences(args.seed ^ 0xA5A5A5, args.batch_size, seq_len)
            verdict = do_get(kvmanager, model_config, cache_config,
                             seqs, slots, neg_seqs, neg_slots, args.hit_threshold)
            exit_code = 0 if verdict else 2

        elif args.mode == "probe":
            # observability-only: just check whether keys exist in the store now
            tot_e, tot = probe_all_pools(seqs, cache_config, "probe")
            print(f"  [probe] {tot_e}/{tot} keys present.")
            exit_code = 0 if (tot_e == tot and tot > 0) else 2

    finally:
        print("\nShutting down...")
        shutdown_tp_clients(tp_procs)
        kvmanager.shutdown()
        try:
            atexit.unregister(_cleanup)
        except Exception:
            pass
        try:
            os.remove(store_cfg_path)
        except OSError:
            pass
        print("Done.")

    sys.exit(exit_code)


def parse_args():
    p = argparse.ArgumentParser(description="FlexKV mooncake-store two-node shared-KVCache E2E test")
    p.add_argument("--config", type=str,
                   default="benchmarks/dist_benchmark/example_dist_direct_mooncake_store_config.yml")
    p.add_argument("--mode", type=str, default="get", choices=["put", "get", "probe"],
                   help="put=Node A writes & stays alive; get=Node B fetches+verifies; probe=key existence only")
    p.add_argument("--seed", type=int, default=20240701,
                   help="MUST be identical on both nodes for the shared sequences")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--sequence-length", type=int, default=1024)
    p.add_argument("--hit-threshold", type=float, default=0.95,
                   help="Min GET hit ratio on Node B to consider sharing confirmed")
    # mooncake per-node overrides
    p.add_argument("--master-addr", type=str, default=None, help="mooncake_master addr, e.g. 10.206.0.9:50051")
    p.add_argument("--local-hostname", type=str, default=None, help="THIS node's RDMA IP (differs per node)")
    p.add_argument("--device-name", type=str, default=None, help="RDMA NIC, e.g. mlx5_0 ('' = auto)")
    p.add_argument("--global-segment-size-gb", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
