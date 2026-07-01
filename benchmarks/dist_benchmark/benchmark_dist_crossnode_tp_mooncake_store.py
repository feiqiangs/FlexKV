"""
Cross-node Tensor-Parallel (TP16 = 2 nodes x 8 GPU) end-to-end validation for the
FlexKV mooncake-store backend, with the DSA/NSA *indexer* side-car pool.

This is the "step 2" of the design review: validate that a SINGLE FlexKV instance
whose TP group spans TWO machines can PUT/GET KV (+ indexer) through the shared
mooncake-store distributed pool.

Topology (nnodes=2, tp_size=16, pp_size=1 -> tp_size_per_node=8, nnodes_per_tp_group=2)
--------------------------------------------------------------------------------
  Node A (--node-rank 0)  = TransferManager MASTER
      * creates KVManager (direct mode): local InterProcess TM (8 local GPUs)
        + a TransferManagerMultiNodeHandle that binds master_ports and pushes
          config to Node B.
      * registers 8 local GPU clients (device_id = local rank 0..7).
      * issues PUT / GET; the transfer graph is submitted to BOTH the local TM
        and (via the MultiNodeHandle) Node B's remote TM.
  Node B (--node-rank 1)  = TransferManager REMOTE
      * launches TransferManagerOnRemote (connects to Node A master_ports,
        receives config incl. the shared gpu_register_port, binds it locally,
        waits for its own 8 local GPU registrations, runs the data plane).
      * registers 8 local GPU clients (device_id = local rank 0..7); then stays
        alive so its RDMA segment + CPU pool keep serving.

Both nodes run the SAME mooncake_master (started on Node A) and each binds its
OWN transfer engine to its bond0 IP (MC_TCP_BIND_ADDRESS, set by the runner).

Notes / scope
-------------
* GLM5 is MLA-based: the KV latent is REPLICATED across TP ranks (not head-sharded),
  so both nodes of the TP group store content-identical blocks under the same
  content-hash key (build_key has no TP dimension) -> no key collision. For MHA/GQA
  models this benchmark would need a per-node key discriminator (not the case here).
* Buffers are synthetic (plumbing/bootstrap correctness), so this validates the
  multi-node bring-up + data-plane + multi-pool sharing, not KV byte-content.
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

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flexkv.server.client import KVTPClient
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.config import (
    ModelConfig, CacheConfig, UserConfig, RankInfo,
    IndexerCacheConfig,
    update_default_config_from_user_config,
    GLOBAL_CONFIG_FROM_ENV,
)
from flexkv.common.debug import flexkv_logger
from flexkv.kvmanager import KVManager
from flexkv.kvtask import KVResponseStatus
from flexkv.transfer_manager import TransferManagerOnRemote

# reuse the verified single-node helpers (same directory)
from benchmark_dist_direct_mooncake_store import (
    gen_sequences, do_put, do_get, probe_all_pools,
)

flexkv_logger.set_level("INFO")

# A FIXED gpu-registration IPC endpoint, identical on BOTH nodes. Node A's
# KVManager binds it; the string is propagated to Node B via config, where Node
# B's TransferManagerOnRemote binds the SAME path on its own filesystem and Node
# B's tp_clients connect to it locally. (ipc:// is per-node, so no cross-node
# clash.)
GPU_REG_PORT = "ipc:///tmp/flexkv_xnode_gpu_reg"


def load_config(config_path, cli):
    """Load config, enable mooncake-store + indexer, and set the cross-node
    (nnodes=2) topology BEFORE the config is frozen."""
    import yaml
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    print(f"[INFO] Loaded config: {config}")

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
    model_config.tp_size = config["tp_size"]          # 16 (global)
    model_config.dp_size = config["dp_size"]
    # ---- cross-node topology (must be set before freeze) ----
    model_config.nnodes = int(cli["nnodes"])
    model_config.master_host = cli["master_host"]
    model_config.master_ports = tuple(cli["master_ports"])

    # ---- mooncake_store.json ----
    master_addr = cli.get("master_addr") or config["mooncake_master_addr"]
    local_hostname = cli.get("local_hostname") or config["mooncake_local_hostname"]
    device_name = cli.get("device_name")
    if device_name is None:
        device_name = config.get("mooncake_device_name", "")
    seg_gb = cli.get("global_segment_size_gb") or config.get("mooncake_global_segment_size_gb", 8)
    mooncake_store_json = {
        "master_addr": master_addr,
        "metadata_server": config.get("mooncake_metadata_server", "P2PHANDSHAKE"),
        "protocol": config.get("mooncake_protocol", "tcp"),
        "device_name": device_name,
        "local_hostname": local_hostname,
        "global_segment_size": int(seg_gb),
        "enable_ssd_offload": False,
        "ssd_offload_path": None,
        "master_metrics_port": config.get("mooncake_master_metrics_port", 9003),
    }
    # IMPORTANT (cross-node): use a FIXED path (identical string on both nodes)
    # instead of mkstemp. cache_config.mooncake_store_config_path is propagated
    # from Node A to Node B via the TM config message; a per-process mkstemp path
    # would NOT exist on Node B's filesystem, so its remote mooncake worker could
    # not read the store config and would hang during setup. Each node writes THIS
    # fixed path locally with ITS OWN local_hostname, so Node B's remote worker
    # reads Node B's file (correct bond0 IP).
    store_cfg_path = "/tmp/flexkv_xnode_mooncake_store.json"
    with open(store_cfg_path, "w") as f:
        json.dump(mooncake_store_json, f, indent=2)
    os.environ["FLEXKV_MOONCAKE_STORE_CONFIG_PATH"] = store_cfg_path
    user_config.use_mooncake_store_backend = True
    user_config.mooncake_store_config_path = store_cfg_path
    print(f"[INFO] mooncake_store config: {json.dumps(mooncake_store_json, indent=2)}")

    cache_config = CacheConfig()
    cache_config.tokens_per_block = config["tokens_per_block"]
    if "cpu_cache_gb" in config:
        user_config.cpu_cache_gb = config["cpu_cache_gb"]

    if config.get("indexer"):
        idx = config["indexer"]
        idx_dtype = eval(f"torch.{idx.get('dtype', 'uint8')}")
        cache_config.indexer = IndexerCacheConfig(
            head_size=int(idx["head_size"]),
            num_kv_heads=int(idx.get("num_kv_heads", 1)),
            dtype=idx_dtype,
        )
        print(f"[INFO] indexer pool ENABLED: head_size={cache_config.indexer.head_size} "
              f"dtype={cache_config.indexer.dtype}")

    update_default_config_from_user_config(
        RankInfo(model_config=model_config), cache_config, user_config
    )
    cache_config.use_mooncake_store_backend = True
    cache_config.mooncake_store_config_path = store_cfg_path
    assert cache_config.use_mooncake_store_backend
    print(f"[INFO] topology: nnodes={model_config.nnodes} tp_size={model_config.tp_size} "
          f"gpus_per_node={model_config.gpus_per_node} "
          f"tp_size_per_node={model_config.tp_size_per_node} "
          f"nnodes_per_tp_group={model_config.nnodes_per_tp_group} "
          f"master={model_config.master_host}:{model_config.master_ports}")
    return model_config, cache_config, store_cfg_path


def run_tp_client(local_rank, gpu_register_port, model_config, cache_config, num_gpu_blocks):
    """Register ONE local GPU (device_id = local_rank, physical cuda:local_rank).

    Mirrors the sglang connector: on every node the tp_clients register with the
    node-local rank (0..gpus_per_node-1); each node's own TransferManager tracks
    its own gpus_per_node GPUs.
    """
    device_id = local_rank
    tp_client = KVTPClient(gpu_register_port, dp_client_id=0, device_id=device_id)
    gpu_kv_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=model_config.num_layers,
        num_block=num_gpu_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        is_mla=model_config.use_mla,
    )
    gpu_blocks = [
        torch.empty(size=tuple(gpu_kv_layout.kv_shape[1:]), dtype=model_config.dtype).cuda(local_rank)
        for _ in range(model_config.num_layers)
    ]
    indexer_buffers = None
    indexer_layout = None
    idx_cfg = getattr(cache_config, "indexer", None)
    if idx_cfg is not None:
        indexer_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=model_config.num_layers,
            num_block=num_gpu_blocks,
            tokens_per_block=1,
            num_head=idx_cfg.num_kv_heads,
            head_size=idx_cfg.head_size,
            is_mla=True,
        )
        indexer_buffers = [
            torch.empty(size=tuple(indexer_layout.kv_shape[1:]), dtype=idx_cfg.dtype).cuda(local_rank)
            for _ in range(model_config.num_layers)
        ]
    tp_client.register_to_server(
        gpu_blocks, gpu_kv_layout,
        indexer_buffers=indexer_buffers, indexer_layout=indexer_layout,
    )
    while True:
        time.sleep(1)


def spawn_tp_clients(model_config, cache_config, num_gpu_blocks):
    spawn_ctx = mp.get_context("spawn")
    procs = []
    for local_rank in range(model_config.tp_size_per_node):
        p = spawn_ctx.Process(
            target=run_tp_client,
            args=(local_rank, GPU_REG_PORT, model_config, cache_config, num_gpu_blocks),
            daemon=True,
        )
        p.start()
        procs.append(p)
    return procs


def kill_procs(procs):
    for p in procs:
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join(timeout=2)


def main(args):
    cli = {
        "nnodes": args.nnodes,
        "master_host": args.master_host,
        "master_ports": args.master_ports.split(","),
        "master_addr": args.master_addr,
        "local_hostname": args.local_hostname,
        "device_name": args.device_name,
        "global_segment_size_gb": args.global_segment_size_gb,
    }
    model_config, cache_config, store_cfg_path = load_config(args.config, cli)

    seq_len = ((args.sequence_length - 1) // cache_config.tokens_per_block + 1) * cache_config.tokens_per_block
    num_gpu_blocks = int(seq_len * args.batch_size / cache_config.tokens_per_block * 1.5) + 64

    if model_config.tp_size_per_node > torch.cuda.device_count():
        raise ValueError(f"tp_size_per_node={model_config.tp_size_per_node} exceeds "
                         f"available GPUs={torch.cuda.device_count()}")

    print("=" * 72)
    print(f"  CROSS-NODE TP  |  node_rank={args.node_rank}  mode={args.mode}  seed={args.seed}")
    print(f"  local_hostname={cli['local_hostname']}  mooncake_master={cli['master_addr']}")
    print(f"  TM master={model_config.master_host}:{model_config.master_ports}")
    print("=" * 72)

    if args.node_rank == 0:
        _main_node0(args, model_config, cache_config, store_cfg_path, num_gpu_blocks, seq_len)
    else:
        _main_node1(args, model_config, cache_config, store_cfg_path, num_gpu_blocks)


def _main_node0(args, model_config, cache_config, store_cfg_path, num_gpu_blocks, seq_len):
    kvmanager = KVManager(model_config, cache_config, gpu_register_port=GPU_REG_PORT)
    kvmanager.start() if hasattr(kvmanager, "start") else None  # KVManager starts in __init__
    tp_procs = []

    def _cleanup():
        kill_procs(tp_procs)
        try:
            kvmanager.shutdown()
        except Exception:
            pass
    atexit.register(_cleanup)

    def _sig(signum, frame):
        _cleanup(); signal.signal(signum, signal.SIG_DFL); os.kill(os.getpid(), signum)
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    tp_procs = spawn_tp_clients(model_config, cache_config, num_gpu_blocks)

    print("\n[Node A] waiting for FlexKV ready (needs BOTH nodes' 8+8 GPUs)...")
    t0 = time.time()
    while not kvmanager.is_ready():
        time.sleep(1)
        if time.time() - t0 > 600:
            raise TimeoutError("Timeout (600s) waiting for cross-node FlexKV ready")
    print(f"[Node A] FlexKV ready ({time.time()-t0:.1f}s)")

    exit_code = 0
    try:
        seqs, slots = gen_sequences(args.seed, args.batch_size, seq_len)
        if args.mode == "put":
            ok = do_put(kvmanager, model_config, cache_config, seqs, slots)
            exit_code = 0 if ok else 2
            print("\n[Node A] PUT done. Staying alive (segment holds data). Ctrl+C to stop...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        elif args.mode == "get":
            neg_seqs, neg_slots = gen_sequences(args.seed ^ 0xA5A5A5, args.batch_size, seq_len)
            verdict = do_get(kvmanager, model_config, cache_config,
                             seqs, slots, neg_seqs, neg_slots, args.hit_threshold)
            exit_code = 0 if verdict else 2
        elif args.mode == "putget":
            # Single-process PUT then cold GET. Node A stays alive throughout, so
            # BOTH nodes' mooncake segments keep serving (no data loss). We drop
            # the local CPU cache between the two so the GET is a genuine cold
            # read from the shared store across nodes (exercising REMOTE2H + H2D
            # and the multi-node completion path in the read direction too).
            ok = do_put(kvmanager, model_config, cache_config, seqs, slots)
            if not ok:
                exit_code = 2
            else:
                print("\n[Node A] PUT done; clearing local CPU cache to force a "
                      "COLD cross-node store read for GET...")
                kvmanager._clear_cpu_cache()
                time.sleep(2)
                neg_seqs, neg_slots = gen_sequences(args.seed ^ 0xA5A5A5, args.batch_size, seq_len)
                verdict = do_get(kvmanager, model_config, cache_config,
                                 seqs, slots, neg_seqs, neg_slots, args.hit_threshold)
                exit_code = 0 if verdict else 2
        elif args.mode == "probe":
            e, t = probe_all_pools(seqs, cache_config, "probe")
            print(f"  [probe] {e}/{t} keys present.")
            exit_code = 0 if (e == t and t > 0) else 2
    finally:
        kill_procs(tp_procs)
        try:
            kvmanager.shutdown()
        except Exception:
            pass
        try:
            atexit.unregister(_cleanup)
        except Exception:
            pass
        try:
            os.remove(store_cfg_path)
        except OSError:
            pass
    sys.exit(exit_code)


def _main_node1(args, model_config, cache_config, store_cfg_path, num_gpu_blocks):
    print("[Node B] launching TransferManagerOnRemote (connect to master)...")
    remote = TransferManagerOnRemote.create_process(
        master_host=model_config.master_host,
        master_ports=tuple(model_config.master_ports),
    )
    tp_procs = []

    def _cleanup():
        kill_procs(tp_procs)
        try:
            if remote is not None and remote.is_alive():
                remote.terminate()
        except Exception:
            pass
    atexit.register(_cleanup)

    def _sig(signum, frame):
        _cleanup(); signal.signal(signum, signal.SIG_DFL); os.kill(os.getpid(), signum)
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    # give the remote TM a moment to connect + receive config + bind the
    # gpu_register_port before local tp_clients try to register.
    time.sleep(8)
    tp_procs = spawn_tp_clients(model_config, cache_config, num_gpu_blocks)
    print("[Node B] 8 local GPU clients spawned; remote data-plane serving. "
          "Staying alive until Node A finishes. Ctrl+C to stop...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        _cleanup()
        try:
            os.remove(store_cfg_path)
        except OSError:
            pass


def parse_args():
    p = argparse.ArgumentParser(description="FlexKV cross-node TP16 mooncake-store E2E")
    p.add_argument("--config", type=str,
                   default="benchmarks/dist_benchmark/example_dist_crossnode_tp16_mooncake_store_config.yml")
    p.add_argument("--mode", type=str, default="put", choices=["put", "get", "putget", "probe"])
    p.add_argument("--node-rank", type=int, required=True, choices=[0, 1],
                   help="0=TM master (KVManager + PUT/GET); 1=TM remote (data-plane only)")
    p.add_argument("--nnodes", type=int, default=2)
    p.add_argument("--master-host", type=str, required=True,
                   help="Node A bond0 IP (TransferManager rendezvous)")
    p.add_argument("--master-ports", type=str, default="5556,5557,5558",
                   help="comma-separated command,result,query ports")
    p.add_argument("--seed", type=int, default=20240701)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--hit-threshold", type=float, default=0.95)
    p.add_argument("--master-addr", type=str, default=None, help="mooncake_master ip:port")
    p.add_argument("--local-hostname", type=str, default=None, help="THIS node bond0 IP")
    p.add_argument("--device-name", type=str, default=None)
    p.add_argument("--global-segment-size-gb", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
