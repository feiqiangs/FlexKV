import time
import os
import shutil
import random

import pytest
import torch
import multiprocessing as mp
from multiprocessing import Process, Pipe

from flexkv.common.config import ModelConfig, CacheConfig, RankInfo
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.request import KVResponseStatus
from flexkv.kvtask import KVTaskEngine
from flexkv.kvmanager import KVManager
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.server.client import KVTPClient
import traceback

from flexkv.common.debug import flexkv_logger

# Import utilities from common_utils
from common_utils import (
    DEFAULT_MODEL_CONFIG, DEFAULT_CACHE_CONFIG, DEFAULT_TEST_CONFIG,
    generate_request_pair, block_ids_2_slot_mapping,
    skip_if_insufficient_gpus,create_gpu_kv_layout, GPUKVCacheVerifier
)


def _fp8_cuda_ops_unavailable():
    """True if fp8 dtype exists but CUDA ops (e.g. mul_cuda) are not implemented."""
    if not hasattr(torch, "float8_e4m3fn"):
        return True
    if not torch.cuda.is_available():
        return False
    try:
        t = torch.tensor([1.0], dtype=torch.float8_e4m3fn, device="cuda")
        t.mul(1.0)
        return False
    except (NotImplementedError, RuntimeError):
        return True

def run_tp_client(dp_client_id,
                  tp_rank,
                  server_recv_port,
                  model_config,
                  cache_config,
                  num_gpu_blocks,
                  child_conn,
                  gpu_layout_type):
    """Run tp_client process"""
    try:
        device_id = tp_rank + dp_client_id * model_config.tp_size
        tp_client = KVTPClient(server_recv_port,
                               dp_client_id=dp_client_id, pp_rank=0,
                               device_id=device_id)

        gpu_kv_layout = create_gpu_kv_layout(model_config, cache_config, num_gpu_blocks, gpu_layout_type)

        # Create GPU blocks for this tp_rank in the tp_client process
        gpu_blocks_for_tp = []
        if gpu_layout_type == 0:
            for _ in range(model_config.num_layers):
                gpu_blocks_for_tp.append(
                    torch.empty(size=tuple(gpu_kv_layout.kv_shape[1:]), dtype=model_config.dtype).cuda(device_id)
                )
        elif gpu_layout_type == 1:
            gpu_blocks_for_tp.append(
                torch.empty(size=tuple(gpu_kv_layout.kv_shape[:]), dtype=model_config.dtype).cuda(device_id)
            )
        elif gpu_layout_type == 2:
            kv_dim = model_config.kv_dim
            for _ in range(model_config.num_layers * kv_dim):
                gpu_blocks_for_tp.append(
                    torch.empty(size=tuple(gpu_kv_layout.kv_shape[2:]), dtype=model_config.dtype).cuda(device_id)
                )
        else:
            raise ValueError(f"Invalid GPU layout type: {gpu_layout_type}")
        tp_client.register_to_server(gpu_blocks_for_tp, gpu_kv_layout)

        # Send GPU blocks back to main process via pipe if connection provided
        if child_conn is not None:
            print(f"[TP Client {tp_rank}] Converting {len(gpu_blocks_for_tp)} GPU blocks to TensorSharedHandle")
            shared_gpu_blocks = [TensorSharedHandle(tensor) for tensor in gpu_blocks_for_tp]
            child_conn.send(shared_gpu_blocks)
            print(f"[TP Client {tp_rank}] Sent GPU blocks to main process via pipe")
            child_conn.close()

        # Keep the process running
        while True:
            time.sleep(1)
    except Exception as e:
        print(f"[TP Client {tp_rank}] Exception occurred: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        if child_conn is not None:
            child_conn.send(None)
            child_conn.close()

def shutdown_tp_client(tp_client_processes):
    for tp_process in tp_client_processes:
        if tp_process.is_alive():
            tp_process.terminate()
            tp_process.join(timeout=5)
            if tp_process.is_alive():
                print(f"Force killing tp_client process {tp_process.pid}")
                tp_process.kill()
                tp_process.join(timeout=2)

@pytest.mark.parametrize(
    "model_config",
    [
        {"tp_size": 1, "dp_size": 1},
        {"tp_size": 2, "dp_size": 2},
        {"dtype": torch.float32},
        {"use_mla": True},
        {"tp_size": 4, "dp_size": 1, "use_mla": True},
        {"tp_size": 4, "dp_size": 1},
        # fp8 端到端流程覆盖（仅在当前 PyTorch 支持 float8_e4m3fn 且 CUDA 具备 mul 等算子时启用）
        pytest.param(
            {"dtype": torch.float8_e4m3fn},
            marks=pytest.mark.skipif(
                _fp8_cuda_ops_unavailable(),
                reason="fp8 dtype or CUDA ops (e.g. mul_cuda) not available in this PyTorch build",
            ),
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("cache_config", [
    {'enable_cpu': True, 'enable_ssd': False, 'num_cpu_blocks': 1024},
    {'enable_cpu': True, 'enable_ssd': True, 'num_cpu_blocks': 256, 'num_ssd_blocks': 2048},
    # GDS test configs
    {'enable_cpu': True, 'enable_gds': True, 'enable_ssd': True, \
        'enable_remote': False, 'num_cpu_blocks':256, 'num_ssd_blocks': 1024},
], indirect=True)
@pytest.mark.parametrize("test_config", [
    {'num_gpu_blocks': 512, 'requests_per_block': 16, 'initial_write_ratio': 0.4},
    {'num_gpu_blocks': 512, 'requests_per_block': 16, 'initial_write_ratio': 0.4, 'namespace': ['test_namespace']},
], indirect=True)
@pytest.mark.parametrize("gpu_layout_type", [
    0,
    1,
    2,
])
def test_kvmanager(model_config, cache_config, test_config, gpu_layout_type):
    tp_size = model_config.tp_size
    dp_size = model_config.dp_size

    tokens_per_block = cache_config.tokens_per_block
    num_cpu_blocks = cache_config.num_cpu_blocks
    num_ssd_blocks = cache_config.num_ssd_blocks

    enable_cpu = cache_config.enable_cpu
    enable_ssd = cache_config.enable_ssd
    enable_remote = cache_config.enable_remote
    enable_gds = cache_config.enable_gds

    num_gpu_blocks = test_config["num_gpu_blocks"]
    block_per_request = test_config['requests_per_block']
    initial_write_ratio = test_config['initial_write_ratio']
    namespace = test_config.get('namespace', None)

    num_requests = num_gpu_blocks // block_per_request

    # Skip tests based on GPU availability and configuration
    skip_if_insufficient_gpus(tp_size * dp_size)

    if enable_gds and os.environ.get("FLEXKV_ENABLE_GDS", "0") == "0":
        pytest.skip("skip because GDS test is not enabled")

    if enable_remote:
        pytest.skip("skip because enable_remote is not supported")

    if dp_size > 1:
         #note that for now only dp_size=1 is supported
        pytest.skip("skip because server-client mode is not ready for dp_size > 1")

    kvmanager = KVManager(
        model_config=model_config,
        cache_config=cache_config,
        dp_client_id=0,
    )
    kvmanager.start()

    # Create pipes for each tp_client to send GPU blocks back
    mp_ctx = mp.get_context('spawn')
    pipe_connections = []
    tp_client_processes = []

    for tp_rank in range(tp_size):
        parent_conn, child_conn = mp_ctx.Pipe()
        pipe_connections.append(parent_conn)

        tp_client_process = mp_ctx.Process(
            target=run_tp_client,
            args=(0, tp_rank, kvmanager.gpu_register_port, model_config, cache_config, \
                num_gpu_blocks + tp_rank, child_conn, gpu_layout_type),
            daemon=True
        )
        tp_client_processes.append(tp_client_process)
        tp_client_process.start()

    # Collect GPU blocks from all tp_client processes
    print(f"[Main Process] Waiting to receive GPU blocks from {tp_size} TP client processes...")
    all_gpu_blocks = []

    for tp_rank, parent_conn in enumerate(pipe_connections):
        try:
            shared_gpu_blocks = parent_conn.recv()
            if shared_gpu_blocks is not None:
                all_gpu_blocks.append(shared_gpu_blocks)
                print(f"[Main Process] Received GPU blocks from TP client {tp_rank}")
            else:
                print(f"[Main Process] TP client {tp_rank} failed to create GPU blocks")
            parent_conn.close()
        except Exception as e:
            print(f"[Main Process] Error receiving from TP client {tp_rank}: {e}")

    # Create GPUKVCacheVerifier with collected GPU blocks
    if all_gpu_blocks and len(all_gpu_blocks) == tp_size:
        print(f"[Main Process] Creating GPUKVCacheVerifier with GPU blocks from {len(all_gpu_blocks)} TP clients")

        # Get gpu_kv_layout from cache_config for GPUKVCacheVerifier
        gpu_kv_layout = create_gpu_kv_layout(model_config, cache_config, num_gpu_blocks, gpu_layout_type)

        gpu_kv_verifier = GPUKVCacheVerifier(
            shared_gpu_blocks=all_gpu_blocks,
            gpu_kv_layout=gpu_kv_layout,
            tp_size=model_config.tp_size,
            tokens_per_block=cache_config.tokens_per_block,
            dtype=model_config.dtype,
            gpu_layout_type=gpu_layout_type
        )
        print("[Main Process] GPUKVCacheVerifier created successfully")
    else:
        print(f"[Main Process] Failed to collect GPU blocks from all TP clients. "
              f"Got {len(all_gpu_blocks)} out of {tp_size}")
        gpu_kv_verifier = None

    while not kvmanager.is_ready():
        time.sleep(1)
        flexkv_logger.info("waiting for flexkv to be ready")

    num_remote_blocks = cache_config.num_remote_blocks
    request_pairs = [generate_request_pair(i, block_per_request, num_gpu_blocks, tokens_per_block, dp_size)
                     for i in range(num_requests)]
    initial_write_num = int(num_requests * initial_write_ratio)
    print("writing initial data...")
    put_ids = []
    for token_ids, block_ids, dp_client_id in request_pairs[:initial_write_num]:
        if gpu_kv_verifier is not None:
            gpu_kv_verifier.fill_gpu_blocks(token_ids, block_ids)
        write_request = kvmanager.put_async(
            token_ids=token_ids,
            slot_mapping=block_ids_2_slot_mapping(block_ids, tokens_per_block),
            token_mask=None,
            namespace=namespace,
        )
        kvmanager.wait([write_request], completely=True)
        if gpu_kv_verifier is not None:
            gpu_kv_verifier.clear_gpu_blocks(block_ids)

    #corner case: input token length for put is less than tokens_per_block
    write_request = kvmanager.put_async(
        token_ids=torch.randint(0, 100, size=(8,), dtype=torch.int64),
        slot_mapping=block_ids_2_slot_mapping(torch.arange(0,1, dtype=torch.int64), tokens_per_block, actual_length=8),
        token_mask=None,
        namespace=namespace,
    )
    kvmanager.wait([write_request], completely=True)
    #corner case: input token length is long enough, but the mask is less than tokens_per_block
    #my_mask = torch.zeros(16, dtype=torch.bool)
    #my_mask[0:8] = True
    #write_request = kvmanager.put_async(
    #    token_ids=torch.randint(0, 100, size=(16,), dtype=torch.int64),
    #    slot_mapping=block_ids_2_slot_mapping(torch.arange(0,1, dtype=torch.int64), tokens_per_block, actual_length=8),
    #    token_mask=my_mask,
    #)
    #kvmanager.wait_for_graph_finished(write_request)

    print(f"initial data {initial_write_num} written")
    total_cache_hit = 0
    total_cache_miss = 0
    running_get_requests = []
    running_put_requests = []
    req_id2block_ids = {}
    req_id2token_ids = {}
    flexkv_id2req_id = {}
    start_time = time.time()
    print(f"the initial {initial_write_num} write done,performing mixed read/write...")
    for i in range(initial_write_num, num_requests):
        print(f"performing mixed read/write {i} / {num_requests} ...")
        read_idx = i - initial_write_num
        token_ids, block_ids, dp_client_id = request_pairs[read_idx]
        slot_mapping = block_ids_2_slot_mapping(block_ids, tokens_per_block)
        request_id, _ = kvmanager.get_match(
            token_ids=token_ids,
            token_mask=None,
            namespace=namespace,
        )
        kvmanager.launch(request_id, slot_mapping)
        flexkv_id2req_id[request_id] = read_idx
        running_get_requests.append(request_id)
        req_id2block_ids[request_id] = block_ids
        req_id2token_ids[request_id] = token_ids
        token_ids, block_ids, dp_client_id = request_pairs[i]
        if gpu_kv_verifier is not None:
            gpu_kv_verifier.fill_gpu_blocks(token_ids, block_ids)
        request_id = kvmanager.put_async(
            token_ids=token_ids,
            slot_mapping=block_ids_2_slot_mapping(block_ids, tokens_per_block),
            token_mask=None,
            namespace=namespace,
        )
        req_id2block_ids[request_id] = block_ids
        flexkv_id2req_id[request_id] = i
        print(f"write flexkv request_id {request_id} to req_id {i}")
        running_put_requests.append(request_id)
        min_block_num = min(num_cpu_blocks, num_gpu_blocks)
        if (len(running_get_requests) + len(running_put_requests) >= min_block_num // block_per_request - 2 or
            i % initial_write_num == initial_write_num - 1 or
            i == num_requests - 1):
            if len(running_put_requests) > 0:
                kvmanager.wait(running_put_requests, completely=True)
                if gpu_kv_verifier is not None:
                    for req_id in running_put_requests:
                        gpu_kv_verifier.clear_gpu_blocks(req_id2block_ids[req_id])
            if len(running_get_requests) > 0:
                return_results = kvmanager.wait(running_get_requests, completely=True)
                if gpu_kv_verifier is not None:
                    for req_id, kvresponse in return_results.items():
                        assert kvresponse.status == KVResponseStatus.SUCCESS
                        valid_fetched_tokens = kvresponse.return_mask.sum().item() // \
                            tokens_per_block * tokens_per_block
                        token_ids = req_id2token_ids[req_id]
                        block_ids = req_id2block_ids[req_id]
                        assert gpu_kv_verifier.verify_kv_blocks(
                            token_ids[:valid_fetched_tokens],
                            block_ids[:valid_fetched_tokens//tokens_per_block])
                for kvresponse in return_results.values():
                    assert kvresponse.status == KVResponseStatus.SUCCESS
                    total_cache_hit += kvresponse.return_mask.sum().item()
                    total_cache_miss += len(kvresponse.return_mask) - kvresponse.return_mask.sum().item()
            running_get_requests = []
            running_put_requests = []
    if len(running_get_requests) > 0:
        return_results = kvmanager.wait(running_get_requests, completely=True)
        if gpu_kv_verifier is not None:
            for req_id, kvresponse in return_results.items():
                assert kvresponse.status == KVResponseStatus.SUCCESS
                valid_fetched_tokens = kvresponse.return_mask.sum().item() // tokens_per_block * tokens_per_block
                token_ids = req_id2token_ids[req_id]
                block_ids = req_id2block_ids[req_id]
                assert gpu_kv_verifier.verify_kv_blocks(
                    token_ids[:valid_fetched_tokens],
                    block_ids[:valid_fetched_tokens//tokens_per_block])
        running_get_requests = []
    if len(running_put_requests) > 0:
        kvmanager.wait(running_put_requests, completely=True)
        running_put_requests = []
    print("mixed read/write done")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time: {total_time} s")
    print(f"Total cache hit rate: {total_cache_hit / (total_cache_hit + total_cache_miss)}")

    # =============== Test batched launched get ===============
    if not enable_gds:
        print("\n========== Testing batched launched get ==========")

        # Use the first few request_pairs that were written in initial phase
        batch_size = 6

        batched_get_task_ids = []
        batched_slot_mappings = []
        batched_req_info = []  # Store (token_ids, block_ids) for verification

        # Create multiple get_match requests
        for i in range(batch_size):
            token_ids, block_ids, dp_client_id = request_pairs[random.randint(0, num_requests - 1)]
            slot_mapping = block_ids_2_slot_mapping(block_ids, tokens_per_block)

            request_id, return_mask = kvmanager.get_match(
                token_ids=token_ids,
                token_mask=None,
                namespace=namespace,
            )
            batched_get_task_ids.append(request_id)
            batched_slot_mappings.append(slot_mapping)
            batched_req_info.append((token_ids, block_ids, request_id))
            print(f"Created get_match request {request_id} for request_pair[{i}]")

        # Launch all get requests as a batch
        print(f"Launching {len(batched_get_task_ids)} get requests as batch...")
        batch_id = kvmanager.launch(
            task_ids=batched_get_task_ids,
            slot_mappings=batched_slot_mappings,
            as_batch=True
        )[0]
        print(f"Returned task_ids after batch launch: {batch_id}")

        # Wait for the batched get to complete
        # When as_batch=True, launch returns [batch_id], we need to wait on batch_id
        batch_results = kvmanager.wait(batch_id, completely=True)
        print(f"Batch wait returned {len(batch_results)} results")

        # Verify results
        batched_cache_hit = 0
        batched_cache_miss = 0
        kvresponse = batch_results[batch_id]
        assert kvresponse.status == KVResponseStatus.SUCCESS, \
            f"Batched get task {batch_id} failed with status {kvresponse.status}"
        for mask in kvresponse.return_mask:
            batched_cache_hit += return_mask.sum().item()
            batched_cache_miss += len(return_mask) - return_mask.sum().item()
            print(f"Task {batch_id}: cache_hit={batched_cache_hit}, cache_miss={batched_cache_miss}")

        # GPU KV cache verification for batched get
        if gpu_kv_verifier is not None:
            for idx, (token_ids, block_ids, req_id) in enumerate(batched_req_info):
                # Find the corresponding response
                # Note: when batched, the returned task_id might be the batch_id
                # We need to verify based on the actual data
                valid_fetched_tokens = kvresponse.return_mask[idx].sum().item() // tokens_per_block * tokens_per_block
                if valid_fetched_tokens > 0:
                    # Verify that GPU blocks contain correct data
                    verify_result = gpu_kv_verifier.verify_kv_blocks(
                        token_ids[:valid_fetched_tokens],
                        block_ids[:valid_fetched_tokens // tokens_per_block]
                    )

        print(f"Batched get test completed: hit={batched_cache_hit}, miss={batched_cache_miss}")

        # Since we read data that was written before, cache hit should be high
        if enable_cpu and num_cpu_blocks >= num_gpu_blocks:
            assert batched_cache_miss == 0, \
                f"Expected 0 cache miss for batched get, but got {batched_cache_miss}"
            print("  ✓ Batched launched get verification PASSED (100% cache hit)")
        else:
            print(f"  Batched launched get completed (cache hit rate: "
                    f"{batched_cache_hit / (batched_cache_hit + batched_cache_miss):.2%})")

    if enable_cpu and num_cpu_blocks >= num_gpu_blocks or \
        enable_ssd and num_ssd_blocks >= num_gpu_blocks or \
        enable_remote and num_remote_blocks >= num_gpu_blocks or \
        enable_gds and num_ssd_blocks >= num_gpu_blocks:
        assert total_cache_miss == 0
    shutdown_tp_client(tp_client_processes)
    kvmanager.shutdown()

    # Only verify data in direct mode
    # verify_data(gpu_blocks, dp_wise_gpu_blocks_gt, num_kv_heads, tp_size, dp_size, num_layers, use_mla)
    if total_cache_miss == 0:
        return
    elif total_cache_miss > 0:
        print(f"verify skipped, because of total_cache_miss={total_cache_miss} > 0")


class GPUIndexerCacheVerifier:
    def __init__(self,
                 shared_indexer_blocks,
                 indexer_kv_layout: KVCacheLayout,
                 tp_size: int,
                 dtype: torch.dtype) -> None:
        if not shared_indexer_blocks:
            raise ValueError("shared_indexer_blocks must not be empty")

        if isinstance(shared_indexer_blocks[0][0], torch.Tensor):
            self.gpu_blocks = shared_indexer_blocks
        else:
            imported_gpu_blocks = []
            for handles_in_one_gpu in shared_indexer_blocks:
                imported_gpu_blocks.append([handle.get_tensor() for handle in handles_in_one_gpu])
            self.gpu_blocks = imported_gpu_blocks

        self.num_layers = indexer_kv_layout.num_layer
        self.tokens_per_block = indexer_kv_layout.tokens_per_block
        self.head_size = indexer_kv_layout.head_size
        self.tp_size = tp_size
        self.dtype = dtype

    def hash_all_values(self, layer_id, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        token_hash = 0
        for i, token_id in enumerate(token_ids):
            token_hash += int(token_id) * (i + 17)
        return torch.tensor(((layer_id + 1) * 29 + token_hash) % 251 + 1, dtype=self.dtype).item()

    def fill_gpu_blocks(self, block_ids, main_kv_tokens_per_block, token_ids):
        """Fill indexer GPU blocks with deterministic hash values.

        Indexer uses tokens_per_block=1 on CPU/SSD side.  Each indexer block
        corresponds to one main-KV block (1:1 page mapping).  We hash the
        *entire page* of token_ids from the main KV request to produce a
        single deterministic value per (layer, block).

        Args:
            block_ids: block IDs to fill (same as main KV block_ids).
            main_kv_tokens_per_block: tokens_per_block of main KV (e.g. 16).
            token_ids: full token_ids tensor from the request.
        """
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.int64)
        if not isinstance(block_ids, torch.Tensor):
            block_ids = torch.tensor(block_ids, dtype=torch.int64)

        for tp_id in range(self.tp_size):
            for layer_id in range(self.num_layers):
                gpu_tensor = self.gpu_blocks[tp_id][layer_id]
                for block_idx, block_id in enumerate(block_ids):
                    start_token_idx = block_idx * main_kv_tokens_per_block
                    end_token_idx = start_token_idx + main_kv_tokens_per_block
                    hash_value = self.hash_all_values(
                        layer_id,
                        token_ids[start_token_idx:end_token_idx],
                    )
                    # gpu_tensor shape: (num_blocks, tokens_per_block=1, head_size)
                    gpu_tensor[block_id, :, :] = hash_value

    def clear_gpu_blocks(self, block_ids):
        if not isinstance(block_ids, torch.Tensor):
            block_ids = torch.tensor(block_ids, dtype=torch.int64)

        for tp_id in range(self.tp_size):
            for layer_id in range(self.num_layers):
                self.gpu_blocks[tp_id][layer_id][block_ids, :, :] = 0

    def verify_gpu_blocks(self, block_ids, main_kv_tokens_per_block, token_ids) -> bool:
        """Verify indexer GPU blocks after round-trip transfer.

        Args:
            block_ids: block IDs to verify.
            main_kv_tokens_per_block: tokens_per_block of main KV.
            token_ids: full token_ids tensor from the request.
        """
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.int64)
        if not isinstance(block_ids, torch.Tensor):
            block_ids = torch.tensor(block_ids, dtype=torch.int64)

        verification_passed = True
        errors = []

        for tp_id in range(self.tp_size):
            for layer_id in range(self.num_layers):
                gpu_tensor = self.gpu_blocks[tp_id][layer_id]
                for block_idx, block_id in enumerate(block_ids):
                    start_token_idx = block_idx * main_kv_tokens_per_block
                    end_token_idx = start_token_idx + main_kv_tokens_per_block
                    expected_hash_value = self.hash_all_values(
                        layer_id,
                        token_ids[start_token_idx:end_token_idx],
                    )
                    actual_values = gpu_tensor[block_id, :, :]
                    expected_tensor = torch.full_like(actual_values, expected_hash_value)
                    if not torch.equal(actual_values, expected_tensor):
                        verification_passed = False
                        max_abs_diff = (
                            actual_values.to(torch.int32) - expected_tensor.to(torch.int32)
                        ).abs().max().item()
                        errors.append(
                            f"Mismatch at tp={tp_id}, layer={layer_id}, block={block_id}: "
                            f"expected={expected_hash_value}, max_abs_diff={max_abs_diff}"
                        )

        if not verification_passed:
            print(f"Indexer verification failed with {len(errors)} errors:")
            for error in errors[:10]:
                print(f"  {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
        else:
            print("Indexer GPU blocks verification passed!")
        assert verification_passed
        return verification_passed


def run_tp_client_with_indexer(dp_client_id,
                               tp_rank,
                               server_recv_port,
                               model_config,
                               cache_config,
                               num_gpu_blocks,
                               child_conn,
                               gpu_layout_type):
    """Run tp_client process with indexer support (shadow transfer mode).

    Indexer configuration is read from cache_config.indexer (IndexerCacheConfig).
    """
    try:
        device_id = tp_rank + dp_client_id * model_config.tp_size

        gpu_kv_layout = create_gpu_kv_layout(model_config, cache_config, num_gpu_blocks, gpu_layout_type)

        # Create main GPU blocks
        gpu_blocks_for_tp = []
        if gpu_layout_type == 0:
            for _ in range(model_config.num_layers):
                gpu_blocks_for_tp.append(
                    torch.empty(size=tuple(gpu_kv_layout.kv_shape[1:]), dtype=model_config.dtype).cuda(device_id)
                )
        elif gpu_layout_type == 2:
            kv_dim = model_config.kv_dim
            for _ in range(model_config.num_layers * kv_dim):
                gpu_blocks_for_tp.append(
                    torch.empty(size=tuple(gpu_kv_layout.kv_shape[2:]), dtype=model_config.dtype).cuda(device_id)
                )
        else:
            raise ValueError(f"Invalid GPU layout type for indexer test: {gpu_layout_type}")

        # Derive indexer params from cache_config.indexer (IndexerCacheConfig).
        # Indexer uses tokens_per_block=1 (one indexer entry per page/block),
        # matching the CPU/SSD layout in StorageEngine.
        indexer_cfg = cache_config.indexer
        assert indexer_cfg is not None, "cache_config.indexer must be set for indexer shadow transfer tests"
        indexer_tokens_per_block = 1  # indexer: 1 entry per page (not main KV tokens_per_block)
        indexer_num_layers = model_config.num_layers

        # Create indexer GPU blocks (MLA-style: 3D tensors)
        indexer_blocks = []
        for _ in range(indexer_num_layers):
            indexer_blocks.append(
                torch.empty(
                    num_gpu_blocks,
                    indexer_tokens_per_block,
                    indexer_cfg.head_size,
                    dtype=indexer_cfg.dtype,
                ).cuda(device_id)
            )

        from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
        indexer_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=indexer_num_layers,
            num_block=num_gpu_blocks,
            tokens_per_block=indexer_tokens_per_block,
            num_head=indexer_cfg.num_kv_heads,
            head_size=indexer_cfg.head_size,
            is_mla=True,
        )

        # Use KVTPClient directly with indexer buffers (shadow transfer mode)
        tp_client = KVTPClient(
            gpu_register_port=server_recv_port + "_gpu_register",
            dp_client_id=dp_client_id, pp_rank=0,
            device_id=device_id,
        )
        tp_client.register_to_server(
            kv_caches=gpu_blocks_for_tp,
            kv_layout=gpu_kv_layout,
            indexer_buffers=indexer_blocks,
            indexer_layout=indexer_layout,
        )

        # Send GPU blocks back to main process via pipe
        if child_conn is not None:
            shared_gpu_blocks = [TensorSharedHandle(tensor) for tensor in gpu_blocks_for_tp]
            shared_indexer_blocks = [TensorSharedHandle(tensor) for tensor in indexer_blocks]
            child_conn.send({
                "main": shared_gpu_blocks,
                "indexer": shared_indexer_blocks,
            })
            child_conn.close()

        # Keep the process running
        while True:
            time.sleep(1)
    except Exception as e:
        print(f"[TP Client {tp_rank}] Exception occurred: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        if child_conn is not None:
            child_conn.send(None)
            child_conn.close()


def _run_indexer_test(model_config, cache_config, test_config, gpu_layout_type, test_label="indexer", layerwise=False):
    """Core test logic for KVManager with indexer shadow transfer.

    Shared by test_kvmanager_with_indexer (non-layerwise) and
    test_kvmanager_with_indexer_layerwise (layerwise mode).
    """
    tp_size = model_config.tp_size
    tokens_per_block = cache_config.tokens_per_block
    num_gpu_blocks = test_config["num_gpu_blocks"]
    block_per_request = test_config['requests_per_block']
    initial_write_ratio = test_config['initial_write_ratio']
    num_requests = num_gpu_blocks // block_per_request

    skip_if_insufficient_gpus(tp_size)

    from flexkv.common.config import IndexerCacheConfig
    cache_config.indexer = IndexerCacheConfig(
        head_size=64,
        num_kv_heads=1,
        dtype=torch.uint8,
    )

    kvmanager = KVManager(
        model_config=model_config,
        cache_config=cache_config,
        dp_client_id=0,
    )
    kvmanager.start()

    mp_ctx = mp.get_context('spawn')
    pipe_connections = []
    tp_client_processes = []

    for tp_rank in range(tp_size):
        parent_conn, child_conn = mp_ctx.Pipe()
        pipe_connections.append(parent_conn)

        tp_client_process = mp_ctx.Process(
            target=run_tp_client_with_indexer,
            args=(0, tp_rank, kvmanager.server_recv_port,
                  model_config, cache_config, num_gpu_blocks, child_conn,
                  gpu_layout_type),
            daemon=True
        )
        tp_client_processes.append(tp_client_process)
        tp_client_process.start()

    all_gpu_blocks = []
    all_indexer_blocks = []
    for tp_rank, parent_conn in enumerate(pipe_connections):
        try:
            shared_payload = parent_conn.recv()
            if shared_payload is not None:
                if isinstance(shared_payload, dict):
                    shared_gpu_blocks = shared_payload.get("main")
                    shared_indexer_blocks = shared_payload.get("indexer")
                else:
                    shared_gpu_blocks = shared_payload
                    shared_indexer_blocks = None
                if shared_gpu_blocks is not None:
                    all_gpu_blocks.append(shared_gpu_blocks)
                    print(f"[Main Process] Received GPU blocks from TP client {tp_rank}")
                if shared_indexer_blocks is not None:
                    all_indexer_blocks.append(shared_indexer_blocks)
            parent_conn.close()
        except Exception as e:
            print(f"[Main Process] Error receiving from TP client {tp_rank}: {e}")

    gpu_kv_verifier = None
    if all_gpu_blocks and len(all_gpu_blocks) == tp_size:
        gpu_kv_layout = create_gpu_kv_layout(model_config, cache_config, num_gpu_blocks, gpu_layout_type)
        gpu_kv_verifier = GPUKVCacheVerifier(
            shared_gpu_blocks=all_gpu_blocks,
            gpu_kv_layout=gpu_kv_layout,
            tp_size=model_config.tp_size,
            tokens_per_block=cache_config.tokens_per_block,
            dtype=model_config.dtype,
            gpu_layout_type=gpu_layout_type,
        )

    indexer_kv_verifier = None
    indexer_cfg = cache_config.indexer
    if all_indexer_blocks and len(all_indexer_blocks) == tp_size and indexer_cfg is not None:
        indexer_gpu_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=model_config.num_layers,
            num_block=num_gpu_blocks,
            tokens_per_block=1,  # indexer: 1 entry per page
            num_head=indexer_cfg.num_kv_heads,
            head_size=indexer_cfg.head_size,
            is_mla=True,
        )
        indexer_kv_verifier = GPUIndexerCacheVerifier(
            shared_indexer_blocks=all_indexer_blocks,
            indexer_kv_layout=indexer_gpu_layout,
            tp_size=model_config.tp_size,
            dtype=indexer_cfg.dtype,
        )

    while not kvmanager.is_ready():
        time.sleep(1)
        flexkv_logger.info(f"waiting for flexkv ({test_label}) to be ready")
    print(f"[Test] KVManager ({test_label}) is ready")

    request_pairs = [generate_request_pair(i, block_per_request, num_gpu_blocks, tokens_per_block, 1)
                     for i in range(num_requests)]
    initial_write_num = int(num_requests * initial_write_ratio)

    print(f"[Test] Testing put flow ({test_label})...")
    for token_ids, block_ids, dp_client_id in request_pairs[:initial_write_num]:
        if gpu_kv_verifier is not None:
            gpu_kv_verifier.fill_gpu_blocks(token_ids, block_ids)
        if indexer_kv_verifier is not None:
            indexer_kv_verifier.fill_gpu_blocks(block_ids, tokens_per_block, token_ids)
        write_request = kvmanager.put_async(
            token_ids=token_ids,
            slot_mapping=block_ids_2_slot_mapping(block_ids, tokens_per_block),
            token_mask=None,
        )
        put_results = kvmanager.wait([write_request], completely=True)
        assert put_results[write_request].status == KVResponseStatus.SUCCESS
        if gpu_kv_verifier is not None:
            gpu_kv_verifier.clear_gpu_blocks(block_ids)
        if indexer_kv_verifier is not None:
            indexer_kv_verifier.clear_gpu_blocks(block_ids)
    print(f"[Test] Initial {initial_write_num} put operations completed ({test_label})")

    print(f"[Test] Testing get flow ({test_label})...")
    total_cache_hit = 0
    total_cache_miss = 0
    running_get_requests = []
    req_id2block_ids = {}
    req_id2token_ids = {}

    batch_task_ids = []
    batch_slot_mappings = []

    for i in range(min(initial_write_num, num_requests)):
        token_ids, block_ids, dp_client_id = request_pairs[i]
        slot_mapping = block_ids_2_slot_mapping(block_ids, tokens_per_block)
        request_id, _ = kvmanager.get_match(
            token_ids=token_ids,
            token_mask=None,
        )
        batch_task_ids.append(request_id)
        batch_slot_mappings.append(slot_mapping)
        req_id2block_ids[request_id] = block_ids
        req_id2token_ids[request_id] = token_ids

    if layerwise:
        # Layerwise mode: launch all GETs as a single batch so that
        # merge_to_batch_graph produces a LAYERWISE op (fused DISK2H+H2D).
        returned_ids = kvmanager.launch(
            task_ids=batch_task_ids,
            slot_mappings=batch_slot_mappings,
            as_batch=True,
            layerwise_transfer=True,
        )
        batch_id = returned_ids[0]
        batch_results = kvmanager.wait(batch_id, completely=True)
        kvresponse = batch_results[batch_id]
        assert kvresponse.status == KVResponseStatus.SUCCESS, \
            f"Layerwise batch GET failed: {kvresponse.status}"
        for idx, orig_req_id in enumerate(batch_task_ids):
            mask = kvresponse.return_mask[idx]
            total_cache_hit += mask.sum().item()
            total_cache_miss += len(mask) - mask.sum().item()
            if gpu_kv_verifier is not None:
                valid_fetched_tokens = mask.sum().item() // tokens_per_block * tokens_per_block
                if valid_fetched_tokens > 0:
                    assert gpu_kv_verifier.verify_kv_blocks(
                        req_id2token_ids[orig_req_id][:valid_fetched_tokens],
                        req_id2block_ids[orig_req_id][:valid_fetched_tokens // tokens_per_block])
            if indexer_kv_verifier is not None:
                valid_fetched_blocks = mask.sum().item() // tokens_per_block
                if valid_fetched_blocks > 0:
                    assert indexer_kv_verifier.verify_gpu_blocks(
                        req_id2block_ids[orig_req_id][:valid_fetched_blocks],
                        tokens_per_block,
                        req_id2token_ids[orig_req_id][:valid_fetched_blocks * tokens_per_block])
    else:
        # Non-layerwise: launch each GET individually
        for req_id in batch_task_ids:
            kvmanager.launch(req_id, batch_slot_mappings[batch_task_ids.index(req_id)])
            running_get_requests.append(req_id)

        if running_get_requests:
            return_results = kvmanager.wait(running_get_requests, completely=True)
            for req_id, kvresponse in return_results.items():
                assert kvresponse.status == KVResponseStatus.SUCCESS
                total_cache_hit += kvresponse.return_mask.sum().item()
                total_cache_miss += len(kvresponse.return_mask) - kvresponse.return_mask.sum().item()
                if gpu_kv_verifier is not None:
                    valid_fetched_tokens = kvresponse.return_mask.sum().item() // tokens_per_block * tokens_per_block
                    if valid_fetched_tokens > 0:
                        assert gpu_kv_verifier.verify_kv_blocks(
                            req_id2token_ids[req_id][:valid_fetched_tokens],
                            req_id2block_ids[req_id][:valid_fetched_tokens // tokens_per_block])
                if indexer_kv_verifier is not None:
                    valid_fetched_blocks = kvresponse.return_mask.sum().item() // tokens_per_block
                    if valid_fetched_blocks > 0:
                        assert indexer_kv_verifier.verify_gpu_blocks(
                            req_id2block_ids[req_id][:valid_fetched_blocks],
                            tokens_per_block,
                            req_id2token_ids[req_id][:valid_fetched_blocks * tokens_per_block])
    print(f"[Test] Get flow completed ({test_label}): hit={total_cache_hit}, miss={total_cache_miss}")

    print(f"[Test] Testing try_wait flow ({test_label})...")
    if initial_write_num < num_requests:
        token_ids, block_ids, dp_client_id = request_pairs[initial_write_num]
        if gpu_kv_verifier is not None:
            gpu_kv_verifier.fill_gpu_blocks(token_ids, block_ids)
        if indexer_kv_verifier is not None:
            indexer_kv_verifier.fill_gpu_blocks(block_ids, tokens_per_block, token_ids)
        write_request = kvmanager.put_async(
            token_ids=token_ids,
            slot_mapping=block_ids_2_slot_mapping(block_ids, tokens_per_block),
            token_mask=None,
        )
        finished = {}
        for _ in range(200):
            finished = kvmanager.try_wait([write_request])
            if write_request in finished:
                break
            time.sleep(0.1)
        assert write_request in finished, "try_wait should eventually return the completed task"
        assert finished[write_request].status == KVResponseStatus.SUCCESS
        if gpu_kv_verifier is not None:
            gpu_kv_verifier.clear_gpu_blocks(block_ids)
        if indexer_kv_verifier is not None:
            indexer_kv_verifier.clear_gpu_blocks(block_ids)
    print(f"[Test] try_wait flow completed ({test_label})")

    # Cache miss assertion: when total capacity >= GPU blocks, expect 0 miss
    enable_cpu = cache_config.enable_cpu
    enable_ssd = cache_config.enable_ssd
    num_cpu_blocks = cache_config.num_cpu_blocks
    num_ssd_blocks = cache_config.num_ssd_blocks
    if (enable_cpu and num_cpu_blocks >= num_gpu_blocks) or \
       (enable_ssd and num_ssd_blocks >= num_gpu_blocks):
        assert total_cache_miss == 0, f"Expected 0 cache miss, got {total_cache_miss}"

    shutdown_tp_client(tp_client_processes)
    kvmanager.shutdown()
    print(f"[Test] {test_label} PASSED")


@pytest.mark.parametrize(
    "model_config",
    [
        {"tp_size": 1, "dp_size": 1},
    ],    indirect=True,
)
@pytest.mark.parametrize("cache_config", [
    {'enable_cpu': True, 'enable_ssd': False, 'num_cpu_blocks': 1024},
    {'enable_cpu': True, 'enable_ssd': True, 'num_cpu_blocks': 256, 'num_ssd_blocks': 2048},
], indirect=True)
@pytest.mark.parametrize("test_config", [
    {'num_gpu_blocks': 256, 'requests_per_block': 16, 'initial_write_ratio': 0.4},
], indirect=True)
@pytest.mark.parametrize("gpu_layout_type", [0])
def test_kvmanager_with_indexer(model_config, cache_config, test_config, gpu_layout_type):
    """Test KVManager with indexer: GPU↔CPU (and optionally ↔SSD) data correctness."""
    ssd_label = "+ssd" if cache_config.enable_ssd else ""
    _run_indexer_test(model_config, cache_config, test_config, gpu_layout_type,
                      test_label=f"indexer{ssd_label}")


import ctypes
import socket
import struct
import threading

# ---- Mock SGLang eventfd client for layerwise unit tests ----

_libc = ctypes.CDLL("libc.so.6", use_errno=True)


def _sys_eventfd(initval: int = 0, flags: int = 0) -> int:
    """Create an eventfd file descriptor via libc."""
    fd = _libc.eventfd(ctypes.c_uint(initval), ctypes.c_int(flags))
    if fd == -1:
        err = ctypes.get_errno()
        raise OSError(err, f"eventfd failed: {os.strerror(err)}")
    return fd


_EFD_SEMAPHORE = 0x1


def _send_fds_via_scm(sock: socket.socket, fds: list, extra_data: bytes = b"x"):
    """Send fds via SCM_RIGHTS (mirrors SGLang's send_fds)."""
    fds_packed = struct.pack(f"{len(fds)}i", *fds)
    ancdata = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds_packed)]
    sock.sendmsg([extra_data], ancdata)


def _mock_sglang_eventfd_client(socket_path: str,
                                tp_rank: int,
                                tp_size: int,
                                num_layers: int,
                                num_counters: int = 3,
                                max_retries: int = 120,
                                retry_interval: float = 0.5):
    """Simulate SGLang sending eventfds to the LayerwiseTransferWorker.

    Runs in a background thread.  Creates real eventfds so the C++
    LayerwiseTransferGroup receives valid file descriptors.  The eventfds
    are never read by anyone in the test, but that is fine: the C++
    ``enable_eventfd_`` flag will be ``true`` and ``eventfd_write`` will
    simply increment the counter without blocking.
    """
    created_fds = []
    try:
        # Create real eventfds
        for _ in range(num_counters * num_layers):
            created_fds.append(_sys_eventfd(0, _EFD_SEMAPHORE))

        # Retry connecting until the worker process binds the socket
        sock = None
        for attempt in range(max_retries):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(socket_path)
                print(f"[MockEventfdClient] Connected to {socket_path} "
                      f"(attempt {attempt + 1})")
                break
            except (FileNotFoundError, ConnectionRefusedError):
                sock.close()
                sock = None
                time.sleep(retry_interval)

        if sock is None:
            print(f"[MockEventfdClient] FAILED to connect to {socket_path} "
                  f"after {max_retries} attempts")
            return

        metadata = struct.pack("iiii",
                               tp_rank, tp_size,
                               num_layers, num_counters)
        sock.sendall(metadata)

        # Send eventfds for each counter via SCM_RIGHTS
        fd_idx = 0
        for counter_id in range(num_counters):
            fds = created_fds[fd_idx:fd_idx + num_layers]
            fd_idx += num_layers
            _send_fds_via_scm(sock, fds, struct.pack("i", counter_id))

        # Wait for ACK
        sock.settimeout(30.0)
        ack = sock.recv(1)
        if ack and ack[0] == 1:
            print(f"[MockEventfdClient] Eventfd handshake OK "
                  f"(counters={num_counters}, layers={num_layers})")
        else:
            print(f"[MockEventfdClient] Unexpected ACK: {ack!r}")
        sock.close()
    except Exception as e:
        print(f"[MockEventfdClient] Error: {e}")
        traceback.print_exc()
    # Note: we intentionally do NOT close the eventfds here.
    # They must remain valid for the lifetime of the LayerwiseTransferGroup
    # in the worker subprocess.  They will be cleaned up when the worker
    # process exits and the OS reclaims the file descriptors.


@pytest.mark.parametrize(
    "model_config",
    [
        {"tp_size": 1, "dp_size": 1},
    ],    indirect=True,
)
@pytest.mark.parametrize("cache_config", [
    {'enable_cpu': True, 'enable_ssd': False, 'num_cpu_blocks': 1024},
    {'enable_cpu': True, 'enable_ssd': True, 'num_cpu_blocks': 256, 'num_ssd_blocks': 2048},
], indirect=True)
@pytest.mark.parametrize("test_config", [
    {'num_gpu_blocks': 256, 'requests_per_block': 16, 'initial_write_ratio': 0.4},
], indirect=True)
@pytest.mark.parametrize("gpu_layout_type", [0])
def test_kvmanager_with_indexer_layerwise(model_config, cache_config, test_config, gpu_layout_type):
    """Test KVManager with indexer in LAYERWISE mode.

    Validates the full round-trip:
      PUT: D2H + H2DISK (non-layerwise, same as normal)
      GET: LAYERWISE (fused DISK2H + H2D)
    Data correctness is verified for both the main KV cache and the
    indexer (DSA) KV cache after the round-trip.

    A background thread simulates the SGLang eventfd client so the
    LayerwiseTransferWorker can complete its initialization handshake
    without any source-code changes.
    """
    from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV

    # Save original values
    orig_layerwise_env = os.environ.get('FLEXKV_ENABLE_LAYERWISE_TRANSFER')
    orig_layerwise_flag = GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer

    # Determine the socket path that the worker will listen on.
    # For tp_size=1, pp_size=1, dp_size=1, there is no suffix.
    socket_path = os.environ.get('FLEXKV_LAYERWISE_EVENTFD_SOCKET',
                                 '/tmp/flexkv_layerwise_eventfd.sock')

    try:
        # Enable layerwise transfer
        os.environ['FLEXKV_ENABLE_LAYERWISE_TRANSFER'] = '1'
        GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer = True

        # Start mock SGLang eventfd client thread BEFORE kvmanager.start()
        # so it is ready to connect once the worker process binds the socket.
        eventfd_thread = threading.Thread(
            target=_mock_sglang_eventfd_client,
            args=(socket_path, 0, 1, model_config.num_layers),
            daemon=True,
        )
        eventfd_thread.start()

        ssd_label = "+ssd" if cache_config.enable_ssd else ""
        _run_indexer_test(model_config, cache_config, test_config, gpu_layout_type,
                          test_label=f"layerwise+indexer{ssd_label}", layerwise=True)

        eventfd_thread.join(timeout=10)
    finally:
        # Restore original environment and config
        if orig_layerwise_env is None:
            os.environ.pop('FLEXKV_ENABLE_LAYERWISE_TRANSFER', None)
        else:
            os.environ['FLEXKV_ENABLE_LAYERWISE_TRANSFER'] = orig_layerwise_env
        GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer = orig_layerwise_flag


# ============================================================
# CE Transfer Mode Tests (copy engine D2H/H2D for P800)
# ============================================================
#
# These tests exercise the three-path optimization in transfer.cc:
#   Path 0: src+dst both contiguous → single cudaMemcpyAsync
#   Path 1: few segments (≤ threshold) → per-segment cudaMemcpyAsync
#   Path 2: many segments (> threshold) → CPU gather + H2D + GPU scatter
#
# Block layout patterns are used to control which path gets selected:
#   "contiguous"  → arange(N)           → triggers Path 0 (1 segment)
#   "few_gaps"    → 4 segments × 4 blocks → triggers Path 1 (4 segments)
#   "many_gaps"   → 16 single-block segments → triggers Path 2 (16 segments)
#
# The CE (copy engine) mode is toggled via FLEXKV_USE_CE_TRANSFER_H2D /
# FLEXKV_USE_CE_TRANSFER_D2H environment variables, which must be set
# before KVManager construction because GLOBAL_CONFIG_FROM_ENV is read
# at import time.

def _generate_block_pattern(pattern_name: str, num_blocks: int, block_per_request: int):
    """Generate (block_ids, token_ids) pairs with specific contiguity patterns.

    Returns a list of (token_ids, block_ids) tuples, each with block_per_request blocks.
    The patterns control how many segments transfer.cc will see:
      - "contiguous": block_ids = arange(start, start+N) → 1 segment → Path 0
      - "few_gaps":   4 segments of 4 blocks with gaps → Path 1 (≤ 8 segments)
      - "many_gaps":  16 single-block segments with gaps → Path 2 (> 8 segments)

    All patterns use the SAME token_ids per block so that get_match can find them.
    """
    pairs = []
    tokens_per_block = 16  # default

    if pattern_name == "contiguous":
        # Sequential blocks: [0,1,2,...,N-1] → 1 segment → Path 0
        for i in range(0, num_blocks, block_per_request):
            block_ids = torch.arange(i, min(i + block_per_request, num_blocks), dtype=torch.int64)
            token_ids = torch.randint(0, 10000, (len(block_ids) * tokens_per_block,), dtype=torch.int64)
            pairs.append((token_ids, block_ids))

    elif pattern_name == "few_gaps":
        # 4 segments of 4 blocks with 8-block gaps → 4 segments → Path 1
        # e.g. block_per_request=16 → [0,1,2,3, 12,13,14,15, 24,25,26,27, 36,37,38,39]
        seg_size = block_per_request // 4
        gap = seg_size * 2
        for i in range(0, num_blocks, block_per_request * 3):
            block_ids_list = []
            for seg in range(4):
                start = i + seg * (seg_size + gap)
                if start + seg_size <= num_blocks:
                    block_ids_list.extend(range(start, start + seg_size))
            block_ids = torch.tensor(block_ids_list[:block_per_request], dtype=torch.int64)
            token_ids = torch.randint(0, 10000, (len(block_ids) * tokens_per_block,), dtype=torch.int64)
            if len(block_ids) >= block_per_request:
                pairs.append((token_ids, block_ids))

    elif pattern_name == "many_gaps":
        # Every other block with gap → 16 segments of 1 block each → Path 2
        # e.g. [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30] → 16 segments
        # With threshold=8, this is > 8 → Path 2 (merged gather/scatter)
        for i in range(0, num_blocks, block_per_request * 2):
            block_ids_list = []
            current = i
            while len(block_ids_list) < block_per_request and current < num_blocks:
                block_ids_list.append(current)
                current += 2  # skip 1 block between each
            block_ids = torch.tensor(block_ids_list[:block_per_request], dtype=torch.int64)
            token_ids = torch.randint(0, 10000, (len(block_ids) * tokens_per_block,), dtype=torch.int64)
            if len(block_ids) > 0:
                pairs.append((token_ids, block_ids))

    else:
        raise ValueError(f"Unknown pattern: {pattern_name}")

    return pairs


def _setup_ce_env(ce_h2d: bool, ce_d2h: bool):
    """Set CE transfer env vars and update GLOBAL_CONFIG_FROM_ENV in-place.

    Must be called before KVManager construction. Returns (orig_h2d, orig_d2h)
    for restoration.
    """
    from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV

    orig_h2d = GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d
    orig_d2h = GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h
    orig_env_h2d = os.environ.get('FLEXKV_USE_CE_TRANSFER_H2D')
    orig_env_d2h = os.environ.get('FLEXKV_USE_CE_TRANSFER_D2H')

    GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d = ce_h2d
    GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h = ce_d2h
    os.environ['FLEXKV_USE_CE_TRANSFER_H2D'] = '1' if ce_h2d else '0'
    os.environ['FLEXKV_USE_CE_TRANSFER_D2H'] = '1' if ce_d2h else '0'

    return (orig_h2d, orig_d2h, orig_env_h2d, orig_env_d2h)


def _restore_ce_env(saved):
    """Restore CE transfer env vars."""
    from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV

    orig_h2d, orig_d2h, orig_env_h2d, orig_env_d2h = saved
    GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d = orig_h2d
    GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h = orig_d2h
    for key, val in [('FLEXKV_USE_CE_TRANSFER_H2D', orig_env_h2d),
                     ('FLEXKV_USE_CE_TRANSFER_D2H', orig_env_d2h)]:
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def _run_ce_transfer_test(model_config, cache_config, num_gpu_blocks,
                          block_pattern, ce_h2d, ce_d2h, layerwise,
                          gpu_layout_type=0):
    """Core test: D2H (put) → H2D (get) round-trip with specific CE + pattern.

    Verifies that GPU KV cache values are preserved through the round-trip
    for the given block contiguity pattern and CE transfer mode.
    """
    tokens_per_block = cache_config.tokens_per_block
    block_per_request = 16

    skip_if_insufficient_gpus(model_config.tp_size)

    # Save and set CE env
    ce_saved = _setup_ce_env(ce_h2d, ce_d2h)

    # Save and set layerwise env
    from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
    orig_lw_flag = GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer
    orig_lw_env = os.environ.get('FLEXKV_ENABLE_LAYERWISE_TRANSFER')
    orig_sock_env = os.environ.get('FLEXKV_LAYERWISE_EVENTFD_SOCKET')
    GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer = layerwise
    os.environ['FLEXKV_ENABLE_LAYERWISE_TRANSFER'] = '1' if layerwise else '0'

    # For layerwise mode, need mock eventfd client
    eventfd_thread = None
    if layerwise:
        # Use a unique socket path to avoid conflicts with production SGLang
        import uuid
        unique_sock = f'/tmp/flexkv_ce_test_eventfd_{uuid.uuid4().hex[:8]}.sock'
        os.environ['FLEXKV_LAYERWISE_EVENTFD_SOCKET'] = unique_sock
        # Also need to set FLEXKV_LAYERWISE_EVENTFD_CONNECT_MAX_RETRIES for slow init
        os.environ.setdefault('FLEXKV_LAYERWISE_EVENTFD_CONNECT_MAX_RETRIES', '120')
        eventfd_thread = threading.Thread(
            target=_mock_sglang_eventfd_client,
            args=(unique_sock, 0, model_config.tp_size, model_config.num_layers),
            daemon=True,
        )
        eventfd_thread.start()

    ce_label = f"ce_h2d={int(ce_h2d)},ce_d2h={int(ce_d2h)}"
    lw_label = "lw" if layerwise else "nolw"
    test_label = f"CE[{ce_label},{lw_label},{block_pattern}]"

    # Disable MPS for test environments without nvidia-cuda-mps-control
    orig_mps = GLOBAL_CONFIG_FROM_ENV.enable_mps
    GLOBAL_CONFIG_FROM_ENV.enable_mps = False
    os.environ['FLEXKV_ENABLE_MPS'] = '0'

    try:
        kvmanager = KVManager(
            model_config=model_config,
            cache_config=cache_config,
            dp_client_id=0,
        )
        kvmanager.start()

        # Spawn tp_client to register GPU blocks
        mp_ctx = mp.get_context('spawn')
        pipe_connections = []
        tp_client_processes = []

        for tp_rank in range(model_config.tp_size):
            parent_conn, child_conn = mp_ctx.Pipe()
            pipe_connections.append(parent_conn)

            tp_client_process = mp_ctx.Process(
                target=run_tp_client,
                args=(0, tp_rank, kvmanager.gpu_register_port, model_config, cache_config,
                      num_gpu_blocks + tp_rank, child_conn, gpu_layout_type),
                daemon=True
            )
            tp_client_processes.append(tp_client_process)
            tp_client_process.start()

        # Collect GPU blocks
        all_gpu_blocks = []
        for tp_rank, parent_conn in enumerate(pipe_connections):
            try:
                shared_gpu_blocks = parent_conn.recv()
                if shared_gpu_blocks is not None:
                    all_gpu_blocks.append(shared_gpu_blocks)
                parent_conn.close()
            except Exception as e:
                print(f"[Main] Error receiving from TP client {tp_rank}: {e}")

        # Create verifier
        gpu_kv_verifier = None
        if all_gpu_blocks and len(all_gpu_blocks) == model_config.tp_size:
            gpu_kv_layout = create_gpu_kv_layout(
                model_config, cache_config, num_gpu_blocks, gpu_layout_type)
            gpu_kv_verifier = GPUKVCacheVerifier(
                shared_gpu_blocks=all_gpu_blocks,
                gpu_kv_layout=gpu_kv_layout,
                tp_size=model_config.tp_size,
                tokens_per_block=tokens_per_block,
                dtype=model_config.dtype,
                gpu_layout_type=gpu_layout_type,
            )

        while not kvmanager.is_ready():
            time.sleep(1)
            flexkv_logger.info(f"waiting for flexkv ({test_label}) to be ready")
        print(f"[Test] KVManager ({test_label}) is ready")

        # Generate block patterns
        request_pairs = _generate_block_pattern(block_pattern, num_gpu_blocks, block_per_request)
        assert len(request_pairs) > 0, f"No request pairs generated for pattern={block_pattern}"
        num_requests = len(request_pairs)

        # ---- Phase 1: PUT (D2H) all requests ----
        print(f"[Test] PUT phase: {num_requests} requests ({test_label})...")
        put_id2info = {}
        for idx, (token_ids, block_ids) in enumerate(request_pairs):
            if gpu_kv_verifier is not None:
                gpu_kv_verifier.fill_gpu_blocks(token_ids, block_ids)
            slot_mapping = block_ids_2_slot_mapping(block_ids, tokens_per_block)
            put_request = kvmanager.put_async(
                token_ids=token_ids,
                slot_mapping=slot_mapping,
                token_mask=None,
            )
            # Wait for each PUT individually (layerwise mode requires this)
            kvmanager.wait([put_request], completely=True)
            put_id2info[put_request] = (token_ids, block_ids, slot_mapping)

            # Clear GPU blocks after D2H
            if gpu_kv_verifier is not None:
                gpu_kv_verifier.clear_gpu_blocks(block_ids)

        print(f"[Test] PUT phase done ({num_requests} requests)")

        # ---- Phase 2: GET (H2D) all requests ----
        print(f"[Test] GET phase: {num_requests} requests ({test_label})...")

        if layerwise:
            # Batch all GETs as a single layerwise launch
            batch_task_ids = []
            batch_slot_mappings = []
            req_id2info = {}

            for idx, (token_ids, block_ids) in enumerate(request_pairs):
                slot_mapping = block_ids_2_slot_mapping(block_ids, tokens_per_block)
                request_id, _ = kvmanager.get_match(
                    token_ids=token_ids,
                    token_mask=None,
                )
                batch_task_ids.append(request_id)
                batch_slot_mappings.append(slot_mapping)
                req_id2info[request_id] = (token_ids, block_ids, slot_mapping)

            returned_ids = kvmanager.launch(
                task_ids=batch_task_ids,
                slot_mappings=batch_slot_mappings,
                as_batch=True,
                layerwise_transfer=True,
            )
            batch_id = returned_ids[0]
            batch_results = kvmanager.wait(batch_id, completely=True)
            kvresponse = batch_results[batch_id]
            assert kvresponse.status == KVResponseStatus.SUCCESS, \
                f"Layerwise batch GET failed: {kvresponse.status}"

            # Verify each request
            return_masks = kvresponse.return_mask
            if not isinstance(return_masks, (list, tuple)):
                # Single mask case - treat as one batch
                return_masks = [return_masks]
            for idx, orig_req_id in enumerate(batch_task_ids):
                if idx >= len(return_masks):
                    print(f"[WARN] return_mask has only {len(return_masks)} entries, expected {len(batch_task_ids)}")
                    break
                mask = return_masks[idx]
                token_ids, block_ids, _ = req_id2info[orig_req_id]
                valid_tokens = mask.sum().item() // tokens_per_block * tokens_per_block
                if valid_tokens > 0 and gpu_kv_verifier is not None:
                    assert gpu_kv_verifier.verify_kv_blocks(
                        token_ids[:valid_tokens],
                        block_ids[:valid_tokens // tokens_per_block])

        else:
            # Non-layerwise: launch each GET individually
            running_get_requests = []
            req_id2info = {}

            for idx, (token_ids, block_ids) in enumerate(request_pairs):
                slot_mapping = block_ids_2_slot_mapping(block_ids, tokens_per_block)
                request_id, _ = kvmanager.get_match(
                    token_ids=token_ids,
                    token_mask=None,
                )
                kvmanager.launch(request_id, slot_mapping)
                running_get_requests.append(request_id)
                req_id2info[request_id] = (token_ids, block_ids, slot_mapping)

            # Wait for all gets
            return_results = kvmanager.wait(running_get_requests, completely=True)

            for req_id, kvresponse in return_results.items():
                assert kvresponse.status == KVResponseStatus.SUCCESS, \
                    f"GET failed for req_id={req_id}: {kvresponse.status}"
                token_ids, block_ids, _ = req_id2info[req_id]
                valid_tokens = kvresponse.return_mask.sum().item() // tokens_per_block * tokens_per_block
                if valid_tokens > 0 and gpu_kv_verifier is not None:
                    assert gpu_kv_verifier.verify_kv_blocks(
                        token_ids[:valid_tokens],
                        block_ids[:valid_tokens // tokens_per_block])

        print(f"[Test] GET phase done ({test_label})")
        print(f"[Test] {test_label} PASSED")

        shutdown_tp_client(tp_client_processes)
        kvmanager.shutdown()

    finally:
        # Restore env
        _restore_ce_env(ce_saved)
        GLOBAL_CONFIG_FROM_ENV.enable_mps = orig_mps
        os.environ['FLEXKV_ENABLE_MPS'] = '1' if orig_mps else '0'
        if orig_lw_env is None:
            os.environ.pop('FLEXKV_ENABLE_LAYERWISE_TRANSFER', None)
        else:
            os.environ['FLEXKV_ENABLE_LAYERWISE_TRANSFER'] = orig_lw_env
        GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer = orig_lw_flag
        # Restore socket env
        if orig_sock_env is None:
            os.environ.pop('FLEXKV_LAYERWISE_EVENTFD_SOCKET', None)
        else:
            os.environ['FLEXKV_LAYERWISE_EVENTFD_SOCKET'] = orig_sock_env

        if eventfd_thread is not None:
            eventfd_thread.join(timeout=5)


@pytest.mark.parametrize(
    "model_config",
    [
        {"tp_size": 1, "dp_size": 1, "use_mla": True},
    ], indirect=True,
)
@pytest.mark.parametrize("cache_config", [
    {'enable_cpu': True, 'enable_ssd': False, 'num_cpu_blocks': 1024},
], indirect=True)
@pytest.mark.parametrize("test_config", [
    {'num_gpu_blocks': 128, 'requests_per_block': 16, 'initial_write_ratio': 0.4},
], indirect=True)
@pytest.mark.parametrize("block_pattern", [
    "contiguous",
    "few_gaps",
    "many_gaps",
])
@pytest.mark.parametrize("ce_mode", [
    {"ce_h2d": True, "ce_d2h": True},
    {"ce_h2d": False, "ce_d2h": False},
    {"ce_h2d": True, "ce_d2h": False},
    {"ce_h2d": False, "ce_d2h": True},
])
def test_kvmanager_ce_transfer(model_config, cache_config, test_config,
                               block_pattern, ce_mode):
    """Test D2H/H2D round-trip correctness under different CE modes and block patterns.

    This test is designed to catch precision bugs in the copy-engine transfer
    paths (transfer.cc Path 0/1/2) by varying:
      - Block contiguity: contiguous (Path 0), few_gaps (Path 1), many_gaps (Path 2)
      - CE mode: both on, both off, H2D-only, D2H-only

    The test performs:
      1. Fill GPU KV cache with deterministic hash values
      2. PUT (D2H) to FlexKV CPU cache
      3. Clear GPU KV cache
      4. GET (H2D) to restore from FlexKV CPU cache
      5. Verify GPU KV cache matches original values
    """
    num_gpu_blocks = test_config["num_gpu_blocks"]
    _run_ce_transfer_test(
        model_config=model_config,
        cache_config=cache_config,
        num_gpu_blocks=num_gpu_blocks,
        block_pattern=block_pattern,
        ce_h2d=ce_mode["ce_h2d"],
        ce_d2h=ce_mode["ce_d2h"],
        layerwise=False,
        gpu_layout_type=0,
    )


@pytest.mark.parametrize(
    "model_config",
    [
        {"tp_size": 1, "dp_size": 1, "use_mla": True},
    ], indirect=True,
)
@pytest.mark.parametrize("cache_config", [
    {'enable_cpu': True, 'enable_ssd': False, 'num_cpu_blocks': 1024},
], indirect=True)
@pytest.mark.parametrize("test_config", [
    {'num_gpu_blocks': 128, 'requests_per_block': 16, 'initial_write_ratio': 0.4},
], indirect=True)
@pytest.mark.parametrize("block_pattern", [
    "contiguous",
    "few_gaps",
    "many_gaps",
])
def test_kvmanager_ce_transfer_layerwise(model_config, cache_config, test_config,
                                         block_pattern):
    """Test D2H/H2D round-trip with CE mode + layerwise transfer.

    Layerwise mode fuses DISK2H + H2D into a single layer-by-layer pipeline.
    This test uses CE mode (both H2D and D2H) with layerwise transfer enabled,
    covering the exact configuration used in the P800 production environment:
      FLEXKV_USE_CE_TRANSFER_H2D=1
      FLEXKV_USE_CE_TRANSFER_D2H=1
      FLEXKV_ENABLE_LAYERWISE_TRANSFER=1
    """
    num_gpu_blocks = test_config["num_gpu_blocks"]
    _run_ce_transfer_test(
        model_config=model_config,
        cache_config=cache_config,
        num_gpu_blocks=num_gpu_blocks,
        block_pattern=block_pattern,
        ce_h2d=True,
        ce_d2h=True,
        layerwise=True,
        gpu_layout_type=0,
    )


@pytest.mark.parametrize(
    "model_config",
    [
        {"tp_size": 1, "dp_size": 1, "use_mla": True},
    ], indirect=True,
)
@pytest.mark.parametrize("cache_config", [
    {'enable_cpu': True, 'enable_ssd': False, 'num_cpu_blocks': 1024},
], indirect=True)
@pytest.mark.parametrize("test_config", [
    {'num_gpu_blocks': 128, 'requests_per_block': 16, 'initial_write_ratio': 0.4},
], indirect=True)
@pytest.mark.parametrize("block_pattern", [
    "contiguous",
    "few_gaps",
    "many_gaps",
])
@pytest.mark.parametrize("transfer_threshold", [1, 8, 1000])
def test_kvmanager_ce_transfer_threshold(model_config, cache_config, test_config,
                                         block_pattern, transfer_threshold):
    """Test D2H/H2D correctness with varying XSGL_TRANSFER_SEGMENT_THRESHOLD.

    This forces different path selections for the same block pattern:
      - threshold=1:    Path 1 almost never used (most patterns → Path 2)
      - threshold=8:    Default, balanced path selection
      - threshold=1000: Path 2 almost never used (all patterns → Path 1)

    CE mode is always ON (both H2D and D2H), non-layerwise.
    """
    orig_threshold = os.environ.get('XSGL_TRANSFER_SEGMENT_THRESHOLD')
    os.environ['XSGL_TRANSFER_SEGMENT_THRESHOLD'] = str(transfer_threshold)

    num_gpu_blocks = test_config["num_gpu_blocks"]
    try:
        _run_ce_transfer_test(
            model_config=model_config,
            cache_config=cache_config,
            num_gpu_blocks=num_gpu_blocks,
            block_pattern=block_pattern,
            ce_h2d=True,
            ce_d2h=True,
            layerwise=False,
            gpu_layout_type=0,
        )
    finally:
        if orig_threshold is None:
            os.environ.pop('XSGL_TRANSFER_SEGMENT_THRESHOLD', None)
        else:
            os.environ['XSGL_TRANSFER_SEGMENT_THRESHOLD'] = orig_threshold


@pytest.mark.parametrize(
    "model_config",
    [
        {"tp_size": 1, "dp_size": 1, "use_mla": True},
    ], indirect=True,
)
@pytest.mark.parametrize("cache_config", [
    {'enable_cpu': True, 'enable_ssd': False, 'num_cpu_blocks': 1024},
], indirect=True)
@pytest.mark.parametrize("test_config", [
    {'num_gpu_blocks': 128, 'requests_per_block': 16, 'initial_write_ratio': 0.4},
], indirect=True)
@pytest.mark.parametrize("block_pattern", [
    "contiguous",
    "few_gaps",
    "many_gaps",
])
@pytest.mark.parametrize("merged_mode", [False, True])
def test_kvmanager_ce_transfer_merged(model_config, cache_config, test_config,
                                      block_pattern, merged_mode):
    """Test D2H/H2D correctness with XSGL_TRANSFER_MERGED (Path 2 force) on/off.

    When merged_mode=True, Path 1 is forced to Path 2 regardless of segment count.
    This tests the CPU gather + H2D + GPU scatter path independently.
    """
    orig_merged = os.environ.get('XSGL_TRANSFER_MERGED')
    os.environ['XSGL_TRANSFER_MERGED'] = '1' if merged_mode else '0'

    num_gpu_blocks = test_config["num_gpu_blocks"]
    try:
        _run_ce_transfer_test(
            model_config=model_config,
            cache_config=cache_config,
            num_gpu_blocks=num_gpu_blocks,
            block_pattern=block_pattern,
            ce_h2d=True,
            ce_d2h=True,
            layerwise=False,
            gpu_layout_type=0,
        )
    finally:
        if orig_merged is None:
            os.environ.pop('XSGL_TRANSFER_MERGED', None)
        else:
            os.environ['XSGL_TRANSFER_MERGED'] = orig_merged



