"""End-to-end-ish integration tests for the mooncake-store adapter using an
in-memory **fake** ``mooncake.store.MooncakeDistributedStore``.

No real RDMA fabric, GPU, or mooncake SDK is required: a fake store backed by
a python dict is injected into ``sys.modules['mooncake.store']`` *before* the
client module is imported.  This exercises the real ``MooncakeStoreClient``
put / exists / get plumbing (de-dup, longest-prefix, success decoding) and the
real ``MooncakeStoreCacheEngine.match`` joint-existence logic against that
fake store.

The fake store models ``batch_put_from`` / ``batch_get_into`` /
``batch_is_exist`` over raw ``(ptr, size)`` ranges, copying bytes through the
process address space via ``ctypes`` so a put->get round-trip is verified
byte-for-byte.

Tests that genuinely need real RDMA / GPU hardware are marked ``mooncake`` and
``skipif`` (pending the real test environment, available tomorrow).
"""
import _mooncake_store_testkit as kit  # installs fake flexkv.c_ext

import ctypes
import os
import sys
import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fake mooncake.store.MooncakeDistributedStore (in-memory, ptr-based)
# ---------------------------------------------------------------------------
class _FakeMooncakeDistributedStore:
    def __init__(self):
        # key -> bytes
        self._data = {}
        self._registered = []

    def setup(self, *args, **kwargs):
        return 0

    def register_buffer(self, ptr, size):
        self._registered.append((ptr, size))
        return 0

    def unregister_buffer(self, ptr):
        return 0

    # --- single-key helpers (used by warm_up) -----------------------------
    def put(self, key, value):
        self._data[key] = bytes(value)
        return 0

    def get(self, key):
        return self._data.get(key, b"")

    def is_exist(self, key):
        return 1 if key in self._data else 0

    # --- batch helpers -----------------------------------------------------
    def batch_put_from(self, keys, ptrs, sizes):
        results = []
        for k, p, s in zip(keys, ptrs, sizes):
            buf = (ctypes.c_char * s).from_address(int(p))
            self._data[k] = bytes(buf)
            results.append(0)
        return results

    def batch_get_into(self, keys, ptrs, sizes):
        results = []
        for k, p, s in zip(keys, ptrs, sizes):
            if k not in self._data:
                results.append(-1)
                continue
            payload = self._data[k][:s]
            dst = (ctypes.c_char * len(payload)).from_address(int(p))
            dst[: len(payload)] = payload
            results.append(len(payload))
        return results

    def batch_is_exist(self, keys):
        return [1 if k in self._data else 0 for k in keys]

    def _batch_exist(self, keys):
        return self.batch_is_exist(keys)

    def remove_all(self):
        self._data.clear()


def _install_fake_mooncake_store():
    if "mooncake.store" in sys.modules:
        return
    mooncake_pkg = types.ModuleType("mooncake")
    store_mod = types.ModuleType("mooncake.store")
    store_mod.MooncakeDistributedStore = _FakeMooncakeDistributedStore
    mooncake_pkg.store = store_mod
    sys.modules["mooncake"] = mooncake_pkg
    sys.modules["mooncake.store"] = store_mod


_install_fake_mooncake_store()

# Now it is safe to import the adapter (its lazy ``import mooncake.store`` will
# resolve to our fake).
from flexkv.external.mooncake_store_utils import (  # noqa: E402
    MooncakeStoreConfig,
    MooncakeStoreClient,
    MooncakeStoreCacheEngine,
)
from flexkv.external.mooncake_store_keys import PoolKind, PoolSpec, build_key  # noqa: E402
from flexkv.common.block import SequenceMeta  # noqa: E402


TOKENS_PER_BLOCK = 16


@pytest.fixture
def client(monkeypatch):
    """A fully-wired MooncakeStoreClient over the fake store.

    ``check_server`` performs an HTTP poll against the master metrics port; we
    stub it out so no network access happens in CI.
    """
    cfg = MooncakeStoreConfig(
        master_addr="127.0.0.1:50051",
        protocol="rdma",
        global_segment_size=1024 * 1024,
    )
    # Avoid the HTTP server poll in check_server (offline).
    monkeypatch.setattr(MooncakeStoreClient, "check_server", lambda self: None)
    c = MooncakeStoreClient(cfg, query_only=False)
    return c


@pytest.fixture
def buffer_and_ptrs():
    """Allocate a raw byte buffer and return per-block (ptr, size) lists."""
    num_blocks = 4
    block_size_bytes = 64
    total = num_blocks * block_size_bytes
    buf = (ctypes.c_char * total)()
    base = ctypes.addressof(buf)
    ptrs = [base + i * block_size_bytes for i in range(num_blocks)]
    sizes = [block_size_bytes] * num_blocks
    return buf, ptrs, sizes, num_blocks, block_size_bytes


def _seq(num_blocks):
    token_ids = np.arange(num_blocks * TOKENS_PER_BLOCK, dtype=np.int64)
    return SequenceMeta(token_ids=token_ids, tokens_per_block=TOKENS_PER_BLOCK)


# ---------------------------------------------------------------------------
# Client: put -> exists -> get round-trip
# ---------------------------------------------------------------------------
def test_client_batch_put_then_exists(client, buffer_and_ptrs):
    buf, ptrs, sizes, num_blocks, _ = buffer_and_ptrs
    keys = [f"rt_{i}" for i in range(3)]
    ok = client.batch_put(keys, ptrs[:3], sizes[:3])
    assert all(ok)

    # all 3 present -> longest prefix == 3
    assert client.batch_exists(keys) == 3
    # a hole in the middle truncates the prefix to 1
    assert client.batch_exists([keys[0], "missing", keys[2]]) == 1


def test_client_put_get_roundtrip_byte_for_byte(client, buffer_and_ptrs):
    buf, ptrs, sizes, num_blocks, block_size_bytes = buffer_and_ptrs
    keys = [f"xfer_{i}" for i in range(num_blocks)]

    # seed each block with a recognisable byte pattern.
    # ctypes char arrays expose a memoryview with format '<c'; cast to 'B'
    # (unsigned bytes) so slice assignment / comparison is supported.
    mv = memoryview(buf).cast("B")
    for i in range(num_blocks):
        mv[i * block_size_bytes : (i + 1) * block_size_bytes] = bytes(
            [(i + 1) % 256] * block_size_bytes
        )

    assert all(client.batch_put(keys, ptrs, sizes))

    # wipe local buffer, then read back from the (fake) store
    mv[:] = bytes(len(buf))
    assert all(client.batch_get(keys, ptrs, sizes))
    for i in range(num_blocks):
        chunk = bytes(mv[i * block_size_bytes : (i + 1) * block_size_bytes])
        assert chunk == bytes([(i + 1) % 256] * block_size_bytes), (
            f"block {i} content mismatch after round-trip"
        )


def test_client_batch_put_is_idempotent(client, buffer_and_ptrs):
    buf, ptrs, sizes, _, _ = buffer_and_ptrs
    keys = ["dup_0", "dup_1"]
    assert all(client.batch_put(keys, ptrs[:2], sizes[:2]))
    # second put of the same keys: dedup path returns success without error
    assert all(client.batch_put(keys, ptrs[:2], sizes[:2]))


# ---------------------------------------------------------------------------
# CacheEngine.match end-to-end over the fake store (single + multi pool)
# ---------------------------------------------------------------------------
def _make_engine_over_client(client, pool_specs):
    engine = object.__new__(MooncakeStoreCacheEngine)
    engine.tokens_per_block = TOKENS_PER_BLOCK
    engine.pp_rank = 0
    engine.pp_size = 1
    engine.node_layer_start = 0
    engine.node_layer_end = 0
    engine.total_layers = 0
    engine.pool_specs = pool_specs
    engine.hit_pool_specs = [s for s in pool_specs if s.required_for_hit]
    engine.mooncake_store_client = client
    return engine


def test_e2e_match_single_pool(client, buffer_and_ptrs):
    buf, ptrs, sizes, num_blocks, _ = buffer_and_ptrs
    seq = _seq(num_blocks)
    kv_keys = [build_key(seq.block_hashes[i], PoolKind.KV) for i in range(num_blocks)]
    assert all(client.batch_put(kv_keys, ptrs, sizes))

    engine = _make_engine_over_client(client, [PoolSpec(PoolKind.KV)])
    result = engine.match(seq)
    assert result.matched_pos == MooncakeStoreCacheEngine.MATCHED_POS
    assert result.num_matched_blocks == num_blocks


def test_e2e_match_single_pool_partial(client, buffer_and_ptrs):
    buf, ptrs, sizes, num_blocks, _ = buffer_and_ptrs
    seq = _seq(num_blocks)
    kv_keys = [build_key(seq.block_hashes[i], PoolKind.KV) for i in range(num_blocks)]
    # publish only first 2 blocks
    assert all(client.batch_put(kv_keys[:2], ptrs[:2], sizes[:2]))

    engine = _make_engine_over_client(client, [PoolSpec(PoolKind.KV)])
    assert engine.match(seq).num_matched_blocks == 2


def test_e2e_match_multi_pool_joint(client, buffer_and_ptrs):
    buf, ptrs, sizes, num_blocks, _ = buffer_and_ptrs
    seq = _seq(num_blocks)
    kv_keys = [build_key(seq.block_hashes[i], PoolKind.KV) for i in range(num_blocks)]
    idx_keys = [build_key(seq.block_hashes[i], PoolKind.INDEXER) for i in range(num_blocks)]
    assert all(client.batch_put(kv_keys, ptrs, sizes))
    # indexer only first 2 -> joint prefix truncates at 2
    assert all(client.batch_put(idx_keys[:2], ptrs[:2], sizes[:2]))

    engine = _make_engine_over_client(
        client, [PoolSpec(PoolKind.KV), PoolSpec(PoolKind.INDEXER)]
    )
    assert engine.match(seq).num_matched_blocks == 2


# ---------------------------------------------------------------------------
# Hardware-only path (real RDMA / GPU): skipped until the test environment.
# ---------------------------------------------------------------------------
_REAL_MOONCAKE = bool(os.environ.get("FLEXKV_MOONCAKE_STORE_CONFIG_PATH")) and (
    "_FakeMooncakeDistributedStore" not in repr(sys.modules.get("mooncake.store"))
)


@pytest.mark.mooncake
@pytest.mark.skipif(
    not _REAL_MOONCAKE,
    reason="needs a real mooncake-store cluster + RDMA fabric "
    "(pending the real test environment available tomorrow)",
)
def test_real_cluster_roundtrip_placeholder():  # pragma: no cover
    """Placeholder for the real-hardware round-trip.

    Run on the real test environment with a live cluster and
    ``FLEXKV_MOONCAKE_STORE_CONFIG_PATH`` pointing at a valid JSON config.
    """
    raise AssertionError("must run on real hardware; see test_mooncake_store_integration "
                         "reference suite in mooncake/FlexKV/tests")
