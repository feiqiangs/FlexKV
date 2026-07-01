"""Unit tests for ``MooncakeStoreCacheEngine.match`` and its no-op interface.

The cache engine is instantiated **without** running ``__init__`` (which would
require a real ``MooncakeStoreConfig`` file and a live ``MooncakeStoreClient``).
Instead we allocate a bare instance via ``object.__new__`` and inject:

* a fake ``mooncake_store_client`` (in-memory key set)
* ``hit_pool_specs`` describing the active pools
* PP / layer-range fields (defaults -> no key suffix)

Covered:
* single-pool (KV only) longest-prefix via ``batch_exists``
* multi-pool (KV + INDEXER) joint-existence: a block counts as a hit only if
  ALL required pools have it; prefix truncates at the first missing pool
* empty sequence -> 0 matched
* ``matched_pos == "global"`` sentinel
* insert / take / lock / unlock / set_ready are no-ops
"""
import _mooncake_store_testkit as kit  # installs fake flexkv.c_ext

import numpy as np
import pytest

from flexkv.common.block import SequenceMeta
from flexkv.external.mooncake_store_keys import PoolKind, PoolSpec, build_key
from flexkv.external.mooncake_store_utils import MooncakeStoreCacheEngine


TOKENS_PER_BLOCK = 16


def _make_seq(num_blocks: int) -> SequenceMeta:
    token_ids = np.arange(num_blocks * TOKENS_PER_BLOCK, dtype=np.int64)
    seq = SequenceMeta(token_ids=token_ids, tokens_per_block=TOKENS_PER_BLOCK)
    assert seq.num_blocks == num_blocks
    return seq


def _make_engine(pool_specs, present_keys=None) -> MooncakeStoreCacheEngine:
    """Build a MooncakeStoreCacheEngine bypassing its heavy __init__."""
    engine = object.__new__(MooncakeStoreCacheEngine)
    engine.tokens_per_block = TOKENS_PER_BLOCK
    engine.pp_rank = 0
    engine.pp_size = 1
    engine.node_layer_start = 0
    engine.node_layer_end = 0
    engine.total_layers = 0
    engine.pool_specs = pool_specs
    engine.hit_pool_specs = [s for s in pool_specs if s.required_for_hit]
    engine.mooncake_store_client = kit.FakeMooncakeStoreClient(present_keys or [])
    return engine


def _kv_key(seq, i):
    return build_key(seq.block_hashes[i], PoolKind.KV)


def _idx_key(seq, i):
    return build_key(seq.block_hashes[i], PoolKind.INDEXER)


# ---------------------------------------------------------------------------
# Single-pool (KV only)
# ---------------------------------------------------------------------------
def test_match_single_pool_full_hit():
    seq = _make_seq(4)
    keys = [_kv_key(seq, i) for i in range(4)]
    engine = _make_engine([PoolSpec(PoolKind.KV)], present_keys=keys)

    result = engine.match(seq)
    assert result.matched_pos == MooncakeStoreCacheEngine.MATCHED_POS
    assert result.num_matched_blocks == 4
    assert result.num_ready_matched_blocks == 4
    assert result.physical_blocks.shape == (4,)
    assert result.physical_blocks.dtype == np.int64
    # single pool -> fast path uses batch_exists (not the impl scan)
    assert engine.mooncake_store_client.batch_exists_calls
    assert not engine.mooncake_store_client.batch_exists_impl_calls


def test_match_single_pool_partial_prefix():
    seq = _make_seq(4)
    # publish only the first 2 KV keys -> longest prefix is 2
    keys = [_kv_key(seq, i) for i in range(2)]
    engine = _make_engine([PoolSpec(PoolKind.KV)], present_keys=keys)

    result = engine.match(seq)
    assert result.num_matched_blocks == 2


def test_match_single_pool_no_hit():
    seq = _make_seq(3)
    engine = _make_engine([PoolSpec(PoolKind.KV)], present_keys=[])
    result = engine.match(seq)
    assert result.num_matched_blocks == 0


def test_match_single_pool_gap_truncates_prefix():
    """A missing block in the middle truncates the prefix at that point."""
    seq = _make_seq(4)
    # publish blocks 0,1,3 (block 2 missing) -> prefix stops at 2
    keys = [_kv_key(seq, i) for i in (0, 1, 3)]
    engine = _make_engine([PoolSpec(PoolKind.KV)], present_keys=keys)
    result = engine.match(seq)
    assert result.num_matched_blocks == 2


# ---------------------------------------------------------------------------
# Multi-pool (KV + INDEXER) joint-existence
# ---------------------------------------------------------------------------
def _multi_specs():
    return [PoolSpec(PoolKind.KV), PoolSpec(PoolKind.INDEXER)]


def test_match_multi_pool_full_joint_hit():
    seq = _make_seq(4)
    keys = [_kv_key(seq, i) for i in range(4)] + [_idx_key(seq, i) for i in range(4)]
    engine = _make_engine(_multi_specs(), present_keys=keys)
    assert len(engine.hit_pool_specs) == 2

    result = engine.match(seq)
    assert result.num_matched_blocks == 4
    # multi-pool -> uses batch_exists_impl scan, not the fast batch_exists
    assert engine.mooncake_store_client.batch_exists_impl_calls
    assert not engine.mooncake_store_client.batch_exists_calls


def test_match_multi_pool_indexer_missing_truncates():
    """KV present for all, indexer only first 2 -> joint prefix is 2."""
    seq = _make_seq(4)
    keys = [_kv_key(seq, i) for i in range(4)]
    keys += [_idx_key(seq, i) for i in range(2)]
    engine = _make_engine(_multi_specs(), present_keys=keys)

    result = engine.match(seq)
    assert result.num_matched_blocks == 2


def test_match_multi_pool_kv_missing_truncates():
    """Symmetric: indexer all, KV only first 1 -> joint prefix is 1."""
    seq = _make_seq(4)
    keys = [_kv_key(seq, i) for i in range(1)]
    keys += [_idx_key(seq, i) for i in range(4)]
    engine = _make_engine(_multi_specs(), present_keys=keys)

    result = engine.match(seq)
    assert result.num_matched_blocks == 1


def test_match_multi_pool_first_block_missing_zero():
    seq = _make_seq(4)
    # KV ok for all, but indexer missing block 0 -> 0 joint hit
    keys = [_kv_key(seq, i) for i in range(4)]
    keys += [_idx_key(seq, i) for i in range(1, 4)]
    engine = _make_engine(_multi_specs(), present_keys=keys)

    result = engine.match(seq)
    assert result.num_matched_blocks == 0


# ---------------------------------------------------------------------------
# Empty sequence
# ---------------------------------------------------------------------------
def test_match_empty_sequence_returns_zero():
    seq = _make_seq(0)
    engine = _make_engine([PoolSpec(PoolKind.KV)], present_keys=[])
    result = engine.match(seq)
    assert result.num_matched_blocks == 0
    assert result.num_ready_matched_blocks == 0
    assert result.matched_pos == MooncakeStoreCacheEngine.MATCHED_POS
    assert result.physical_blocks.shape == (0,)
    # no RPC issued for an empty sequence
    assert not engine.mooncake_store_client.batch_exists_calls
    assert not engine.mooncake_store_client.batch_exists_impl_calls


# ---------------------------------------------------------------------------
# No-op interface
# ---------------------------------------------------------------------------
def test_insert_take_and_lifecycle_are_noops():
    seq = _make_seq(2)
    engine = _make_engine([PoolSpec(PoolKind.KV)], present_keys=[])

    node = engine.insert(seq, np.arange(2, dtype=np.int64))
    assert node.size() == 0

    blocks = engine.take(5)
    assert blocks.shape == (5,)
    assert blocks.dtype == np.int64
    assert np.all(blocks == 0)

    # these must not raise
    engine.reset()
    engine.start()
    engine.stop()
    engine.lock_node(node)
    engine.unlock(node)
    engine.set_ready(node, True, 0)
    engine.recycle(np.arange(2, dtype=np.int64))
    assert engine.insert_and_publish(node) is True
