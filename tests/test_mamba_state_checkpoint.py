"""Unit tests for P3: MambaCheckpointManager.

Tests cover:
  - checkpoint_active_state (store path: clone → quantize → mount)
  - restore_to_active (load path: dequantize → CoW restore)
  - should_checkpoint (strategy triggers)
  - LRU eviction when pool is full
  - gap recompute (ancestor checkpoint fallback)
  - evict_all

Uses mock radix tree (from P1 tests) + real pools (from P0).
"""
import os
import sys
import threading
from typing import Optional

from unittest.mock import MagicMock
import numpy as np
import pytest
import torch

# Mock c_ext if not available
if "flexkv.c_ext" not in sys.modules:
    sys.modules["flexkv.c_ext"] = MagicMock()

from flexkv.mamba_state.config import MambaStatePoolConfig
from flexkv.mamba_state.active_state_pool import MambaActiveStatePool
from flexkv.mamba_state.checkpoint_pool import MambaCheckpointPool
from flexkv.mamba_state.checkpoint_manager import (
    MambaCheckpointManager,
    CheckpointResult,
)
from flexkv.mamba_state.radix_extension import (
    MambaStateMixin,
    init_linear_state_fields,
    has_mamba_state,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Mock radix tree (reused from P1 tests)
# ---------------------------------------------------------------------------

class MockNode:
    def __init__(self, node_id: int, parent=None, block_start=0, block_end=0):
        self.node_id = node_id
        self.parent = parent
        self.block_start = block_start
        self.block_end = block_end
        self.lock_cnt = 0
        self.children: dict = {}
        self.swa_host_slot = -1
        self.swa_tombstone = True
        init_linear_state_fields(self)

    @property
    def has_swa(self):
        return self.swa_host_slot >= 0 and not self.swa_tombstone


class MockRadixTree(MambaStateMixin):
    def __init__(self):
        self.root_node = MockNode(0)
        self._mamba_state_lru_head = None
        self._mamba_state_enabled = False
        self._lock = threading.Lock()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(num_active=8, num_cpu=16):
    return MambaStatePoolConfig(
        num_linear_layers=4,
        num_heads=8,
        head_v_dim=16,
        head_k_dim=16,
        conv_shape=(3, 128),
        num_active_slots=num_active,
        num_cpu_slots=num_cpu,
    )


@pytest.fixture
def manager():
    config = _make_config()
    active_pool = MambaActiveStatePool(config, device="cpu")
    ckpt_pool = MambaCheckpointPool(config, device="cpu")
    tree = MockRadixTree()
    return MambaCheckpointManager(active_pool, ckpt_pool, tree)


@pytest.fixture
def tree_with_nodes(manager):
    """Create a tree: root -> A -> B -> C with an active slot filled."""
    tree = manager.tree
    a = MockNode(1, parent=tree.root_node, block_start=0, block_end=4)
    b = MockNode(2, parent=a, block_start=4, block_end=8)
    c = MockNode(3, parent=b, block_start=8, block_end=12)
    tree.root_node.children[1] = a
    a.children[2] = b
    b.children[3] = c
    # Fill an active slot with known data
    active_slot = manager.active_pool.allocate()
    manager.active_pool.temporal_state[:, active_slot] = 42.0
    manager.active_pool.conv_state[:, active_slot] = 7.0
    return {
        "A": a, "B": b, "C": c,
        "active_slot": active_slot,
    }


# ---------------------------------------------------------------------------
# checkpoint_active_state (store path)
# ---------------------------------------------------------------------------

class TestCheckpointStore:

    def test_checkpoint_creates_slot_and_mounts(self, manager, tree_with_nodes):
        """Checkpoint should allocate a pool slot and mount on radix node."""
        result = manager.checkpoint_active_state(
            tree_with_nodes["active_slot"],
            tree_with_nodes["A"],
        )
        assert result.success
        assert result.checkpoint_slot >= 1
        assert has_mamba_state(tree_with_nodes["A"])
        assert tree_with_nodes["A"].mamba_state_slot == result.checkpoint_slot

    def test_checkpoint_quantizes_state(self, manager, tree_with_nodes):
        """Checkpoint should store int8-quantized state in the pool."""
        result = manager.checkpoint_active_state(
            tree_with_nodes["active_slot"],
            tree_with_nodes["A"],
        )
        # Verify the checkpoint pool has the data
        temporal, conv = manager.checkpoint_pool.load_to_active(result.checkpoint_slot)
        # Conv should be exact (not quantized)
        assert torch.allclose(conv.float(), torch.full(conv.shape, 7.0, dtype=torch.float32))
        # Temporal should be close to 42.0 (int8 quantized)
        assert torch.allclose(temporal.float(), torch.full(temporal.shape, 42.0, dtype=torch.float32), atol=1.0)

    def test_checkpoint_does_not_modify_active(self, manager, tree_with_nodes):
        """Checkpointing should not modify the active slot."""
        original = manager.active_pool.temporal_state[:, tree_with_nodes["active_slot"]].clone()
        manager.checkpoint_active_state(
            tree_with_nodes["active_slot"],
            tree_with_nodes["A"],
        )
        assert torch.equal(
            manager.active_pool.temporal_state[:, tree_with_nodes["active_slot"]],
            original,
        )

    def test_checkpoint_multiple_nodes(self, manager, tree_with_nodes):
        """Multiple checkpoints on different nodes."""
        slot = tree_with_nodes["active_slot"]
        r1 = manager.checkpoint_active_state(slot, tree_with_nodes["A"])
        r2 = manager.checkpoint_active_state(slot, tree_with_nodes["B"])
        r3 = manager.checkpoint_active_state(slot, tree_with_nodes["C"])
        assert r1.success and r2.success and r3.success
        assert r1.checkpoint_slot != r2.checkpoint_slot
        assert r2.checkpoint_slot != r3.checkpoint_slot
        assert has_mamba_state(tree_with_nodes["A"])
        assert has_mamba_state(tree_with_nodes["B"])
        assert has_mamba_state(tree_with_nodes["C"])

    def test_checkpoint_pool_full_triggers_eviction(self, manager, tree_with_nodes):
        """When checkpoint pool is full, LRU eviction should free a slot."""
        # Fill checkpoint pool almost full
        config = manager.checkpoint_pool.config
        usable = config.num_cpu_slots - 1
        slot = tree_with_nodes["active_slot"]
        # Create checkpoints to fill the pool
        nodes = [tree_with_nodes["A"], tree_with_nodes["B"], tree_with_nodes["C"]]
        created = 0
        for i in range(usable):
            node = MockNode(100 + i, parent=manager.tree.root_node, block_end=i)
            r = manager.checkpoint_active_state(slot, node)
            assert r.success
            created += 1
        # Pool should be full now
        assert manager.num_free_checkpoint_slots == 0
        # One more checkpoint should trigger eviction
        node = MockNode(999, parent=manager.tree.root_node, block_end=999)
        r = manager.checkpoint_active_state(slot, node)
        assert r.success  # eviction freed a slot


# ---------------------------------------------------------------------------
# restore_to_active (load path, CoW)
# ---------------------------------------------------------------------------

class TestRestore:

    def test_restore_into_new_slot(self, manager, tree_with_nodes):
        """Restore checkpoint into a new active slot (CoW)."""
        # Create checkpoint
        src_slot = tree_with_nodes["active_slot"]
        result = manager.checkpoint_active_state(src_slot, tree_with_nodes["A"])
        # Restore into a new slot
        dst_slot = manager.active_pool.allocate()
        success = manager.restore_to_active(result.checkpoint_slot, dst_slot)
        assert success
        # dst should have ~42.0 (int8 quantized)
        assert torch.allclose(
            manager.active_pool.temporal_state[:, dst_slot].float(),
            torch.full_like(manager.active_pool.temporal_state[:, dst_slot].float(), 42.0),
            atol=1.0,
        )

    def test_restore_does_not_modify_checkpoint(self, manager, tree_with_nodes):
        """Restoring should not modify the checkpoint pool data."""
        src_slot = tree_with_nodes["active_slot"]
        result = manager.checkpoint_active_state(src_slot, tree_with_nodes["A"])
        # Load once
        temporal1, _ = manager.checkpoint_pool.load_to_active(result.checkpoint_slot)
        # Restore into a slot
        dst_slot = manager.active_pool.allocate()
        manager.restore_to_active(result.checkpoint_slot, dst_slot)
        # Modify the active slot
        manager.active_pool.temporal_state[:, dst_slot] = 0.0
        # Load again — should be unchanged
        temporal2, _ = manager.checkpoint_pool.load_to_active(result.checkpoint_slot)
        assert torch.equal(temporal1, temporal2)


# ---------------------------------------------------------------------------
# should_checkpoint (strategy triggers)
# ---------------------------------------------------------------------------

class TestStrategy:

    def test_prompt_end_strategy(self, manager):
        assert manager.should_checkpoint("prompt_end", is_prompt_end=True)
        assert not manager.should_checkpoint("prompt_end", is_prompt_end=False)
        assert not manager.should_checkpoint("prompt_end", is_radix_branch=True)

    def test_radix_branch_strategy(self, manager):
        assert manager.should_checkpoint("radix_branch", is_radix_branch=True)
        assert not manager.should_checkpoint("radix_branch", is_radix_branch=False)
        assert not manager.should_checkpoint("radix_branch", is_prompt_end=True)

    def test_all_strategy(self, manager):
        assert manager.should_checkpoint("all", is_prompt_end=True)
        assert manager.should_checkpoint("all", is_radix_branch=True)
        assert not manager.should_checkpoint("all", is_prompt_end=False, is_radix_branch=False)

    def test_unknown_strategy(self, manager):
        assert not manager.should_checkpoint("unknown", is_prompt_end=True)


# ---------------------------------------------------------------------------
# gap recompute
# ---------------------------------------------------------------------------

class TestGapRecompute:

    def test_direct_hit_no_gap(self, manager, tree_with_nodes):
        """When matched node has checkpoint, no gap recompute needed."""
        slot = tree_with_nodes["active_slot"]
        manager.checkpoint_active_state(slot, tree_with_nodes["B"])
        dst = manager.active_pool.allocate()
        success, ancestor, ckpt = manager.restore_with_gap_recompute(
            tree_with_nodes["B"], dst
        )
        assert success
        assert ancestor is None  # no gap recompute

    def test_ancestor_hit_needs_gap(self, manager, tree_with_nodes):
        """When matched node has no checkpoint, find ancestor."""
        slot = tree_with_nodes["active_slot"]
        manager.checkpoint_active_state(slot, tree_with_nodes["A"])
        dst = manager.active_pool.allocate()
        success, ancestor, ckpt = manager.restore_with_gap_recompute(
            tree_with_nodes["C"], dst
        )
        assert success
        assert ancestor is tree_with_nodes["A"]  # gap recompute from A to C

    def test_no_checkpoint_anywhere(self, manager, tree_with_nodes):
        """No checkpoint anywhere → full recompute."""
        dst = manager.active_pool.allocate()
        success, ancestor, ckpt = manager.restore_with_gap_recompute(
            tree_with_nodes["C"], dst
        )
        assert not success
        assert ancestor is None

    def test_tombstoned_ancestor_skipped(self, manager, tree_with_nodes):
        """Tombstoned ancestors should be skipped."""
        slot = tree_with_nodes["active_slot"]
        # Checkpoint A and B
        manager.checkpoint_active_state(slot, tree_with_nodes["A"])
        manager.checkpoint_active_state(slot, tree_with_nodes["B"])
        # Evict A (LRU)
        manager._evict_to_free_slots(1)
        # A is now tombstoned
        assert not has_mamba_state(tree_with_nodes["A"])
        # Restore for C → should find B, not A
        dst = manager.active_pool.allocate()
        success, ancestor, ckpt = manager.restore_with_gap_recompute(
            tree_with_nodes["C"], dst
        )
        assert success
        assert ancestor is tree_with_nodes["B"]


# ---------------------------------------------------------------------------
# eviction
# ---------------------------------------------------------------------------

class TestEviction:

    def test_evict_all(self, manager, tree_with_nodes):
        """evict_all should free all checkpoints."""
        slot = tree_with_nodes["active_slot"]
        manager.checkpoint_active_state(slot, tree_with_nodes["A"])
        manager.checkpoint_active_state(slot, tree_with_nodes["B"])
        manager.checkpoint_active_state(slot, tree_with_nodes["C"])
        assert manager.num_checkpoints == 3
        freed = manager.evict_all()
        assert freed == 3
        assert manager.num_checkpoints == 0
        assert not has_mamba_state(tree_with_nodes["A"])
        assert not has_mamba_state(tree_with_nodes["B"])
        assert not has_mamba_state(tree_with_nodes["C"])

    def test_evict_returns_freed_slots(self, manager, tree_with_nodes):
        """Evicted slots should be returned to checkpoint pool."""
        slot = tree_with_nodes["active_slot"]
        r = manager.checkpoint_active_state(slot, tree_with_nodes["A"])
        freed = manager._evict_to_free_slots(1)
        assert freed == 1
        # The freed slot should be reusable
        assert manager.num_free_checkpoint_slots > 0


# ---------------------------------------------------------------------------
# properties
# ---------------------------------------------------------------------------

class TestProperties:

    def test_num_checkpoints(self, manager, tree_with_nodes):
        slot = tree_with_nodes["active_slot"]
        assert manager.num_checkpoints == 0
        manager.checkpoint_active_state(slot, tree_with_nodes["A"])
        assert manager.num_checkpoints == 1
        manager.checkpoint_active_state(slot, tree_with_nodes["B"])
        assert manager.num_checkpoints == 2

    def test_num_free_decreases(self, manager, tree_with_nodes):
        initial_free = manager.num_free_checkpoint_slots
        slot = tree_with_nodes["active_slot"]
        manager.checkpoint_active_state(slot, tree_with_nodes["A"])
        assert manager.num_free_checkpoint_slots == initial_free - 1
