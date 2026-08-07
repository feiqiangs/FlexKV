"""Unit tests for P1: radix tree linear-state extension + CoW.

Tests cover:
  - set_mamba_state / has_mamba_state
  - evict_mamba_state (tombstone)
  - LRU ordering
  - linear_state_cannot_split
  - find_nearest_ancestor_with_state (gap recompute)
  - extend_match_result_for_mamba_state (boundary truncation)

Uses a minimal mock radix tree (no real RadixTreeIndex) to keep tests
pure-Python and dependency-free.
"""
import time
from typing import Optional

import pytest

from flexkv.mamba_state.radix_extension import (
    MambaStateMixin,
    LINEAR_STATE_FIELDS,
    init_linear_state_fields,
    has_mamba_state,
    extend_match_result_for_mamba_state,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Minimal mock radix tree for testing
# ---------------------------------------------------------------------------

class MockNode:
    """Minimal RadixNode mock for linear-state tests."""
    def __init__(self, node_id: int, parent=None, block_start=0, block_end=0):
        self.node_id = node_id
        self.parent = parent
        self.block_start = block_start
        self.block_end = block_end
        self.lock_cnt = 0
        self.children: dict = {}
        # SWA fields (not used, but present for compatibility)
        self.swa_host_slot = -1
        self.swa_tombstone = True
        # Initialize linear state fields
        init_linear_state_fields(self)

    @property
    def has_swa(self):
        return self.swa_host_slot >= 0 and not self.swa_tombstone

    def __repr__(self):
        return f"MockNode({self.node_id}, slot={self.mamba_state_slot}, tomb={self.linear_state_tombstone})"


class MockMatchResult:
    """Minimal MatchResult mock."""
    def __init__(self, last_node, matched_blocks):
        self.last_node = last_node
        self.matched_blocks = matched_blocks
        self.usable_blocks = matched_blocks
        self.last_mamba_state_node = None
        self.mamba_state_hit_blocks = 0


class MockRadixTree(MambaStateMixin):
    """Minimal RadixTreeIndex mock with MambaStateMixin."""
    def __init__(self):
        self.root_node = MockNode(0)
        self._mamba_state_lru_head = None
        self._mamba_state_enabled = False
        self._lock = __import__("threading").Lock()


@pytest.fixture
def tree():
    return MockRadixTree()


@pytest.fixture
def nodes(tree):
    """Create a simple tree: root -> A -> B -> C."""
    a = MockNode(1, parent=tree.root_node, block_start=0, block_end=4)
    b = MockNode(2, parent=a, block_start=4, block_end=8)
    c = MockNode(3, parent=b, block_start=8, block_end=12)
    tree.root_node.children[1] = a
    a.children[2] = b
    b.children[3] = c
    return {"A": a, "B": b, "C": c}


# ---------------------------------------------------------------------------
# set_mamba_state / has_mamba_state
# ---------------------------------------------------------------------------

class TestSetAndGetState:

    def test_set_mamba_state(self, tree, nodes):
        """set_mamba_state mounts a checkpoint slot on a node."""
        tree.set_mamba_state(nodes["A"], slot=100)
        assert nodes["A"].mamba_state_slot == 100
        assert not nodes["A"].linear_state_tombstone
        assert has_mamba_state(nodes["A"])

    def test_has_mamba_state_false_for_unset(self, tree, nodes):
        """A fresh node (no set_mamba_state) should not have state."""
        assert not has_mamba_state(nodes["A"])

    def test_has_mamba_state_false_for_tombstone(self, tree, nodes):
        """A tombstoned node should not have state."""
        tree.set_mamba_state(nodes["A"], slot=100)
        tree.evict_mamba_state(1)
        assert not has_mamba_state(nodes["A"])

    def test_set_mamba_state_enables_tree(self, tree, nodes):
        """Setting state should enable _mamba_state_enabled flag."""
        assert not tree._mamba_state_enabled
        tree.set_mamba_state(nodes["A"], slot=100)
        assert tree._mamba_state_enabled

    def test_cannot_set_on_root(self, tree):
        """set_mamba_state on root should assert."""
        with pytest.raises(AssertionError):
            tree.set_mamba_state(tree.root_node, slot=0)

    def test_remount_same_slot_refreshes_lru(self, tree, nodes):
        """Remounting the same slot should refresh LRU recency."""
        tree.set_mamba_state(nodes["A"], slot=100)
        time.sleep(0.01)
        tree.set_mamba_state(nodes["A"], slot=100)
        # Should not raise, and state should still be live
        assert has_mamba_state(nodes["A"])


# ---------------------------------------------------------------------------
# evict_mamba_state (tombstone)
# ---------------------------------------------------------------------------

class TestEvictState:

    def test_evict_creates_tombstone(self, tree, nodes):
        """Evicting state from a node creates a tombstone (slot freed, node stays)."""
        tree.set_mamba_state(nodes["A"], slot=100)
        tree.set_mamba_state(nodes["B"], slot=200)
        freed = tree.evict_mamba_state(1)
        assert len(freed) == 1
        # One of the nodes should be tombstoned
        # LRU order: A was set first, so A is LRU
        assert freed[0] == 100
        assert not has_mamba_state(nodes["A"])
        assert nodes["A"].linear_state_tombstone
        # B should still have state
        assert has_mamba_state(nodes["B"])

    def test_evict_multiple(self, tree, nodes):
        """Evict multiple state slots at once."""
        tree.set_mamba_state(nodes["A"], slot=100)
        tree.set_mamba_state(nodes["B"], slot=200)
        tree.set_mamba_state(nodes["C"], slot=300)
        freed = tree.evict_mamba_state(2)
        assert len(freed) == 2
        # A and B should be tombstoned (LRU order)
        assert not has_mamba_state(nodes["A"])
        assert not has_mamba_state(nodes["B"])
        # C should still have state
        assert has_mamba_state(nodes["C"])

    def test_evict_more_than_available(self, tree, nodes):
        """Evicting more than available should only free what exists."""
        tree.set_mamba_state(nodes["A"], slot=100)
        freed = tree.evict_mamba_state(5)
        assert len(freed) == 1
        assert freed[0] == 100

    def test_evict_returns_freed_slots(self, tree, nodes):
        """Evicted slot ids should be returned for recycling."""
        tree.set_mamba_state(nodes["A"], slot=42)
        freed = tree.evict_mamba_state(1)
        assert 42 in freed


# ---------------------------------------------------------------------------
# LRU ordering
# ---------------------------------------------------------------------------

class TestLRUOrdering:

    def test_promote_moves_to_mru(self, tree, nodes):
        """promote_mamba_state should move node to MRU."""
        tree.set_mamba_state(nodes["A"], slot=100)
        tree.set_mamba_state(nodes["B"], slot=200)
        # A is LRU, B is MRU
        # Promote A to MRU
        tree.promote_mamba_state(nodes["A"])
        # Now B should be LRU
        freed = tree.evict_mamba_state(1)
        assert freed[0] == 200  # B evicted first

    def test_lru_order_is_insertion_order(self, tree, nodes):
        """Without promote, LRU order = insertion order."""
        tree.set_mamba_state(nodes["A"], slot=100)
        tree.set_mamba_state(nodes["B"], slot=200)
        tree.set_mamba_state(nodes["C"], slot=300)
        freed = tree.evict_mamba_state(3)
        # LRU order: A, B, C
        assert freed == [100, 200, 300]

    def test_promote_noop_for_tombstone(self, tree, nodes):
        """promote on a tombstoned node should be a no-op."""
        tree.set_mamba_state(nodes["A"], slot=100)
        tree.evict_mamba_state(1)
        # A is now tombstoned
        tree.promote_mamba_state(nodes["A"])
        # Should not raise, should not re-add to LRU
        assert not has_mamba_state(nodes["A"])


# ---------------------------------------------------------------------------
# linear_state_cannot_split
# ---------------------------------------------------------------------------

class TestCannotSplit:

    def test_split_keeps_state_on_old(self, tree, nodes):
        """When a node splits, state stays on old_node, new_node gets no state."""
        tree.set_mamba_state(nodes["A"], slot=100)
        new_node = MockNode(99, parent=nodes["A"].parent, block_start=2, block_end=4)
        MambaStateMixin.linear_state_cannot_split(nodes["A"], new_node)
        # Old node keeps state
        assert has_mamba_state(nodes["A"])
        assert nodes["A"].mamba_state_slot == 100
        # New node has no state
        assert not has_mamba_state(new_node)
        assert new_node.mamba_state_slot == -1
        assert new_node.linear_state_tombstone

    def test_split_init_fields_on_new_node(self, tree, nodes):
        """New node should have linear state fields initialized."""
        new_node = MockNode(99)
        # Before init, should not have fields
        # (MockNode.__init__ already calls init_linear_state_fields)
        MambaStateMixin.linear_state_cannot_split(nodes["A"], new_node)
        assert hasattr(new_node, "mamba_state_slot")
        assert hasattr(new_node, "linear_state_tombstone")
        assert hasattr(new_node, "mamba_state_lock_ref")


# ---------------------------------------------------------------------------
# find_nearest_ancestor_with_state (gap recompute)
# ---------------------------------------------------------------------------

class TestFindAncestor:

    def test_find_immediate_ancestor(self, tree, nodes):
        """Find ancestor with state when node itself has none."""
        tree.set_mamba_state(nodes["A"], slot=100)
        # C has no state; A is its grandparent
        ancestor, pos = tree.find_nearest_ancestor_with_state(nodes["C"])
        assert ancestor is nodes["A"]

    def test_find_parent_ancestor(self, tree, nodes):
        """Find parent with state."""
        tree.set_mamba_state(nodes["B"], slot=200)
        ancestor, pos = tree.find_nearest_ancestor_with_state(nodes["C"])
        assert ancestor is nodes["B"]

    def test_no_ancestor_with_state(self, tree, nodes):
        """No ancestor has state → returns (None, -1)."""
        ancestor, pos = tree.find_nearest_ancestor_with_state(nodes["C"])
        assert ancestor is None
        assert pos == -1

    def test_skip_tombstoned_ancestor(self, tree, nodes):
        """Should skip tombstoned ancestors and find a live one."""
        tree.set_mamba_state(nodes["A"], slot=100)
        tree.set_mamba_state(nodes["B"], slot=200)
        # Tombstone B (evict it)
        tree.evict_mamba_state(1)  # evicts A (LRU)
        # Now A is tombstoned, B has state
        ancestor, pos = tree.find_nearest_ancestor_with_state(nodes["C"])
        assert ancestor is nodes["B"]

    def test_all_ancestors_tombstoned(self, tree, nodes):
        """All ancestors tombstoned → returns (None, -1)."""
        tree.set_mamba_state(nodes["A"], slot=100)
        tree.set_mamba_state(nodes["B"], slot=200)
        tree.evict_mamba_state(2)  # evict both
        ancestor, pos = tree.find_nearest_ancestor_with_state(nodes["C"])
        assert ancestor is None


# ---------------------------------------------------------------------------
# free_mamba_state (node deletion)
# ---------------------------------------------------------------------------

class TestFreeState:

    def test_free_returns_slot(self, tree, nodes):
        """free_mamba_state should return the freed slot id."""
        tree.set_mamba_state(nodes["A"], slot=100)
        freed = tree.free_mamba_state(nodes["A"])
        assert freed == 100
        assert not has_mamba_state(nodes["A"])

    def test_free_on_no_state_returns_none(self, tree, nodes):
        """free on a node with no state returns None."""
        result = tree.free_mamba_state(nodes["A"])
        assert result is None

    def test_free_removes_from_lru(self, tree, nodes):
        """free should remove the node from LRU list."""
        tree.set_mamba_state(nodes["A"], slot=100)
        tree.set_mamba_state(nodes["B"], slot=200)
        tree.free_mamba_state(nodes["A"])
        # Only B should be in LRU now
        freed = tree.evict_mamba_state(5)
        assert freed == [200]


# ---------------------------------------------------------------------------
# extend_match_result_for_mamba_state (boundary truncation)
# ---------------------------------------------------------------------------

class TestMatchResultExtension:

    def test_match_with_state_on_last_node(self, tree, nodes):
        """When last matched node has state, usable = matched."""
        tree.set_mamba_state(nodes["C"], slot=300)
        tree._mamba_state_enabled = True
        mr = MockMatchResult(last_node=nodes["C"], matched_blocks=12)
        extend_match_result_for_mamba_state(mr, tree)
        assert mr.last_mamba_state_node is nodes["C"]
        assert mr.usable_blocks == 12

    def test_match_truncated_to_ancestor_state(self, tree, nodes):
        """When last node has no state but ancestor does, usable is truncated."""
        tree.set_mamba_state(nodes["A"], slot=100)
        tree._mamba_state_enabled = True
        # Matched up to C (12 blocks), but state only on A (4 blocks)
        mr = MockMatchResult(last_node=nodes["C"], matched_blocks=12)
        extend_match_result_for_mamba_state(mr, tree)
        assert mr.last_mamba_state_node is nodes["A"]
        assert mr.usable_blocks == 4  # truncated to A's boundary

    def test_match_no_state_anywhere(self, tree, nodes):
        """When no node has state, usable = matched (no truncation)."""
        tree._mamba_state_enabled = True
        mr = MockMatchResult(last_node=nodes["C"], matched_blocks=12)
        extend_match_result_for_mamba_state(mr, tree)
        assert mr.last_mamba_state_node is None
        assert mr.usable_blocks == 12  # unchanged

    def test_match_not_enabled_no_truncation(self, tree, nodes):
        """When linear state is not enabled, no truncation."""
        tree.set_mamba_state(nodes["A"], slot=100)
        tree._mamba_state_enabled = False  # simulate disabled
        mr = MockMatchResult(last_node=nodes["C"], matched_blocks=12)
        extend_match_result_for_mamba_state(mr, tree)
        assert mr.usable_blocks == 12  # no truncation

    def test_match_tombstoned_state_skipped(self, tree, nodes):
        """Tombstoned state should be skipped during match walk."""
        tree.set_mamba_state(nodes["A"], slot=100)
        tree.set_mamba_state(nodes["B"], slot=200)
        tree.evict_mamba_state(1)  # evict A (LRU)
        tree._mamba_state_enabled = True
        # Matched up to C, A is tombstoned, B has state
        mr = MockMatchResult(last_node=nodes["C"], matched_blocks=12)
        extend_match_result_for_mamba_state(mr, tree)
        # Should find B, not A
        assert mr.last_mamba_state_node is nodes["B"]
        assert mr.usable_blocks == 8  # B's boundary


# ---------------------------------------------------------------------------
# init_linear_state_fields
# ---------------------------------------------------------------------------

class TestInitFields:

    def test_init_sets_defaults(self):
        """init_linear_state_fields sets all defaults."""
        class Bare:
            pass
        node = Bare()
        init_linear_state_fields(node)
        assert node.mamba_state_slot == -1
        assert node.linear_state_tombstone is True
        assert node.mamba_state_lock_ref == 0
        assert node.linear_state_last_access == 0.0
        assert node.mamba_state_lru_prev is None
        assert node.mamba_state_lru_next is None
        assert node.on_mamba_state_lru is False

    def test_init_idempotent(self):
        """init_linear_state_fields is idempotent (doesn't overwrite existing)."""
        class Bare:
            pass
        node = Bare()
        node.mamba_state_slot = 42  # pre-set
        init_linear_state_fields(node)
        assert node.mamba_state_slot == 42  # not overwritten

    def test_has_mamba_state_on_bare_node(self):
        """has_mamba_state returns False on a node without fields."""
        class Bare:
            pass
        node = Bare()
        assert not has_mamba_state(node)
