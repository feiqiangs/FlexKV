"""Linear state support for RadixNode and RadixTreeIndex.

Extends the existing Python spec radix tree (flexkv.cache.radixtree) with
node-mounted linear-attention recurrent state checkpoint slots, mirroring
the SWA pattern.

Design:
  - Each RadixNode may carry at most one mamba_state_slot (checkpoint pool id)
  - state_slot is set at checkpoint boundaries (prompt_end / radix_branch)
  - state cannot be split (cumulative, not token-indexed)
  - tombstone: evict state but keep node structure + token KV
  - match_prefix returns prefix truncated to nearest state boundary

This module adds the linear-state fields and methods WITHOUT modifying the
existing RadixNode/RadixTreeIndex classes — it uses a mixin pattern so the
linear state support can be layered on top.

For C++ production path, equivalent fields need to be added to CRadixNode
in csrc/radix_tree.h and exposed in csrc/bindings.cpp.
"""
import time
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Node-level linear state fields (added to RadixNode via duck-typing)
# ---------------------------------------------------------------------------

# Field defaults — set on RadixNode instances that support linear state.
# These mirror the SWA fields (swa_host_slot, swa_tombstone, etc.) but for
# linear-attention recurrent state checkpoints.
LINEAR_STATE_FIELDS = {
    "mamba_state_slot": -1,           # checkpoint pool slot id (-1 = no state)
    "linear_state_tombstone": True,    # True = no live state on this node
    "mamba_state_lock_ref": 0,        # state lock ref (invariant: <= lock_cnt)
    "linear_state_last_access": 0.0,   # LRU timestamp
    "mamba_state_lru_prev": None,     # intrusive LRU doubly-linked list
    "mamba_state_lru_next": None,
    "on_mamba_state_lru": False,
}


def init_linear_state_fields(node) -> None:
    """Initialize linear-state fields on a RadixNode instance.

    Call this after creating a node (or on existing nodes when enabling
    linear state support). Uses setattr to add fields dynamically.
    """
    for name, default in LINEAR_STATE_FIELDS.items():
        if not hasattr(node, name):
            setattr(node, name, default)


def has_mamba_state(node) -> bool:
    """Check if a node has a live (non-tombstone) linear state checkpoint."""
    return (
        hasattr(node, "mamba_state_slot")
        and node.mamba_state_slot >= 0
        and not getattr(node, "linear_state_tombstone", True)
    )


# ---------------------------------------------------------------------------
# MambaStateMixin — methods to add to RadixTreeIndex
# ---------------------------------------------------------------------------

class MambaStateMixin:
    """Reference implementation: radix tree mixin for mamba state.

    NOT used in production sglang integration (UnifiedRadixCache handles
    mamba CoW/tombstone natively). Kept for future frameworks needing
    radix-tree-integrated mamba state management.

    Mixin for radix tree nodes to track mamba state.
    
    Host class must call init_mamba_state_lru() before using mamba state methods.
    """
    
    def init_mamba_state_lru(self):
        """Initialize mamba state LRU tracking. Call this in host class __init__."""
        self._mamba_state_lru_head = None
        self._mamba_state_lru_tail = None

    """Mixin providing linear-state checkpoint management on RadixTreeIndex.

    Usage:
        class MyRadixTree(RadixTreeIndex, MambaStateMixin):
            ...

    Or assign methods at runtime. The host class must have:
      - self.root_node
      - self._lock (threading lock)
      - node.has_swa() / node.lock_cnt / etc. (standard RadixNode fields)
    """

    def _mamba_state_lru_add_mru(self, node) -> None:
        """Add/move node to MRU end of the linear-state LRU list."""
        if getattr(node, "on_mamba_state_lru", False):
            self._mamba_state_lru_remove(node)
        # Insert at MRU (head)
        node.mamba_state_lru_prev = None
        node.mamba_state_lru_next = getattr(self, "_mamba_state_lru_head", None)
        if getattr(self, "_mamba_state_lru_head", None) is not None:
            getattr(self, "_mamba_state_lru_head", None).mamba_state_lru_prev = node
        self._mamba_state_lru_head = node
        node.on_mamba_state_lru = True

    def _mamba_state_lru_remove(self, node) -> None:
        """Remove node from the linear-state LRU list."""
        if not getattr(node, "on_mamba_state_lru", False):
            return
        prev = node.mamba_state_lru_prev
        nxt = node.mamba_state_lru_next
        if prev is not None:
            prev.mamba_state_lru_next = nxt
        else:
            self._mamba_state_lru_head = nxt
        if nxt is not None:
            nxt.mamba_state_lru_prev = prev
        node.mamba_state_lru_prev = None
        node.mamba_state_lru_next = None
        node.on_mamba_state_lru = False

    def _mamba_state_lru_get_lru(self):
        """Get the LRU (tail) node from the linear-state LRU list."""
        node = getattr(self, "_mamba_state_lru_head", None)
        if node is None:
            return None
        while node.mamba_state_lru_next is not None:
            node = node.mamba_state_lru_next
        return node

    def set_mamba_state(self, node, slot: int) -> None:
        """Mount a linear-state checkpoint slot on a node (store side).

        The caller guarantees the node is at a checkpoint boundary.
        A different existing slot must be explicitly unmounted first.
        """
        assert node is not self.root_node
        assert getattr(node, "mamba_state_slot", -1) < 0 or \
               node.mamba_state_slot == slot
        node.mamba_state_slot = slot
        node.linear_state_tombstone = False
        node.linear_state_last_access = time.time()
        self._mamba_state_lru_add_mru(node)
        self._mamba_state_enabled = True

    def promote_mamba_state(self, node) -> None:
        """Refresh a node's linear-state recency on read-hit."""
        if node is self.root_node or not has_mamba_state(node):
            return
        node.linear_state_last_access = time.time()
        self._mamba_state_lru_add_mru(node)

    def evict_mamba_state(self, num_to_evict: int) -> List[int]:
        """Evict linear-state slots from the LRU tail.

        Internal nodes become tombstones (state freed, node + token KV stay).
        Leaf nodes are left alone (they'll be evicted with full KV).

        Returns list of freed checkpoint slot ids.
        """
        freed_slots = []
        for _ in range(num_to_evict):
            node = self._mamba_state_lru_get_lru()
            if node is None:
                break
            self._mamba_state_lru_remove(node)
            freed_slots.append(node.mamba_state_slot)
            # Tombstone: free the slot but keep the node
            node.mamba_state_slot = -1
            node.linear_state_tombstone = True
        return freed_slots

    def free_mamba_state(self, node) -> Optional[int]:
        """Free a node's linear-state slot (when the node itself is deleted).

        Returns the freed slot id, or None if no state.
        """
        if not has_mamba_state(node):
            return None
        slot = node.mamba_state_slot
        self._mamba_state_lru_remove(node)
        node.mamba_state_slot = -1
        node.linear_state_tombstone = True
        return slot

    def find_nearest_ancestor_with_state(self, node):
        """Walk up the tree to find the nearest ancestor with a live state checkpoint.

        Used for gap recompute: when a node has no state (tombstone or never set),
        find the closest ancestor that does, restore from there, and recompute
        the gap tokens.

        Returns (ancestor_node, ancestor_token_pos) or (None, -1).
        """
        ancestor = node.parent
        while ancestor is not None and ancestor is not self.root_node:
            if has_mamba_state(ancestor):
                return ancestor, getattr(ancestor, "mamba_state_token_pos", -1)
            ancestor = ancestor.parent
        return None, -1

    @staticmethod
    def linear_state_cannot_split(old_node, new_node) -> None:
        """Enforce that linear-state does NOT split when radix node splits.

        State is cumulative (not token-indexed), so it cannot be divided.
        The state stays with old_node; new_node gets no state.

        Call this from the tree's split logic.
        """
        # new_node inherits NO state (state is cumulative, can't split)
        init_linear_state_fields(new_node)
        # old_node keeps its state (already has the fields)


# ---------------------------------------------------------------------------
# MatchResult extension
# ---------------------------------------------------------------------------

def extend_match_result_for_mamba_state(match_result, tree) -> None:
    """Extend a MatchResult with linear-state information after match_prefix.

    Walks the matched path to find the deepest node with a live linear-state
    checkpoint, and truncates the usable prefix to that node's boundary.

    This implements the "state boundary constraint": recurrent state cannot
    be restored from the middle of a node's token range, only from checkpoint
    boundaries.

    Modifies match_result in-place:
      - match_result.last_mamba_state_node: deepest node with live state
      - match_result.mamba_state_hit_blocks: block count at that node
      - match_result.usable_blocks: min(full_hit, mamba_state_hit_blocks)
        (if linear state is enabled; otherwise unchanged)
    """
    match_result.last_mamba_state_node = None
    match_result.mamba_state_hit_blocks = 0
    match_result.usable_blocks = match_result.matched_blocks

    # If linear state is not enabled, no truncation
    if not getattr(tree, "_mamba_state_enabled", False):
        return

    # Walk from the matched node up to find the deepest ancestor with live state
    last_state_node = None
    state_hit_blocks = 0
    node = match_result.last_node
    while node is not None and node is not tree.root_node:
        if has_mamba_state(node):
            last_state_node = node
            state_hit_blocks = node.block_end if hasattr(node, "block_end") else 0
            break
        node = node.parent

    match_result.last_mamba_state_node = last_state_node
    match_result.mamba_state_hit_blocks = state_hit_blocks

    # Truncate usable prefix to state boundary
    if last_state_node is not None:
        if state_hit_blocks < match_result.matched_blocks:
            match_result.usable_blocks = state_hit_blocks
            match_result.last_node = last_state_node
        else:
            match_result.usable_blocks = match_result.matched_blocks
    # else: no state found, usable = matched (already set)
