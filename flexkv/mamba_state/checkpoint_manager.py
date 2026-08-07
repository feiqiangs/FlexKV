"""Checkpoint strategy for linear-attention recurrent state.

Integrates P0 (pools) and P1 (radix extension) to implement:
  - prompt_end: snapshot state when a request's prompt is fully processed
  - radix_branch: snapshot state at radix tree fork points
  - clone snapshot → int8 quantize → store in checkpoint pool → mount on radix node
  - LRU eviction when checkpoint pool is full
  - gap recompute: when no checkpoint at matched boundary, find nearest
    ancestor with checkpoint, restore from there, recompute the gap

This module provides the orchestration logic; the actual transfer (D2H/H2D)
is handled by MambaStateOpConstructor (P2) and the transfer engine.
"""
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import threading
import torch

from flexkv.mamba_state.config import MambaStatePoolConfig
from flexkv.mamba_state.active_state_pool import MambaActiveStatePool
from flexkv.mamba_state.checkpoint_pool import MambaCheckpointPool
from flexkv.mamba_state.radix_extension import (
    MambaStateMixin,
    has_mamba_state,
    init_linear_state_fields,
)


@dataclass
class CheckpointResult:
    """Result of a checkpoint operation."""
    success: bool
    checkpoint_slot: int = -1
    error: str = ""


class MambaCheckpointManager:
    """Reference implementation: radix-tree-mounted checkpoint manager.

    NOT used in production — production uses MambaStateConnectorBase which
    manages checkpoints via SHA256 prefix-hash + OrderedDict LRU,
    independent of the radix tree. This module is kept for future
    frameworks that need radix-tree-integrated checkpoint management.
    """
    """Manages checkpoint lifecycle for linear-attention recurrent state.

    Orchestrates:
      1. Snapshot: clone active state → quantize int8 → store in checkpoint pool
      2. Mount: attach checkpoint slot to radix tree node
      3. Eviction: LRU eviction when checkpoint pool is full
      4. Restore: load checkpoint → dequantize → restore to active slot (CoW)
      5. Gap recompute: find nearest ancestor checkpoint when no direct hit

    The manager holds references to:
      - active_pool: MambaActiveStatePool (GPU, bf16)
      - checkpoint_pool: MambaCheckpointPool (CPU, int8)
      - tree: RadixTreeIndex with MambaStateMixin
    """

    def __init__(
        self,
        active_pool: MambaActiveStatePool,
        checkpoint_pool: MambaCheckpointPool,
        tree: MambaStateMixin,
    ):
        self.active_pool = active_pool
        self.checkpoint_pool = checkpoint_pool
        self.tree = tree

    # --- checkpoint creation (store path) ----------------------------------

    def checkpoint_active_state(
        self,
        active_slot: int,
        radix_node,
        strategy: str = "prompt_end",
    ) -> CheckpointResult:
        """Snapshot active state, quantize, store, and mount on radix node.

        Uses clone-then-quantize (MVP approach C): the active state is
        cloned before quantization so the active slot can continue updating
        while quantization proceeds asynchronously.

        Args:
            active_slot: GPU active pool slot id
            radix_node: radix tree node to mount the checkpoint on
            strategy: checkpoint strategy name (for logging/tracking)

        Returns:
            CheckpointResult with success status and slot id
        """
        # 1. Allocate a checkpoint pool slot
        ckpt_slot = self.checkpoint_pool.allocate()
        if ckpt_slot is None:
            # Pool full: evict LRU and retry
            self._evict_to_free_slots(1)
            ckpt_slot = self.checkpoint_pool.allocate()
            if ckpt_slot is None:
                return CheckpointResult(False, error="checkpoint pool exhausted")

        # 2. Clone active state (safe for async quantization)
        temporal_clone, conv_clone = self.active_pool.clone_snapshot(active_slot)

        # 3. Quantize + store
        self.checkpoint_pool.store_from_active(ckpt_slot, temporal_clone, conv_clone)

        # 4. Mount on radix node
        self.tree.set_mamba_state(radix_node, ckpt_slot)

        return CheckpointResult(True, checkpoint_slot=ckpt_slot)

    # --- checkpoint restore (load path, CoW) -------------------------------

    def restore_to_active(
        self,
        checkpoint_slot: int,
        target_active_slot: int,
    ) -> bool:
        """Load checkpoint, dequantize, restore to a new active slot (CoW).

        Args:
            checkpoint_slot: CPU checkpoint pool slot id
            target_active_slot: GPU active pool slot to restore into

        Returns:
            True on success, False on failure
        """
        temporal_bf16, conv_bf16 = self.checkpoint_pool.load_to_active(checkpoint_slot)
        self.active_pool.restore_from_tensors(target_active_slot, temporal_bf16, conv_bf16)
        return True

    # --- gap recompute -----------------------------------------------------

    def find_ancestor_checkpoint(self, node) -> Tuple[Optional[object], int]:
        """Find nearest ancestor with a live checkpoint.

        Returns (ancestor_node, checkpoint_slot) or (None, -1).
        """
        ancestor, _ = self.tree.find_nearest_ancestor_with_state(node)
        if ancestor is not None and has_mamba_state(ancestor):
            return ancestor, ancestor.mamba_state_slot
        return None, -1

    def restore_with_gap_recompute(
        self,
        matched_node,
        target_active_slot: int,
    ) -> Tuple[bool, Optional[object], int]:
        """Restore from nearest ancestor checkpoint, flagging gap recompute needed.

        When the matched node has no live checkpoint (tombstone or never set),
        walk up the tree to find the nearest ancestor with a checkpoint,
        restore from there, and return the ancestor so the caller can
        recompute the gap tokens.

        Returns:
            (success, ancestor_node_or_none, checkpoint_slot_or_minus1)
            If ancestor_node is not None, the caller must recompute the gap
            between ancestor and matched_node.
        """
        if has_mamba_state(matched_node):
            # Direct hit: restore from this node's checkpoint
            self.restore_to_active(matched_node.mamba_state_slot, target_active_slot)
            return True, None, -1  # no gap recompute needed

        # Find ancestor
        ancestor, ckpt_slot = self.find_ancestor_checkpoint(matched_node)
        if ancestor is None:
            return False, None, -1  # no checkpoint anywhere, full recompute

        # Restore from ancestor
        self.restore_to_active(ckpt_slot, target_active_slot)
        # Caller must recompute gap between ancestor and matched_node
        return True, ancestor, ckpt_slot

    # --- eviction ----------------------------------------------------------

    def _evict_to_free_slots(self, num_slots: int) -> int:
        """Evict LRU checkpoints to free at least num_slots.

        Returns actual number of slots freed.
        """
        freed = self.tree.evict_mamba_state(num_slots)
        for slot_id in freed:
            self.checkpoint_pool.free(slot_id)
        return len(freed)

    def evict_all(self) -> int:
        """Evict all checkpoints (e.g. on reset)."""
        freed_count = 0
        while True:
            freed = self._evict_to_free_slots(1)
            if freed == 0:
                break
            freed_count += freed
        return freed_count

    # --- checkpoint strategy triggers --------------------------------------

    def should_checkpoint(
        self,
        strategy: str,
        is_prompt_end: bool = False,
        is_radix_branch: bool = False,
    ) -> bool:
        """Decide whether to checkpoint based on strategy and triggers.

        Args:
            strategy: "prompt_end", "radix_branch", or "all"
            is_prompt_end: True if the request's prompt just finished
            is_radix_branch: True if this is a radix tree fork point

        Returns:
            True if a checkpoint should be created
        """
        if strategy == "prompt_end":
            return is_prompt_end
        elif strategy == "radix_branch":
            return is_radix_branch
        elif strategy == "all":
            return is_prompt_end or is_radix_branch
        return False

    # --- properties --------------------------------------------------------

    @property
    def num_checkpoints(self) -> int:
        return self.checkpoint_pool.num_used

    @property
    def num_free_checkpoint_slots(self) -> int:
        return self.checkpoint_pool.num_free
