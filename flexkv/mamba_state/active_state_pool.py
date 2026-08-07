"""MambaActiveStatePool — GPU-side per-request recurrent state pool.

Holds full-precision (bf16) state tensors that the linear-attention kernel
(KDA/GDN/Mamba2) reads/writes every decode step. One slot = one request's
complete state across all linear layers.

This pool does NOT participate in prefix caching — it is purely the runtime
working memory. Checkpointing (snapshot → int8 → CPU) is done by
MambaCheckpointPool.

Design references: sglang MambaPool.
"""
from typing import List, Optional, Tuple

import torch

from flexkv.mamba_state.config import MambaStatePoolConfig


class MambaActiveStatePool:
    """GPU-side pool of recurrent states for running requests.

    Tensor layout (matches sglang MambaPool for kernel compatibility):
        temporal_state: [L, max_reqs, H, d_v, d_k]  bf16
        conv_state:     [L, max_reqs, *conv_shape]   bf16

    Slot 0 is reserved as dummy write target for padded tokens
    (matches sglang MambaSlotAllocator design).
    """

    def __init__(self, config: MambaStatePoolConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.max_reqs = config.num_active_slots

        L = config.num_linear_layers
        H = config.num_heads
        dv = config.head_v_dim
        dk = config.head_k_dim

        # Active recurrent states (kernel-facing, updated every step)
        self.temporal_state = torch.zeros(
            L, self.max_reqs, H, dv, dk,
            dtype=torch.bfloat16, device=device,
        )
        # Conv window (kept at full precision, updated every step)
        conv_shape = (L, self.max_reqs) + config.conv_shape
        self.conv_state = torch.zeros(
            conv_shape,
            dtype=torch.bfloat16, device=device,
        )
        # Free-list: slot 0 reserved as dummy write target
        # Usable slots: 1..max_reqs-1 (total max_reqs-1)
        self._free_slots: List[int] = list(range(self.max_reqs - 1, 0, -1))

    # --- allocation --------------------------------------------------------

    def allocate(self) -> Optional[int]:
        """Allocate a slot for a new request. Returns slot_id or None if full."""
        if not self._free_slots:
            return None
        return self._free_slots.pop()

    def free(self, slot_id: int) -> None:
        """Return a slot to the free list."""
        slot_id = int(slot_id)
        if slot_id < 1 or slot_id >= self.max_reqs:
            raise ValueError(f"Invalid slot id: {slot_id} (valid: 1..{self.max_reqs - 1})")
        if slot_id in self._free_slots:
            return  # already free
        self._free_slots.append(slot_id)

    # --- CoW core ----------------------------------------------------------

    def copy_from(self, src_slot: int, dst_slot: int) -> None:
        """Copy temporal + conv state from src to dst (CoW core).

        After this call, dst has an independent copy of src's state.
        Subsequent updates to dst do not affect src.

        The src_slot may be an active slot or a checkpoint-restored slot.
        The dst_slot must be an already-allocated active slot.
        """
        self.temporal_state[:, dst_slot].copy_(self.temporal_state[:, src_slot])
        self.conv_state[:, dst_slot].copy_(self.conv_state[:, src_slot])

    def restore_from_tensors(self, dst_slot: int, temporal: torch.Tensor, conv: torch.Tensor) -> None:
        """Load state from external tensors (e.g. dequantized checkpoint) into a slot.

        Args:
            dst_slot: target active slot
            temporal: [L, H, d_v, d_k] bf16
            conv: [L, *conv_shape] bf16
        """
        self.temporal_state[:, dst_slot].copy_(temporal)
        self.conv_state[:, dst_slot].copy_(conv)

    # --- snapshot ----------------------------------------------------------

    def snapshot(self, slot_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract a (temporal, conv) snapshot from an active slot for caching.

        Returns references (not copies) — the caller must clone if the source
        slot will continue to update (which it will during decode).
        """
        return (
            self.temporal_state[:, slot_id],  # [L, H, d_v, d_k]
            self.conv_state[:, slot_id],      # [L, *conv_shape]
        )

    def clone_snapshot(self, slot_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Clone a snapshot from an active slot (safe for async quantization).

        Use this instead of snapshot() when the source slot may be updated
        before the quantization completes.
        """
        return (
            self.temporal_state[:, slot_id].clone(),
            self.conv_state[:, slot_id].clone(),
        )

    # --- properties --------------------------------------------------------

    @property
    def num_free(self) -> int:
        return len(self._free_slots)

    @property
    def num_used(self) -> int:
        return (self.max_reqs - 1) - self.num_free  # -1 for reserved slot 0

    @property
    def num_slots(self) -> int:
        return self.max_reqs

    def reset(self) -> None:
        """Return every slot to the free list (all state dropped)."""
        self._free_slots = list(range(self.max_reqs - 1, 0, -1))
