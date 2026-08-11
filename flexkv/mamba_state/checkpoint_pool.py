"""MambaCheckpointPool — CPU-side store for cached linear-attention recurrent states.

Stores temporal + conv state at native dtype (bf16), no compression.
Future int8 quantization can be added as an extension.

Design references: sglang HiCache MambaPoolHost (bf16 host pool).
"""
from typing import Any, List, Optional, Tuple

import torch

from flexkv.mamba_state.config import MambaStatePoolConfig


class MambaCheckpointPool:
    """CPU-side store for cached recurrent states (native dtype, no compression).

    Tensors:
        temporal: [L, num_slots, H, d_v, d_k]  native dtype (bf16)
        conv:     [L, num_slots, *conv_shape]  native dtype (bf16)
    """

    def __init__(self, config: MambaStatePoolConfig, device: str = "cpu"):
        self.config = config
        self.device = device

        L = config.num_linear_layers
        H = config.num_heads
        dv = config.head_v_dim
        dk = config.head_k_dim
        num_slots = config.num_cpu_slots

        # temporal state at native dtype
        self.temporal = torch.empty(
            L, num_slots, H, dv, dk,
            dtype=config.temporal_dtype, device=device,
        )
        # conv window at native dtype
        # Support multi-conv-type: allocate separate buffer per conv shape
        conv_shapes = config.conv_shapes  # always a list (default_factory=list)
        self.conv = [
            torch.empty(
                (L, num_slots) + shape,
                dtype=config.conv_dtype, device=device,
            )
            for shape in conv_shapes
        ]
        self._num_conv_types = len(conv_shapes)
        # slot allocator: slot 0 reserved (dummy), usable from 1
        self._free_slots: List[int] = list(range(num_slots - 1, 0, -1))

    # --- allocation --------------------------------------------------------

    def allocate(self) -> Optional[int]:
        if not self._free_slots:
            return None
        return self._free_slots.pop()

    def free(self, slot_id: int) -> None:
        slot_id = int(slot_id)
        if slot_id < 1 or slot_id >= self.config.num_cpu_slots:
            raise ValueError(f"Invalid checkpoint slot id: {slot_id}")
        if slot_id in self._free_slots:
            return
        self._free_slots.append(slot_id)

    # --- store / load (native dtype, no compression) ----------------------

    def store_from_active(
        self,
        slot_id: int,
        temporal_bf16: torch.Tensor,
        conv_bf16,
    ) -> None:
        """Store one active snapshot into a cached slot (direct copy).

        Args:
            slot_id: checkpoint pool slot
            temporal_bf16: [L, H, d_v, d_k] from active pool
            conv_bf16: [L, *conv_shape] or List of bf16 (multi-conv-type)
        """
        self.temporal[:, slot_id].copy_(temporal_bf16)
        if isinstance(conv_bf16, (list, tuple)):
            for i, cb in enumerate(conv_bf16):
                self.conv[i][:, slot_id].copy_(cb)
        else:
            self.conv[0][:, slot_id].copy_(conv_bf16)

    def load_to_active(self, slot_id: int) -> Tuple[torch.Tensor, Any]:
        """Load one cached slot back to tensors (CoW target).

        Returns (temporal, conv) — caller copies into active slot.
        conv is a single tensor if 1 conv type, or a list if multi.
        """
        temporal_bf16 = self.temporal[:, slot_id].clone()
        if self._num_conv_types == 1:
            conv_bf16 = self.conv[0][:, slot_id].clone()
        else:
            conv_bf16 = [c[:, slot_id].clone() for c in self.conv]
        return temporal_bf16, conv_bf16

    # --- properties --------------------------------------------------------

    @property
    def num_free(self) -> int:
        return len(self._free_slots)

    @property
    def num_used(self) -> int:
        return (self.config.num_cpu_slots - 1) - self.num_free

    def reset(self) -> None:
        """Return every slot to the free list."""
        self._free_slots = list(range(self.config.num_cpu_slots - 1, 0, -1))
