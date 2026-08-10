"""MambaCheckpointPool — CPU-side int8-compressed store for cached
linear-attention recurrent states.

Decouples cached states (radix-owned, idle, int8 compressed) from the active
MambaActiveStatePool (running requests, full precision, kernel-facing).

Per cached slot it holds:
  * temporal state in int8 (per-(head, k-channel) symmetric quantization)
    — ~2x more cached states than bf16, quality-safe
  * conv1d window at native dtype (tiny, W-1 tokens, not worth quantizing)

Why int8 not fp8: a cached checkpoint is loaded ONCE on a cache hit, then
decoding continues at full precision, so the only error is a single rounding
of S. The temporal state is roughly uniformly distributed, so int8
per-(head, k-channel) beats fp8-e4m3 at the same 1 byte (fp8 wastes bits on
the exponent). The scale axis (reduces over d_v) matches the per-k-channel
decay diag(alpha), so large state entries keep ~bf16 precision and error
concentrates on small entries that barely affect the readout.

Design references: sglang MambaCheckpointPool / Int8CheckpointStore.
"""
from typing import Any, List, Optional, Tuple

import torch

from flexkv.mamba_state.config import MambaStatePoolConfig


class MambaCheckpointPool:
    """CPU-side int8-compressed store for cached recurrent states.

    Model-agnostic: treats state as structured tensors (not opaque bytes),
    applying per-(head, k-channel) symmetric int8 quantization.

    Tensors:
        qdata: [L, num_slots, H, d_v, d_k]  int8         (quantized temporal)
        scale: [L, num_slots, H, 1,   d_k]  scale_dtype  (per head,k-channel)
        conv:  [L, num_slots, *conv_shape]  bf16         (not quantized)
    """

    QMAX = 127

    def __init__(self, config: MambaStatePoolConfig, device: str = "cpu"):
        self.config = config
        self.device = device

        L = config.num_linear_layers
        H = config.num_heads
        dv = config.head_v_dim
        dk = config.head_k_dim
        num_slots = config.num_cpu_slots

        # int8 quantized temporal state
        self.qdata = torch.empty(
            L, num_slots, H, dv, dk,
            dtype=config.compress_dtype, device=device,
        )
        # per-(head, k-channel) scale: reduction axis = d_v (dim=-2)
        self.scale = torch.empty(
            L, num_slots, H, 1, dk,
            dtype=config.scale_dtype, device=device,
        )
        # conv window at native dtype (not quantized)
        # Support multi-conv-type: allocate separate buffer per conv shape
        conv_shapes = config.conv_shapes if config.conv_shapes else [config.conv_shape]
        self.conv = [
            torch.empty(
                (L, num_slots) + shape,
                dtype=torch.bfloat16, device=device,
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

    # --- quantization (store path: active bf16 → cached int8) -------------

    @classmethod
    def quantize(cls, state_bf16: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize temporal state to int8.

        Args:
            state_bf16: [..., H, d_v, d_k] bf16/fp16/fp32

        Returns:
            qdata: [..., H, d_v, d_k] int8
            scale: [..., H, 1, d_k] same dtype as input

        Algorithm (matches sglang Int8CheckpointStore):
          1. All math in float32 to avoid low-precision intermediate loss
          2. scale = amax(|state|, dim=d_v) / 127, rounded to state dtype
             BEFORE division — so quantize and dequantize use identical scale
          3. scale reduction axis = d_v (dim=-2), aligned with per-k-channel
             decay diag(alpha)
          4. Symmetric quantization (no zero-point), clamp [-127, 127]
        """
        state_fp32 = state_bf16.to(torch.float32)
        # amax over d_v (dim=-2) -> [..., H, 1, d_k]
        amax = state_fp32.abs().amax(dim=-2, keepdim=True).clamp(min=1e-8)
        scale = (amax / cls.QMAX).to(state_bf16.dtype)  # round to source dtype first
        qdata = torch.round(state_fp32 / scale.to(torch.float32)).clamp(
            -cls.QMAX, cls.QMAX
        ).to(torch.int8)
        return qdata, scale

    @staticmethod
    def dequantize(qdata: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize int8 back to bf16.

        Args:
            qdata: [..., H, d_v, d_k] int8
            scale: [..., H, 1, d_k] scale_dtype

        Returns:
            [..., H, d_v, d_k] bf16
        """
        return (qdata.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)

    def store_from_active(
        self,
        slot_id: int,
        temporal_bf16: torch.Tensor,
        conv_bf16,
    ) -> None:
        """Compress and store one active snapshot into a cached slot.

        Args:
            slot_id: checkpoint pool slot
            temporal_bf16: [L, H, d_v, d_k] bf16 from active pool
            conv_bf16: [L, *conv_shape] bf16 or List of bf16 (multi-conv-type)
        """
        qdata, scale = self.quantize(temporal_bf16)
        self.qdata[:, slot_id].copy_(qdata)
        self.scale[:, slot_id].copy_(scale)
        if isinstance(conv_bf16, (list, tuple)):
            for i, cb in enumerate(conv_bf16):
                self.conv[i][:, slot_id].copy_(cb)
        else:
            self.conv[0][:, slot_id].copy_(conv_bf16)

    # --- dequantization (load path: cached int8 → active bf16) ------------

    def load_to_active(self, slot_id: int) -> Tuple[torch.Tensor, Any]:
        """Decompress one cached slot back to bf16 tensors (CoW target).

        Returns (temporal_bf16, conv_bf16) — caller copies into active slot.
        conv_bf16 is a single tensor if 1 conv type, or a list if multi.
        """
        temporal_bf16 = self.dequantize(
            self.qdata[:, slot_id], self.scale[:, slot_id]
        )
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
