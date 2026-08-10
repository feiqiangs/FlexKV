"""Linear-attention recurrent state pool configuration.

Internal config auto-constructed by FlexKVConnector from sglang's MambaPool
tensor shapes at registration time. Not user-facing — no YAML needed.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


@dataclass
class MambaStatePoolConfig:
    """Configuration for linear-attention recurrent state pools.

    Model-agnostic: KDA (Kimi) and GDN (Qwen) share the same structure,
    differing only in parameter values read from sglang's MambaPool.
    """
    # --- model geometry (auto-read from sglang MambaPool) ---
    num_linear_layers: int = 0
    num_heads: int = 0
    head_v_dim: int = 0
    head_k_dim: int = 0
    conv_shape: Tuple[int, ...] = ()
    conv_shapes: List[Tuple[int, ...]] = ()  # multi-conv-type: list of shapes per conv type
    conv_shard_groups: Optional[List[int]] = None
    is_kda: bool = False

    # --- pool sizing (derived from FlexKV CacheConfig) ---
    num_active_slots: int = 256
    num_cpu_slots: int = 2048
    evict_ratio: float = 0.1

    # --- compression ---
    compress_dtype: torch.dtype = torch.int8
    scale_dtype: torch.dtype = torch.bfloat16

    @property
    def state_bytes_per_layer(self) -> int:
        """One linear layer's temporal state: [H, d_v, d_k] bf16."""
        return self.num_heads * self.head_v_dim * self.head_k_dim * 2

    @property
    def conv_bytes_per_layer(self) -> int:
        """Conv window bytes per layer (layout-dependent)."""
        elem = 1
        for s in self.conv_shape:
            elem *= s
        return elem * 2  # bf16

    @property
    def total_active_bytes(self) -> int:
        """All linear layers' active state for one request."""
        return self.num_linear_layers * (self.state_bytes_per_layer + self.conv_bytes_per_layer)

    @property
    def total_checkpoint_bytes_int8(self) -> int:
        """All linear layers' int8 compressed checkpoint (state + scale + conv)."""
        L = self.num_linear_layers
        state_int8 = L * self.num_heads * self.head_v_dim * self.head_k_dim * 1
        scale = L * self.num_heads * 1 * self.head_k_dim * 2
        conv = L * self.conv_bytes_per_layer
        return state_int8 + scale + conv

    @property
    def enabled(self) -> bool:
        return self.num_linear_layers > 0 and self.num_heads > 0
