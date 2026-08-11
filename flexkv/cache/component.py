"""Generic component system for FlexKV radix tree nodes.

Aligned with sglang's UnifiedRadixCache ComponentType / ComponentData pattern.
Each node carries a list of ComponentData, indexed by ComponentType enum.
Adding a new data type (SWA, mamba, etc.) only requires adding an enum value —
no new fields on the node.
"""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from flexkv.cache.radixtree import RadixNode


class ComponentType(IntEnum):
    """Types of cached data that can be mounted on a radix tree node."""
    FULL = 0    # Main token KV (page-granular)
    SWA = 1     # Sliding window attention KV (page-granular)
    MAMBA = 2   # Mamba/linear-attn recurrent state (checkpoint-granular)


_NUM_COMPONENT_TYPES = len(ComponentType)


@dataclass
class ComponentData:
    """Per-component state on a radix tree node.

    Mirrors sglang's ComponentData — holds slot ID, tombstone flag, lock ref,
    and LRU pointers for one component type on one node.
    """
    slot_id: int = -1               # Host pool slot id (-1 = not allocated)
    tombstone: bool = True          # True = no live data (slot evicted, node shell remains)
    lock_ref: int = 0               # Eviction pin (0 = evictable)
    last_access_time: float = 0.0   # LRU timestamp

    # Intrusive LRU doubly-linked list pointers (independent per component type)
    lru_prev: Optional['RadixNode'] = None
    lru_next: Optional['RadixNode'] = None
    on_lru: bool = False

    def has_data(self) -> bool:
        """True iff this component carries live (non-tombstone) data."""
        return (not self.tombstone) and self.slot_id >= 0

    def is_locked(self) -> bool:
        """True iff this component is pinned against eviction."""
        return self.lock_ref > 0

    def reset(self) -> None:
        """Clear to default state (after eviction or node deletion)."""
        self.slot_id = -1
        self.tombstone = True
        self.lock_ref = 0
        self.last_access_time = 0.0
        self.lru_prev = None
        self.lru_next = None
        self.on_lru = False
