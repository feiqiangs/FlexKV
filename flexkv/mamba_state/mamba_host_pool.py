"""MambaHostPool — CPU-side slot-id allocator for mamba state checkpoints.

Same pattern as SWAHostPool: fixed-size free-list (stack) that hands out
and reclaims integer slot ids. Does NOT hold the mamba state bytes — those
live in the StorageEngine buffer, addressed by slot id.
"""
from typing import Optional


class MambaHostPool:
    """Fixed-size mamba slot-id allocator (free-list); holds no state bytes."""

    def __init__(self, num_slots: int):
        self._num_slots = num_slots
        self._free_slots = list(range(num_slots - 1, -1, -1))

    def allocate(self) -> Optional[int]:
        """Allocate a slot. Returns slot_id or None if pool is full."""
        if not self._free_slots:
            return None
        return self._free_slots.pop()

    def free(self, slot_id: int) -> None:
        """Return a slot to the free list."""
        slot_id = int(slot_id)
        if slot_id < 0 or slot_id >= self._num_slots:
            raise ValueError(f"Invalid mamba slot id: {slot_id}")
        if slot_id in self._free_slots:
            return
        self._free_slots.append(slot_id)

    def reset(self) -> None:
        """Return every slot to the free list."""
        self._free_slots = list(range(self._num_slots - 1, -1, -1))

    @property
    def num_free(self) -> int:
        return len(self._free_slots)

    @property
    def num_used(self) -> int:
        return self._num_slots - self.num_free
