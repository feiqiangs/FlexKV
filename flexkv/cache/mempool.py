import os
from collections import deque
from typing import List, Optional

import numpy as np


class Mempool:
    def __init__(
        self,
        num_total_blocks: int,
    ):
        assert num_total_blocks > 0
        self.num_total_blocks = num_total_blocks

        self._free_mask = np.ones(self.num_total_blocks, dtype=np.bool_)
        self._num_free = num_total_blocks
        self._free_ids = self._free_mask.nonzero()[0]
        self._free_ids_offset = 0

        # [P2] Prefer allocating a physically-contiguous run of block ids.
        # When the stored prefix occupies contiguous CPU blocks, the matched
        # block ids handed back on load are contiguous too, so the CE transfer
        # hits Path 0 (one big memcpy per layer, no CPU gather) instead of the
        # scattered Path 2 gather. Controlled by FLEXKV_CONTIGUOUS_ALLOC (default
        # on). Falls back to the original front-cursor allocation when no run of
        # the requested length exists.
        self._prefer_contiguous = (
            os.environ.get("FLEXKV_CONTIGUOUS_ALLOC", "1") != "0"
        )

    def reset(self) -> None:
        self._free_mask.fill(True)
        self._num_free = self.num_total_blocks
        self._free_ids = self._free_mask.nonzero()[0]
        self._free_ids_offset = 0

    def _find_contiguous_run(self, num: int) -> Optional[int]:
        """Return the start id of a free contiguous run of length >= num, else None.

        Vectorized scan over the free mask (O(num_total_blocks)). Only used when
        FLEXKV_CONTIGUOUS_ALLOC is enabled and num > 1.
        """
        free = self._free_mask
        n = free.shape[0]
        if num > n:
            return None
        if num == 1:
            nz = np.flatnonzero(free)
            return int(nz[0]) if nz.size else None
        # window sum of `num` consecutive entries == num  =>  all free
        ci = np.concatenate(([0], np.cumsum(free, dtype=np.int64)))
        window = ci[num:] - ci[:-num]  # length n - num + 1
        idx = np.flatnonzero(window == num)
        return int(idx[0]) if idx.size else None

    def allocate_blocks(self, num: int) -> np.ndarray:
        if num < 0:
            raise ValueError(f"num must be greater than 0, but got {num}")
        if num > self._num_free:
            raise ValueError(f"Not enough free blocks, required: {num}, available: {self._num_free}")

        # [P2] Try a contiguous run first.
        if self._prefer_contiguous and num > 1:
            run_start = self._find_contiguous_run(num)
            if run_start is not None:
                free_ids = np.arange(run_start, run_start + num, dtype=np.int64)
                self._free_mask[free_ids] = False
                self._num_free -= num
                # Invalidate the front cursor so the next non-contiguous
                # allocation rebuilds _free_ids from the (now-updated) mask.
                self._free_ids_offset = len(self._free_ids)
                return free_ids

        if num > len(self._free_ids) - self._free_ids_offset:
            self._update_free_ids()

        free_ids = self._free_ids[self._free_ids_offset:self._free_ids_offset+num]
        self._free_ids_offset += num

        self._free_mask[free_ids] = False
        self._num_free -= num
        return free_ids

    def recycle_blocks(self, block_ids: np.ndarray) -> None:
        if block_ids.ndim != 1 or block_ids.dtype != np.int64:
            raise ValueError("block_ids must be a 1D tensor of int64")
        
        block_ids = np.unique(block_ids)

        already_free = self._free_mask[block_ids]
        if already_free.any():
            raise ValueError(f"block_ids {block_ids[already_free]} are already free")
        self._free_mask[block_ids] = True
        self._num_free += len(block_ids)

    def _update_free_ids(self) -> None:
        self._free_ids = self._free_mask.nonzero()[0]
        self._free_ids_offset = 0

    @property
    def num_free_blocks(self) -> int:
        return self._num_free

    @property
    def num_used_blocks(self) -> int:
        return self.num_total_blocks - self._num_free
