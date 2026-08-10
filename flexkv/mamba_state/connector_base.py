"""Framework-agnostic mamba state L2/L3 management for FlexKV connectors.

This module provides the bridge between a framework's mamba state pool
(sglang MambaPool / vLLM mamba pool) and FlexKV's multi-tier storage
(CPU int8 / SSD / GDS / Remote).

Design:
  - L1 (GPU active state) = framework native (sglang MambaPool / vLLM pool)
  - L2 (CPU int8 checkpoint) = FlexKV MambaCheckpointPool
  - L3 (SSD / GDS / Remote) = FlexKV transfer engine

The framework connector reads/writes mamba state tensors from/to the
framework's pool. This module handles int8 quantization, prefix-keyed
storage, LRU eviction, and multi-tier transfer.

Framework-agnostic: works with any inference framework that exposes
mamba state tensors (temporal_state + conv_state) indexed by a
per-request slot id.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from flexkv.mamba_state.checkpoint_pool import MambaCheckpointPool

logger = logging.getLogger(__name__)


@dataclass
class MambaStateSnapshot:
    """A snapshot of mamba state read from a framework's GPU pool."""

    temporal: torch.Tensor  # [L, H, d_v, d_k] bf16
    conv: List[torch.Tensor]  # list of [L, ...conv_shape...] bf16
    token_ids: List[int]  # prefix token ids this state corresponds to


@dataclass
class MambaStateConfig:
    """Configuration for mamba state L2/L3 management.

    Auto-constructed from the framework's mamba pool shape — users do
    not configure this directly.
    """

    num_linear_layers: int
    num_heads: int
    head_v_dim: int
    head_k_dim: int
    conv_shapes: List[Tuple[int, ...]]
    conv_dtype: torch.dtype
    temporal_dtype: torch.dtype = torch.bfloat16

    # L2 sizing (CPU int8 checkpoint pool)
    num_cpu_slots: int = 2048

    # L3 (SSD / Remote) — 0 means disabled
    num_ssd_slots: int = 0
    num_remote_slots: int = 0

    # int8 compression
    enable_int8_compress: bool = True

    # L3 SSD
    ssd_dir: str = ""  # directory for SSD spill (empty = disabled)

    # Write policy: "write_back" (default, store on eviction) or
    # "write_through" (proactively D2H+int8 after N hits).
    write_policy: str = "write_back"
    write_through_threshold: int = 2  # hit count to trigger proactive backup

    # Donate: GPU-side bf16 checkpoint tier (no D2H, no int8).
    # Stores a GPU clone of the active state as a hot cache layer above
    # the CPU int8 pool. Retrieve hits donate first (GPU→GPU copy),
    # falling back to CPU int8 (H2D + dequant) on miss.
    enable_donate: bool = False
    num_donate_slots: int = 64


class MambaStateConnectorBase:
    """Framework-agnostic mamba state L2/L3 manager.

    Owns a :class:`MambaCheckpointPool` (CPU int8) and coordinates
    multi-tier transfer (SSD / Remote) through FlexKV's transfer engine.

    The framework connector calls:
      - :meth:`store`   — snapshot from GPU → int8 → CPU/SSD/Remote
      - :meth:`lookup`  — check if mamba state exists for a prefix
      - :meth:`retrieve`— load from CPU/SSD/Remote → dequantize → GPU

    Thread-safety: all public methods are guarded by an internal lock.
    The framework connector is responsible for CUDA stream ordering.
    """

    def __init__(self, config: MambaStateConfig, transfer_engine: Any = None):
        self._config = config
        self._transfer_engine = transfer_engine  # FlexKV TransferManager (optional, for L3)

        # L2: CPU int8 checkpoint pool
        from flexkv.mamba_state.config import MambaStatePoolConfig
        pool_config = MambaStatePoolConfig(
            num_linear_layers=config.num_linear_layers,
            num_heads=config.num_heads,
            head_v_dim=config.head_v_dim,
            head_k_dim=config.head_k_dim,
            conv_shape=config.conv_shapes[0] if config.conv_shapes else (),
            conv_shapes=config.conv_shapes,
            # P1-4: only single conv type supported (all known models have 1)
            # If future models have multiple conv types, pool needs per-type buffers
            num_cpu_slots=config.num_cpu_slots,
            compress_dtype=torch.int8 if config.enable_int8_compress else torch.bfloat16,
            scale_dtype=config.temporal_dtype,
        )
        self._ckpt_pool = MambaCheckpointPool(pool_config, device="cpu")

        # L1.5: GPU bf16 donate pool (hot cache, zero-D2H path)
        self._enable_donate = config.enable_donate
        self._donated_slots: OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._max_donate_slots = config.num_donate_slots
        self._donate_hit_count: Dict[str, int] = {}

        # L1.8: CPU bf16 host pool (intermediate tier, no quantization)
        # GPU donate → host bf16 → CPU int8 (progressive demotion)
        self._host_cache: OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._max_host_slots = config.num_donate_slots * 4  # host is cheaper than GPU

        # Write policy
        self._write_policy = config.write_policy
        self._write_through_threshold = config.write_through_threshold

        # Prefix hash → checkpoint slot mapping (LRU ordered)
        self._prefix_map: OrderedDict[str, int] = OrderedDict()
        self._slot_to_prefix: Dict[int, str] = {}
        self._prefix_lengths: set = set()  # P1-6: known prefix lengths for fast lookup
        self._branch_points: set = set()  # radix_branch: high-priority prefixes (don't evict)
        self._lock = threading.Lock()

        # Lock reference: prevent evicting checkpoints that active retrieves hold.
        # Key = prefix_hash, value = refcount. _evict_lru skips refcount > 0 entries.
        self._prefix_refcount: Dict[str, int] = {}

        # L3 transfer tracking (SSD / Remote paths)
        self._ssd_dir: str = config.ssd_dir
        self._ssd_paths: Dict[str, str] = {}  # prefix_hash → ssd file path
        self._remote_keys: Dict[str, str] = {}  # prefix_hash → remote key

        import os
        if self._ssd_dir:
            os.makedirs(self._ssd_dir, exist_ok=True)

        # In-flight store tracking (P1-1/P1-3: completion tracking)
        self._inflight_stores: set = set()
        self._inflight_lock = threading.Lock()

        # Stats
        self._store_count = 0
        self._lookup_count = 0
        self._hit_count = 0
        self._retrieve_count = 0

        # Prefetch: early retrieve on background thread
        self._prefetch_results: Dict[str, Any] = {}
        self._prefetch_threads: Dict[str, Any] = {}
        self._prefetch_lock = threading.Lock()

        logger.info(
            "[FlexKV-Mamba] MambaStateConnectorBase initialized: "
            "layers=%d heads=%d d_v=%d d_k=%d cpu_slots=%d int8=%s",
            config.num_linear_layers,
            config.num_heads,
            config.head_v_dim,
            config.head_k_dim,
            config.num_cpu_slots,
            config.enable_int8_compress,
        )

    # ------------------------------------------------------------------
    # Prefix hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_prefix(token_ids: List[int]) -> str:
        """Fast hash of token_ids for prefix matching (P2-1: blake2b)."""
        h = hashlib.blake2b(digest_size=16)
        arr = np.asarray(token_ids, dtype=np.int64)
        h.update(arr.tobytes())
        return h.hexdigest()

    # ------------------------------------------------------------------
    # Store: GPU snapshot → int8 → CPU/SSD/Remote
    # ------------------------------------------------------------------

    def store(
        self,
        token_ids: List[int],
        temporal_state: torch.Tensor,
        conv_state: List[torch.Tensor],
    ) -> str:
        """Store a mamba state snapshot to FlexKV L2/L3.

        Args:
            token_ids: prefix token ids this state corresponds to.
            temporal_state: [L, H, d_v, d_k] bf16 GPU tensor.
            conv_state: list of [L, ...] bf16 GPU tensors.

        Returns:
            prefix_hash key under which the state was stored.
        """
        prefix_hash = self._hash_prefix(token_ids)

        # Track in-flight store (P1-1/P1-3)
        with self._inflight_lock:
            self._inflight_stores.add(prefix_hash)

        try:
            with self._lock:
                # If already stored, skip (dedup)
                if prefix_hash in self._prefix_map:
                    logger.debug("[FlexKV-Mamba] store: prefix already cached, skip")
                    return prefix_hash

            # Allocate a checkpoint slot
            ckpt_slot = self._ckpt_pool.allocate()
            if ckpt_slot is None:
                # LRU evict
                evicted_slot = self._evict_lru()
                if evicted_slot is not None:
                    ckpt_slot = self._ckpt_pool.allocate()
                if ckpt_slot is None:
                    logger.warning("[FlexKV-Mamba] store: checkpoint pool full, cannot store")
                    return prefix_hash

            # Move GPU → CPU (non-blocking), then quantize + store
            temporal_cpu = temporal_state.detach().to("cpu", non_blocking=False)  # sync — avoid reading uninitialized data
            conv_cpu = [c.detach().to("cpu", non_blocking=False) for c in conv_state]  # sync

            # int8 quantize temporal (conv stays bf16)
            # Pool expects single conv tensor [L, *conv_shape]; connector
            # may receive a list (one per conv layer type). Use first
            # element or stack if multiple.
            if isinstance(conv_cpu, (list, tuple)):
                conv_tensor = conv_cpu[0] if len(conv_cpu) == 1 else torch.stack(conv_cpu, dim=0)
            else:
                conv_tensor = conv_cpu
            self._ckpt_pool.store_from_active(
                ckpt_slot, temporal_cpu, conv_tensor
            )

            # Register in prefix map
            self._prefix_map[prefix_hash] = ckpt_slot
            self._slot_to_prefix[ckpt_slot] = prefix_hash
            self._prefix_lengths.add(len(token_ids))  # P1-6: track length
            self._store_count += 1

            # L3 spill to SSD (if enabled)
            if self._ssd_dir:
                self._spill_to_ssd(prefix_hash, ckpt_slot)

            logger.debug(
                "[FlexKV-Mamba] store: prefix=%s... slot=%d (total stored=%d)",
                prefix_hash[:12],
                ckpt_slot,
                self._store_count,
            )

        finally:
            with self._inflight_lock:
                self._inflight_stores.discard(prefix_hash)

        return prefix_hash

    # ------------------------------------------------------------------
    # Donate: GPU bf16 snapshot (no D2H, no int8)
    # ------------------------------------------------------------------

    def store_donate(
        self,
        token_ids: List[int],
        temporal_state: torch.Tensor,
        conv_state: List[torch.Tensor],
    ) -> str:
        """Donate a GPU snapshot as a hot bf16 checkpoint (zero-D2H path).

        Clones the GPU tensors (same-device copy) and holds them as a
        cache tier above the CPU int8 pool. Retrieve checks donate first
        (GPU→GPU CoW), falling back to CPU int8 on miss.

        Use when the source slot will be freed or reused after this call
        — the clone decouples the checkpoint from the active slot.
        """
        if not self._enable_donate:
            return self.store(token_ids, temporal_state, conv_state)

        prefix_hash = self._hash_prefix(token_ids)

        with self._inflight_lock:
            self._inflight_stores.add(prefix_hash)

        try:
            with self._lock:
                if prefix_hash in self._donated_slots or prefix_hash in self._prefix_map:
                    logger.debug("[FlexKV-Mamba] store_donate: prefix already cached, skip")
                    return prefix_hash

                if len(self._donated_slots) >= self._max_donate_slots:
                    self._evict_donated()

            # GPU clone (same-device copy, no D2H sync)
            temporal_clone = temporal_state.clone()
            conv_tensor = conv_state[0] if isinstance(conv_state, (list, tuple)) else conv_state
            conv_clone = conv_tensor.clone()

            with self._lock:
                self._donated_slots[prefix_hash] = (temporal_clone, conv_clone)
                self._prefix_lengths.add(len(token_ids))
                self._store_count += 1

            logger.debug(
                "[FlexKV-Mamba] store_donate: prefix=%s... (donated=%d/%d)",
                prefix_hash[:12], len(self._donated_slots), self._max_donate_slots,
            )

        finally:
            with self._inflight_lock:
                self._inflight_stores.discard(prefix_hash)

        return prefix_hash

    def _evict_donated(self) -> None:
        """Demote the LRU donated slot to host bf16 cache (intermediate tier)."""
        if not self._donated_slots:
            return

        prefix_hash, (temporal_gpu, conv_gpu) = next(iter(self._donated_slots))
        del self._donated_slots[prefix_hash]
        self._donate_hit_count.pop(prefix_hash, None)

        # Demote to host bf16 (D2H, no quantization)
        temporal_cpu = temporal_gpu.detach().to("cpu", non_blocking=False)
        conv_cpu = conv_gpu.detach().to("cpu", non_blocking=False)

        if len(self._host_cache) >= self._max_host_slots:
            self._evict_host()

        self._host_cache[prefix_hash] = (temporal_cpu, conv_cpu)
        logger.debug("[FlexKV-Mamba] demoted donate→host: prefix=%s...", prefix_hash[:12])

    def _evict_host(self) -> None:
        """Demote the LRU host bf16 slot to CPU int8 (quantize + compress)."""
        if not self._host_cache:
            return

        prefix_hash, (temporal_cpu, conv_cpu) = next(iter(self._host_cache))
        del self._host_cache[prefix_hash]

        ckpt_slot = self._ckpt_pool.allocate()
        if ckpt_slot is None:
            evicted = self._evict_lru()
            if evicted is not None:
                ckpt_slot = self._ckpt_pool.allocate()

        if ckpt_slot is not None:
            self._ckpt_pool.store_from_active(ckpt_slot, temporal_cpu, conv_cpu)
            self._prefix_map[prefix_hash] = ckpt_slot
            self._slot_to_prefix[ckpt_slot] = prefix_hash
            logger.debug("[FlexKV-Mamba] demoted host→int8: prefix=%s...", prefix_hash[:12])
        else:
            logger.debug("[FlexKV-Mamba] dropped host (pool full): prefix=%s...", prefix_hash[:12])

    def _write_through_demote(self, prefix_hash: str) -> None:
        """Write-through: proactively create host bf16 copy while keeping GPU donate slot."""
        if prefix_hash not in self._donated_slots or prefix_hash in self._host_cache:
            return
        temporal_gpu, conv_gpu = self._donated_slots[prefix_hash]
        temporal_cpu = temporal_gpu.detach().to("cpu", non_blocking=False)
        conv_cpu = conv_gpu.detach().to("cpu", non_blocking=False)
        if len(self._host_cache) >= self._max_host_slots:
            self._evict_host()
        self._host_cache[prefix_hash] = (temporal_cpu, conv_cpu)
        logger.debug("[FlexKV-Mamba] write-through demote→host: prefix=%s...", prefix_hash[:12])

    # ------------------------------------------------------------------
    # Lookup: check if mamba state exists for a prefix
    # ------------------------------------------------------------------

    def lookup(self, token_ids: List[int]) -> Tuple[bool, int]:
        """Check if mamba state exists for the given prefix.

        Returns:
            (hit, matched_token_count). If hit, matched_token_count is
            the number of tokens in the longest cached prefix (may be
            <= len(token_ids) if only a shorter prefix is cached).
        """
        # P1-7: stats increment inside lock for thread safety
        with self._lock:
            self._lookup_count += 1

        # Mamba state is stored at checkpoint boundaries (e.g. prompt end),
        # not every token. Check exact token_ids first (O(n) for one SHA256).
        prefix_hash = self._hash_prefix(token_ids)
        with self._lock:
            if prefix_hash in self._donated_slots:
                self._hit_count += 1
                self._donated_slots.move_to_end(prefix_hash)
                return True, len(token_ids)
            if prefix_hash in self._host_cache:
                self._hit_count += 1
                self._host_cache.move_to_end(prefix_hash)
                return True, len(token_ids)
            if prefix_hash in self._prefix_map:
                self._hit_count += 1
                self._prefix_map.move_to_end(prefix_hash)
                return True, len(token_ids)

        # P1-6: Only check known prefix lengths (stored at checkpoint boundaries)
        # instead of scanning every possible length. This is O(k) where k is
        # the number of distinct checkpoint lengths, typically 1-5.
        candidates = sorted(
            [l for l in self._prefix_lengths if l < len(token_ids)],
            reverse=True
        )

        for end in candidates:
            prefix_hash = self._hash_prefix(token_ids[:end])
            with self._lock:
                if prefix_hash in self._donated_slots:
                    self._hit_count += 1
                    self._donated_slots.move_to_end(prefix_hash)
                    return True, end
                if prefix_hash in self._prefix_map:
                    self._hit_count += 1
                    self._prefix_map.move_to_end(prefix_hash)
                    return True, end

        return False, 0

    # ------------------------------------------------------------------
    # Retrieve: CPU/SSD/Remote → dequantize → GPU
    # ------------------------------------------------------------------

    def retrieve(
        self,
        token_ids: List[int],
    ) -> Optional[Tuple[str, torch.Tensor, List[torch.Tensor]]]:
        """Retrieve mamba state from FlexKV L1.5/L2/L3.

        Returns:
            (prefix_hash, temporal_state, conv_state) if found, None if miss.
            Tensors may be on GPU (donate hit, bf16) or CPU (int8 pool,
            dequantized to bf16). The connector copies to the active slot.

        The caller MUST call release_retrieve(prefix_hash) after the GPU
        copy completes, to allow eviction of this checkpoint.
        """
        with self._lock:
            self._retrieve_count += 1

        # Check prefetch cache first (early retrieve may have completed)
        prefix_hash = self._hash_prefix(token_ids)
        with self._prefetch_lock:
            if prefix_hash in self._prefetch_results:
                result = self._prefetch_results.pop(prefix_hash)
                self._prefetch_threads.pop(prefix_hash, None)
                if result is not None:
                    self._acquire_ref(result[0])
                    return result
                return None

        # Check donate tier first (GPU bf16, zero-D2H), then CPU int8
        prefix_hash = self._hash_prefix(token_ids)
        with self._lock:
            if prefix_hash in self._donated_slots:
                temporal, conv = self._donated_slots[prefix_hash]
                self._donated_slots.move_to_end(prefix_hash)
                self._hit_count += 1
                self._acquire_ref(prefix_hash)
                # write_through: proactively demote to CPU int8 after N hits
                if (self._write_policy == "write_through"
                        and prefix_hash not in self._prefix_map):
                    cnt = self._donate_hit_count.get(prefix_hash, 0) + 1
                    self._donate_hit_count[prefix_hash] = cnt
                    if cnt >= self._write_through_threshold:
                        self._write_through_demote(prefix_hash)
                logger.debug(
                    "[FlexKV-Mamba] retrieve(donate): prefix=%s... matched=%d",
                    prefix_hash[:12], len(token_ids),
                )
                return prefix_hash, temporal, [conv]

            if prefix_hash in self._host_cache:
                temporal, conv = self._host_cache[prefix_hash]
                self._host_cache.move_to_end(prefix_hash)
                self._hit_count += 1
                self._acquire_ref(prefix_hash)
                return prefix_hash, temporal, [conv]

            if prefix_hash in self._host_cache:
                temporal, conv = self._host_cache[prefix_hash]
                self._host_cache.move_to_end(prefix_hash)
                self._hit_count += 1
                return prefix_hash, temporal, [conv]

            if prefix_hash in self._prefix_map:
                ckpt_slot = self._prefix_map[prefix_hash]
                self._prefix_map.move_to_end(prefix_hash)
                self._acquire_ref(prefix_hash)
                temporal, conv = self._ckpt_pool.load_to_active(ckpt_slot)
                logger.debug(
                    "[FlexKV-Mamba] retrieve: prefix=%s... slot=%d matched_tokens=%d",
                    prefix_hash[:12], ckpt_slot, len(token_ids),
                )
                return prefix_hash, temporal, [conv] if not isinstance(conv, (list, tuple)) else conv

        candidates = sorted(
            [l for l in self._prefix_lengths if l < len(token_ids)],
            reverse=True
        )

        for end in candidates:
            prefix_hash = self._hash_prefix(token_ids[:end])
            with self._lock:
                if prefix_hash in self._donated_slots:
                    temporal, conv = self._donated_slots[prefix_hash]
                    self._donated_slots.move_to_end(prefix_hash)
                    self._hit_count += 1
                    self._acquire_ref(prefix_hash)
                    logger.debug(
                        "[FlexKV-Mamba] retrieve(donate): prefix=%s... matched=%d",
                        prefix_hash[:12], end,
                    )
                    return prefix_hash, temporal, [conv]

                if prefix_hash in self._host_cache:
                    temporal, conv = self._host_cache[prefix_hash]
                    self._host_cache.move_to_end(prefix_hash)
                    self._hit_count += 1
                    self._acquire_ref(prefix_hash)
                    return prefix_hash, temporal, [conv]

                if prefix_hash in self._host_cache:
                    temporal, conv = self._host_cache[prefix_hash]
                    self._host_cache.move_to_end(prefix_hash)
                    self._hit_count += 1
                    return prefix_hash, temporal, [conv]

                if prefix_hash in self._prefix_map:
                    ckpt_slot = self._prefix_map[prefix_hash]
                    self._prefix_map.move_to_end(prefix_hash)
                    self._acquire_ref(prefix_hash)
                    temporal, conv = self._ckpt_pool.load_to_active(ckpt_slot)
                    logger.debug(
                        "[FlexKV-Mamba] retrieve: prefix=%s... slot=%d matched_tokens=%d",
                        prefix_hash[:12], ckpt_slot, end,
                    )
                    return prefix_hash, temporal, [conv] if not isinstance(conv, (list, tuple)) else conv

        # L3 SSD fallback: check all prefix lengths in SSD
        if self._ssd_dir:
            ssd_prefix = self._hash_prefix(token_ids)
            if ssd_prefix in self._ssd_paths:
                return self._load_from_ssd(ssd_prefix)
            for end in candidates:
                ssd_prefix = self._hash_prefix(token_ids[:end])
                if ssd_prefix in self._ssd_paths:
                    return self._load_from_ssd(ssd_prefix)

        logger.debug("[FlexKV-Mamba] retrieve: miss (no matching prefix)")
        return None

    # ------------------------------------------------------------------
    # L3 SSD spill / load
    # ------------------------------------------------------------------

    def _spill_to_ssd(self, prefix_hash: str, ckpt_slot: int) -> None:
        """Serialize int8 checkpoint slot to SSD file."""
        import os
        file_path = os.path.join(self._ssd_dir, f"{prefix_hash}.pt")
        try:
            torch.save({
                "qdata": self._ckpt_pool.qdata[:, ckpt_slot].clone(),
                "scale": self._ckpt_pool.scale[:, ckpt_slot].clone(),
                "conv": [c[:, ckpt_slot].clone() for c in self._ckpt_pool.conv],
            }, file_path)
            self._ssd_paths[prefix_hash] = file_path
            self._prefix_lengths.add(0)  # ensure miss path checks SSD
        except Exception as exc:
            logger.debug("[FlexKV-Mamba] SSD spill failed: %s", exc)

    def _load_from_ssd(self, prefix_hash: str) -> Optional[Tuple[str, torch.Tensor, List[torch.Tensor]]]:
        """Load from SSD, store to int8 pool, return (prefix_hash, temporal, conv)."""
        import os
        file_path = self._ssd_paths.get(prefix_hash)
        if file_path is None or not os.path.exists(file_path):
            return None
        try:
            data = torch.load(file_path, map_location="cpu")
            ckpt_slot = self._ckpt_pool.allocate()
            if ckpt_slot is None:
                evicted = self._evict_lru()
                if evicted is not None:
                    ckpt_slot = self._ckpt_pool.allocate()
            if ckpt_slot is None:
                return None
            self._ckpt_pool.qdata[:, ckpt_slot].copy_(data["qdata"])
            self._ckpt_pool.scale[:, ckpt_slot].copy_(data["scale"])
            for i, c in enumerate(self._ckpt_pool.conv):
                c[:, ckpt_slot].copy_(data["conv"][i])
            self._prefix_map[prefix_hash] = ckpt_slot
            self._slot_to_prefix[ckpt_slot] = prefix_hash
            temporal, conv = self._ckpt_pool.load_to_active(ckpt_slot)
            logger.debug("[FlexKV-Mamba] SSD load: prefix=%s...", prefix_hash[:12])
            return prefix_hash, temporal, [conv] if not isinstance(conv, (list, tuple)) else conv
        except Exception as exc:
            logger.debug("[FlexKV-Mamba] SSD load failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Lock reference: prevent evicting checkpoints in active use
    # ------------------------------------------------------------------

    def _acquire_ref(self, prefix_hash: str) -> None:
        """Increment refcount for a prefix (called before retrieve reads slot)."""
        self._prefix_refcount[prefix_hash] = self._prefix_refcount.get(prefix_hash, 0) + 1

    def _release_ref(self, prefix_hash: str) -> None:
        """Decrement refcount for a prefix (called after copy to GPU completes)."""
        cnt = self._prefix_refcount.get(prefix_hash, 0)
        if cnt <= 1:
            self._prefix_refcount.pop(prefix_hash, None)
        else:
            self._prefix_refcount[prefix_hash] = cnt - 1

    def release_retrieve(self, prefix_hash: str) -> None:
        """Release a retrieve reference. Call after GPU copy completes."""
        with self._lock:
            self._release_ref(prefix_hash)

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _evict_lru(self) -> Optional[int]:
        """Evict the least recently used checkpoint slot.

        Skips entries with refcount > 0 (active retrieve in flight).
        Skips branch points (high priority) unless all are locked.
        """
        if not self._prefix_map:
            return None

        # Find first evictable entry: non-branch-point AND refcount == 0
        prefix_hash = None
        ckpt_slot = None
        fallback = None  # branch-point but unlocked — last resort
        for ph, slot in self._prefix_map.items():
            if self._prefix_refcount.get(ph, 0) > 0:
                continue  # locked by active retrieve
            if ph in self._branch_points:
                if fallback is None:
                    fallback = (ph, slot)
                continue
            prefix_hash = ph
            ckpt_slot = slot
            break

        if prefix_hash is None:
            if fallback is not None:
                prefix_hash, ckpt_slot = fallback
            else:
                logger.debug("[FlexKV-Mamba] evict LRU: no evictable slots (all locked or branch points)")
                return None

        del self._prefix_map[prefix_hash]
        self._slot_to_prefix.pop(ckpt_slot, None)
        self._ckpt_pool.free(ckpt_slot)

        logger.debug(
            "[FlexKV-Mamba] evict LRU: prefix=%s... slot=%d",
            prefix_hash[:12],
            ckpt_slot,
        )
        return ckpt_slot

    def mark_branch_point(self, token_ids: List[int]) -> None:
        """Mark a prefix as a radix branch point (high priority, don't evict)."""
        prefix_hash = self._hash_prefix(token_ids)
        with self._lock:
            if prefix_hash in self._prefix_map:
                self._branch_points.add(prefix_hash)
                self._prefix_map.move_to_end(prefix_hash)  # move to MRU
                logger.debug("[FlexKV-Mamba] marked branch point: %s...", prefix_hash[:12])

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        """Return mamba state cache statistics."""
        with self._lock:
            return {
                "mamba_store_count": self._store_count,
                "mamba_lookup_count": self._lookup_count,
                "mamba_hit_count": self._hit_count,
                "mamba_retrieve_count": self._retrieve_count,
                "mamba_cached_count": len(self._prefix_map),
                "mamba_free_slots": self._ckpt_pool.num_free,
            }

    # ------------------------------------------------------------------
    # Reset / shutdown
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all cached mamba state. Waits for in-flight stores."""
        # P1-3: Wait for in-flight stores to complete before clearing
        import time
        for _ in range(100):  # max 10s
            with self._inflight_lock:
                if not self._inflight_stores:
                    break
            time.sleep(0.1)
        with self._lock:
            self._donated_slots.clear()
            self._donate_hit_count.clear()
            self._host_cache.clear()
            self._prefix_map.clear()
            self._slot_to_prefix.clear()
            self._prefix_lengths.clear()
            self._branch_points.clear()
            self._prefix_refcount.clear()
            self._ckpt_pool.reset()
            self._ssd_paths.clear()
            self._remote_keys.clear()
            with self._prefetch_lock:
                self._prefetch_results.clear()
                self._prefetch_threads.clear()

    # --- P2-1: Prefetch support (early retrieve for overlap) ---

    def prefetch_mamba_state(self, token_ids: List[int]) -> bool:
        """Prefetch mamba state: start retrieve on background thread.

        Kicks off the hash + dict lookup + dequant early (during match_prefix),
        so init_load_back's retrieve returns the cached result immediately.
        Returns True if prefetch was initiated, False if miss.
        """
        import threading
        prefix_hash = self._hash_prefix(token_ids)
        with self._lock:
            if prefix_hash not in self._prefix_map and prefix_hash not in self._donated_slots:
                return False
            if prefix_hash in self._prefetch_results:
                return True  # already prefetched

        def _do_prefetch():
            result = self._retrieve_internal(token_ids)
            with self._prefetch_lock:
                self._prefetch_results[prefix_hash] = result

        t = threading.Thread(target=_do_prefetch, daemon=True)
        with self._prefetch_lock:
            self._prefetch_threads[prefix_hash] = t
        t.start()
        return True

    def _retrieve_internal(self, token_ids: List[int]):
        """Internal retrieve without acquiring ref (prefetch uses this)."""
        with self._lock:
            self._retrieve_count += 1
        prefix_hash = self._hash_prefix(token_ids)
        with self._lock:
            if prefix_hash in self._donated_slots:
                temporal, conv = self._donated_slots[prefix_hash]
                self._donated_slots.move_to_end(prefix_hash)
                self._hit_count += 1
                return prefix_hash, temporal, [conv]
            if prefix_hash in self._prefix_map:
                ckpt_slot = self._prefix_map[prefix_hash]
                self._prefix_map.move_to_end(prefix_hash)
                temporal, conv = self._ckpt_pool.load_to_active(ckpt_slot)
                return prefix_hash, temporal, [conv] if not isinstance(conv, (list, tuple)) else conv
        candidates = sorted([l for l in self._prefix_lengths if l < len(token_ids)], reverse=True)
        for end in candidates:
            prefix_hash = self._hash_prefix(token_ids[:end])
            with self._lock:
                if prefix_hash in self._donated_slots:
                    temporal, conv = self._donated_slots[prefix_hash]
                    self._donated_slots.move_to_end(prefix_hash)
                    self._hit_count += 1
                    return prefix_hash, temporal, [conv]
                if prefix_hash in self._prefix_map:
                    ckpt_slot = self._prefix_map[prefix_hash]
                    self._prefix_map.move_to_end(prefix_hash)
                    temporal, conv = self._ckpt_pool.load_to_active(ckpt_slot)
                    return prefix_hash, temporal, [conv] if not isinstance(conv, (list, tuple)) else conv
        return None

    def check_prefetch_progress(self, token_ids: List[int]) -> bool:
        """Check if prefetch has completed. Returns True if ready."""
        prefix_hash = self._hash_prefix(token_ids)
        with self._prefetch_lock:
            return prefix_hash in self._prefetch_results

    def cancel_prefetch(self, token_ids: List[int]) -> None:
        """Cancel an in-flight prefetch."""
        prefix_hash = self._hash_prefix(token_ids)
        with self._prefetch_lock:
            self._prefetch_results.pop(prefix_hash, None)
            self._prefetch_threads.pop(prefix_hash, None)
        pass  # stub

    def evict_by_prefix(self, token_ids: List[int]) -> bool:
        """Evict mamba state for a specific prefix (P1-4: linked eviction).

        Called when the radix cache evicts a prefix node — the corresponding
        mamba state should also be evicted to avoid stale entries.

        Returns True if a state was evicted, False if not found.
        """
        prefix_hash = self._hash_prefix(token_ids)
        with self._lock:
            if prefix_hash not in self._prefix_map:
                return False
            ckpt_slot = self._prefix_map.pop(prefix_hash)
            self._slot_to_prefix.pop(ckpt_slot, None)
            self._ckpt_pool.free(ckpt_slot)
            logger.debug("[FlexKV-Mamba] evict_by_prefix: %s...", prefix_hash[:12])
            return True

    def shutdown(self) -> None:
        """Cleanup resources."""
        self.reset()
