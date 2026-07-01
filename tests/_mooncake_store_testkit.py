"""Shared test helpers for the mooncake-store port unit tests.

These tests must be collectable and runnable on a CI host **without** a real
RDMA fabric, without the ``mooncake.store`` SDK, and (optionally) without the
compiled ``flexkv.c_ext`` PyTorch extension.

To make the pure-python source modules importable in that reduced environment
this module installs a lightweight fake ``flexkv.c_ext`` into ``sys.modules``
*before* any flexkv source module that needs it (``flexkv.common.hash_utils``)
is imported.  The fake provides a deterministic, dependency-free hashing
implementation so that ``SequenceMeta.block_hashes`` is stable across a run.

Importing this module has the side effect of installing the fake; it is
idempotent and never overrides an already-present real ``flexkv.c_ext``.
"""
from __future__ import annotations

import sys
import types
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Fake flexkv.c_ext (only installed when the real extension is unavailable)
# ---------------------------------------------------------------------------
def _install_fake_c_ext() -> None:
    """Install a deterministic, torch-free fake ``flexkv.c_ext``.

    No-op if a (real or fake) ``flexkv.c_ext`` is already importable.
    """
    try:
        import flexkv.c_ext  # noqa: F401  (real extension present -> keep it)
        return
    except Exception:
        pass

    class _FakeHasher:
        """A tiny FNV-1a style 64-bit rolling hasher operating on int64 data."""

        _OFFSET = np.uint64(1469598103934665603)
        _PRIME = np.uint64(1099511628211)

        def __init__(self) -> None:
            self._state = self._OFFSET

        def reset(self) -> None:
            self._state = self._OFFSET

        def update(self, tensor) -> None:
            # Accept either a torch tensor (has .numpy()) or a numpy array.
            arr = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
            flat = np.ascontiguousarray(arr).astype(np.int64).ravel()
            state = self._state
            with np.errstate(over="ignore"):
                for v in flat.view(np.uint64):
                    state = (state ^ np.uint64(v)) * self._PRIME
            self._state = state

        def digest(self) -> int:
            return int(self._state)

    def _gen_hashes(hasher, token_ids_tensor, tokens_per_block, out_tensor) -> None:
        token_ids = (
            token_ids_tensor.numpy()
            if hasattr(token_ids_tensor, "numpy")
            else np.asarray(token_ids_tensor)
        )
        out = out_tensor.numpy() if hasattr(out_tensor, "numpy") else np.asarray(out_tensor)
        num_blocks = token_ids.size // tokens_per_block
        # Reuse the supplied hasher's accumulated prefix state so that the
        # block hash depends on all preceding tokens (prefix property), which
        # the radix/prefix matching logic relies on.
        for b in range(num_blocks):
            blk = token_ids[b * tokens_per_block : (b + 1) * tokens_per_block]
            hasher.update(blk if hasattr(blk, "numpy") else blk)
            out[b] = np.uint64(hasher.digest() & ((1 << 64) - 1))

    fake = types.ModuleType("flexkv.c_ext")
    fake.Hasher = _FakeHasher
    fake.get_hash_size = lambda: 8  # np.uint64 itemsize
    fake.gen_hashes = _gen_hashes

    # ``flexkv.transfer.worker`` (and the modules it transitively imports, e.g.
    # ``flexkv.cache.cache_engine`` / ``radix_remote`` / ``redis_meta``) pull a
    # large, evolving set of symbols out of the native extension via
    # ``from flexkv.c_ext import <Name>`` (CMatchResult, CRadixNode,
    # CRadixTreeIndex, DistributedRadixTree, transfer_kv_blocks, ...).  Rather
    # than enumerate that moving target, expose a module-level ``__getattr__``
    # (PEP 562, py3.7+) that hands back a generic, callable dummy for ANY name
    # not explicitly defined above.  This lets those modules import cleanly
    # under the fake while keeping the real hashing implementation intact.
    class _CDummy:  # generic placeholder for any native class/function
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return None

    def _module_getattr(name):
        # Callable class so both ``X()`` (construct) and ``from ... import X``
        # then ``X(...)`` work; cache on the module so identity is stable.
        dummy = type(name, (_CDummy,), {})
        setattr(fake, name, dummy)
        return dummy

    fake.__getattr__ = _module_getattr

    # Ensure the parent package object exists and exposes the attribute too,
    # because flexkv.common.hash_utils does ``from flexkv import c_ext``.
    flexkv_pkg = sys.modules.get("flexkv")
    if flexkv_pkg is None:
        import flexkv as flexkv_pkg  # noqa: F811  (triggers package __init__)
    sys.modules["flexkv.c_ext"] = fake
    setattr(flexkv_pkg, "c_ext", fake)


_install_fake_c_ext()


# ---------------------------------------------------------------------------
# Generic fakes shared by several test modules
# ---------------------------------------------------------------------------
class FakeMooncakeStoreClient:
    """In-memory stand-in for ``MooncakeStoreClient`` used by cache-engine /
    worker unit tests.  Records every call so assertions can inspect them.

    ``store`` is a ``set`` of keys that are considered "present".
    """

    def __init__(self, present_keys=None) -> None:
        self.store = set(present_keys or [])
        self.put_calls: List[tuple] = []
        self.get_calls: List[tuple] = []
        self.batch_exists_calls: List[List[str]] = []
        self.batch_exists_impl_calls: List[List[str]] = []

    # --- existence helpers (mirror the real client semantics) -------------
    def batch_exists(self, keys: List[str]) -> int:
        self.batch_exists_calls.append(list(keys))
        for i, k in enumerate(keys):
            if k not in self.store:
                return i
        return len(keys)

    def batch_exists_impl(self, keys: List[str]) -> List[int]:
        self.batch_exists_impl_calls.append(list(keys))
        return [1 if k in self.store else 0 for k in keys]

    # --- transfer helpers --------------------------------------------------
    def batch_put(self, key_strs, buffer_ptrs, buffer_sizes) -> List[bool]:
        self.put_calls.append((list(key_strs), list(buffer_ptrs), list(buffer_sizes)))
        for k in key_strs:
            self.store.add(k)
        return [True] * len(key_strs)

    def batch_get(self, key_strs, buffer_ptrs, buffer_sizes) -> List[bool]:
        self.get_calls.append((list(key_strs), list(buffer_ptrs), list(buffer_sizes)))
        return [k in self.store for k in key_strs]

    def register_buffer(self, *a, **k) -> None:
        pass

    def unregister_buffer(self, *a, **k) -> None:
        pass


def make_cache_config(**overrides):
    """Build a CacheConfig that targets the mooncake-store backend without
    requiring a real config file on disk.

    ``use_mooncake_store_backend`` is intentionally left False by default so
    ``CacheConfig.__post_init__`` does not insist on a config-file path; the
    mooncake fields are set directly which is all the unit-under-test reads.
    """
    from flexkv.common.config import CacheConfig

    kwargs = dict(
        tokens_per_block=16,
    )
    kwargs.update(overrides)
    return CacheConfig(**kwargs)
