"""Unit tests for ``MooncakeStoreTransferWorker._preprocess`` / ``_transfer_impl``.

Importing ``flexkv.transfer.worker`` normally pulls in the compiled
``flexkv.c_ext`` extension, the ``flexkv.mooncakeEngineWrapper`` module and a
few transport libs.  To keep the test offline-collectable we install fakes for
the heavy native dependencies *before* importing the worker module (the
``_mooncake_store_testkit`` import already provides a fake ``flexkv.c_ext``;
here we top it up with the extra symbols ``worker.py`` imports at module load).

The worker instance is built with ``object.__new__`` so the real ``__init__``
(which calls ``cudaHostRegister`` + creates a live ``MooncakeStoreClient`` +
``register_buffer``) never runs.  We set only the attributes the two methods
under test read, and inject a fake client.

Covered:
* ``_preprocess`` pointer math: ``base_ptr + blk_id * block_size_bytes``
* keys built via ``build_key`` (pool kind + PP/layer fields)
* H2REMOTE reads src_block_ids, REMOTE2H reads dst_block_ids
* ``_transfer_impl`` dispatch: H2REMOTE -> batch_put, REMOTE2H -> batch_get
* invalid transfer type raises ValueError
"""
import _mooncake_store_testkit as kit  # installs fake flexkv.c_ext

import sys
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Top up the fake native deps that flexkv.transfer.worker imports at load time.
# ---------------------------------------------------------------------------
_c_ext = sys.modules.get("flexkv.c_ext")
if _c_ext is not None and not hasattr(_c_ext, "transfer_kv_blocks"):
    _c_ext.transfer_kv_blocks = lambda *a, **k: None
    _c_ext.transfer_kv_blocks_ssd = lambda *a, **k: None

    class _TPTransferThreadGroup:  # pragma: no cover - placeholder
        def __init__(self, *a, **k):
            pass

    _c_ext.TPTransferThreadGroup = _TPTransferThreadGroup

if "flexkv.mooncakeEngineWrapper" not in sys.modules:
    _mc_wrap = types.ModuleType("flexkv.mooncakeEngineWrapper")

    class _MoonCakeTransferEngineWrapper:  # pragma: no cover - placeholder
        def __init__(self, *a, **k):
            pass

    _mc_wrap.MoonCakeTransferEngineWrapper = _MoonCakeTransferEngineWrapper
    sys.modules["flexkv.mooncakeEngineWrapper"] = _mc_wrap

# Import the worker module.  The fake ``flexkv.c_ext`` (installed by the
# testkit, with a module-level ``__getattr__`` returning dummies for any native
# symbol) plus the fakes above let this import succeed offline.  On the real
# test environment the compiled extension is used instead.  We deliberately use
# a hard import (not importorskip) so a regression that breaks worker import is
# reported as a FAILURE rather than being silently skipped.
import flexkv.transfer.worker as worker_mod  # noqa: E402

from flexkv.common.transfer import TransferType  # noqa: E402
from flexkv.external.mooncake_store_keys import PoolKind, build_key  # noqa: E402

MooncakeStoreTransferWorker = worker_mod.MooncakeStoreTransferWorker


class _FakeLayout:
    """Minimal stand-in for KVCacheLayout exposing get_elements_per_block."""

    def __init__(self, elements_per_block):
        self._epb = elements_per_block

    def get_elements_per_block(self):
        return self._epb


class _FakeBuffer:
    """Stand-in for the registered hot-CPU tensor; only data_ptr() is used."""

    def __init__(self, base_ptr):
        self._ptr = base_ptr

    def data_ptr(self):
        return self._ptr


class _FakeOp:
    def __init__(self, transfer_type, src, dst, hashes):
        self.transfer_type = transfer_type
        self.src_block_ids = np.array(src, dtype=np.int64)
        self.dst_block_ids = np.array(dst, dtype=np.int64)
        self.mooncake_store_block_hashes = np.array(hashes, dtype=np.int64)


def _make_worker(*, pool_kind=PoolKind.KV, base_ptr=0x1000,
                 elements_per_block=32, dtype=None,
                 pp_rank=0, pp_size=1, node_layer_start=0,
                 node_layer_end=0, total_layers=0):
    import torch

    worker = object.__new__(MooncakeStoreTransferWorker)
    worker.pp_rank = pp_rank
    worker.pp_size = pp_size
    worker.node_layer_start = node_layer_start
    worker.node_layer_end = node_layer_end
    worker.total_layers = total_layers
    worker.pool_kind = pool_kind
    worker.suffix_str = pool_kind.value
    worker.dtype = dtype if dtype is not None else torch.bfloat16
    worker.cpu_kv_layout = _FakeLayout(elements_per_block)
    worker._cpu_buffer = _FakeBuffer(base_ptr)
    worker.mooncake_client = kit.FakeMooncakeStoreClient()
    return worker


def _block_size_bytes(worker):
    return worker.cpu_kv_layout.get_elements_per_block() * worker.dtype.itemsize


# ---------------------------------------------------------------------------
# _preprocess
# ---------------------------------------------------------------------------
def test_preprocess_h2remote_uses_src_block_ids_and_pointer_math():
    worker = _make_worker(base_ptr=0x2000, elements_per_block=16)
    bsb = _block_size_bytes(worker)
    op = _FakeOp(TransferType.H2REMOTE, src=[0, 2, 5], dst=[9, 9, 9],
                 hashes=[100, 101, 102])

    cpu_ptrs, block_sizes, keys = worker._preprocess(op)

    # H2REMOTE -> uses src_block_ids
    assert cpu_ptrs == [0x2000 + 0 * bsb, 0x2000 + 2 * bsb, 0x2000 + 5 * bsb]
    assert block_sizes == [bsb, bsb, bsb]
    assert keys == [build_key(h, PoolKind.KV) for h in (100, 101, 102)]


def test_preprocess_remote2h_uses_dst_block_ids():
    worker = _make_worker(base_ptr=0x4000, elements_per_block=8)
    bsb = _block_size_bytes(worker)
    op = _FakeOp(TransferType.REMOTE2H, src=[9, 9], dst=[1, 3],
                 hashes=[200, 201])

    cpu_ptrs, block_sizes, keys = worker._preprocess(op)

    # REMOTE2H -> uses dst_block_ids
    assert cpu_ptrs == [0x4000 + 1 * bsb, 0x4000 + 3 * bsb]
    assert keys == [build_key(h, PoolKind.KV) for h in (200, 201)]


def test_preprocess_uses_pool_kind_and_pp_suffix():
    worker = _make_worker(
        pool_kind=PoolKind.INDEXER,
        pp_rank=1, pp_size=2, node_layer_start=0, node_layer_end=16,
        total_layers=32,
    )
    op = _FakeOp(TransferType.H2REMOTE, src=[0], dst=[0], hashes=[7])
    _, _, keys = worker._preprocess(op)
    # cross-node PP (16 != 32) -> indexer suffix + pp_rank suffix
    assert keys == ["7_FlexKV_indexer_pp_rank_1_of_2"]


def test_preprocess_full_model_node_no_suffix():
    worker = _make_worker(
        pool_kind=PoolKind.KV,
        pp_rank=1, pp_size=2, node_layer_start=0, node_layer_end=32,
        total_layers=32,
    )
    op = _FakeOp(TransferType.H2REMOTE, src=[0], dst=[0], hashes=[7])
    _, _, keys = worker._preprocess(op)
    assert keys == ["7_FlexKV"]


def test_preprocess_requires_hashes():
    worker = _make_worker()
    op = _FakeOp(TransferType.H2REMOTE, src=[0], dst=[0], hashes=[1])
    op.mooncake_store_block_hashes = None
    with pytest.raises(AssertionError):
        worker._preprocess(op)


# ---------------------------------------------------------------------------
# _transfer_impl dispatch
# ---------------------------------------------------------------------------
def test_transfer_impl_h2remote_calls_batch_put():
    worker = _make_worker()
    keys = ["k0", "k1"]
    ptrs = [10, 20]
    sizes = [4, 4]
    worker._transfer_impl(ptrs, sizes, keys, TransferType.H2REMOTE)
    assert worker.mooncake_client.put_calls == [(keys, ptrs, sizes)]
    assert worker.mooncake_client.get_calls == []


def test_transfer_impl_remote2h_calls_batch_get():
    worker = _make_worker()
    keys = ["k0"]
    ptrs = [10]
    sizes = [4]
    worker._transfer_impl(ptrs, sizes, keys, TransferType.REMOTE2H)
    assert worker.mooncake_client.get_calls == [(keys, ptrs, sizes)]
    assert worker.mooncake_client.put_calls == []


@pytest.mark.parametrize(
    "bad_type",
    [TransferType.H2D, TransferType.D2H, TransferType.DISK2H, TransferType.H2DISK],
)
def test_transfer_impl_invalid_type_raises_value_error(bad_type):
    worker = _make_worker()
    with pytest.raises(ValueError):
        worker._transfer_impl([1], [4], ["k"], bad_type)
