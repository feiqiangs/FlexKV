"""Unit tests for the mooncake-store transfer merge logic in
``flexkv.common.transfer``:

* ``_merge_remote2h_ops``: REMOTE2H / H2REMOTE merge preserves the extra
  fields a plain ``_merge_ops`` would drop -- ``mooncake_store_block_hashes``
  (concatenated) and ``src_block_node_ids``.
* ``merge_to_batch_graph`` orchestration (non-layerwise):
    - GET path: H2D depends on REMOTE2H (REMOTE2H prefetch gates H2D)
    - PUT path: H2REMOTE depends on D2H
    - ``batch_end_op_id`` priority:
        GET: H2D > REMOTE2H > DISK2H
        PUT: H2REMOTE > H2DISK > D2H
* unsupported transfer types raise NotImplementedError.

These tests use only numpy + the (torch-importing) flexkv package; no RDMA,
GPU, or mooncake SDK is required.
"""
import _mooncake_store_testkit  # noqa: F401  (installs fake flexkv.c_ext)

import numpy as np
import pytest

from flexkv.common.transfer import (
    TransferOp,
    TransferType,
    TransferOpGraph,
    _merge_remote2h_ops,
    merge_to_batch_graph,
)


def _arr(*vals):
    return np.array(vals, dtype=np.int64)


def _make_op(transfer_type, src, dst, hashes=None, node_ids=None):
    return TransferOp(
        graph_id=0,
        transfer_type=transfer_type,
        src_block_ids=_arr(*src),
        dst_block_ids=_arr(*dst),
        mooncake_store_block_hashes=(_arr(*hashes) if hashes is not None else None),
        src_block_node_ids=(_arr(*node_ids) if node_ids is not None else None),
    )


def _graph_with(*ops):
    """Build a TransferOpGraph containing the given ops."""
    graph = TransferOpGraph()
    for op in ops:
        op.graph_id = graph.graph_id
        graph.add_transfer_op(op)
    return graph


# ---------------------------------------------------------------------------
# _merge_remote2h_ops
# ---------------------------------------------------------------------------
def test_merge_remote2h_concats_block_hashes():
    op1 = _make_op(TransferType.REMOTE2H, [0, 1], [10, 11], hashes=[100, 101])
    op2 = _make_op(TransferType.REMOTE2H, [2], [12], hashes=[102])
    graph = TransferOpGraph()
    merged = _merge_remote2h_ops(
        [op1, op2], TransferType.REMOTE2H, graph, [], {}
    )
    assert merged is not None
    assert merged.transfer_type == TransferType.REMOTE2H
    np.testing.assert_array_equal(merged.src_block_ids, _arr(0, 1, 2))
    np.testing.assert_array_equal(merged.dst_block_ids, _arr(10, 11, 12))
    np.testing.assert_array_equal(
        merged.mooncake_store_block_hashes, _arr(100, 101, 102)
    )


def test_merge_remote2h_concats_node_ids():
    op1 = _make_op(TransferType.H2REMOTE, [0], [10], hashes=[1], node_ids=[7])
    op2 = _make_op(TransferType.H2REMOTE, [1], [11], hashes=[2], node_ids=[8])
    merged = _merge_remote2h_ops(
        [op1, op2], TransferType.H2REMOTE, TransferOpGraph(), [], {}
    )
    np.testing.assert_array_equal(merged.src_block_node_ids, _arr(7, 8))
    np.testing.assert_array_equal(merged.mooncake_store_block_hashes, _arr(1, 2))


def test_merge_remote2h_hashes_none_when_any_missing():
    """If any op lacks hashes, the merged op carries None (no partial concat)."""
    op1 = _make_op(TransferType.REMOTE2H, [0], [10], hashes=[1])
    op2 = _make_op(TransferType.REMOTE2H, [1], [11], hashes=None)
    merged = _merge_remote2h_ops(
        [op1, op2], TransferType.REMOTE2H, TransferOpGraph(), [], {}
    )
    assert merged.mooncake_store_block_hashes is None


def test_merge_remote2h_empty_returns_none():
    assert _merge_remote2h_ops([], TransferType.REMOTE2H, TransferOpGraph(), [], {}) is None


def test_merge_remote2h_preserves_callback():
    op1 = _make_op(TransferType.REMOTE2H, [0], [10], hashes=[1])
    called = {}

    def cb(*a, **k):
        called["hit"] = True

    op_cb = {}
    merged = _merge_remote2h_ops(
        [op1], TransferType.REMOTE2H, TransferOpGraph(), [cb], op_cb
    )
    assert merged.op_id in op_cb
    op_cb[merged.op_id]()
    assert called.get("hit") is True


# ---------------------------------------------------------------------------
# merge_to_batch_graph - GET path (REMOTE2H -> H2D)
# ---------------------------------------------------------------------------
def test_batch_merge_get_h2d_depends_on_remote2h():
    remote2h = _make_op(TransferType.REMOTE2H, [0, 1], [5, 6], hashes=[100, 101])
    h2d = _make_op(TransferType.H2D, [5, 6], [20, 21])
    graph = _graph_with(remote2h, h2d)

    merged_graph, batch_end_op_id, _ = merge_to_batch_graph(
        batch_id=999,
        transfer_graphs=[graph],
        task_end_op_ids=[h2d.op_id],
        op_callback_dict={},
    )

    # find the merged H2D and REMOTE2H ops in the new graph
    ops = list(merged_graph._op_map.values())
    h2d_ops = [o for o in ops if o.transfer_type == TransferType.H2D]
    remote_ops = [o for o in ops if o.transfer_type == TransferType.REMOTE2H]
    assert len(h2d_ops) == 1 and len(remote_ops) == 1
    # H2D must depend on REMOTE2H (REMOTE2H prefetch gates H2D)
    assert remote_ops[0].op_id in h2d_ops[0].predecessors
    # hashes preserved on the merged REMOTE2H op
    np.testing.assert_array_equal(
        remote_ops[0].mooncake_store_block_hashes, _arr(100, 101)
    )
    # GET batch_end priority: H2D wins
    assert batch_end_op_id == h2d_ops[0].op_id


def test_batch_merge_get_remote2h_only_end_id():
    """REMOTE2H with no H2D / DISK2H -> batch_end_op_id is the REMOTE2H op."""
    remote2h = _make_op(TransferType.REMOTE2H, [0], [5], hashes=[100])
    graph = _graph_with(remote2h)
    merged_graph, batch_end_op_id, _ = merge_to_batch_graph(
        batch_id=1000,
        transfer_graphs=[graph],
        task_end_op_ids=[remote2h.op_id],
        op_callback_dict={},
    )
    remote_ops = [
        o for o in merged_graph._op_map.values()
        if o.transfer_type == TransferType.REMOTE2H
    ]
    assert len(remote_ops) == 1
    assert batch_end_op_id == remote_ops[0].op_id


# ---------------------------------------------------------------------------
# merge_to_batch_graph - PUT path (D2H -> H2REMOTE)
# ---------------------------------------------------------------------------
def test_batch_merge_put_h2remote_depends_on_d2h():
    d2h = _make_op(TransferType.D2H, [20, 21], [5, 6])
    h2remote = _make_op(TransferType.H2REMOTE, [5, 6], [0, 1], hashes=[100, 101])
    graph = _graph_with(d2h, h2remote)

    merged_graph, batch_end_op_id, _ = merge_to_batch_graph(
        batch_id=1001,
        transfer_graphs=[graph],
        task_end_op_ids=[h2remote.op_id],
        op_callback_dict={},
    )

    ops = list(merged_graph._op_map.values())
    d2h_ops = [o for o in ops if o.transfer_type == TransferType.D2H]
    h2remote_ops = [o for o in ops if o.transfer_type == TransferType.H2REMOTE]
    assert len(d2h_ops) == 1 and len(h2remote_ops) == 1
    # H2REMOTE depends on D2H (data must reach CPU before pushing to store)
    assert d2h_ops[0].op_id in h2remote_ops[0].predecessors
    np.testing.assert_array_equal(
        h2remote_ops[0].mooncake_store_block_hashes, _arr(100, 101)
    )
    # PUT batch_end priority: H2REMOTE wins
    assert batch_end_op_id == h2remote_ops[0].op_id


def test_batch_merge_put_d2h_only_end_id():
    d2h = _make_op(TransferType.D2H, [20], [5])
    graph = _graph_with(d2h)
    _, batch_end_op_id, _ = merge_to_batch_graph(
        batch_id=1002,
        transfer_graphs=[graph],
        task_end_op_ids=[d2h.op_id],
        op_callback_dict={},
    )
    assert batch_end_op_id != -1


# ---------------------------------------------------------------------------
# batch_end_op_id priority ordering
# ---------------------------------------------------------------------------
def test_batch_merge_end_id_h2d_over_remote2h_over_disk2h():
    disk2h = _make_op(TransferType.DISK2H, [0], [5])
    remote2h = _make_op(TransferType.REMOTE2H, [1], [6], hashes=[100])
    h2d = _make_op(TransferType.H2D, [5, 6], [20, 21])
    graph = _graph_with(disk2h, remote2h, h2d)
    merged_graph, batch_end_op_id, _ = merge_to_batch_graph(
        batch_id=1003,
        transfer_graphs=[graph],
        task_end_op_ids=[h2d.op_id],
        op_callback_dict={},
    )
    h2d_ops = [
        o for o in merged_graph._op_map.values()
        if o.transfer_type == TransferType.H2D
    ]
    assert batch_end_op_id == h2d_ops[0].op_id


# ---------------------------------------------------------------------------
# Unsupported transfer types
# ---------------------------------------------------------------------------
def test_batch_merge_rejects_unsupported_type():
    bad = _make_op(TransferType.PEERH2H, [0], [5])
    graph = _graph_with(bad)
    with pytest.raises(NotImplementedError):
        merge_to_batch_graph(
            batch_id=1004,
            transfer_graphs=[graph],
            task_end_op_ids=[bad.op_id],
            op_callback_dict={},
        )


def test_batch_merge_empty_graphs_returns_minus_one():
    merged_graph, batch_end_op_id, cb = merge_to_batch_graph(
        batch_id=1005,
        transfer_graphs=[],
        task_end_op_ids=[],
        op_callback_dict={},
    )
    assert batch_end_op_id == -1
    assert cb == {}
