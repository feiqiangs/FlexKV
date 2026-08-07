"""Unit tests for P2: MambaStateOpConstructor + TransferOp is_mamba_state.

Tests cover:
  - TransferOp with is_mamba_state=True
  - build_get_chain topology (H2D + optional DISK2H/REMOTE2H deps)
  - build_put_chain topology (D2H + optional H2DISK/H2REMOTE deps)
  - disabled state (no-op when enable_linear_state_transfer=False)
  - empty slot handling

Uses MagicMock for GlobalCacheEngine (no real engine needed).
"""
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mock c_ext if not available
if "flexkv.c_ext" not in sys.modules:
    sys.modules["flexkv.c_ext"] = MagicMock()

from flexkv.common.transfer import (
    DeviceType,
    TransferOp,
    TransferOpGraph,
    TransferType,
)
from flexkv.mamba_state.state_op_constructor import (
    MambaStateOpConstructor,
    LinearStateGetChainOpIds,
    LinearStatePutChainOpIds,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_gce(enabled: bool = True, has_cpu_pool: bool = True):
    """Create a mock GlobalCacheEngine."""
    gce = MagicMock()
    gce.cache_config.enable_linear_state_transfer = enabled
    cpu_engine = MagicMock()
    cpu_engine.mamba_state_enabled = has_cpu_pool
    gce.cache_engines = {DeviceType.CPU: cpu_engine}
    return gce


def _slot_ids(*xs):
    return np.array(xs, dtype=np.int64)


@pytest.fixture
def enabled_constructor():
    return MambaStateOpConstructor(_make_gce(enabled=True))


@pytest.fixture
def disabled_constructor():
    return MambaStateOpConstructor(_make_gce(enabled=False))


@pytest.fixture
def graph():
    return TransferOpGraph()


# ---------------------------------------------------------------------------
# TransferOp is_mamba_state field
# ---------------------------------------------------------------------------

class TestTransferOpField:

    def test_default_is_false(self):
        op = TransferOp(
            graph_id=0,
            transfer_type=TransferType.H2D,
            src_block_ids=_slot_ids(1),
            dst_block_ids=_slot_ids(2),
        )
        assert op.is_mamba_state is False

    def test_can_set_true(self):
        op = TransferOp(
            graph_id=0,
            transfer_type=TransferType.H2D,
            src_block_ids=_slot_ids(1),
            dst_block_ids=_slot_ids(2),
            is_mamba_state=True,
        )
        assert op.is_mamba_state is True

    def test_independent_from_is_swa(self):
        op = TransferOp(
            graph_id=0,
            transfer_type=TransferType.H2D,
            src_block_ids=_slot_ids(1),
            dst_block_ids=_slot_ids(2),
            is_swa=True,
            is_mamba_state=True,
        )
        assert op.is_swa is True
        assert op.is_mamba_state is True


# ---------------------------------------------------------------------------
# build_state_op
# ---------------------------------------------------------------------------

class TestBuildStateOp:

    def test_disabled_returns_none(self, disabled_constructor, graph):
        result = disabled_constructor.build_state_op(
            graph, TransferType.H2D, _slot_ids(1), _slot_ids(2)
        )
        assert result is None

    def test_empty_slots_returns_none(self, enabled_constructor, graph):
        result = enabled_constructor.build_state_op(
            graph, TransferType.H2D, _slot_ids(), _slot_ids(2)
        )
        assert result is None

    def test_creates_op_with_is_mamba_state(self, enabled_constructor, graph):
        op_id = enabled_constructor.build_state_op(
            graph, TransferType.H2D, _slot_ids(1), _slot_ids(2)
        )
        assert op_id is not None
        op = graph._op_map[op_id]
        assert op.is_mamba_state is True
        assert op.is_swa is False
        assert op.transfer_type == TransferType.H2D

    def test_op_added_to_graph(self, enabled_constructor, graph):
        assert len(graph._op_map) == 0
        enabled_constructor.build_state_op(
            graph, TransferType.D2H, _slot_ids(1), _slot_ids(2)
        )
        assert len(graph._op_map) == 1


# ---------------------------------------------------------------------------
# build_get_chain
# ---------------------------------------------------------------------------

class TestGetChain:

    def test_get_chain_cpu_only(self, enabled_constructor, graph):
        """GET chain with only CPU source: just H2D."""
        ids = enabled_constructor.build_get_chain(
            graph,
            gpu_slot_ids=_slot_ids(10),
            cpu_slot_ids=_slot_ids(20),
        )
        assert ids is not None
        assert ids.h2d_id is not None
        assert ids.disk2h_id is None
        assert ids.remote2h_id is None
        # Only 1 op in graph
        assert len(graph._op_map) == 1

    def test_get_chain_with_ssd(self, enabled_constructor, graph):
        """GET chain with SSD source: H2D + DISK2H, H2D depends on DISK2H."""
        ids = enabled_constructor.build_get_chain(
            graph,
            gpu_slot_ids=_slot_ids(10),
            cpu_slot_ids=_slot_ids(20),
            ssd_slot_ids=_slot_ids(30),
        )
        assert ids is not None
        assert ids.h2d_id is not None
        assert ids.disk2h_id is not None
        assert ids.remote2h_id is None
        # 2 ops
        assert len(graph._op_map) == 2
        # H2D depends on DISK2H
        h2d_op = graph._op_map[ids.h2d_id]
        assert ids.disk2h_id in h2d_op.predecessors

    def test_get_chain_with_remote(self, enabled_constructor, graph):
        """GET chain with REMOTE source: H2D + REMOTE2H."""
        ids = enabled_constructor.build_get_chain(
            graph,
            gpu_slot_ids=_slot_ids(10),
            cpu_slot_ids=_slot_ids(20),
            remote_slot_ids=_slot_ids(40),
        )
        assert ids is not None
        assert ids.h2d_id is not None
        assert ids.remote2h_id is not None
        assert ids.disk2h_id is None
        assert len(graph._op_map) == 2
        h2d_op = graph._op_map[ids.h2d_id]
        assert ids.remote2h_id in h2d_op.predecessors

    def test_get_chain_with_ssd_and_remote(self, enabled_constructor, graph):
        """GET chain with both SSD and REMOTE: H2D + DISK2H + REMOTE2H."""
        ids = enabled_constructor.build_get_chain(
            graph,
            gpu_slot_ids=_slot_ids(10),
            cpu_slot_ids=_slot_ids(20),
            ssd_slot_ids=_slot_ids(30),
            remote_slot_ids=_slot_ids(40),
        )
        assert ids is not None
        assert ids.h2d_id is not None
        assert ids.disk2h_id is not None
        assert ids.remote2h_id is not None
        assert len(graph._op_map) == 3
        h2d_op = graph._op_map[ids.h2d_id]
        assert ids.disk2h_id in h2d_op.predecessors
        assert ids.remote2h_id in h2d_op.predecessors

    def test_get_chain_disabled_returns_none(self, disabled_constructor, graph):
        ids = disabled_constructor.build_get_chain(
            graph,
            gpu_slot_ids=_slot_ids(10),
            cpu_slot_ids=_slot_ids(20),
        )
        assert ids is None


# ---------------------------------------------------------------------------
# build_put_chain
# ---------------------------------------------------------------------------

class TestPutChain:

    def test_put_chain_cpu_only(self, enabled_constructor, graph):
        """PUT chain with only CPU: just D2H."""
        ids = enabled_constructor.build_put_chain(
            graph,
            gpu_slot_ids=_slot_ids(10),
            cpu_slot_ids=_slot_ids(20),
        )
        assert ids is not None
        assert ids.d2h_id is not None
        assert ids.h2disk_id is None
        assert ids.h2remote_id is None
        assert len(graph._op_map) == 1

    def test_put_chain_with_ssd(self, enabled_constructor, graph):
        """PUT chain with SSD: D2H + H2DISK, H2DISK depends on D2H."""
        ids = enabled_constructor.build_put_chain(
            graph,
            gpu_slot_ids=_slot_ids(10),
            cpu_slot_ids=_slot_ids(20),
            ssd_slot_ids=_slot_ids(30),
        )
        assert ids is not None
        assert ids.d2h_id is not None
        assert ids.h2disk_id is not None
        assert ids.h2remote_id is None
        assert len(graph._op_map) == 2
        h2disk_op = graph._op_map[ids.h2disk_id]
        assert ids.d2h_id in h2disk_op.predecessors

    def test_put_chain_with_remote(self, enabled_constructor, graph):
        """PUT chain with REMOTE: D2H + H2REMOTE."""
        ids = enabled_constructor.build_put_chain(
            graph,
            gpu_slot_ids=_slot_ids(10),
            cpu_slot_ids=_slot_ids(20),
            remote_slot_ids=_slot_ids(40),
        )
        assert ids is not None
        assert ids.d2h_id is not None
        assert ids.h2remote_id is not None
        assert len(graph._op_map) == 2
        h2remote_op = graph._op_map[ids.h2remote_id]
        assert ids.d2h_id in h2remote_op.predecessors

    def test_put_chain_with_ssd_and_remote(self, enabled_constructor, graph):
        """PUT chain with both SSD and REMOTE."""
        ids = enabled_constructor.build_put_chain(
            graph,
            gpu_slot_ids=_slot_ids(10),
            cpu_slot_ids=_slot_ids(20),
            ssd_slot_ids=_slot_ids(30),
            remote_slot_ids=_slot_ids(40),
        )
        assert ids is not None
        assert ids.d2h_id is not None
        assert ids.h2disk_id is not None
        assert ids.h2remote_id is not None
        assert len(graph._op_map) == 3

    def test_put_chain_disabled_returns_none(self, disabled_constructor, graph):
        ids = disabled_constructor.build_put_chain(
            graph,
            gpu_slot_ids=_slot_ids(10),
            cpu_slot_ids=_slot_ids(20),
        )
        assert ids is None


# ---------------------------------------------------------------------------
# enabled property
# ---------------------------------------------------------------------------

class TestEnabled:

    def test_enabled_requires_config_flag(self):
        gce = _make_gce(enabled=False, has_cpu_pool=True)
        constructor = MambaStateOpConstructor(gce)
        assert not constructor.enabled

    def test_enabled_requires_cpu_pool(self):
        gce = _make_gce(enabled=True, has_cpu_pool=False)
        constructor = MambaStateOpConstructor(gce)
        assert not constructor.enabled

    def test_enabled_when_both_present(self):
        gce = _make_gce(enabled=True, has_cpu_pool=True)
        constructor = MambaStateOpConstructor(gce)
        assert constructor.enabled
