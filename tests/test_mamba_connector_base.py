"""Unit tests for MambaStateConnectorBase (framework-agnostic L2/L3 bridge).

Tests the store/lookup/retrieve/evict lifecycle without GPU — uses
CPU tensors as mock mamba state.
"""

import pytest
import torch
import numpy as np

from flexkv.mamba_state.connector_base import (
    MambaStateConfig,
    MambaStateConnectorBase,
)


@pytest.fixture
def config():
    return MambaStateConfig(
        num_linear_layers=4,
        num_heads=8,
        head_v_dim=16,
        head_k_dim=16,
        conv_shapes=[(3, 128)],
        conv_dtype=torch.bfloat16,
        temporal_dtype=torch.bfloat16,
        num_cpu_slots=16,
    )


@pytest.fixture
def connector(config):
    return MambaStateConnectorBase(config)


@pytest.fixture
def mock_state():
    """Mock mamba state: temporal [L, H, d_v, d_k] + conv [L, *conv_shape]."""
    temporal = torch.randn(4, 8, 16, 16, dtype=torch.bfloat16)
    conv = [torch.randn(4, 3, 128, dtype=torch.bfloat16)]
    return temporal, conv


class TestStore:
    def test_store_and_lookup_hit(self, connector, mock_state):
        token_ids = list(range(100))
        temporal, conv = mock_state
        connector.store(token_ids, temporal, conv)
        hit, matched = connector.lookup(token_ids)
        assert hit is True
        assert matched == 100

    def test_store_and_retrieve(self, connector, mock_state):
        token_ids = list(range(50))
        temporal, conv = mock_state
        connector.store(token_ids, temporal, conv)
        result = connector.retrieve(token_ids)
        assert result is not None
        temporal_ret, conv_ret = result
        assert temporal_ret.shape == temporal.shape
        rel_err = (temporal_ret.float() - temporal.float()).abs().max() / \
                  (temporal.float().abs().max() + 1e-6)
        assert rel_err < 0.1

    def test_store_dedup(self, connector, mock_state):
        token_ids = list(range(100))
        connector.store(token_ids, *mock_state)
        connector.store(token_ids, *mock_state)
        stats = connector.get_stats()
        assert stats["mamba_cached_count"] == 1

    def test_store_multiple_prefixes(self, connector, mock_state):
        for i in range(5):
            token_ids = list(range(i * 100, (i + 1) * 100))
            connector.store(token_ids, *mock_state)
        stats = connector.get_stats()
        assert stats["mamba_cached_count"] == 5


class TestLookup:
    def test_lookup_miss(self, connector):
        hit, matched = connector.lookup(list(range(100)))
        assert hit is False
        assert matched == 0

    def test_lookup_empty(self, connector):
        hit, matched = connector.lookup([])
        assert hit is False


class TestRetrieve:
    def test_retrieve_miss(self, connector):
        result = connector.retrieve(list(range(100)))
        assert result is None

    def test_retrieve_returns_list_conv(self, connector, mock_state):
        token_ids = list(range(50))
        connector.store(token_ids, *mock_state)
        result = connector.retrieve(token_ids)
        assert result is not None
        temporal, conv = result
        assert isinstance(conv, (list, tuple))


class TestEviction:
    def test_evict_by_prefix(self, connector, mock_state):
        token_ids = list(range(100))
        connector.store(token_ids, *mock_state)
        evicted = connector.evict_by_prefix(token_ids)
        assert evicted is True
        hit, _ = connector.lookup(token_ids)
        assert hit is False

    def test_evict_nonexistent(self, connector):
        evicted = connector.evict_by_prefix(list(range(100)))
        assert evicted is False

    def test_lru_eviction_on_full(self, config, mock_state):
        config.num_cpu_slots = 4
        conn = MambaStateConnectorBase(config)
        for i in range(6):
            conn.store(list(range(i * 10, (i + 1) * 10)), *mock_state)
        stats = conn.get_stats()
        assert stats["mamba_cached_count"] <= 4


class TestStats:
    def test_stats_after_operations(self, connector, mock_state):
        token_ids = list(range(100))
        connector.store(token_ids, *mock_state)
        connector.lookup(token_ids)
        connector.retrieve(token_ids)
        stats = connector.get_stats()
        assert stats["mamba_store_count"] == 1
        assert stats["mamba_lookup_count"] == 1
        assert stats["mamba_hit_count"] == 1
        assert stats["mamba_retrieve_count"] == 1


class TestReset:
    def test_reset_clears_all(self, connector, mock_state):
        connector.store(list(range(100)), *mock_state)
        connector.reset()
        stats = connector.get_stats()
        assert stats["mamba_cached_count"] == 0

    def test_reset_idempotent(self, connector):
        connector.reset()
        connector.reset()


class TestPrefetch:
    def test_prefetch_stub(self, connector):
        result = connector.prefetch_mamba_state(list(range(100)))
        assert result is False

    def test_check_prefetch_progress_stub(self, connector):
        result = connector.check_prefetch_progress(list(range(100)))
        assert result is True

    def test_cancel_prefetch_stub(self, connector):
        connector.cancel_prefetch(list(range(100)))
