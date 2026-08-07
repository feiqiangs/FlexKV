"""Unit tests for P0: MambaActiveStatePool + MambaCheckpointPool.

Tests cover:
  - Pool allocation / free / reset
  - int8 quantization / dequantization accuracy
  - CoW copy_from correctness
  - Snapshot / store / load round-trip
  - Edge cases (slot 0 reserved, pool full, invalid slot)

Uses CPU-only torch tensors (no GPU needed). Marked as unit tests.
"""
import math

import pytest
import torch

from flexkv.mamba_state.config import MambaStatePoolConfig
from flexkv.mamba_state.active_state_pool import MambaActiveStatePool
from flexkv.mamba_state.checkpoint_pool import MambaCheckpointPool

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(
    num_linear_layers: int = 4,
    num_heads: int = 8,
    head_v_dim: int = 16,
    head_k_dim: int = 16,
    conv_shape=(3, 128),
    num_active_slots: int = 8,
    num_cpu_slots: int = 16,
) -> MambaStatePoolConfig:
    return MambaStatePoolConfig(
        num_linear_layers=num_linear_layers,
        num_heads=num_heads,
        head_v_dim=head_v_dim,
        head_k_dim=head_k_dim,
        conv_shape=conv_shape,
        num_active_slots=num_active_slots,
        num_cpu_slots=num_cpu_slots,
    )


@pytest.fixture
def config():
    return _make_config()


@pytest.fixture
def active_pool(config):
    return MambaActiveStatePool(config, device="cpu")


@pytest.fixture
def ckpt_pool(config):
    return MambaCheckpointPool(config, device="cpu")


# ---------------------------------------------------------------------------
# ActiveStatePool: allocation / free / reset
# ---------------------------------------------------------------------------

class TestActivePoolAllocation:

    def test_allocate_returns_valid_slots(self, active_pool):
        """Allocate should return slot ids in [1, max_reqs)."""
        s1 = active_pool.allocate()
        s2 = active_pool.allocate()
        assert s1 is not None
        assert s2 is not None
        assert s1 >= 1 and s1 < active_pool.max_reqs
        assert s2 >= 1 and s2 < active_pool.max_reqs
        assert s1 != s2

    def test_slot_zero_never_allocated(self, active_pool):
        """Slot 0 is reserved for dummy writes."""
        allocated = set()
        for _ in range(active_pool.max_reqs - 1):
            s = active_pool.allocate()
            assert s is not None
            allocated.add(s)
        assert 0 not in allocated

    def test_pool_full_returns_none(self, active_pool):
        """When all usable slots are taken, allocate returns None."""
        # max_reqs - 1 usable slots (slot 0 reserved)
        for _ in range(active_pool.max_reqs - 1):
            assert active_pool.allocate() is not None
        assert active_pool.allocate() is None

    def test_free_then_reallocate(self, active_pool):
        """Freed slots can be reused."""
        s = active_pool.allocate()
        assert s is not None
        active_pool.free(s)
        s2 = active_pool.allocate()
        assert s2 is not None
        # Free-list is LIFO (stack), so should get the same slot back
        assert s2 == s

    def test_free_invalid_slot_raises(self, active_pool):
        with pytest.raises(ValueError):
            active_pool.free(0)   # reserved
        with pytest.raises(ValueError):
            active_pool.free(999)  # out of range

    def test_double_free_is_noop(self, active_pool):
        s = active_pool.allocate()
        active_pool.free(s)
        active_pool.free(s)  # should not raise
        # Should still have the right count
        assert active_pool.num_used == active_pool.max_reqs - 1 - active_pool.num_free

    def test_reset_frees_all(self, active_pool):
        for _ in range(3):
            active_pool.allocate()
        active_pool.reset()
        assert active_pool.num_free == active_pool.max_reqs - 1
        assert active_pool.num_used == 0

    def test_num_used_and_free(self, active_pool):
        total_usable = active_pool.max_reqs - 1
        assert active_pool.num_free == total_usable
        assert active_pool.num_used == 0
        active_pool.allocate()
        assert active_pool.num_free == total_usable - 1
        assert active_pool.num_used == 1


# ---------------------------------------------------------------------------
# ActiveStatePool: CoW copy_from
# ---------------------------------------------------------------------------

class TestActivePoolCoW:

    def test_copy_from_produces_identical_state(self, active_pool):
        """copy_from should make dst identical to src."""
        src = active_pool.allocate()
        dst = active_pool.allocate()
        # Write some data to src
        active_pool.temporal_state[:, src] = torch.randn_like(
            active_pool.temporal_state[:, src]
        )
        active_pool.conv_state[:, src] = torch.randn_like(
            active_pool.conv_state[:, src]
        )
        # Copy
        active_pool.copy_from(src, dst)
        # Verify identical
        assert torch.equal(active_pool.temporal_state[:, src],
                          active_pool.temporal_state[:, dst])
        assert torch.equal(active_pool.conv_state[:, src],
                          active_pool.conv_state[:, dst])

    def test_cow_independence_after_update(self, active_pool):
        """After copy_from, updating dst should NOT affect src."""
        src = active_pool.allocate()
        dst = active_pool.allocate()
        active_pool.temporal_state[:, src] = torch.ones_like(
            active_pool.temporal_state[:, src]
        )
        active_pool.copy_from(src, dst)
        # Now modify dst
        active_pool.temporal_state[:, dst] = torch.zeros_like(
            active_pool.temporal_state[:, dst]
        )
        # src should still be ones
        assert torch.all(active_pool.temporal_state[:, src] == 1.0)
        assert torch.all(active_pool.temporal_state[:, dst] == 0.0)

    def test_snapshot_returns_references(self, active_pool):
        """snapshot returns references, not copies."""
        s = active_pool.allocate()
        active_pool.temporal_state[:, s] = 42.0
        temporal_ref, conv_ref = active_pool.snapshot(s)
        # Modifying the pool should be visible through the reference
        assert torch.all(temporal_ref == 42.0)

    def test_clone_snapshot_is_independent(self, active_pool):
        """clone_snapshot returns copies independent of the source."""
        s = active_pool.allocate()
        active_pool.temporal_state[:, s] = 42.0
        temporal_clone, conv_clone = active_pool.clone_snapshot(s)
        # Modify source
        active_pool.temporal_state[:, s] = 0.0
        # Clone should still have the old value
        assert torch.all(temporal_clone == 42.0)

    def test_restore_from_tensors(self, active_pool):
        """restore_from_tensors loads external data into a slot."""
        s = active_pool.allocate()
        L, H, dv, dk = 4, 8, 16, 16
        temporal = torch.randn(L, H, dv, dk, dtype=torch.bfloat16)
        conv = torch.randn(L, 3, 128, dtype=torch.bfloat16)
        active_pool.restore_from_tensors(s, temporal, conv)
        assert torch.equal(active_pool.temporal_state[:, s], temporal)
        assert torch.equal(active_pool.conv_state[:, s], conv)


# ---------------------------------------------------------------------------
# CheckpointPool: int8 quantization / dequantization
# ---------------------------------------------------------------------------

class TestInt8Quantization:

    def test_quantize_dequantize_small_values(self):
        """Quantize -> dequantize round-trip for small uniform values."""
        state = torch.randn(2, 4, 16, 16, dtype=torch.bfloat16) * 0.1
        qdata, scale = MambaCheckpointPool.quantize(state)
        assert qdata.dtype == torch.int8
        assert scale.dtype == torch.bfloat16
        assert scale.shape == (2, 4, 1, 16)
        recovered = MambaCheckpointPool.dequantize(qdata, scale)
        # Use norm-based comparison (avoids div-by-zero on near-zero values)
        orig_norm = state.float().norm().item()
        diff_norm = (recovered.float() - state.float()).norm().item()
        rel_norm_err = diff_norm / (orig_norm + 1e-8)
        assert rel_norm_err < 0.05  # < 5% relative norm error

    def test_quantize_dequantize_large_values(self):
        """Large values should be preserved accurately."""
        state = torch.randn(4, 8, 16, 16, dtype=torch.bfloat16) * 100
        qdata, scale = MambaCheckpointPool.quantize(state)
        recovered = MambaCheckpointPool.dequantize(qdata, scale)
        orig_norm = state.float().norm().item()
        diff_norm = (recovered.float() - state.float()).norm().item()
        rel_norm_err = diff_norm / (orig_norm + 1e-8)
        assert rel_norm_err < 0.02  # < 2% for large values

    def test_quantize_clamps_to_qmax(self):
        """Quantized values should be in [-127, 127]."""
        state = torch.randn(2, 4, 16, 16, dtype=torch.bfloat16) * 1000
        qdata, _ = MambaCheckpointPool.quantize(state)
        assert qdata.min().item() >= -127
        assert qdata.max().item() <= 127

    def test_scale_reduction_axis_is_d_v(self):
        """Scale should reduce over d_v (dim=-2), not d_k (dim=-1)."""
        L, H, dv, dk = 2, 4, 8, 6
        state = torch.randn(L, H, dv, dk, dtype=torch.bfloat16)
        _, scale = MambaCheckpointPool.quantize(state)
        # scale shape should be [L, H, 1, dk] (reduced over dv)
        assert scale.shape == (L, H, 1, dk)

    def test_scale_matches_amax_over_d_v(self):
        """Scale = amax(|state|, dim=d_v) / 127."""
        state = torch.randn(2, 4, 8, 6, dtype=torch.bfloat16)
        _, scale = MambaCheckpointPool.quantize(state)
        expected_amax = state.abs().to(torch.float32).amax(dim=-2, keepdim=True)
        expected_scale = (expected_amax / 127.0).to(torch.bfloat16)
        assert torch.allclose(scale.to(torch.float32), expected_scale.to(torch.float32), atol=1e-4)

    def test_quantize_preserves_zero(self):
        """Zero values should quantize to exactly 0."""
        state = torch.zeros(2, 4, 8, 6, dtype=torch.bfloat16)
        qdata, _ = MambaCheckpointPool.quantize(state)
        assert torch.all(qdata == 0)

    def test_dequantize_dtype_is_bf16(self):
        """Dequantized output should be bf16."""
        state = torch.randn(2, 4, 8, 6, dtype=torch.bfloat16)
        qdata, scale = MambaCheckpointPool.quantize(state)
        recovered = MambaCheckpointPool.dequantize(qdata, scale)
        assert recovered.dtype == torch.bfloat16

    def test_quantize_norm_preservation(self):
        """Quantized state's Frobenius norm should be close to original."""
        state = torch.randn(4, 8, 16, 16, dtype=torch.bfloat16)
        qdata, scale = MambaCheckpointPool.quantize(state)
        recovered = MambaCheckpointPool.dequantize(qdata, scale)
        orig_norm = state.float().norm().item()
        rec_norm = recovered.float().norm().item()
        rel_diff = abs(orig_norm - rec_norm) / (orig_norm + 1e-8)
        assert rel_diff < 1e-3  # < 0.1% norm difference


# ---------------------------------------------------------------------------
# CheckpointPool: store / load round-trip
# ---------------------------------------------------------------------------

class TestCheckpointStoreLoad:

    def test_store_and_load_round_trip(self, active_pool, ckpt_pool):
        """Store from active -> load back -> compare."""
        s = active_pool.allocate()
        ckpt_s = ckpt_pool.allocate()
        # Fill active with known data
        active_pool.temporal_state[:, s] = torch.randn_like(
            active_pool.temporal_state[:, s]
        ) * 10
        active_pool.conv_state[:, s] = torch.randn_like(
            active_pool.conv_state[:, s]
        )
        # Snapshot + store
        temporal, conv = active_pool.clone_snapshot(s)
        ckpt_pool.store_from_active(ckpt_s, temporal, conv)
        # Load back
        temporal_recovered, conv_recovered = ckpt_pool.load_to_active(ckpt_s)
        # Conv should be exact (not quantized)
        assert torch.equal(conv_recovered, active_pool.conv_state[:, s])
        # Temporal: norm-based comparison (int8 quantized)
        orig = active_pool.temporal_state[:, s].float()
        recov = temporal_recovered.float()
        rel_norm_err = (recov - orig).norm().item() / (orig.norm().item() + 1e-8)
        assert rel_norm_err < 0.05

    def test_store_does_not_modify_source(self, active_pool, ckpt_pool):
        """Storing should not modify the active slot."""
        s = active_pool.allocate()
        ckpt_s = ckpt_pool.allocate()
        original = active_pool.temporal_state[:, s].clone()
        temporal, conv = active_pool.clone_snapshot(s)
        ckpt_pool.store_from_active(ckpt_s, temporal, conv)
        assert torch.equal(active_pool.temporal_state[:, s], original)

    def test_load_into_different_slot(self, active_pool, ckpt_pool):
        """Load checkpoint into a different active slot (CoW scenario)."""
        src = active_pool.allocate()
        ckpt_s = ckpt_pool.allocate()
        dst = active_pool.allocate()
        # Store from src
        active_pool.temporal_state[:, src] = 42.0
        temporal, conv = active_pool.clone_snapshot(src)
        ckpt_pool.store_from_active(ckpt_s, temporal, conv)
        # Load into dst
        temporal_recovered, conv_recovered = ckpt_pool.load_to_active(ckpt_s)
        active_pool.restore_from_tensors(dst, temporal_recovered, conv_recovered)
        # dst should have ~42.0 (int8 quantized)
        assert torch.allclose(
            active_pool.temporal_state[:, dst].float(),
            torch.full_like(active_pool.temporal_state[:, dst].float(), 42.0),
            atol=1.0,
        )

    def test_multiple_stores_and_loads(self, active_pool, ckpt_pool):
        """Multiple store/load cycles should work independently."""
        for i in range(3):
            s = active_pool.allocate()
            ckpt_s = ckpt_pool.allocate()
            val = float(i + 1) * 10.0
            active_pool.temporal_state[:, s] = val
            temporal, conv = active_pool.clone_snapshot(s)
            ckpt_pool.store_from_active(ckpt_s, temporal, conv)
            active_pool.free(s)
        # Now load each checkpoint
        for i in range(3):
            ckpt_s = i + 1  # slots allocated in order 1,2,3
            temporal, conv = ckpt_pool.load_to_active(ckpt_s)
            val = float(i + 1) * 10.0
            assert torch.allclose(
                temporal.float(),
                torch.full(temporal.shape, val, dtype=torch.float32),
                atol=1.0,
            )


# ---------------------------------------------------------------------------
# CheckpointPool: allocation / free
# ---------------------------------------------------------------------------

class TestCheckpointPoolAllocation:

    def test_allocate_and_free(self, ckpt_pool):
        s = ckpt_pool.allocate()
        assert s is not None
        assert s >= 1
        ckpt_pool.free(s)
        s2 = ckpt_pool.allocate()
        assert s2 == s  # LIFO

    def test_pool_full(self, ckpt_pool):
        usable = ckpt_pool.config.num_cpu_slots - 1
        for _ in range(usable):
            assert ckpt_pool.allocate() is not None
        assert ckpt_pool.allocate() is None

    def test_reset(self, ckpt_pool):
        for _ in range(3):
            ckpt_pool.allocate()
        ckpt_pool.reset()
        assert ckpt_pool.num_free == ckpt_pool.config.num_cpu_slots - 1


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:

    def test_enabled_requires_nonzero(self):
        cfg = MambaStatePoolConfig()
        assert not cfg.enabled
        cfg.num_linear_layers = 4
        cfg.num_heads = 8
        assert cfg.enabled

    def test_state_bytes_per_layer(self, config):
        expected = 8 * 16 * 16 * 2  # H * dv * dk * bf16
        assert config.state_bytes_per_layer == expected

    def test_total_active_bytes(self, config):
        per_layer = config.state_bytes_per_layer + config.conv_bytes_per_layer
        assert config.total_active_bytes == 4 * per_layer

    def test_total_checkpoint_bytes_int8(self, config):
        L, H, dv, dk = 4, 8, 16, 16
        state_int8 = L * H * dv * dk * 1
        scale = L * H * 1 * dk * 2
        conv = L * config.conv_bytes_per_layer
        assert config.total_checkpoint_bytes_int8 == state_int8 + scale + conv
