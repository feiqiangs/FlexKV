"""Unit tests for ``flexkv.external.mooncake_store_keys``.

Covers every branch of ``build_key`` plus ``PoolKind`` / ``PoolSpec``:

* base form ``"<hash>_FlexKV"`` / ``"<hash>_FlexKV_indexer"``
* Case 1 (single-node / full-model node, no suffix):
  ``total_layers > 0`` and ``node_layer_end - node_layer_start == total_layers``
* Case 2 (cross-node PP, suffix ``"_pp_rank_{i}_of_{N}"``):
  ``pp_size > 1`` and the node only holds part of the model
* legacy compatibility: ``total_layers == 0`` falls back to legacy logic
* T5 layer-range mapping logic (single-node ``end==total`` -> no suffix /
  cross-node ``end<total`` -> suffix), mirroring transfer_manager.

These tests are pure-python (no torch / RDMA / mooncake SDK required), but the
shared testkit is imported so the package import path is consistent with the
other mooncake test modules.
"""
import _mooncake_store_testkit  # noqa: F401  (installs fake flexkv.c_ext)

import pytest

from flexkv.external.mooncake_store_keys import (
    PoolKind,
    PoolSpec,
    build_key,
)


# ---------------------------------------------------------------------------
# PoolKind / PoolSpec
# ---------------------------------------------------------------------------
def test_pool_kind_values():
    assert PoolKind.KV.value == "FlexKV"
    assert PoolKind.INDEXER.value == "FlexKV_indexer"
    assert PoolKind.SWA.value == "FlexKV_swa"
    # PoolKind is a str Enum so members compare equal to their string value.
    assert PoolKind.KV == "FlexKV"


def test_pool_spec_defaults_required_for_hit():
    spec = PoolSpec(PoolKind.KV)
    assert spec.kind is PoolKind.KV
    assert spec.required_for_hit is True

    spec2 = PoolSpec(PoolKind.INDEXER, required_for_hit=False)
    assert spec2.required_for_hit is False


def test_pool_spec_is_frozen_and_hashable():
    spec = PoolSpec(PoolKind.KV)
    with pytest.raises(Exception):
        spec.kind = PoolKind.INDEXER  # frozen dataclass
    # frozen -> hashable -> usable in sets / dict keys
    assert {PoolSpec(PoolKind.KV), PoolSpec(PoolKind.KV)} == {PoolSpec(PoolKind.KV)}


# ---------------------------------------------------------------------------
# build_key - base form
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "block_hash, kind, expected",
    [
        (123, PoolKind.KV, "123_FlexKV"),
        ("abc", PoolKind.INDEXER, "abc_FlexKV_indexer"),
        (0, PoolKind.SWA, "0_FlexKV_swa"),
    ],
)
def test_build_key_base_form(block_hash, kind, expected):
    """Default call (no PP / layer fields) -> bare '<hash>_<suffix>'."""
    assert build_key(block_hash, kind) == expected


def test_build_key_format_matches_worker_contract():
    # Mirrors the reference integration test's contract assertion.
    assert build_key(123, PoolKind.KV) == "123_FlexKV"
    assert build_key("abc", PoolKind.INDEXER) == "abc_FlexKV_indexer"


# ---------------------------------------------------------------------------
# build_key - Case 1: node CPU pool covers the whole model -> NO suffix
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pp_size", [1, 2, 4])
def test_build_key_case1_full_model_node_no_suffix(pp_size):
    """When (node_layer_end - node_layer_start) == total_layers, no suffix is
    appended regardless of pp_size (single-node deployment)."""
    key = build_key(
        7,
        PoolKind.KV,
        pp_rank=pp_size - 1,
        pp_size=pp_size,
        node_layer_start=0,
        node_layer_end=32,
        total_layers=32,
    )
    assert key == "7_FlexKV"


def test_build_key_case1_nonzero_start_still_full_range():
    """start>0 but the covered range still equals total_layers -> no suffix."""
    key = build_key(
        7,
        PoolKind.KV,
        pp_rank=1,
        pp_size=2,
        node_layer_start=8,
        node_layer_end=40,  # 40 - 8 == 32
        total_layers=32,
    )
    assert key == "7_FlexKV"


# ---------------------------------------------------------------------------
# build_key - Case 2: cross-node PP (partial model) -> suffix
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "pp_rank, pp_size, node_layer_end, total_layers, expected_suffix",
    [
        (0, 2, 16, 32, "_pp_rank_0_of_2"),
        (1, 2, 16, 32, "_pp_rank_1_of_2"),
        (3, 4, 8, 32, "_pp_rank_3_of_4"),
    ],
)
def test_build_key_case2_cross_node_pp_suffix(
    pp_rank, pp_size, node_layer_end, total_layers, expected_suffix
):
    """Cross-node PP (node only holds part of the model: end-start < total)
    -> '<hash>_<suffix>_pp_rank_<i>_of_<N>'."""
    key = build_key(
        99,
        PoolKind.KV,
        pp_rank=pp_rank,
        pp_size=pp_size,
        node_layer_start=0,
        node_layer_end=node_layer_end,
        total_layers=total_layers,
    )
    assert key == f"99_FlexKV{expected_suffix}"


def test_build_key_case2_indexer_pool_keeps_suffix():
    key = build_key(
        99,
        PoolKind.INDEXER,
        pp_rank=1,
        pp_size=2,
        node_layer_start=0,
        node_layer_end=16,
        total_layers=32,
    )
    assert key == "99_FlexKV_indexer_pp_rank_1_of_2"


# ---------------------------------------------------------------------------
# build_key - legacy compatibility (total_layers == 0)
# ---------------------------------------------------------------------------
def test_build_key_legacy_total_layers_zero_pp1_no_suffix():
    """Pre-PR call sites pass no layer fields (total_layers == 0). With
    pp_size==1 the result is the bare base key."""
    assert build_key(5, PoolKind.KV) == "5_FlexKV"
    assert build_key(5, PoolKind.KV, total_layers=0) == "5_FlexKV"


def test_build_key_legacy_total_layers_zero_pp_gt1_gets_suffix():
    """When total_layers==0 the Case-1 short-circuit is skipped, so a pp_size>1
    call still falls through to the legacy per-stage suffix."""
    key = build_key(5, PoolKind.KV, pp_rank=1, pp_size=2, total_layers=0)
    assert key == "5_FlexKV_pp_rank_1_of_2"


# ---------------------------------------------------------------------------
# T5 transfer_manager layer-range mapping logic (small unit test on build_key)
# ---------------------------------------------------------------------------
def _node_layer_end_from_num_layers_on_node(num_layers_on_node: int) -> int:
    """Mirror transfer_manager: node_layer_start=0, end=num_layers_on_node."""
    return num_layers_on_node


@pytest.mark.parametrize(
    "num_layers_on_node, total, pp_rank, pp_size, expect_suffix",
    [
        # single-node / full-model node: num_layers_on_node == total -> no suffix
        (32, 32, 0, 1, False),
        (32, 32, 1, 2, False),   # single-node PP=2 still full model
        (32, 32, 3, 4, False),   # single-node PP=4 still full model
        # cross-node PP: num_layers_on_node < total -> suffix
        (16, 32, 0, 2, True),
        (16, 32, 1, 2, True),
        (8, 32, 2, 4, True),
    ],
)
def test_transfer_manager_layer_range_to_key(
    num_layers_on_node, total, pp_rank, pp_size, expect_suffix
):
    node_layer_start = 0
    node_layer_end = _node_layer_end_from_num_layers_on_node(num_layers_on_node)
    key = build_key(
        42,
        PoolKind.KV,
        pp_rank=pp_rank,
        pp_size=pp_size,
        node_layer_start=node_layer_start,
        node_layer_end=node_layer_end,
        total_layers=total,
    )
    if expect_suffix:
        assert key == f"42_FlexKV_pp_rank_{pp_rank}_of_{pp_size}"
    else:
        assert key == "42_FlexKV"
