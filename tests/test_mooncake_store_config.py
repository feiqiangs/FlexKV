"""Unit tests for ``MooncakeStoreConfig.from_file``.

All cases write a temporary JSON file (via the ``tmp_path`` fixture) and verify:

* every field is parsed from the JSON
* ``global_segment_size`` is converted from GB (JSON) to bytes (config)
* ``override_global_segment_size=0`` produces a pure-client config
  (segment size 0, JSON value ignored)
* an explicit override value wins over the JSON value
* a missing file path raises ValueError
* a non-existent file raises FileNotFoundError

The ``MooncakeStoreClient`` / ``MooncakeStoreCacheEngine`` (which would need a
real cluster) are never instantiated here.
"""
import _mooncake_store_testkit  # noqa: F401  (installs fake flexkv.c_ext)

import json
import os
from types import SimpleNamespace

import pytest

from flexkv.external.mooncake_store_utils import MooncakeStoreConfig


_GB = 1024 * 1024 * 1024


def _write_config(tmp_path, **overrides):
    payload = {
        "master_addr": "192.168.1.1:50051",
        "metadata_server": "P2PHANDSHAKE",
        "protocol": "rdma",
        "device_name": "mlx5_0",
        "local_hostname": "10.0.0.5",
        "global_segment_size": 8,  # GB
        "enable_ssd_offload": False,
        "ssd_offload_path": None,
        "master_metrics_port": 9003,
    }
    payload.update(overrides)
    path = tmp_path / "mooncake_store.json"
    path.write_text(json.dumps(payload))
    return str(path)


def _cache_config_with_path(path):
    # from_file only needs ``getattr(cache_config, "mooncake_store_config_path")``.
    return SimpleNamespace(mooncake_store_config_path=path)


# ---------------------------------------------------------------------------
# Happy path: all fields parsed, GB -> bytes conversion
# ---------------------------------------------------------------------------
def test_from_file_parses_all_fields_and_converts_gb(tmp_path):
    path = _write_config(tmp_path)
    cfg = MooncakeStoreConfig.from_file(_cache_config_with_path(path))

    assert cfg.master_addr == "192.168.1.1:50051"
    assert cfg.metadata_server == "P2PHANDSHAKE"
    assert cfg.protocol == "rdma"
    assert cfg.device_name == "mlx5_0"
    assert cfg.local_hostname == "10.0.0.5"
    # GB -> bytes
    assert cfg.global_segment_size == 8 * _GB
    assert cfg.enable_ssd_offload is False
    assert cfg.ssd_offload_path is None
    assert cfg.master_metrics_port == 9003


def test_from_file_segment_size_other_value(tmp_path):
    path = _write_config(tmp_path, global_segment_size=256)
    cfg = MooncakeStoreConfig.from_file(_cache_config_with_path(path))
    assert cfg.global_segment_size == 256 * _GB


# ---------------------------------------------------------------------------
# override_global_segment_size
# ---------------------------------------------------------------------------
def test_from_file_override_zero_pure_client(tmp_path):
    """override=0 -> pure-client config, segment size is exactly 0 (the JSON
    value is ignored, NOT multiplied)."""
    path = _write_config(tmp_path, global_segment_size=64)
    cfg = MooncakeStoreConfig.from_file(
        _cache_config_with_path(path), override_global_segment_size=0
    )
    assert cfg.global_segment_size == 0


def test_from_file_override_nonzero_wins_over_json(tmp_path):
    """A non-None override is used verbatim (bytes, no GB multiply)."""
    path = _write_config(tmp_path, global_segment_size=64)
    cfg = MooncakeStoreConfig.from_file(
        _cache_config_with_path(path), override_global_segment_size=12345
    )
    assert cfg.global_segment_size == 12345


def test_from_file_override_none_falls_back_to_json(tmp_path):
    path = _write_config(tmp_path, global_segment_size=4)
    cfg = MooncakeStoreConfig.from_file(
        _cache_config_with_path(path), override_global_segment_size=None
    )
    assert cfg.global_segment_size == 4 * _GB


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------
def test_from_file_missing_path_raises_value_error(monkeypatch):
    """No path in cache_config AND no env var -> ValueError."""
    monkeypatch.delenv("FLEXKV_MOONCAKE_STORE_CONFIG_PATH", raising=False)
    cache_config = SimpleNamespace(mooncake_store_config_path=None)
    with pytest.raises(ValueError):
        MooncakeStoreConfig.from_file(cache_config)


def test_from_file_nonexistent_file_raises(tmp_path):
    missing = str(tmp_path / "does_not_exist.json")
    cache_config = _cache_config_with_path(missing)
    with pytest.raises(FileNotFoundError):
        MooncakeStoreConfig.from_file(cache_config)


def test_from_file_path_from_env_var(tmp_path, monkeypatch):
    """When cache_config has no path, the env var is consulted."""
    path = _write_config(tmp_path, global_segment_size=2)
    monkeypatch.setenv("FLEXKV_MOONCAKE_STORE_CONFIG_PATH", path)
    cache_config = SimpleNamespace(mooncake_store_config_path=None)
    cfg = MooncakeStoreConfig.from_file(cache_config)
    assert cfg.global_segment_size == 2 * _GB
