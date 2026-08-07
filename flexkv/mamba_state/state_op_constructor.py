"""MambaStateOpConstructor — linear-attention state peer-op graph builder.

Mirrors SWAOpConstructor: given resolved state slot ids, appends peer state
ops into the same TransferOpGraph as the token-KV ops.

Responsibilities (control plane only — no byte movement):
  * Build linear-state *peer* ops into the SAME TransferOpGraph as the
    token-KV ops, with tier dependencies that mirror the token-KV graph.
  * State ops reuse STANDARD transfer types (H2D / D2H / DISK2H / H2DISK /
    REMOTE2H / H2REMOTE) and carry ``is_mamba_state=True`` so the transfer
    engine routes them to the dedicated state worker; their src/dst block
    ids are state-pool slot ids.

State is a first-class PEER op, NOT a child derived from the token-KV op
(independent slot space; the state-only case has no token-KV op to derive
from). The data-plane routes on the ``is_mamba_state`` flag (a plain
TransferOp field) rather than dedicated transfer types, so the graph stays
homogeneous and routing is a single boolean.

Everything here is gated by ``cache_config.enable_linear_state_transfer``
(default False): until the dedicated state transfer worker (data plane) is
registered, the build helpers are no-ops so a state op never reaches the
transfer engine.
"""
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

from flexkv.common.transfer import DeviceType, TransferOp, TransferOpGraph, TransferType


@dataclass
class LinearStatePutChainOpIds:
    """Op ids for a PUT-side state transfer chain."""
    d2h_id: Optional[int] = None
    h2disk_id: Optional[int] = None
    h2remote_id: Optional[int] = None


@dataclass
class LinearStateGetChainOpIds:
    """Op ids for a GET-side state transfer chain."""
    h2d_id: Optional[int] = None
    disk2h_id: Optional[int] = None
    remote2h_id: Optional[int] = None


class MambaStateOpConstructor:
    """Linear-attention state peer-op graph construction.

    Holds a back-reference to the owning ``GlobalCacheEngine`` to reach the
    cache config. Per-tier state primitives live on the engines themselves;
    the token-KV get/put implementations choose the source/destination slots
    and this class only appends the corresponding state peer ops.
    """

    def __init__(self, global_cache_engine) -> None:
        self._gce = global_cache_engine

    # --- tier access -------------------------------------------------------

    def _engine(self, device_type: DeviceType):
        return self._gce.cache_engines.get(device_type)

    def _state_enabled_tier(self, device_type: DeviceType) -> bool:
        """True iff the tier's engine has a linear-state checkpoint pool."""
        engine = self._engine(device_type)
        return bool(getattr(engine, "mamba_state_enabled", False)) if engine is not None else False

    @property
    def enabled(self) -> bool:
        """True when state transfer is gated on AND the CPU tier has a state pool."""
        cfg = getattr(self._gce, "cache_config", None)
        return bool(getattr(cfg, "enable_linear_state_transfer", False)) and \
            self._state_enabled_tier(DeviceType.CPU)

    # --- single op builder -------------------------------------------------

    def build_state_op(
        self,
        graph: TransferOpGraph,
        transfer_type: TransferType,
        src_slot_ids: np.ndarray,
        dst_slot_ids: np.ndarray,
        dp_client_id: int = 0,
    ) -> Optional[int]:
        """Build a single ``is_mamba_state=True`` TransferOp.

        Returns op_id, or None if disabled / empty slots.
        """
        if not self.enabled:
            return None
        if src_slot_ids.size == 0 or dst_slot_ids.size == 0:
            return None
        op = TransferOp(
            graph_id=graph.graph_id,
            transfer_type=transfer_type,
            src_block_ids=src_slot_ids,
            dst_block_ids=dst_slot_ids,
            dp_client_id=dp_client_id,
            is_mamba_state=True,
        )
        graph.add_transfer_op(op)
        return op.op_id

    # --- GET chain: load state from CPU/SSD/Remote → GPU active slot -------

    def build_get_chain(
        self,
        graph: TransferOpGraph,
        gpu_slot_ids: np.ndarray,
        cpu_slot_ids: np.ndarray,
        ssd_slot_ids: Optional[np.ndarray] = None,
        remote_slot_ids: Optional[np.ndarray] = None,
        dp_client_id: int = 0,
    ) -> Optional[LinearStateGetChainOpIds]:
        """Build GET-side state load chain.

        Topology mirrors token-KV GET:
          state H2D (CPU → GPU)  ← reported (joins VIRTUAL barrier)
            ├── depends on → state DISK2H (SSD → CPU)   [if SSD source]
            └── depends on → state REMOTE2H (remote → CPU) [if REMOTE source]
        """
        ids = LinearStateGetChainOpIds()

        # Terminal H2D: CPU slot → GPU active slot
        ids.h2d_id = self.build_state_op(
            graph, TransferType.H2D, cpu_slot_ids, gpu_slot_ids, dp_client_id
        )
        if ids.h2d_id is None:
            return None

        # Optional DISK2H staging
        if ssd_slot_ids is not None and ssd_slot_ids.size > 0:
            ids.disk2h_id = self.build_state_op(
                graph, TransferType.DISK2H, ssd_slot_ids, cpu_slot_ids, dp_client_id
            )
            if ids.disk2h_id is not None:
                graph.add_dependency(ids.h2d_id, ids.disk2h_id)

        # Optional REMOTE2H staging
        if remote_slot_ids is not None and remote_slot_ids.size > 0:
            ids.remote2h_id = self.build_state_op(
                graph, TransferType.REMOTE2H, remote_slot_ids, cpu_slot_ids, dp_client_id
            )
            if ids.remote2h_id is not None:
                graph.add_dependency(ids.h2d_id, ids.remote2h_id)

        return ids

    # --- PUT chain: store state from GPU → CPU/SSD/Remote ------------------

    def build_put_chain(
        self,
        graph: TransferOpGraph,
        gpu_slot_ids: np.ndarray,
        cpu_slot_ids: np.ndarray,
        ssd_slot_ids: Optional[np.ndarray] = None,
        remote_slot_ids: Optional[np.ndarray] = None,
        dp_client_id: int = 0,
    ) -> Optional[LinearStatePutChainOpIds]:
        """Build PUT-side state store chain.

        Topology mirrors token-KV PUT:
          state D2H (GPU → CPU)  ← reported
            ├── state H2DISK depends on D2H   [fire-and-forget, NOT reported]
            └── state H2REMOTE depends on D2H  [fire-and-forget, NOT reported]
        """
        ids = LinearStatePutChainOpIds()

        # D2H: GPU active slot → CPU checkpoint slot
        ids.d2h_id = self.build_state_op(
            graph, TransferType.D2H, gpu_slot_ids, cpu_slot_ids, dp_client_id
        )
        if ids.d2h_id is None:
            return None

        # Optional H2DISK write-through
        if ssd_slot_ids is not None and ssd_slot_ids.size > 0:
            ids.h2disk_id = self.build_state_op(
                graph, TransferType.H2DISK, cpu_slot_ids, ssd_slot_ids, dp_client_id
            )
            if ids.h2disk_id is not None:
                graph.add_dependency(ids.h2disk_id, ids.d2h_id)

        # Optional H2REMOTE write-through
        if remote_slot_ids is not None and remote_slot_ids.size > 0:
            ids.h2remote_id = self.build_state_op(
                graph, TransferType.H2REMOTE, cpu_slot_ids, remote_slot_ids, dp_client_id
            )
            if ids.h2remote_id is not None:
                graph.add_dependency(ids.h2remote_id, ids.d2h_id)

        return ids
