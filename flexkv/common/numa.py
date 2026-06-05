# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NUMA policy utilities for FlexKV subprocesses.

This module is intentionally light-weight (pure-stdlib + ctypes) so that it can
be imported very early in subprocess entrypoints, before ``import torch`` or any
other heavy module that would touch ``mempolicy``.

Entry points:
    * ``apply_numa_policy(role)``    - reset current process mempolicy according
                                       to ``FLEXKV_NUMA_POLICY``.
    * ``wrap_with_numactl(cmd)``     - wrap a ``subprocess.Popen`` argv list with
                                       a ``numactl`` prefix according to the
                                       same env vars (so child processes start
                                       with the desired policy regardless of
                                       parent inheritance).
    * ``dump_numa_maps_summary``     - parse ``/proc/<pid>/numa_maps`` and log
                                       per-NUMA-node RSS in MiB, used to verify
                                       the policy actually took effect.

Recognised environment variables (all read at function-call time, never cached):
    FLEXKV_NUMA_POLICY          interleave (default) | follow_sglang | preferred
                                | bind | none
    FLEXKV_NUMA_BIND_NODES      comma-separated NUMA node ids, e.g. "0,3,4,5"
                                used by ``bind`` and ``follow_sglang``.
    FLEXKV_NUMA_PREFERRED_NODE  single NUMA node id, used by ``preferred``.
    FLEXKV_DISABLE_NUMA_POLICY  if set to "1", every entry-point becomes a
                                no-op (rollback switch).
"""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
from typing import Iterable, List, Optional

from flexkv.common.debug import flexkv_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENV_POLICY = "FLEXKV_NUMA_POLICY"
ENV_BIND_NODES = "FLEXKV_NUMA_BIND_NODES"
ENV_PREFERRED_NODE = "FLEXKV_NUMA_PREFERRED_NODE"
ENV_DISABLE = "FLEXKV_DISABLE_NUMA_POLICY"

POLICY_INTERLEAVE = "interleave"
POLICY_FOLLOW_SGLANG = "follow_sglang"
POLICY_PREFERRED = "preferred"
POLICY_BIND = "bind"
POLICY_NONE = "none"
DEFAULT_POLICY = POLICY_INTERLEAVE
VALID_POLICIES = {
    POLICY_INTERLEAVE,
    POLICY_FOLLOW_SGLANG,
    POLICY_PREFERRED,
    POLICY_BIND,
    POLICY_NONE,
}

# Track whether we already emitted a warning about libnuma being unavailable,
# so that we do not spam the log on every call.
_warned_unavailable = False


# ---------------------------------------------------------------------------
# libnuma binding
# ---------------------------------------------------------------------------
class _LibNuma:
    """Lazy-loaded ctypes wrapper around libnuma.so."""

    _instance: Optional["_LibNuma"] = None

    def __init__(self, lib: ctypes.CDLL):
        self.lib = lib

        # int numa_available(void);
        lib.numa_available.restype = ctypes.c_int
        lib.numa_available.argtypes = []

        # int numa_max_node(void);
        lib.numa_max_node.restype = ctypes.c_int
        lib.numa_max_node.argtypes = []

        # struct bitmask *numa_allocate_nodemask(void);
        lib.numa_allocate_nodemask.restype = ctypes.c_void_p
        lib.numa_allocate_nodemask.argtypes = []

        # void numa_bitmask_clearall(struct bitmask *);
        lib.numa_bitmask_clearall.restype = ctypes.c_void_p
        lib.numa_bitmask_clearall.argtypes = [ctypes.c_void_p]

        # struct bitmask *numa_bitmask_setbit(struct bitmask *, unsigned int);
        lib.numa_bitmask_setbit.restype = ctypes.c_void_p
        lib.numa_bitmask_setbit.argtypes = [ctypes.c_void_p, ctypes.c_uint]

        # void numa_bitmask_free(struct bitmask *);
        lib.numa_bitmask_free.restype = None
        lib.numa_bitmask_free.argtypes = [ctypes.c_void_p]

        # void numa_set_interleave_mask(struct bitmask *);
        lib.numa_set_interleave_mask.restype = None
        lib.numa_set_interleave_mask.argtypes = [ctypes.c_void_p]

        # void numa_set_membind(struct bitmask *);
        lib.numa_set_membind.restype = None
        lib.numa_set_membind.argtypes = [ctypes.c_void_p]

        # void numa_set_preferred(int);
        lib.numa_set_preferred.restype = None
        lib.numa_set_preferred.argtypes = [ctypes.c_int]

        # void numa_set_localalloc(void);
        lib.numa_set_localalloc.restype = None
        lib.numa_set_localalloc.argtypes = []

        # ``numa_all_nodes_ptr`` is a global ``struct bitmask *`` exported by
        # libnuma that already represents the set of all nodes available to the
        # caller. We read it lazily because it is only initialised after the
        # first ``numa_available()`` call.
        try:
            self._numa_all_nodes_ptr_addr = ctypes.addressof(
                ctypes.c_void_p.in_dll(lib, "numa_all_nodes_ptr")
            )
        except Exception:
            self._numa_all_nodes_ptr_addr = None

    # ----- lifecycle ------------------------------------------------------
    @classmethod
    def get(cls) -> Optional["_LibNuma"]:
        global _warned_unavailable
        if cls._instance is not None:
            return cls._instance
        for soname in ("libnuma.so.1", "libnuma.so"):
            try:
                lib = ctypes.CDLL(soname)
            except OSError:
                continue
            inst = cls(lib)
            if inst.lib.numa_available() < 0:
                if not _warned_unavailable:
                    flexkv_logger.warning(
                        "[FlexKV][NUMA] libnuma loaded but numa_available()<0; "
                        "skipping NUMA policy."
                    )
                    _warned_unavailable = True
                return None
            cls._instance = inst
            return inst
        if not _warned_unavailable:
            flexkv_logger.warning(
                "[FlexKV][NUMA] libnuma.so not found; skipping NUMA policy."
            )
            _warned_unavailable = True
        return None

    # ----- helpers --------------------------------------------------------
    def numa_all_nodes_ptr(self) -> Optional[int]:
        """Return the libnuma global ``numa_all_nodes_ptr`` (a ``bitmask*``)."""
        if self._numa_all_nodes_ptr_addr is None:
            return None
        ptr = ctypes.c_void_p.from_address(self._numa_all_nodes_ptr_addr).value
        return ptr if ptr else None

    def build_nodemask(self, nodes: Iterable[int]) -> Optional[int]:
        """Allocate a ``bitmask`` with only the given node ids set.

        Caller owns the returned pointer and must free it via
        ``numa_bitmask_free``.
        """
        mask = self.lib.numa_allocate_nodemask()
        if not mask:
            return None
        self.lib.numa_bitmask_clearall(mask)
        for n in nodes:
            self.lib.numa_bitmask_setbit(mask, ctypes.c_uint(int(n)))
        return mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_disabled() -> bool:
    return os.environ.get(ENV_DISABLE, "0") == "1"


def _read_policy() -> str:
    p = (os.environ.get(ENV_POLICY) or DEFAULT_POLICY).strip().lower()
    if p not in VALID_POLICIES:
        flexkv_logger.warning(
            f"[FlexKV][NUMA] unknown {ENV_POLICY}={p!r}; falling back to "
            f"{DEFAULT_POLICY!r}"
        )
        p = DEFAULT_POLICY
    return p


def _parse_node_list(raw: Optional[str]) -> List[int]:
    """Parse a comma-separated list like ``"0,3,4,5"`` -> ``[0,3,4,5]``.

    Empty / unparseable inputs return ``[]`` (caller decides how to handle).
    """
    if not raw:
        return []
    out: List[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            flexkv_logger.warning(
                f"[FlexKV][NUMA] ignore non-integer node id: {tok!r}"
            )
    return out


def _read_bind_nodes() -> List[int]:
    return _parse_node_list(os.environ.get(ENV_BIND_NODES))


def _read_preferred_node() -> Optional[int]:
    raw = os.environ.get(ENV_PREFERRED_NODE)
    if raw is None or raw == "":
        return None
    try:
        return int(raw.strip())
    except ValueError:
        flexkv_logger.warning(
            f"[FlexKV][NUMA] invalid {ENV_PREFERRED_NODE}={raw!r}"
        )
        return None


def _all_node_ids(libnuma: _LibNuma) -> List[int]:
    max_node = libnuma.lib.numa_max_node()
    if max_node < 0:
        return []
    return list(range(max_node + 1))


# ---------------------------------------------------------------------------
# Public API: apply_numa_policy
# ---------------------------------------------------------------------------
def apply_numa_policy(role: str = "unknown") -> None:
    """Reset current process NUMA mempolicy according to ``FLEXKV_NUMA_POLICY``.

    Should be called *before* any large host allocation (in particular before
    constructing :class:`flexkv.storage.storage_engine.StorageEngine`).

    The function is best-effort: any failure only logs a warning and returns.
    """
    if _is_disabled():
        flexkv_logger.info(
            f"[FlexKV][NUMA] pid={os.getpid()} role={role} "
            f"{ENV_DISABLE}=1 -> skip"
        )
        return

    policy = _read_policy()
    if policy == POLICY_NONE:
        flexkv_logger.info(
            f"[FlexKV][NUMA] pid={os.getpid()} role={role} policy=none -> skip"
        )
        return

    libnuma = _LibNuma.get()
    if libnuma is None:
        # Already warned inside ``get``.
        return

    try:
        if policy == POLICY_INTERLEAVE:
            mask = libnuma.numa_all_nodes_ptr()
            if not mask:
                # Fallback: build a mask that covers all reachable nodes.
                nodes = _all_node_ids(libnuma)
                owned = libnuma.build_nodemask(nodes)
                if not owned:
                    flexkv_logger.warning(
                        "[FlexKV][NUMA] failed to build all-nodes bitmask"
                    )
                    return
                try:
                    libnuma.lib.numa_set_interleave_mask(owned)
                finally:
                    libnuma.lib.numa_bitmask_free(owned)
                _log_success(role, policy, nodes)
            else:
                libnuma.lib.numa_set_interleave_mask(mask)
                _log_success(role, policy, _all_node_ids(libnuma))

        elif policy == POLICY_FOLLOW_SGLANG:
            nodes = _read_bind_nodes()
            if not nodes:
                # Degrade to interleave-all.
                flexkv_logger.info(
                    f"[FlexKV][NUMA] pid={os.getpid()} role={role} "
                    f"follow_sglang has no {ENV_BIND_NODES}; "
                    f"falling back to interleave-all"
                )
                mask = libnuma.numa_all_nodes_ptr()
                if mask:
                    libnuma.lib.numa_set_interleave_mask(mask)
                _log_success(role, "interleave(fallback)", _all_node_ids(libnuma))
                return
            owned = libnuma.build_nodemask(nodes)
            if not owned:
                flexkv_logger.warning(
                    "[FlexKV][NUMA] failed to build subset bitmask"
                )
                return
            try:
                libnuma.lib.numa_set_interleave_mask(owned)
            finally:
                libnuma.lib.numa_bitmask_free(owned)
            _log_success(role, policy, nodes)

        elif policy == POLICY_PREFERRED:
            node = _read_preferred_node()
            if node is None:
                flexkv_logger.warning(
                    f"[FlexKV][NUMA] preferred policy needs "
                    f"{ENV_PREFERRED_NODE}; falling back to interleave-all"
                )
                mask = libnuma.numa_all_nodes_ptr()
                if mask:
                    libnuma.lib.numa_set_interleave_mask(mask)
                _log_success(role, "interleave(fallback)", _all_node_ids(libnuma))
                return
            libnuma.lib.numa_set_preferred(ctypes.c_int(node))
            _log_success(role, policy, [node])

        elif policy == POLICY_BIND:
            nodes = _read_bind_nodes()
            if not nodes:
                flexkv_logger.warning(
                    f"[FlexKV][NUMA] bind policy needs {ENV_BIND_NODES}; "
                    f"falling back to interleave-all"
                )
                mask = libnuma.numa_all_nodes_ptr()
                if mask:
                    libnuma.lib.numa_set_interleave_mask(mask)
                _log_success(role, "interleave(fallback)", _all_node_ids(libnuma))
                return
            owned = libnuma.build_nodemask(nodes)
            if not owned:
                flexkv_logger.warning(
                    "[FlexKV][NUMA] failed to build bind bitmask"
                )
                return
            try:
                libnuma.lib.numa_set_membind(owned)
            finally:
                libnuma.lib.numa_bitmask_free(owned)
            _log_success(role, policy, nodes)

        else:  # pragma: no cover - guarded by _read_policy
            flexkv_logger.warning(
                f"[FlexKV][NUMA] unhandled policy={policy!r}; skip"
            )

    except Exception as e:  # noqa: BLE001 - best-effort
        flexkv_logger.warning(
            f"[FlexKV][NUMA] apply_numa_policy(role={role}) failed: {e!r}"
        )


def _log_success(role: str, policy: str, nodes: List[int]) -> None:
    flexkv_logger.info(
        f"[FlexKV][NUMA] pid={os.getpid()} role={role} policy={policy} "
        f"nodes={nodes}"
    )


# ---------------------------------------------------------------------------
# Public API: wrap_with_numactl
# ---------------------------------------------------------------------------
def wrap_with_numactl(cmd: List[str]) -> List[str]:
    """Return a new argv list prefixed with ``numactl`` according to env vars.

    If the rollback switch is on, ``numactl`` is missing, or the policy is
    ``none``, the original ``cmd`` is returned unchanged.
    """
    if _is_disabled():
        return list(cmd)

    policy = _read_policy()
    if policy == POLICY_NONE:
        return list(cmd)

    numactl = shutil.which("numactl")
    if numactl is None:
        flexkv_logger.warning(
            "[FlexKV][NUMA] `numactl` binary not found in PATH; "
            "subprocess will rely on in-process apply_numa_policy() only."
        )
        return list(cmd)

    prefix: List[str] = [numactl]

    if policy == POLICY_INTERLEAVE:
        prefix.append("--interleave=all")

    elif policy == POLICY_FOLLOW_SGLANG:
        nodes = _read_bind_nodes()
        if not nodes:
            prefix.append("--interleave=all")
        else:
            joined = ",".join(str(n) for n in nodes)
            prefix.append(f"--cpunodebind={joined}")
            prefix.append(f"--interleave={joined}")

    elif policy == POLICY_PREFERRED:
        node = _read_preferred_node()
        if node is None:
            prefix.append("--interleave=all")
        else:
            prefix.append(f"--preferred={node}")

    elif policy == POLICY_BIND:
        nodes = _read_bind_nodes()
        if not nodes:
            prefix.append("--interleave=all")
        else:
            joined = ",".join(str(n) for n in nodes)
            prefix.append(f"--cpunodebind={joined}")
            prefix.append(f"--membind={joined}")

    else:  # pragma: no cover
        return list(cmd)

    flexkv_logger.info(
        f"[FlexKV][NUMA] wrap_with_numactl: prefix={prefix} for cmd[0]={cmd[0]!r}"
    )
    return prefix + list(cmd)


# ---------------------------------------------------------------------------
# Public API: env propagation list
# ---------------------------------------------------------------------------
NUMA_ENV_KEYS = (
    ENV_POLICY,
    ENV_BIND_NODES,
    ENV_PREFERRED_NODE,
    ENV_DISABLE,
)


def merge_numa_env(env: dict) -> dict:
    """Ensure ``FLEXKV_NUMA_*`` env vars from current process are kept in
    the given child env dict.

    Useful when the caller built a fresh env dict (``inherit_env=False``) and
    wants to propagate the NUMA configuration.
    """
    for k in NUMA_ENV_KEYS:
        if k not in env and k in os.environ:
            env[k] = os.environ[k]
    return env


# ---------------------------------------------------------------------------
# Public API: dump_numa_maps_summary
# ---------------------------------------------------------------------------
def dump_numa_maps_summary(pid: Optional[int] = None, tag: str = "") -> None:
    """Parse ``/proc/<pid>/numa_maps`` and log per-NUMA-node RSS in MiB.

    The format of each line ends with tokens like ``N0=12345 N1=234`` where
    the integer is the *number of pages* (not bytes) on that node. We
    multiply by the system page size to get bytes.
    """
    if _is_disabled():
        return
    if pid is None:
        pid = os.getpid()
    path = f"/proc/{pid}/numa_maps"
    try:
        page_size = os.sysconf("SC_PAGESIZE")
    except (ValueError, OSError):
        page_size = 4096

    per_node_pages: dict = {}
    try:
        with open(path, "r") as f:
            for line in f:
                for tok in line.split():
                    if not tok.startswith("N"):
                        continue
                    eq = tok.find("=")
                    if eq < 0:
                        continue
                    head = tok[1:eq]
                    if not head.isdigit():
                        continue
                    try:
                        node = int(head)
                        pages = int(tok[eq + 1 :])
                    except ValueError:
                        continue
                    per_node_pages[node] = per_node_pages.get(node, 0) + pages
    except FileNotFoundError:
        flexkv_logger.warning(
            f"[FlexKV][NUMA] dump_numa_maps_summary: {path} not found"
        )
        return
    except Exception as e:  # noqa: BLE001
        flexkv_logger.warning(
            f"[FlexKV][NUMA] dump_numa_maps_summary failed: {e!r}"
        )
        return

    if not per_node_pages:
        flexkv_logger.info(
            f"[FlexKV][NUMA] dump_numa_maps_summary pid={pid} tag={tag!r} "
            f"(no NUMA-tagged pages found)"
        )
        return

    parts = []
    for node in sorted(per_node_pages):
        mib = (per_node_pages[node] * page_size) / (1024 * 1024)
        parts.append(f"N{node}={mib:.1f}MiB")
    flexkv_logger.info(
        f"[FlexKV][NUMA] numa_maps_summary pid={pid} tag={tag!r} "
        + " ".join(parts)
    )


__all__ = [
    "apply_numa_policy",
    "wrap_with_numactl",
    "merge_numa_env",
    "dump_numa_maps_summary",
    "NUMA_ENV_KEYS",
    "ENV_POLICY",
    "ENV_BIND_NODES",
    "ENV_PREFERRED_NODE",
    "ENV_DISABLE",
]
