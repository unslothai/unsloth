# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the NUMA auto-interleave decision (core/inference/numa.py).

Models the user's dual-NUMA Xeon (HF screenshots #23): node 0 ~465 GB free, node 1
~223 GB free; a 583 GB GGUF exceeds the largest single node but fits across both, so
`numactl --interleave=all` is the right call. The decision is pure and topology is
injected, so these are deterministic on any host (no /sys, no numactl needed).
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Importing core.inference.numa runs core/inference/__init__.py (orchestrator + structlog
# + loggers + httpx); stub those when absent so a dependency-light run can collect this.
try:
    import structlog  # noqa: F401
except ImportError:
    _s = _types.ModuleType("structlog")
    _s.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
    _s.BoundLogger = type("BoundLogger", (), {})
    sys.modules["structlog"] = _s
try:
    import loggers  # noqa: F401
except ImportError:
    _loggers_stub = _types.ModuleType("loggers")
    _loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
    sys.modules["loggers"] = _loggers_stub
try:
    import httpx  # noqa: F401
except ImportError:
    _httpx_stub = _types.ModuleType("httpx")
    for _exc in (
        "ConnectError",
        "TimeoutException",
        "ReadTimeout",
        "ReadError",
        "RemoteProtocolError",
        "CloseError",
        "HTTPError",
        "RequestError",
    ):
        setattr(_httpx_stub, _exc, type(_exc, (Exception,), {}))
    _httpx_stub.Timeout = type("T", (), {"__init__": lambda s, *a, **k: None})
    _httpx_stub.Response = type("Response", (), {})
    _httpx_stub.Client = type(
        "C",
        (),
        {
            "__init__": lambda s, **kw: None,
            "__enter__": lambda s: s,
            "__exit__": lambda s, *a: None,
        },
    )
    sys.modules["httpx"] = _httpx_stub

from core.inference.numa import (  # noqa: E402
    InterleaveDecision,
    NumaTopology,
    _parse_online,
    decide_interleave,
)

_GiB = 1024**3
_MiB = 1024**2

# The user's box, in MiB free per node (from `numactl --hardware`).
_USER_TOPO = NumaTopology(node_free_mib = {0: 465594, 1: 223814})
_SINGLE = NumaTopology(node_free_mib = {0: 900_000})


def test_parse_online_ranges():
    assert _parse_online("0-1") == [0, 1]
    assert _parse_online("0,2-3") == [0, 2, 3]
    assert _parse_online("3") == [3]
    assert _parse_online("") == []


def test_topology_aggregates():
    assert _USER_TOPO.node_count == 2
    assert _USER_TOPO.largest_node_free_mib == 465594
    assert _USER_TOPO.smallest_node_free_mib == 223814
    assert _USER_TOPO.total_free_mib == 465594 + 223814


def test_interleaves_when_model_exceeds_largest_node_but_fits_across():
    # 583 GB model: > 465 GB (node 0) but < ~689 GB total.
    d = decide_interleave(583 * _GiB, cpu_only = True, topology = _USER_TOPO, has_numactl = True)
    assert d.interleave is True
    assert d.prefix == ("numactl", "--interleave=all")
    assert "interleave=all" in d.reason


def test_no_interleave_when_model_fits_every_node():
    # A 200 GB model fits even the smaller node (223 GB) -> safe on any node, keep local.
    d = decide_interleave(200 * _GiB, cpu_only = True, topology = _USER_TOPO, has_numactl = True)
    assert d.interleave is False
    assert d.prefix == ()
    assert "fits every node" in d.reason


def test_interleaves_when_fits_larger_node_but_not_smaller():
    # 300 GB fits node 0 (465) but not node 1 (223); the loader is not bound, so first-
    # touch could land on node 1. Interleave instead of gambling on placement (PR review).
    d = decide_interleave(300 * _GiB, cpu_only = True, topology = _USER_TOPO, has_numactl = True)
    assert d.interleave is True
    assert d.prefix == ("numactl", "--interleave=all")


def test_no_interleave_on_gpu_host():
    d = decide_interleave(583 * _GiB, cpu_only = False, topology = _USER_TOPO, has_numactl = True)
    assert d.interleave is False
    assert "not cpu-only" in d.reason


def test_no_interleave_single_node():
    d = decide_interleave(583 * _GiB, cpu_only = True, topology = _SINGLE, has_numactl = True)
    assert d.interleave is False
    assert "single NUMA node" in d.reason


def test_model_too_big_for_all_nodes_blocks_with_message():
    # 800 GB > ~689 GB total free across both nodes -> interleave can't help.
    d = decide_interleave(800 * _GiB, cpu_only = True, topology = _USER_TOPO, has_numactl = True)
    assert d.interleave is False
    assert "exceeds total free RAM" in d.reason


def test_numactl_missing_surfaces_actionable_warning():
    d = decide_interleave(583 * _GiB, cpu_only = True, topology = _USER_TOPO, has_numactl = False)
    assert d.interleave is False
    assert "numactl` is not installed" in d.reason


def test_too_big_and_numactl_missing_prefers_total_ram_message():
    # Impossible across all nodes AND no numactl: the total-RAM guidance must win, so the
    # user is not told to install numactl when interleaving could never help (PR review fix).
    d = decide_interleave(800 * _GiB, cpu_only = True, topology = _USER_TOPO, has_numactl = False)
    assert d.interleave is False
    assert "exceeds total free RAM" in d.reason
    assert "numactl` is not installed" not in d.reason


def test_unknown_model_size_does_not_force_interleave():
    for size in (None, 0, -1):
        d = decide_interleave(size, cpu_only = True, topology = _USER_TOPO, has_numactl = True)
        assert d.interleave is False


def test_topology_restricted_to_cpuset_allowed_nodes(tmp_path, monkeypatch):
    """A cpuset that allows only node 0 must drop host node 1 from the topology, so a
    container is not told a node's RAM is usable when the child can't allocate there
    (PR review fix)."""
    import core.inference.numa as m

    (tmp_path / "online").write_text("0-1")
    for nid, free_kb in ((0, 100 * 1024), (1, 200 * 1024)):
        d = tmp_path / f"node{nid}"
        d.mkdir()
        (d / "meminfo").write_text(f"Node {nid} MemFree: {free_kb} kB\n")
    monkeypatch.setattr(m, "_NODE_ROOT", tmp_path)

    monkeypatch.setattr(m, "_mems_allowed", lambda: {0})
    assert m.read_numa_topology().node_free_mib == {0: 100}
    # No cpuset info (None) -> trust the online set, both nodes present.
    monkeypatch.setattr(m, "_mems_allowed", lambda: None)
    assert m.read_numa_topology().node_free_mib == {0: 100, 1: 200}


def test_decision_is_frozen_dataclass():
    d = InterleaveDecision(True, "x", ("numactl", "--interleave=all"))
    try:
        d.interleave = False  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("InterleaveDecision should be immutable")
