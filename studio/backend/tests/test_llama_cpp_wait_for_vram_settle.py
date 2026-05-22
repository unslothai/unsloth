# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``_wait_for_vram_settle`` helper contract.

Pins the bounded poll over ``_get_gpu_free_memory`` that bridges the
kill -> spawn VRAM-reclaim window. Patches ``_get_gpu_free_memory``;
no real llama-server or nvidia-smi involved.
"""

from __future__ import annotations

import sys
import time
import types as _types
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Same external-dep stubs as the other llama_cpp tests so this module
# imports cleanly without httpx / structlog / loggers installed.
# ---------------------------------------------------------------------------
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)
# Ensure get_logger is set even if a previous test module already
# inserted a bare ``structlog`` stub via ``setdefault``.
if not hasattr(sys.modules["structlog"], "get_logger"):
    sys.modules["structlog"].get_logger = _structlog_stub.get_logger

_httpx_stub = _types.ModuleType("httpx")
for _exc in (
    "ConnectError",
    "TimeoutException",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
    "CloseError",
    "WriteError",
):
    setattr(_httpx_stub, _exc, type(_exc, (Exception,), {}))
_httpx_stub.Timeout = type("T", (), {"__init__": lambda s, *a, **k: None})
_httpx_stub.Client = type(
    "C",
    (),
    {
        "__init__": lambda s, **kw: None,
        "__enter__": lambda s: s,
        "__exit__": lambda s, *a: None,
    },
)
sys.modules.setdefault("httpx", _httpx_stub)

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_probe(samples):
    """Patch ``_get_gpu_free_memory`` to yield ``samples`` in order.

    Each entry is a list[(idx, free_mib)], a callable, or an exception
    (instance or class). Calls past the end repeat the last entry so
    tests can assert "stopped polling" via the call count.
    """
    state = {"i": 0, "calls": 0}

    def _side_effect():
        state["calls"] += 1
        idx = min(state["i"], len(samples) - 1)
        state["i"] += 1
        item = samples[idx]
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, type) and issubclass(item, BaseException):
            raise item()
        if callable(item):
            return item()
        return item

    return patch.object(
        LlamaCppBackend,
        "_get_gpu_free_memory",
        staticmethod(_side_effect),
    ), state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _kw(**extra):
    """Helper kwargs that engage the wait path (``since_kill=now()``)."""
    base = {"since_kill": time.monotonic()}
    base.update(extra)
    return base


def test_cold_start_returns_immediately_without_probing():
    """Default ``since_kill=0.0`` is cold-start: no kill recorded,
    helper short-circuits without ever invoking the probe."""
    ctx, state = _patch_probe([[(0, 10000)], [(0, 10000)]])
    with ctx:
        start = time.monotonic()
        LlamaCppBackend._wait_for_vram_settle(max_wait = 2.0, interval = 0.25)
        elapsed = time.monotonic() - start
    assert state["calls"] == 0, "cold start must skip the probe entirely"
    assert elapsed < 0.05


def test_stale_kill_skips_wait():
    """Kill older than the settle window (~15 s default): no wait."""
    ctx, state = _patch_probe([[(0, 10000)]])
    long_ago = time.monotonic() - 60.0
    with ctx:
        LlamaCppBackend._wait_for_vram_settle(
            **_kw(since_kill = long_ago, max_wait = 2.0, interval = 0.25)
        )
    assert state["calls"] == 0, (
        "kill older than _VRAM_SETTLE_WINDOW_S must skip the wait"
    )


def test_empty_first_sample_returns_immediately():
    """CPU-only host: probe returns [] → no wait, no further polls."""
    ctx, state = _patch_probe([[]])
    with ctx:
        start = time.monotonic()
        LlamaCppBackend._wait_for_vram_settle(**_kw(max_wait = 2.0, interval = 0.25))
        elapsed = time.monotonic() - start
    assert state["calls"] == 1
    assert elapsed < 0.5, (
        "CPU-only short-circuit must not sleep through the interval"
    )


def test_first_probe_raises_returns_without_polling():
    """nvidia-smi gone away at the start: helper returns silently."""
    ctx, state = _patch_probe([OSError("nvidia-smi missing")])
    with ctx:
        LlamaCppBackend._wait_for_vram_settle(**_kw(max_wait = 2.0, interval = 0.25))
    assert state["calls"] == 1


def test_two_consecutive_samples_within_tolerance_settles():
    """The reclaim ramp from 10000 → 11500 → 11550: third sample within
    256 MiB of the second so the helper returns after exactly three probes."""
    ctx, state = _patch_probe(
        [
            [(0, 10000)],
            [(0, 11500)],
            [(0, 11550)],
        ]
    )
    with ctx:
        start = time.monotonic()
        LlamaCppBackend._wait_for_vram_settle(**_kw(max_wait = 2.0, interval = 0.05))
        elapsed = time.monotonic() - start
    assert state["calls"] == 3
    # interval * 2 sleeps = 0.10; allow generous slack for scheduler jitter.
    assert elapsed < 1.0


def test_probe_raises_mid_loop_returns():
    """Probe disappears between polls: helper bails without infinite-looping."""
    ctx, state = _patch_probe(
        [
            [(0, 10000)],
            OSError("nvidia-smi crashed"),
        ]
    )
    with ctx:
        LlamaCppBackend._wait_for_vram_settle(**_kw(max_wait = 2.0, interval = 0.05))
    assert state["calls"] == 2


def test_max_wait_respected_when_never_settles():
    """Probe always drifts: helper returns within ``max_wait`` regardless."""

    drift = {"v": 10000}

    def _drifty():
        drift["v"] += 500
        return [(0, drift["v"])]

    ctx, _state = _patch_probe([_drifty])
    with ctx:
        start = time.monotonic()
        LlamaCppBackend._wait_for_vram_settle(**_kw(max_wait = 0.5, interval = 0.1))
        elapsed = time.monotonic() - start
    # We must stop near max_wait, not run forever. Generous upper bound for CI.
    assert 0.3 <= elapsed < 2.0, (
        f"helper ignored max_wait: elapsed={elapsed:.3f}s"
    )


def test_max_wait_respected_when_probe_is_slow():
    """Slow probe: clipped sleep keeps the wall-clock bound honest."""

    def _slow_probe():
        time.sleep(0.30)
        return [(0, 10000)]

    ctx, _state = _patch_probe([_slow_probe])
    with ctx:
        start = time.monotonic()
        LlamaCppBackend._wait_for_vram_settle(
            **_kw(max_wait = 0.4, interval = 0.25),
        )
        elapsed = time.monotonic() - start
    # First probe (0.30 s) + at most one short clipped sleep + bail.
    # Hard cap well below the old behaviour of 0.30 + 0.25 + 0.30 = 0.85.
    assert elapsed < 0.85, (
        f"helper exceeded the deadline due to slow probes: {elapsed:.3f}s"
    )


def test_gpu_index_set_change_returns():
    """Driver re-enumeration mid-wait: helper stops and lets the caller
    re-probe in the main GPU-selection block."""
    ctx, state = _patch_probe(
        [
            [(0, 10000), (1, 8000)],
            [(0, 11000)],
        ]
    )
    with ctx:
        LlamaCppBackend._wait_for_vram_settle(**_kw(max_wait = 2.0, interval = 0.05))
    assert state["calls"] == 2


def test_per_gpu_stability_one_still_draining():
    """Per-GPU stability: returns only once every card is within tol."""
    ctx, state = _patch_probe(
        [
            [(0, 10000), (1, 5000)],
            [(0, 10050), (1, 6500)],   # GPU 1 still draining (1500 jump)
            [(0, 10080), (1, 6520)],   # GPU 1 settles (20 delta)
        ]
    )
    with ctx:
        LlamaCppBackend._wait_for_vram_settle(**_kw(max_wait = 2.0, interval = 0.05))
    assert state["calls"] == 3


def test_tolerance_two_percent_for_large_cards():
    """80 GB card with sub-1 % noise: adaptive 2 % tol settles fast."""
    ctx, state = _patch_probe(
        [
            [(0, 80000)],
            [(0, 80700)],   # 700 MiB delta < 2% of 80000 = 1600 MiB
        ]
    )
    with ctx:
        LlamaCppBackend._wait_for_vram_settle(**_kw(max_wait = 2.0, interval = 0.05))
    assert state["calls"] == 2


def test_load_model_calls_helper_outside_lock_and_uses_last_kill_timestamp():
    """Pin the call site: outside Phase 3 lock, gated on the timestamp,
    no ``had_live_process`` in-band flag regression. Mirrors the
    ``inspect.getsource`` pattern from ``test_llama_cpp_no_context_shift``.
    """
    import inspect
    src = inspect.getsource(LlamaCppBackend.load_model)
    assert "_wait_for_vram_settle" in src
    assert "since_kill" in src
    assert "self._last_kill_monotonic" in src
    # Must be invoked before Phase 3's broad lock so /unload, /cancel,
    # /status are not blocked during the wait.
    assert src.index("_wait_for_vram_settle") < src.index("# ── Phase 3:")
    # An in-band ``had_live_process`` flag would silently regress the
    # frontend /unload+/load Apply path; use the timestamp instead.
    assert "had_live_process" not in src


def test_kill_process_records_timestamp_on_actual_kill():
    """Cold-call no-op leaves the sentinel; real kill stamps monotonic."""
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = None
    backend._healthy = False
    backend._stdout_thread = None
    backend._llama_log_fh = None
    backend._last_kill_monotonic = 0.0

    backend._kill_process()
    assert backend._last_kill_monotonic == 0.0

    class _FakeProcess:
        def terminate(self): pass
        def wait(self, timeout = None): return 0
        def kill(self): pass
        def poll(self): return 0

    backend._process = _FakeProcess()
    before = time.monotonic()
    backend._kill_process()
    after = time.monotonic()
    assert before <= backend._last_kill_monotonic <= after


def test_helper_is_static_method_callable_off_class():
    """Pin the @staticmethod binding so call sites can invoke off the class."""
    ctx, _state = _patch_probe([[]])
    with ctx:
        LlamaCppBackend._wait_for_vram_settle(
            **_kw(max_wait = 0.1, interval = 0.05),
        )
