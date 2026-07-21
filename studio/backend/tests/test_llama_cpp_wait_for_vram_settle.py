# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``_wait_for_vram_settle`` helper contract.

Pins the bounded poll over ``_get_gpu_free_memory`` bridging the kill -> spawn
VRAM-reclaim window. Patches ``_get_gpu_free_memory``; no real llama-server or
nvidia-smi involved.
"""

from __future__ import annotations

import sys
import time
import types as _types
from pathlib import Path
from unittest.mock import patch

import pytest


# External-dep stubs so this module imports without httpx / structlog / loggers.
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)
# Set get_logger even if a prior test inserted a bare ``structlog`` stub.
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
    (instance or class). Calls past the end repeat the last entry so tests
    can assert "stopped polling" via the call count.
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
    """Default ``since_kill=0.0`` is cold-start: no kill recorded, so the
    helper short-circuits without invoking the probe."""
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
    assert (
        state["calls"] == 0
    ), "kill older than _VRAM_SETTLE_WINDOW_S must skip the wait"


def test_empty_first_sample_returns_immediately():
    """CPU-only host: probe returns [] → no wait, no further polls."""
    ctx, state = _patch_probe([[]])
    with ctx:
        start = time.monotonic()
        LlamaCppBackend._wait_for_vram_settle(**_kw(max_wait = 2.0, interval = 0.25))
        elapsed = time.monotonic() - start
    assert state["calls"] == 1
    assert elapsed < 0.5, "CPU-only short-circuit must not sleep through the interval"


def test_first_probe_raises_returns_without_polling():
    """nvidia-smi gone away at the start: helper returns silently."""
    ctx, state = _patch_probe([OSError("nvidia-smi missing")])
    with ctx:
        LlamaCppBackend._wait_for_vram_settle(**_kw(max_wait = 2.0, interval = 0.25))
    assert state["calls"] == 1


def test_two_consecutive_samples_within_tolerance_settles():
    """Reclaim ramp 10000 → 11500 → 11550: third sample within 256 MiB of
    the second, so the helper returns after exactly three probes."""
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
    # interval * 2 sleeps = 0.10; allow slack for scheduler jitter.
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
    # Must stop near max_wait, not run forever. Generous upper bound for CI.
    assert 0.3 <= elapsed < 2.0, f"helper ignored max_wait: elapsed={elapsed:.3f}s"


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
    # First probe (0.30 s) + at most one clipped sleep + bail.
    # Hard cap well below the old 0.30 + 0.25 + 0.30 = 0.85.
    assert (
        elapsed < 0.85
    ), f"helper exceeded the deadline due to slow probes: {elapsed:.3f}s"


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
            [(0, 10050), (1, 6500)],  # GPU 1 still draining (1500 jump)
            [(0, 10080), (1, 6520)],  # GPU 1 settles (20 delta)
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
            [(0, 80700)],  # 700 MiB delta < 2% of 80000 = 1600 MiB
        ]
    )
    with ctx:
        LlamaCppBackend._wait_for_vram_settle(**_kw(max_wait = 2.0, interval = 0.05))
    assert state["calls"] == 2


def test_load_model_calls_helper_outside_lock_and_uses_last_kill_timestamp():
    """Pin the call site: outside Phase 3 lock, gated on the timestamp, no
    ``had_live_process`` in-band flag regression."""
    import inspect

    src = inspect.getsource(LlamaCppBackend.load_model)
    assert "_wait_for_vram_settle" in src
    assert "since_kill" in src
    assert "self._last_kill_monotonic" in src
    # Must run before Phase 3's broad lock so /unload, /cancel, /status
    # are not blocked during the wait.
    assert src.index("_wait_for_vram_settle") < src.index("# ── Phase 3:")
    # An in-band ``had_live_process`` flag would regress the frontend
    # /unload+/load Apply path; use the timestamp instead.
    assert "had_live_process" not in src


def test_kill_process_records_timestamp_on_actual_kill():
    """Cold-call no-op leaves the sentinel; real kill stamps monotonic."""
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = None
    backend._healthy = False
    backend._stats_logger = None  # _kill_process stops it in finally
    backend._stdout_thread = None
    backend._llama_log_fh = None
    backend._last_kill_monotonic = 0.0

    backend._kill_process()
    assert backend._last_kill_monotonic == 0.0

    class _FakeProcess:
        def terminate(self):
            pass

        def wait(self, timeout = None):
            return 0

        def kill(self):
            pass

        def poll(self):
            return 0

    backend._process = _FakeProcess()
    before = time.monotonic()
    backend._kill_process()
    after = time.monotonic()
    assert before <= backend._last_kill_monotonic <= after


def test_kill_process_tolerates_partially_constructed_backend():
    # Teardown must not AttributeError on a __new__-built backend that never ran
    # __init__: _stats_logger / _stdout_thread / _llama_log_fh are left unset.
    backend = LlamaCppBackend.__new__(LlamaCppBackend)

    class _FakeProcess:
        def terminate(self):
            pass

        def wait(self, timeout = None):
            return 0

        def kill(self):
            pass

    backend._process = _FakeProcess()
    backend._kill_process()
    assert backend._process is None


def test_helper_is_static_method_callable_off_class():
    """Pin the @staticmethod binding so call sites can invoke off the class."""
    ctx, _state = _patch_probe([[]])
    with ctx:
        LlamaCppBackend._wait_for_vram_settle(
            **_kw(max_wait = 0.1, interval = 0.05),
        )


# ---------------------------------------------------------------------------
# Startup orphan-reaper arms the settle clock (the "wrong card after restart"
# root cause: reaped VRAM frees lazily, so the first load must wait).
# ---------------------------------------------------------------------------


def test_kill_orphaned_servers_returns_count():
    """The reaper reports how many owned orphans it killed, so __init__ can
    arm the settle wait. Only Unsloth-owned llama-server procs count."""
    import os

    mypid = os.getpid()
    fake_path = "/tmp/unsloth-test-llama/llama-server"
    killed: list[int] = []

    class _FakeProc:
        def __init__(self, pid, name, exe):
            self.info = {"pid": pid, "name": name, "exe": exe}

        def kill(self):
            killed.append(self.info["pid"])

    owned = _FakeProc(mypid + 1, "llama-server", fake_path)  # exact-path match
    foreign = _FakeProc(mypid + 2, "llama-server", "/usr/bin/llama-server")
    unrelated = _FakeProc(mypid + 3, "python3", "/usr/bin/python3")

    fake_psutil = _types.ModuleType("psutil")
    fake_psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
    fake_psutil.AccessDenied = type("AccessDenied", (Exception,), {})
    fake_psutil.ZombieProcess = type("ZombieProcess", (Exception,), {})
    fake_psutil.process_iter = lambda attrs = None: [owned, foreign, unrelated]

    with (
        patch.dict(sys.modules, {"psutil": fake_psutil}),
        patch.dict(os.environ, {"LLAMA_SERVER_PATH": fake_path}),
        patch.object(
            LlamaCppBackend, "_pid_parent_is_alive", staticmethod(lambda pid: False)
        ),
    ):
        n = LlamaCppBackend._kill_orphaned_servers()
    assert n == 1, "only the Unsloth-owned orphan should be counted"
    assert killed == [mypid + 1]

    # No owned orphans -> zero, so __init__ leaves the cold-start sentinel.
    fake_psutil.process_iter = lambda attrs = None: [foreign, unrelated]
    killed.clear()
    with (
        patch.dict(sys.modules, {"psutil": fake_psutil}),
        patch.dict(os.environ, {"LLAMA_SERVER_PATH": fake_path}),
        patch.object(
            LlamaCppBackend, "_pid_parent_is_alive", staticmethod(lambda pid: False)
        ),
    ):
        assert LlamaCppBackend._kill_orphaned_servers() == 0
    assert killed == []


def test_kill_orphaned_servers_spares_live_parent():
    """An Unsloth-owned llama-server whose parent is still running is not an
    orphan (a live Unsloth or the user's shell owns it) and must never be
    killed; only the true orphan (parent gone) is reaped."""
    import os

    mypid = os.getpid()
    fake_path = "/tmp/unsloth-test-llama/llama-server"
    killed: list[int] = []

    class _FakeProc:
        def __init__(self, pid, name, exe):
            self.info = {"pid": pid, "name": name, "exe": exe}

        def kill(self):
            killed.append(self.info["pid"])

    live_parent = _FakeProc(mypid + 1, "llama-server", fake_path)
    true_orphan = _FakeProc(mypid + 2, "llama-server", fake_path)

    fake_psutil = _types.ModuleType("psutil")
    fake_psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
    fake_psutil.AccessDenied = type("AccessDenied", (Exception,), {})
    fake_psutil.ZombieProcess = type("ZombieProcess", (Exception,), {})
    fake_psutil.process_iter = lambda attrs = None: [live_parent, true_orphan]

    with (
        patch.dict(sys.modules, {"psutil": fake_psutil}),
        patch.dict(os.environ, {"LLAMA_SERVER_PATH": fake_path}),
        patch.object(LlamaCppBackend, "_reap_recorded_pid", staticmethod(lambda: 0)),
        patch.object(
            LlamaCppBackend,
            "_pid_parent_is_alive",
            staticmethod(lambda pid: pid == mypid + 1),
        ),
    ):
        n = LlamaCppBackend._kill_orphaned_servers()
    assert n == 1, "only the true orphan should be reaped"
    assert killed == [mypid + 2], "the live-parent server must be spared"


def test_startup_reaper_arms_settle_timestamp():
    """__init__ arms ``_last_kill_monotonic`` when the startup reaper kills an
    orphan (so the first load_model waits for VRAM to settle), and leaves the
    0.0 cold-start sentinel when nothing was reaped."""
    with patch.object(
        LlamaCppBackend, "_kill_orphaned_servers", staticmethod(lambda: 1)
    ):
        before = time.monotonic()
        backend = LlamaCppBackend()
        after = time.monotonic()
    assert (
        before <= backend._last_kill_monotonic <= after
    ), "a positive reap count must arm the settle clock"

    with patch.object(
        LlamaCppBackend, "_kill_orphaned_servers", staticmethod(lambda: 0)
    ):
        backend_cold = LlamaCppBackend()
    assert (
        backend_cold._last_kill_monotonic == 0.0
    ), "no reap must leave the cold-start sentinel so the wait is skipped"


# ---------------------------------------------------------------------------
# Cross-session backstop: a server PID recorded at spawn is reaped on the next
# startup even when parent-death cleanup did not run (macOS, a best-effort
# PR_SET_PDEATHSIG / Job Object failure, or a pre-existing orphan), but ONLY when
# it is a true orphan (its parent is gone), it still is a llama-server, and its
# start-time identity matches. A live server (parent still running) is spared so a
# helper backend built in-process can never kill the active chat server.
# ---------------------------------------------------------------------------


class _FakeKillProc:
    def terminate(self):
        pass

    def wait(self, timeout = None):
        return 0

    def kill(self):
        pass

    def poll(self):
        return 0


def test_kill_process_clears_pidfile(tmp_path):
    """A real kill removes the recorded pidfile so a clean eject leaves no orphan marker."""
    pidfile = tmp_path / "llama-server.pid"
    pidfile.write_text("12345")
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = _FakeKillProc()
    backend._healthy = False
    backend._stdout_thread = None
    backend._llama_log_fh = None
    backend._last_kill_monotonic = 0.0
    backend._stats_logger = None
    with patch.object(
        LlamaCppBackend, "_server_pidfile_path", staticmethod(lambda: pidfile)
    ):
        backend._kill_process()
    assert not pidfile.exists()


def test_reap_recorded_pid_kills_recorded_server(tmp_path):
    """An orphaned recorded PID (parent gone) is killed and the pidfile cleared
    when it is still a llama-server."""
    import subprocess

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    pidfile = tmp_path / "llama-server.pid"
    pidfile.write_text(str(proc.pid))
    try:
        with (
            patch.object(
                LlamaCppBackend, "_server_pidfile_path", staticmethod(lambda: pidfile)
            ),
            patch.object(
                LlamaCppBackend, "_pid_parent_is_alive", staticmethod(lambda pid: False)
            ),
            patch.object(
                LlamaCppBackend,
                "_pid_is_llama_server",
                staticmethod(lambda pid: pid == proc.pid),
            ),
        ):
            n = LlamaCppBackend._reap_recorded_pid()
        assert n == 1
        assert not pidfile.exists()
        proc.wait(timeout = 5)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout = 5)


def test_record_then_reap_round_trip_identity_matches(tmp_path):
    """Full round trip: _record_server_pid writes pid:starttime, and an orphaned
    reap whose recorded identity still matches DOES kill it."""
    import subprocess

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    pidfile = tmp_path / "llama-server.pid"
    try:
        with patch.object(
            LlamaCppBackend, "_server_pidfile_path", staticmethod(lambda: pidfile)
        ):
            LlamaCppBackend._record_server_pid(proc.pid)
        assert ":" in pidfile.read_text(), "a start-time identity must be recorded"
        with (
            patch.object(
                LlamaCppBackend, "_server_pidfile_path", staticmethod(lambda: pidfile)
            ),
            patch.object(
                LlamaCppBackend, "_pid_parent_is_alive", staticmethod(lambda pid: False)
            ),
            patch.object(
                LlamaCppBackend, "_pid_is_llama_server", staticmethod(lambda pid: True)
            ),
        ):
            n = LlamaCppBackend._reap_recorded_pid()
        assert n == 1, "a matching identity on a true orphan must be reaped"
        proc.wait(timeout = 5)
        assert proc.poll() is not None
        assert not pidfile.exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout = 5)


def test_reap_recorded_pid_spares_live_server(tmp_path):
    """A recorded server whose parent is still alive (the running Unsloth) is NEVER
    reaped, and its pidfile is kept. This is the finding-3 guard: a helper backend
    constructed in-process must not kill the active chat server. Uses the REAL
    _pid_parent_is_alive (the child's parent is this live test process)."""
    import subprocess

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    pidfile = tmp_path / "llama-server.pid"
    pidfile.write_text(str(proc.pid))
    try:
        with (
            patch.object(
                LlamaCppBackend, "_server_pidfile_path", staticmethod(lambda: pidfile)
            ),
            # Force the name check True so ONLY the parent-alive guard can spare it.
            patch.object(
                LlamaCppBackend, "_pid_is_llama_server", staticmethod(lambda pid: True)
            ),
        ):
            n = LlamaCppBackend._reap_recorded_pid()
        assert n == 0, "a live server with a running parent must not be reaped"
        assert proc.poll() is None, "the live server must still be running"
        assert pidfile.exists(), "the record is kept so a later orphan reap still works"
    finally:
        proc.kill()
        proc.wait(timeout = 5)


def test_reap_recorded_pid_skips_pid_reuse(tmp_path):
    """A recorded PID recycled to a non-llama-server must NOT be killed (only the
    stale pidfile is cleaned), so the user's vllm/games are never touched."""
    import subprocess

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    pidfile = tmp_path / "llama-server.pid"
    pidfile.write_text(str(proc.pid))
    try:
        with (
            patch.object(
                LlamaCppBackend, "_server_pidfile_path", staticmethod(lambda: pidfile)
            ),
            patch.object(
                LlamaCppBackend, "_pid_parent_is_alive", staticmethod(lambda pid: False)
            ),
            patch.object(
                LlamaCppBackend, "_pid_is_llama_server", staticmethod(lambda pid: False)
            ),
        ):
            n = LlamaCppBackend._reap_recorded_pid()
        assert n == 0
        assert proc.poll() is None, "an unrelated reused PID must not be killed"
        assert not pidfile.exists(), "stale pidfile is cleaned up"
    finally:
        proc.kill()
        proc.wait(timeout = 5)


def test_reap_recorded_pid_skips_identity_mismatch(tmp_path):
    """An orphaned PID whose recorded start-time identity no longer matches has been
    recycled; it must NOT be killed even if it now looks like a llama-server."""
    import subprocess

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    pidfile = tmp_path / "llama-server.pid"
    pidfile.write_text(f"{proc.pid}:0.0")  # stale identity that cannot match
    try:
        with (
            patch.object(
                LlamaCppBackend, "_server_pidfile_path", staticmethod(lambda: pidfile)
            ),
            patch.object(
                LlamaCppBackend, "_pid_parent_is_alive", staticmethod(lambda pid: False)
            ),
            patch.object(
                LlamaCppBackend, "_pid_is_llama_server", staticmethod(lambda pid: True)
            ),
        ):
            n = LlamaCppBackend._reap_recorded_pid()
        assert n == 0, "a PID whose start-time identity changed must not be killed"
        assert proc.poll() is None, "the recycled process must survive"
        assert not pidfile.exists(), "stale pidfile is cleaned up"
    finally:
        proc.kill()
        proc.wait(timeout = 5)


def test_reap_recorded_pid_windows_sigkill_fallback(tmp_path, monkeypatch):
    """On Windows signal.SIGKILL is undefined; the reaper must fall back to SIGTERM
    (os.kill -> TerminateProcess) instead of crashing and leaving the orphan."""
    import os as _os
    import signal as _signal

    monkeypatch.delattr(_signal, "SIGKILL", raising = False)
    captured = {}

    def _fake_kill(pid, sig):
        captured["pid"] = pid
        captured["sig"] = sig  # recorded; do not actually signal anything

    pidfile = tmp_path / "llama-server.pid"
    pidfile.write_text("424242")
    with (
        patch.object(
            LlamaCppBackend, "_server_pidfile_path", staticmethod(lambda: pidfile)
        ),
        patch.object(
            LlamaCppBackend, "_pid_parent_is_alive", staticmethod(lambda pid: False)
        ),
        patch.object(
            LlamaCppBackend, "_pid_is_llama_server", staticmethod(lambda pid: True)
        ),
        patch.object(_os, "kill", _fake_kill),
    ):
        n = LlamaCppBackend._reap_recorded_pid()
    assert n == 1
    assert (
        captured.get("sig") == _signal.SIGTERM
    ), "must fall back to SIGTERM when SIGKILL is absent"
    assert not pidfile.exists()


def test_reap_recorded_pid_no_pidfile(tmp_path):
    """No pidfile -> nothing reaped, no error."""
    pidfile = tmp_path / "llama-server.pid"  # never created
    with patch.object(
        LlamaCppBackend, "_server_pidfile_path", staticmethod(lambda: pidfile)
    ):
        assert LlamaCppBackend._reap_recorded_pid() == 0
