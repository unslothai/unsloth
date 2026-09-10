# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for LlamaCppBackend._wait_for_health resilience.

The probe loop must swallow transient httpx errors and fall through to the
subprocess.poll() branch so a crashed llama-server surfaces a structured
"exited with code X" log instead of bubbling an opaque exception up to the
/api/inference/load route.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import types as _types
from pathlib import Path
from unittest import mock

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Mirror sibling tests' stubbing so the module imports without fastapi.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
sys.modules.setdefault("structlog", _types.ModuleType("structlog"))

import httpx  # noqa: E402

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402
from core.inference.llama_cpp import GgufLoadIntent

# Sibling tests install lightweight httpx stubs, so when collected together our `httpx`
# may be a stub lacking `get`. Fill in the gaps so collection order does not matter.
if not hasattr(httpx, "get"):
    httpx.get = None  # placeholder; every test below monkeypatches it
for _exc_name in (
    "ConnectError",
    "TimeoutException",
    "ReadError",
    "RemoteProtocolError",
    "WriteError",
):
    if not hasattr(httpx, _exc_name):
        setattr(httpx, _exc_name, type(_exc_name, (Exception,), {}))


def _make_backend(port: int = 12345) -> LlamaCppBackend:
    """Barebones LlamaCppBackend with only the attributes _wait_for_health touches (bypasses __init__)."""
    b = LlamaCppBackend.__new__(LlamaCppBackend)
    b._port = port
    b._stdout_thread = None
    b._stdout_lines = []
    b._process = mock.Mock()
    return b


class TestWaitForHealthResilience:
    def test_returns_true_on_first_200(self, monkeypatch):
        b = _make_backend()
        b._process.poll.return_value = None
        ok_resp = mock.Mock(status_code = 200)
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: ok_resp)
        assert b._wait_for_health(timeout = 1.0, interval = 0.01) is True

    def test_timeout_records_marker_for_classification(self, monkeypatch):
        """A live-but-never-healthy server leaves a marker so the failure is
        classified as a /health timeout, not a bad GGUF (#5740)."""
        b = _make_backend()
        b._process.poll.return_value = None
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock.Mock(status_code = 503))
        assert b._wait_for_health(timeout = 0.02, interval = 0.01) is False
        assert any("health check timed out" in ln for ln in b._stdout_lines)

    def test_a_teardown_during_the_wait_is_a_failed_load_not_a_crash(self, monkeypatch):
        """Shutdown clears the reference mid-wait; reading it once and treating
        its absence as "gone" is what stops the AttributeError the user saw as
        "Error loading model: 'NoneType' object has no attribute 'poll'" (#10353)."""
        b = _make_backend()
        b._process.poll.return_value = None
        probes = []

        def probe(*a, **kw):
            probes.append(1)
            if len(probes) == 2:
                b._process = None  # what _kill_process does on shutdown
            return mock.Mock(status_code = 503)

        monkeypatch.setattr(httpx, "get", probe)
        assert b._wait_for_health(timeout = 30.0, interval = 0.01) is False
        assert len(probes) == 2
        assert not any("health check timed out" in ln for ln in b._stdout_lines)
        assert b._health_wait_cancelled is True

    def test_a_teardown_before_the_first_probe_is_still_not_a_crash(self, monkeypatch):
        b = _make_backend()
        b._process = None
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock.Mock(status_code = 200))
        assert b._wait_for_health(timeout = 30.0, interval = 0.01) is False
        assert b._health_wait_cancelled is True

    def test_a_teardown_is_terminal_so_the_caller_reads_no_exit_code(self, monkeypatch):
        """False alone is not enough: _spawn_and_wait stops only on this flag, else
        it reads the cleared reference and raises one frame up, and respawns."""
        b = _make_backend()
        b._process = None
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock.Mock(status_code = 503))
        assert b._wait_for_health(timeout = 30.0, interval = 0.01) is False
        assert b._health_wait_cancelled is True
        # Exactly what the caller does at the `if not healthy` branch.
        assert (
            b._process is not None and b._process.poll() is not None and b._process.returncode != 0
        ) is False

    def test_a_crash_leaves_the_wait_unmarked_so_the_retries_still_run(self, monkeypatch):
        """The other side: a real crash must NOT look terminal, or the --fit off
        and CPU fallbacks stop recovering loads that used to recover."""
        b = _make_backend()
        b._process.poll.return_value = 1
        b._process.returncode = 1
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock.Mock(status_code = 503))
        assert b._wait_for_health(timeout = 1.0, interval = 0.01) is False
        assert b._health_wait_cancelled is False

    def test_a_teardown_marker_does_not_leak_into_the_next_load(self, monkeypatch):
        b = _make_backend()
        b._process = None
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock.Mock(status_code = 200))
        assert b._wait_for_health(timeout = 30.0, interval = 0.01) is False
        assert b._health_wait_cancelled is True
        b._process = mock.Mock()
        b._process.poll.return_value = None
        assert b._wait_for_health(timeout = 30.0, interval = 0.01) is True
        assert b._health_wait_cancelled is False

    def test_a_teardown_from_a_real_thread_is_not_a_crash(self, monkeypatch):
        """Two real threads, not a probe callback: the once-per-iteration read is
        what makes it safe, so drive it that way."""
        b = _make_backend()
        b._process.poll.return_value = None
        in_probe = threading.Event()
        cleared = threading.Event()

        def probe(*a, **kw):
            in_probe.set()
            cleared.wait(2.0)
            return mock.Mock(status_code = 503)

        monkeypatch.setattr(httpx, "get", probe)

        def shutdown():
            in_probe.wait(2.0)
            b._process = None  # _kill_process, from run.py's shutdown path
            cleared.set()

        t = threading.Thread(target = shutdown)
        t.start()
        try:
            assert b._wait_for_health(timeout = 30.0, interval = 0.01) is False
        finally:
            t.join(5.0)
        assert b._health_wait_cancelled is True

    def test_a_signalled_exit_during_teardown_is_not_a_startup_crash(self, monkeypatch):
        """_kill_process holds the reference across its terminate/wait, so for up to
        ~10s the loop sees a populated process whose poll() is the SIGTERM code. The
        None check alone only covers teardown AFTER cleanup finished, so the kill
        publishes its terminal state before signalling; otherwise -15 reads as a
        startup crash and the retry ladder respawns a server during shutdown."""
        b = _make_backend()
        b._process.poll.return_value = None

        def probe(*a, **kw):
            # The shutdown thread mid-wait: publish, then signal, reference still
            # set. Here rather than before the call because the wait resets on
            # entry, so only a teardown landing DURING a wait is this case.
            b._torn_down_process = b._process
            b._process.poll.return_value = -15
            b._process.returncode = -15
            return mock.Mock(status_code = 503)

        monkeypatch.setattr(httpx, "get", probe)
        with mock.patch("core.inference.llama_cpp.logger") as log:
            assert b._wait_for_health(timeout = 1.0, interval = 0.01) is False
        assert b._health_wait_cancelled is True
        assert not any("exited with code" in str(c) for c in log.error.call_args_list)

    def test_a_teardown_between_the_spawn_and_the_wait_still_counts(self, monkeypatch):
        """Shutdown can land after Popen publishes the child but before the wait
        starts. A marker reset on wait entry would erase it, and the loop would
        then read the signalled exit as a startup crash and let the fallbacks
        respawn during shutdown, which is the race this whole change is about.
        Keyed to the process, so it survives until the wait that owns it reads it."""
        b = _make_backend()
        b._torn_down_process = b._process  # teardown, before the wait is entered
        b._process.poll.return_value = -15
        b._process.returncode = -15
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock.Mock(status_code = 503))
        with mock.patch("core.inference.llama_cpp.logger") as log:
            assert b._wait_for_health(timeout = 1.0, interval = 0.01) is False
        assert b._health_wait_cancelled is True
        assert not any("exited with code" in str(c) for c in log.error.call_args_list)

    def test_shutdown_is_terminal_however_late_it_lands(self, monkeypatch):
        """The durable half of the guard. Every per-process snapshot has an instant
        after it where teardown can still land, so this flag only goes false to
        true and is re-read each iteration; a wait in flight when shutdown begins
        ends terminally no matter where in the loop the flag was set."""
        b = _make_backend()
        b._process.poll.return_value = None

        def probe(*a, **kw):
            b._shutting_down = True  # run.py's shutdown, at an arbitrary instant
            return mock.Mock(status_code = 503)

        monkeypatch.setattr(httpx, "get", probe)
        assert b._wait_for_health(timeout = 30.0, interval = 0.01) is False
        assert b._health_wait_cancelled is True
        assert not any("health check timed out" in ln for ln in b._stdout_lines)

    def test_kill_process_marks_shutdown_only_for_a_teardown(self):
        """The flag must not be set by the retry ladder reaping its own child, or
        the first crash retry would look like a shutdown and abort the load."""
        for teardown in (True, False):
            b = _make_backend()
            b._stop_mtp_crash_watchdog = lambda *a, **kw: None
            b._reset_effective_parallel_slots = lambda *a, **kw: None
            b._leading_process_group = lambda *a, **kw: None
            b._collect_descendants = lambda *a, **kw: []
            b._kill_process_group = lambda *a, **kw: None
            b._terminate_descendants = lambda *a, **kw: None
            b._process.poll.return_value = 0
            b._kill_process(teardown = teardown)
            assert getattr(b, "_shutting_down", False) is teardown

    def test_a_started_shutdown_refuses_to_spawn(self, monkeypatch):
        """_start_llama_process is the chokepoint the mmproj text-only retry uses
        without passing _spawn_and_wait's boundary check, so it refuses too, and
        says so: a silent refusal leaves the caller health-waiting on the previous
        child and then reading a reference the teardown is clearing."""
        import subprocess

        b = _make_backend()
        b._shutting_down = True
        spawned = []
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: spawned.append(1))
        assert b._start_llama_process(["llama-server"], {}, child_gpu_physical_ids = None) is False
        assert spawned == [], "started a server after shutdown had begun"

    def test_a_teardown_with_no_process_still_marks_shutdown(self):
        """Quitting during a download or staging has nothing to kill, so
        _kill_process returns early. Marking teardown only on the path that has a
        process let the load spawn a server after the shutdown sweep finished."""
        b = _make_backend()
        b._process = None
        b._stop_mtp_crash_watchdog = lambda *a, **kw: None
        b._reset_effective_parallel_slots = lambda *a, **kw: None
        b._kill_process(teardown = True)
        assert b._shutting_down is True

    def test_the_published_child_survives_post_spawn_setup(self, monkeypatch):
        """Shutdown can take the spawn lock the instant it is released and clear
        the reference, so post-spawn setup reads the Popen it just made rather than
        self._process. Otherwise recording the pid raises the same AttributeError
        this PR exists to remove."""
        import subprocess

        b = _make_backend()
        b._shutting_down = False
        recorded = []
        b._record_server_pid = lambda pid: recorded.append(pid)
        b._drain_stdout = lambda *a, **kw: None
        b._llama_log_fh = None
        b._process = None
        proc = mock.Mock()
        proc.pid = 4242
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: proc)

        class _TeardownWinsTheLock:
            """Shutdown taking the lock the instant the spawn releases it, which is
            the whole window: it clears the reference before the pid is read."""

            def __enter__(self):
                return None

            def __exit__(self, *_exc):
                b._process = None
                return False

        b._spawn_lock = _TeardownWinsTheLock()
        assert b._start_llama_process(["llama-server"], {}, child_gpu_physical_ids = None) is True
        assert recorded == [4242]

    def test_a_new_server_lifecycle_clears_the_shutdown_state(self):
        """The backend is a module singleton and an embedded host may call
        run_server() again in the same process, so this state is scoped to a
        lifecycle. Left latched, every launch of the second session is refused."""
        b = _make_backend()
        b._shutting_down = True
        b._torn_down_process = b._process
        b._begin_server_lifecycle()
        assert b._shutting_down is False
        assert b._torn_down_process is None

    def test_a_healthy_probe_during_shutdown_is_not_a_healthy_server(self, monkeypatch):
        """The probe blocks for up to 2s. A 200 arriving as teardown begins must not
        publish a child the shutdown is already killing, exactly as for a cancel."""
        b = _make_backend()
        b._process.poll.return_value = None

        def probe(*a, **kw):
            b._shutting_down = True
            return mock.Mock(status_code = 200)

        monkeypatch.setattr(httpx, "get", probe)
        assert b._wait_for_health(timeout = 5.0, interval = 0.01) is False
        assert b._health_wait_cancelled is True

    def test_a_teardown_during_the_last_probe_is_not_reported_as_a_timeout(self, monkeypatch):
        """The deadline is the other way out of the loop. A teardown landing in the
        final probe leaves no iteration to notice it, so without a check here the
        wait returns a plain timeout, the caller sees something retryable, and the
        fallbacks respawn after shutdown killed the first server."""
        b = _make_backend()
        b._process.poll.return_value = None

        def probe(*a, **kw):
            b._torn_down_process = b._process  # shutdown, during the last probe
            return mock.Mock(status_code = 503)

        monkeypatch.setattr(httpx, "get", probe)
        assert b._wait_for_health(timeout = 0.02, interval = 0.01) is False
        assert b._health_wait_cancelled is True
        # Not a live-but-never-healthy load, so it must not be classified as one.
        assert not any("health check timed out" in ln for ln in b._stdout_lines)

    def test_a_timeout_with_no_teardown_still_reports_a_timeout(self, monkeypatch):
        """The check above must not swallow the ordinary #5740 classification."""
        b = _make_backend()
        b._process.poll.return_value = None
        b._torn_down_process = mock.Mock()  # an earlier child, unrelated
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock.Mock(status_code = 503))
        assert b._wait_for_health(timeout = 0.02, interval = 0.01) is False
        assert b._health_wait_cancelled is False
        assert any("health check timed out" in ln for ln in b._stdout_lines)

    def test_an_earlier_childs_teardown_does_not_end_a_later_wait(self, monkeypatch):
        """The other half of keying on identity: the marker names one child, so a
        load that replaced a torn-down one is not aborted by its predecessor."""
        b = _make_backend()
        b._torn_down_process = mock.Mock()  # the previous child, already reaped
        b._process.poll.return_value = 1  # this one really did crash
        b._process.returncode = 1
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock.Mock(status_code = 503))
        with mock.patch("core.inference.llama_cpp.logger") as log:
            assert b._wait_for_health(timeout = 1.0, interval = 0.01) is False
        assert b._health_wait_cancelled is False, "a stale teardown aborted a live load"
        assert any("exited with code 1" in str(c) for c in log.error.call_args_list)

    @pytest.mark.parametrize("teardown", [True, False])
    def test_kill_process_publishes_a_teardown_before_it_signals(self, teardown):
        """The ordering the test above depends on: published before terminate(),
        not in the finally that clears the reference.

        Only for a teardown. The retry ladder reaps a crashed child through this
        same method between attempts, and marking that terminal would abort loads
        the --fit off and CPU fallbacks currently recover."""
        b = _make_backend()
        b._torn_down_process = None
        b._stop_mtp_crash_watchdog = lambda *a, **kw: None
        b._reset_effective_parallel_slots = lambda *a, **kw: None
        b._leading_process_group = lambda *a, **kw: None
        b._collect_descendants = lambda *a, **kw: []
        b._kill_process_group = lambda *a, **kw: None
        b._terminate_descendants = lambda *a, **kw: None
        seen = {}

        def _terminate():
            # What a racing _wait_for_health would observe at this instant.
            seen["flag_at_signal"] = getattr(b, "_torn_down_process", None) is b._process
            seen["process_still_set"] = b._process is not None

        b._process.terminate = _terminate
        b._process.poll.return_value = -15
        b._kill_process(teardown = teardown)
        assert seen["process_still_set"] is True, "the window this guards would not exist"
        assert (
            seen["flag_at_signal"] is teardown
        ), "teardown published late, or a reap marked terminal"

    def test_a_crash_still_reports_the_exit_code(self, monkeypatch):
        """The teardown guard must not swallow the crash branch: an exited
        process still has to name its exit code and its output."""
        b = _make_backend()
        b._process.poll.return_value = 1
        b._process.returncode = 1
        b._stdout_lines = ["ROCm error: out of memory"]
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock.Mock(status_code = 503))
        with mock.patch("core.inference.llama_cpp.logger") as log:
            assert b._wait_for_health(timeout = 1.0, interval = 0.01) is False
        assert any("exited with code 1" in str(c) for c in log.error.call_args_list)

    def test_cancel_stops_the_wait_without_a_timeout_marker(self, monkeypatch):
        b = _make_backend()
        b._process.poll.return_value = None
        b._cancel_event = threading.Event()
        b._cancel_event.set()
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock.Mock(status_code = 503))
        started = time.monotonic()
        assert b._wait_for_health(timeout = 30.0, interval = 0.01) is False
        assert time.monotonic() - started < 1.0
        assert not any("health check timed out" in ln for ln in b._stdout_lines)

    def test_cancel_midway_stops_a_wait_already_in_progress(self, monkeypatch):
        b = _make_backend()
        b._process.poll.return_value = None
        b._cancel_event = threading.Event()
        probes = []

        def probe(*a, **kw):
            probes.append(1)
            if len(probes) == 3:
                b._cancel_event.set()
            return mock.Mock(status_code = 503)

        monkeypatch.setattr(httpx, "get", probe)
        assert b._wait_for_health(timeout = 30.0, interval = 0.01) is False
        assert len(probes) == 3

    def test_cancel_at_the_deadline_is_not_reported_as_a_timeout(self, monkeypatch):
        b = _make_backend()
        b._process.poll.return_value = None
        b._cancel_event = threading.Event()
        b._health_probe_event = mock.Mock()
        b._health_probe_event.wait.side_effect = lambda _interval: b._cancel_event.set()
        ticks = iter((0.0, 0.0, 1.0))
        monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock.Mock(status_code = 503))

        assert b._wait_for_health(timeout = 0.5, interval = 0.5) is False
        assert b._health_wait_cancelled is True
        assert not any("health check timed out" in ln for ln in b._stdout_lines)

    def test_a_scoped_load_cancel_stops_the_wait(self, monkeypatch):
        """An auto-switch or a /load carrying load_request_id cancels through its own
        event and never calls unload_model, so _cancel_event stays clear. The wait has
        to honor the predicate load_model passes down or it polls the full timeout."""
        b = _make_backend()
        b._process.poll.return_value = None
        b._cancel_event = threading.Event()  # no unload was issued
        scoped = threading.Event()
        scoped.set()
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock.Mock(status_code = 503))
        started = time.monotonic()
        assert b._wait_for_health(timeout = 30.0, interval = 0.01, cancelled = scoped.is_set) is False
        assert time.monotonic() - started < 1.0
        assert not any("health check timed out" in ln for ln in b._stdout_lines)

    def test_a_cancel_during_the_probe_does_not_publish_the_model(self, monkeypatch):
        """The probe blocks for up to 2s. A cancel landing inside that window used to be
        ignored because the 200 returned first, so the unwanted model went live."""
        b = _make_backend()
        b._process.poll.return_value = None
        b._cancel_event = threading.Event()
        scoped = threading.Event()

        def probe(*a, **kw):
            scoped.set()  # the user cancels while this request is in flight
            return mock.Mock(status_code = 200)

        monkeypatch.setattr(httpx, "get", probe)
        assert b._wait_for_health(timeout = 30.0, interval = 0.01, cancelled = scoped.is_set) is False

    def test_read_error_loops_to_subprocess_poll(self, monkeypatch):
        """WinError 10054 (httpx.ReadError) must be swallowed; the next iteration sees the dead subprocess and returns False with a structured exit-code log."""
        b = _make_backend()
        # Iter 1: alive (reach probe); iter 2: exited (exit-code branch -> False).
        b._process.poll.side_effect = [None, 1]
        b._process.returncode = 1
        b._stdout_lines = ["llama-server: ggml-cuda.dll failed to load"]

        def raise_read_error(*a, **kw):
            raise httpx.ReadError("WinError 10054")

        monkeypatch.setattr(httpx, "get", raise_read_error)
        assert b._wait_for_health(timeout = 5.0, interval = 0.01) is False
        # Both loop iterations ran -- the ReadError did not bubble.
        assert b._process.poll.call_count >= 2

    def test_remote_protocol_error_also_swallowed(self, monkeypatch):
        """A partial/malformed probe response (server crashed mid-headers)
        raises RemoteProtocolError -- also non-fatal."""
        b = _make_backend()
        b._process.poll.side_effect = [None, -1]
        b._process.returncode = -1

        def raise_rpe(*a, **kw):
            raise httpx.RemoteProtocolError("partial response")

        monkeypatch.setattr(httpx, "get", raise_rpe)
        assert b._wait_for_health(timeout = 5.0, interval = 0.01) is False
        assert b._process.poll.call_count >= 2

    def test_write_error_also_swallowed(self, monkeypatch):
        """Send-side socket failure mid-request raises WriteError --
        same recovery path as ReadError."""
        b = _make_backend()
        b._process.poll.side_effect = [None, 1]
        b._process.returncode = 1

        def raise_we(*a, **kw):
            raise httpx.WriteError("connection broken on write")

        monkeypatch.setattr(httpx, "get", raise_we)
        assert b._wait_for_health(timeout = 5.0, interval = 0.01) is False
        assert b._process.poll.call_count >= 2

    def test_connect_error_swallowed_until_success(self, monkeypatch):
        """Sanity: existing ConnectError swallowing still works -- the loop
        retries until llama-server answers 200."""
        b = _make_backend()
        b._process.poll.return_value = None
        calls = {"n": 0}
        ok_resp = mock.Mock(status_code = 200)

        def cycling(*a, **kw):
            calls["n"] += 1
            if calls["n"] < 3:
                raise httpx.ConnectError("not yet")
            return ok_resp

        monkeypatch.setattr(httpx, "get", cycling)
        assert b._wait_for_health(timeout = 5.0, interval = 0.01) is True
        assert calls["n"] >= 3

    def test_stdout_readiness_wakes_probe_before_fallback_interval(self, monkeypatch):
        b = _make_backend()
        b._process.poll.return_value = None
        b._health_probe_event = threading.Event()
        calls = {"n": 0}

        def becomes_healthy(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("not yet")
            return mock.Mock(status_code = 200)

        monkeypatch.setattr(httpx, "get", becomes_healthy)
        wake = threading.Timer(0.02, b._health_probe_event.set)
        wake.start()
        start = time.monotonic()
        try:
            assert b._wait_for_health(timeout = 1.0, interval = 0.5) is True
        finally:
            wake.cancel()
        assert time.monotonic() - start < 0.25
        assert calls["n"] == 2

    def test_stdout_drain_sets_health_event_on_readiness_line(self):
        b = _make_backend()
        b._health_probe_event = threading.Event()
        b._process.stdout = iter(["main: server is listening on http://127.0.0.1:12345\n"])
        event_seen_while_draining = []
        b._llama_log_fh = mock.Mock()
        b._llama_log_fh.write.side_effect = lambda _line: event_seen_while_draining.append(
            b._health_probe_event.is_set()
        )

        b._drain_stdout()

        assert event_seen_while_draining == [True]

    def test_dead_process_before_probe_returns_false(self, monkeypatch):
        """poll() != None on entry: _wait_for_health returns False
        immediately without calling httpx."""
        b = _make_backend()
        b._process.poll.return_value = 137
        b._process.returncode = 137
        b._stdout_lines = ["llama-server: out of memory"]
        called = {"n": 0}

        def should_not_be_called(*a, **kw):
            called["n"] += 1
            raise AssertionError("httpx.get must not run when subprocess is dead")

        monkeypatch.setattr(httpx, "get", should_not_be_called)
        assert b._wait_for_health(timeout = 5.0, interval = 0.01) is False
        assert called["n"] == 0


class TestCrashLogTail:
    """The "exited with code X" log must keep the TAIL of the output.

    Crash diagnostics (abort reason, ROCm/CUDA error text) print last,
    after the long startup banner; head truncation has cut off exactly
    the diagnostic line in field reports (gfx1151 fit-step abort)."""

    @staticmethod
    def _capture_error_logs(monkeypatch) -> list:
        """Capture module-logger .error() messages directly -- immune to
        whatever logging/structlog config sibling test modules installed."""
        import core.inference.llama_cpp as _llama_mod

        records: list = []
        fake_logger = mock.Mock()
        fake_logger.error = mock.Mock(side_effect = lambda msg, *a, **k: records.append(msg))
        monkeypatch.setattr(_llama_mod, "logger", fake_logger)
        return records

    def test_crash_log_keeps_tail_not_head(self, monkeypatch):
        records = self._capture_error_logs(monkeypatch)
        b = _make_backend()
        b._process.poll.return_value = 1
        b._process.returncode = 1
        # >2000 chars of banner, diagnostic on the final line.
        banner = [f"load_model: tensor blk.{i} buffer ROCm0" for i in range(80)]
        diagnostic = "ggml-cuda.cu:103: ROCm error: out of memory"
        b._stdout_lines = banner + [diagnostic]

        assert b._wait_for_health(timeout = 1.0, interval = 0.01) is False

        crash_logs = [m for m in records if "exited with code" in m]
        assert crash_logs, "crash must produce an exited-with-code log"
        assert diagnostic in crash_logs[-1]
        assert "Output (tail)" in crash_logs[-1]
        # The head of the banner must be the part sacrificed to truncation.
        assert "blk.0 buffer" not in crash_logs[-1]

    def test_crash_log_mentions_log_file_when_present(self, monkeypatch):
        records = self._capture_error_logs(monkeypatch)
        b = _make_backend()
        b._process.poll.return_value = 1
        b._process.returncode = 1
        b._stdout_lines = ["boom"]
        b._llama_log_path = Path("C:/logs/llama-123-port-1234.log")

        assert b._wait_for_health(timeout = 1.0, interval = 0.01) is False

        crash_logs = [m for m in records if "exited with code" in m]
        assert crash_logs and "llama-123-port-1234.log" in crash_logs[-1]


class TestRetryLogFilenameUnique:
    """The --fit off retry can respawn within the same epoch second; the log
    filename must carry the attempt index or the second open ("w") truncates
    the crash log the retry warning just referenced (found by simulation:
    frozen time.time -> single file, crash evidence gone)."""

    def test_log_name_includes_attempt_index(self):
        src = (
            Path(__file__).resolve().parent.parent / "core" / "inference" / "llama_cpp.py"
        ).read_text(encoding = "utf-8")
        assert "-try{_spawn_attempt}.log" in src


class TestFitOffRetryEligible:
    """Gate for the one-shot --fit off startup-crash retry.

    Retry only when Unsloth's own VRAM math placed the model and nothing
    on the command line chose the fit mode explicitly."""

    def test_eligible_for_plain_ngl_launch(self):
        cmd = ["llama-server", "-m", "x.gguf", "-ngl", "-1", "--jinja"]
        assert LlamaCppBackend._fit_off_retry_eligible(cmd, use_fit = False) is True

    def test_not_eligible_when_use_fit(self):
        cmd = ["llama-server", "-m", "x.gguf", "--fit", "on"]
        assert LlamaCppBackend._fit_off_retry_eligible(cmd, use_fit = True) is False

    @pytest.mark.parametrize(
        "fit_args",
        [
            ["--fit", "on"],
            ["--fit", "off"],
            ["-fit", "off"],
            ["--fit=on"],
            ["-fit=off"],
        ],
    )
    def test_not_eligible_with_explicit_fit_flag(self, fit_args):
        cmd = ["llama-server", "-m", "x.gguf", *fit_args]
        assert LlamaCppBackend._fit_off_retry_eligible(cmd, use_fit = False) is False

    @pytest.mark.parametrize(
        "tuning_args",
        [
            ["--fit-ctx", "8192"],
            ["--fit-target", "1024"],
            ["-fitc", "4096"],
            ["-fitt", "512"],
            ["--fit-ctx=8192"],
        ],
    )
    def test_fit_tuning_flags_do_not_block_retry(self, tuning_args):
        cmd = ["llama-server", "-m", "x.gguf", *tuning_args]
        assert LlamaCppBackend._fit_off_retry_eligible(cmd, use_fit = False) is True


class TestCancelledWaitEndsTheLoad:
    """An auto-switch cancels through its own event and never unloads, so the child is
    still alive and poll() reports no exit code. Classifying that as a start failure
    raises a 500 at the caller before it can run its own cancellation handling."""

    def test_a_cancelled_health_wait_returns_false_instead_of_raising(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\0" * 4096)

        b = LlamaCppBackend()
        b._find_llama_server_binary = lambda *a, **kw: "/usr/bin/true"
        # The header refusals read a real GGUF; this fixture stands in for a chat model.
        b._non_chat_gguf_refusal_for_path = lambda *a, **kw: None
        b._non_chat_gguf_refusal = lambda *a, **kw: None
        b._kill_process = lambda *a, **kw: None

        def _start(cmd, env, **kw):
            proc = mock.Mock()
            proc.poll.return_value = None  # alive: the cancel kills it, not a crash
            proc.pid = 424242
            b._process = proc
            b._stdout_lines = ["build: 6543", "main: loading model"]
            return proc

        b._start_llama_process = _start
        scoped = threading.Event()
        real_wait = b._wait_for_health

        def _wait(
            timeout = 600.0,
            interval = 0.5,
            cancelled = None,
        ):
            scoped.set()
            return real_wait(timeout = 2.0, interval = 0.05, cancelled = cancelled)

        b._wait_for_health = _wait

        assert (
            b.load_model(
                GgufLoadIntent(
                    model_identifier = "owner/model",
                    gguf_path = str(gguf),
                    n_ctx = 4096,
                ),
                load_cancel_event = scoped,
            )
            is False
        )

    def test_a_teardown_mid_load_ends_the_load_instead_of_raising(self, tmp_path, monkeypatch):
        """The whole race through the real caller (#10353): the waiter tests above
        never reach _spawn_and_wait, which is where the traceback resurfaced."""
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\0" * 4096)

        b = LlamaCppBackend()
        b._find_llama_server_binary = lambda *a, **kw: "/usr/bin/true"
        b._non_chat_gguf_refusal_for_path = lambda *a, **kw: None
        b._non_chat_gguf_refusal = lambda *a, **kw: None
        # The drain thread would iterate a Mock's stdout forever.
        b._drain_stdout = lambda *a, **kw: None

        def _kill():
            b._process = None  # the one line of _kill_process this race turns on

        b._kill_process = _kill

        spawns = []
        _real_popen = subprocess.Popen

        def _popen(*a, **kw):
            argv = [str(x) for x in (a[0] if a else kw.get("args") or [])]
            # `--help` and nvidia-smi are not launches; run() needs a real Popen.
            if str(gguf) not in argv:
                return _real_popen(*a, **kw)
            spawns.append(argv)
            proc = mock.Mock()
            proc.poll.return_value = None  # alive: shutdown kills it, it does not crash
            proc.pid = 424242
            return proc

        monkeypatch.setattr(subprocess, "Popen", _popen)

        def probe(*a, **kw):
            _kill()  # run.py's shutdown, arriving while the load waits
            return mock.Mock(status_code = 503)

        monkeypatch.setattr(httpx, "get", probe)

        assert (
            b.load_model(
                GgufLoadIntent(
                    model_identifier = "owner/model",
                    gguf_path = str(gguf),
                    n_ctx = 4096,
                ),
            )
            is False
        )
        # A second server would outlive the app whose shutdown killed the first.
        assert len(spawns) == 1, f"respawned after shutdown (spawns={len(spawns)})"


def test_a_cancelled_diffusion_start_reaps_the_runner():
    """An automatic switch cancels without unloading, so nothing else reaps the shim
    and the visual server; they would keep loading and holding memory."""
    b = LlamaCppBackend()
    kills = []
    b._kill_process = lambda *a, **kw: kills.append(1)
    b._find_diffusion_assets = lambda *a, **kw: (["/bin/true"], "/bin/true", None)
    b._find_free_port = lambda *a, **kw: 45999

    def _wait(
        timeout = 600.0,
        interval = 0.5,
        cancelled = None,
    ):
        b._health_wait_cancelled = True
        return False

    b._wait_for_health = _wait
    proc = mock.Mock()
    proc.poll.return_value = None
    proc.pid = 999
    with mock.patch.object(subprocess, "Popen", return_value = proc):
        assert (
            b._start_diffusion_server(
                model_path = "/m/x.gguf",
                gguf_path = "/m/x.gguf",
                hf_repo = None,
                hf_variant = None,
                model_identifier = "o/m",
                n_ctx = 4096,
                extra_args = None,
                cancelled = lambda: True,
            )
            is False
        )
    # One teardown before the launch, one for the cancelled runner.
    assert len(kills) == 2, f"the cancelled runner was left running (kills={len(kills)})"


def test_a_diffusion_cancel_after_health_reaps_the_runner():
    b = LlamaCppBackend()
    kills = []
    b._kill_process = lambda *a, **kw: kills.append(1)
    b._find_diffusion_assets = lambda *a, **kw: (["/bin/true"], "/bin/true", None)
    b._find_free_port = lambda *a, **kw: 45999
    scoped = threading.Event()

    def _wait(
        timeout = 600.0,
        interval = 0.5,
        cancelled = None,
    ):
        scoped.set()
        return True

    b._wait_for_health = _wait
    proc = mock.Mock()
    proc.poll.return_value = None
    proc.pid = 999
    with mock.patch.object(subprocess, "Popen", return_value = proc):
        assert (
            b._start_diffusion_server(
                model_path = "/m/x.gguf",
                gguf_path = "/m/x.gguf",
                hf_repo = None,
                hf_variant = None,
                model_identifier = "o/m",
                n_ctx = 4096,
                extra_args = None,
                cancelled = scoped.is_set,
            )
            is False
        )
    assert b._healthy is False
    assert len(kills) == 2, f"the cancelled runner was left running (kills={len(kills)})"


def _cancel_scaffold(tmp_path, kills):
    """A backend wired far enough to run load_model without weights or a server."""
    b = LlamaCppBackend()
    b._find_llama_server_binary = lambda *a, **kw: "/usr/bin/true"
    b._non_chat_gguf_refusal_for_path = lambda *a, **kw: None
    b._non_chat_gguf_refusal = lambda *a, **kw: None
    b._kill_process = lambda *a, **kw: kills.append(1)

    def _start(cmd, env, **kw):
        proc = mock.Mock()
        proc.poll.return_value = None
        proc.pid = 424242
        b._process = proc
        b._stdout_lines = ["build: 6543", "main: loading model"]
        return proc

    b._start_llama_process = _start
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF" + b"\0" * 4096)
    return b, gguf


def test_combined_download_cancel_wait_observes_the_scoped_event():
    from core.inference.llama_cpp import _CombinedCancelEvent

    shared = threading.Event()
    scoped = threading.Event()
    timer = threading.Timer(0.01, scoped.set)
    timer.start()
    try:
        assert _CombinedCancelEvent(shared, scoped).wait(0.5)
    finally:
        timer.cancel()


@pytest.mark.parametrize("cancel_source", ["shared", "scoped"])
def test_remote_download_observes_both_load_cancel_sources(tmp_path, cancel_source):
    kills = []
    b, gguf = _cancel_scaffold(tmp_path, kills)
    b._remote_non_chat_gguf_refusal = lambda **_kw: None
    scoped = threading.Event()
    seen = []

    def _download(**kwargs):
        cancel_event = kwargs["cancel_event"]
        seen.append(cancel_event)
        (b._cancel_event if cancel_source == "shared" else scoped).set()
        assert cancel_event.is_set()
        return str(gguf)

    b._download_gguf = _download
    assert (
        b.load_model(
            GgufLoadIntent(
                model_identifier = "owner/model",
                hf_repo = "owner/model-GGUF",
                speculative_type = "none",
                n_ctx = 4096,
            ),
            load_cancel_event = scoped,
        )
        is False
    )
    assert len(seen) == 1


def test_a_stale_cancel_marker_does_not_abort_the_next_load(tmp_path):
    """The marker belongs to one attempt. Left set, a guard that runs before this
    load's first health wait reads the previous request's cancellation and swallows
    a genuine start failure."""
    kills = []
    b, gguf = _cancel_scaffold(tmp_path, kills)
    b._health_wait_cancelled = True  # left over from an earlier cancelled load
    # A genuine failure: the wait fails and does NOT mark the load cancelled.
    b._wait_for_health = lambda timeout = 600.0, interval = 0.5, cancelled = None: False

    with pytest.raises(RuntimeError):
        b.load_model(
            GgufLoadIntent(
                model_identifier = "owner/model",
                gguf_path = str(gguf),
                n_ctx = 4096,
            )
        )


def test_cancelled_health_wait_removes_the_staged_cpu_runtime(tmp_path):
    kills = []
    cleaned = []
    waits = []
    b, gguf = _cancel_scaffold(tmp_path, kills)
    scoped = threading.Event()

    def _wait(
        timeout = 600.0,
        interval = 0.5,
        cancelled = None,
    ):
        waits.append(1)
        b._process = mock.Mock(returncode = -9)
        b._process.poll.return_value = -9
        b._cpu_fallback_runtime = _types.SimpleNamespace(
            tempdir = _types.SimpleNamespace(cleanup = lambda: cleaned.append(1))
        )
        scoped.set()
        b._health_wait_cancelled = True
        return False

    b._wait_for_health = _wait
    assert (
        b.load_model(
            GgufLoadIntent(
                model_identifier = "owner/model",
                gguf_path = str(gguf),
                n_ctx = 4096,
            ),
            load_cancel_event = scoped,
        )
        is False
    )
    assert waits == [1]
    assert b._cpu_fallback_runtime is None
    assert cleaned == [1]


def test_a_cancel_after_the_server_is_healthy_does_not_publish_it(tmp_path):
    """Cancelling during the post-health setup used to leave the child resident while
    the cancel route saw is_loaded and skipped teardown."""
    kills = []
    cleaned = []
    b, gguf = _cancel_scaffold(tmp_path, kills)

    def _wait(
        timeout = 600.0,
        interval = 0.5,
        cancelled = None,
    ):
        b._cpu_fallback_runtime = _types.SimpleNamespace(
            tempdir = _types.SimpleNamespace(cleanup = lambda: cleaned.append(1))
        )
        b._cancel_event.set()  # the user cancels while the load finishes publishing
        return True

    b._wait_for_health = _wait
    assert (
        b.load_model(
            GgufLoadIntent(
                model_identifier = "owner/model",
                gguf_path = str(gguf),
                n_ctx = 4096,
            )
        )
        is False
    )
    assert kills, "the healthy-but-cancelled child was left resident"
    assert b._cpu_fallback_runtime is None
    assert cleaned == [1]


def test_a_cancel_after_audio_setup_unloads_the_codec(tmp_path, monkeypatch):
    kills = []
    emptied = []
    b, gguf = _cancel_scaffold(tmp_path, kills)
    scoped = threading.Event()
    codec = mock.Mock()
    monkeypatch.setattr(LlamaCppBackend, "_codec_mgr", None)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _types.SimpleNamespace(
            cuda = _types.SimpleNamespace(
                is_available = lambda: True,
                empty_cache = lambda: emptied.append(1),
            )
        ),
    )

    def _apply_audio(_detected):
        LlamaCppBackend._codec_mgr = codec
        scoped.set()
        return True

    b._wait_for_health = lambda **_kwargs: True
    b._detect_audio_type_strict = lambda: "snac"
    b._apply_detected_audio = _apply_audio

    assert (
        b.load_model(
            GgufLoadIntent(
                model_identifier = "owner/model",
                gguf_path = str(gguf),
                n_ctx = 4096,
            ),
            load_cancel_event = scoped,
        )
        is False
    )
    codec.unload.assert_called_once_with()
    assert LlamaCppBackend._codec_mgr is None
    assert emptied == [1]


def _run_server_body() -> str:
    """run_server's source, without importing run.py.

    Importing it pulls in fastapi and the whole route tree, which is exactly the
    cost the ordering below exists to keep out of the early startup path.
    """
    import ast

    run_py = Path(__file__).resolve().parent.parent / "run.py"
    tree = ast.parse(run_py.read_text(encoding = "utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run_server":
            return ast.get_source_segment(run_py.read_text(encoding = "utf-8"), node) or ""
    raise AssertionError("run_server not found in run.py")


def test_the_lifecycle_reset_does_not_import_the_route_module_early():
    """The reset must not be what first builds the llama-server backend.

    `from routes.inference import ...` constructs the module singleton, which
    sweeps orphan llama-servers and registers an atexit handler. run_server
    documents five things as having to happen before any of that: the Windows
    UTF-8 reconfigure, the session log that catches import-time crashes, the
    structlog setup (whose cache_logger_on_first_use pins any logger that has
    already emitted), initialize_parent_lifetime() before a child can spawn, and
    write_startup_marker() before a sweep can reap a sibling's server.

    Ordering it after `from main import app` makes the import free: main imports
    the route package, so the singleton already exists and this is a lookup.
    """
    body = _run_server_body()

    main_import = body.index("from main import app")
    reset = body.index("_begin_server_lifecycle()")
    assert reset > main_import, (
        "the lifecycle reset imports routes.inference before `from main import app`, "
        "so it, not main, builds the backend singleton -- ahead of the startup steps "
        "run_server requires to come first"
    )

    for earlier in (
        'sys.stdout.reconfigure(encoding = "utf-8"',
        "_setup_server_disk_logging()",
        "LogConfig.setup_logging(",
        "initialize_parent_lifetime()",
        "write_startup_marker()",
    ):
        assert body.index(earlier) < reset, f"{earlier} must precede the lifecycle reset"


def test_the_lifecycle_reset_stays_below_the_argument_checks():
    """A rejected invocation must not touch the backend at all."""
    body = _run_server_body()

    assert body.index("choose an explicit port.") < body.index(
        "_begin_server_lifecycle()"
    ), "the reset runs before run_server has finished rejecting bad arguments"


def test_the_lifecycle_reset_is_the_last_thing_before_the_serve():
    """Clearing the shutdown flag is what lets a spawn through, so nothing that can
    abort startup may follow it.

    An embedded host restarting into an occupied port (_resolve_port) or a missing
    frontend (SystemExit) would otherwise leave _shutting_down False with the
    previous session's still-unwinding load free to start a child the shutdown
    sweep has already run past.
    """
    body = _run_server_body()
    # Statements, not text: the comments around these calls name them too, and
    # matching the prose made this compare the wrong offsets.
    reset = body.index("_llama_cpp_backend._begin_server_lifecycle()")
    serve = body.index("\n    thread.start()")
    assert reset < serve, "the lifecycle reset no longer precedes the serve"

    # Only the window between the reset and the serve. The sys.exit further down is
    # a startup FAILURE after the thread is running, by which point a spawn is
    # legitimate, so the whole tail is the wrong scope.
    #
    # sys.exit( as well as raise SystemExit: the first version of this test checked
    # only the latter and missed the admin-password gate, which exits the other way.
    window = body[reset:serve]
    for fail_fast in ("raise SystemExit", "sys.exit(", "_resolve_port("):
        assert fail_fast not in window, (
            f"{fail_fast} runs after the lifecycle reset, so a failed restart leaves the "
            "spawn guard cleared"
        )


class TestARefusedSpawnClosesItsLog:
    """Each spawn opens a per-attempt tee log just before Popen. A refusal returns
    without a process, and _kill_process returns early when there is none, so
    nothing else ever closes it: the next attempt overwrites the attribute, leaking
    the descriptor and holding the file lock on Windows."""

    def _backend(self, tmp_path):
        b = _make_backend()
        b._shutting_down = True
        b._spawn_lock = threading.Lock()
        b._stop_mtp_crash_watchdog = lambda: None
        b._process = None
        b._redacted_cmd_for_log = lambda c: c
        b._llama_log_path = str(tmp_path / "llama.log")
        b._llama_log_fh = None
        return b

    def test_start_llama_process_closes_it(self, tmp_path):
        """The handle under test is the one the method opens for itself, not one
        set beforehand: _start_llama_process clears the attribute and opens a fresh
        per-attempt log before it reaches the shutdown check."""
        b = self._backend(tmp_path)
        opened = []
        real_open = open

        def tracking_open(*a, **k):
            fh = real_open(*a, **k)
            opened.append(fh)
            return fh

        import builtins

        with mock.patch.object(builtins, "open", tracking_open):
            assert (
                b._start_llama_process(["llama-server"], {}, child_gpu_physical_ids = None) is False
            )

        assert opened, "the method never opened an attempt log; this proves nothing"
        assert all(fh.closed for fh in opened), "the refused attempt left its tee log open"
        assert b._llama_log_fh is None

    def test_the_kill_path_does_not_cover_it(self, tmp_path):
        """Why the explicit close is needed: _kill_process is the only other closer
        and it returns before that, since a refusal published no process."""
        b = self._backend(tmp_path)
        fh = open(tmp_path / "kill.log", "w", encoding = "utf-8")
        b._llama_log_fh = fh
        b._reset_effective_parallel_slots = lambda: None

        b._kill_process(teardown = True)

        assert not fh.closed, (
            "_kill_process now closes the handle with no process; if that is "
            "deliberate this test and the explicit close are both redundant"
        )
        fh.close()


def test_the_pid_is_recorded_before_the_spawn_lock_is_released():
    """Teardown takes the lock the instant publication ends. Recording the pid after
    that would write it back behind a sweep that just forgot it, and
    _pid_start_identity cannot read a start time for a reaped process -- so the
    record is a bare pid, which a later launch kills without an identity check."""
    b = _make_backend()
    b._shutting_down = False
    b._stop_mtp_crash_watchdog = lambda: None
    b._process = None
    b._redacted_cmd_for_log = lambda c: c
    b._llama_log_path = None
    b._llama_log_fh = None
    order = []

    class _Lock:
        def __enter__(self):
            order.append("lock")
            return None

        def __exit__(self, *_exc):
            order.append("unlock")
            return False

    b._spawn_lock = _Lock()
    b._record_server_pid = lambda pid: order.append("record")

    class _Proc:
        pid = 4242

        def poll(self):
            return None

    with mock.patch.object(subprocess, "Popen", lambda *a, **k: _Proc()):
        with mock.patch.object(threading, "Thread", lambda **k: mock.Mock()):
            assert b._start_llama_process(["llama-server"], {}, child_gpu_physical_ids = None) is True

    assert order.index("record") < order.index("unlock"), (
        f"the pid is recorded after the lock is released ({order}), so a teardown "
        "between the two writes a bare pid record for an already-reaped process"
    )


class TestHealthPublicationIsAtomicWithTeardown:
    """A successful probe and the _healthy commit are separate steps, so a teardown
    landing between them left the backend advertising a model whose child had
    already been killed. The recheck inside the wait narrows that window; only
    taking the same lock the teardown mark is set under closes it."""

    def _backend(self):
        b = _make_backend()
        b._stop_mtp_crash_watchdog = lambda: None
        b._healthy = False
        b._spawn_lock = threading.Lock()
        return b

    def test_a_teardown_before_the_commit_refuses_publication(self):
        b = self._backend()
        b._shutting_down = True

        assert b._publish_healthy() is False
        assert b._healthy is False, "a torn-down backend was published as healthy"

    def test_an_ordinary_load_still_publishes(self):
        b = self._backend()
        b._shutting_down = False

        assert b._publish_healthy() is True
        assert b._healthy is True

    def test_the_teardown_mark_and_the_commit_cannot_interleave(self):
        """The two orders the lock permits are the only safe ones: publish then
        teardown (which clears _healthy on its way out), or teardown then a refused
        publish. This drives the first order and asserts the second is impossible."""
        b = self._backend()
        b._shutting_down = False
        # A real child, not None: _kill_process returns before clearing _healthy when
        # there is nothing to kill, and a backend that published healthy by
        # definition has a process, so None would be testing the wrong ordering.
        b._process = object()
        b._reset_effective_parallel_slots = lambda: None
        b._leading_process_group = lambda _pid: None
        b._collect_descendants = lambda _pid: []
        b._diffusion_requested_ngl = None

        published = b._publish_healthy()
        assert published is True and b._healthy is True

        b._kill_process(teardown = True)

        assert (
            b._healthy is False
        ), "teardown left _healthy set, so a publication that won the race is never undone"
        assert b._publish_healthy() is False, "a later publication slipped past the teardown"


def test_the_deadline_asks_the_durable_flag_too(monkeypatch):
    """_kill_process publishes its two signals apart: _shutting_down under the
    spawn lock on entry, _torn_down_process only after collecting descendants. A
    wait whose deadline lands between them saw no marker and recorded a plain
    timeout, sending a deliberate teardown through startup-failure handling."""
    b = _make_backend()
    b._process.poll.return_value = None
    b._shutting_down = False
    b._torn_down_process = None

    # Set during the LAST probe, not before the wait: setting it up front is caught
    # by the top-of-iteration check and never reaches the deadline at all, so that
    # version of this test passed with or without the fix.
    def probe(*a, **kw):
        b._shutting_down = True  # teardown, after the final iteration's check
        return mock.Mock(status_code = 503)

    monkeypatch.setattr(httpx, "get", probe)

    assert b._wait_for_health(timeout = 0.01, interval = 0.05) is False
    assert b._health_wait_cancelled is True, "a teardown was recorded as a timeout"
    assert not any("health check timed out" in ln for ln in b._stdout_lines)


class TestAStaleLoadIsDroppedAtTheSerialScope:
    """Refusing a previous lifecycle's load only at the spawn is too late.

    A lock gives a waiter no priority, so the stale load can take the serial scope
    after the restart, and the duplicate-adoption phase inside it calls
    _kill_process() on whatever is loaded -- by then the NEW lifecycle's model. It
    would unload the restarted server and then decline to replace it.
    """

    def test_the_check_precedes_the_replacement_work(self):
        """Checked over the AST, not the text: a first version matched the words
        '_kill_process' in the explanatory comment above the guard and failed on
        correct code."""
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(LlamaCppBackend.load_model)))
        scope = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.With)
            and any("_serial_load_scope" in ast.unparse(i.context_expr) for i in n.items)
        )

        guard_line = next(
            n.lineno
            for n in ast.walk(scope)
            if isinstance(n, ast.Call) and "_spawn_is_stale" in ast.unparse(n.func)
        )

        early_kills = [
            n.lineno
            for n in ast.walk(scope)
            if isinstance(n, ast.Call)
            and "_kill_process" in ast.unparse(n.func)
            and n.lineno < guard_line
        ]
        assert not early_kills, (
            f"a stale load calls _kill_process at {early_kills}, before it is refused at "
            f"line {guard_line}, so it unloads the new lifecycle's model"
        )

        # And the refusal has to stop the load rather than just log it.
        src = inspect.getsource(LlamaCppBackend.load_model)
        stale = src.index("_spawn_is_stale()")
        assert "return False" in src[stale : stale + 300], "the stale-load branch does not return"


def test_a_lifecycle_cannot_reopen_while_a_teardown_is_still_killing():
    """_kill_process(teardown) used to release the lock as soon as it set the flag,
    but it goes on reading self._process and finally clears it. An embedded host
    that saw the server thread stop could reopen the lifecycle in that gap, let a
    new load spawn, and have this teardown drop or terminate its child."""
    b = _make_backend()
    b._stop_mtp_crash_watchdog = lambda: None
    b._reset_effective_parallel_slots = lambda: None
    b._diffusion_requested_ngl = None
    b._leading_process_group = lambda _pid: None
    b._collect_descendants = lambda _pid: []
    b._spawn_lock = threading.RLock()  # so the probe below can observe, not deadlock

    reopened_during_kill = []
    in_kill = threading.Event()
    release = threading.Event()

    class _SlowProcess:
        pid = 999

        def terminate(self):
            in_kill.set()
            release.wait(2.0)

        def wait(self, timeout = None):
            return 0

        def kill(self):
            pass

    b._process = _SlowProcess()

    def restarter():
        in_kill.wait(2.0)
        # _teardown_lock, not _spawn_lock: the kill holds the former for its whole
        # duration and the latter only long enough to set the flag, so a spawn can
        # be refused promptly. This probes the lock that carries the property.
        got = b._teardown_lock.acquire(blocking = False)
        if got:
            b._teardown_lock.release()
        reopened_during_kill.append(got)
        release.set()

    t = threading.Thread(target = restarter)
    t.start()
    try:
        b._kill_process(teardown = True)
    finally:
        release.set()
        t.join(5.0)

    assert reopened_during_kill == [False], (
        "the teardown lock was free while the kill was still running, so "
        "_begin_server_lifecycle could reopen the lifecycle mid-kill"
    )


def test_the_shutdown_cancels_loads_before_it_kills_the_server():
    """Order matters: killing first leaves the load to respawn into the teardown."""
    import ast
    import textwrap
    from pathlib import Path

    run_py = (Path(__file__).resolve().parent.parent / "run.py").read_text(encoding = "utf-8")
    tree = ast.parse(run_py)
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_graceful_shutdown"
    )
    src = textwrap.dedent(ast.get_source_segment(run_py, fn) or "")

    assert "cancel_pending_loads()" in src, "shutdown does not cancel in-flight loads"
    assert src.index("cancel_pending_loads()") < src.index(
        "_kill_process(teardown = True)"
    ), "the loads are cancelled after the kill, so one can still spawn into the teardown"


class TestATeardownDoesNotBlockASpawnItWillRefuse:
    """The kill needs a long hold so a lifecycle cannot reopen mid-terminate, but a
    spawn only needs to read the flag. Holding one lock for both made a spawn queue
    behind a SIGTERM/SIGKILL escalation for seconds before being told no, on a
    thread shutdown is already waiting for."""

    def _backend(self, on_terminate):
        b = _make_backend()
        b._stop_mtp_crash_watchdog = lambda: None
        b._reset_effective_parallel_slots = lambda: None
        b._diffusion_requested_ngl = None
        b._leading_process_group = lambda _p: None
        b._collect_descendants = lambda _p: []

        class _Stubborn:
            pid = 4242

            def __init__(self):
                self.killed = False

            def terminate(self):
                on_terminate()

            def wait(self, timeout = None):
                if not self.killed:
                    raise subprocess.TimeoutExpired("llama-server", timeout)

            def kill(self):
                self.killed = True

        b._process = _Stubborn()
        return b

    def test_a_spawn_is_refused_without_waiting_for_the_kill(self):
        in_kill = threading.Event()
        release = threading.Event()
        b = self._backend(lambda: (in_kill.set(), release.wait(5.0)))
        seen = []

        def spawner():
            in_kill.wait(5.0)
            start = time.monotonic()
            with b._spawn_lock:
                stale = b._spawn_is_stale()
            seen.append((time.monotonic() - start, stale))
            release.set()

        t = threading.Thread(target = spawner)
        t.start()
        try:
            b._kill_process(teardown = True)
        finally:
            release.set()
            t.join(10)

        waited, refused = seen[0]
        assert refused is True, "the spawn was allowed through during a teardown"
        assert waited < 1.0, (
            f"the spawn waited {waited:.2f}s for the kill before being refused; the "
            "teardown hold and the flag read must not share one lock"
        )

    def test_a_lifecycle_reset_still_waits_for_the_kill(self):
        """The other half, and the reason the long hold exists at all: clearing the
        flag mid-kill would let a new load spawn a child this teardown then drops."""
        in_kill = threading.Event()
        release = threading.Event()
        b = self._backend(lambda: (in_kill.set(), release.wait(2.0)))
        order = []

        def resetter():
            in_kill.wait(5.0)
            b._begin_server_lifecycle()
            order.append("reset")

        t = threading.Thread(target = resetter)
        t.start()
        try:
            b._kill_process(teardown = True)
            order.append("kill_done")
        finally:
            release.set()
            t.join(10)

        assert order == [
            "kill_done",
            "reset",
        ], f"the lifecycle reopened before the kill finished: {order}"


def test_the_lock_order_is_teardown_then_spawn():
    """Two locks means an order, and taking them the other way round deadlocks."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(LlamaCppBackend))
    inversions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.With) and any(
            "_spawn_lock" in ast.unparse(i.context_expr) for i in node.items
        ):
            for inner in ast.walk(node):
                if inner is node:
                    continue
                if isinstance(inner, ast.With) and any(
                    "_teardown_lock" in ast.unparse(i.context_expr) for i in inner.items
                ):
                    inversions.append(inner.lineno)
    assert not inversions, f"_teardown_lock taken inside _spawn_lock at {inversions}"
