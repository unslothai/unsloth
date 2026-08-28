# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for LlamaCppBackend._wait_for_health resilience.

The probe loop must swallow transient httpx errors and fall through to the
subprocess.poll() branch so a crashed llama-server surfaces a structured
"exited with code X" log instead of bubbling an opaque exception up to the
/api/inference/load route.
"""

from __future__ import annotations

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

    def test_cancel_leaves_a_marker_the_classifier_reads_as_a_cancel(self, monkeypatch):
        """The cancel kills the child on purpose, so poll() reports no exit code and the
        startup log holds nothing diagnostic. Without a marker the load is classified as
        an unknown start failure and the user is told to check their GGUF."""
        b = _make_backend()
        b._process.poll.return_value = None
        b._cancel_event = threading.Event()
        b._cancel_event.set()
        monkeypatch.setattr(httpx, "get", lambda *a, **kw: mock.Mock(status_code = 503))
        assert b._wait_for_health(timeout = 30.0, interval = 0.01) is False

        detail = LlamaCppBackend._classify_llama_start_failure(
            "\n".join(b._stdout_lines),
            "/models/model.gguf",
            "owner/model",
            returncode = None,
        )
        assert "cancelled" in detail
        assert "GGUF file is valid" not in detail

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


def test_every_health_wait_in_load_model_forwards_a_cancel_predicate():
    """A _wait_for_health call that does not carry the per-load predicate polls the
    full 600s on an auto-switch cancel, which sets only the generation event."""
    import ast

    src = Path(_BACKEND_DIR, "core", "inference", "llama_cpp.py").read_text(encoding = "utf-8")
    calls = [
        node
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_wait_for_health"
    ]
    assert calls, "the health wait moved; re-pin this test"
    missing = [c.lineno for c in calls if not any(k.arg == "cancelled" for k in c.keywords)]
    assert not missing, (
        f"_wait_for_health called without a cancelled predicate at lines {missing}; "
        "a cancel that is not the unload flag would be ignored there"
    )


class TestCancelledWaitEndsTheLoad:
    """An auto-switch cancels through its own event and never unloads, so the child is
    still alive and poll() reports no exit code. Classifying that as a start failure
    raises a 500 at the caller before it can run its own cancellation handling."""

    def test_a_cancelled_health_wait_returns_false_instead_of_raising(self, tmp_path):
        from core.inference.llama_cpp import GgufLoadIntent

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
        scoped.set()
        real_wait = b._wait_for_health
        b._wait_for_health = lambda timeout = 600.0, interval = 0.5, cancelled = None: real_wait(
            timeout = 2.0,
            interval = 0.05,
            cancelled = scoped.is_set,
        )

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


def test_every_failed_health_wait_in_load_model_checks_the_cancel_flag():
    """Each health wait inside load_model has its own failure branch, and each one
    classifies a startup error. A branch that skips the flag raises a 500 for a
    deliberate cancel instead of returning the load."""
    import ast

    src = Path(_BACKEND_DIR, "core", "inference", "llama_cpp.py").read_text(encoding = "utf-8")
    load_model = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "load_model"
    )
    waits = [
        n.lineno
        for n in ast.walk(load_model)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_wait_for_health"
    ]
    guards = [
        n.lineno
        for n in ast.walk(load_model)
        if isinstance(n, ast.Constant) and n.value == "_health_wait_cancelled"
    ]
    assert waits, "the health waits moved; re-pin this test"
    assert len(guards) >= len(waits), (
        f"{len(waits)} health wait(s) in load_model but only {len(guards)} cancel guard(s); "
        "a failure branch classifies a cancelled load as a server-start failure"
    )


def test_a_cancelled_diffusion_start_reaps_the_runner():
    """An automatic switch cancels without unloading, so nothing else reaps the shim
    and the visual server; they would keep loading and holding memory."""
    import subprocess

    b = LlamaCppBackend()
    kills = []
    b._kill_process = lambda *a, **kw: kills.append(1)
    b._find_diffusion_assets = lambda *a, **kw: (["/bin/true"], "/bin/true", None)
    b._find_free_port = lambda *a, **kw: 45999

    def _wait(timeout = 600.0, interval = 0.5, cancelled = None):
        b._health_wait_cancelled = True
        return False

    b._wait_for_health = _wait
    proc = mock.Mock()
    proc.poll.return_value = None
    proc.pid = 999
    with mock.patch.object(subprocess, "Popen", return_value = proc):
        assert b._start_diffusion_server(
            model_path = "/m/x.gguf", gguf_path = "/m/x.gguf", hf_repo = None,
            hf_variant = None, model_identifier = "o/m", n_ctx = 4096,
            extra_args = None, cancelled = lambda: True,
        ) is False
    # One teardown before the launch, one for the cancelled runner.
    assert len(kills) == 2, f"the cancelled runner was left running (kills={len(kills)})"
