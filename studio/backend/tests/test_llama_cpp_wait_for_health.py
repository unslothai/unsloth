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

# Sibling tests install lightweight httpx stubs via sys.modules.setdefault.
# When collected together, our `httpx` may be such a stub lacking `get`. Add
# the missing attributes so production code finds a working `httpx.get` and
# the standard exception types regardless of collection order.
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
        fake_logger.error = mock.Mock(
            side_effect = lambda msg, *a, **k: records.append(msg)
        )
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
            Path(__file__).resolve().parent.parent
            / "core"
            / "inference"
            / "llama_cpp.py"
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
