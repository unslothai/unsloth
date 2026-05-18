# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for LlamaCppBackend._wait_for_health resilience.

The probe loop must swallow transient httpx errors and fall through to
the subprocess.poll() branch so a crashed llama-server surfaces a
structured "exited with code X" log instead of bubbling an opaque
exception up to the /api/inference/load route.
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

# Match the stubbing pattern in sibling tests so the module imports in
# a lightweight env without fastapi.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
sys.modules.setdefault("structlog", _types.ModuleType("structlog"))

import httpx  # noqa: E402

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

# Sibling tests in this directory install lightweight httpx stubs via
# sys.modules.setdefault. When collected together, our `httpx` symbol
# may be one of those stubs, which lacks `get`. Ensure the production
# code finds a working `httpx.get` and the standard exception types
# regardless of collection order by adding the missing attributes.
if not hasattr(httpx, "get"):
    httpx.get = None  # placeholder; every test below monkeypatches it
for _exc_name in (
    "ConnectError",
    "TimeoutException",
    "ReadError",
    "RemoteProtocolError",
):
    if not hasattr(httpx, _exc_name):
        setattr(httpx, _exc_name, type(_exc_name, (Exception,), {}))


def _make_backend(port: int = 12345) -> LlamaCppBackend:
    """Build a barebones LlamaCppBackend instance with only the
    attributes _wait_for_health touches. Bypasses __init__ so we do not
    pull in the full subprocess + logging stack."""
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

    def test_read_error_loops_to_subprocess_poll(self, monkeypatch):
        """WinError 10054 maps to httpx.ReadError. The loop must swallow
        it and the next iteration must detect the dead subprocess via
        poll() != None, returning False with a structured exit-code log
        instead of bubbling the ReadError."""
        b = _make_backend()
        # First iteration: process alive (so we reach the httpx probe).
        # Second iteration: process has exited (so we hit the structured
        # exit-code branch and return False).
        b._process.poll.side_effect = [None, 1]
        b._process.returncode = 1
        b._stdout_lines = ["llama-server: ggml-cuda.dll failed to load"]

        def raise_read_error(*a, **kw):
            raise httpx.ReadError("WinError 10054")

        monkeypatch.setattr(httpx, "get", raise_read_error)
        assert b._wait_for_health(timeout = 5.0, interval = 0.01) is False
        # Both iterations of the loop ran -- the ReadError did not bubble.
        assert b._process.poll.call_count >= 2

    def test_remote_protocol_error_also_swallowed(self, monkeypatch):
        """Partial / malformed response on the probe (server crashed
        mid-headers) raises RemoteProtocolError -- also non-fatal."""
        b = _make_backend()
        b._process.poll.side_effect = [None, -1]
        b._process.returncode = -1

        def raise_rpe(*a, **kw):
            raise httpx.RemoteProtocolError("partial response")

        monkeypatch.setattr(httpx, "get", raise_rpe)
        assert b._wait_for_health(timeout = 5.0, interval = 0.01) is False
        assert b._process.poll.call_count >= 2

    def test_connect_error_swallowed_until_success(self, monkeypatch):
        """Sanity: existing ConnectError swallowing still works -- the
        loop retries until llama-server eventually answers 200."""
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
        """If poll() != None on entry, _wait_for_health must return
        False immediately without calling httpx at all."""
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
