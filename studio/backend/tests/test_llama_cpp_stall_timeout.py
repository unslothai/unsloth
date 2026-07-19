# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression test for the post-first-token stall timeout in the cancel-aware read.

httpcore snapshots ``request.extensions["timeout"]["read"]`` once at body start,
so when ``_iter_text_cancellable`` lowers the read timeout to the stall timeout
after the first token, a one-token-then-silent server hangs for the full prefill
window. The fix makes the wrapper re-read the live extensions timeout per call.
Exercised with a fake clock and always-silent stream: the read must give up after
the live stall timeout, not the stale prefill timeout passed in by httpcore.
"""

from __future__ import annotations

import inspect
import sys
import threading
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Mirror sibling tests' stubbing so the module imports without fastapi.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
sys.modules.setdefault("structlog", _types.ModuleType("structlog"))

import httpcore  # noqa: E402

from core.inference import llama_cpp as llama_cpp_mod  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

_PREFILL_TIMEOUT = 1200.0  # what httpcore snapshots from the prefill timeout
_STALL_TIMEOUT = 120.0  # the post-first-token stall timeout the wrapper must honor


class _Obj:
    pass


def _install(response, clock, silent_stream):
    """Wire fake client/pool objects so _install_cancel_aware_read finds the
    stream, then return the wrapped stream.read."""
    inner = _Obj()
    inner._network_stream = silent_stream
    connection = _Obj()
    connection._connection = inner
    pool = _Obj()
    pool._connections = [connection]
    transport = _Obj()
    transport._pool = pool
    client = _Obj()
    client._transport = transport

    cancel_event = threading.Event()  # never set: we test the stall path, not cancel
    sig = inspect.signature(LlamaCppBackend._install_cancel_aware_read)
    if "response" in sig.parameters:
        # Fixed signature: wrapper reads the live extensions timeout.
        LlamaCppBackend._install_cancel_aware_read(client, cancel_event, response)
    else:
        # Pre-fix signature: no response, so the stall assertion fails (proves the bug).
        LlamaCppBackend._install_cancel_aware_read(client, cancel_event)
    return silent_stream.read


def test_stall_timeout_honored_after_first_token(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(llama_cpp_mod.time, "monotonic", lambda: clock["t"])

    # One token then silence: every read times out, advancing fake time by its timeout.
    def silent_read(max_bytes, timeout = None):
        clock["t"] += timeout if timeout is not None else 0.0
        raise httpcore.ReadTimeout("slice timed out on silence")

    stream = _Obj()
    stream.read = silent_read

    # First token seen: the live read timeout is lowered to the stall timeout.
    request = _Obj()
    request.extensions = {"timeout": {"read": _STALL_TIMEOUT}}
    response = _Obj()
    response.request = request

    wrapped_read = _install(response, clock, stream)

    # httpcore still passes the stale prefill timeout it snapshotted at body start.
    with pytest.raises(httpcore.ReadTimeout):
        wrapped_read(65536, timeout = _PREFILL_TIMEOUT)

    # Must give up ~stall timeout after the last token, not the prefill window.
    assert clock["t"] <= _STALL_TIMEOUT * 1.5, (
        f"stall timeout not honored: waited {clock['t']}s "
        f"(expected ~{_STALL_TIMEOUT}s, not {_PREFILL_TIMEOUT}s)"
    )
    assert clock["t"] >= _STALL_TIMEOUT * 0.5


def test_prefill_timeout_used_when_no_live_override(monkeypatch):
    """Without a lowered live timeout, the wrapper honors the passed prefill
    timeout, so the normal first-token wait is unchanged."""
    clock = {"t": 0.0}
    monkeypatch.setattr(llama_cpp_mod.time, "monotonic", lambda: clock["t"])

    def silent_read(max_bytes, timeout = None):
        clock["t"] += timeout if timeout is not None else 0.0
        raise httpcore.ReadTimeout("slice timed out on silence")

    stream = _Obj()
    stream.read = silent_read

    # No timeout extension: wrapper falls back to httpcore's passed timeout.
    request = _Obj()
    request.extensions = {}
    response = _Obj()
    response.request = request

    wrapped_read = _install(response, clock, stream)

    with pytest.raises(httpcore.ReadTimeout):
        wrapped_read(65536, timeout = _PREFILL_TIMEOUT)

    assert clock["t"] >= _PREFILL_TIMEOUT * 0.9
