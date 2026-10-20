# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#14298: an HTTP/SSE MCP tool call must run for the whole Max Tool Call Duration, not stop at the
SDK's 300s httpx read timeout. The e2e test shrinks that SDK default to 1s and runs a 2s tool against
a real local server, so it fails in seconds without the fix instead of after five minutes."""

from __future__ import annotations

import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import mcp_client

_SERVER = textwrap.dedent(
    """
    import asyncio, sys
    from fastmcp import FastMCP

    mcp = FastMCP("slow")

    @mcp.tool
    async def slow(seconds: float) -> str:
        await asyncio.sleep(seconds)
        return f"slept {seconds}"

    mcp.run(transport = "http", host = "127.0.0.1", port = int(sys.argv[1]), show_banner = False)
    """
)


@pytest.fixture
def slow_server():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, "-c", _SERVER, str(port)])
    try:
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), timeout = 0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.terminate()
        proc.wait(timeout = 10)


@pytest.mark.parametrize("restricted", [False, True])
def test_factory_lifts_only_the_read_cap(monkeypatch, restricted):
    monkeypatch.setattr(mcp_client, "_managed_mcp_restricted", lambda: restricted)
    from mcp.shared._httpx_utils import create_mcp_http_client

    timeout = type(create_mcp_http_client().timeout)(5.0, read = 300.0)  # what sse_client passes
    for kwargs in ({}, {"timeout": timeout, "follow_redirects": True}):
        t = mcp_client._tool_http_client_factory(**kwargs).timeout
        assert t.read is None
        assert t.connect == (5.0 if kwargs else 30.0)


@pytest.mark.parametrize("scope", [None, "chat-1"])  # one-shot client and cached session
def test_tool_call_outlives_sdk_read_timeout(slow_server, monkeypatch, scope):
    import mcp.shared._httpx_utils as httpx_utils

    monkeypatch.setattr(httpx_utils, "MCP_DEFAULT_SSE_READ_TIMEOUT", 1.0)
    out = mcp_client.call_tool_sync(
        slow_server, None, "slow", {"seconds": 2}, timeout = 30, scope = scope
    )
    assert "slept 2" in out, out


def test_call_budget_still_enforced(slow_server):
    out = mcp_client.call_tool_sync(slow_server, None, "slow", {"seconds": 10}, timeout = 2)
    assert out.startswith("Error") and "2s" in out, out


# With the read cap lifted, a reply abandoned by a timed-out or stopped call stays open on the transport until the
# server closes it. These pin the bounded cleanup that replaces the cap: the cached session is retired once an
# abandoned reply is overdue or too many have piled up, without waiting on a liveness probe that could not see them.


@pytest.fixture
def cached_scope():
    scope = f"test-{time.monotonic_ns()}"
    yield scope
    mcp_client.close_mcp_sessions()


def _cached_session(url, scope):
    return mcp_client._mcp_sessions[mcp_client._session_key(url, None, scope)]


def _abandon_call(url, scope):
    out = mcp_client.call_tool_sync(url, None, "slow", {"seconds": 30}, timeout = 0.5, scope = scope)
    assert out.startswith("Error"), out


def test_abandoned_reply_is_recorded_on_the_cached_session(slow_server, cached_scope):
    _abandon_call(slow_server, cached_scope)
    session = _cached_session(slow_server, cached_scope)
    assert session.dirty and session.abandoned == 1 and session.abandoned_at is not None
    assert not session.abandoned_overdue()


def test_overdue_abandoned_reply_retires_the_session_without_probing(
    slow_server, cached_scope, monkeypatch
):
    monkeypatch.setattr(mcp_client, "_ABANDONED_REPLY_TIMEOUT", 0.2)
    _abandon_call(slow_server, cached_scope)
    stale = _cached_session(slow_server, cached_scope)
    time.sleep(0.3)
    assert stale.abandoned_overdue()

    def _no_probe(*args, **kwargs):
        raise AssertionError("an overdue session must be closed, not probed")

    monkeypatch.setattr(mcp_client, "_session_responsive", _no_probe)
    out = mcp_client.call_tool_sync(
        slow_server, None, "slow", {"seconds": 0}, timeout = 30, scope = cached_scope
    )
    assert "slept 0" in out, out
    assert stale.closed.is_set() and stale.client is None
    fresh = _cached_session(slow_server, cached_scope)
    assert fresh is not stale and fresh.abandoned_at is None


def test_too_many_abandoned_replies_retire_the_session(slow_server, cached_scope, monkeypatch):
    monkeypatch.setattr(mcp_client, "_MAX_ABANDONED_REPLIES", 2)
    _abandon_call(slow_server, cached_scope)
    stale = _cached_session(slow_server, cached_scope)
    _abandon_call(slow_server, cached_scope)
    assert _cached_session(slow_server, cached_scope) is stale and stale.abandoned == 2
    out = mcp_client.call_tool_sync(
        slow_server, None, "slow", {"seconds": 0}, timeout = 30, scope = cached_scope
    )
    assert "slept 0" in out, out
    assert stale.closed.is_set()
    assert _cached_session(slow_server, cached_scope) is not stale


def test_reaper_closes_an_idle_session_with_an_overdue_reply(
    slow_server, cached_scope, monkeypatch
):
    monkeypatch.setattr(mcp_client, "_ABANDONED_REPLY_TIMEOUT", 0.2)
    _abandon_call(slow_server, cached_scope)
    stale = _cached_session(slow_server, cached_scope)
    mcp_client._reap_idle_sessions()  # not overdue yet, and far inside the idle TTL
    assert not stale.closed.is_set()
    time.sleep(0.3)
    mcp_client._reap_idle_sessions()
    assert stale.closed.is_set()
    assert mcp_client._session_key(slow_server, None, cached_scope) not in mcp_client._mcp_sessions
