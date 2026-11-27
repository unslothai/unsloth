# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#14298: an HTTP/SSE MCP tool call must run for the whole Max Tool Call Duration, not stop at the
SDK's 300s httpx read timeout. The e2e test shrinks that SDK default to 1s and runs a 2s tool against
a real local server, so it fails in seconds without the fix instead of after five minutes."""

from __future__ import annotations

import asyncio
import importlib
import json
import socket
import subprocess
import sys
import textwrap
import threading
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
    try:
        out = mcp_client.call_tool_sync(
            slow_server, None, "slow", {"seconds": 2}, timeout = 30, scope = scope
        )
    finally:
        mcp_client.close_mcp_sessions()  # the scoped case caches a session
    assert "slept 2" in out, out


def test_call_budget_still_enforced(slow_server):
    out = mcp_client.call_tool_sync(slow_server, None, "slow", {"seconds": 10}, timeout = 2)
    assert out.startswith("Error") and "2s" in out, out


# With the read cap lifted, a reply abandoned by a timed-out or stopped call stays open on the transport until the
# server closes it (SDKs that abort the POST leave nothing open). These pin the bounded cleanup that replaces the cap.


@pytest.fixture
def cached_scope():
    scope = f"test-{time.monotonic_ns()}"
    yield scope
    mcp_client.close_mcp_sessions()


def _cached_session(url, scope):
    return mcp_client._mcp_sessions[mcp_client._session_key(url, None, scope)]


def _call(
    url,
    scope,
    seconds,
    timeout = 30,
):
    return mcp_client.call_tool_sync(
        url, None, "slow", {"seconds": seconds}, timeout = timeout, scope = scope
    )


def _warm(url, scope):
    assert "slept 0" in _call(url, scope, 0)
    return _cached_session(url, scope)


def _open_post(
    session,
    age = 0.0,
    abandoned = True,
):
    token, tracker = object(), session.posts
    with tracker.lock:
        tracker.open.add(token)
        if abandoned:
            tracker.abandoned[token] = time.monotonic() - age
    return token


def _slot(tracker, key):
    slot = mcp_client._PostSlot(key)
    tracker.enqueue(slot)
    return slot


def test_redirected_post_stays_with_its_call(slow_server, cached_scope):
    # A 307/308 resends the same JSON-RPC request: the slot follows it to the final POST.
    tracker, key = _warm(slow_server, cached_scope).posts, ("tools/call", "slow")
    slot, hop, final = _slot(tracker, key), object(), object()
    tracker._sent(hop, key, 41)
    tracker._closed(hop)
    tracker._sent(final, key, 41)
    tracker.settle(slot, abandoned = True)
    assert list(tracker.abandoned) == [final]
    tracker.settle(slot, abandoned = False)
    tracker._closed(final)
    assert tracker.pending() is None and not tracker.bound


def test_only_call_replies_lose_the_read_cap():
    from mcp.shared._httpx_utils import create_mcp_http_client

    client = create_mcp_http_client()
    http = importlib.import_module(type(client).__module__.split(".")[0])
    seen = {}

    def handler(request):
        seen[json.loads(request.content)["method"]] = request.extensions["timeout"]["read"]
        return http.Response(200, json = {})

    client._transport, client._mounts = http.MockTransport(handler), {}
    client.timeout = type(client.timeout)(30.0, read = None)
    mcp_client._PostTracker().install(client)

    async def post(method):
        await client.post("http://mcp.test/mcp", json = {"jsonrpc": "2.0", "id": 1, "method": method})

    for method in ("tools/call", "resources/read", "tools/list", "initialize"):
        asyncio.run(post(method))
    assert seen["tools/call"] is None and seen["resources/read"] is None
    assert seen["tools/list"] == seen["initialize"] == mcp_client._ABANDONED_REPLY_TIMEOUT


def test_abandoned_call_is_bound_to_its_own_post(slow_server, cached_scope):
    # Call A is abandoned while a later call B to the same tool is still running: only A's POST may count.
    session = _warm(slow_server, cached_scope)
    assert session.posts.installed
    sent, real_sent = [], session.posts._sent
    session.posts._sent = lambda token, *a: (sent.append(token), real_sent(token, *a))
    a = threading.Thread(target = _call, args = (slow_server, cached_scope, 30, 1.0))
    a.start()
    time.sleep(0.3)
    b = threading.Thread(target = _call, args = (slow_server, cached_scope, 3))
    b.start()
    a.join()
    time.sleep(0.3)
    assert len(sent) == 2 and sent[1] in session.posts.open  # B is still running
    assert set(session.posts.abandoned) <= {sent[0]}
    b.join()


def test_retirement_deadline_starts_at_abandonment(slow_server, cached_scope, monkeypatch):
    # A reply that ran past the deadline before it was abandoned still gets the full cleanup window.
    monkeypatch.setattr(mcp_client, "_ABANDONED_REPLY_TIMEOUT", 0.5)
    session = _warm(slow_server, cached_scope)
    tracker, key = session.posts, ("tools/call", "slow")
    slot, token = _slot(tracker, key), object()
    tracker._sent(token, key, 1)
    time.sleep(0.6)
    session.note_abandoned(slot)
    assert not session.abandoned_overdue()
    time.sleep(0.6)
    assert session.abandoned_overdue()
    tracker._closed(token)
    assert not session.abandoned_overdue()


def test_abandonment_bookkeeping_is_thread_safe(slow_server, cached_scope):
    session = _warm(slow_server, cached_scope)
    tracker, key, errors = session.posts, ("tools/call", "slow"), []

    def churn():
        try:
            for _ in range(2000):
                slot, token = _slot(tracker, key), object()
                tracker._sent(token, key, id(token))
                session.note_abandoned(slot)
                session.abandoned_overdue()
                tracker._closed(token)
                tracker.settle(slot, abandoned = False)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target = churn) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and not tracker.abandoned and not tracker.waiting and not tracker.bound


def test_overdue_abandoned_reply_retires_the_session_without_probing(
    slow_server, cached_scope, monkeypatch
):
    monkeypatch.setattr(mcp_client, "_ABANDONED_REPLY_TIMEOUT", 0.2)
    stale = _warm(slow_server, cached_scope)
    _open_post(stale, age = 1)
    assert stale.abandoned_overdue()

    def _no_probe(*args, **kwargs):
        raise AssertionError("an overdue session must be closed, not probed")

    stale.dirty = True
    monkeypatch.setattr(mcp_client, "_session_responsive", _no_probe)
    assert "slept 0" in _call(slow_server, cached_scope, 0)
    assert stale.closed.is_set() and stale.client is None
    fresh = _cached_session(slow_server, cached_scope)
    assert fresh is not stale and fresh.posts.pending() is None


def test_too_many_abandoned_replies_retire_the_session(slow_server, cached_scope, monkeypatch):
    monkeypatch.setattr(mcp_client, "_MAX_ABANDONED_REPLIES", 2)
    stale = _warm(slow_server, cached_scope)
    _open_post(stale)
    assert not stale.abandoned_overdue()
    _open_post(stale)
    assert "slept 0" in _call(slow_server, cached_scope, 0)
    assert stale.closed.is_set()
    assert _cached_session(slow_server, cached_scope) is not stale


def test_reaper_closes_an_idle_session_with_an_overdue_reply(
    slow_server, cached_scope, monkeypatch
):
    monkeypatch.setattr(mcp_client, "_ABANDONED_REPLY_TIMEOUT", 0.2)
    stale = _warm(slow_server, cached_scope)
    token = _open_post(stale)
    mcp_client._reap_idle_sessions()  # not overdue yet, and far inside the idle TTL
    assert not stale.closed.is_set()
    stale.posts.abandoned[token] -= 1
    mcp_client._reap_idle_sessions()
    assert stale.closed.is_set()
    assert mcp_client._session_key(slow_server, None, cached_scope) not in mcp_client._mcp_sessions


def test_finished_abandoned_reply_keeps_the_session(slow_server, cached_scope, monkeypatch):
    # A Stop whose reply the server then finished leaks nothing, so the chat keeps its session (and server state).
    monkeypatch.setattr(mcp_client, "_ABANDONED_REPLY_TIMEOUT", 0.2)
    session = _warm(slow_server, cached_scope)
    assert _call(slow_server, cached_scope, 1, timeout = 0.3).startswith("Error")
    time.sleep(1.5)
    assert not session.abandoned_overdue() and session.posts.pending() is None
    assert "slept 0" in _call(slow_server, cached_scope, 0)
    assert _cached_session(slow_server, cached_scope) is session and not session.closed.is_set()
