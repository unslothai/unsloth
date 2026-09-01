# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared HTTP MCP sessions: routing, identity, timeout budget, OAuth safety and
concurrency.

test_mcp_stdio_sessions.py owns the machinery these share; this file owns what is
specific to HTTP now that it is cached too, and the regressions that came with it.
Races use explicit barriers rather than sleeps so they cannot pass by luck.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import mcp_client
from core.inference.mcp_client import call_tool_sync, close_mcp_sessions

STDIO_URL = "npx fake-stateful-server"
HTTP_URL = "https://mcp.example.test/mcp"
HTTP_URL_2 = "https://other.example.test/mcp"
SSE_URL = "https://mcp.example.test/sse"
SCOPE = "s=sess1:t=threadA"
SCOPE_B = "s=sess1:t=threadB"


def _result(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        content = [SimpleNamespace(type = "text", text = text)],
        is_error = False,
        structured_content = None,
    )


class RecordingClient:
    """Records what _client() was actually given. The stdio-session fixture drops
    headers and use_oauth, which is exactly where a credential bug would hide."""

    instances: list["RecordingClient"] = []

    def __init__(self, url: str, headers, use_oauth: bool):
        self.url = url
        self.headers = headers
        self.use_oauth = use_oauth
        self.entered = 0
        self.exited = 0
        self.calls: list[tuple[str, dict]] = []
        self.connected = False
        self.probes = 0
        self.probe_error = False
        self.connect_delay = 0.0
        self.call_delay = 0.0
        self.live = 0
        self.max_live = 0
        self._lock = threading.Lock()
        self.transport = SimpleNamespace()  # no _is_session_dead: real HTTP has none
        RecordingClient.instances.append(self)

    async def list_tools_mcp(self):
        self.probes += 1
        if self.probe_error:
            raise RuntimeError("session expired")
        return SimpleNamespace(tools = [])

    async def __aenter__(self):
        if self.connect_delay:
            await asyncio.sleep(self.connect_delay)
        self.entered += 1
        self.connected = True
        return self

    async def __aexit__(self, *exc):
        self.exited += 1
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    async def call_tool(self, name: str, args: dict, raise_on_error: bool = True):
        with self._lock:
            self.live += 1
            self.max_live = max(self.max_live, self.live)
        try:
            if self.call_delay:
                await asyncio.sleep(self.call_delay)
            with self._lock:
                self.calls.append((name, args))
                return _result(f"call-{len(self.calls)}")
        finally:
            with self._lock:
                self.live -= 1


@pytest.fixture
def clients(monkeypatch):
    RecordingClient.instances = []
    monkeypatch.setattr(
        mcp_client, "_client",
        lambda url, headers, use_oauth = False: RecordingClient(url, headers, use_oauth),
    )
    yield RecordingClient.instances
    close_mcp_sessions()


def _call(url, name = "t", args = None, **kw):
    kw.setdefault("timeout", 30.0)
    return call_tool_sync(url, kw.pop("headers", None), name, args or {}, **kw)


# --------------------------------------------------------------------------
# Routing and identity
# --------------------------------------------------------------------------


def test_scoped_http_reuses_one_client(clients):
    assert _call(HTTP_URL, scope = SCOPE) == "call-1"
    assert _call(HTTP_URL, scope = SCOPE) == "call-2"
    assert len(clients) == 1
    assert clients[0].entered == 1 and clients[0].exited == 0


def test_sse_urls_are_cached_too(clients):
    _call(SSE_URL, scope = SCOPE)
    _call(SSE_URL, scope = SCOPE)
    assert len(clients) == 1


def test_different_scopes_do_not_share(clients):
    _call(HTTP_URL, scope = SCOPE)
    _call(HTTP_URL, scope = SCOPE_B)
    assert len(clients) == 2


def test_different_urls_do_not_share(clients):
    _call(HTTP_URL, scope = SCOPE)
    _call(HTTP_URL_2, scope = SCOPE)
    assert len(clients) == 2


def test_different_header_values_do_not_share(clients):
    _call(HTTP_URL, scope = SCOPE, headers = {"Authorization": "Bearer a"})
    _call(HTTP_URL, scope = SCOPE, headers = {"Authorization": "Bearer b"})
    assert len(clients) == 2
    assert clients[0].headers != clients[1].headers


def test_headers_reach_the_client_unchanged(clients):
    _call(HTTP_URL, scope = SCOPE, headers = {"Authorization": "Bearer a"})
    assert clients[0].headers == {"Authorization": "Bearer a"}


def test_header_order_does_not_split_the_session(clients):
    _call(HTTP_URL, scope = SCOPE, headers = {"A": "1", "B": "2"})
    _call(HTTP_URL, scope = SCOPE, headers = {"B": "2", "A": "1"})
    assert len(clients) == 1


def test_empty_headers_and_none_share_a_key(clients):
    _call(HTTP_URL, scope = SCOPE, headers = None)
    _call(HTTP_URL, scope = SCOPE, headers = {})
    assert len(clients) == 1


def test_unscoped_http_stays_one_shot(clients):
    _call(HTTP_URL, scope = None)
    _call(HTTP_URL, scope = None)
    assert len(clients) == 2
    assert all(c.entered == 1 and c.exited == 1 for c in clients)
    assert mcp_client._mcp_sessions == {}


def test_empty_scope_is_treated_as_unscoped(clients):
    _call(HTTP_URL, scope = "")
    _call(HTTP_URL, scope = "")
    assert len(clients) == 2
    assert mcp_client._mcp_sessions == {}


def test_oauth_http_stays_one_shot(clients):
    _call(HTTP_URL, scope = SCOPE, use_oauth = True)
    _call(HTTP_URL, scope = SCOPE, use_oauth = True)
    assert len(clients) == 2
    assert all(c.use_oauth for c in clients)
    assert mcp_client._mcp_sessions == {}


def test_a_shared_session_is_never_built_for_an_oauth_server():
    """Defence in depth: the routing above is what keeps OAuth out of the cache,
    so if that ever slips the session must refuse rather than talk to an OAuth
    server with no credentials."""
    with pytest.raises(ValueError):
        mcp_client._McpSession(HTTP_URL, None, use_oauth = True)


# --------------------------------------------------------------------------
# Timeout budget
# --------------------------------------------------------------------------


def test_http_connect_uses_the_whole_caller_budget(monkeypatch, clients):
    """Regression: routing HTTP through the shared cache must not import stdio's
    cold-start cap. Scaled down so the test costs a second, not a minute."""
    monkeypatch.setattr(mcp_client, "_STDIO_CONNECT_TIMEOUT", 0.3)
    monkeypatch.setattr(
        mcp_client, "_client",
        lambda url, headers, use_oauth = False: _slow_connect(url, headers, use_oauth, 0.8),
    )
    out = _call(HTTP_URL, scope = SCOPE, timeout = 5.0)
    assert out == "call-1", out


def _slow_connect(url, headers, use_oauth, delay):
    client = RecordingClient(url, headers, use_oauth)
    client.connect_delay = delay
    return client


def test_stdio_connect_keeps_its_cold_start_cap(monkeypatch, clients):
    monkeypatch.setattr(mcp_client, "_STDIO_CONNECT_TIMEOUT", 0.3)
    monkeypatch.setattr(
        mcp_client, "_client",
        lambda url, headers, use_oauth = False: _slow_connect(url, headers, use_oauth, 0.8),
    )
    out = _call(STDIO_URL, scope = SCOPE, timeout = 5.0)
    assert "timed out connecting" in out, out


def test_connect_timeout_reports_the_window_that_expired(monkeypatch, clients):
    """It used to report the caller's timeout even when the much tighter connect
    cap was what fired, which sends anyone debugging it to the wrong knob."""
    monkeypatch.setattr(mcp_client, "_STDIO_CONNECT_TIMEOUT", 0.3)
    monkeypatch.setattr(
        mcp_client, "_client",
        lambda url, headers, use_oauth = False: _slow_connect(url, headers, use_oauth, 5.0),
    )
    out = _call(STDIO_URL, scope = SCOPE, timeout = 9.0)
    assert "0.3s" in out and "9s" not in out, out


def test_total_deadline_still_bounds_a_slow_http_connect(clients):
    """The complement: giving HTTP the full budget must not remove the budget."""
    mcp_client._client = lambda url, headers, use_oauth = False: _slow_connect(
        url, headers, use_oauth, 1.0
    )
    out = _call(HTTP_URL, scope = SCOPE, timeout = 0.25)
    assert "timed out" in out, out


def test_connect_and_call_share_one_deadline(monkeypatch, clients):
    def _client(url, headers, use_oauth = False):
        c = _slow_connect(url, headers, use_oauth, 0.25)
        c.call_delay = 0.25
        return c

    monkeypatch.setattr(mcp_client, "_client", _client)
    out = _call(HTTP_URL, scope = SCOPE, timeout = 0.35)
    assert "timed out" in out, out


def test_unlimited_timeout_stays_unlimited(monkeypatch, clients):
    monkeypatch.setattr(mcp_client, "_STDIO_CONNECT_TIMEOUT", 0.05)
    monkeypatch.setattr(
        mcp_client, "_client",
        lambda url, headers, use_oauth = False: _slow_connect(url, headers, use_oauth, 0.3),
    )
    assert _call(HTTP_URL, scope = SCOPE, timeout = None) == "call-1"


# --------------------------------------------------------------------------
# Concurrency
# --------------------------------------------------------------------------


def _parallel(url, scopes, delay = 0.4):
    out: list[str] = []
    lock = threading.Lock()

    def run(scope):
        r = _call(url, "delayed", scope = scope)
        with lock:
            out.append(r)

    started = time.monotonic()
    threads = [threading.Thread(target = run, args = (s,)) for s in scopes]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    return out, time.monotonic() - started


def test_two_http_calls_in_one_chat_run_concurrently(monkeypatch, clients):
    """Regression: the shared session must not serialize HTTP. Every JSON-RPC
    message is its own POST and responses carry request ids, so there is nothing
    to interleave; before this, a parallel tool batch took twice as long."""
    def _client(url, headers, use_oauth = False):
        c = RecordingClient(url, headers, use_oauth)
        c.call_delay = 0.4
        return c

    monkeypatch.setattr(mcp_client, "_client", _client)
    out, elapsed = _parallel(HTTP_URL, [SCOPE, SCOPE])
    assert len(out) == 2
    assert len(clients) == 1, "the two calls should share one client"
    assert clients[0].max_live == 2, "the server never saw them overlap"
    assert elapsed < 0.75, f"calls were serialized: {elapsed:.2f}s"


def test_two_stdio_calls_in_one_chat_stay_serialized(monkeypatch, clients):
    """The counterpart that must not change: one subprocess is one ordered byte
    stream, so a browser or REPL must not see two calls interleaved."""
    def _client(url, headers, use_oauth = False):
        c = RecordingClient(url, headers, use_oauth)
        c.call_delay = 0.3
        return c

    monkeypatch.setattr(mcp_client, "_client", _client)
    out, _ = _parallel(STDIO_URL, [SCOPE, SCOPE], delay = 0.3)
    assert len(out) == 2
    assert len(clients) == 1
    assert clients[0].max_live == 1, "stdio calls overlapped on one subprocess"


def test_calls_in_different_chats_run_concurrently(monkeypatch, clients):
    def _client(url, headers, use_oauth = False):
        c = RecordingClient(url, headers, use_oauth)
        c.call_delay = 0.4
        return c

    monkeypatch.setattr(mcp_client, "_client", _client)
    out, elapsed = _parallel(HTTP_URL, [SCOPE, SCOPE_B])
    assert len(out) == 2
    assert len(clients) == 2
    assert elapsed < 0.75, f"different chats were serialized: {elapsed:.2f}s"


def test_concurrent_first_calls_publish_one_session(monkeypatch, clients):
    def _client(url, headers, use_oauth = False):
        return _slow_connect(url, headers, use_oauth, 0.3)

    monkeypatch.setattr(mcp_client, "_client", _client)
    out, _ = _parallel(HTTP_URL, [SCOPE, SCOPE])
    assert len(out) == 2
    assert len(clients) == 1, "a connect race opened two clients for one key"
    assert len(mcp_client._mcp_sessions) == 1


# --------------------------------------------------------------------------
# Stale configuration
# --------------------------------------------------------------------------


def test_oauth_flip_before_connect_blocks_dispatch(clients):
    """The window: the call read a non-OAuth row, the row flipped, and the route's
    close found nothing cached and no connect in flight, so it had no generation
    to bump. Only the config snapshot can reject this call."""
    row = {"is_enabled": 1, "url": HTTP_URL, "headers": None, "use_oauth": 0}

    def config_check() -> bool:
        return (
            bool(row["is_enabled"])
            and row["url"] == HTTP_URL
            and row["headers"] is None
            and bool(row["use_oauth"]) is False
        )

    reached = threading.Event()
    release = threading.Event()
    real_slot = mcp_client._connect_slot

    def paused_slot(url, headers):
        reached.set()
        release.wait(10)
        return real_slot(url, headers)

    mcp_client._connect_slot = paused_slot
    try:
        out: list[str] = []
        worker = threading.Thread(
            target = lambda: out.append(
                _call(HTTP_URL, scope = SCOPE, config_check = config_check)
            )
        )
        worker.start()
        assert reached.wait(10)
        row["use_oauth"] = 1
        close_mcp_sessions(HTTP_URL, None)
        release.set()
        worker.join(20)
    finally:
        mcp_client._connect_slot = real_slot

    assert not [c for c in clients if c.calls], "dispatched on a stale non-OAuth client"
    assert mcp_client._mcp_sessions == {}
    assert "updated or removed" in out[0], out


def test_oauth_flip_before_dispatch_blocks_a_cached_session(clients):
    row = {"use_oauth": 0}
    _call(HTTP_URL, scope = SCOPE, config_check = lambda: not row["use_oauth"])
    assert len(mcp_client._mcp_sessions) == 1
    row["use_oauth"] = 1
    out = _call(HTTP_URL, scope = SCOPE, config_check = lambda: not row["use_oauth"])
    assert "updated or removed" in out, out
    assert mcp_client._mcp_sessions == {}


def test_url_change_blocks_a_cached_session(clients):
    row = {"url": HTTP_URL}
    _call(HTTP_URL, scope = SCOPE, config_check = lambda: row["url"] == HTTP_URL)
    row["url"] = HTTP_URL_2
    out = _call(HTTP_URL, scope = SCOPE, config_check = lambda: row["url"] == HTTP_URL)
    assert "updated or removed" in out, out


def test_a_raising_config_check_fails_closed(clients):
    def boom() -> bool:
        raise RuntimeError("db gone")

    out = _call(HTTP_URL, scope = SCOPE, config_check = boom)
    assert "updated or removed" in out, out
    assert mcp_client._mcp_sessions == {}


# --------------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------------


def test_a_transport_failure_is_never_replayed(clients):
    """The tool may already have run on the server, so a retry could double a
    side effect. Drop the session; the next call reconnects."""
    _call(HTTP_URL, scope = SCOPE)
    attempts = []

    async def _boom(name, args, raise_on_error = True):
        attempts.append(name)
        raise RuntimeError("stream closed")

    clients[0].call_tool = _boom
    out = _call(HTTP_URL, scope = SCOPE)
    assert out.startswith("Error:"), out
    assert len(attempts) == 1, f"the failed tool was dispatched {len(attempts)} times"
    assert mcp_client._mcp_sessions == {}
    # The session is gone, so the next call reconnects rather than reusing it.
    assert _call(HTTP_URL, scope = SCOPE) == "call-1"
    assert len(clients) == 2


def test_a_tool_error_keeps_the_session(clients):
    from fastmcp.exceptions import ToolError

    _call(HTTP_URL, scope = SCOPE)

    async def _tool_error(name, args, raise_on_error = True):
        raise ToolError("nope")

    clients[0].call_tool = _tool_error
    assert _call(HTTP_URL, scope = SCOPE).startswith("Error:")
    assert len(mcp_client._mcp_sessions) == 1
    assert len(clients) == 1


def test_an_expired_idle_http_session_is_replaced_before_dispatch(monkeypatch, clients):
    """A server may drop an HTTP session whenever it likes and no HTTP transport
    exposes a liveness probe, so the only honest check is to ask it."""
    monkeypatch.setattr(mcp_client, "_HTTP_IDLE_RECHECK", 0.0)
    _call(HTTP_URL, scope = SCOPE)
    clients[0].probe_error = True
    assert _call(HTTP_URL, scope = SCOPE) == "call-1"
    assert len(clients) == 2
    assert clients[0].exited == 1


def test_transport_dead_is_unknown_for_http():
    """Documents why the idle recheck exists at all: _is_session_dead and
    _connect_task are StdioTransport internals, absent from both HTTP transports
    on every fastmcp this repo supports."""
    from fastmcp.client.transports import SSETransport, StreamableHttpTransport

    for cls in (StreamableHttpTransport, SSETransport):
        transport = cls(url = "https://x.test/mcp")
        assert not hasattr(transport, "_is_session_dead")
        assert not hasattr(transport, "_connect_task")
        assert mcp_client._transport_dead(SimpleNamespace(client = SimpleNamespace(
            transport = transport
        ))) is False
