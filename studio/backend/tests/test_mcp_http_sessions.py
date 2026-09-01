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


def _protocol_error(code: int, message: str) -> Exception:
    """Build a JSON-RPC error exception for whichever mcp is installed.

    single-env/constraints.txt allows mcp>=1.24,<2, where the class is McpError
    and takes an ErrorData; mcp 2 renamed it MCPError and takes the fields
    directly. Production code reads whichever of the two names exists, so the
    tests have to be able to raise it under both."""
    import mcp.shared.exceptions as mcp_exceptions
    from mcp.types import ErrorData

    cls = getattr(mcp_exceptions, "MCPError", None) or mcp_exceptions.McpError
    try:
        return cls(code = code, message = message)
    except TypeError:
        return cls(ErrorData(code = code, message = message))


def _settled(client, expected: int = 1, timeout: float = 10.0) -> int:
    """Wait out an asynchronous close.

    A discarded session is closed by the cleanup worker rather than on the
    request thread, so its close lands just after the call returns."""
    deadline = time.monotonic() + timeout
    while client.exited < expected and time.monotonic() < deadline:
        time.sleep(0.005)
    return client.exited


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
        self.probe_delay = 0.0
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
        if self.probe_delay:
            await asyncio.sleep(self.probe_delay)
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

    async def call_tool(
        self,
        name: str,
        args: dict,
        raise_on_error: bool = True,
    ):
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
        mcp_client,
        "_client",
        lambda url, headers, use_oauth = False: RecordingClient(url, headers, use_oauth),
    )
    yield RecordingClient.instances
    close_mcp_sessions()


def _call(
    url,
    name = "t",
    args = None,
    **kw,
):
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
    assert all(c.entered == 1 and _settled(c) == 1 for c in clients)
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
        mcp_client,
        "_client",
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
        mcp_client,
        "_client",
        lambda url, headers, use_oauth = False: _slow_connect(url, headers, use_oauth, 0.8),
    )
    out = _call(STDIO_URL, scope = SCOPE, timeout = 5.0)
    assert "timed out connecting" in out, out


def test_connect_timeout_reports_the_window_that_expired(monkeypatch, clients):
    """It used to report the caller's timeout even when the much tighter connect
    cap was what fired, which sends anyone debugging it to the wrong knob."""
    monkeypatch.setattr(mcp_client, "_STDIO_CONNECT_TIMEOUT", 0.3)
    monkeypatch.setattr(
        mcp_client,
        "_client",
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
    def _client(
        url,
        headers,
        use_oauth = False,
    ):
        c = _slow_connect(url, headers, use_oauth, 0.25)
        c.call_delay = 0.25
        return c

    monkeypatch.setattr(mcp_client, "_client", _client)
    out = _call(HTTP_URL, scope = SCOPE, timeout = 0.35)
    assert "timed out" in out, out


def test_unlimited_timeout_stays_unlimited(monkeypatch, clients):
    monkeypatch.setattr(mcp_client, "_STDIO_CONNECT_TIMEOUT", 0.05)
    monkeypatch.setattr(
        mcp_client,
        "_client",
        lambda url, headers, use_oauth = False: _slow_connect(url, headers, use_oauth, 0.3),
    )
    assert _call(HTTP_URL, scope = SCOPE, timeout = None) == "call-1"


# --------------------------------------------------------------------------
# Concurrency
# --------------------------------------------------------------------------


def _parallel(
    url,
    scopes,
    delay = 0.4,
):
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

    def _client(
        url,
        headers,
        use_oauth = False,
    ):
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

    def _client(
        url,
        headers,
        use_oauth = False,
    ):
        c = RecordingClient(url, headers, use_oauth)
        c.call_delay = 0.3
        return c

    monkeypatch.setattr(mcp_client, "_client", _client)
    out, _ = _parallel(STDIO_URL, [SCOPE, SCOPE], delay = 0.3)
    assert len(out) == 2
    assert len(clients) == 1
    assert clients[0].max_live == 1, "stdio calls overlapped on one subprocess"


def test_calls_in_different_chats_run_concurrently(monkeypatch, clients):
    def _client(
        url,
        headers,
        use_oauth = False,
    ):
        c = RecordingClient(url, headers, use_oauth)
        c.call_delay = 0.4
        return c

    monkeypatch.setattr(mcp_client, "_client", _client)
    out, elapsed = _parallel(HTTP_URL, [SCOPE, SCOPE_B])
    assert len(out) == 2
    assert len(clients) == 2
    assert elapsed < 0.75, f"different chats were serialized: {elapsed:.2f}s"


def test_concurrent_first_calls_publish_one_session(monkeypatch, clients):
    def _client(
        url,
        headers,
        use_oauth = False,
    ):
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
            target = lambda: out.append(_call(HTTP_URL, scope = SCOPE, config_check = config_check))
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

    async def _boom(
        name,
        args,
        raise_on_error = True,
    ):
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

    async def _tool_error(
        name,
        args,
        raise_on_error = True,
    ):
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
    assert _settled(clients[0]) == 1


def test_a_concurrent_checkout_cannot_cancel_another_borrowers_recheck(monkeypatch, clients):
    """Two HTTP borrowers now run at once, so the idle gap has to belong to the
    borrower. Held on the session, the second checkout would overwrite the first
    one's long gap with a near-zero one and talk it out of proving a session that
    really had gone stale."""
    monkeypatch.setattr(mcp_client, "_HTTP_IDLE_RECHECK", 0.0)

    def _client(
        url,
        headers,
        use_oauth = False,
    ):
        c = RecordingClient(url, headers, use_oauth)
        c.call_delay = 0.3
        return c

    monkeypatch.setattr(mcp_client, "_client", _client)
    _call(HTTP_URL, scope = SCOPE)  # connect and publish
    out, _ = _parallel(HTTP_URL, [SCOPE, SCOPE])
    assert len(out) == 2
    # Both were reused after an idle gap, so both must have proved the session.
    assert clients[0].probes == 2, f"a borrower skipped its recheck: {clients[0].probes}"


def test_a_second_borrower_still_proves_a_session_that_went_idle(monkeypatch, clients):
    """last_used is refreshed at checkout, so a borrower arriving while the first
    one's probe is still outstanding would see a near-zero gap and dispatch on a
    session nobody has proved yet. If the server expired it, that user's call is
    the thing that finds out, and it cannot be replayed."""
    monkeypatch.setattr(mcp_client, "_HTTP_IDLE_RECHECK", 0.2)
    _call(HTTP_URL, scope = SCOPE)
    client = clients[0]
    probes = {"n": 0}
    started = threading.Event()
    real_list = client.list_tools_mcp

    async def gated():
        probes["n"] += 1
        if probes["n"] == 1:
            started.set()
            await asyncio.sleep(0.5)  # hold the first probe open
        return await real_list()

    client.list_tools_mcp = gated
    time.sleep(0.3)  # past the recheck threshold
    out: list[str] = []
    first = threading.Thread(target = lambda: out.append(_call(HTTP_URL, scope = SCOPE)))
    first.start()
    assert started.wait(10), "the first borrower never began its probe"
    out.append(_call(HTTP_URL, scope = SCOPE))
    first.join(30)
    assert len(out) == 2 and not any(r.startswith("Error:") for r in out), out
    assert probes["n"] == 2, "the second borrower dispatched on an unproven session"


def test_closing_many_sessions_does_not_run_serially(monkeypatch, clients):
    """A popular HTTP server holds a session per chat, and close runs on the
    request thread during an edit or delete."""
    closes = []

    class SlowExit(RecordingClient):
        async def __aexit__(self, *exc):
            closes.append(time.monotonic())
            await asyncio.sleep(0.4)
            return await super().__aexit__(*exc)

    monkeypatch.setattr(
        mcp_client,
        "_client",
        lambda url, headers, use_oauth = False: SlowExit(url, headers, use_oauth),
    )
    for i in range(6):
        _call(HTTP_URL, scope = f"chat-{i}")
    started = time.monotonic()
    close_mcp_sessions()
    elapsed = time.monotonic() - started
    assert len(closes) == 6
    assert elapsed < 6 * 0.4 * 0.75, f"closes ran serially: {elapsed:.2f}s"


def test_a_slow_but_live_idle_session_survives_the_recheck(monkeypatch, clients):
    """The recheck exists to catch a session the server dropped. A server that is
    merely slow to answer tools/list has not dropped anything, and retiring it
    there would discard exactly the state this cache is for."""
    monkeypatch.setattr(mcp_client, "_HTTP_IDLE_RECHECK", 0.0)
    monkeypatch.setattr(mcp_client, "_SESSION_LIVENESS_TIMEOUT", 0.2)
    _call(HTTP_URL, scope = SCOPE)
    clients[0].probe_delay = 0.6  # answers, but well past the probe window
    assert _call(HTTP_URL, scope = SCOPE, timeout = 30.0) == "call-2"
    assert len(clients) == 1, "a slow probe retired a healthy session"


def test_a_tight_deadline_skips_the_probe_and_still_runs_the_tool(monkeypatch, clients):
    """The recheck is there to save someone a failed call, so it must not become
    the reason one fails. tool_call_timeout goes down to 1s, and a probe that
    quietly eats the whole budget leaves nothing to dispatch with."""
    monkeypatch.setattr(mcp_client, "_HTTP_IDLE_RECHECK", 0.0)
    monkeypatch.setattr(mcp_client, "_SESSION_LIVENESS_TIMEOUT", 0.5)
    _call(HTTP_URL, scope = SCOPE)
    clients[0].probe_delay = 1.5
    clients[0].call_delay = 0.2
    assert _call(HTTP_URL, scope = SCOPE, timeout = 0.8) == "call-2"
    assert clients[0].probes == 0, "the probe ran with no budget left for the call"


def test_evicting_another_scope_does_not_run_on_the_callers_deadline(monkeypatch, clients):
    """Whoever happens to trip the cap is mid tool call. The victim belongs to a
    different chat and nobody is waiting on it, so its teardown must not be
    charged to that caller's budget."""
    monkeypatch.setattr(mcp_client, "_MAX_SESSIONS", 1)
    closed = threading.Event()

    class SlowExit(RecordingClient):
        async def __aexit__(self, *exc):
            await asyncio.sleep(1.5)
            out = await super().__aexit__(*exc)
            closed.set()
            return out

    monkeypatch.setattr(
        mcp_client,
        "_client",
        lambda url, headers, use_oauth = False: SlowExit(url, headers, use_oauth),
    )
    _call(HTTP_URL, scope = SCOPE)  # fills the cache
    started = time.monotonic()
    assert _call(HTTP_URL, scope = SCOPE_B) == "call-1"
    elapsed = time.monotonic() - started
    assert elapsed < 0.5, f"the caller paid for an unrelated eviction: {elapsed:.2f}s"
    assert closed.wait(10), "the evicted session was never closed"


def test_a_json_rpc_error_keeps_the_chats_session(monkeypatch, clients):
    """A FastMCP server answers an unknown tool with a result carrying is_error,
    but the spec also lets a server report it as a protocol error and non-FastMCP
    servers do. Receiving that reply proves the connection works, so discarding
    the session would throw away the chat's server-side state over a tool name
    the model got wrong."""
    class ProtocolError(RecordingClient):
        async def call_tool(
            self,
            name: str,
            args: dict,
            raise_on_error: bool = True,
        ):
            if name == "nope":
                raise _protocol_error(-32602, "Unknown tool: nope")
            return await super().call_tool(name, args, raise_on_error)

    monkeypatch.setattr(
        mcp_client,
        "_client",
        lambda url, headers, use_oauth = False: ProtocolError(url, headers, use_oauth),
    )
    _call(HTTP_URL, scope = SCOPE)
    assert _call(HTTP_URL, "nope", scope = SCOPE).startswith("Error:")
    assert _call(HTTP_URL, scope = SCOPE) == "call-2"
    assert len(clients) == 1, "a protocol error discarded the session"
    # Kept, but no longer taken on trust: the next call proves it first, in case
    # the error was the server saying it no longer knows this session.
    assert clients[0].probes == 1


def test_a_failed_session_is_not_closed_on_the_retry_budget(monkeypatch, clients):
    """The session that fails the pre-dispatch probe is the one most likely to
    hang on close, and the caller still has a reconnect and a retry to do on the
    same deadline. Paying for its teardown first is what leaves the retry with
    nothing."""
    monkeypatch.setattr(mcp_client, "_HTTP_IDLE_RECHECK", 0.0)

    class HangingExit(RecordingClient):
        async def __aexit__(self, *exc):
            if self.probe_error:  # only the session that failed its probe
                await asyncio.sleep(1.5)
            return await super().__aexit__(*exc)

    monkeypatch.setattr(
        mcp_client,
        "_client",
        lambda url, headers, use_oauth = False: HangingExit(url, headers, use_oauth),
    )
    _call(HTTP_URL, scope = SCOPE)
    clients[0].probe_error = True  # the server dropped it while it sat idle
    started = time.monotonic()
    assert _call(HTTP_URL, scope = SCOPE) == "call-1"
    elapsed = time.monotonic() - started
    assert len(clients) == 2
    assert elapsed < 1.0, f"the retry waited for the dead session to close: {elapsed:.2f}s"
    assert _settled(clients[0]) == 1


def test_a_synchronous_close_waits_for_work_already_started(monkeypatch, clients):
    """close_mcp_sessions promises a server edit, and atexit, that the teardown
    has happened. Draining the queue does not recall the session the worker had
    already picked up."""
    monkeypatch.setattr(mcp_client, "_MAX_SESSIONS", 1)
    gate = threading.Event()

    class SlowExit(RecordingClient):
        slow = False

        async def __aexit__(self, *exc):
            if self.slow:
                gate.set()
                await asyncio.sleep(0.6)
            return await super().__aexit__(*exc)

    monkeypatch.setattr(
        mcp_client,
        "_client",
        lambda url, headers, use_oauth = False: SlowExit(url, headers, use_oauth),
    )
    _call(HTTP_URL, scope = SCOPE)
    victim = clients[0]
    # Only the evicted session is slow, so the assertion cannot be satisfied by
    # close_mcp_sessions happening to take just as long on the others.
    victim.slow = True
    _call(HTTP_URL, scope = SCOPE_B)  # evicts the first, worker picks it up
    assert gate.wait(10), "the worker never started on the evicted session"
    close_mcp_sessions()
    assert victim.exited == 1, "close_mcp_sessions returned mid-teardown"


def test_a_surviving_call_does_not_pay_for_the_retirement(monkeypatch, clients):
    """Parallel borrowers share one session, so one transport failure retires it
    while another call is still succeeding. That makes the survivor the last
    borrower, and its result is already waiting on this thread when the close
    would run."""
    closing = threading.Event()

    class SlowExit(RecordingClient):
        async def __aexit__(self, *exc):
            closing.set()
            await asyncio.sleep(1.5)
            return await super().__aexit__(*exc)

    monkeypatch.setattr(
        mcp_client,
        "_client",
        lambda url, headers, use_oauth = False: SlowExit(url, headers, use_oauth),
    )
    _call(HTTP_URL, scope = SCOPE)
    session = next(iter(mcp_client._mcp_sessions.values()))
    client = clients[0]
    client.call_delay = 0.3

    out: list[str] = []
    slow = threading.Thread(target = lambda: out.append(_call(HTTP_URL, scope = SCOPE)))
    slow.start()
    while session.in_flight < 1:
        time.sleep(0.01)
    # A sibling borrower's transport error retires the session under it.
    mcp_client._drop_session(next(iter(mcp_client._mcp_sessions)), session)
    started = time.monotonic()
    slow.join(30)
    elapsed = time.monotonic() - started
    assert out == ["call-2"], out
    assert elapsed < 1.0, f"the surviving call waited on the close: {elapsed:.2f}s"
    assert closing.wait(10), "the retired session was never closed"


def test_evictions_do_not_spawn_a_thread_each(monkeypatch, clients):
    """A run of new chat scopes against a server that hangs on shutdown would
    otherwise leave a cleanup thread per eviction alive for the close timeout."""
    monkeypatch.setattr(mcp_client, "_MAX_SESSIONS", 1)
    release = threading.Event()

    class HangingExit(RecordingClient):
        async def __aexit__(self, *exc):
            await asyncio.get_running_loop().run_in_executor(None, release.wait, 30)
            return await super().__aexit__(*exc)

    monkeypatch.setattr(
        mcp_client,
        "_client",
        lambda url, headers, use_oauth = False: HangingExit(url, headers, use_oauth),
    )
    try:
        before = threading.active_count()
        for i in range(12):  # 11 evictions, all of them stuck in __aexit__
            _call(HTTP_URL, scope = f"chat-{i}")
        # Not a total thread count: a transport stuck in __aexit__ keeps its own
        # session loop thread alive whatever closes it. What must stay bounded is
        # the cleanup machinery itself.
        cleanup = [t for t in threading.enumerate() if t.name in ("mcp-cleanup", "mcp-evict")]
        assert len(cleanup) <= 1, f"a cleanup thread per eviction: {len(cleanup)}"
        assert threading.active_count() >= before
    finally:
        release.set()


def test_a_json_rpc_error_from_the_probe_keeps_the_session(monkeypatch, clients):
    """The probe asks whether the server is still there. A rate limit or a
    permission rule on tools/list answers that question with a yes, so treating
    it as a dead transport would lose the chat's state over a reply that proves
    the connection works."""
    monkeypatch.setattr(mcp_client, "_HTTP_IDLE_RECHECK", 0.0)

    class ProbeRefused(RecordingClient):
        async def list_tools_mcp(self):
            self.probes += 1
            raise _protocol_error(-32000, "Rate limit exceeded")

    monkeypatch.setattr(
        mcp_client,
        "_client",
        lambda url, headers, use_oauth = False: ProbeRefused(url, headers, use_oauth),
    )
    _call(HTTP_URL, scope = SCOPE)
    assert _call(HTTP_URL, scope = SCOPE) == "call-2"
    assert len(clients) == 1, "a protocol error on the probe replaced the session"
    assert clients[0].probes == 1


def test_the_queue_of_pending_closes_is_bounded(monkeypatch, clients):
    """One worker bounds the cleanup threads but not the queue: a server that
    hangs on shutdown is closed slower than new chats evict, and every session
    waiting in the queue still holds its own loop thread and connection."""
    monkeypatch.setattr(mcp_client, "_MAX_SESSIONS", 1)
    monkeypatch.setattr(mcp_client, "_MAX_PENDING_CLOSES", 2)
    release = threading.Event()
    depths: list[int] = []

    class HangingExit(RecordingClient):
        async def __aexit__(self, *exc):
            await asyncio.get_running_loop().run_in_executor(None, release.wait, 30)
            return await super().__aexit__(*exc)

    monkeypatch.setattr(
        mcp_client,
        "_client",
        lambda url, headers, use_oauth = False: HangingExit(url, headers, use_oauth),
    )
    try:
        for i in range(6):  # 5 evictions, none of which can finish closing
            threading.Thread(target = _call, args = (HTTP_URL,), kwargs = {"scope": f"c{i}"}).start()
            deadline = time.monotonic() + 5.0
            while len(mcp_client._mcp_cleanup_queue) < min(i, 2) and time.monotonic() < deadline:
                time.sleep(0.01)
            depths.append(len(mcp_client._mcp_cleanup_queue))
        assert max(depths) <= 2, f"the queue grew past the bound: {depths}"
    finally:
        release.set()
        for t in threading.enumerate():
            if t.name.startswith("mcp-") and t is not threading.current_thread():
                t.join(5)


def test_closing_many_sessions_does_not_spawn_a_thread_each(monkeypatch, clients):
    """The cache is allowed to overshoot _MAX_SESSIONS while every session in it
    is busy, so the list close_mcp_sessions is handed has no fixed length, and a
    shutdown is the worst moment to ask for an unbounded number of threads."""
    monkeypatch.setattr(mcp_client, "_MAX_CLOSE_THREADS", 3)
    release = threading.Event()
    live = []
    lock = threading.Lock()

    class HangingExit(RecordingClient):
        async def __aexit__(self, *exc):
            with lock:
                live.append(len([t for t in threading.enumerate() if t.name == "mcp-close"]))
            await asyncio.get_running_loop().run_in_executor(None, release.wait, 30)
            return await super().__aexit__(*exc)

    monkeypatch.setattr(
        mcp_client,
        "_client",
        lambda url, headers, use_oauth = False: HangingExit(url, headers, use_oauth),
    )
    for i in range(9):
        _call(HTTP_URL, scope = f"chat-{i}")
    closer = threading.Thread(target = close_mcp_sessions)
    closer.start()
    deadline = time.monotonic() + 10.0
    while len(live) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        assert max(live) <= 3, f"one close thread per session: {live}"
        assert len(live) == 3, f"only {len(live)} closes started; the fan-out stalled"
    finally:
        release.set()
        closer.join(60)


def test_a_slow_probe_still_condemns_a_dirty_session(monkeypatch, clients):
    """The counterpart that must not change: a session whose last call was
    abandoned is under suspicion, so silence within the window condemns it."""
    monkeypatch.setattr(mcp_client, "_SESSION_LIVENESS_TIMEOUT", 0.2)
    _call(HTTP_URL, scope = SCOPE)
    clients[0].probe_delay = 0.6
    mcp_client._mcp_sessions[next(iter(mcp_client._mcp_sessions))].dirty = True
    assert _call(HTTP_URL, scope = SCOPE, timeout = 30.0) == "call-1"
    assert len(clients) == 2, "a wedged session was reused"


def test_a_failed_session_is_uncached_before_the_borrow_is_released(clients):
    """HTTP callers do not queue on call_lock, so between releasing the borrow and
    dropping the key another same-scope call could check the broken transport out
    and dispatch on it."""
    _call(HTTP_URL, scope = SCOPE)
    key = next(iter(mcp_client._mcp_sessions))
    seen = {}
    real_release = mcp_client._release_session

    def watching_release(session, **kw):
        # Whatever a concurrent caller could observe at this instant.
        seen["cached"] = mcp_client._mcp_sessions.get(key) is session
        seen["defunct"] = session.defunct
        return real_release(session, **kw)

    async def _boom(name, args, raise_on_error = True):
        raise RuntimeError("stream closed")

    clients[0].call_tool = _boom
    mcp_client._release_session = watching_release
    try:
        assert _call(HTTP_URL, scope = SCOPE).startswith("Error:")
    finally:
        mcp_client._release_session = real_release
    assert seen["cached"] is False, "the failed session was still checkout-able"
    assert seen["defunct"] is True


def test_closing_sessions_works_during_interpreter_exit():
    """close_mcp_sessions is the atexit handler. Python tears the
    ThreadPoolExecutor machinery down before normal atexit callbacks, so a pooled
    close raises there and the cleanup aborts with stdio subprocesses still up.

    Run in a child process: the failure only exists at real interpreter exit."""
    import subprocess
    import textwrap

    script = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, %r)
        from types import SimpleNamespace
        from core.inference import mcp_client

        closed = []

        class S:
            def __init__(self, n): self.n = n
            def close(self): closed.append(self.n)

        # More than one, so the parallel path is taken.
        import atexit
        atexit.register(lambda: print("CLOSED:" + ",".join(sorted(closed))))
        atexit.register(mcp_client._close_all, [S("a"), S("b"), S("c")])
        """
    ) % (_BACKEND_DIR,)
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output = True, text = True, timeout = 120
    )
    assert "CLOSED:a,b,c" in proc.stdout, f"stdout={proc.stdout!r} stderr={proc.stderr[-2000:]!r}"
    assert "can't register atexit" not in proc.stderr, proc.stderr[-2000:]
    assert "cannot schedule new futures" not in proc.stderr, proc.stderr[-2000:]


def test_a_fork_resets_the_inherited_cache(clients):
    """Only the forking thread survives, so every inherited session's loop thread
    is gone while its client still reports connected. A child that checked one
    out would wait on a loop that never runs."""
    if not hasattr(mcp_client.os, "register_at_fork"):
        pytest.skip("no register_at_fork on this platform")
    _call(HTTP_URL, scope = SCOPE)
    assert len(mcp_client._mcp_sessions) == 1
    # The real hook runs in a child that is about to exec or exit, so dropping the
    # entries is the whole point. Here it runs in the parent, where those objects
    # are live: hold on to them and close them by hand, or this test leaks a loop
    # thread and its descriptors into every test that follows.
    inherited = list(mcp_client._mcp_sessions.values())
    reaper_was_started = mcp_client._mcp_reaper_started
    try:
        mcp_client._reset_after_fork()
        assert mcp_client._mcp_sessions == {}
        assert mcp_client._mcp_key_locks == {}
        assert mcp_client._mcp_connects_in_flight == 0
        assert mcp_client._mcp_reaper_started is False
    finally:
        for session in inherited:
            session.close()
        # The parent's reaper thread outlived the call above; leaving the flag
        # False would start a second one on the next connect.
        mcp_client._mcp_reaper_started = reaper_was_started


def test_transport_dead_is_unknown_for_http():
    """Documents why the idle recheck exists at all: _is_session_dead and
    _connect_task are StdioTransport internals, absent from both HTTP transports
    on every fastmcp this repo supports."""
    from fastmcp.client.transports import SSETransport, StreamableHttpTransport
    for cls in (StreamableHttpTransport, SSETransport):
        transport = cls(url = "https://x.test/mcp")
        assert not hasattr(transport, "_is_session_dead")
        assert not hasattr(transport, "_connect_task")
        assert (
            mcp_client._transport_dead(SimpleNamespace(client = SimpleNamespace(transport = transport)))
            is False
        )
