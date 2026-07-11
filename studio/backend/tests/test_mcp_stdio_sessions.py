# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
from core.inference.mcp_client import call_tool_sync, close_stdio_sessions

STDIO_URL = "npx fake-stateful-server"
HTTP_URL = "https://mcp.example.test/mcp"


def _result(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        content = [SimpleNamespace(type = "text", text = text)],
        is_error = False,
        structured_content = None,
    )


class FakeClient:
    instances: list["FakeClient"] = []

    def __init__(self, url: str):
        self.url = url
        self.entered = 0
        self.exited = 0
        self.calls: list[tuple[str, dict]] = []
        self.connected = False
        self.fail_next = False
        self.call_delay = 0.0
        FakeClient.instances.append(self)

    async def __aenter__(self):
        self.entered += 1
        self.connected = True
        return self

    async def __aexit__(self, *exc):
        self.exited += 1
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    async def call_tool(self, name: str, args: dict):
        if self.call_delay:
            await asyncio.sleep(self.call_delay)
        if self.fail_next:
            self.fail_next = False
            self.connected = False
            raise RuntimeError("transport closed")
        self.calls.append((name, args))
        return _result(f"call-{len(self.calls)}")


@pytest.fixture
def fake_clients(monkeypatch):
    FakeClient.instances = []
    monkeypatch.setattr(
        mcp_client, "_client", lambda url, headers, use_oauth = False: FakeClient(url)
    )
    yield FakeClient.instances
    close_stdio_sessions()


def test_stdio_call_without_scope_is_one_shot(fake_clients):
    r1 = call_tool_sync(STDIO_URL, None, "browser_navigate", {"url": "https://x.test"})
    r2 = call_tool_sync(STDIO_URL, None, "browser_take_screenshot", {})
    assert r1 == "call-1"
    assert r2 == "call-1"
    assert len(fake_clients) == 2
    assert all(client.entered == 1 and client.exited == 1 for client in fake_clients)


def test_stdio_sessions_keyed_by_url_and_env(fake_clients):
    call_tool_sync(STDIO_URL, None, "t", {})
    call_tool_sync("npx other-server", None, "t", {})
    call_tool_sync(STDIO_URL, {"ENV_VAR": "1"}, "t", {})
    assert len(fake_clients) == 3


def test_stdio_sessions_scoped_per_chat(fake_clients):
    # Two conversations must not share one stateful server process.
    call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat-a")
    call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat-b")
    call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat-a")
    assert len(fake_clients) == 2
    assert fake_clients[0].calls and len(fake_clients[0].calls) == 2


def test_dead_stdio_session_recovers(fake_clients):
    assert call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat") == "call-1"
    # Subprocess dies between calls: next call reconnects instead of failing.
    fake_clients[0].fail_next = True
    assert call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat") == "call-1"
    assert len(fake_clients) == 2
    assert fake_clients[0].exited == 1


def test_tool_error_does_not_recycle_session(fake_clients, monkeypatch):
    class ToolFailure(FakeClient):
        async def call_tool(self, name, args):
            if name == "boom":
                raise RuntimeError("tool exploded")  # session stays connected
            return await super().call_tool(name, args)

    monkeypatch.setattr(
        mcp_client, "_client", lambda url, headers, use_oauth = False: ToolFailure(url)
    )
    assert call_tool_sync(STDIO_URL, None, "boom", {}, scope = "chat").startswith(
        "Error: MCP tool"
    )
    assert call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat") == "call-1"
    assert len(fake_clients) == 1


def test_http_stays_one_shot(fake_clients):
    call_tool_sync(HTTP_URL, None, "t", {})
    call_tool_sync(HTTP_URL, None, "t", {})
    assert len(fake_clients) == 2
    assert all(c.entered == 1 and c.exited == 1 for c in fake_clients)


def test_timeout_discards_stdio_session(fake_clients):
    call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat")
    key = mcp_client._session_key(STDIO_URL, None, "chat")
    fake_clients[0].call_delay = 0.5
    out = call_tool_sync(
        STDIO_URL,
        None,
        "slow",
        {},
        timeout = 0.05,
        cancel_event = threading.Event(),
        scope = "chat",
    )
    assert "timed out" in out
    assert fake_clients[0].exited == 1
    assert key not in mcp_client._stdio_key_locks
    assert call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat") == "call-1"
    assert len(fake_clients) == 2


def test_no_timeout_allows_long_call(fake_clients):
    call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat")
    fake_clients[0].call_delay = 0.2
    # timeout=None means no deadline: the call must not be treated as wedged.
    assert call_tool_sync(STDIO_URL, None, "slow", {}, timeout = None, scope = "chat") == "call-2"


def test_connect_races_cancel_event(fake_clients, monkeypatch):
    class SlowStart(FakeClient):
        async def __aenter__(self):
            await asyncio.sleep(5.0)
            return await super().__aenter__()

    monkeypatch.setattr(mcp_client, "_client", lambda url, headers, use_oauth = False: SlowStart(url))
    ev = threading.Event()
    threading.Timer(0.1, ev.set).start()
    start = time.monotonic()
    out = call_tool_sync(STDIO_URL, None, "t", {}, cancel_event = ev)
    assert out == "Error: MCP tool 't' cancelled"
    assert time.monotonic() - start < 3.0
    assert mcp_client._stdio_sessions == {}


def test_connect_respects_caller_timeout(fake_clients, monkeypatch):
    class SlowStart(FakeClient):
        async def __aenter__(self):
            await asyncio.sleep(5.0)
            return await super().__aenter__()

    monkeypatch.setattr(mcp_client, "_client", lambda url, headers, use_oauth = False: SlowStart(url))
    start = time.monotonic()
    out = call_tool_sync(STDIO_URL, None, "t", {}, timeout = 0.2)
    assert "timed out" in out
    assert time.monotonic() - start < 3.0
    assert mcp_client._stdio_sessions == {}


def test_cancel_pre_set_spawns_nothing(fake_clients):
    ev = threading.Event()
    ev.set()
    out = call_tool_sync(STDIO_URL, None, "t", {}, cancel_event = ev)
    assert out == "Error: MCP tool 't' cancelled"
    assert fake_clients == []


def test_idle_reap_closes_session(fake_clients, monkeypatch):
    call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat")
    key = mcp_client._session_key(STDIO_URL, None, "chat")
    assert key in mcp_client._stdio_key_locks
    monkeypatch.setattr(mcp_client, "_STDIO_SESSION_IDLE_TTL", 0.0)
    mcp_client._reap_idle_stdio_sessions()
    assert fake_clients[0].exited == 1
    assert mcp_client._stdio_sessions == {}
    assert key not in mcp_client._stdio_key_locks
    # Next call transparently opens a fresh session.
    assert call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat") == "call-1"
    assert len(fake_clients) == 2


def test_reap_skips_in_flight_session(fake_clients, monkeypatch):
    call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat")
    monkeypatch.setattr(mcp_client, "_STDIO_SESSION_IDLE_TTL", 0.0)
    session = next(iter(mcp_client._stdio_sessions.values()))
    with mcp_client._stdio_sessions_lock:
        session.in_flight = 1
    try:
        mcp_client._reap_idle_stdio_sessions()
        assert fake_clients[0].exited == 0
    finally:
        with mcp_client._stdio_sessions_lock:
            session.in_flight = 0


def test_close_stdio_sessions_by_url(fake_clients):
    call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat")
    call_tool_sync("npx other-server", None, "t", {}, scope = "chat")
    key = mcp_client._session_key(STDIO_URL, None, "chat")
    close_stdio_sessions(STDIO_URL)
    assert fake_clients[0].exited == 1
    assert fake_clients[1].exited == 0
    assert len(mcp_client._stdio_sessions) == 1
    assert key not in mcp_client._stdio_key_locks


def test_multi_block_result_flattens_through_session(fake_clients):
    async def _rich_call(name, args):
        return SimpleNamespace(
            content = [
                SimpleNamespace(type = "text", text = "### Page"),
                SimpleNamespace(type = "text", text = "- Page URL: https://example.com/"),
            ],
            is_error = False,
            structured_content = None,
        )

    call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat")
    fake_clients[0].call_tool = _rich_call
    out = call_tool_sync(STDIO_URL, None, "browser_snapshot", {}, scope = "chat")
    assert out == "### Page\n- Page URL: https://example.com/"
