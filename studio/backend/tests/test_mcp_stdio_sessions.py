# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import sys
import threading
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


def test_stdio_session_persists_across_calls(fake_clients):
    r1 = call_tool_sync(STDIO_URL, None, "browser_navigate", {"url": "https://x.test"})
    r2 = call_tool_sync(STDIO_URL, None, "browser_take_screenshot", {})
    assert r1 == "call-1"
    assert r2 == "call-2"
    # One subprocess for both calls: navigation state survives to the screenshot.
    assert len(fake_clients) == 1
    assert fake_clients[0].entered == 1
    assert fake_clients[0].exited == 0
    assert [name for name, _ in fake_clients[0].calls] == [
        "browser_navigate",
        "browser_take_screenshot",
    ]


def test_stdio_sessions_keyed_by_url_and_env(fake_clients):
    call_tool_sync(STDIO_URL, None, "t", {})
    call_tool_sync("npx other-server", None, "t", {})
    call_tool_sync(STDIO_URL, {"ENV_VAR": "1"}, "t", {})
    assert len(fake_clients) == 3


def test_dead_stdio_session_recovers(fake_clients):
    assert call_tool_sync(STDIO_URL, None, "t", {}) == "call-1"
    # Subprocess dies between calls: next call reconnects instead of failing.
    fake_clients[0].fail_next = True
    assert call_tool_sync(STDIO_URL, None, "t", {}) == "call-1"
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
    assert call_tool_sync(STDIO_URL, None, "boom", {}).startswith("Error: MCP tool")
    assert call_tool_sync(STDIO_URL, None, "t", {}) == "call-1"
    assert len(fake_clients) == 1


def test_http_stays_one_shot(fake_clients):
    call_tool_sync(HTTP_URL, None, "t", {})
    call_tool_sync(HTTP_URL, None, "t", {})
    assert len(fake_clients) == 2
    assert all(c.entered == 1 and c.exited == 1 for c in fake_clients)


def test_timeout_keeps_stdio_session(fake_clients):
    call_tool_sync(STDIO_URL, None, "t", {})
    fake_clients[0].call_delay = 0.5
    out = call_tool_sync(STDIO_URL, None, "slow", {}, timeout = 0.05)
    assert "timed out" in out
    fake_clients[0].call_delay = 0.0
    # The session survives a slow tool call (the browser is not torn down).
    assert call_tool_sync(STDIO_URL, None, "t", {}) == "call-2"
    assert len(fake_clients) == 1


def test_cancel_pre_set_spawns_nothing(fake_clients):
    ev = threading.Event()
    ev.set()
    out = call_tool_sync(STDIO_URL, None, "t", {}, cancel_event = ev)
    assert out == "Error: MCP tool 't' cancelled"
    assert fake_clients == []


def test_idle_reap_closes_session(fake_clients, monkeypatch):
    call_tool_sync(STDIO_URL, None, "t", {})
    monkeypatch.setattr(mcp_client, "_STDIO_SESSION_IDLE_TTL", 0.0)
    mcp_client._reap_idle_stdio_sessions()
    assert fake_clients[0].exited == 1
    assert mcp_client._stdio_sessions == {}
    # Next call transparently opens a fresh session.
    assert call_tool_sync(STDIO_URL, None, "t", {}) == "call-1"
    assert len(fake_clients) == 2


def test_reap_skips_in_flight_session(fake_clients, monkeypatch):
    call_tool_sync(STDIO_URL, None, "t", {})
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
    call_tool_sync(STDIO_URL, None, "t", {})
    call_tool_sync("npx other-server", None, "t", {})
    close_stdio_sessions(STDIO_URL)
    assert fake_clients[0].exited == 1
    assert fake_clients[1].exited == 0
    assert len(mcp_client._stdio_sessions) == 1


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

    call_tool_sync(STDIO_URL, None, "t", {})
    fake_clients[0].call_tool = _rich_call
    out = call_tool_sync(STDIO_URL, None, "browser_snapshot", {})
    assert out == "### Page\n- Page URL: https://example.com/"
