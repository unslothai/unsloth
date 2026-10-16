# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Session threads and descriptors must be released; counts are polled until they settle."""

from __future__ import annotations

import gc
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

HTTP_URL = "https://mcp.example.test/mcp"
STDIO_URL = "npx fake-stateful-server"


def _settled(
    client,
    expected: int = 1,
    timeout: float = 10.0,
) -> int:
    """Discarded sessions close on the cleanup worker, so a check must wait for the close to land."""
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


class TinyClient:
    instances: list["TinyClient"] = []

    def __init__(self, url: str):
        self.url = url
        self.connected = False
        self.exited = 0
        self.transport = SimpleNamespace()
        TinyClient.instances.append(self)

    async def list_tools_mcp(self):
        return SimpleNamespace(tools = [])

    async def __aenter__(self):
        self.connected = True
        return self

    async def __aexit__(self, *exc):
        self.exited += 1
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    async def call_tool(
        self,
        name,
        args,
        raise_on_error = True,
    ):
        return _result("ok")


@pytest.fixture
def tiny(monkeypatch):
    TinyClient.instances = []
    monkeypatch.setattr(
        mcp_client, "_client", lambda url, headers, use_oauth = False: TinyClient(url)
    )
    yield TinyClient.instances
    close_mcp_sessions()


def _session_threads() -> int:
    return sum(1 for t in threading.enumerate() if t.name == "mcp-session")


def _settle(predicate, timeout = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _open_fds() -> int:
    try:
        return len(list(Path("/proc/self/fd").iterdir()))
    except OSError:
        pytest.skip("no /proc on this platform")


def test_a_closed_session_gives_its_thread_back(tiny):
    before = _session_threads()
    call_tool_sync(HTTP_URL, None, "t", {}, scope = "chat")
    assert _session_threads() > before
    close_mcp_sessions()
    assert _settle(lambda: _session_threads() <= before), "the session thread outlived close()"


def test_repeated_open_close_does_not_accumulate_threads(tiny):
    before = _session_threads()
    for i in range(12):
        call_tool_sync(HTTP_URL, None, "t", {}, scope = f"chat-{i}")
        close_mcp_sessions()
    assert _settle(
        lambda: _session_threads() <= before
    ), f"leaked threads after 12 cycles: {_session_threads()} vs {before}"


def test_repeated_open_close_does_not_accumulate_descriptors(tiny):
    for i in range(5):
        call_tool_sync(HTTP_URL, None, "t", {}, scope = f"warm-{i}")
        close_mcp_sessions()
    baseline = _session_threads()
    _settle(lambda: _session_threads() <= baseline)
    before = _open_fds()
    for i in range(12):
        call_tool_sync(HTTP_URL, None, "t", {}, scope = f"chat-{i}")
        close_mcp_sessions()
    _settle(lambda: _session_threads() <= baseline)
    gc.collect()
    # An event loop costs a few descriptors; a leak shows up as growth per cycle.
    assert _open_fds() <= before + 4, f"descriptors grew {before} -> {_open_fds()}"


def test_every_cached_client_is_exited_on_close(tiny):
    for i in range(6):
        call_tool_sync(HTTP_URL, None, "t", {}, scope = f"chat-{i}")
    assert len(mcp_client._mcp_sessions) == 6
    close_mcp_sessions()
    assert all(_settled(c) == 1 for c in tiny), [c.exited for c in tiny]


def test_the_cache_stays_within_its_cap(monkeypatch, tiny):
    # Thread counts are process-global, so compare against a baseline rather than zero.
    before = _session_threads()
    monkeypatch.setattr(mcp_client, "_MAX_SESSIONS", 3)
    for i in range(10):
        call_tool_sync(HTTP_URL, None, "t", {}, scope = f"chat-{i}")
    assert len(mcp_client._mcp_sessions) <= 3
    assert _settle(lambda: _session_threads() - before <= 3), _session_threads() - before


def test_http_and_stdio_sessions_share_one_cap(monkeypatch, tiny):
    """Worth pinning: the cap used to bound stdio subprocesses only, so a chat
    that talks to HTTP servers can now evict a stateful stdio session."""
    monkeypatch.setattr(mcp_client, "_MAX_SESSIONS", 2)
    call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat-1")
    for i in range(5):
        call_tool_sync(HTTP_URL, None, "t", {}, scope = f"http-{i}")
    assert len(mcp_client._mcp_sessions) <= 2


def test_key_locks_do_not_pile_up(tiny):
    for i in range(20):
        call_tool_sync(HTTP_URL, None, "t", {}, scope = f"chat-{i}")
        close_mcp_sessions()
    assert mcp_client._mcp_key_locks == {}, mcp_client._mcp_key_locks


def test_a_close_with_nothing_cached_leaves_no_tombstone(tiny):
    # The generation maps are module-global and earlier tests leave entries, so use a fresh url.
    url = "https://never-used.example.test/mcp"
    cfg = mcp_client._cfg_close_key(url, None)
    url_key = mcp_client._url_close_key(url)
    close_mcp_sessions(url, None)
    assert cfg not in mcp_client._mcp_cfg_close_gen
    assert url_key not in mcp_client._mcp_url_close_gen


def test_idle_sessions_are_reaped(monkeypatch, tiny):
    before = _session_threads()
    call_tool_sync(HTTP_URL, None, "t", {}, scope = "chat")
    assert len(mcp_client._mcp_sessions) == 1
    mcp_client._reap_idle_sessions(now = time.monotonic() + mcp_client._SESSION_IDLE_TTL + 1)
    assert mcp_client._mcp_sessions == {}
    assert _settle(lambda: _session_threads() <= before)
    assert _settled(tiny[0]) == 1


def test_shutdown_of_a_full_cache_is_bounded(monkeypatch, tiny):
    """close_mcp_sessions() runs on the request thread during a server edit, so a
    cache full of sessions must not stall it for minutes."""
    monkeypatch.setattr(mcp_client, "_MAX_SESSIONS", 16)
    for i in range(16):
        call_tool_sync(HTTP_URL, None, "t", {}, scope = f"chat-{i}")
    started = time.monotonic()
    close_mcp_sessions()
    assert time.monotonic() - started < 20.0, "closing a full cache took too long"


def test_sessions_are_collectable_after_close(tiny):
    import weakref

    call_tool_sync(HTTP_URL, None, "t", {}, scope = "chat")
    ref = weakref.ref(next(iter(mcp_client._mcp_sessions.values())))
    threads = _session_threads()
    close_mcp_sessions()
    _settle(lambda: _session_threads() < threads)
    gc.collect()
    assert ref() is None, "a closed session is still referenced"
