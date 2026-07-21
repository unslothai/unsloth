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
        # Models a dead stdio transport: real Client.is_connected() stays True
        # after the subprocess dies, so liveness is probed via the transport.
        self.dead = False
        self.transport = SimpleNamespace(_is_session_dead = lambda: self.dead)
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
    # Subprocess dies between calls: the dead transport is detected before the
    # next dispatch, so the call reconnects on a fresh session instead of failing.
    fake_clients[0].dead = True
    assert call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat") == "call-1"
    assert len(fake_clients) == 2
    assert fake_clients[0].exited == 1


def test_tool_error_does_not_recycle_session(fake_clients, monkeypatch):
    from fastmcp.exceptions import ToolError

    class ToolFailure(FakeClient):
        async def call_tool(self, name, args):
            if name == "boom":
                raise ToolError("tool exploded")  # tool-level: session stays connected
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
    assert (
        call_tool_sync(STDIO_URL, None, "slow", {}, timeout = None, scope = "chat")
        == "call-2"
    )


def test_connect_races_cancel_event(fake_clients, monkeypatch):
    class SlowStart(FakeClient):
        async def __aenter__(self):
            await asyncio.sleep(5.0)
            return await super().__aenter__()

    monkeypatch.setattr(
        mcp_client, "_client", lambda url, headers, use_oauth = False: SlowStart(url)
    )
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

    monkeypatch.setattr(
        mcp_client, "_client", lambda url, headers, use_oauth = False: SlowStart(url)
    )
    start = time.monotonic()
    out = call_tool_sync(STDIO_URL, None, "t", {}, timeout = 0.2)
    assert "timed out" in out
    assert time.monotonic() - start < 3.0
    assert mcp_client._stdio_sessions == {}


def test_connect_failure_timeout_surfaces_immediately(fake_clients, monkeypatch):
    class InitTimeout(FakeClient):
        async def __aenter__(self):
            raise asyncio.TimeoutError  # e.g. fastmcp's own init timeout

    monkeypatch.setattr(
        mcp_client, "_client", lambda url, headers, use_oauth = False: InitTimeout(url)
    )
    start = time.monotonic()
    out = call_tool_sync(STDIO_URL, None, "t", {}, timeout = 30.0)
    assert "timed out" in out
    # Must fail fast, not wait out the 30s/60s connect window.
    assert time.monotonic() - start < 5.0
    assert mcp_client._stdio_sessions == {}


def test_key_lock_wait_honors_cancel_and_timeout(fake_clients, monkeypatch):
    class SlowStart(FakeClient):
        async def __aenter__(self):
            await asyncio.sleep(1.5)
            return await super().__aenter__()

    monkeypatch.setattr(
        mcp_client, "_client", lambda url, headers, use_oauth = False: SlowStart(url)
    )
    first = threading.Thread(
        target = lambda: call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat")
    )
    first.start()
    key = mcp_client._session_key(STDIO_URL, None, "chat")
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        key_lock = mcp_client._stdio_key_locks.get(key)
        if key_lock is not None and key_lock.lock.locked():
            break
        time.sleep(0.01)
    # Second same-scope call is stuck behind the first slow connect: Stop must
    # interrupt the key-lock wait, and a short tool timeout must bound it.
    ev = threading.Event()
    threading.Timer(0.2, ev.set).start()
    start = time.monotonic()
    out = call_tool_sync(STDIO_URL, None, "t", {}, cancel_event = ev, scope = "chat")
    assert out == "Error: MCP tool 't' cancelled"
    assert time.monotonic() - start < 1.0
    start = time.monotonic()
    out = call_tool_sync(STDIO_URL, None, "t", {}, timeout = 0.2, scope = "chat")
    assert "timed out" in out
    assert time.monotonic() - start < 1.0
    first.join(10.0)
    assert not first.is_alive()


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


def test_close_during_connect_is_not_cached(fake_clients, monkeypatch):
    class SlowStart(FakeClient):
        async def __aenter__(self):
            await asyncio.sleep(0.5)
            return await super().__aenter__()

    monkeypatch.setattr(
        mcp_client, "_client", lambda url, headers, use_oauth = False: SlowStart(url)
    )
    results: list[str] = []
    worker = threading.Thread(
        target = lambda: results.append(
            call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat")
        )
    )
    worker.start()
    deadline = time.monotonic() + 5.0
    while not fake_clients and time.monotonic() < deadline:
        time.sleep(0.01)
    assert fake_clients  # connect is in progress
    # Server deleted/updated mid-connect: the session must not be cached after.
    close_stdio_sessions(STDIO_URL)
    worker.join(10.0)
    assert results and results[0].startswith("Error: MCP tool 't' failed")
    assert mcp_client._stdio_sessions == {}
    assert fake_clients[0].exited == 1


def test_connect_abort_race_still_closes_client(fake_clients, monkeypatch):
    class WinsRace(FakeClient):
        async def __aenter__(self):
            try:
                await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                pass  # connect finishes just as the abort lands
            return await super().__aenter__()

    monkeypatch.setattr(
        mcp_client, "_client", lambda url, headers, use_oauth = False: WinsRace(url)
    )
    out = call_tool_sync(STDIO_URL, None, "t", {}, timeout = 0.1)
    assert "timed out" in out
    assert fake_clients[0].entered == 1
    assert fake_clients[0].exited == 1  # no orphaned subprocess
    assert mcp_client._stdio_sessions == {}


def test_close_unblocks_no_limit_call(fake_clients):
    call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat")
    session = next(iter(mcp_client._stdio_sessions.values()))
    fake_clients[0].call_delay = 30.0
    results: list[str] = []
    worker = threading.Thread(
        target = lambda: results.append(
            call_tool_sync(STDIO_URL, None, "slow", {}, timeout = None, scope = "chat")
        )
    )
    worker.start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with mcp_client._stdio_sessions_lock:
            if session.in_flight >= 1:
                break
        time.sleep(0.01)
    # Server deleted while a no-limit call is in flight: the request thread
    # must not hang forever on the stopped session loop.
    close_stdio_sessions(STDIO_URL)
    worker.join(5.0)
    assert not worker.is_alive()
    assert results and results[0].startswith("Error: MCP tool 'slow' failed")


def test_lock_wait_timeout_spares_the_borrowed_session(fake_clients):
    call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat")
    session = next(iter(mcp_client._stdio_sessions.values()))
    fake_clients[0].call_delay = 1.0
    results: list[str] = []
    slow = threading.Thread(
        target = lambda: results.append(
            call_tool_sync(STDIO_URL, None, "slow", {}, timeout = None, scope = "chat")
        )
    )
    slow.start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with mcp_client._stdio_sessions_lock:
            if session.in_flight >= 1:
                break
        time.sleep(0.01)
    # A second same-scope call times out waiting for the call lock; it never
    # touched the transport, so the shared session must stay alive and cached.
    out = call_tool_sync(STDIO_URL, None, "fast", {}, timeout = 0.05, scope = "chat")
    assert "timed out" in out
    assert fake_clients[0].exited == 0
    slow.join(10.0)
    assert results == ["call-2"]
    assert fake_clients[0].exited == 0
    assert len(mcp_client._stdio_sessions) == 1


def test_stale_session_close_deferred_until_borrower_drains(fake_clients):
    call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat")
    session = next(iter(mcp_client._stdio_sessions.values()))
    fake_clients[0].call_delay = 0.8
    results: list[str] = []
    slow = threading.Thread(
        target = lambda: results.append(
            call_tool_sync(STDIO_URL, None, "slow", {}, timeout = None, scope = "chat")
        )
    )
    slow.start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with mcp_client._stdio_sessions_lock:
            if session.in_flight >= 1:
                break
        time.sleep(0.01)
    # The subprocess "dies" mid-call: a new caller replaces the stale session,
    # but its close must wait for the slow borrower instead of killing its call.
    fake_clients[0].connected = False
    out = call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat")
    assert out == "call-1"
    assert len(fake_clients) == 2
    assert fake_clients[0].exited == 0
    slow.join(10.0)
    assert results == ["call-2"]
    assert fake_clients[0].exited == 1  # last borrower performed the deferred close
    assert len(mcp_client._stdio_sessions) == 1


def test_error_on_closed_session_does_not_retry(fake_clients):
    call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat")
    session = next(iter(mcp_client._stdio_sessions.values()))
    # A close can surface at the borrower as a plain transport error instead
    # of _SessionClosed; that must not be treated as a crash and retried.
    fake_clients[0].fail_next = True
    session.closed.set()
    out = call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat")
    assert (
        out
        == "Error: MCP tool 't' failed: MCP server was updated or removed during the call"
    )
    assert len(fake_clients) == 1  # no respawn for the removed config


def test_config_check_blocks_stale_publish(fake_clients):
    # Simulates a caller that read the server row before an update/delete:
    # the row re-check runs after connect and must block caching.
    out = call_tool_sync(
        STDIO_URL, None, "t", {}, scope = "chat", config_check = lambda: False
    )
    assert out.startswith("Error: MCP tool 't' failed")
    assert mcp_client._stdio_sessions == {}
    assert fake_clients[0].exited == 1


def test_close_generation_keys_hold_no_secrets(fake_clients):
    secret_url = "npx server --token sk-url-secret"
    close_stdio_sessions(secret_url, {"API_KEY": "sk-env-secret"})
    close_stdio_sessions(secret_url)
    gen_keys = list(mcp_client._stdio_cfg_close_gen) + list(
        mcp_client._stdio_url_close_gen
    )
    assert gen_keys
    # These maps are never pruned: neither command/URL nor env may persist.
    assert all(
        "sk-url-secret" not in repr(k) and "sk-env-secret" not in repr(k)
        for k in gen_keys
    )


def test_overlapping_calls_serialize_on_shared_session(fake_clients, monkeypatch):
    class OverlapDetect(FakeClient):
        active = 0
        max_active = 0

        async def call_tool(self, name, args):
            OverlapDetect.active += 1
            OverlapDetect.max_active = max(
                OverlapDetect.max_active, OverlapDetect.active
            )
            try:
                await asyncio.sleep(0.2)
                return await super().call_tool(name, args)
            finally:
                OverlapDetect.active -= 1

    monkeypatch.setattr(
        mcp_client, "_client", lambda url, headers, use_oauth = False: OverlapDetect(url)
    )
    call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat")
    workers = [
        threading.Thread(
            target = lambda: call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat")
        )
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(10.0)
    # A stateful server must never see interleaved same-scope operations.
    assert OverlapDetect.max_active == 1
    assert len(fake_clients) == 1


def test_timeout_budget_spans_connect_and_call(fake_clients, monkeypatch):
    class SlowBoth(FakeClient):
        async def __aenter__(self):
            await asyncio.sleep(0.4)
            return await super().__aenter__()

        async def call_tool(self, name, args):
            await asyncio.sleep(0.5)
            return await super().call_tool(name, args)

    monkeypatch.setattr(
        mcp_client, "_client", lambda url, headers, use_oauth = False: SlowBoth(url)
    )
    start = time.monotonic()
    # 0.4s connect + 0.5s call vs a 0.6s budget: the call must inherit only
    # the remaining ~0.2s, not a fresh full window.
    out = call_tool_sync(STDIO_URL, None, "t", {}, timeout = 0.6, scope = "chat")
    assert "timed out" in out
    assert time.monotonic() - start < 2.0


def test_close_narrowed_by_headers_spares_other_env(fake_clients):
    call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat")
    call_tool_sync(STDIO_URL, {"ENV_VAR": "b"}, "t", {}, scope = "chat")
    # Two server rows can share a command with different envs; editing one
    # must only close its own sessions.
    close_stdio_sessions(STDIO_URL, None)
    assert fake_clients[0].exited == 1
    assert fake_clients[1].exited == 0
    assert len(mcp_client._stdio_sessions) == 1
    close_stdio_sessions(STDIO_URL)  # headers omitted: any env for the command
    assert fake_clients[1].exited == 1
    assert mcp_client._stdio_sessions == {}


def test_close_stdio_sessions_by_url(fake_clients):
    call_tool_sync(STDIO_URL, None, "t", {}, scope = "chat")
    call_tool_sync("npx other-server", None, "t", {}, scope = "chat")
    key = mcp_client._session_key(STDIO_URL, None, "chat")
    close_stdio_sessions(STDIO_URL)
    assert fake_clients[0].exited == 1
    assert fake_clients[1].exited == 0
    assert len(mcp_client._stdio_sessions) == 1
    assert key not in mcp_client._stdio_key_locks


def test_execute_tool_mcp_scope_is_per_thread(tmp_path, monkeypatch):
    # session_id is the sandbox id and can be shared project-wide; the stdio
    # session scope must also carry the per-conversation thread id.
    from core.inference import tools as tools_mod
    from storage import mcp_servers_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", False)
    monkeypatch.setattr(tools_mod, "stdio_mcp_enabled", lambda: True)
    mcp_servers_db.create_server(
        id = "s1", display_name = "S", url = STDIO_URL, is_enabled = True
    )

    scopes: list = []

    def fake_call_tool_sync(**kwargs):
        scopes.append(kwargs["scope"])
        return "ok"

    monkeypatch.setattr(tools_mod, "call_tool_sync", fake_call_tool_sync)
    tools_mod.execute_tool(
        "mcp__s1__t", {}, session_id = "project-p1", thread_id = "thread-a"
    )
    tools_mod.execute_tool(
        "mcp__s1__t", {}, session_id = "project-p1", thread_id = "thread-b"
    )
    tools_mod.execute_tool("mcp__s1__t", {}, session_id = "sess-only")
    tools_mod.execute_tool("mcp__s1__t", {}, thread_id = "thread-a")
    # Persist only with a thread_id; session_id alone stays one-shot (None) so a
    # project-wide id can't leak state across conversations. Fields are tagged.
    assert scopes == [
        "s=project-p1:t=thread-a",
        "s=project-p1:t=thread-b",
        None,
        "s=:t=thread-a",
    ]
    # IDs containing ":" must not collapse distinct conversations into one scope,
    # and a session-only id must never collide with a thread-only id.
    tools_mod.execute_tool("mcp__s1__t", {}, session_id = "a:b", thread_id = "c")
    tools_mod.execute_tool("mcp__s1__t", {}, session_id = "a", thread_id = "b:c")
    assert scopes[-2] != scopes[-1]
    tools_mod.execute_tool("mcp__s1__t", {}, session_id = "same")
    tools_mod.execute_tool("mcp__s1__t", {}, thread_id = "same")
    assert scopes[-2] != scopes[-1]  # session-only "same" != thread-only "same"


def test_execute_tool_config_check_tracks_row(tmp_path, monkeypatch):
    from core.inference import tools as tools_mod
    from storage import mcp_servers_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", False)
    monkeypatch.setattr(tools_mod, "stdio_mcp_enabled", lambda: True)
    mcp_servers_db.create_server(
        id = "s1", display_name = "S", url = STDIO_URL, is_enabled = True
    )

    captured: dict = {}
    monkeypatch.setattr(
        tools_mod, "call_tool_sync", lambda **kw: captured.update(kw) or "ok"
    )
    tools_mod.execute_tool("mcp__s1__t", {})
    check = captured["config_check"]
    assert check() is True
    mcp_servers_db.update_server("s1", {"url": "npx different-server"})
    assert check() is False


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


def test_stdio_cache_trims_overshoot_after_burst(fake_clients, monkeypatch):
    # A concurrent burst of distinct-scope calls can overshoot the cap while every
    # session is busy (insert-time eviction only reclaims idle sessions). Once the
    # calls finish, release-time trimming must bring the cache back within cap.
    monkeypatch.setattr(mcp_client, "_STDIO_MAX_SESSIONS", 2)

    def slow_client(
        url,
        headers,
        use_oauth = False,
    ):
        client = FakeClient(url)
        client.call_delay = 0.5  # keep every session in-flight during the burst
        return client

    monkeypatch.setattr(mcp_client, "_client", slow_client)
    errors: list = []

    def worker(i: int):
        try:
            call_tool_sync(STDIO_URL, None, "t", {}, scope = f"chat-{i}")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target = worker, args = (i,)) for i in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10.0)
    assert not errors, errors
    assert len(mcp_client._stdio_sessions) <= 2


def test_close_http_server_creates_no_stdio_tombstone(fake_clients):
    # HTTP/SSE servers are never cached as stdio sessions, so closing one on
    # update/delete must not accrue a close-generation entry (an unbounded leak).
    before_cfg = len(mcp_client._stdio_cfg_close_gen)
    before_url = len(mcp_client._stdio_url_close_gen)
    for i in range(50):
        close_stdio_sessions(f"https://mcp-{i}.example/mcp", {"K": str(i)})
        close_stdio_sessions(f"https://mcp-{i}.example/mcp")
    assert len(mcp_client._stdio_cfg_close_gen) == before_cfg
    assert len(mcp_client._stdio_url_close_gen) == before_url
    # a real stdio command still registers a generation (the guard is non-stdio only)
    close_stdio_sessions(STDIO_URL, {"K": "v"})
    assert len(mcp_client._stdio_cfg_close_gen) == before_cfg + 1
