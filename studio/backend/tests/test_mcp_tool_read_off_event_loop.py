# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep MCP server-list reads off the event loop."""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

import core.inference.tools as tools
import routes.inference as inference_routes
from storage import mcp_servers_db

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

# Reuse the token-count handler fixtures.
from test_openai_auto_switch import (  # noqa: E402
    MCP_TOOL_PAYLOAD,
    _count_request,
    _count_tokens_backend,
    _enabled_mcp_server,
)


def test_the_server_list_is_read_off_the_event_loop_thread(monkeypatch):
    threads: list[int] = []

    def _list_servers():
        threads.append(threading.get_ident())
        return []

    monkeypatch.setattr(tools.mcp_servers_db, "list_servers", _list_servers)

    async def _drive():
        assert await tools.get_enabled_mcp_tools() == []
        return threading.get_ident()

    loop_thread = asyncio.run(_drive())

    assert threads, "the tool list never read the servers"
    assert threads[0] != loop_thread, "list_servers ran on the event loop thread"


def test_the_post_probe_re_read_stays_on_the_event_loop_thread(monkeypatch):
    """The race-protection re-read must NOT leave the event loop.

    It is the guard for the cache_tools / record_probe_failure writes right after it: awaiting
    it lets an MCP edit invalidate the cache in between, and the stale probe result then
    overwrites the invalidation and is served indefinitely.
    """
    server = {"id": "s1", "url": "http://127.0.0.1:1/mcp", "is_enabled": True, "use_oauth": False}
    threads: list[int] = []

    def _list_servers():
        threads.append(threading.get_ident())
        return [server]

    async def _probe(**_kwargs):
        return []

    monkeypatch.setattr(tools.mcp_servers_db, "list_servers", _list_servers)
    monkeypatch.setattr(tools, "stdio_mcp_enabled", lambda: True)
    monkeypatch.setattr(tools, "get_cached_tools", lambda _id: None)
    monkeypatch.setattr(tools, "in_failure_cooloff", lambda _id: False)
    monkeypatch.setattr(tools, "list_tools_async", _probe)

    async def _drive():
        await tools.get_enabled_mcp_tools()
        return threading.get_ident()

    loop_thread = asyncio.run(_drive())

    assert len(threads) == 2, f"expected the read and the post-probe re-read, got {len(threads)}"
    assert threads[0] != loop_thread, "the first list_servers ran on the event loop thread"
    assert threads[1] == loop_thread, "the post-probe re-read left the event loop thread"


def test_the_token_count_reads_the_cached_tools_off_the_event_loop_thread(tmp_path, monkeypatch):
    """Exercise the cache-only read through the token-count handler."""
    _count_tokens_backend(monkeypatch, count = 1234, supports_tools = True)
    _enabled_mcp_server(tmp_path, monkeypatch, cached = MCP_TOOL_PAYLOAD)

    threads: list[int] = []
    real_list_servers = mcp_servers_db.list_servers

    def _list_servers():
        threads.append(threading.get_ident())
        return real_list_servers()

    monkeypatch.setattr(mcp_servers_db, "list_servers", _list_servers)

    payload = _count_request(
        [{"role": "user", "content": "hello"}], mcp_enabled = True, enabled_tools = []
    )
    # asyncio.run uses the current thread for the event loop.
    loop_thread = threading.get_ident()
    asyncio.run(inference_routes.chat_count_tokens(payload, "tester"))

    assert threads, "the count never read the enabled MCP servers"
    assert loop_thread not in threads, "cached_mcp_tools read mcp_servers on the event loop thread"
