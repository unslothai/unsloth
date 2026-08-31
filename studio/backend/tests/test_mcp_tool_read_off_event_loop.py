# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep MCP server-list reads off the event loop."""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

from fastapi import HTTPException

import core.inference.tools as tools
import core.inference.mcp_client as mcp_client
import routes.inference as inference_routes
import routes.mcp_servers as mcp_routes
from models.mcp_servers import McpServerUpdate
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
    loop_thread = threading.get_ident()
    asyncio.run(inference_routes.chat_count_tokens(payload, "tester"))

    assert threads, "the count never read the enabled MCP servers"
    assert loop_thread not in threads, "cached_mcp_tools read mcp_servers on the event loop thread"


def test_the_cached_row_and_tools_are_one_snapshot_during_an_edit(monkeypatch):
    state = {
        "row": {
            "id": "s1",
            "display_name": "Saved",
            "url": "https://old.example/mcp",
            "headers_json": None,
            "is_enabled": True,
            "use_oauth": False,
        },
        "cache": {"s1": [{"name": "old_tool"}]},
    }
    row_read = threading.Event()
    release_row = threading.Event()

    def _list_servers():
        row = dict(state["row"])
        row_read.set()
        assert release_row.wait(2), "the concurrent update never reached the row read"
        return [row]

    def _update_server(_server_id, changes):
        state["row"].update(changes)
        return True

    def _invalidate(_server_id):
        state["cache"].pop("s1", None)

    monkeypatch.setattr(tools.mcp_servers_db, "list_servers", _list_servers)
    monkeypatch.setattr(tools, "stdio_mcp_enabled", lambda: True)
    monkeypatch.setattr(tools, "get_cached_tools", lambda server_id: state["cache"].get(server_id))
    monkeypatch.setattr(tools, "in_failure_cooloff", lambda _server_id: False)
    monkeypatch.setattr(
        tools,
        "_mcp_specs_for_server",
        lambda server, payload: [(server["url"], payload[0]["name"])],
    )
    monkeypatch.setattr(
        mcp_routes.mcp_servers_db,
        "get_server",
        lambda _server_id, **_kwargs: dict(state["row"]),
    )
    monkeypatch.setattr(mcp_routes.mcp_servers_db, "update_server", _update_server)
    monkeypatch.setattr(mcp_routes, "invalidate_tool_cache", _invalidate)
    monkeypatch.setattr(mcp_routes, "close_stdio_sessions", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mcp_routes, "_row_to_response", lambda row, **_kwargs: row)

    async def _read_snapshot():
        async with mcp_client.mcp_server_snapshot_guard():
            return await asyncio.to_thread(tools.cached_mcp_tools)

    async def _drive():
        read = asyncio.create_task(_read_snapshot())
        assert await asyncio.to_thread(row_read.wait, 2), "the cached read never started"
        update = asyncio.create_task(
            mcp_routes.update_mcp_server(
                "s1",
                McpServerUpdate(url = "https://new.example/mcp"),
                current_subject = "u",
                via_api_key = False,
            )
        )
        release_row.set()
        return await read, await update

    (specs, complete), _response = asyncio.run(_drive())
    assert complete is True
    assert specs == [("https://old.example/mcp", "old_tool")]
    assert state["row"]["url"] == "https://new.example/mcp"
    assert state["cache"] == {}


def test_queued_updates_recheck_the_stdio_api_key_gate(monkeypatch):
    state = {
        "row": {
            "id": "s1",
            "display_name": "Saved",
            "url": "https://old.example/mcp",
            "headers_json": None,
            "is_enabled": True,
            "use_oauth": False,
        }
    }

    def _update_server(_server_id, changes):
        state["row"].update(changes)
        return True

    monkeypatch.setattr(mcp_routes, "stdio_mcp_enabled", lambda: True)
    monkeypatch.setattr(
        mcp_routes.mcp_servers_db, "get_server", lambda _id, **_kwargs: dict(state["row"])
    )
    monkeypatch.setattr(mcp_routes.mcp_servers_db, "update_server", _update_server)
    monkeypatch.setattr(mcp_routes, "invalidate_tool_cache", lambda _server_id: None)
    monkeypatch.setattr(mcp_routes, "close_stdio_sessions", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mcp_routes, "_row_to_response", lambda row, **_kwargs: row)

    async def _drive():
        async with mcp_client.mcp_server_snapshot_guard():
            ui_update = asyncio.create_task(
                mcp_routes.update_mcp_server(
                    "s1",
                    McpServerUpdate(url = "npx local-server"),
                    current_subject = "u",
                    via_api_key = False,
                )
            )
            api_update = asyncio.create_task(
                mcp_routes.update_mcp_server(
                    "s1",
                    McpServerUpdate(headers = {"API_KEY": "secret"}),
                    current_subject = "u",
                    via_api_key = True,
                )
            )
            await asyncio.sleep(0)
        return await asyncio.gather(ui_update, api_update, return_exceptions = True)

    _ui_result, api_result = asyncio.run(_drive())
    assert isinstance(api_result, HTTPException)
    assert api_result.status_code == 403
    assert state["row"]["url"] == "npx local-server"
    assert state["row"]["headers_json"] is None
