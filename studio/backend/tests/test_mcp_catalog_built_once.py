# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Loaded MCP tools are looked up from one catalog build, and the listing is priced off the event loop."""

import asyncio
import json
import threading

from core.inference import mcp_client
from core.inference import tool_loop_controller as controller_mod
from core.inference import tools as tools_mod
from storage import mcp_servers_db

SEARCH = {
    "type": "function",
    "function": {"name": tools_mod.MCP_TOOL_SEARCH_NAME, "parameters": {}},
}


def _servers(
    tmp_path,
    monkeypatch,
    per_server = 6,
    servers = 2,
):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", set())
    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    for s in range(servers):
        sid = f"{s:016x}"
        mcp_servers_db.create_server(
            id = sid, display_name = f"S{s}", url = f"https://s{s}.example.com/mcp"
        )
        mcp_client.cache_tools(
            sid,
            [
                {"name": f"tool_{i}", "inputSchema": {"type": "object", "properties": {}}}
                for i in range(per_server)
            ],
        )
    return [spec["function"]["name"] for spec in tools_mod._cached_mcp_catalog()]


def _count_builds(monkeypatch):
    builds = []
    real = tools_mod._cached_mcp_catalog

    def counted():
        builds.append(1)
        return real()

    monkeypatch.setattr(tools_mod, "_cached_mcp_catalog", counted)
    return builds


def test_active_tools_builds_the_catalog_once_for_every_loaded_tool(tmp_path, monkeypatch):
    names = _servers(tmp_path, monkeypatch)
    controller = controller_mod.ToolLoopController(tools = [SEARCH])
    loaded = names[:8] + ["mcp__nope__missing", "not_an_mcp_tool"]
    controller._allowed_tool_names.update(loaded)
    builds = _count_builds(monkeypatch)

    active = controller.active_tools(include_deferred = True)

    assert len(builds) == 1
    assert [tool["function"]["name"] for tool in active] == [SEARCH["function"]["name"]] + sorted(
        names[:8]
    )
    assert all(
        tool == controller_mod._mcp_tool_spec(tool["function"]["name"]) for tool in active[1:]
    )


def test_active_tools_skips_the_catalog_with_nothing_deferred(tmp_path, monkeypatch):
    _servers(tmp_path, monkeypatch)
    controller = controller_mod.ToolLoopController(tools = [SEARCH])
    builds = _count_builds(monkeypatch)

    assert controller.active_tools(include_deferred = True) == [SEARCH]
    assert builds == []


def test_search_result_specs_build_the_catalog_once_and_keep_their_order(tmp_path, monkeypatch):
    names = _servers(tmp_path, monkeypatch)
    picked = [names[5], names[0], names[5], "mcp__nope__missing", names[9]]
    result = "\n".join(
        f"  schema: {json.dumps({'type': 'function', 'function': {'name': name}})}"
        for name in picked
    )
    builds = _count_builds(monkeypatch)

    specs = controller_mod._mcp_tool_search_specs(result)

    assert len(builds) == 1
    assert [spec["function"]["name"] for spec in specs] == [names[5], names[0], names[5], names[9]]


def test_a_catalog_edit_is_seen_by_the_next_lookup(tmp_path, monkeypatch):
    names = _servers(tmp_path, monkeypatch)
    assert tools_mod.mcp_tool_spec(names[0]) is not None
    mcp_servers_db.update_server(f"{0:016x}", {"is_enabled": False})

    assert tools_mod.mcp_tool_spec(names[0]) is None
    assert tools_mod.mcp_tool_specs(names) == {
        name: spec for name, spec in zip(names[6:], tools_mod._cached_mcp_catalog())
    }


def test_listing_is_priced_off_the_event_loop(tmp_path, monkeypatch):
    _servers(tmp_path, monkeypatch, per_server = 30, servers = 3)
    counted_on = []

    def counter(_ctx):
        def count(text):
            counted_on.append(threading.get_ident())
            return len(text) // 4

        return count

    monkeypatch.setattr(tools_mod, "_loaded_token_counter", counter)

    async def send():
        tools_mod._MCP_LISTING_CONTEXT_TOKENS.set(4096)
        tools_mod._MCP_LISTING_QUERY.set("tool 3")
        return threading.get_ident(), await tools_mod.get_enabled_mcp_tools()

    loop_thread, listing = asyncio.run(send())

    assert listing[0]["function"]["name"] == tools_mod.MCP_TOOL_SEARCH_NAME
    assert counted_on and loop_thread not in counted_on
