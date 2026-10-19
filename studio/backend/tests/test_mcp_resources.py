# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MCP resources reach the model as one read tool per server (#9925), against a real stdio server."""

import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastmcp")

from core.inference import mcp_client
from core.inference import tools as tools_mod
from storage import mcp_servers_db


FIXTURE = Path(__file__).parent / "fixtures" / "mcp_resource_server.py"
READ = "mcp__srv1__read_resource"


def _server(tmp_path, monkeypatch, *arguments):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", set())
    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    url = mcp_client.join_stdio_command([sys.executable, str(FIXTURE), *arguments])
    mcp_servers_db.create_server(id = "srv1", display_name = "Docs", url = url)
    return url


def _names(specs):
    return [spec["function"]["name"] for spec in specs]


@pytest.mark.timeout(90)
def test_listed_resources_become_a_read_tool(tmp_path, monkeypatch):
    url = _server(tmp_path, monkeypatch)
    tools = asyncio.run(mcp_client.list_tools_async(url, timeout = 30))
    # The server's own tool count is unchanged: the UI reports it.
    assert [tool["name"] for tool in tools] == ["ping"]
    assert [r["uri"] for r in tools.resources] == ["docs://readme", "docs://logo", "docs://secret"]

    specs = asyncio.run(tools_mod.get_enabled_mcp_tools())
    assert _names(specs) == ["mcp__srv1__ping", READ]
    description = specs[1]["function"]["description"]
    assert description.startswith("[Docs] Read a resource")
    assert "- docs://readme (readme): The project readme." in description
    assert "ui://" not in description
    assert specs[1]["function"]["parameters"]["required"] == ["uri"]
    # Idempotent: a second discovery reuses the cache and renders the same specs.
    assert asyncio.run(tools_mod.get_enabled_mcp_tools()) == specs

    assert tools_mod.execute_tool(READ, {"uri": "docs://readme"}) == "Hello from the readme."
    assert tools_mod.execute_tool(READ, {"uri": " docs://secret "}) == "listed but fine"
    assert tools_mod.execute_tool(READ, {"uri": "docs://logo"}) == "[binary image/png, not shown]"
    assert tools_mod.execute_tool("mcp__srv1__ping", {}) == "pong"


@pytest.mark.timeout(90)
def test_unlisted_uris_never_reach_the_server(tmp_path, monkeypatch):
    url = _server(tmp_path, monkeypatch)
    mcp_client.cache_tools("srv1", asyncio.run(mcp_client.list_tools_async(url, timeout = 30)))
    calls = []
    monkeypatch.setattr(tools_mod, "read_resource_sync", lambda *a, **k: calls.append(a))
    for uri in ("file:///etc/passwd", "ui://widget/app.html", None):
        out = tools_mod.execute_tool(READ, {"uri": uri})
        assert out.startswith("Error: ") and "is not a resource 'Docs' lists" in out
    assert "docs://readme" in out
    assert calls == []


@pytest.mark.timeout(90)
def test_servers_without_resources_are_unchanged(tmp_path, monkeypatch):
    url = _server(tmp_path, monkeypatch, "--no-resources")
    tools = asyncio.run(mcp_client.list_tools_async(url, timeout = 30))
    assert tools.resources is None
    assert _names(asyncio.run(tools_mod.get_enabled_mcp_tools())) == ["mcp__srv1__ping"]


def test_a_servers_own_read_resource_tool_wins(tmp_path, monkeypatch):
    _server(tmp_path, monkeypatch)
    tools = mcp_client.McpToolList([{"name": "read_resource", "description": "Own reader."}])
    tools.resources = [{"uri": "docs://readme"}]
    mcp_client.cache_tools("srv1", tools)
    specs = tools_mod.cached_mcp_tools()[0]
    assert _names(specs) == [READ]
    assert specs[0]["function"]["description"] == "[Docs] Own reader."
    monkeypatch.setattr(tools_mod, "call_tool_sync", lambda **kwargs: f"own:{kwargs['name']}")
    assert tools_mod.execute_tool(READ, {"uri": "docs://readme"}) == "own:read_resource"


def test_slow_resource_listing_keeps_the_tools(monkeypatch):
    class _Tool:
        def model_dump(self, **_):
            return {"name": "ping"}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def list_tools(self):
            return [_Tool()]

        async def list_resources_mcp(self):
            await asyncio.sleep(30)

    monkeypatch.setattr(mcp_client, "_client", lambda *a, **k: _Client())
    tools = asyncio.run(mcp_client.list_tools_async("https://x.example/mcp", timeout = 0.5))
    assert tools == [{"name": "ping"}] and tools.resources is None
