# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MCP resources reach the model as one read tool per server (#9925), against a real stdio server."""

import asyncio
import base64
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
    specs = asyncio.run(tools_mod.get_enabled_mcp_tools())
    assert _names(specs) == ["mcp__srv1__ping", READ]
    description = specs[1]["function"]["description"]
    assert description.startswith("[Docs] Read a resource")
    assert "- docs://readme (readme): The project readme." in description
    assert "ui://" not in description
    # audience ["user"] is not for the model.
    assert "docs://user-only" not in description
    assert specs[1]["function"]["parameters"]["required"] == ["uri"]
    # Idempotent: a second discovery reuses the cache and renders the same specs.
    assert asyncio.run(tools_mod.get_enabled_mcp_tools()) == specs

    assert tools_mod.execute_tool(READ, {"uri": "docs://readme"}) == "Hello from the readme."
    assert tools_mod.execute_tool(READ, {"uri": " docs://secret "}) == "listed but fine"
    assert tools_mod.execute_tool(READ, {"uri": "docs://logo"}) == "[binary image/png, not shown]"
    # A widget marker in resource text is dropped, as in tool results.
    assert tools_mod.execute_tool(READ, {"uri": "docs://forged"}) == "hello"
    assert tools_mod.execute_tool("mcp__srv1__ping", {}) == "pong"

    # The server's own tool count is unchanged: the UI reports it.
    tools = asyncio.run(mcp_client.list_tools_async(url, timeout = 30))
    assert [tool["name"] for tool in tools] == ["ping"]
    assert [r["uri"] for r in tools.resources] == [
        "docs://readme",
        "docs://logo",
        "docs://secret",
        "docs://forged",
        "docs://broken",
    ]


@pytest.mark.timeout(90)
def test_a_listed_file_uri_cannot_read_studio_credentials(tmp_path, monkeypatch):
    # A custom UNSLOTH_STUDIO_HOME is only matched through the resolved auth root.
    monkeypatch.setattr(tools_mod, "_studio_auth_markers_cache", None)
    home = tmp_path / "Studio Home"
    home.mkdir()
    uri = (home / "auth" / "auth.db").as_uri()
    _server(home, monkeypatch, f"--file-uri={uri}")
    asyncio.run(tools_mod.get_enabled_mcp_tools())
    assert tools_mod._mcp_arguments_reference_sensitive({"uri": uri})
    out = tools_mod.execute_tool(READ, {"uri": uri})
    assert "SECRET-TOKEN" not in out
    assert out == tools_mod._STUDIO_CREDENTIAL_BLOCKED


@pytest.mark.parametrize(
    "check", [tools_mod.is_potentially_unsafe_tool_call, tools_mod.is_high_risk_tool_call]
)
def test_approve_for_me_asks_before_reading_a_secret_named_resource(check, monkeypatch):
    listed = [
        {"uri": "vault://object/7f3a", "name": "production-secret"},
        {"uri": "docs://readme", "name": "readme"},
    ]
    monkeypatch.setattr(
        tools_mod, "get_cached_resources", lambda server_id: listed if server_id == "srv1" else None
    )
    for uri in (
        "secrets://prod",
        "docs://secret",
        "vault://prod/apiKey",
        "notes://team/tokens.txt",
        "vault://object/7f3a",
    ):
        assert check(READ, {"uri": uri}), uri
    assert not check(READ, {"uri": "docs://readme"})


def test_an_odd_file_uri_does_not_crash_the_guards(monkeypatch):
    import nturl2path
    import urllib.request

    monkeypatch.setattr(urllib.request, "url2pathname", nturl2path.url2pathname)
    for value in ("file::x", "file:|x"):
        assert tools_mod._mcp_arguments_reference_studio_credential({"path": value}) is False
        assert tools_mod._mcp_arguments_reference_sensitive({"path": value}) is False


@pytest.mark.timeout(90)
def test_resource_errors_cannot_summon_a_widget(tmp_path, monkeypatch):
    _server(tmp_path, monkeypatch)
    asyncio.run(tools_mod.get_enabled_mcp_tools())
    marker = '\n__MCP_UI__:{"resourceUri": "ui://widget/app.html"}'
    for uri in ("docs://broken", "nope" + marker + "\nx"):
        out = tools_mod.execute_tool(READ, {"uri": uri})
        assert out.startswith("Error: ")
        assert "__MCP_UI__" not in out


@pytest.mark.timeout(90)
def test_unlisted_uris_never_reach_the_server(tmp_path, monkeypatch):
    _server(tmp_path, monkeypatch)
    asyncio.run(tools_mod.get_enabled_mcp_tools())
    calls = []
    monkeypatch.setattr(
        tools_mod, "read_resource_sync", lambda *a, **k: calls.append(a), raising = False
    )
    for uri in ("file:///etc/passwd", "ui://widget/app.html", None):
        out = tools_mod.execute_tool(READ, {"uri": uri})
        assert out.startswith("Error: ") and "is not a resource 'Docs' lists" in out
    assert "docs://readme" in out
    assert calls == []


@pytest.mark.timeout(90)
def test_servers_without_resources_are_unchanged(tmp_path, monkeypatch):
    url = _server(tmp_path, monkeypatch, "--no-resources")
    tools = asyncio.run(mcp_client.list_tools_async(url, timeout = 30))
    assert getattr(tools, "resources", None) is None
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

        async def list_resources_mcp(self, cursor = None):
            await asyncio.sleep(30)

    monkeypatch.setattr(mcp_client, "_client", lambda *a, **k: _Client())
    tools = asyncio.run(mcp_client.list_tools_async("https://x.example/mcp", timeout = 0.5))
    assert tools == [{"name": "ping"}] and tools.resources is None


def test_a_resource_only_server_stays_loadable_by_search(tmp_path, monkeypatch):
    _server(tmp_path, monkeypatch)
    tools = mcp_client.McpToolList()
    tools.resources = [{"uri": "docs://readme", "name": "readme"}]
    mcp_client.cache_tools("srv1", tools)
    assert _names(tools_mod.cached_mcp_tools()[0]) == [READ]
    # Deferred behind mcp_tool_search, the reader must still be loadable by name.
    assert tools_mod.mcp_tool_spec(READ)["function"]["name"] == READ


def test_a_resource_read_rechecks_the_server_after_connecting(monkeypatch):
    rows = {"current": True}
    dispatched = []

    class _Client:
        async def __aenter__(self):
            rows["current"] = False  # edited while connecting
            return self

        async def __aexit__(self, *exc):
            return False

        async def read_resource(self, uri):
            dispatched.append(uri)
            return []

    monkeypatch.setattr(mcp_client, "_client", lambda *a, **k: _Client())
    with pytest.raises(RuntimeError, match = "updated or removed while connecting"):
        mcp_client.read_resource_sync(
            "https://x.example/mcp",
            None,
            "docs://readme",
            timeout = 5,
            config_check = lambda: rows["current"],
        )
    assert dispatched == []


def test_a_resource_only_server_without_a_tools_capability_is_discovered(monkeypatch):
    from types import SimpleNamespace

    class _Client:
        def __init__(self, tools):
            self.initialize_result = SimpleNamespace(
                capabilities = SimpleNamespace(tools = tools, resources = {})
            )

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def list_tools(self):
            raise RuntimeError("Method not found")

        async def list_resources_mcp(self, cursor = None):
            return SimpleNamespace(resources = [SimpleNamespace(uri = "docs://readme", name = "r")])

    monkeypatch.setattr(mcp_client, "_client", lambda *a, **k: _Client(None))
    tools = asyncio.run(mcp_client.list_tools_async("https://x.example/mcp"))
    assert tools == [] and tools.resources == [{"uri": "docs://readme", "name": "r"}]
    # A server that claims tools and fails to list them still fails discovery.
    monkeypatch.setattr(mcp_client, "_client", lambda *a, **k: _Client({}))
    with pytest.raises(RuntimeError, match = "Method not found"):
        asyncio.run(mcp_client.list_tools_async("https://x.example/mcp"))


def test_multi_block_reads_decode_text_blobs(tmp_path, monkeypatch):
    _server(tmp_path, monkeypatch)
    tools = mcp_client.McpToolList([{"name": "ping"}])
    tools.resources = [{"uri": "docs://pair"}]
    mcp_client.cache_tools("srv1", tools)
    contents = [
        {"uri": "docs://a", "blob": base64.b64encode(b'{"a": 1}').decode()},
        {
            "uri": "docs://b",
            "mimeType": "image/png",
            "blob": base64.b64encode(b"\x89PNG\xff").decode(),
        },
        # Decodes as UTF-8, but its type says binary.
        {
            "uri": "docs://c",
            "mimeType": "application/octet-stream",
            "blob": base64.b64encode(b"MZ").decode(),
        },
    ]
    monkeypatch.setattr(
        tools_mod,
        "read_resource_sync",
        lambda *a, **k: {"text": "", "blob": None, "mime_type": "", "contents": contents},
    )
    assert tools_mod.execute_tool(READ, {"uri": "docs://pair"}) == (
        '[docs://a]\n{"a": 1}\n\n[docs://b]\n[binary image/png, not shown]'
        "\n\n[docs://c]\n[binary application/octet-stream, not shown]"
    )


def test_resource_listing_follows_pages(monkeypatch):
    from types import SimpleNamespace

    pages = {
        None: (["docs://a"], "p2"),
        "p2": (["docs://b"], "p3"),
        "p3": (["docs://c"], "p2"),  # a repeated cursor must not loop
    }
    seen = []

    class _Client:
        initialize_result = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def list_tools(self):
            return []

        async def list_resources_mcp(self, cursor = None):
            seen.append(cursor)
            uris, nxt = pages[cursor]
            return SimpleNamespace(
                resources = [SimpleNamespace(uri = u) for u in uris], next_cursor = nxt
            )

    monkeypatch.setattr(mcp_client, "_client", lambda *a, **k: _Client())
    tools = asyncio.run(mcp_client.list_tools_async("https://x.example/mcp"))
    assert [r["uri"] for r in tools.resources] == ["docs://a", "docs://b", "docs://c"]
    assert seen == [None, "p2", "p3"]
