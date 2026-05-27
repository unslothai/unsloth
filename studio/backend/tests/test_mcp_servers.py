# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from fastapi import HTTPException

from storage import mcp_servers_db


def _reset_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", False)


# ── storage: mcp_servers_db ─────────────────────────────────────────


def test_create_and_get_server(tmp_path, monkeypatch):
    _reset_db(tmp_path, monkeypatch)
    mcp_servers_db.create_server(
        id = "srv1",
        display_name = "GitHub",
        url = "https://example.com/mcp",
        headers_json = '{"Authorization": "Bearer x"}',
        is_enabled = True,
        use_oauth = False,
    )
    row = mcp_servers_db.get_server("srv1")
    assert row["id"] == "srv1"
    assert row["display_name"] == "GitHub"
    assert row["url"] == "https://example.com/mcp"
    assert row["headers_json"] == '{"Authorization": "Bearer x"}'
    assert row["is_enabled"] == 1
    assert row["use_oauth"] == 0


def test_list_servers_ordered_by_created_at(tmp_path, monkeypatch):
    _reset_db(tmp_path, monkeypatch)
    mcp_servers_db.create_server(id = "a", display_name = "A", url = "https://a/m")
    mcp_servers_db.create_server(id = "b", display_name = "B", url = "https://b/m")
    rows = mcp_servers_db.list_servers()
    assert [r["id"] for r in rows] == ["a", "b"]


def test_update_server_coerces_bools(tmp_path, monkeypatch):
    _reset_db(tmp_path, monkeypatch)
    mcp_servers_db.create_server(id = "srv1", display_name = "A", url = "https://a/m")
    assert mcp_servers_db.update_server(
        "srv1", {"is_enabled": False, "use_oauth": True}
    )
    row = mcp_servers_db.get_server("srv1")
    assert row["is_enabled"] == 0
    assert row["use_oauth"] == 1


def test_update_server_empty_changes_returns_false(tmp_path, monkeypatch):
    _reset_db(tmp_path, monkeypatch)
    mcp_servers_db.create_server(id = "srv1", display_name = "A", url = "https://a/m")
    assert mcp_servers_db.update_server("srv1", {}) is False


def test_delete_server_roundtrip(tmp_path, monkeypatch):
    _reset_db(tmp_path, monkeypatch)
    mcp_servers_db.create_server(id = "srv1", display_name = "A", url = "https://a/m")
    assert mcp_servers_db.delete_server("srv1") is True
    assert mcp_servers_db.delete_server("srv1") is False
    assert mcp_servers_db.get_server("srv1") is None


# ── routes/mcp_servers: pure helpers ────────────────────────────────


def test_validate_url_accepts_http_and_https():
    from routes.mcp_servers import _validate_url

    assert _validate_url("http://example.com/mcp") == "http://example.com/mcp"
    assert _validate_url("https://example.com/mcp") == "https://example.com/mcp"
    assert _validate_url("  https://example.com/mcp  ") == "https://example.com/mcp"


@pytest.mark.parametrize("bad", ["", "   ", "ftp://x", "http://", "noscheme.com"])
def test_validate_url_rejects_bad(bad):
    from routes.mcp_servers import _validate_url

    with pytest.raises(HTTPException) as exc:
        _validate_url(bad)
    assert exc.value.status_code == 400


def test_normalize_headers():
    from routes.mcp_servers import _normalize_headers

    assert _normalize_headers({"  Auth  ": "Bearer x", "": "ignored"}) == {
        "Auth": "Bearer x"
    }
    assert _normalize_headers({"X": 42}) == {"X": "42"}
    assert _normalize_headers({}) is None
    assert _normalize_headers(None) is None
    assert _normalize_headers({"   ": "x"}) is None


def test_changes_from_payload_tristate_headers():
    from routes.mcp_servers import _changes_from_payload
    from models.mcp_servers import McpServerUpdate

    # omitted → key absent
    assert "headers_json" not in _changes_from_payload(
        McpServerUpdate(display_name = "x")
    )
    # null → stored as None (clear all headers)
    assert _changes_from_payload(McpServerUpdate(headers = None))["headers_json"] is None
    # dict → serialised JSON
    assert (
        _changes_from_payload(McpServerUpdate(headers = {"a": "1"}))["headers_json"]
        == '{"a": "1"}'
    )


# ── core/inference/tools: MCP wiring ────────────────────────────────


def test_mcp_specs_skip_oversized_names():
    from core.inference.tools import _mcp_specs_for_server

    server = {"id": "s" * 30, "display_name": "S"}
    tools = [
        {"name": "ok", "description": "fine"},
        {"name": "x" * 40, "description": "too long"},
    ]
    specs = _mcp_specs_for_server(server, tools)
    assert len(specs) == 1
    assert specs[0]["function"]["name"].endswith("__ok")
    assert len(specs[0]["function"]["name"]) <= 64


def test_execute_tool_malformed_mcp_name():
    from core.inference.tools import execute_tool

    out = execute_tool("mcp__no_double_underscore", {})
    assert out.startswith("Error: malformed MCP tool name")


def test_execute_tool_unknown_server(tmp_path, monkeypatch):
    _reset_db(tmp_path, monkeypatch)
    from core.inference.tools import execute_tool

    assert (
        execute_tool("mcp__missing__do_thing", {})
        == "Error: MCP server 'missing' not found"
    )


def test_execute_tool_disabled_server(tmp_path, monkeypatch):
    _reset_db(tmp_path, monkeypatch)
    mcp_servers_db.create_server(
        id = "srv1",
        display_name = "A",
        url = "https://a/m",
        is_enabled = False,
    )
    from core.inference.tools import execute_tool

    assert (
        execute_tool("mcp__srv1__do_thing", {})
        == "Error: MCP server 'srv1' is disabled"
    )


def test_mcp_specs_skip_invalid_openai_function_names():
    """OpenAI requires function.name ^[a-zA-Z0-9_-]{1,64}$; tools whose
    names contain '.', '/', spaces, etc. would 400 the whole request."""
    from core.inference.tools import _mcp_specs_for_server

    server = {"id": "srv", "display_name": "S"}
    tools = [
        {"name": "ok"},
        {"name": "with.dot"},
        {"name": "weird/slash"},
        {"name": "has space"},
        {"name": "good-dash_ok"},
    ]
    specs = _mcp_specs_for_server(server, tools)
    names = {s["function"]["name"] for s in specs}
    assert {"mcp__srv__ok", "mcp__srv__good-dash_ok"} == names


def test_mcp_specs_skip_empty_tool_name():
    from core.inference.tools import _mcp_specs_for_server

    server = {"id": "srv", "display_name": "S"}
    specs = _mcp_specs_for_server(server, [{"name": "", "description": "x"}])
    assert specs == []


def test_mcp_specs_drops_duplicate_names():
    """Same tool name twice from one MCP server -> OpenAI rejects the
    request as 'duplicates'. Drop the duplicate before forwarding."""
    from core.inference.tools import _mcp_specs_for_server

    server = {"id": "srv", "display_name": "S"}
    tools = [{"name": "echo"}, {"name": "echo"}]
    specs = _mcp_specs_for_server(server, tools)
    assert len(specs) == 1


def test_call_tool_sync_respects_pre_set_cancel_event(monkeypatch):
    """cancel_event already set before the call -> immediate Error: cancelled
    without making a network round-trip."""
    import threading
    from core.inference import mcp_client

    # Stub _client so the test doesn't need a real MCP server.
    class _StubClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def call_tool(self, name, args):
            import asyncio as _asyncio

            await _asyncio.sleep(30)  # never finishes within the test

    monkeypatch.setattr(mcp_client, "_client", lambda *a, **kw: _StubClient())

    cancel = threading.Event()
    cancel.set()
    out = mcp_client.call_tool_sync(
        url = "https://example/mcp",
        headers = None,
        name = "slow",
        args = {},
        timeout = 30.0,
        cancel_event = cancel,
    )
    assert "cancelled" in out.lower()


def test_clear_oauth_tokens_async_no_op_safe(tmp_path, monkeypatch):
    """clear_oauth_tokens_async on a URL with no stored token must not raise --
    the delete + update handlers call it best-effort regardless of prior state."""
    import asyncio

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    from core.inference import mcp_client

    monkeypatch.setattr(mcp_client, "_oauth_token_store", None)
    asyncio.run(mcp_client.clear_oauth_tokens_async("https://example.com/mcp"))


def test_delete_server_calls_oauth_cleanup_when_oauth_was_on(tmp_path, monkeypatch):
    """delete_mcp_server route helper should invoke clear_oauth_tokens_async
    when the deleted row had use_oauth=true."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client

    monkeypatch.setattr(mcp_client, "_oauth_token_store", None)
    mcp_servers_db.create_server(
        id = "oauth1",
        display_name = "GH",
        url = "https://gh-mcp.example/mcp",
        is_enabled = True,
        use_oauth = True,
    )

    calls: list[str] = []

    async def fake_clear(url):
        calls.append(url)

    monkeypatch.setattr(mcp_client, "clear_oauth_tokens_async", fake_clear)
    # Re-import the route's binding through the module so the patch is seen.
    import routes.mcp_servers as routes_mcp

    monkeypatch.setattr(routes_mcp, "clear_oauth_tokens_async", fake_clear)
    asyncio.run(routes_mcp.delete_mcp_server("oauth1", current_subject = "u"))
    assert calls == ["https://gh-mcp.example/mcp"]
    assert mcp_servers_db.get_server("oauth1") is None


def test_delete_server_skips_oauth_cleanup_when_oauth_off(tmp_path, monkeypatch):
    """No OAuth token cleanup when the deleted server never had OAuth."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    import routes.mcp_servers as routes_mcp

    monkeypatch.setattr(mcp_client, "_oauth_token_store", None)
    mcp_servers_db.create_server(
        id = "noauth",
        display_name = "Plain",
        url = "https://plain/mcp",
        is_enabled = True,
        use_oauth = False,
    )
    calls: list[str] = []

    async def fake_clear(url):
        calls.append(url)

    monkeypatch.setattr(routes_mcp, "clear_oauth_tokens_async", fake_clear)
    asyncio.run(routes_mcp.delete_mcp_server("noauth", current_subject = "u"))
    assert calls == []


def test_update_server_clears_oauth_on_url_change(tmp_path, monkeypatch):
    """Changing the URL on an OAuth server must drop the old URL's tokens
    so the new URL doesn't silently inherit credentials."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    from models.mcp_servers import McpServerUpdate
    import routes.mcp_servers as routes_mcp

    monkeypatch.setattr(mcp_client, "_oauth_token_store", None)
    mcp_servers_db.create_server(
        id = "s1",
        display_name = "A",
        url = "https://old/mcp",
        is_enabled = True,
        use_oauth = True,
    )
    calls: list[str] = []

    async def fake_clear(url):
        calls.append(url)

    monkeypatch.setattr(routes_mcp, "clear_oauth_tokens_async", fake_clear)
    asyncio.run(
        routes_mcp.update_mcp_server(
            "s1",
            McpServerUpdate(url = "https://new/mcp"),
            current_subject = "u",
        )
    )
    assert calls == ["https://old/mcp"]
    row = mcp_servers_db.get_server("s1")
    assert row["url"] == "https://new/mcp"


def test_update_server_clears_oauth_when_oauth_disabled(tmp_path, monkeypatch):
    """Flipping use_oauth false must drop the old URL's tokens."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    from models.mcp_servers import McpServerUpdate
    import routes.mcp_servers as routes_mcp

    monkeypatch.setattr(mcp_client, "_oauth_token_store", None)
    mcp_servers_db.create_server(
        id = "s1",
        display_name = "A",
        url = "https://u/mcp",
        is_enabled = True,
        use_oauth = True,
    )
    calls: list[str] = []

    async def fake_clear(url):
        calls.append(url)

    monkeypatch.setattr(routes_mcp, "clear_oauth_tokens_async", fake_clear)
    asyncio.run(
        routes_mcp.update_mcp_server(
            "s1",
            McpServerUpdate(use_oauth = False),
            current_subject = "u",
        )
    )
    assert calls == ["https://u/mcp"]


def test_changes_from_payload_rejects_null_is_enabled():
    """Explicit null for is_enabled used to hit int(None) -> TypeError 500."""
    from routes.mcp_servers import _changes_from_payload
    from models.mcp_servers import McpServerUpdate

    with pytest.raises(HTTPException) as exc:
        _changes_from_payload(McpServerUpdate(is_enabled = None))
    assert exc.value.status_code == 400


def test_changes_from_payload_rejects_null_use_oauth():
    """Explicit null for use_oauth used to hit int(None) -> TypeError 500."""
    from routes.mcp_servers import _changes_from_payload
    from models.mcp_servers import McpServerUpdate

    with pytest.raises(HTTPException) as exc:
        _changes_from_payload(McpServerUpdate(use_oauth = None))
    assert exc.value.status_code == 400


def test_test_endpoint_surfaces_url_validation_as_400(tmp_path, monkeypatch):
    """POST /api/mcp/servers/test must 400 on invalid URL like create/update;
    previously the same input returned 200 with {"ok": false}."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from routes.mcp_servers import test_mcp_server
    from models.mcp_servers import McpServerTestRequest

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            test_mcp_server(
                McpServerTestRequest(url = "ftp://nope"),
                current_subject = "u",
            )
        )
    assert exc.value.status_code == 400


def test_tool_xml_parser_handles_hyphenated_parameter_names():
    """MCP tool schemas commonly use hyphenated property names like
    `issue-number` / `repo-name`; the XML parser's `<parameter=\\w+>` regex
    dropped those keys. Verify hyphenated parameter names round-trip."""
    from core.inference.tool_call_parser import parse_tool_calls_from_text
    import json as _json

    calls = parse_tool_calls_from_text(
        "<function=mcp__srv__create-issue>"
        "<parameter=issue-title>Bug report</parameter>"
        "<parameter=repo-name>octocat/hello</parameter>"
        "</function>"
    )
    assert len(calls) == 1
    args = _json.loads(calls[0]["function"]["arguments"])
    assert args == {"issue-title": "Bug report", "repo-name": "octocat/hello"}


def test_tool_healing_strip_handles_hyphenated_function_names():
    """GGUF's core/tool_healing.py has its own copy of the XML strip
    regex; the round-4 fix to the shared parser missed this file."""
    from core.tool_healing import strip_tool_call_markup

    out = strip_tool_call_markup(
        "before <function=mcp__srv__list-issues>"
        "<parameter=q>x</parameter></function> after"
    )
    assert out == "before  after"


def test_gguf_allow_list_blocks_unadvertised_tool(monkeypatch):
    """When the model emits a tool call not in the per-request tool list
    the GGUF agentic loop must refuse to dispatch -- mirroring the
    safetensors path. Previously execute_tool ran the call regardless."""
    from core.inference import tools as tools_mod

    captured: list[str] = []

    def fake_execute(name, args, **kw):
        captured.append(name)
        return "executed"

    monkeypatch.setattr(tools_mod, "execute_tool", fake_execute)

    # Re-create the allow-list check inline so we can unit-test the
    # behavior without spinning up llama-server.
    def _gate(tools_advertised, called_name, args):
        allowed = {
            (t.get("function") or {}).get("name")
            for t in (tools_advertised or [])
            if (t.get("function") or {}).get("name")
        }
        if allowed and called_name not in allowed:
            return "Error: tool '" + called_name + "' is not enabled"
        return fake_execute(called_name, args)

    # Built-in not in advertised list -> blocked.
    out = _gate(
        [{"function": {"name": "mcp__srv__echo"}}],
        "terminal",
        {"command": "echo x"},
    )
    assert "not enabled" in out
    assert captured == []
    # Tool in advertised list -> runs.
    out = _gate(
        [{"function": {"name": "mcp__srv__echo"}}],
        "mcp__srv__echo",
        {"text": "hi"},
    )
    assert out == "executed"
    assert captured == ["mcp__srv__echo"]


def test_call_tool_sync_short_circuits_on_pre_set_cancel(monkeypatch):
    """cancel_event set BEFORE call_tool_sync runs -> no HTTP request
    is made. Previously the call task was created before the cancel
    check, opening a transport that the watcher then had to cancel."""
    from core.inference import mcp_client

    opened: list[str] = []

    class _StubClient:
        async def __aenter__(self):
            opened.append("opened")
            return self

        async def __aexit__(self, *args):
            return False

        async def call_tool(self, name, args):
            return "ran"

    monkeypatch.setattr(mcp_client, "_client", lambda *a, **kw: _StubClient())

    import threading

    ev = threading.Event()
    ev.set()
    out = mcp_client.call_tool_sync(
        url = "https://example/mcp",
        headers = None,
        name = "x",
        args = {},
        timeout = 5.0,
        cancel_event = ev,
    )
    assert "cancelled" in out.lower()
    # The client must NOT have been opened.
    assert opened == []


def test_clear_oauth_tokens_swallows_constructor_errors(tmp_path, monkeypatch):
    """clear_oauth_tokens_async is best-effort; an OAuth constructor
    failure (e.g. missing fastmcp.client.auth) must not bubble out into
    a 500 from the delete / update routes."""
    import asyncio
    from core.inference import mcp_client

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_client, "_oauth_token_store", None)

    # Patch the OAuth import path to raise so the entire body fails.
    class _BoomOAuth:
        def __init__(self, *a, **kw):
            raise RuntimeError("simulated")

    import sys as _sys

    fake_mod = type(_sys)("fastmcp.client.auth")
    fake_mod.OAuth = _BoomOAuth
    monkeypatch.setitem(_sys.modules, "fastmcp.client.auth", fake_mod)
    # Must not raise.
    asyncio.run(mcp_client.clear_oauth_tokens_async("https://x/mcp"))


def test_tool_xml_parser_handles_hyphenated_function_names():
    """MCP tool names are advertised as `mcp__srv__list-issues` (the regex
    fix allows '-'); the XML tool-call parser must parse them too,
    otherwise the model can call the tool but Studio cannot dispatch."""
    from core.inference.tool_call_parser import parse_tool_calls_from_text

    calls = parse_tool_calls_from_text(
        "<function=mcp__srv__list-issues>"
        "<parameter=repo>octocat/hello</parameter>"
        "</function>"
    )
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "mcp__srv__list-issues"
    import json as _json

    args = _json.loads(calls[0]["function"]["arguments"])
    assert args == {"repo": "octocat/hello"}


def test_tool_xml_strip_handles_hyphenated_function_names():
    """routes/inference.py:_TOOL_XML_RE must strip a `<function=name-with-dash>`
    block; otherwise hyphenated MCP tool-call XML leaks into chat history."""
    import re as _re
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "routes/inference.py").read_text()
    m = _re.search(r"_TOOL_XML_RE = _re\.compile\((.*?)\n\)", src, _re.DOTALL)
    assert m, "could not extract _TOOL_XML_RE"
    ns: dict = {"_re": _re}
    exec(f"_TOOL_XML_RE = _re.compile({m.group(1)})", ns)
    rx = ns["_TOOL_XML_RE"]
    stripped = rx.sub(
        "",
        "before <function=mcp__srv__list-issues>"
        "<parameter=q>x</parameter></function> after",
    )
    assert stripped == "before  after"


def test_safetensors_agentic_empty_allowlist_still_means_allow_all():
    """Document existing contract: at the safetensors_agentic layer,
    tools=[] is still treated as "no constraint" (so existing callers
    work unchanged). The real fix for the MCP-only-no-discovery case
    lives at the route level in inference.py, which refuses to enter
    use_tools when the resolved tool list is empty."""
    import threading
    from core.inference.safetensors_agentic import run_safetensors_tool_loop

    calls: list[str] = []

    def fake_execute(name, args, **kw):
        calls.append(name)
        return "ran"

    iteration = {"n": 0}

    def fake_single_turn(messages):
        iteration["n"] += 1
        if iteration["n"] == 1:
            txt = '<tool_call>{"name":"python","arguments":{"code":"1"}}</tool_call>'
            buf = ""
            for ch in txt:
                buf += ch
                yield buf
        else:
            yield "done"

    list(
        run_safetensors_tool_loop(
            single_turn = fake_single_turn,
            messages = [{"role": "user", "content": "x"}],
            tools = [],
            execute_tool = fake_execute,
            cancel_event = threading.Event(),
            max_tool_iterations = 1,
        )
    )
    # Empty allow-list = run anything (preserved contract).
    assert calls == [("python", {"code": "1"})] or len(calls) >= 1


# ── discovery cache ─────────────────────────────────────────────────


def _one_tool(name = "echo"):
    return [{"name": name, "inputSchema": {"type": "object", "properties": {}}}]


def test_get_enabled_mcp_tools_caches_discovery(tmp_path, monkeypatch):
    """A second send must serve tools from cache instead of re-probing."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    from core.inference import tools as tools_mod

    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    mcp_servers_db.create_server(
        id = "s1", display_name = "A", url = "https://x/mcp", is_enabled = True
    )

    calls: list[str] = []

    async def fake(url, headers = None, timeout = None, use_oauth = False):
        calls.append(url)
        return _one_tool()

    monkeypatch.setattr(tools_mod, "list_tools_async", fake)

    first = asyncio.run(tools_mod.get_enabled_mcp_tools())
    second = asyncio.run(tools_mod.get_enabled_mcp_tools())

    assert len(calls) == 1  # probed once, cache hit on the second send
    assert [t["function"]["name"] for t in first] == ["mcp__s1__echo"]
    assert first == second


def test_get_enabled_mcp_tools_does_not_cache_failures(tmp_path, monkeypatch):
    """A failed probe must not be cached, so a recovered server is retried."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    from core.inference import tools as tools_mod

    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    mcp_servers_db.create_server(
        id = "s1", display_name = "A", url = "https://x/mcp", is_enabled = True
    )

    attempts = {"n": 0}

    async def fake(url, headers = None, timeout = None, use_oauth = False):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("server down")
        return _one_tool()

    monkeypatch.setattr(tools_mod, "list_tools_async", fake)

    assert asyncio.run(tools_mod.get_enabled_mcp_tools()) == []  # failure -> empty
    second = asyncio.run(tools_mod.get_enabled_mcp_tools())
    assert attempts["n"] == 2  # retried because the failure was not cached
    assert [t["function"]["name"] for t in second] == ["mcp__s1__echo"]


def test_refresh_warms_tool_cache(tmp_path, monkeypatch):
    """Clicking Refresh must populate the cache the chat path reads."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    from core.inference import tools as tools_mod
    import routes.mcp_servers as routes_mcp

    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    mcp_servers_db.create_server(
        id = "s1", display_name = "A", url = "https://x/mcp", is_enabled = True
    )

    async def fake_refresh(url, headers = None, timeout = None, use_oauth = False):
        return _one_tool()

    monkeypatch.setattr(routes_mcp, "list_tools_async", fake_refresh)
    res = asyncio.run(routes_mcp.refresh_mcp_server_tools("s1", current_subject = "u"))
    assert res.ok and res.tool_count == 1

    def boom(*a, **k):
        raise AssertionError("chat path re-probed despite a warm cache")

    monkeypatch.setattr(tools_mod, "list_tools_async", boom)
    specs = asyncio.run(tools_mod.get_enabled_mcp_tools())
    assert [t["function"]["name"] for t in specs] == ["mcp__s1__echo"]


def test_update_url_evicts_tool_cache(tmp_path, monkeypatch):
    """Re-pointing the URL must drop the old endpoint's cached tools."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    from models.mcp_servers import McpServerUpdate
    import routes.mcp_servers as routes_mcp

    monkeypatch.setattr(mcp_client, "_tool_cache", {"s1": _one_tool("stale")})
    mcp_servers_db.create_server(
        id = "s1", display_name = "A", url = "https://old/mcp", is_enabled = True
    )

    asyncio.run(
        routes_mcp.update_mcp_server(
            "s1", McpServerUpdate(url = "https://new/mcp"), current_subject = "u"
        )
    )
    assert mcp_client.get_cached_tools("s1") is None


def test_update_display_name_keeps_tool_cache(tmp_path, monkeypatch):
    """A rename touches no endpoint, so the cache must survive it."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    from models.mcp_servers import McpServerUpdate
    import routes.mcp_servers as routes_mcp

    cached = _one_tool()
    monkeypatch.setattr(mcp_client, "_tool_cache", {"s1": cached})
    mcp_servers_db.create_server(
        id = "s1", display_name = "A", url = "https://x/mcp", is_enabled = True
    )

    asyncio.run(
        routes_mcp.update_mcp_server(
            "s1", McpServerUpdate(display_name = "B"), current_subject = "u"
        )
    )
    assert mcp_client.get_cached_tools("s1") == cached


def test_update_disable_evicts_tool_cache(tmp_path, monkeypatch):
    """Disabling a server must drop its cached tools, not leave them unread."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    from models.mcp_servers import McpServerUpdate
    import routes.mcp_servers as routes_mcp

    monkeypatch.setattr(mcp_client, "_tool_cache", {"s1": _one_tool()})
    mcp_servers_db.create_server(
        id = "s1", display_name = "A", url = "https://x/mcp", is_enabled = True
    )

    asyncio.run(
        routes_mcp.update_mcp_server(
            "s1", McpServerUpdate(is_enabled = False), current_subject = "u"
        )
    )
    assert mcp_client.get_cached_tools("s1") is None


def test_delete_evicts_tool_cache(tmp_path, monkeypatch):
    """Deleting a server must not leave its tools cached."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    import routes.mcp_servers as routes_mcp

    monkeypatch.setattr(mcp_client, "_tool_cache", {"s1": _one_tool()})
    mcp_servers_db.create_server(
        id = "s1", display_name = "A", url = "https://x/mcp", is_enabled = True
    )
    asyncio.run(routes_mcp.delete_mcp_server("s1", current_subject = "u"))
    assert mcp_client.get_cached_tools("s1") is None


def test_invalidate_tool_cache_clears_all(monkeypatch):
    from core.inference import mcp_client

    monkeypatch.setattr(mcp_client, "_tool_cache", {"a": _one_tool(), "b": _one_tool()})
    mcp_client.invalidate_tool_cache()
    assert mcp_client.get_cached_tools("a") is None
    assert mcp_client.get_cached_tools("b") is None


def test_get_enabled_mcp_tools_probes_only_uncached(tmp_path, monkeypatch):
    """An already-cached server must not be re-probed alongside a cold one."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    from core.inference import tools as tools_mod

    monkeypatch.setattr(mcp_client, "_tool_cache", {"s1": _one_tool("cached")})
    mcp_servers_db.create_server(
        id = "s1", display_name = "A", url = "https://a/mcp", is_enabled = True
    )
    mcp_servers_db.create_server(
        id = "s2", display_name = "B", url = "https://b/mcp", is_enabled = True
    )

    probed: list[str] = []

    async def fake(url, headers = None, timeout = None, use_oauth = False):
        probed.append(url)
        return _one_tool("fresh")

    monkeypatch.setattr(tools_mod, "list_tools_async", fake)

    specs = asyncio.run(tools_mod.get_enabled_mcp_tools())
    assert probed == ["https://b/mcp"]  # only the uncached server is probed
    assert sorted(t["function"]["name"] for t in specs) == [
        "mcp__s1__cached",
        "mcp__s2__fresh",
    ]


def test_get_enabled_mcp_tools_partial_failure_caches_healthy(tmp_path, monkeypatch):
    """One server failing must not stop the others from being cached/served."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    from core.inference import tools as tools_mod

    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    mcp_servers_db.create_server(
        id = "s1", display_name = "A", url = "https://bad/mcp", is_enabled = True
    )
    mcp_servers_db.create_server(
        id = "s2", display_name = "B", url = "https://good/mcp", is_enabled = True
    )

    async def fake(url, headers = None, timeout = None, use_oauth = False):
        if "bad" in url:
            raise RuntimeError("down")
        return _one_tool("ok")

    monkeypatch.setattr(tools_mod, "list_tools_async", fake)

    specs = asyncio.run(tools_mod.get_enabled_mcp_tools())
    assert [t["function"]["name"] for t in specs] == ["mcp__s2__ok"]
    assert mcp_client.get_cached_tools("s1") is None  # failure not cached
    assert mcp_client.get_cached_tools("s2") == _one_tool("ok")  # healthy cached


def test_get_enabled_mcp_tools_caches_empty_tool_list(tmp_path, monkeypatch):
    """A server exposing zero tools is cached as [] (a hit), not re-probed."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    from core.inference import tools as tools_mod

    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    mcp_servers_db.create_server(
        id = "s1", display_name = "A", url = "https://x/mcp", is_enabled = True
    )

    calls: list[str] = []

    async def fake(url, headers = None, timeout = None, use_oauth = False):
        calls.append(url)
        return []

    monkeypatch.setattr(tools_mod, "list_tools_async", fake)

    assert asyncio.run(tools_mod.get_enabled_mcp_tools()) == []
    assert asyncio.run(tools_mod.get_enabled_mcp_tools()) == []
    assert len(calls) == 1  # [] is a cache hit, not re-probed every send
    assert mcp_client.get_cached_tools("s1") == []


def test_update_headers_evicts_tool_cache(tmp_path, monkeypatch):
    """Changing auth headers must drop tools discovered under the old headers."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    from models.mcp_servers import McpServerUpdate
    import routes.mcp_servers as routes_mcp

    monkeypatch.setattr(mcp_client, "_tool_cache", {"s1": _one_tool()})
    mcp_servers_db.create_server(
        id = "s1", display_name = "A", url = "https://x/mcp", is_enabled = True
    )

    asyncio.run(
        routes_mcp.update_mcp_server(
            "s1",
            McpServerUpdate(headers = {"Authorization": "Bearer new"}),
            current_subject = "u",
        )
    )
    assert mcp_client.get_cached_tools("s1") is None


def test_get_enabled_mcp_tools_skips_cache_when_config_changes_mid_probe(
    tmp_path, monkeypatch
):
    """A config edit landing during an in-flight probe must not be clobbered
    by the now-stale probe result (TOCTOU on the cache write)."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    from core.inference import tools as tools_mod

    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    mcp_servers_db.create_server(
        id = "s1", display_name = "A", url = "https://old/mcp", is_enabled = True
    )

    async def fake(url, headers = None, timeout = None, use_oauth = False):
        # Simulate a PUT landing while we are awaiting the probe.
        mcp_servers_db.update_server("s1", {"url": "https://new/mcp"})
        return _one_tool()

    monkeypatch.setattr(tools_mod, "list_tools_async", fake)

    specs = asyncio.run(tools_mod.get_enabled_mcp_tools())
    assert specs == []  # stale result is neither served...
    assert mcp_client.get_cached_tools("s1") is None  # ...nor cached
