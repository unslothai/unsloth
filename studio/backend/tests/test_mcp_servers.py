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
    assert mcp_servers_db.update_server("srv1", {"is_enabled": False, "use_oauth": True})
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

    assert _normalize_headers({"  Auth  ": "Bearer x", "": "ignored"}) == {"Auth": "Bearer x"}
    assert _normalize_headers({"X": 42}) == {"X": "42"}
    assert _normalize_headers({}) is None
    assert _normalize_headers(None) is None
    assert _normalize_headers({"   ": "x"}) is None


def test_changes_from_payload_tristate_headers():
    from routes.mcp_servers import _changes_from_payload
    from models.mcp_servers import McpServerUpdate

    # omitted → key absent
    assert "headers_json" not in _changes_from_payload(McpServerUpdate(display_name = "x"))
    # null → stored as None (clear all headers)
    assert _changes_from_payload(McpServerUpdate(headers = None))["headers_json"] is None
    # dict → serialised JSON
    assert (
        _changes_from_payload(McpServerUpdate(headers = {"a": "1"}))["headers_json"] == '{"a": "1"}'
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
    assert execute_tool("mcp__missing__do_thing", {}) == "Error: MCP server 'missing' not found"


def test_execute_tool_disabled_server(tmp_path, monkeypatch):
    _reset_db(tmp_path, monkeypatch)
    mcp_servers_db.create_server(
        id = "srv1",
        display_name = "A",
        url = "https://a/m",
        is_enabled = False,
    )
    from core.inference.tools import execute_tool

    assert execute_tool("mcp__srv1__do_thing", {}) == "Error: MCP server 'srv1' is disabled"


def test_mcp_specs_skip_invalid_openai_function_names():
    """OpenAI requires function.name ^[a-zA-Z0-9_-]{1,64}$; bad names 400 the request."""
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
    """Duplicate tool names from one server -> OpenAI rejects; drop before forwarding."""
    from core.inference.tools import _mcp_specs_for_server

    server = {"id": "srv", "display_name": "S"}
    tools = [{"name": "echo"}, {"name": "echo"}]
    specs = _mcp_specs_for_server(server, tools)
    assert len(specs) == 1


def test_call_tool_sync_respects_pre_set_cancel_event(monkeypatch):
    """Pre-set cancel_event -> immediate cancellation, no network round-trip."""
    import threading
    from core.inference import mcp_client

    # Stub _client so the test doesn't need a real MCP server.
    class _StubClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def call_tool(
            self,
            name,
            args,
            raise_on_error = True,
        ):
            import asyncio as _asyncio
            await _asyncio.sleep(30)  # never finishes during the test

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
    """clear_oauth_tokens_async on a URL with no stored token must not raise;
    the delete + update handlers call it best-effort regardless of state."""
    import asyncio

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    from core.inference import mcp_client

    monkeypatch.setattr(mcp_client, "_oauth_token_store", None)
    asyncio.run(mcp_client.clear_oauth_tokens_async("https://example.com/mcp"))


def test_delete_server_calls_oauth_cleanup_when_oauth_was_on(tmp_path, monkeypatch):
    """delete_mcp_server route helper must call clear_oauth_tokens_async
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
    # Patch the route's module binding too so it's seen.
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
    so the new URL doesn't inherit credentials."""
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
    it previously returned 200 with {"ok": false}."""
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
    """Hyphenated property names like `issue-number` must round-trip through the
    XML parser (the old `<parameter=\\w+>` regex dropped them)."""
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
    """core/tool_healing.py has its own copy of the XML strip regex that the
    shared-parser fix missed."""
    from core.tool_healing import strip_tool_call_markup

    out = strip_tool_call_markup(
        "before <function=mcp__srv__list-issues><parameter=q>x</parameter></function> after"
    )
    assert out == "before  after"


def test_tool_healing_strip_handles_gemma_native_tool_call():
    from core.tool_healing import strip_tool_call_markup
    out = strip_tool_call_markup(
        'before <|tool_call>call:mcp__srv__list-issues{repo:"octocat/hello"}<tool_call|> after'
    )
    assert out == "before  after"


def test_tool_healing_strip_handles_gemma_close_only_marker():
    from core.tool_healing import strip_tool_call_markup
    assert strip_tool_call_markup("before <tool_call|> after") == "before  after"
    assert strip_tool_call_markup("before <tool_call|> after", final = True) == "before  after"


def test_tool_healing_parser_handles_gemma_native_windows_path():
    from core.tool_healing import parse_tool_calls_from_text
    import json as _json

    calls = parse_tool_calls_from_text(
        r'<|tool_call>call:ls{path:<|"|>C:\Users\wasim\repo<|"|>}<tool_call|>'
    )
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "ls"
    assert _json.loads(calls[0]["function"]["arguments"]) == {"path": r"C:\Users\wasim\repo"}


def test_tool_healing_json_parser_preserves_literal_gemma_quote_token():
    from core.tool_healing import parse_tool_calls_from_text
    import json as _json

    text = (
        "<tool_call>"
        + _json.dumps({"name": "python", "arguments": {"code": "print('<|\"|>')"}})
        + "</tool_call>"
    )
    calls = parse_tool_calls_from_text(text, allow_incomplete = False)
    assert len(calls) == 1
    assert _json.loads(calls[0]["function"]["arguments"]) == {"code": "print('<|\"|>')"}


def test_gguf_allow_list_blocks_unadvertised_tool(monkeypatch):
    """A tool call not in the per-request list must be refused by the GGUF
    agentic loop (mirroring the safetensors path)."""
    from core.inference import tools as tools_mod

    captured: list[str] = []

    def fake_execute(name, args, **kw):
        captured.append(name)
        return "executed"

    monkeypatch.setattr(tools_mod, "execute_tool", fake_execute)

    # Inline allow-list check to unit-test behavior without llama-server.
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
    """Pre-set cancel_event -> no HTTP request (task used to open a transport
    before the cancel check)."""
    from core.inference import mcp_client

    opened: list[str] = []

    class _StubClient:
        async def __aenter__(self):
            opened.append("opened")
            return self

        async def __aexit__(self, *args):
            return False

        async def call_tool(
            self,
            name,
            args,
            raise_on_error = True,
        ):
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
    """clear_oauth_tokens_async is best-effort; an OAuth constructor failure
    must not bubble into a 500 from the delete/update routes."""
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
    """Hyphenated tool names like `mcp__srv__list-issues` must parse, else the
    model can call the tool but Unsloth can't dispatch."""
    from core.inference.tool_call_parser import parse_tool_calls_from_text

    calls = parse_tool_calls_from_text(
        "<function=mcp__srv__list-issues><parameter=repo>octocat/hello</parameter></function>"
    )
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "mcp__srv__list-issues"
    import json as _json

    args = _json.loads(calls[0]["function"]["arguments"])
    assert args == {"repo": "octocat/hello"}


def test_tool_xml_strip_handles_hyphenated_function_names():
    """routes/inference.py:_TOOL_XML_RE must strip a `<function=name-with-dash>`
    block; else hyphenated MCP tool-call XML leaks into chat history."""
    import re as _re
    from pathlib import Path

    from core.inference.tool_call_parser import _DEEPSEEK_OPEN_RE_SRC as _DS_OPEN_SRC

    src = (Path(__file__).resolve().parent.parent / "routes/inference.py").read_text()
    m = _re.search(r"_TOOL_XML_RE = _re\.compile\((.*?)\n\)", src, _re.DOTALL)
    assert m, "could not extract _TOOL_XML_RE"
    ns: dict = {"_re": _re, "_DS_OPEN_SRC": _DS_OPEN_SRC}
    exec(f"_TOOL_XML_RE = _re.compile({m.group(1)})", ns)
    rx = ns["_TOOL_XML_RE"]
    stripped = rx.sub(
        "",
        "before <function=mcp__srv__list-issues><parameter=q>x</parameter></function> after",
    )
    assert stripped == "before  after"


def test_safetensors_agentic_empty_allowlist_still_means_allow_all():
    """Contract: at the safetensors_agentic layer tools=[] means "no
    constraint". The MCP-only-no-discovery fix lives at the route level in
    inference.py, which refuses use_tools when the resolved list is empty."""
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
    mcp_servers_db.create_server(id = "s1", display_name = "A", url = "https://x/mcp", is_enabled = True)

    calls: list[str] = []

    async def fake(
        url,
        headers = None,
        timeout = None,
        use_oauth = False,
    ):
        calls.append(url)
        return _one_tool()

    monkeypatch.setattr(tools_mod, "list_tools_async", fake)

    first = asyncio.run(tools_mod.get_enabled_mcp_tools())
    second = asyncio.run(tools_mod.get_enabled_mcp_tools())

    assert len(calls) == 1  # probed once, cache hit on the second send
    assert [t["function"]["name"] for t in first] == ["mcp__s1__echo"]
    assert first == second


def test_get_enabled_mcp_tools_does_not_cache_failures(tmp_path, monkeypatch):
    """A failed probe isn't cached: once the cool-off elapses, it's retried."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    from core.inference import tools as tools_mod

    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    monkeypatch.setattr(mcp_client, "_probe_cooloff_until", {})
    mcp_servers_db.create_server(id = "s1", display_name = "A", url = "https://x/mcp", is_enabled = True)

    attempts = {"n": 0}

    async def fake(
        url,
        headers = None,
        timeout = None,
        use_oauth = False,
    ):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("server down")
        return _one_tool()

    monkeypatch.setattr(tools_mod, "list_tools_async", fake)

    assert asyncio.run(tools_mod.get_enabled_mcp_tools()) == []  # failure -> empty
    # Expire the cool-off (an until-time in the past) so the server is retried.
    mcp_client._probe_cooloff_until["s1"] = 0.0
    second = asyncio.run(tools_mod.get_enabled_mcp_tools())
    assert attempts["n"] == 2  # retried after the cool-off, not cached
    assert [t["function"]["name"] for t in second] == ["mcp__s1__echo"]


def test_refresh_warms_tool_cache(tmp_path, monkeypatch):
    """Clicking Refresh must populate the cache the chat path reads."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    from core.inference import tools as tools_mod
    import routes.mcp_servers as routes_mcp

    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    mcp_servers_db.create_server(id = "s1", display_name = "A", url = "https://x/mcp", is_enabled = True)

    async def fake_refresh(
        url,
        headers = None,
        timeout = None,
        use_oauth = False,
    ):
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
    mcp_servers_db.create_server(id = "s1", display_name = "A", url = "https://old/mcp", is_enabled = True)

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
    mcp_servers_db.create_server(id = "s1", display_name = "A", url = "https://x/mcp", is_enabled = True)

    asyncio.run(
        routes_mcp.update_mcp_server("s1", McpServerUpdate(display_name = "B"), current_subject = "u")
    )
    assert mcp_client.get_cached_tools("s1") == cached


def test_update_rename_keeps_stdio_session(tmp_path, monkeypatch):
    """The edit dialog resends url/headers/oauth unchanged on a rename, so gating
    the close on field presence would drop the live stdio session. Only a real
    endpoint/auth change may close it."""
    import asyncio
    import json

    _reset_db(tmp_path, monkeypatch)
    from models.mcp_servers import McpServerUpdate
    import routes.mcp_servers as routes_mcp

    closed: list = []
    monkeypatch.setattr(routes_mcp, "stdio_mcp_enabled", lambda: True)
    monkeypatch.setattr(routes_mcp, "close_stdio_sessions", lambda *a, **k: closed.append(a))
    mcp_servers_db.create_server(
        id = "s1",
        display_name = "A",
        url = "npx demo-server",
        headers_json = json.dumps({"API_KEY": "x"}),
        is_enabled = True,
    )
    asyncio.run(
        routes_mcp.update_mcp_server(
            "s1",
            McpServerUpdate(
                display_name = "B",
                url = "npx demo-server",
                headers = {"API_KEY": "x"},
                use_oauth = False,
            ),
            current_subject = "u",
        )
    )
    assert closed == []
    assert mcp_servers_db.get_server("s1")["display_name"] == "B"


def test_update_stdio_command_change_closes_session(tmp_path, monkeypatch):
    """A real command change must still close the old stdio session."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from models.mcp_servers import McpServerUpdate
    import routes.mcp_servers as routes_mcp

    closed: list = []
    monkeypatch.setattr(routes_mcp, "stdio_mcp_enabled", lambda: True)
    monkeypatch.setattr(routes_mcp, "close_stdio_sessions", lambda *a, **k: closed.append(a))
    mcp_servers_db.create_server(
        id = "s1",
        display_name = "A",
        url = "npx demo-server",
        is_enabled = True,
    )
    asyncio.run(
        routes_mcp.update_mcp_server(
            "s1", McpServerUpdate(url = "npx other-server"), current_subject = "u"
        )
    )
    assert len(closed) == 1


def test_update_disable_evicts_tool_cache(tmp_path, monkeypatch):
    """Disabling a server must drop its cached tools, not leave them unread."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    from models.mcp_servers import McpServerUpdate
    import routes.mcp_servers as routes_mcp

    monkeypatch.setattr(mcp_client, "_tool_cache", {"s1": _one_tool()})
    mcp_servers_db.create_server(id = "s1", display_name = "A", url = "https://x/mcp", is_enabled = True)

    asyncio.run(
        routes_mcp.update_mcp_server("s1", McpServerUpdate(is_enabled = False), current_subject = "u")
    )
    assert mcp_client.get_cached_tools("s1") is None


def test_delete_evicts_tool_cache(tmp_path, monkeypatch):
    """Deleting a server must not leave its tools cached."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    import routes.mcp_servers as routes_mcp

    monkeypatch.setattr(mcp_client, "_tool_cache", {"s1": _one_tool()})
    mcp_servers_db.create_server(id = "s1", display_name = "A", url = "https://x/mcp", is_enabled = True)
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
    mcp_servers_db.create_server(id = "s1", display_name = "A", url = "https://a/mcp", is_enabled = True)
    mcp_servers_db.create_server(id = "s2", display_name = "B", url = "https://b/mcp", is_enabled = True)

    probed: list[str] = []

    async def fake(
        url,
        headers = None,
        timeout = None,
        use_oauth = False,
    ):
        probed.append(url)
        return _one_tool("fresh")

    monkeypatch.setattr(tools_mod, "list_tools_async", fake)

    specs = asyncio.run(tools_mod.get_enabled_mcp_tools())
    assert probed == ["https://b/mcp"]  # only the uncached server is probed
    assert sorted(t["function"]["name"] for t in specs) == ["mcp__s1__cached", "mcp__s2__fresh"]


def test_get_enabled_mcp_tools_partial_failure_caches_healthy(tmp_path, monkeypatch):
    """One server failing must not stop the others from being cached/served."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    from core.inference import tools as tools_mod

    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    monkeypatch.setattr(mcp_client, "_probe_cooloff_until", {})
    mcp_servers_db.create_server(id = "s1", display_name = "A", url = "https://bad/mcp", is_enabled = True)
    mcp_servers_db.create_server(id = "s2", display_name = "B", url = "https://good/mcp", is_enabled = True)

    async def fake(
        url,
        headers = None,
        timeout = None,
        use_oauth = False,
    ):
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
    mcp_servers_db.create_server(id = "s1", display_name = "A", url = "https://x/mcp", is_enabled = True)

    calls: list[str] = []

    async def fake(
        url,
        headers = None,
        timeout = None,
        use_oauth = False,
    ):
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
    mcp_servers_db.create_server(id = "s1", display_name = "A", url = "https://x/mcp", is_enabled = True)

    asyncio.run(
        routes_mcp.update_mcp_server(
            "s1",
            McpServerUpdate(headers = {"Authorization": "Bearer new"}),
            current_subject = "u",
        )
    )
    assert mcp_client.get_cached_tools("s1") is None


def test_get_enabled_mcp_tools_skips_cache_when_config_changes_mid_probe(tmp_path, monkeypatch):
    """A config edit landing during an in-flight probe must not be clobbered
    by the now-stale probe result (TOCTOU on the cache write)."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    from core.inference import tools as tools_mod

    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    mcp_servers_db.create_server(id = "s1", display_name = "A", url = "https://old/mcp", is_enabled = True)

    async def fake(
        url,
        headers = None,
        timeout = None,
        use_oauth = False,
    ):
        # Simulate a PUT landing while we are awaiting the probe.
        mcp_servers_db.update_server("s1", {"url": "https://new/mcp"})
        return _one_tool()

    monkeypatch.setattr(tools_mod, "list_tools_async", fake)

    specs = asyncio.run(tools_mod.get_enabled_mcp_tools())
    assert specs == []  # stale result is neither served...
    assert mcp_client.get_cached_tools("s1") is None  # ...nor cached


def test_get_enabled_mcp_tools_no_cooloff_when_config_changes_mid_failed_probe(
    tmp_path, monkeypatch
):
    """An edit landing while a probe of the OLD config is failing must not park
    a cool-off on the now-fresh config -- else the re-pointed server the user
    just fixed is needlessly skipped for the whole cool-off window."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    from core.inference import tools as tools_mod

    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    monkeypatch.setattr(mcp_client, "_probe_cooloff_until", {})
    mcp_servers_db.create_server(id = "s1", display_name = "A", url = "https://old/mcp", is_enabled = True)

    async def fake(
        url,
        headers = None,
        timeout = None,
        use_oauth = False,
    ):
        # The user re-points the server while the old endpoint's probe fails.
        mcp_servers_db.update_server("s1", {"url": "https://new/mcp"})
        raise RuntimeError("old endpoint down")

    monkeypatch.setattr(tools_mod, "list_tools_async", fake)

    assert asyncio.run(tools_mod.get_enabled_mcp_tools()) == []
    # The failure was for the OLD config, so the new one must stay re-probable.
    assert not mcp_client.in_failure_cooloff("s1")


def test_get_enabled_mcp_tools_no_cooloff_when_server_deleted_mid_failed_probe(
    tmp_path, monkeypatch
):
    """A delete landing while a probe fails must not leave an orphan cool-off
    entry keyed by the since-removed server id."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    from core.inference import tools as tools_mod

    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    monkeypatch.setattr(mcp_client, "_probe_cooloff_until", {})
    mcp_servers_db.create_server(id = "s1", display_name = "A", url = "https://x/mcp", is_enabled = True)

    async def fake(
        url,
        headers = None,
        timeout = None,
        use_oauth = False,
    ):
        mcp_servers_db.delete_server("s1")
        raise RuntimeError("down")

    monkeypatch.setattr(tools_mod, "list_tools_async", fake)

    assert asyncio.run(tools_mod.get_enabled_mcp_tools()) == []
    assert "s1" not in mcp_client._probe_cooloff_until  # no orphan cool-off


def test_get_enabled_mcp_tools_skips_failed_server_during_cooloff(tmp_path, monkeypatch):
    """A down server is probed once, then skipped during the cool-off instead
    of being re-probed (and re-hung) on every send."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    from core.inference import tools as tools_mod

    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    monkeypatch.setattr(mcp_client, "_probe_cooloff_until", {})
    mcp_servers_db.create_server(id = "s1", display_name = "A", url = "https://x/mcp", is_enabled = True)

    attempts = {"n": 0}

    async def fake(
        url,
        headers = None,
        timeout = None,
        use_oauth = False,
    ):
        attempts["n"] += 1
        raise RuntimeError("down")

    monkeypatch.setattr(tools_mod, "list_tools_async", fake)

    assert asyncio.run(tools_mod.get_enabled_mcp_tools()) == []  # probes, fails
    assert asyncio.run(tools_mod.get_enabled_mcp_tools()) == []  # within cool-off
    assert asyncio.run(tools_mod.get_enabled_mcp_tools()) == []  # still skipped
    assert attempts["n"] == 1  # only the first send probed


def test_cache_tools_clears_failure_cooloff(monkeypatch):
    """A successful probe lifts a server's failure cool-off."""
    from core.inference import mcp_client

    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    monkeypatch.setattr(mcp_client, "_probe_cooloff_until", {})
    mcp_client.record_probe_failure("s1")
    assert mcp_client.in_failure_cooloff("s1")
    mcp_client.cache_tools("s1", _one_tool())
    assert not mcp_client.in_failure_cooloff("s1")


def test_oauth_failure_cools_off_longer_than_plain(monkeypatch):
    """An OAuth server's failure cools off longer than a plain server's, so its
    multi-minute probe hang doesn't recur every minute."""
    from core.inference import mcp_client

    monkeypatch.setattr(mcp_client, "_probe_cooloff_until", {})
    mcp_client.record_probe_failure("plain", use_oauth = False)
    mcp_client.record_probe_failure("oauth", use_oauth = True)
    assert mcp_client._probe_cooloff_until["oauth"] > mcp_client._probe_cooloff_until["plain"]


def test_invalidate_clears_failure_cooloff(monkeypatch):
    """Eviction drops the failure cool-off so an edited server re-probes at once."""
    from core.inference import mcp_client

    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    monkeypatch.setattr(mcp_client, "_probe_cooloff_until", {"s1": 1.0, "s2": 2.0})
    mcp_client.invalidate_tool_cache("s1")
    assert "s1" not in mcp_client._probe_cooloff_until
    assert "s2" in mcp_client._probe_cooloff_until
    mcp_client.invalidate_tool_cache()
    assert mcp_client._probe_cooloff_until == {}


def test_refresh_failure_records_cooloff(tmp_path, monkeypatch):
    """A failed manual refresh starts the cool-off so the next chat send does
    not immediately hang on the down server."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    import routes.mcp_servers as routes_mcp

    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    monkeypatch.setattr(mcp_client, "_probe_cooloff_until", {})
    mcp_servers_db.create_server(id = "s1", display_name = "A", url = "https://x/mcp", is_enabled = True)

    async def boom(
        url,
        headers = None,
        timeout = None,
        use_oauth = False,
    ):
        raise RuntimeError("down")

    monkeypatch.setattr(routes_mcp, "list_tools_async", boom)
    res = asyncio.run(routes_mcp.refresh_mcp_server_tools("s1", current_subject = "u"))
    assert res.ok is False
    assert mcp_client.in_failure_cooloff("s1")


def test_refresh_drops_result_when_config_changes_mid_probe(tmp_path, monkeypatch):
    """A manual refresh must not warm the chat cache with tools discovered
    under an old config if the server is edited while the probe is in flight."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    import routes.mcp_servers as routes_mcp

    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    mcp_servers_db.create_server(id = "s1", display_name = "A", url = "https://old/mcp", is_enabled = True)

    async def fake_refresh(
        url,
        headers = None,
        timeout = None,
        use_oauth = False,
    ):
        mcp_servers_db.update_server("s1", {"url": "https://new/mcp"})
        return _one_tool("stale")

    monkeypatch.setattr(routes_mcp, "list_tools_async", fake_refresh)
    res = asyncio.run(routes_mcp.refresh_mcp_server_tools("s1", current_subject = "u"))
    assert res.ok and res.tool_count == 1
    assert mcp_client.get_cached_tools("s1") is None


def test_refresh_failure_no_cooloff_when_config_changes_mid_probe(tmp_path, monkeypatch):
    """A manual refresh failure for an old config must not cool off the freshly
    edited server."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    import routes.mcp_servers as routes_mcp

    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    monkeypatch.setattr(mcp_client, "_probe_cooloff_until", {})
    mcp_servers_db.create_server(id = "s1", display_name = "A", url = "https://old/mcp", is_enabled = True)

    async def boom(
        url,
        headers = None,
        timeout = None,
        use_oauth = False,
    ):
        mcp_servers_db.update_server("s1", {"url": "https://new/mcp"})
        raise RuntimeError("old endpoint down")

    monkeypatch.setattr(routes_mcp, "list_tools_async", boom)
    res = asyncio.run(routes_mcp.refresh_mcp_server_tools("s1", current_subject = "u"))
    assert res.ok is False
    assert not mcp_client.in_failure_cooloff("s1")


def test_get_enabled_mcp_tools_drops_result_when_server_deleted_mid_probe(tmp_path, monkeypatch):
    """A delete landing while a probe is in flight must drop the now-orphan
    result -- the `fresh is None` arm of the mid-probe TOCTOU guard. The
    result is neither served nor cached under the since-removed id."""
    import asyncio

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    from core.inference import tools as tools_mod

    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    monkeypatch.setattr(mcp_client, "_probe_cooloff_until", {})
    mcp_servers_db.create_server(id = "s1", display_name = "A", url = "https://x/mcp", is_enabled = True)

    async def fake(
        url,
        headers = None,
        timeout = None,
        use_oauth = False,
    ):
        # Simulate a DELETE landing while we await the probe.
        mcp_servers_db.delete_server("s1")
        return _one_tool()

    monkeypatch.setattr(tools_mod, "list_tools_async", fake)

    specs = asyncio.run(tools_mod.get_enabled_mcp_tools())
    assert specs == []  # orphan result not served
    assert mcp_client.get_cached_tools("s1") is None  # nor cached under a gone id


def test_oauth_probe_failure_in_chat_path_uses_long_cooloff(tmp_path, monkeypatch):
    """When an OAuth server fails discovery during a send, the chat path must
    record the OAuth (long) cool-off, not the plain one -- otherwise its
    multi-minute browser hang recurs every minute."""
    import asyncio
    import time

    _reset_db(tmp_path, monkeypatch)
    from core.inference import mcp_client
    from core.inference import tools as tools_mod

    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    monkeypatch.setattr(mcp_client, "_probe_cooloff_until", {})
    mcp_servers_db.create_server(
        id = "s1",
        display_name = "A",
        url = "https://x/mcp",
        is_enabled = True,
        use_oauth = True,
    )

    async def boom(
        url,
        headers = None,
        timeout = None,
        use_oauth = False,
    ):
        raise RuntimeError("oauth down")

    monkeypatch.setattr(tools_mod, "list_tools_async", boom)

    assert asyncio.run(tools_mod.get_enabled_mcp_tools()) == []
    assert mcp_client.in_failure_cooloff("s1")
    # The recorded window must exceed the plain cool-off, proving the OAuth
    # branch (use_oauth=True) fired -- not the 60 s default.
    remaining = mcp_client._probe_cooloff_until["s1"] - time.monotonic()
    assert remaining > mcp_client.FAILED_PROBE_COOLOFF_SECONDS
