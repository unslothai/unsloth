"""Verification tests for PR #5863 (stdio MCP server support).

Covers the pure helpers (is_stdio / parse_stdio_command / stdio_mcp_enabled /
probe_timeout), the route-level _validate_url gate, and - most importantly -
that the UNSLOTH_STUDIO_ALLOW_STDIO_MCP gate blocks the stdio transport at all
five enforcement points (create, update, test, refresh, discovery, execute)
when disabled, and reaches it when enabled. The transport (_client) is stubbed
so no real subprocess is spawned; a recorder asserts whether it was reached.

Run from studio/backend:  python -m pytest tests/test_mcp_stdio_pr5863.py -q
"""

import sys

import pytest
from fastapi import HTTPException

from core.inference import mcp_client
from storage import mcp_servers_db


def _reset_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", False)


def _enable(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")


def _disable(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", raising=False)


# ── transport stub + recorder ───────────────────────────────────────


class _FakeTool:
    def __init__(self, name):
        self._name = name

    def model_dump(self, exclude_none=True):
        return {"name": self._name, "description": f"{self._name} tool"}


class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResult:
    is_error = False

    def __init__(self, text):
        self.content = [_Block(text)]


class _RecordingClient:
    """Stands in for fastmcp.Client; records that the transport was opened."""

    def __init__(self, url, headers, use_oauth, recorder):
        recorder.append({"url": url, "headers": headers, "use_oauth": use_oauth})

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def list_tools(self):
        return [_FakeTool("list_directory"), _FakeTool("write_file")]

    async def call_tool(self, name, args):
        return _FakeResult(f"called {name}")


@pytest.fixture
def transport(monkeypatch):
    """Patch mcp_client._client with a recorder. Returns the recorder list;
    empty == the stdio transport was never reached."""
    recorder = []
    monkeypatch.setattr(
        mcp_client,
        "_client",
        lambda url, headers, use_oauth=False: _RecordingClient(
            url, headers, use_oauth, recorder
        ),
    )
    return recorder


# ── 1. is_stdio ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "addr",
    [
        "http://localhost:8000/mcp",
        "https://example.com/mcp",
        "  https://example.com/mcp  ",
        "HTTPS://EXAMPLE.COM/mcp",
    ],
)
def test_is_stdio_false_for_http(addr):
    assert mcp_client.is_stdio(addr) is False


@pytest.mark.parametrize(
    "addr",
    [
        "npx -y @modelcontextprotocol/server-filesystem /tmp",
        "python -m some.module",
        "uvx some-server --flag",
        "/usr/local/bin/my-server",
    ],
)
def test_is_stdio_true_for_commands(addr):
    assert mcp_client.is_stdio(addr) is True


# ── 2. parse_stdio_command ──────────────────────────────────────────


def test_parse_basic_argv():
    assert mcp_client.parse_stdio_command(
        "npx -y @modelcontextprotocol/server-filesystem /tmp"
    ) == ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]


def test_parse_keeps_url_argument_as_one_command():
    # gemini "high": a :// inside an ARGUMENT must not break the command.
    assert mcp_client.parse_stdio_command(
        "npx server --endpoint https://example.com/mcp"
    ) == ["npx", "server", "--endpoint", "https://example.com/mcp"]


def test_parse_quoted_arg():
    assert mcp_client.parse_stdio_command('python -m mod --name "a b"') == [
        "python",
        "-m",
        "mod",
        "--name",
        "a b",
    ]


def test_parse_empty_returns_empty_list():
    assert mcp_client.parse_stdio_command("   ") == []


def test_parse_unclosed_quote_raises_valueerror():
    with pytest.raises(ValueError):
        mcp_client.parse_stdio_command('npx "unclosed')


def test_parse_windows_strips_wrapping_quotes(monkeypatch):
    # gemini "medium": posix=False keeps backslash paths but also the wrapping
    # quotes; the PR strips a matched pair so argv[0] reaches the OS clean.
    monkeypatch.setattr(sys, "platform", "win32")
    parts = mcp_client.parse_stdio_command(
        r'"C:\Program Files\node\node.exe" server.js'
    )
    assert parts[0] == r"C:\Program Files\node\node.exe"
    assert parts[1] == "server.js"


# ── 3. stdio_mcp_enabled ────────────────────────────────────────────


@pytest.mark.parametrize("val", ["0", "false", "true", "", " 1 ", "yes", "2"])
def test_stdio_disabled_for_non_exact_one(monkeypatch, val):
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", val)
    assert mcp_client.stdio_mcp_enabled() is False


def test_stdio_enabled_only_for_exact_one(monkeypatch):
    _disable(monkeypatch)
    assert mcp_client.stdio_mcp_enabled() is False
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    assert mcp_client.stdio_mcp_enabled() is True


# ── 4. probe_timeout ────────────────────────────────────────────────


def test_probe_timeout_matrix():
    assert mcp_client.probe_timeout("https://x/mcp", False) == 8.0
    assert mcp_client.probe_timeout("https://x/mcp", True) == 305.0
    assert mcp_client.probe_timeout("npx server", False) == 60.0
    # oauth wins regardless of address kind (documented behaviour)
    assert mcp_client.probe_timeout("npx server", True) == 305.0


# ── 5. _validate_url gate ───────────────────────────────────────────


def test_validate_url_gate_off_rejects_stdio(monkeypatch):
    _disable(monkeypatch)
    from routes.mcp_servers import _validate_url

    assert _validate_url("https://example.com/mcp") == "https://example.com/mcp"
    for bad in ["npx server", "python -m mod", "ftp://host"]:
        with pytest.raises(HTTPException) as exc:
            _validate_url(bad)
        assert exc.value.status_code == 400


def test_validate_url_gate_on_accepts_stdio(monkeypatch):
    _enable(monkeypatch)
    from routes.mcp_servers import _validate_url

    assert _validate_url("npx -y server /tmp") == "npx -y server /tmp"
    # http still works when stdio is on
    assert _validate_url("https://x/mcp") == "https://x/mcp"
    # url-bearing argument accepted as a command
    assert _validate_url("npx server --url https://x/mcp") == (
        "npx server --url https://x/mcp"
    )
    # empty / unparseable still rejected
    for bad in ["   ", '"unclosed']:
        with pytest.raises(HTTPException) as exc:
            _validate_url(bad)
        assert exc.value.status_code == 400


# ── 6. gate enforcement at every spawn path (mocked transport) ──────


def test_create_route_gate(tmp_path, monkeypatch, transport):
    import asyncio

    from models.mcp_servers import McpServerCreate
    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    payload = McpServerCreate(display_name="FS", url="npx -y server /tmp")

    _disable(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(routes_mcp.create_mcp_server(payload, current_subject="u"))
    assert exc.value.status_code == 400

    _enable(monkeypatch)
    resp = asyncio.run(routes_mcp.create_mcp_server(payload, current_subject="u"))
    assert resp.url == "npx -y server /tmp"


def test_update_http_to_stdio_blocked_when_off(tmp_path, monkeypatch):
    import asyncio

    from models.mcp_servers import McpServerUpdate
    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    _disable(monkeypatch)
    mcp_servers_db.create_server(id="s1", display_name="A", url="https://a/mcp")
    # editing url -> stdio command must 400 (http->stdio edit bypass closed)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            routes_mcp.update_mcp_server(
                "s1", McpServerUpdate(url="npx server"), current_subject="u"
            )
        )
    assert exc.value.status_code == 400


def test_test_route_gate(tmp_path, monkeypatch, transport):
    import asyncio

    from models.mcp_servers import McpServerTestRequest
    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    req = McpServerTestRequest(url="npx -y server /tmp")

    _disable(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(routes_mcp.test_mcp_server(req, current_subject="u"))
    assert exc.value.status_code == 400
    assert transport == []  # transport never opened

    _enable(monkeypatch)
    res = asyncio.run(routes_mcp.test_mcp_server(req, current_subject="u"))
    assert res.ok and res.tool_count == 2
    assert len(transport) == 1


def test_refresh_route_gate(tmp_path, monkeypatch, transport):
    import asyncio

    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    # a stdio row as if carried over from a desktop DB
    mcp_servers_db.create_server(id="stdio1", display_name="FS", url="npx server")

    _disable(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(routes_mcp.refresh_mcp_server_tools("stdio1", current_subject="u"))
    assert exc.value.status_code == 400
    assert transport == []

    _enable(monkeypatch)
    res = asyncio.run(routes_mcp.refresh_mcp_server_tools("stdio1", current_subject="u"))
    assert res.ok and res.tool_count == 2
    assert len(transport) == 1


def test_discovery_gate(tmp_path, monkeypatch, transport):
    import asyncio

    from core.inference.tools import get_enabled_mcp_tools

    _reset_db(tmp_path, monkeypatch)
    mcp_servers_db.create_server(
        id="stdio1", display_name="FS", url="npx server", is_enabled=True
    )

    _disable(monkeypatch)
    assert asyncio.run(get_enabled_mcp_tools()) == []
    assert transport == []  # filtered out before any probe

    _enable(monkeypatch)
    specs = asyncio.run(get_enabled_mcp_tools())
    assert len(specs) == 2
    assert len(transport) == 1


def test_execute_gate(tmp_path, monkeypatch, transport):
    from core.inference.tools import execute_tool

    _reset_db(tmp_path, monkeypatch)
    mcp_servers_db.create_server(
        id="stdio1", display_name="FS", url="npx server", is_enabled=True
    )

    _disable(monkeypatch)
    out = execute_tool("mcp__stdio1__list_directory", {"path": "/tmp"})
    assert "disabled on this host" in out
    assert transport == []

    _enable(monkeypatch)
    out = execute_tool("mcp__stdio1__list_directory", {"path": "/tmp"})
    assert out == "called list_directory"
    assert len(transport) == 1


# ── 7. env vars ride headers_json as the subprocess env ─────────────


def test_stdio_env_passed_through(tmp_path, monkeypatch, transport):
    from core.inference.tools import execute_tool

    _reset_db(tmp_path, monkeypatch)
    _enable(monkeypatch)
    mcp_servers_db.create_server(
        id="stdio1",
        display_name="FS",
        url="npx server",
        headers_json='{"API_KEY": "sk-test"}',
        is_enabled=True,
    )
    execute_tool("mcp__stdio1__list_directory", {})
    assert transport[-1]["headers"] == {"API_KEY": "sk-test"}
