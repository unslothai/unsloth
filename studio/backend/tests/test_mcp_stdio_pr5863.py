"""Verification tests for PR #5863 (stdio MCP server support).

Covers the pure helpers, the route-level _validate_url gate, and that the
UNSLOTH_STUDIO_ALLOW_STDIO_MCP gate blocks the stdio transport at every
enforcement point (create/update/test/refresh/discovery/execute) when disabled
and reaches it when enabled. The transport is stubbed so no subprocess spawns;
a recorder asserts whether it was reached.
"""

import os
import sys

import pytest
from fastapi import HTTPException

from core.inference import mcp_client
from storage import mcp_servers_db
from utils import host_policy


def _reset_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", False)
    # The discovered-tool cache is process-global and keyed by server id; tests reuse "stdio1".
    mcp_client.invalidate_tool_cache()


def _enable(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")


def _disable(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", raising = False)


@pytest.fixture(autouse = True)
def _isolate_stdio_env():
    # These three mutate process state monkeypatch cannot roll back; snapshot/restore by hand.
    from state import tool_policy

    saved = os.environ.get("UNSLOTH_STUDIO_ALLOW_STDIO_MCP")
    saved_policy = tool_policy.get_tool_policy()
    host_policy._reset_loopback_default_state()
    yield
    host_policy._reset_loopback_default_state()
    tool_policy.set_tool_policy(saved_policy)
    if saved is None:
        os.environ.pop("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", None)
    else:
        os.environ["UNSLOTH_STUDIO_ALLOW_STDIO_MCP"] = saved




class _FakeTool:
    def __init__(self, name):
        self._name = name

    def model_dump(self, exclude_none = True):
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
    """Stand-in for fastmcp.Client; records that the transport was opened."""

    def __init__(self, url, headers, use_oauth, recorder):
        recorder.append({"url": url, "headers": headers, "use_oauth": use_oauth})

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def list_tools(self):
        return [_FakeTool("list_directory"), _FakeTool("write_file")]

    async def call_tool(
        self,
        name,
        args,
        raise_on_error = True,
    ):
        return _FakeResult(f"called {name}")


@pytest.fixture
def transport(monkeypatch):
    """Patch mcp_client._client with a recorder. Returns the recorder list;
    empty == stdio transport never reached."""
    recorder = []
    monkeypatch.setattr(
        mcp_client,
        "_client",
        lambda url, headers, use_oauth = False: _RecordingClient(url, headers, use_oauth, recorder),
    )
    return recorder




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




def test_parse_basic_argv():
    assert mcp_client.parse_stdio_command(
        "npx -y @modelcontextprotocol/server-filesystem /tmp"
    ) == ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]


def test_parse_keeps_url_argument_as_one_command():
    # A :// inside an ARGUMENT must not break the command.
    assert mcp_client.parse_stdio_command("npx server --endpoint https://example.com/mcp") == [
        "npx",
        "server",
        "--endpoint",
        "https://example.com/mcp",
    ]


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
    # posix=False keeps backslash paths but also the wrapping quotes; a matched pair is stripped.
    monkeypatch.setattr(sys, "platform", "win32")
    parts = mcp_client.parse_stdio_command(r'"C:\Program Files\node\node.exe" server.js')
    assert parts[0] == r"C:\Program Files\node\node.exe"
    assert parts[1] == "server.js"




@pytest.mark.parametrize("val", ["0", "false", "true", "", " 1 ", "yes", "2"])
def test_stdio_disabled_for_non_exact_one(monkeypatch, val):
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", val)
    assert mcp_client.stdio_mcp_enabled() is False


def test_stdio_enabled_only_for_exact_one(monkeypatch):
    _disable(monkeypatch)
    assert mcp_client.stdio_mcp_enabled() is False
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    assert mcp_client.stdio_mcp_enabled() is True


def test_disabled_reason_generic(monkeypatch):
    _disable(monkeypatch)
    assert "UNSLOTH_STUDIO_ALLOW_STDIO_MCP=1" in mcp_client.stdio_mcp_disabled_reason()


def test_disabled_reason_remote_access(monkeypatch):
    _disable(monkeypatch)
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")
    host_policy.set_remote_connector_active(True)
    assert mcp_client.stdio_mcp_enabled() is False
    reason = mcp_client.stdio_mcp_disabled_reason()
    assert "Remote Access" in reason
    assert "UNSLOTH_STUDIO_ALLOW_STDIO_MCP" not in reason


def test_disabled_reason_tools_disabled(monkeypatch):
    from state import tool_policy

    _disable(monkeypatch)
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")
    tool_policy.set_tool_policy(False)
    assert mcp_client.stdio_mcp_enabled() is False
    reason = mcp_client.stdio_mcp_disabled_reason()
    assert "disable-tools" in reason
    assert "UNSLOTH_STUDIO_ALLOW_STDIO_MCP" not in reason


def test_disabled_reason_explicit_optin_not_suspended(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    host_policy.set_remote_connector_active(True)
    assert mcp_client.stdio_mcp_enabled() is True




@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "LOCALHOST", "::1"])
def test_is_external_host_false_for_loopback(host):
    assert host_policy.is_external_host(host) is False


# 127.0.0.2 is loopback, but the stack hard-codes 127.0.0.1, so only exact aliases count.
@pytest.mark.parametrize("host", ["0.0.0.0", "::", "127.0.0.2", "192.168.1.10", "example.com"])
def test_is_external_host_true_for_network(host):
    assert host_policy.is_external_host(host) is True


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "LOCALHOST", "::1"])
def test_loopback_bind_enables_stdio(monkeypatch, host):
    _disable(monkeypatch)
    host_policy.apply_stdio_mcp_loopback_default(host)
    assert mcp_client.stdio_mcp_enabled() is True


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "127.0.0.2", "192.168.1.10", "example.com"])
def test_network_bind_leaves_stdio_off(monkeypatch, host):
    _disable(monkeypatch)
    host_policy.apply_stdio_mcp_loopback_default(host)
    assert mcp_client.stdio_mcp_enabled() is False


def test_colab_loopback_does_not_auto_enable(monkeypatch):
    # Colab loopback is a hosted VM reachable via the proxy, so it stays off.
    _disable(monkeypatch)
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1", is_colab = True)
    assert mcp_client.stdio_mcp_enabled() is False


def test_explicit_enable_survives_colab(monkeypatch):
    # apply_ early-returns on an explicit value, before the is_colab check.
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1", is_colab = True)
    assert mcp_client.stdio_mcp_enabled() is True


def test_explicit_disable_survives_loopback(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "0")
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")
    assert mcp_client.stdio_mcp_enabled() is False


def test_explicit_enable_survives_network_bind(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    host_policy.apply_stdio_mcp_loopback_default("0.0.0.0")
    assert mcp_client.stdio_mcp_enabled() is True


def test_loopback_default_not_inherited_by_later_public_bind(monkeypatch):
    # A later 0.0.0.0 launch must take the auto-enable back down, not inherit it as an opt-in.
    _disable(monkeypatch)
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")
    assert mcp_client.stdio_mcp_enabled() is True
    host_policy.apply_stdio_mcp_loopback_default("0.0.0.0")
    assert mcp_client.stdio_mcp_enabled() is False


def test_remote_access_suspends_only_automatic_stdio_default(monkeypatch):
    _disable(monkeypatch)
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")
    assert mcp_client.stdio_mcp_enabled() is True
    host_policy.set_remote_connector_active(True)
    assert mcp_client.stdio_mcp_enabled() is False
    host_policy.set_remote_connector_active(False)
    assert mcp_client.stdio_mcp_enabled() is True

    host_policy._reset_loopback_default_state()
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    host_policy.set_remote_connector_active(True)
    assert mcp_client.stdio_mcp_enabled() is True


@pytest.mark.parametrize("second_host", ["127.0.0.1", "0.0.0.0"])
def test_force_disable_after_auto_default_in_same_process(monkeypatch, second_host):
    # A force-disable must win whether the later bind is loopback or public.
    _disable(monkeypatch)
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")
    assert mcp_client.stdio_mcp_enabled() is True
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "0")
    host_policy.apply_stdio_mcp_loopback_default(second_host)
    assert mcp_client.stdio_mcp_enabled() is False


def test_cleared_env_after_auto_default_falls_back_to_host_default(monkeypatch):
    # Unsetting the var (unlike =0) is "no preference", so a loopback re-apply re-enables.
    _disable(monkeypatch)
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")
    monkeypatch.delenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", raising = False)
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")
    assert mcp_client.stdio_mcp_enabled() is True


def test_disable_tools_overrides_loopback_default(monkeypatch):
    # --disable-tools is the only way tool policy is False on a loopback bind.
    from state import tool_policy

    _disable(monkeypatch)
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")
    assert mcp_client.stdio_mcp_enabled() is True
    tool_policy.set_tool_policy(False)
    assert mcp_client.stdio_mcp_enabled() is False


def test_explicit_env_opt_in_survives_external_default_policy(monkeypatch):
    # Tool policy False by the external-host default, not --disable-tools: the env opt-in wins.
    from state import tool_policy

    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    host_policy.apply_stdio_mcp_loopback_default("0.0.0.0")
    tool_policy.set_tool_policy(False)
    assert mcp_client.stdio_mcp_enabled() is True


def test_explicit_env_opt_in_beats_disable_tools_on_loopback(monkeypatch):
    # A hand-set =1 outranks --disable-tools: apply_ leaves the auto-default inactive.
    from state import tool_policy

    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")
    tool_policy.set_tool_policy(False)
    assert mcp_client.stdio_mcp_enabled() is True


@pytest.mark.parametrize("policy", [None, True])
def test_non_false_tool_policy_defers_to_env(monkeypatch, policy):
    # Only an explicit --disable-tools (False) gates stdio; None/True fall through to the env var.
    from state import tool_policy

    tool_policy.set_tool_policy(policy)
    _disable(monkeypatch)
    assert mcp_client.stdio_mcp_enabled() is False
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    assert mcp_client.stdio_mcp_enabled() is True




def test_probe_timeout_matrix():
    assert mcp_client.probe_timeout("https://x/mcp", False) == 8.0
    assert mcp_client.probe_timeout("https://x/mcp", True) == 305.0
    assert mcp_client.probe_timeout("npx server", False) == 60.0
    # oauth wins regardless of address kind (documented behaviour)
    assert mcp_client.probe_timeout("npx server", True) == 305.0




def test_validate_url_gate_off_rejects_stdio(monkeypatch):
    _disable(monkeypatch)
    from routes.mcp_servers import _validate_url

    assert _validate_url("https://example.com/mcp") == "https://example.com/mcp"
    # urlparse reads "localhost:8000" scheme as "localhost", so it lands here too.
    for bad in [
        "npx server",
        "python -m mod",
        "ftp://host",
        "example.com",
        "localhost:8000",
        r"C:\node\node.exe server.js",
    ]:
        with pytest.raises(HTTPException) as exc:
            _validate_url(bad)
        assert exc.value.status_code == 400


def test_validate_url_gate_off_message_depends_on_whitespace(monkeypatch):
    # Never says "desktop app only": self-hosted can opt in via the env var.
    _disable(monkeypatch)
    from routes.mcp_servers import _validate_url

    with pytest.raises(HTTPException) as exc:
        _validate_url("npx -y @modelcontextprotocol/server-filesystem /tmp")
    cmd = exc.value.detail.lower()
    assert "http://" in cmd and "https://" in cmd
    assert "local command" in cmd
    assert "desktop app" not in cmd

    with pytest.raises(HTTPException) as exc:
        _validate_url("example.com")
    lone = exc.value.detail.lower()
    assert "http://" in lone and "https://" in lone
    assert "local command" not in lone


def test_validate_url_gate_on_accepts_stdio(monkeypatch):
    _enable(monkeypatch)
    from routes.mcp_servers import _validate_url

    assert _validate_url("npx -y server /tmp") == "npx -y server /tmp"
    assert _validate_url("https://x/mcp") == "https://x/mcp"
    assert _validate_url("npx server --url https://x/mcp") == ("npx server --url https://x/mcp")
    # A lone token is ambiguous; accept it as a command rather than guessing a URL.
    assert _validate_url("/usr/local/bin/my-mcp-server") == "/usr/local/bin/my-mcp-server"
    assert _validate_url("mcp-server-sqlite") == "mcp-server-sqlite"
    for bad in ["   ", '"unclosed']:
        with pytest.raises(HTTPException) as exc:
            _validate_url(bad)
        assert exc.value.status_code == 400




def test_create_route_gate(tmp_path, monkeypatch, transport):
    import asyncio

    from models.mcp_servers import McpServerCreate
    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    payload = McpServerCreate(display_name = "FS", url = "npx -y server /tmp")

    _disable(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(routes_mcp.create_mcp_server(payload, current_subject = "u"))
    assert exc.value.status_code == 400

    _enable(monkeypatch)
    resp = asyncio.run(routes_mcp.create_mcp_server(payload, current_subject = "u"))
    assert resp.url == "npx -y server /tmp"


def test_update_http_to_stdio_blocked_when_off(tmp_path, monkeypatch):
    import asyncio

    from models.mcp_servers import McpServerUpdate
    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    _disable(monkeypatch)
    mcp_servers_db.create_server(id = "s1", display_name = "A", url = "https://a/mcp")
    # editing url -> stdio command must 400 (http->stdio bypass closed)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            routes_mcp.update_mcp_server(
                "s1", McpServerUpdate(url = "npx server"), current_subject = "u"
            )
        )
    assert exc.value.status_code == 400


def test_test_route_gate(tmp_path, monkeypatch, transport):
    import asyncio

    from models.mcp_servers import McpServerTestRequest
    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    req = McpServerTestRequest(url = "npx -y server /tmp")

    _disable(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(routes_mcp.test_mcp_server(req, current_subject = "u"))
    assert exc.value.status_code == 400
    assert transport == []

    _enable(monkeypatch)
    res = asyncio.run(routes_mcp.test_mcp_server(req, current_subject = "u"))
    assert res.ok and res.tool_count == 2
    assert len(transport) == 1


def test_refresh_route_gate(tmp_path, monkeypatch, transport):
    import asyncio

    import routes.mcp_servers as routes_mcp

    _reset_db(tmp_path, monkeypatch)
    # a stdio row, as if carried over from a desktop DB
    mcp_servers_db.create_server(id = "stdio1", display_name = "FS", url = "npx server")

    _disable(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(routes_mcp.refresh_mcp_server_tools("stdio1", current_subject = "u"))
    assert exc.value.status_code == 400
    assert transport == []

    _enable(monkeypatch)
    res = asyncio.run(routes_mcp.refresh_mcp_server_tools("stdio1", current_subject = "u"))
    assert res.ok and res.tool_count == 2
    assert len(transport) == 1


def test_discovery_gate(tmp_path, monkeypatch, transport):
    import asyncio

    from core.inference.tools import get_enabled_mcp_tools

    _reset_db(tmp_path, monkeypatch)
    mcp_servers_db.create_server(id = "stdio1", display_name = "FS", url = "npx server", is_enabled = True)

    _disable(monkeypatch)
    assert asyncio.run(get_enabled_mcp_tools()) == []
    assert transport == []

    _enable(monkeypatch)
    specs = asyncio.run(get_enabled_mcp_tools())
    assert len(specs) == 2
    assert len(transport) == 1


def test_execute_gate(tmp_path, monkeypatch, transport):
    from core.inference.tools import execute_tool

    _reset_db(tmp_path, monkeypatch)
    mcp_servers_db.create_server(id = "stdio1", display_name = "FS", url = "npx server", is_enabled = True)

    _disable(monkeypatch)
    out = execute_tool("mcp__stdio1__list_directory", {"path": "/tmp"})
    assert "disabled on this host" in out
    assert transport == []

    _enable(monkeypatch)
    out = execute_tool("mcp__stdio1__list_directory", {"path": "/tmp"})
    assert out == "called list_directory"
    assert len(transport) == 1




def test_stdio_env_passed_through(tmp_path, monkeypatch, transport):
    from core.inference.tools import execute_tool

    _reset_db(tmp_path, monkeypatch)
    _enable(monkeypatch)
    mcp_servers_db.create_server(
        id = "stdio1",
        display_name = "FS",
        url = "npx server",
        headers_json = '{"API_KEY": "sk-test"}',
        is_enabled = True,
    )
    execute_tool("mcp__stdio1__list_directory", {})
    assert transport[-1]["headers"] == {"API_KEY": "sk-test"}
