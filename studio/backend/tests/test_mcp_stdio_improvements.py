"""Tests for the proposed PR #5863 improvements.

Covers: _client() self-gating + keep_alive, OAuth normalised off for stdio
(create + update), env/header dropped on a transport-type switch, and rejecting
a command whose first token is a URL scheme.

Run from studio/backend:  python -m pytest tests/test_mcp_stdio_improvements.py -q
"""

import asyncio

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
    monkeypatch.delenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", raising = False)


# ── P1: _client() self-gates the stdio sink ─────────────────────────


def test_client_refuses_stdio_when_disabled(monkeypatch):
    _disable(monkeypatch)
    with pytest.raises(PermissionError):
        mcp_client._client("npx -y server /tmp", None)


def test_client_builds_stdio_when_enabled_without_spawning(monkeypatch):
    _enable(monkeypatch)
    # Constructing the Client must not spawn the subprocess (spawn happens on
    # __aenter__); only assert it builds.
    client = mcp_client._client("npx -y server /tmp", {"K": "v"})
    assert client is not None


def test_client_http_unaffected_by_gate(monkeypatch):
    _disable(monkeypatch)
    assert mcp_client._client("https://example.com/mcp", None) is not None


# ── P3: OAuth normalised off for stdio (create + update) ────────────


def test_create_forces_oauth_off_for_stdio(tmp_path, monkeypatch):
    import routes.mcp_servers as routes_mcp
    from models.mcp_servers import McpServerCreate

    _reset_db(tmp_path, monkeypatch)
    _enable(monkeypatch)
    resp = asyncio.run(
        routes_mcp.create_mcp_server(
            McpServerCreate(
                display_name = "FS", url = "npx -y server /tmp", use_oauth = True
            ),
            current_subject = "u",
        )
    )
    assert resp.use_oauth is False
    assert mcp_servers_db.get_server(resp.id)["use_oauth"] == 0


def test_create_keeps_oauth_for_http(tmp_path, monkeypatch):
    import routes.mcp_servers as routes_mcp
    from models.mcp_servers import McpServerCreate

    _reset_db(tmp_path, monkeypatch)
    _enable(monkeypatch)
    resp = asyncio.run(
        routes_mcp.create_mcp_server(
            McpServerCreate(display_name = "GH", url = "https://gh/mcp", use_oauth = True),
            current_subject = "u",
        )
    )
    assert resp.use_oauth is True


def test_update_url_to_stdio_clears_oauth(tmp_path, monkeypatch):
    import routes.mcp_servers as routes_mcp
    from models.mcp_servers import McpServerUpdate

    _reset_db(tmp_path, monkeypatch)
    _enable(monkeypatch)
    monkeypatch.setattr(mcp_client, "_oauth_token_store", None)
    monkeypatch.setattr(
        routes_mcp, "clear_oauth_tokens_async", lambda *a, **k: asyncio.sleep(0)
    )
    mcp_servers_db.create_server(
        id = "s1", display_name = "A", url = "https://a/mcp", use_oauth = True
    )
    resp = asyncio.run(
        routes_mcp.update_mcp_server(
            "s1", McpServerUpdate(url = "npx -y server /tmp"), current_subject = "u"
        )
    )
    assert resp.use_oauth is False


# ── P4: env/headers dropped on a transport-type switch ──────────────


def test_switch_stdio_to_http_drops_env(tmp_path, monkeypatch):
    import routes.mcp_servers as routes_mcp
    from models.mcp_servers import McpServerUpdate

    _reset_db(tmp_path, monkeypatch)
    _enable(monkeypatch)
    mcp_servers_db.create_server(
        id = "s1",
        display_name = "A",
        url = "npx server",
        headers_json = '{"API_KEY": "secret"}',
    )
    resp = asyncio.run(
        routes_mcp.update_mcp_server(
            "s1", McpServerUpdate(url = "https://remote/mcp"), current_subject = "u"
        )
    )
    # stdio env must NOT survive as HTTP headers on the remote endpoint
    assert resp.headers == {}
    assert mcp_servers_db.get_server("s1")["headers_json"] is None


def test_switch_keeps_explicitly_supplied_headers(tmp_path, monkeypatch):
    import routes.mcp_servers as routes_mcp
    from models.mcp_servers import McpServerUpdate

    _reset_db(tmp_path, monkeypatch)
    _enable(monkeypatch)
    mcp_servers_db.create_server(
        id = "s1",
        display_name = "A",
        url = "npx server",
        headers_json = '{"API_KEY": "secret"}',
    )
    resp = asyncio.run(
        routes_mcp.update_mcp_server(
            "s1",
            McpServerUpdate(
                url = "https://remote/mcp", headers = {"Authorization": "Bearer new"}
            ),
            current_subject = "u",
        )
    )
    assert resp.headers == {"Authorization": "Bearer new"}


def test_same_transport_edit_keeps_headers(tmp_path, monkeypatch):
    import routes.mcp_servers as routes_mcp
    from models.mcp_servers import McpServerUpdate

    _reset_db(tmp_path, monkeypatch)
    _enable(monkeypatch)
    mcp_servers_db.create_server(
        id = "s1",
        display_name = "A",
        url = "npx server",
        headers_json = '{"API_KEY": "secret"}',
    )
    # editing only the display name (still stdio) must keep env vars
    resp = asyncio.run(
        routes_mcp.update_mcp_server(
            "s1", McpServerUpdate(display_name = "B"), current_subject = "u"
        )
    )
    assert resp.headers == {"API_KEY": "secret"}


# ── P5: reject a command whose first token is a URL scheme ───────────


def test_validate_url_rejects_url_scheme_command_when_enabled(monkeypatch):
    from routes.mcp_servers import _validate_url
    _enable(monkeypatch)
    for bad in ["ftp://host/x", "file:///etc/passwd", "ws://h/y"]:
        with pytest.raises(HTTPException) as exc:
            _validate_url(bad)
        assert exc.value.status_code == 400


def test_validate_url_allows_url_in_argument(monkeypatch):
    from routes.mcp_servers import _validate_url
    _enable(monkeypatch)
    # :// inside an ARGUMENT (not the first token) is a valid command
    assert _validate_url("npx server --url https://x/mcp") == (
        "npx server --url https://x/mcp"
    )


# ── P6: Data Recipe stdio path obeys the same host gate ─────────────
# build_mcp_providers needs the Unsloth-only data_designer plugin; skip if absent.

_STDIO_RECIPE = {
    "mcp_providers": [
        {
            "provider_type": "stdio",
            "name": "fs",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            "env": {},
        }
    ]
}


def test_data_recipe_skips_stdio_when_disabled(monkeypatch):
    pytest.importorskip("data_designer")
    _disable(monkeypatch)
    from core.data_recipe.service import build_mcp_providers

    # gate off -> the stdio provider is dropped (no subprocess spawned)
    assert build_mcp_providers(_STDIO_RECIPE) == []


def test_data_recipe_builds_stdio_when_enabled(monkeypatch):
    pytest.importorskip("data_designer")
    _enable(monkeypatch)
    from core.data_recipe.service import build_mcp_providers

    built = build_mcp_providers(_STDIO_RECIPE)
    assert len(built) == 1  # constructed (not spawned) only when enabled
