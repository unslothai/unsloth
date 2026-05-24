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
    assert _normalize_headers({"  Auth  ": "Bearer x", "": "ignored"}) == {"Auth": "Bearer x"}
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
    assert execute_tool("mcp__missing__do_thing", {}) == \
        "Error: MCP server 'missing' not found"


def test_execute_tool_disabled_server(tmp_path, monkeypatch):
    _reset_db(tmp_path, monkeypatch)
    mcp_servers_db.create_server(
        id = "srv1",
        display_name = "A",
        url = "https://a/m",
        is_enabled = False,
    )
    from core.inference.tools import execute_tool
    assert execute_tool("mcp__srv1__do_thing", {}) == \
        "Error: MCP server 'srv1' is disabled"
