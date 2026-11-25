# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Saved HTTP MCP headers stay with their origin: repointing a row elsewhere drops them unless resent."""

import asyncio
import json

import routes.mcp_servers as routes_mcp
from models.mcp_servers import McpServerUpdate
from storage import mcp_servers_db

_SECRET = {"Authorization": "Bearer saved-secret"}


def _seed(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", set())
    mcp_servers_db.create_server(
        id = "srv",
        display_name = "srv",
        url = "https://mcp.example.com/mcp",
        headers_json = json.dumps(_SECRET),
        is_enabled = True,
        use_oauth = False,
    )


def _update(**fields):
    asyncio.run(routes_mcp.update_mcp_server("srv", McpServerUpdate(**fields), current_subject = "u"))
    return mcp_servers_db.get_server("srv")


def test_new_origin_without_headers_drops_saved_headers(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    row = _update(url = "https://attacker.example.net/mcp")
    assert row["url"] == "https://attacker.example.net/mcp"
    assert row["headers_json"] is None


def test_new_port_or_scheme_drops_saved_headers(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    assert _update(url = "http://mcp.example.com/mcp")["headers_json"] is None


def test_same_origin_path_change_keeps_headers(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    row = _update(url = "https://MCP.example.com:443/v2/mcp")
    assert json.loads(row["headers_json"]) == _SECRET


def test_new_origin_with_resent_headers_keeps_them(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    row = _update(url = "https://other.example.org/mcp", headers = _SECRET)
    assert json.loads(row["headers_json"]) == _SECRET


def test_non_url_edit_keeps_headers(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    assert json.loads(_update(display_name = "renamed")["headers_json"]) == _SECRET
