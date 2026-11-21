# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import sqlite3

import pytest
from fastapi import HTTPException
from core.inference import mcp_client
from models.mcp_servers import McpServerCreate, McpServerTestRequest, McpServerUpdate
from routes import mcp_servers as routes
from storage import mcp_servers_db

URL = "https://calendarmcp.googleapis.com/mcp/v1"
CREDS = {"oauth_client_id": "client-id", "oauth_client_secret": "client-secret"}


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", set())
    cleared, invalidated = [], []

    async def clear_tokens(url):
        cleared.append(url)

    monkeypatch.setattr(routes, "clear_oauth_tokens_async", clear_tokens)
    monkeypatch.setattr(routes, "invalidate_tool_cache", invalidated.append)
    mcp_servers_db.create_server(
        id = "cal", display_name = "Calendar", url = URL, use_oauth = True, **CREDS
    )
    return cleared, invalidated


def _update(**fields):
    return asyncio.run(
        routes.update_mcp_server("cal", McpServerUpdate(**fields), current_subject = "u")
    )


def test_real_fastmcp_oauth_gets_the_preregistered_client(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_client, "_oauth_token_store", None)
    auth = mcp_client._client(URL, None, use_oauth = True, **CREDS).transport.auth
    assert (auth._client_id, auth._client_secret) == ("client-id", "client-secret")


def test_list_call_and_widget_paths_forward_the_client(monkeypatch):
    captured = []

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def list_tools(self):
            return []

        async def call_tool(self, *_args, **_kwargs):
            return "ok"

    def fake_client(
        url,
        headers,
        use_oauth = False,
        **oauth,
    ):
        captured.append(oauth)
        return FakeClient()

    monkeypatch.setattr(mcp_client, "_client", fake_client)
    asyncio.run(mcp_client.list_tools_async(URL, use_oauth = True, **CREDS))
    mcp_client.call_tool_sync(URL, None, "t", {}, use_oauth = True, **CREDS)
    mcp_client._ui_request_sync(
        URL, None, "r", lambda c: c.list_tools(), timeout = 5, use_oauth = True, **CREDS
    )
    asyncio.run(mcp_client.list_tools_async(URL, use_oauth = True))
    assert captured == [CREDS, CREDS, CREDS, {}]


def test_stored_client_reaches_chat_and_widget_paths(db):
    row = mcp_servers_db.get_server("cal")
    assert mcp_client.oauth_client_kwargs(row) == CREDS
    assert routes._ui_call_kwargs("cal", row, None, None)["oauth_client_secret"] == "client-secret"


def test_responses_report_the_secret_without_returning_it(db):
    created = asyncio.run(
        routes.create_mcp_server(
            McpServerCreate(display_name = "Other", url = URL, use_oauth = True, **CREDS),
            current_subject = "u",
        )
    )
    assert created.oauth_client_id == "client-id"
    assert created.has_oauth_client_secret is True
    assert "client-secret" not in created.model_dump_json()
    with pytest.raises(HTTPException, match = "requires oauth_client_id"):
        routes._oauth_client(None, "orphan")


@pytest.mark.parametrize(
    "fields",
    [
        {"oauth_client_id": "other-id"},
        {"url": "https://other.example/mcp"},
        {"oauth_client_secret": None},
    ],
)
def test_new_client_id_url_or_removed_secret_drops_secret_and_tokens(db, fields):
    cleared, invalidated = db
    _update(**fields)
    assert mcp_servers_db.get_server("cal")["oauth_client_secret"] is None
    assert cleared == [URL] and invalidated == ["cal"]


def test_rename_resending_the_same_client_keeps_secret_and_tokens(db):
    cleared, invalidated = db
    _update(display_name = "Renamed", url = URL, use_oauth = True, oauth_client_id = "client-id")
    assert mcp_servers_db.get_server("cal")["oauth_client_secret"] == "client-secret"
    assert cleared == [] and invalidated == []


def test_secret_rotation_keeps_stored_id_and_disabling_oauth_clears_both(db):
    _update(oauth_client_secret = "rotated")
    row = mcp_servers_db.get_server("cal")
    assert (row["oauth_client_id"], row["oauth_client_secret"]) == ("client-id", "rotated")
    _update(use_oauth = False, oauth_client_id = "client-id")
    row = mcp_servers_db.get_server("cal")
    assert (row["oauth_client_id"], row["oauth_client_secret"]) == (None, None)


@pytest.mark.parametrize(
    "url, expected", [(URL, "client-secret"), ("https://attacker.example/mcp", None)]
)
def test_connection_test_reuses_stored_secret_only_for_the_same_url(db, monkeypatch, url, expected):
    captured = {}

    async def fake_list_tools(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(routes, "list_tools_async", fake_list_tools)
    request = McpServerTestRequest(
        server_id = "cal", url = url, use_oauth = True, oauth_client_id = "client-id"
    )
    asyncio.run(routes.test_mcp_server(request, current_subject = "u"))
    assert captured["oauth_client_secret"] == expected


def test_legacy_database_gains_the_oauth_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", set())
    path = mcp_servers_db.studio_db_path()
    path.parent.mkdir(parents = True, exist_ok = True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE mcp_servers (id TEXT PRIMARY KEY, display_name TEXT NOT NULL,"
            " url TEXT NOT NULL, headers_json TEXT, is_enabled INTEGER NOT NULL DEFAULT 1,"
            " created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
    mcp_servers_db.create_server(id = "cal", display_name = "C", url = URL, use_oauth = True, **CREDS)
    assert mcp_client.oauth_client_kwargs(mcp_servers_db.get_server("cal")) == CREDS
