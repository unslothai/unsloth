# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import sqlite3

import pytest
from fastapi import HTTPException
from core.inference import mcp_client
from models.mcp_servers import McpServerCreate, McpServerResponse, McpServerTestRequest
from storage import mcp_servers_db


def test_http_client_passes_preregistered_oauth_credentials(monkeypatch):
    captured = {}

    class FakeOAuth:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeTransport:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeClient:
        def __init__(self, transport):
            self.transport = transport

    monkeypatch.setattr("fastmcp.Client", FakeClient)
    monkeypatch.setattr("fastmcp.client.auth.OAuth", FakeOAuth)
    monkeypatch.setattr("fastmcp.client.transports.StreamableHttpTransport", FakeTransport)
    monkeypatch.setattr("fastmcp.mcp_config.infer_transport_type_from_url", lambda _url: "http")

    mcp_client._client(
        "https://calendarmcp.googleapis.com/mcp/v1",
        None,
        use_oauth = True,
        oauth_client_id = "configured-client-id",
        oauth_client_secret = "configured-client-secret",
    )

    assert captured["client_id"] == "configured-client-id"
    assert captured["client_secret"] == "configured-client-secret"


def test_real_fastmcp_oauth_accepts_preregistered_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_client, "_oauth_token_store", None)
    client = mcp_client._client(
        "https://calendarmcp.googleapis.com/mcp/v1",
        None,
        use_oauth = True,
        oauth_client_id = "configured-client-id",
        oauth_client_secret = "configured-client-secret",
    )
    auth = client.transport.auth
    assert auth._client_id == "configured-client-id"
    assert auth._client_secret == "configured-client-secret"


def test_list_and_call_paths_forward_credentials_to_client(monkeypatch):
    captured = []

    class FakeTool:
        def model_dump(self, exclude_none):
            assert exclude_none is True
            return {"name": "list_events"}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def list_tools(self):
            return [FakeTool()]

        async def call_tool(self, _name, _args, raise_on_error):
            assert raise_on_error is False
            return "calendar-result"

    def fake_client(
        url,
        headers,
        use_oauth = False,
        **kwargs,
    ):
        captured.append((url, headers, use_oauth, kwargs))
        return FakeClient()

    monkeypatch.setattr(mcp_client, "_client", fake_client)
    credentials = {
        "oauth_client_id": "configured-client-id",
        "oauth_client_secret": "configured-client-secret",
    }
    tools = asyncio.run(
        mcp_client.list_tools_async(
            "https://calendarmcp.googleapis.com/mcp/v1",
            use_oauth = True,
            **credentials,
        )
    )
    mcp_client.call_tool_sync(
        "https://calendarmcp.googleapis.com/mcp/v1",
        None,
        "list_events",
        {},
        use_oauth = True,
        **credentials,
    )

    assert tools == [{"name": "list_events"}]
    assert [entry[3] for entry in captured] == [credentials, credentials]


def test_oauth_models_accept_credentials_without_exposing_the_secret():
    create = McpServerCreate(
        display_name = "Calendar",
        url = "https://calendarmcp.googleapis.com/mcp/v1",
        use_oauth = True,
        oauth_client_id = "configured-client-id",
        oauth_client_secret = "configured-client-secret",
    )
    probe = McpServerTestRequest(
        url = create.url,
        use_oauth = True,
        oauth_client_id = create.oauth_client_id,
        oauth_client_secret = create.oauth_client_secret,
    )
    response_fields = McpServerResponse.model_fields

    assert probe.oauth_client_id == "configured-client-id"
    assert "oauth_client_secret" not in response_fields
    assert "has_oauth_client_secret" in response_fields


def test_oauth_credentials_round_trip_in_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", False)
    mcp_servers_db.create_server(
        id = "calendar",
        display_name = "Calendar",
        url = "https://calendarmcp.googleapis.com/mcp/v1",
        use_oauth = True,
        oauth_client_id = "configured-client-id",
        oauth_client_secret = "configured-client-secret",
    )

    row = mcp_servers_db.get_server("calendar")
    assert row["oauth_client_id"] == "configured-client-id"
    assert row["oauth_client_secret"] == "configured-client-secret"
    assert row[mcp_servers_db.HAS_OAUTH_CLIENT_SECRET_KEY] is True
    masked = mcp_servers_db.get_server("calendar", include_secret = False)
    assert masked[mcp_servers_db.HAS_OAUTH_CLIENT_SECRET_KEY] is True
    # Presence, never a stand-in value: a masked row must not carry anything
    # that a connection path could hand to the OAuth client as the secret.
    assert masked["oauth_client_secret"] is None
    assert mcp_client.oauth_client_kwargs(masked)["oauth_client_secret"] is None


def test_create_response_masks_oauth_secret(tmp_path, monkeypatch):
    from routes.mcp_servers import create_mcp_server

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", False)
    response = asyncio.run(
        create_mcp_server(
            McpServerCreate(
                display_name = "Calendar",
                url = "https://calendarmcp.googleapis.com/mcp/v1",
                use_oauth = True,
                oauth_client_id = "configured-client-id",
                oauth_client_secret = "configured-client-secret",
            ),
            current_subject = "test-user",
        )
    )

    assert response.oauth_client_id == "configured-client-id"
    assert response.has_oauth_client_secret is True
    assert "oauth_client_secret" not in response.model_dump()


def test_connection_probe_reuses_stored_secret(tmp_path, monkeypatch):
    from routes import mcp_servers as routes

    captured = {}

    async def fake_list_tools(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(routes, "list_tools_async", fake_list_tools)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", False)
    mcp_servers_db.create_server(
        id = "calendar",
        display_name = "Calendar",
        url = "https://calendarmcp.googleapis.com/mcp/v1",
        use_oauth = True,
        oauth_client_id = "configured-client-id",
        oauth_client_secret = "configured-client-secret",
    )
    result = asyncio.run(
        routes.test_mcp_server(
            McpServerTestRequest(
                server_id = "calendar",
                url = "https://calendarmcp.googleapis.com/mcp/v1",
                use_oauth = True,
                oauth_client_id = "configured-client-id",
            ),
            current_subject = "test-user",
        )
    )

    assert result.ok is True
    assert captured["oauth_client_id"] == "configured-client-id"
    assert captured["oauth_client_secret"] == "configured-client-secret"


def test_connection_probe_never_reuses_secret_for_changed_url(tmp_path, monkeypatch):
    from routes import mcp_servers as routes

    _store_calendar_server(tmp_path, monkeypatch)
    captured = {}

    async def fake_list_tools(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(routes, "list_tools_async", fake_list_tools)
    result = asyncio.run(
        routes.test_mcp_server(
            McpServerTestRequest(
                server_id = "calendar",
                url = "https://attacker.example/mcp",
                use_oauth = True,
                oauth_client_id = "configured-client-id",
            ),
            current_subject = "test-user",
        )
    )
    assert result.ok is True
    assert captured["oauth_client_secret"] is None


def test_disable_oauth_clears_credentials_even_when_payload_resends_id():
    from routes.mcp_servers import _changes_from_payload
    from models.mcp_servers import McpServerUpdate

    changes = _changes_from_payload(
        McpServerUpdate(
            use_oauth = False,
            oauth_client_id = "configured-client-id",
        )
    )
    assert changes["oauth_client_id"] is None
    assert changes["oauth_client_secret"] is None


def test_secret_without_client_id_is_rejected():
    from routes.mcp_servers import _oauth_credentials
    with pytest.raises(HTTPException, match = "requires oauth_client_id"):
        _oauth_credentials(None, "orphan-secret")


def test_update_credentials_supports_clear_and_rotation():
    from models.mcp_servers import McpServerUpdate
    from routes.mcp_servers import _changes_from_payload

    cleared = _changes_from_payload(McpServerUpdate(oauth_client_id = None))
    rotated = _changes_from_payload(
        McpServerUpdate(
            oauth_client_id = "configured-client-id",
            oauth_client_secret = "rotated-secret",
        )
    )
    assert cleared["oauth_client_id"] is None
    assert cleared["oauth_client_secret"] is None
    assert rotated["oauth_client_id"] == "configured-client-id"
    assert rotated["oauth_client_secret"] == "rotated-secret"


def test_secret_only_update_reuses_stored_client_id(tmp_path, monkeypatch):
    from models.mcp_servers import McpServerUpdate
    from routes import mcp_servers as routes

    _store_calendar_server(tmp_path, monkeypatch)

    async def clear_tokens(*_args):
        return None

    monkeypatch.setattr(routes, "clear_oauth_tokens_async", clear_tokens)

    asyncio.run(
        routes.update_mcp_server(
            "calendar",
            McpServerUpdate(oauth_client_secret = "rotated-secret"),
            current_subject = "test-user",
        )
    )

    row = mcp_servers_db.get_server("calendar")
    assert row["oauth_client_id"] == "configured-client-id"
    assert row["oauth_client_secret"] == "rotated-secret"


def test_changing_client_id_clears_stored_secret(tmp_path, monkeypatch):
    from models.mcp_servers import McpServerUpdate
    from routes import mcp_servers as routes

    _store_calendar_server(tmp_path, monkeypatch)

    async def clear_tokens(_url):
        return None

    monkeypatch.setattr(routes, "clear_oauth_tokens_async", clear_tokens)
    asyncio.run(
        routes.update_mcp_server(
            "calendar",
            McpServerUpdate(oauth_client_id = "replacement-client-id"),
            current_subject = "test-user",
        )
    )
    row = mcp_servers_db.get_server("calendar")
    assert row["oauth_client_id"] == "replacement-client-id"
    assert row["oauth_client_secret"] is None


def test_removing_the_stored_secret_drops_tokens_and_cached_tools(tmp_path, monkeypatch):
    """Clearing the secret is a credential change like any other.

    `old` is a masked read, so it never carries the stored secret value; a
    comparison against it cannot see the removal. Leaving the persisted tokens
    and the cached tools in place would let the server keep answering under
    credentials the user just revoked."""
    from models.mcp_servers import McpServerUpdate
    from routes import mcp_servers as routes

    _store_calendar_server(tmp_path, monkeypatch)
    cleared_token_urls = []
    invalidated_ids = []

    async def clear_tokens(url):
        cleared_token_urls.append(url)

    monkeypatch.setattr(routes, "clear_oauth_tokens_async", clear_tokens)
    monkeypatch.setattr(routes, "invalidate_tool_cache", invalidated_ids.append)
    asyncio.run(
        routes.update_mcp_server(
            "calendar",
            # Same client ID and URL: only the secret goes away.
            McpServerUpdate(
                oauth_client_id = "configured-client-id",
                oauth_client_secret = None,
            ),
            current_subject = "test-user",
        )
    )

    assert mcp_servers_db.get_server("calendar")["oauth_client_secret"] is None
    assert cleared_token_urls == ["https://calendarmcp.googleapis.com/mcp/v1"]
    assert invalidated_ids == ["calendar"]


def test_resending_an_unchanged_client_id_keeps_tokens_and_cached_tools(tmp_path, monkeypatch):
    """The edit dialog resends every OAuth field on a rename, and a blank secret
    field means "keep the stored one" -- neither is a credential change."""
    from models.mcp_servers import McpServerUpdate
    from routes import mcp_servers as routes

    _store_calendar_server(tmp_path, monkeypatch)
    cleared_token_urls = []
    invalidated_ids = []

    async def clear_tokens(url):
        cleared_token_urls.append(url)

    monkeypatch.setattr(routes, "clear_oauth_tokens_async", clear_tokens)
    monkeypatch.setattr(routes, "invalidate_tool_cache", invalidated_ids.append)
    asyncio.run(
        routes.update_mcp_server(
            "calendar",
            McpServerUpdate(
                display_name = "Google Calendar",
                oauth_client_id = "configured-client-id",
            ),
            current_subject = "test-user",
        )
    )

    assert mcp_servers_db.get_server("calendar")["oauth_client_secret"] == (
        "configured-client-secret"
    )
    assert cleared_token_urls == []
    assert invalidated_ids == []


def test_changing_url_clears_stored_secret(tmp_path, monkeypatch):
    from models.mcp_servers import McpServerUpdate
    from routes import mcp_servers as routes

    _store_calendar_server(tmp_path, monkeypatch)

    async def clear_tokens(_url):
        return None

    monkeypatch.setattr(routes, "clear_oauth_tokens_async", clear_tokens)
    asyncio.run(
        routes.update_mcp_server(
            "calendar",
            McpServerUpdate(url = "https://replacement.example/mcp"),
            current_subject = "test-user",
        )
    )
    row = mcp_servers_db.get_server("calendar")
    assert row["url"] == "https://replacement.example/mcp"
    assert row["oauth_client_secret"] is None


def test_legacy_database_migrates_oauth_credential_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", False)
    db_path = tmp_path / "studio.db"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE mcp_servers (
                id TEXT NOT NULL PRIMARY KEY,
                display_name TEXT NOT NULL,
                url TEXT NOT NULL,
                headers_json TEXT,
                is_enabled INTEGER NOT NULL DEFAULT 1,
                use_oauth INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.commit()
    finally:
        connection.close()

    migrated = mcp_servers_db.get_connection()
    try:
        columns = {
            row["name"] for row in migrated.execute("PRAGMA table_info(mcp_servers)").fetchall()
        }
    finally:
        migrated.close()
    assert {"oauth_client_id", "oauth_client_secret"} <= columns


def _store_calendar_server(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", False)
    mcp_servers_db.create_server(
        id = "calendar",
        display_name = "Calendar",
        url = "https://calendarmcp.googleapis.com/mcp/v1",
        is_enabled = True,
        use_oauth = True,
        oauth_client_id = "configured-client-id",
        oauth_client_secret = "configured-client-secret",
    )


def test_refresh_propagates_stored_oauth_credentials(tmp_path, monkeypatch):
    from routes import mcp_servers as routes

    _store_calendar_server(tmp_path, monkeypatch)
    captured = {}

    async def fake_list_tools(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(routes, "list_tools_async", fake_list_tools)
    result = asyncio.run(routes.refresh_mcp_server_tools("calendar", current_subject = "test-user"))
    assert result.ok is True
    assert captured["oauth_client_id"] == "configured-client-id"
    assert captured["oauth_client_secret"] == "configured-client-secret"


def test_chat_discovery_propagates_stored_oauth_credentials(tmp_path, monkeypatch):
    from core.inference import tools

    _store_calendar_server(tmp_path, monkeypatch)
    captured = {}

    async def fake_list_tools(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(tools, "list_tools_async", fake_list_tools)
    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    monkeypatch.setattr(mcp_client, "_probe_cooloff_until", {})
    assert asyncio.run(tools.get_enabled_mcp_tools()) == []
    assert captured["oauth_client_id"] == "configured-client-id"
    assert captured["oauth_client_secret"] == "configured-client-secret"


def test_tool_execution_propagates_stored_oauth_credentials(tmp_path, monkeypatch):
    from core.inference import tools

    _store_calendar_server(tmp_path, monkeypatch)
    captured = {}

    def fake_call_tool(**kwargs):
        captured.update(kwargs)
        return "calendar-result"

    monkeypatch.setattr(tools, "call_tool_sync", fake_call_tool)
    result = tools.execute_tool("mcp__calendar__list_events", {})
    assert result == "calendar-result"
    assert captured["oauth_client_id"] == "configured-client-id"
    assert captured["oauth_client_secret"] == "configured-client-secret"
