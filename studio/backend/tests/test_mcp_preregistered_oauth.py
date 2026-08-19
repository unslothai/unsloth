# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import sqlite3
import stat

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


def test_preregistered_clients_use_isolated_token_stores(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_client, "_oauth_client_token_stores", {})

    first = mcp_client._oauth_store("client-a")
    second = mcp_client._oauth_store("client-b")

    assert first is not second
    assert first._data_directory != second._data_directory


def test_preregistered_oauth_client_info_is_encrypted_at_rest(tmp_path, monkeypatch):
    from fastmcp.client.auth import OAuth
    from mcp.shared.auth import OAuthClientInformationFull

    client_secret = "configured-client-secret"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_client, "_oauth_client_token_stores", {})
    auth = OAuth(
        mcp_url = "https://calendarmcp.googleapis.com/mcp/v1",
        token_storage = mcp_client._oauth_store("configured-client-id"),
    )
    client_info = OAuthClientInformationFull(
        client_id = "configured-client-id",
        client_secret = client_secret,
        redirect_uris = ["http://localhost/callback"],
    )

    asyncio.run(auth.token_storage_adapter.set_client_info(client_info))

    token_directory = tmp_path / "mcp-oauth-tokens"
    persisted = b"".join(path.read_bytes() for path in token_directory.rglob("*") if path.is_file())
    assert client_secret.encode() not in persisted
    restored = asyncio.run(auth.token_storage_adapter.get_client_info())
    assert restored == client_info


def test_oauth_store_reads_legacy_values_and_rejects_empty_ciphertext():
    adapter = mcp_client._encrypted_oauth_serialization_adapter()
    legacy = {"value": {"access_token": "legacy-token"}}

    assert adapter.prepare_load(legacy) == legacy
    with pytest.raises(ValueError, match = "encrypted OAuth store value is empty"):
        adapter.prepare_load({"value": None})


def test_dynamic_oauth_tokens_are_encrypted_at_rest(tmp_path, monkeypatch):
    from fastmcp.client.auth import OAuth
    from mcp.shared.auth import OAuthToken

    access_token = "distinctive-access-token"
    refresh_token = "distinctive-refresh-token"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_client, "_oauth_token_store", None)
    auth = OAuth(
        mcp_url = "https://dynamic.example/mcp",
        token_storage = mcp_client._oauth_store(),
    )
    tokens = OAuthToken(access_token = access_token, refresh_token = refresh_token)

    asyncio.run(auth.token_storage_adapter.set_tokens(tokens))

    token_directory = tmp_path / "mcp-oauth-tokens"
    persisted = b"".join(path.read_bytes() for path in token_directory.rglob("*") if path.is_file())
    assert access_token.encode() not in persisted
    assert refresh_token.encode() not in persisted
    assert asyncio.run(auth.token_storage_adapter.get_tokens()) == tokens


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
    raw_connection = sqlite3.connect(tmp_path / "studio.db")
    try:
        encrypted = raw_connection.execute(
            "SELECT oauth_client_secret_encrypted FROM mcp_servers WHERE id = ?",
            ("calendar",),
        ).fetchone()[0]
    finally:
        raw_connection.close()
    assert encrypted != "configured-client-secret"
    assert "configured-client-secret" not in (tmp_path / "studio.db").read_text(encoding = "latin-1")
    assert stat.S_IMODE((tmp_path / ".mcp-oauth-client-secret.key").stat().st_mode) == 0o600


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


def test_server_listing_does_not_decrypt_stored_secrets(tmp_path, monkeypatch):
    from routes import mcp_servers as routes

    _store_calendar_server(tmp_path, monkeypatch)
    monkeypatch.setattr(
        mcp_servers_db,
        "decrypt_client_secret",
        lambda _value: (_ for _ in ()).throw(ValueError("corrupt secret")),
    )

    response = asyncio.run(routes.list_mcp_servers(current_subject = "test-user"))

    assert len(response) == 1
    assert response[0].has_oauth_client_secret is True


def test_delete_does_not_decrypt_stored_secret(tmp_path, monkeypatch):
    from routes import mcp_servers as routes

    _store_calendar_server(tmp_path, monkeypatch)
    monkeypatch.setattr(
        mcp_servers_db,
        "decrypt_client_secret",
        lambda _value: (_ for _ in ()).throw(ValueError("corrupt secret")),
    )

    async def clear_tokens(*_args):
        return None

    monkeypatch.setattr(routes, "clear_oauth_tokens_async", clear_tokens)
    asyncio.run(routes.delete_mcp_server("calendar", current_subject = "test-user"))

    assert mcp_servers_db.get_server("calendar") is None


def test_metadata_update_can_clear_corrupt_stored_secret(tmp_path, monkeypatch):
    from models.mcp_servers import McpServerUpdate
    from routes import mcp_servers as routes

    _store_calendar_server(tmp_path, monkeypatch)
    monkeypatch.setattr(
        mcp_servers_db,
        "decrypt_client_secret",
        lambda _value: (_ for _ in ()).throw(ValueError("corrupt secret")),
    )

    response = asyncio.run(
        routes.update_mcp_server(
            "calendar",
            McpServerUpdate(display_name = "Recovered", oauth_client_secret = None),
            current_subject = "test-user",
        )
    )

    assert response.display_name == "Recovered"
    assert response.has_oauth_client_secret is False


def test_disabled_server_secret_is_not_decrypted_during_chat_discovery(tmp_path, monkeypatch):
    from core.inference import tools

    _store_calendar_server(tmp_path, monkeypatch)
    mcp_servers_db.update_server("calendar", {"is_enabled": False})
    monkeypatch.setattr(
        mcp_servers_db,
        "decrypt_client_secret",
        lambda _value: (_ for _ in ()).throw(ValueError("corrupt secret")),
    )

    assert asyncio.run(tools.get_enabled_mcp_tools()) == []


def test_server_disabled_after_metadata_snapshot_is_not_decrypted(tmp_path, monkeypatch):
    from core.inference import tools

    _store_calendar_server(tmp_path, monkeypatch)
    real_list_servers = mcp_servers_db.list_servers

    def list_then_disable(*, decrypt_secrets = True):
        rows = real_list_servers(decrypt_secrets = decrypt_secrets)
        mcp_servers_db.update_server("calendar", {"is_enabled": False})
        return rows

    monkeypatch.setattr(mcp_servers_db, "list_servers", list_then_disable)
    monkeypatch.setattr(
        mcp_servers_db,
        "decrypt_client_secret",
        lambda _value: (_ for _ in ()).throw(ValueError("disabled secret was decrypted")),
    )

    assert asyncio.run(tools.get_enabled_mcp_tools()) == []


def test_corrupt_enabled_server_does_not_hide_healthy_tools(monkeypatch):
    from core.inference import tools

    corrupt_id = "corrupt"
    healthy = {
        "id": "healthy",
        "display_name": "Healthy",
        "url": "https://healthy.example/mcp",
        "is_enabled": True,
        "use_oauth": False,
    }
    metadata = [{"id": corrupt_id, "is_enabled": True}, healthy]

    def get_enabled_server(server_id):
        if server_id == corrupt_id:
            raise ValueError("corrupt secret")
        return healthy

    async def list_tools(**_kwargs):
        return [{"name": "ping", "description": "Ping", "inputSchema": {}}]

    cached_tools = {}
    monkeypatch.setattr(mcp_servers_db, "list_servers", lambda **_kwargs: metadata)
    monkeypatch.setattr(mcp_servers_db, "get_enabled_server", get_enabled_server)
    monkeypatch.setattr(tools, "get_cached_tools", cached_tools.get)
    monkeypatch.setattr(tools, "cache_tools", cached_tools.__setitem__)
    monkeypatch.setattr(tools, "in_failure_cooloff", lambda _server_id: False)
    monkeypatch.setattr(tools, "list_tools_async", list_tools)

    specs = asyncio.run(tools.get_enabled_mcp_tools())

    assert [spec["function"]["name"] for spec in specs] == ["mcp__healthy__ping"]


def test_changing_client_id_clears_stored_secret(tmp_path, monkeypatch):
    from models.mcp_servers import McpServerUpdate
    from routes import mcp_servers as routes

    _store_calendar_server(tmp_path, monkeypatch)

    async def clear_tokens(_url, _client_id):
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


def test_changing_url_clears_stored_secret(tmp_path, monkeypatch):
    from models.mcp_servers import McpServerUpdate
    from routes import mcp_servers as routes

    _store_calendar_server(tmp_path, monkeypatch)

    async def clear_tokens(_url, _client_id):
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
    assert {"oauth_client_id", "oauth_client_secret_encrypted"} <= columns


def test_secret_key_permission_failure_cleans_up_and_raises(tmp_path, monkeypatch):
    from storage import mcp_oauth_secret_crypto

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", False)

    def fail_file_chmod(_path, _mode):
        raise PermissionError("cannot protect secret key")

    monkeypatch.setattr(mcp_oauth_secret_crypto.os, "chmod", fail_file_chmod)
    with pytest.raises(PermissionError, match = "cannot protect secret key"):
        mcp_servers_db.create_server(
            id = "calendar",
            display_name = "Calendar",
            url = "https://calendarmcp.googleapis.com/mcp/v1",
            use_oauth = True,
            oauth_client_id = "configured-client-id",
            oauth_client_secret = "configured-client-secret",
        )
    assert not (tmp_path / ".mcp-oauth-client-secret.key").exists()


def test_secret_key_creation_race_uses_winning_key(tmp_path, monkeypatch):
    from cryptography.fernet import Fernet
    from storage import mcp_oauth_secret_crypto

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    winning_key = Fernet.generate_key()

    def racing_link(_source, destination):
        destination.write_bytes(winning_key)
        destination.chmod(0o600)
        raise FileExistsError

    monkeypatch.setattr(mcp_oauth_secret_crypto.os, "link", racing_link)
    assert mcp_oauth_secret_crypto._load_or_create_key() == winning_key


def test_secret_key_write_failure_closes_descriptor(tmp_path, monkeypatch):
    from storage import mcp_oauth_secret_crypto

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    real_close = mcp_oauth_secret_crypto.os.close
    closed = []

    def fail_fdopen(_descriptor, _mode):
        raise RuntimeError("cannot open key descriptor")

    def record_close(descriptor):
        closed.append(descriptor)
        return real_close(descriptor)

    monkeypatch.setattr(mcp_oauth_secret_crypto.os, "fdopen", fail_fdopen)
    monkeypatch.setattr(mcp_oauth_secret_crypto.os, "close", record_close)
    with pytest.raises(RuntimeError, match = "cannot open key descriptor"):
        mcp_oauth_secret_crypto._load_or_create_key()
    assert closed


def test_secret_key_missing_temp_cleanup_is_safe(tmp_path, monkeypatch):
    from storage import mcp_oauth_secret_crypto

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))

    def remove_then_fail(source, _destination):
        source.unlink()
        raise RuntimeError("link failed")

    monkeypatch.setattr(mcp_oauth_secret_crypto.os, "link", remove_then_fail)
    with pytest.raises(RuntimeError, match = "link failed"):
        mcp_oauth_secret_crypto._load_or_create_key()


def test_tampered_oauth_secret_is_rejected(tmp_path, monkeypatch):
    from storage import mcp_oauth_secret_crypto

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    encrypted = mcp_oauth_secret_crypto.encrypt_client_secret("secret")
    tampered = f"{encrypted[:-1]}{'A' if encrypted[-1] != 'A' else 'B'}"
    with pytest.raises(ValueError, match = "cannot be decrypted"):
        mcp_oauth_secret_crypto.decrypt_client_secret(tampered)


def test_missing_secret_key_is_not_recreated_during_decrypt(tmp_path, monkeypatch):
    from storage import mcp_oauth_secret_crypto

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    encrypted = mcp_oauth_secret_crypto.encrypt_client_secret("secret")
    key_path = tmp_path / ".mcp-oauth-client-secret.key"
    key_path.unlink()

    with pytest.raises(FileNotFoundError, match = "key is missing"):
        mcp_oauth_secret_crypto.decrypt_client_secret(encrypted)
    assert not key_path.exists()


def test_secret_key_publication_fsyncs_parent_directory(tmp_path, monkeypatch):
    from storage import mcp_oauth_secret_crypto

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    synced = []

    def record_fsync(path):
        synced.append(path)

    monkeypatch.setattr(mcp_oauth_secret_crypto, "_fsync_directory", record_fsync)
    mcp_oauth_secret_crypto.encrypt_client_secret("secret")
    assert synced == [tmp_path, tmp_path]


def test_existing_secret_key_is_reused_for_encryption(tmp_path, monkeypatch):
    from storage import mcp_oauth_secret_crypto

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    first = mcp_oauth_secret_crypto.encrypt_client_secret("first")
    second = mcp_oauth_secret_crypto.encrypt_client_secret("second")
    assert mcp_oauth_secret_crypto.decrypt_client_secret(first) == "first"
    assert mcp_oauth_secret_crypto.decrypt_client_secret(second) == "second"


def _install_fake_dpapi(
    monkeypatch,
    crypto,
    *,
    fail = False,
):
    buffers = []
    calls = []

    class Transform:
        argtypes = None
        restype = None

        def __init__(self, decrypt):
            self.decrypt = decrypt

        def __call__(self, input_pointer, *_args):
            calls.append(("decrypt" if self.decrypt else "encrypt", _args[-2]))
            if fail:
                return 0
            input_blob = crypto.ctypes.cast(
                input_pointer,
                crypto.ctypes.POINTER(crypto._DataBlob),
            ).contents
            raw = crypto.ctypes.string_at(input_blob.data, input_blob.size)
            transformed = raw.removeprefix(b"protected:") if self.decrypt else b"protected:" + raw
            buffer = crypto.ctypes.create_string_buffer(transformed)
            buffers.append(buffer)
            output_blob = crypto.ctypes.cast(
                _args[-1],
                crypto.ctypes.POINTER(crypto._DataBlob),
            ).contents
            output_blob.size = len(transformed)
            output_blob.data = crypto.ctypes.cast(
                buffer,
                crypto.ctypes.POINTER(crypto.ctypes.c_ubyte),
            )
            return 1

    class LocalFree:
        argtypes = None
        restype = None

        def __call__(self, _pointer):
            calls.append(("free", None))
            return None

    protect = Transform(decrypt = False)
    unprotect = Transform(decrypt = True)
    local_free = LocalFree()

    def load_dll(name, use_last_error):
        assert use_last_error is True
        if name == "crypt32":
            return type(
                "Crypt32",
                (),
                {
                    "CryptProtectData": protect,
                    "CryptUnprotectData": unprotect,
                },
            )()
        assert name == "kernel32"
        return type("Kernel32", (), {"LocalFree": local_free})()

    monkeypatch.setattr(crypto.ctypes, "WinDLL", load_dll, raising = False)
    monkeypatch.setattr(crypto.ctypes, "set_last_error", lambda _value: None, raising = False)
    monkeypatch.setattr(crypto.ctypes, "get_last_error", lambda: 5, raising = False)
    monkeypatch.setattr(
        crypto.ctypes,
        "WinError",
        lambda code: OSError(code, "DPAPI failed"),
        raising = False,
    )
    return calls


def test_windows_dpapi_round_trip_uses_current_user_protection(tmp_path, monkeypatch):
    from storage import mcp_oauth_secret_crypto

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_oauth_secret_crypto, "_IS_WINDOWS", True)
    calls = _install_fake_dpapi(monkeypatch, mcp_oauth_secret_crypto)

    encrypted = mcp_oauth_secret_crypto.encrypt_client_secret("secret")
    assert encrypted.startswith("dpapi:")
    assert mcp_oauth_secret_crypto.decrypt_client_secret(encrypted) == "secret"
    assert calls == [
        ("encrypt", mcp_oauth_secret_crypto._CRYPTPROTECT_UI_FORBIDDEN),
        ("free", None),
        ("decrypt", mcp_oauth_secret_crypto._CRYPTPROTECT_UI_FORBIDDEN),
        ("free", None),
    ]
    assert not (tmp_path / ".mcp-oauth-client-secret.key").exists()


def test_windows_dpapi_failure_surfaces_immediately(monkeypatch):
    from storage import mcp_oauth_secret_crypto

    monkeypatch.setattr(mcp_oauth_secret_crypto, "_IS_WINDOWS", True)
    _install_fake_dpapi(monkeypatch, mcp_oauth_secret_crypto, fail = True)
    with pytest.raises(OSError, match = "DPAPI failed"):
        mcp_oauth_secret_crypto.encrypt_client_secret("secret")


def test_windows_ciphertext_is_rejected_off_windows(monkeypatch):
    from storage import mcp_oauth_secret_crypto
    monkeypatch.setattr(mcp_oauth_secret_crypto, "_IS_WINDOWS", False)
    with pytest.raises(ValueError, match = "Windows-protected"):
        mcp_oauth_secret_crypto.decrypt_client_secret("dpapi:c2VjcmV0")


def test_posix_ciphertext_is_rejected_on_windows(tmp_path, monkeypatch):
    from storage import mcp_oauth_secret_crypto

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_oauth_secret_crypto, "_IS_WINDOWS", False)
    encrypted = mcp_oauth_secret_crypto.encrypt_client_secret("secret")
    monkeypatch.setattr(mcp_oauth_secret_crypto, "_IS_WINDOWS", True)
    with pytest.raises(ValueError, match = "cannot be decrypted on Windows"):
        mcp_oauth_secret_crypto.decrypt_client_secret(encrypted)


def test_unknown_secret_encryption_format_is_rejected():
    from storage import mcp_oauth_secret_crypto
    with pytest.raises(ValueError, match = "unknown encryption format"):
        mcp_oauth_secret_crypto.decrypt_client_secret("plaintext-secret")


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
