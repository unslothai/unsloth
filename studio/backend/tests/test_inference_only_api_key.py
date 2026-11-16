# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An inference only key reaches the chat surface and nothing else; a full key keeps every route (#17819)."""

import asyncio
import sqlite3

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

from auth import authentication, storage
from core.inference import load_policy as lp
from routes import auth as auth_routes

from .mcp_harness import isolated_auth, owner_auth  # noqa: F401  (fixtures)

ALLOWED = [
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/messages"),
    ("POST", "/v1/embeddings"),
    ("GET", "/v1/models"),
    ("GET", "/v1/models/some-model"),
    ("GET", "/api/inference/loaded-models"),
]
REFUSED = [
    ("POST", "/api/inference/load"),
    ("POST", "/api/inference/unload"),
    ("POST", "/v1/load"),
    ("PUT", "/api/settings/openai-auto-switch"),
    ("DELETE", "/api/models/delete-cached"),
    ("POST", "/api/train/start"),
]


def _app():
    app = FastAPI()

    async def handler(subject: str = Depends(authentication.get_current_subject)):
        from state.tool_policy import get_tool_policy
        return {
            "subject": subject,
            "policy": lp.current_load_policy().value,
            "tools_off": get_tool_policy() is False,
        }

    for method, path in ALLOWED + REFUSED:
        app.add_api_route(path, handler, methods = [method])
    return TestClient(app, raise_server_exceptions = False)


def _key(access = None):
    raw, _row = storage.create_api_key(username = "unsloth", name = "remote", access = access)
    return {"Authorization": f"Bearer {raw}"}


@pytest.mark.usefixtures("owner_auth")
@pytest.mark.parametrize("method, path", ALLOWED + REFUSED)
def test_a_full_key_keeps_every_route(method, path):
    for access in (None, "full"):
        response = _app().request(method, path, headers = _key(access))
        assert response.status_code == 200, response.text
        assert response.json()["policy"] == "any"
        assert response.json()["tools_off"] is False


@pytest.mark.usefixtures("owner_auth")
@pytest.mark.parametrize("method, path", ALLOWED)
def test_an_inference_key_chats_but_never_loads(method, path):
    response = _app().request(method, path, headers = _key("inference"))
    assert response.status_code == 200, response.text
    assert response.json()["policy"] == "never"
    # Server-side tools (python, terminal, skill writes) stay off, as for keyless access.
    assert response.json()["tools_off"] is True


@pytest.mark.usefixtures("owner_auth")
@pytest.mark.parametrize("method, path", REFUSED)
def test_an_inference_key_is_refused_model_management(method, path):
    response = _app().request(method, path, headers = _key("inference"))
    assert response.status_code == 403
    assert "inference only" in response.json()["detail"]


@pytest.mark.usefixtures("owner_auth")
def test_credentials_without_a_route_refuse_an_inference_key():
    raw, _ = storage.create_api_key(username = "unsloth", name = "remote", access = "inference")
    bearer = HTTPAuthorizationCredentials(scheme = "Bearer", credentials = raw)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(authentication._get_current_credential(bearer, allow_password_change = False))
    assert exc.value.status_code == 403


@pytest.mark.usefixtures("owner_auth")
def test_an_unknown_stored_access_reads_as_inference_only():
    raw, row = storage.create_api_key(username = "unsloth", name = "remote")
    conn = storage.get_connection()
    try:
        conn.execute("UPDATE api_keys SET access = 'bogus' WHERE id = ?", (row["id"],))
        conn.commit()
    finally:
        conn.close()
    assert storage.validate_api_key_account(raw, touch = False)[0]["key_access"] == "inference"


@pytest.mark.usefixtures("isolated_auth")
def test_a_database_from_before_the_column_keeps_its_keys_full():
    from .mcp_harness import seed_owner

    seed_owner()
    raw, row = storage.create_api_key(username = "unsloth", name = "old")
    conn = sqlite3.connect(storage.DB_PATH)
    conn.execute("ALTER TABLE api_keys DROP COLUMN access")
    conn.commit()
    conn.close()
    storage._auth_schema_ready.clear()
    storage._api_key_hash_cache.clear()
    for _ in range(2):
        assert storage.validate_api_key_account(raw, touch = False)[0]["key_access"] == "full"
        storage._auth_schema_ready.clear()


@pytest.mark.usefixtures("owner_auth")
def test_the_create_route_stores_and_lists_the_access():
    app = FastAPI()
    app.include_router(auth_routes.router, prefix = "/api/auth")
    client = TestClient(app)
    ui = _key()
    made = client.post(
        "/api/auth/api-keys", json = {"name": "lan", "access": "inference"}, headers = ui
    )
    assert made.status_code == 200, made.text
    assert made.json()["api_key"]["access"] == "inference"
    default = client.post("/api/auth/api-keys", json = {"name": "plain"}, headers = ui)
    assert default.json()["api_key"]["access"] == "full"
    listed = {
        k["name"]: k["access"]
        for k in client.get("/api/auth/api-keys", headers = ui).json()["api_keys"]
    }
    assert listed["lan"] == "inference" and listed["plain"] == "full"
    child = client.post(
        "/api/auth/api-keys",
        json = {"name": "child"},
        headers = {"Authorization": f"Bearer {made.json()['key']}"},
    )
    assert child.status_code == 403


@pytest.mark.usefixtures("owner_auth")
def test_the_mcp_gate_refuses_an_inference_key():
    from studio_mcp import gate

    raw, _ = storage.create_api_key(username = "unsloth", name = "remote", access = "inference")
    assert gate._validate_key(raw) == (None, gate.INFERENCE_ONLY_KEY)
    full, _ = storage.create_api_key(username = "unsloth", name = "remote-full")
    assert gate._validate_key(full)[0] is not None


@pytest.mark.usefixtures("owner_auth")
def test_a_chat_started_by_an_inference_key_gets_an_inference_tools_key():
    from core.inference import studio_tools_mcp
    from utils.account_context import OWNER, as_api_caller
    from dataclasses import replace

    narrow = replace(as_api_caller(OWNER), load_policy = "never")
    _raw, row = studio_tools_mcp._mint(narrow, 60)
    assert row["access"] == "inference"
    _raw, row = studio_tools_mcp._mint(as_api_caller(OWNER), 60)
    assert row["access"] is None


@pytest.mark.usefixtures("owner_auth")
def test_a_managed_key_restored_from_its_copy_stays_inference_only():
    alice = storage.issue_account_setup_code(username = "alice")["account"]["account_id"]
    raw, _row = storage.create_api_key("alice", name = "lan", account_id = alice, access = "inference")
    conn = sqlite3.connect(storage.DB_PATH)
    try:
        # An older build's bulk delete; the next start restores managed keys from their copy.
        conn.execute("DELETE FROM api_keys")
        conn.commit()
        conn.row_factory = sqlite3.Row
        storage._account_keys_synced.clear()
        storage._ensure_account_api_keys(conn, {"account_id"})
    finally:
        conn.close()
    assert storage.validate_api_key_account(raw, touch = False)[0]["key_access"] == "inference"


@pytest.mark.usefixtures("owner_auth")
def test_the_access_can_be_changed_after_the_key_is_shown():
    app = FastAPI()
    app.include_router(auth_routes.router, prefix = "/api/auth")
    client = TestClient(app)
    ui = _key()
    made = client.post("/api/auth/api-keys", json = {"name": "lan"}, headers = ui).json()
    key_id, remote = made["api_key"]["id"], {"Authorization": f"Bearer {made['key']}"}
    assert _app().post("/api/inference/load", headers = remote).status_code == 200

    narrowed = client.patch(
        f"/api/auth/api-keys/{key_id}", json = {"access": "inference"}, headers = ui
    )
    assert narrowed.status_code == 200 and narrowed.json()["access"] == "inference"
    assert _app().post("/api/inference/load", headers = remote).status_code == 403
    assert _app().post("/v1/chat/completions", headers = remote).status_code == 200
    assert (
        client.patch(
            f"/api/auth/api-keys/{key_id}", json = {"access": "full"}, headers = remote
        ).status_code
        == 403
    )

    widened = client.patch(f"/api/auth/api-keys/{key_id}", json = {"access": "full"}, headers = ui)
    assert widened.json()["access"] == "full"
    assert _app().post("/api/inference/load", headers = remote).status_code == 200
    assert (
        client.patch(
            f"/api/auth/api-keys/{key_id}", json = {"access": "full"}, headers = ui
        ).status_code
        == 200
    )
    assert (
        client.patch(
            f"/api/auth/api-keys/{key_id}", json = {"access": "admin"}, headers = ui
        ).status_code
        == 422
    )


@pytest.mark.usefixtures("owner_auth")
def test_internal_and_revoked_keys_keep_their_access():
    _raw, internal = storage.create_api_key(username = "unsloth", name = "Data recipe", internal = True)
    assert storage.set_api_key_access("unsloth", internal["id"], "inference") is None
    _raw, row = storage.create_api_key(username = "unsloth", name = "gone")
    storage.revoke_api_key("unsloth", row["id"])
    assert storage.set_api_key_access("unsloth", row["id"], "inference") is None
    assert storage.set_api_key_access("someone-else", _key_id_of("unsloth"), "inference") is None


def _key_id_of(username):
    _raw, row = storage.create_api_key(username = username, name = "mine")
    return row["id"]
