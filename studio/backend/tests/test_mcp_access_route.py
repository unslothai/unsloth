# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import policy, storage
from auth.authentication import create_access_token
from routes import settings
from utils.keyless_api_access import set_keyless_api_access
from utils.mcp_access import ENV_FORCE, get_mcp_enabled

from .mcp_harness import isolated_auth, owner_auth  # noqa: F401  (fixtures)

UI_ONLY = "Agent access (MCP) can only be changed from the Unsloth UI."
URL = "/api/settings/mcp-access"
pytestmark = pytest.mark.usefixtures("owner_auth")


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(settings.router, prefix = "/api/settings")
    app.state.bind_host = "127.0.0.1"
    app.state.secure = False
    app.state.remote_access_is_colab = False
    app.state.lan_access_is_colab = False
    app.state.lan_access_secure_launch = False
    app.state.cloudflare_url = None
    with TestClient(app, base_url = "http://127.0.0.1:8888", client = ("127.0.0.1", 50000)) as http:
        yield http


def _owner_session():
    return {
        "Authorization": f"Bearer {create_access_token(subject = storage.DEFAULT_ADMIN_USERNAME)}"
    }


def _alice_session():
    storage.issue_account_setup_code(username = "alice")
    record = storage.get_user_record("alice")
    secret = storage.update_account_password(
        "alice",
        "alice-password-123",
        expect_password_hash = record["password_hash"],
        expect_secret = record["jwt_secret"],
    )
    policy.invalidate_account_cache()
    return {"Authorization": f"Bearer {create_access_token(subject = 'alice', secret = secret)}"}


def test_the_owner_session_reads_and_turns_it_on(client):
    got = client.get(URL, headers = _owner_session())
    assert got.status_code == 200, got.text
    assert got.json() == {
        "enabled": False,
        "forced_by_env": False,
        "url": "http://127.0.0.1:8888/mcp/",
    }

    put = client.put(URL, json = {"enabled": True}, headers = _owner_session())
    assert put.status_code == 200, put.text
    assert put.json()["enabled"] is True
    assert put.json()["url"].endswith("/mcp/")
    assert get_mcp_enabled() is True

    off = client.put(URL, json = {"enabled": False}, headers = _owner_session())
    assert off.json()["enabled"] is False
    assert get_mcp_enabled() is False


@pytest.mark.parametrize("value", ["true", 1, None])
def test_the_payload_must_be_a_strict_bool(client, value):
    response = client.put(URL, json = {"enabled": value}, headers = _owner_session())
    assert response.status_code == 422
    assert get_mcp_enabled() is False


@pytest.mark.parametrize("method", ["GET", "PUT"])
def test_an_owner_api_key_is_refused(client, method):
    raw_key, _row = storage.create_api_key(storage.DEFAULT_ADMIN_USERNAME, name = "agent")
    response = client.request(
        method, URL, json = {"enabled": True}, headers = {"Authorization": f"Bearer {raw_key}"}
    )
    assert response.status_code == 403
    assert response.json()["detail"] == UI_ONLY
    assert get_mcp_enabled() is False


@pytest.mark.parametrize("method", ["GET", "PUT"])
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer not-needed"}])
def test_a_keyless_caller_is_refused(client, method, headers):
    set_keyless_api_access("full", tools = True)
    response = client.request(method, URL, json = {"enabled": True}, headers = headers)
    assert response.status_code == 403
    assert response.json()["detail"] == UI_ONLY
    assert get_mcp_enabled() is False


@pytest.mark.parametrize("method", ["GET", "PUT"])
def test_a_managed_account_session_is_refused(client, method):
    response = client.request(method, URL, json = {"enabled": True}, headers = _alice_session())
    assert response.status_code == 403
    assert response.json()["detail"] == "Only the installation owner can do this"
    assert get_mcp_enabled() is False


def test_a_put_while_forced_by_env_is_a_conflict(client, monkeypatch):
    monkeypatch.setenv(ENV_FORCE, "1")
    got = client.get(URL, headers = _owner_session())
    assert got.json()["enabled"] is True
    assert got.json()["forced_by_env"] is True

    response = client.put(URL, json = {"enabled": False}, headers = _owner_session())
    assert response.status_code == 409
    assert response.json()["detail"] == "Set by UNSLOTH_STUDIO_ENABLE_MCP."
    assert get_mcp_enabled() is False
