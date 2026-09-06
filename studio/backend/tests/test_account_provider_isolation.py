# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Provider, MCP and remote-code trust boundaries use the authenticated account DB."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import policy
from auth.authentication import authenticated_via_api_key, get_current_credential, get_current_subject
from hub.services.models import account_access as access
from routes import mcp_servers, provider_credentials, providers
from storage import credential_secrets, mcp_servers_db, providers_db
from utils.account_context import OWNER, AccountContext, arun_as, bind_account, reset_account, run_as
from utils.security import remote_code_approvals as approvals

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")


@pytest.fixture(autouse = True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    # The owner has already initialized these modules: each new DB must still get its schema.
    for module in (credential_secrets, mcp_servers_db, providers_db):
        monkeypatch.setattr(module, "_schema_ready", set())
    monkeypatch.setattr(access, "_schema_paths", set())
    monkeypatch.setattr(credential_secrets, "get_or_create_credential_encryption_key", lambda: b"k" * 32)
    monkeypatch.setattr(providers, "current_credential_write", lambda credential: nullcontext())


def client_for(account):
    app = FastAPI()

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    app.dependency_overrides[get_current_credential] = lambda: (account.username, None)
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    app.include_router(providers.router, prefix = "/providers")
    app.include_router(mcp_servers.router, prefix = "/mcp")
    return TestClient(app)


@pytest.mark.parametrize("account,other", [(ALICE, BOB), (BOB, ALICE)])
def test_provider_object_routes_and_saved_keys_are_private(account, other):
    with client_for(account) as client:
        created = client.post("/providers/", json = {"provider_type": "openai", "display_name": "Private provider"})
        assert created.status_code == 201, created.text
        provider_id = created.json()["id"]
    run_as(account, credential_secrets.save_provider_api_key, provider_id, "private-api-key")
    with client_for(other) as client:
        assert client.get("/providers/").json() == []
        assert client.put(f"/providers/{provider_id}", json = {"display_name": "stolen"}).status_code == 404
        assert client.delete(f"/providers/{provider_id}").status_code == 404
        assert client.put(f"/providers/{provider_id}/api-key/migrate", json = {"encrypted_api_key": "ignored"}).status_code == 404
        for path in ["/providers/test", "/providers/models"]:
            response = client.post(path, json = {"provider_type": "openai", "provider_id": provider_id})
            assert response.status_code == 404, response.text
    assert run_as(other, credential_secrets.get_provider_api_key, provider_id) is None
    assert run_as(account, credential_secrets.get_provider_api_key, provider_id) == "private-api-key"
    with client_for(account) as client:
        assert client.get("/providers/").json()[0]["id"] == provider_id


@pytest.mark.parametrize("account,other", [(ALICE, BOB), (BOB, ALICE)])
def test_mcp_server_ids_are_scoped_before_session_or_tool_access(account, other, monkeypatch):
    with client_for(account) as client:
        created = client.post("/mcp/", json = {"display_name": "Private MCP", "url": "https://8.8.8.8/mcp", "headers": {"Authorization": "Bearer private"}})
        assert created.status_code == 201, created.text
        server_id = created.json()["id"]
    calls = []
    monkeypatch.setattr(mcp_servers, "list_tools_async", lambda **kwargs: calls.append(kwargs))
    with client_for(other) as client:
        assert client.get("/mcp/").json() == []
        assert client.put(f"/mcp/{server_id}", json = {"display_name": "stolen"}).status_code == 404
        assert client.delete(f"/mcp/{server_id}").status_code == 404
        assert client.post(f"/mcp/{server_id}/refresh").status_code == 404
    assert calls == []
    with client_for(account) as client:
        assert client.get("/mcp/").json()[0]["id"] == server_id


def test_same_provider_id_can_hold_different_account_credentials():
    for account, secret in [(ALICE, "alice-secret"), (BOB, "bob-secret")]:
        run_as(account, access.ensure_account_schema, credential_secrets)
        run_as(account, credential_secrets.save_provider_api_key, "same-provider-id", secret)
        run_as(account, credential_secrets.save_hf_token, secret)
    assert run_as(ALICE, provider_credentials.resolve_provider_api_key_or_400, "same-provider-id", None) == "alice-secret"
    assert run_as(BOB, provider_credentials.resolve_provider_api_key_or_400, "same-provider-id", None) == "bob-secret"
    assert run_as(ALICE, credential_secrets.get_hf_token) == "alice-secret"
    assert run_as(BOB, credential_secrets.get_hf_token) == "bob-secret"


def test_provider_config_locks_do_not_serialize_unrelated_account_ids():
    async def check():
        a = await arun_as(ALICE, _get_lock())
        b = await arun_as(BOB, _get_lock())
        assert a is not b
        assert a is await arun_as(ALICE, _get_lock())

    async def _get_lock():
        return provider_credentials.provider_config_guard("same-id")

    asyncio.run(check())


def test_remote_code_approvals_are_account_owned_and_survive_renames(tmp_path):
    run_as(ALICE, approvals.record, "alice", "org/private", commit_sha = "abc", fingerprint = "safe", max_severity = None)
    assert run_as(ALICE, approvals.lookup, "alice", "org/private") is not None
    assert run_as(BOB, approvals.lookup, "alice", "org/private") is None
    renamed = AccountContext(ALICE.account_id, "new-alice")
    assert run_as(renamed, approvals.lookup, "new-alice", "org/private") is not None
    replacement = AccountContext("c" * 32, "alice")
    assert run_as(replacement, approvals.lookup, "alice", "org/private") is None
    run_as(BOB, approvals.forget, "alice", "org/private")
    assert run_as(ALICE, approvals.lookup, "alice", "org/private") is not None
    assert run_as(OWNER, approvals._store_path) == tmp_path / "security" / "remote_code_approvals.json"
    assert run_as(ALICE, approvals._store_path) == tmp_path / "accounts" / ALICE.account_id / "security" / "remote_code_approvals.json"


def test_ambient_hf_credentials_are_never_lent_to_managed_accounts(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "owner-token")
    assert run_as(OWNER, access.ambient_hf_token) == "owner-token"
    assert run_as(ALICE, access.ambient_hf_token) is False
    assert run_as(BOB, access.account_hf_token, None) is False
    assert run_as(ALICE, access.account_hf_token, "alice-token") == "alice-token"
