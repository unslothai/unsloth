# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""The account endpoint reports the authenticated identity without exposing credentials."""

import importlib.util
import secrets
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import policy, storage


@pytest.fixture()
def account_client(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    policy.invalidate_account_cache()
    storage.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(32))
    alice_id = storage.issue_account_setup_code(username = "alice")["account"]["account_id"]
    owner_key, _ = storage.create_api_key("unsloth", "owner")
    alice_key, _ = storage.create_api_key("alice", "alice", account_id = alice_id)
    route_path = Path(storage.__file__).resolve().parents[1] / "routes" / "auth.py"
    spec = importlib.util.spec_from_file_location("_picker_account_auth_route", route_path)
    auth_routes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auth_routes)
    app = FastAPI()
    app.include_router(auth_routes.router, prefix = "/api/auth")
    try:
        with TestClient(app) as client:
            yield client, {"owner": owner_key, "alice": alice_key}, alice_id
    finally:
        policy.invalidate_account_cache()


@pytest.mark.parametrize("name", ["owner", "alice"])
def test_account_endpoint_uses_the_api_keys_account(account_client, name):
    from utils.account_context import current_account

    client, keys, alice_id = account_client
    response = client.get("/api/auth/account", headers = {"Authorization": f"Bearer {keys[name]}"})
    assert response.status_code == 200
    assert response.json() == {
        "account_id": "owner" if name == "owner" else alice_id,
        "username": "unsloth" if name == "owner" else "alice",
        "role": "owner" if name == "owner" else "user",
        "via_api_key": True,
    }
    assert current_account().is_owner


@pytest.mark.parametrize("key", [None, "sk-unsloth-invalid"])
def test_account_endpoint_requires_authentication(account_client, key):
    client, _, _ = account_client
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    response = client.get("/api/auth/account", headers = headers)
    assert response.status_code == 401
    assert "account_id" not in response.json()


@pytest.mark.parametrize(
    "restriction", ["allowlist", "private-repo", "none-cached-allowed", "unrestricted"]
)
def test_model_access_uses_the_authenticated_account_policy(
    account_client, monkeypatch, restriction
):
    from hub.services.models import account_access

    client, keys, alice_id = account_client
    if restriction in ("allowlist", "none-cached-allowed"):
        storage.set_account_model_allowlist(
            alice_id, ["org/NotCached"] if restriction == "none-cached-allowed" else ["org/Allowed"]
        )
    if restriction == "unrestricted":
        storage.set_account_model_allowlist(alice_id, [])
    monkeypatch.setattr(
        account_access,
        "repo_is_public",
        lambda repo, *args, **kwargs: restriction in ("allowlist", "unrestricted")
        or repo.lower() == "org/allowed",
    )
    response = client.post(
        "/api/auth/model-access",
        json = ["org/Denied", "org/Allowed"],
        headers = {"Authorization": f"Bearer {keys['alice']}"},
    )
    assert response.status_code == 200
    expected = ["org/Allowed"]
    if restriction == "none-cached-allowed":
        expected = []
    elif restriction == "unrestricted":
        expected = ["org/Denied", "org/Allowed"]
    assert response.json() == {"allowed": expected}


def test_model_access_preserves_owner_access(account_client):
    client, keys, _ = account_client
    models = ["org/Private", "org/Other", r"\\server\share\model.gguf"]
    response = client.post(
        "/api/auth/model-access",
        json = models,
        headers = {"Authorization": f"Bearer {keys['owner']}"},
    )
    assert response.status_code == 200
    assert response.json() == {"allowed": models}


def test_model_access_does_not_probe_network_paths(account_client, monkeypatch):
    from hub.services.models import account_access

    def no_filesystem_probe(*args, **kwargs):
        pytest.fail("network paths must be refused before a filesystem probe")

    monkeypatch.setattr(account_access, "Path", no_filesystem_probe)
    client, keys, _ = account_client
    response = client.post(
        "/api/auth/model-access",
        json = [r"\\server\share\model.gguf"],
        headers = {"Authorization": f"Bearer {keys['alice']}"},
    )
    assert response.status_code == 200
    assert response.json() == {"allowed": []}


@pytest.mark.parametrize("key", [None, "sk-unsloth-invalid"])
def test_model_access_requires_authentication(account_client, key):
    client, _, _ = account_client
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    response = client.post("/api/auth/model-access", json = ["org/Allowed"], headers = headers)
    assert response.status_code == 401
