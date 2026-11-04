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
