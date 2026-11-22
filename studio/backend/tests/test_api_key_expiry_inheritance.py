# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A key minted with an expiring API key expires no later than that key, so it cannot outlive it."""

import importlib.util
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import storage


@pytest.fixture(autouse = True)
def isolated_auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    monkeypatch.setattr(storage, "_credential_encryption_key_cache", None)
    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = "human-password-123",
        jwt_secret = secrets.token_urlsafe(64),
    )
    yield


@pytest.fixture
def client():
    route_path = Path(__file__).resolve().parents[1] / "routes" / "auth.py"
    spec = importlib.util.spec_from_file_location("_api_key_expiry_auth_route", route_path)
    auth_route = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(auth_route)
    app = FastAPI()
    app.include_router(auth_route.router, prefix = "/api/auth")
    return TestClient(app)


def _parent(expires_at):
    raw, _row = storage.create_api_key(
        storage.DEFAULT_ADMIN_USERNAME, name = "parent", expires_at = expires_at
    )
    return raw


def _mint(client, raw, **body):
    response = client.post(
        "/api/auth/api-keys",
        headers = {"Authorization": f"Bearer {raw}"},
        json = {"name": "child", **body},
    )
    assert response.status_code == 200, response.text
    return response.json()["api_key"]["expires_at"]


def test_child_of_an_expiring_key_inherits_its_expiry(client):
    parent_expiry = (datetime.now(timezone.utc) + timedelta(hours = 2)).isoformat()
    raw = _parent(parent_expiry)
    assert datetime.fromisoformat(_mint(client, raw)) == datetime.fromisoformat(parent_expiry)
    assert datetime.fromisoformat(_mint(client, raw, expires_in_days = 30)) == datetime.fromisoformat(
        parent_expiry
    )


def test_a_shorter_requested_expiry_is_kept(client):
    raw = _parent((datetime.now(timezone.utc) + timedelta(days = 10)).isoformat())
    child = datetime.fromisoformat(_mint(client, raw, expires_in_days = 1))
    assert child < datetime.now(timezone.utc) + timedelta(days = 2)


def test_child_of_a_non_expiring_key_is_unchanged(client):
    raw = _parent(None)
    assert _mint(client, raw) is None


def test_a_parent_key_revoked_mid_request_mints_nothing(client, monkeypatch):
    raw = _parent((datetime.now(timezone.utc) + timedelta(hours = 2)).isoformat())
    # Authentication passed; the key is gone by the time the cap is read.
    monkeypatch.setattr(storage, "validate_api_key_account", lambda key, touch = True: None)
    response = client.post(
        "/api/auth/api-keys", headers = {"Authorization": f"Bearer {raw}"}, json = {"name": "child"}
    )
    assert response.status_code == 401
    assert [row["name"] for row in storage.list_api_keys(storage.DEFAULT_ADMIN_USERNAME)] == [
        "parent"
    ]


@pytest.mark.parametrize("days", [0, -5, 36501, 10**9, 10**12])
def test_out_of_range_expiry_is_rejected(client, days):
    raw = _parent(None)
    response = client.post(
        "/api/auth/api-keys",
        headers = {"Authorization": f"Bearer {raw}"},
        json = {"name": "child", "expires_in_days": days},
    )
    assert response.status_code == 422
    assert [row["name"] for row in storage.list_api_keys(storage.DEFAULT_ADMIN_USERNAME)] == [
        "parent"
    ]
