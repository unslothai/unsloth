# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import importlib.util
import secrets
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import storage
from auth.authentication import create_access_token

START = storage.UNSLOTH_START_API_KEY_NAME


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
    spec = importlib.util.spec_from_file_location("_unsloth_start_key_auth_route", route_path)
    auth_route = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auth_route)
    app = FastAPI()
    app.include_router(auth_route.router, prefix = "/api/auth")
    return TestClient(app)


def _mint(client, bearer, name):
    return client.post(
        "/api/auth/api-keys", headers = {"Authorization": f"Bearer {bearer}"}, json = {"name": name}
    )


def test_only_a_desktop_session_can_mint_the_unsloth_start_key(client):
    owner = storage.DEFAULT_ADMIN_USERNAME
    user_key, _row = storage.create_api_key(owner, name = "my script")

    assert _mint(client, user_key, START).status_code == 403
    assert _mint(client, create_access_token(owner), START).status_code == 403
    assert _mint(client, user_key, "another script").status_code == 200
    minted = _mint(client, create_access_token(owner, desktop = True), START)
    assert minted.status_code == 200, minted.text
    assert minted.json()["api_key"]["name"] == START


def test_only_the_unsloth_start_key_may_use_a_saved_connection():
    from routes import inference as inf

    owner = storage.DEFAULT_ADMIN_USERNAME
    start_key, _ = storage.create_api_key(owner, name = START)
    user_key, _ = storage.create_api_key(owner, name = "my script")
    internal_key, _ = storage.create_api_key(owner, name = START, internal = True)

    def allowed(raw):
        return inf._request_may_use_saved_provider_key(
            SimpleNamespace(headers = {"authorization": f"Bearer {raw}"})
        )

    assert allowed(start_key) is True
    assert allowed(user_key) is False
    assert allowed(internal_key) is False
    assert allowed("sk-unsloth-0000000000000000") is False


def test_the_cli_mints_the_reserved_name():
    start_py = Path(__file__).resolve().parents[3] / "unsloth_cli" / "commands" / "start.py"
    assert f'{{"name": "{START}"}}' in start_py.read_text(encoding = "utf-8")
