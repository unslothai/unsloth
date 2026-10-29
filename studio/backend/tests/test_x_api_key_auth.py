# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import ast
import secrets
from pathlib import Path

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from auth import storage
from auth.authentication import (
    authenticated_via_api_key,
    create_access_token,
    get_current_subject,
)
from auth.x_api_key import XApiKeyMiddleware, bearer_from_x_api_key
from utils.keyless_api_access import (
    KEYLESS_ADMISSION_STATE_KEY,
    KeylessToolPolicyMiddleware,
    _reset_scope_cache,
    set_keyless_api_access,
)


@pytest.fixture(autouse = True)
def isolated_auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    storage._reset_api_key_hash_cache()
    _reset_scope_cache()
    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = "human-password-123",
        jwt_secret = secrets.token_urlsafe(64),
        must_change_password = False,
    )
    yield
    storage._reset_api_key_hash_cache()
    _reset_scope_cache()


def _client() -> TestClient:
    app = FastAPI()
    for name, value in dict(
        bind_host = "127.0.0.1",
        secure = False,
        remote_access_is_colab = False,
        lan_access_is_colab = False,
        lan_access_secure_launch = False,
        cloudflare_url = None,
    ).items():
        setattr(app.state, name, value)

    @app.get("/v1/models")
    async def models(
        request: Request,
        subject: str = Depends(get_current_subject),
        via_api_key: bool = Depends(authenticated_via_api_key),
    ):
        state = request.scope.get("state") or {}
        return {
            "subject": subject,
            "via_api_key": via_api_key,
            "keyless": state.get(KEYLESS_ADMISSION_STATE_KEY),
        }

    app.add_middleware(KeylessToolPolicyMiddleware)
    app.add_middleware(XApiKeyMiddleware)
    return TestClient(app, base_url = "http://127.0.0.1", client = ("127.0.0.1", 50000))


def _api_key() -> str:
    key, _ = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME, name = "office", expires_at = None
    )
    return key


def test_x_api_key_authenticates_like_bearer():
    key = _api_key()
    client = _client()
    via_header = client.get("/v1/models", headers = {"x-api-key": key})
    via_bearer = client.get("/v1/models", headers = {"Authorization": f"Bearer {key}"})
    assert via_header.status_code == 200, via_header.text
    assert via_header.json() == via_bearer.json()
    assert via_header.json()["subject"] == storage.DEFAULT_ADMIN_USERNAME
    assert via_header.json()["via_api_key"] is True


def test_x_api_key_with_empty_bearer_still_authenticates():
    key = _api_key()
    response = _client().get("/v1/models", headers = {"x-api-key": key, "Authorization": "Bearer "})
    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"x-api-key": f"{storage.API_KEY_PREFIX}not-a-real-key"},
        {"x-api-key": "not-needed"},
        {"x-api-key": "sk-ant-api03-something"},
    ],
)
def test_without_a_valid_key_is_still_refused(headers):
    _api_key()
    assert _client().get("/v1/models", headers = headers).status_code == 401


def test_bearer_wins_over_x_api_key():
    key = _api_key()
    session = create_access_token(subject = storage.DEFAULT_ADMIN_USERNAME)
    response = _client().get(
        "/v1/models", headers = {"Authorization": f"Bearer {session}", "x-api-key": key}
    )
    assert response.status_code == 200, response.text
    assert response.json()["via_api_key"] is False


def test_keyless_access_unchanged_and_real_key_is_not_keyless():
    key = _api_key()
    set_keyless_api_access("full")
    client = _client()
    dummy = client.get("/v1/models", headers = {"x-api-key": "not-needed"})
    assert dummy.status_code == 200, dummy.text
    assert dummy.json()["keyless"] is True
    keyed = client.get("/v1/models", headers = {"x-api-key": key})
    assert keyed.status_code == 200, keyed.text
    assert keyed.json()["keyless"] is False
    assert keyed.json()["via_api_key"] is True


def test_rewrite_is_idempotent_and_leaves_other_requests_alone():
    key = f"{storage.API_KEY_PREFIX}abc".encode()
    rewritten = bearer_from_x_api_key([(b"x-api-key", key)])
    assert (b"authorization", b"Bearer " + key) in rewritten
    assert bearer_from_x_api_key(rewritten) is None
    assert bearer_from_x_api_key([(b"authorization", b"Bearer other")]) is None
    assert bearer_from_x_api_key([(b"x-api-key", key), (b"x-api-key", key)]) is None
    assert bearer_from_x_api_key([(b"x-api-key", b"not-needed")]) is None


def test_main_app_runs_the_rewrite_outermost():
    # The last add_middleware is the outermost layer; parsed so the check runs without main's deps.
    tree = ast.parse((Path(__file__).resolve().parents[1] / "main.py").read_text(encoding = "utf-8"))
    added = [
        node.args[0].id
        for node in sorted(ast.walk(tree), key = lambda node: getattr(node, "lineno", 0))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_middleware"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "app"
    ]
    assert added[-1] == XApiKeyMiddleware.__name__
