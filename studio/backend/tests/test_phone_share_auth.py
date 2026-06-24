# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import secrets
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from auth import storage


@pytest.fixture(autouse = True)
def isolated_auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    yield


def seed_user():
    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = "human-password-123",
        jwt_secret = secrets.token_urlsafe(64),
        must_change_password = False,
    )


def _creds(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme = "Bearer", credentials = token)


def test_phone_token_carries_readonly_scope_and_run_id():
    seed_user()
    from auth.authentication import create_phone_token

    token, expires = create_phone_token(storage.DEFAULT_ADMIN_USERNAME, "run-123")
    payload = jwt.decode(
        token,
        storage.get_jwt_secret(storage.DEFAULT_ADMIN_USERNAME),
        algorithms = ["HS256"],
    )
    assert payload["sub"] == storage.DEFAULT_ADMIN_USERNAME
    assert payload["scope"] == "phone_view"
    assert payload["run_id"] == "run-123"
    assert expires > datetime.now(timezone.utc)


def test_phone_viewer_accepts_phone_token():
    seed_user()
    from auth.authentication import create_phone_token, get_phone_viewer

    token, _ = create_phone_token(storage.DEFAULT_ADMIN_USERNAME, "run-123")
    subject, run_id = asyncio.run(get_phone_viewer(_creds(token)))
    assert subject == storage.DEFAULT_ADMIN_USERNAME
    assert run_id == "run-123"


def test_phone_token_rejected_on_full_access_routes():
    seed_user()
    from auth.authentication import create_phone_token, get_current_subject

    token, _ = create_phone_token(storage.DEFAULT_ADMIN_USERNAME, "run-123")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(get_current_subject(_creds(token)))
    assert exc.value.status_code == 401


def test_phone_viewer_rejects_normal_access_token():
    seed_user()
    from auth.authentication import create_access_token, get_phone_viewer

    token = create_access_token(subject = storage.DEFAULT_ADMIN_USERNAME)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(get_phone_viewer(_creds(token)))
    assert exc.value.status_code == 401


def test_expired_phone_token_rejected():
    seed_user()
    from auth.authentication import get_phone_viewer

    secret = storage.get_jwt_secret(storage.DEFAULT_ADMIN_USERNAME)
    expired = jwt.encode(
        {
            "sub": storage.DEFAULT_ADMIN_USERNAME,
            "scope": "phone_view",
            "run_id": "r",
            "exp": datetime.now(timezone.utc) - timedelta(minutes = 1),
        },
        secret,
        algorithm = "HS256",
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(get_phone_viewer(_creds(expired)))
    assert exc.value.status_code == 401
