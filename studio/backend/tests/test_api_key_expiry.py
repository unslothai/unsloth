# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Expiry enforcement for API keys (tz-aware ``expires_at``) and JWT access
tokens (``exp`` claim). Both must surface as 401 on protected routes."""

from __future__ import annotations

import asyncio
import secrets
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from auth import storage
from auth.authentication import create_access_token, get_current_subject


@pytest.fixture(autouse = True)
def isolated_auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    storage._reset_api_key_hash_cache()
    yield
    storage._reset_api_key_hash_cache()


def seed_user():
    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = "human-password-123",
        jwt_secret = secrets.token_urlsafe(64),
    )


def iso_from_now(**delta):
    return (datetime.now(timezone.utc) + timedelta(**delta)).isoformat()


def make_key(expires_at):
    raw, _row = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME,
        name = "test",
        expires_at = expires_at,
    )
    return raw


def subject_of(token):
    """Run the real FastAPI auth dependency against a bearer token."""
    credentials = HTTPAuthorizationCredentials(scheme = "Bearer", credentials = token)
    return asyncio.run(get_current_subject(credentials))


# --- validate_api_key (storage layer) ---------------------------------------


def test_unexpired_key_validates():
    seed_user()
    assert (
        storage.validate_api_key(make_key(iso_from_now(days = 1)))
        == storage.DEFAULT_ADMIN_USERNAME
    )


def test_never_expiring_key_validates():
    seed_user()
    assert storage.validate_api_key(make_key(None)) == storage.DEFAULT_ADMIN_USERNAME


def test_expired_key_rejected():
    seed_user()
    assert storage.validate_api_key(make_key(iso_from_now(seconds = -1))) is None


def test_key_expiring_far_in_past_rejected():
    seed_user()
    assert storage.validate_api_key(make_key(iso_from_now(days = -30))) is None


def test_revoked_key_rejected():
    seed_user()
    raw, row = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME,
        name = "doomed",
        expires_at = iso_from_now(days = 1),
    )
    storage.revoke_api_key(storage.DEFAULT_ADMIN_USERNAME, int(row["id"]))
    assert storage.validate_api_key(raw) is None


def test_unknown_key_rejected():
    seed_user()
    assert (
        storage.validate_api_key(storage.API_KEY_PREFIX + secrets.token_hex(16)) is None
    )


# --- get_current_subject (route dependency) ---------------------------------


def test_dependency_accepts_unexpired_key():
    seed_user()
    assert subject_of(make_key(iso_from_now(days = 1))) == storage.DEFAULT_ADMIN_USERNAME


def test_dependency_rejects_expired_key_as_401():
    seed_user()
    with pytest.raises(HTTPException) as exc:
        subject_of(make_key(iso_from_now(seconds = -1)))
    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid or expired API key"


# --- JWT access-token expiry ------------------------------------------------


def test_dependency_accepts_unexpired_jwt():
    seed_user()
    token = create_access_token(storage.DEFAULT_ADMIN_USERNAME, timedelta(minutes = 5))
    assert subject_of(token) == storage.DEFAULT_ADMIN_USERNAME


def test_dependency_rejects_expired_jwt_as_401():
    seed_user()
    token = create_access_token(storage.DEFAULT_ADMIN_USERNAME, timedelta(seconds = -1))
    with pytest.raises(HTTPException) as exc:
        subject_of(token)
    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid or expired token"


# --- derivation cache: speeds repeats without bypassing checks --------------


def test_cache_skips_pbkdf2_on_repeat(monkeypatch):
    seed_user()
    raw = make_key(iso_from_now(days = 1))
    assert (
        storage.validate_api_key(raw) == storage.DEFAULT_ADMIN_USERNAME
    )  # warms cache

    calls = {"n": 0}
    real = storage._pbkdf2_api_key

    def counting(key):
        calls["n"] += 1
        return real(key)

    monkeypatch.setattr(storage, "_pbkdf2_api_key", counting)
    assert storage.validate_api_key(raw) == storage.DEFAULT_ADMIN_USERNAME
    assert storage.validate_api_key(raw) == storage.DEFAULT_ADMIN_USERNAME
    assert calls["n"] == 0  # served from cache, KDF not re-run


def test_cache_does_not_bypass_revocation():
    seed_user()
    raw, row = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME,
        name = "revoke-after-cache",
        expires_at = iso_from_now(days = 1),
    )
    assert storage.validate_api_key(raw) == storage.DEFAULT_ADMIN_USERNAME  # cached
    storage.revoke_api_key(storage.DEFAULT_ADMIN_USERNAME, int(row["id"]))
    assert storage.validate_api_key(raw) is None  # cache hit still re-checks is_active


def test_cache_does_not_bypass_expiry():
    seed_user()
    # Expires between the two calls: the first warms the cache, the second is still rejected.
    near = (datetime.now(timezone.utc) + timedelta(milliseconds = 600)).isoformat()
    raw = make_key(near)
    assert storage.validate_api_key(raw) == storage.DEFAULT_ADMIN_USERNAME
    import time

    time.sleep(0.8)
    assert storage.validate_api_key(raw) is None


def test_unknown_key_not_cached():
    seed_user()
    bogus = storage.API_KEY_PREFIX + secrets.token_hex(16)
    assert storage.validate_api_key(bogus) is None
    cache_id = storage._api_key_cache_id(bogus)
    assert cache_id not in storage._api_key_hash_cache  # spam can't grow the cache


def test_create_api_key_route_stores_tz_aware_expiry():
    from datetime import datetime as _dt

    seed_user()
    raw, row = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME,
        name = "route",
        expires_at = iso_from_now(days = 30),
    )
    parsed = _dt.fromisoformat(row["expires_at"])
    assert (
        parsed.tzinfo is not None
    )  # tz-aware: comparison in validate_api_key won't raise
    assert storage.validate_api_key(raw) == storage.DEFAULT_ADMIN_USERNAME
