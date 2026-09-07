# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/link-initial-password: a link-token session may set the FIRST password.

Same authority argument as /desktop-initial-password -- the caller proved
possession of an out-of-band secret rather than the account password -- so the
tests here are about keeping that authority narrow: only a link session, only
while must_change_password is set, and never as a way to change an existing
password.
"""

from __future__ import annotations

import importlib.util
import secrets
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from auth import authentication, hashing, storage  # noqa: E402

_SEED = "seeded-bootstrap-123"
_NEW = "a-real-password-456"


@pytest.fixture(autouse = True)
def isolated_auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    yield


def _seed_admin(*, must_change_password: bool = True) -> str:
    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = _SEED,
        jwt_secret = secrets.token_urlsafe(64),
        must_change_password = must_change_password,
    )
    return storage.DEFAULT_ADMIN_USERNAME


def _load_auth_route():
    spec = importlib.util.spec_from_file_location(
        "unsloth_test_auth_route_lip", _BACKEND / "routes" / "auth.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(_load_auth_route().router, prefix = "/api/auth")
    return TestClient(app)


def _authenticates(username: str, candidate: str) -> bool:
    record = storage.get_user_and_secret(username)
    if record is None:
        return False
    salt, pwd_hash, _jwt, _must_change = record
    return hashing.verify_password(candidate, salt, pwd_hash)


def _link_session(client: TestClient, admin: str) -> str:
    token = authentication.create_link_token(admin)
    resp = client.post("/api/auth/link-exchange", json = {"link_token": token})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # The exchange must NOT clear the flag: the seeded password is still the live
    # one until the operator picks a replacement.
    assert body["must_change_password"] is True
    return body["access_token"]


# ── the happy path ───────────────────────────────────────────────────


def test_link_session_sets_the_first_password_without_the_seed():
    admin = _seed_admin()
    client = _client()
    access = _link_session(client, admin)

    resp = client.post(
        "/api/auth/link-initial-password",
        json = {"new_password": _NEW},
        headers = {"Authorization": f"Bearer {access}"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["must_change_password"] is False

    # The chosen password is live, the seed is dead, and the flag is cleared.
    assert _authenticates(admin, _NEW)
    assert not _authenticates(admin, _SEED)
    assert storage.requires_password_change(admin) is False


def test_the_returned_session_is_no_longer_link_scoped():
    # The privilege is spent: the session handed back is an ordinary one, so it
    # cannot be replayed against this route.
    admin = _seed_admin()
    client = _client()
    access = _link_session(client, admin)
    resp = client.post(
        "/api/auth/link-initial-password",
        json = {"new_password": _NEW},
        headers = {"Authorization": f"Bearer {access}"},
    )
    assert resp.status_code == 200, resp.text
    assert authentication.is_link_access_token(resp.json()["access_token"]) is False


# ── keeping the authority narrow ─────────────────────────────────────


def test_an_ordinary_password_session_is_refused():
    admin = _seed_admin(must_change_password = False)
    client = _client()
    login = client.post(
        "/api/auth/login",
        json = {"username": admin, "password": _SEED},
    )
    assert login.status_code == 200, login.text
    resp = client.post(
        "/api/auth/link-initial-password",
        json = {"new_password": _NEW},
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"},
    )
    assert resp.status_code == 403
    assert _authenticates(admin, _SEED), "the password must not have moved"


def test_route_refuses_once_a_password_is_already_set():
    admin = _seed_admin()
    client = _client()
    access = _link_session(client, admin)
    # Someone else completes the change first.
    assert storage.update_password(admin, "chosen-elsewhere-789") is not None

    resp = client.post(
        "/api/auth/link-initial-password",
        json = {"new_password": _NEW},
        headers = {"Authorization": f"Bearer {access}"},
    )
    # 401: rotating the password rotates the JWT secret, so the link session no
    # longer verifies at all. Either way the write must not land.
    assert resp.status_code in (401, 409), resp.text
    assert _authenticates(admin, "chosen-elsewhere-789")
    assert not _authenticates(admin, _NEW)


def test_rejects_a_whitespace_password():
    admin = _seed_admin()
    client = _client()
    access = _link_session(client, admin)
    resp = client.post(
        "/api/auth/link-initial-password",
        json = {"new_password": "has a space in it"},
        headers = {"Authorization": f"Bearer {access}"},
    )
    assert resp.status_code == 400
    assert _authenticates(admin, _SEED)


# ── the two claim paths must not cross ───────────────────────────────


def test_a_link_session_is_not_a_desktop_session():
    admin = _seed_admin()
    client = _client()
    access = _link_session(client, admin)
    assert authentication.is_link_access_token(access) is True
    assert authentication.is_desktop_access_token(access) is False


def test_an_ordinary_access_token_is_not_link_scoped():
    admin = _seed_admin()
    assert (
        authentication.is_link_access_token(authentication.create_access_token(subject = admin))
        is False
    )


def test_a_link_token_is_not_a_bearer_token_and_vice_versa():
    # The link token is a two-segment HMAC blob under a domain-separated key; an
    # access token is a three-segment JWT. Neither may validate on the other path.
    admin = _seed_admin()
    link_token = authentication.create_link_token(admin)
    access = authentication.create_access_token(subject = admin)

    assert authentication.is_link_access_token(link_token) is False
    assert authentication.exchange_link_token(access) is None
