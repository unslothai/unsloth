# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One-time, short-TTL link tokens (opt-in Colab same-tab handoff).

A link token is minted for the admin, signed with a key derived from the user's
JWT secret, single-use, and exchangeable for the normal session JWT exactly once.
Imports the backend auth modules directly, so run under the Unsloth venv."""

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

from auth import authentication, storage  # noqa: E402


@pytest.fixture(autouse = True)
def isolated_auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    yield


def _seed_admin() -> str:
    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = "human-password-123",
        jwt_secret = secrets.token_urlsafe(64),
        must_change_password = False,
    )
    return storage.DEFAULT_ADMIN_USERNAME


# ── mint / exchange primitives ───────────────────────────────────────


def test_link_token_round_trips_once():
    admin = _seed_admin()
    token = authentication.create_link_token(admin)
    # First exchange succeeds and returns the bound subject.
    assert authentication.exchange_link_token(token) == admin
    # Second exchange is rejected: single-use.
    assert authentication.exchange_link_token(token) is None


def test_link_token_is_not_a_valid_access_bearer_token():
    # Domain separation: a link token is signed with a derived key, so it must NOT
    # validate as a normal bearer JWT (which would sidestep single-use).
    import jwt as _jwt

    admin = _seed_admin()
    token = authentication.create_link_token(admin)
    jwt_secret = storage.get_jwt_secret(admin)
    # The compact link token is not a JWT (two segments, derived signing key), so
    # the access-token path cannot accept it as a bearer credential.
    with pytest.raises(_jwt.InvalidTokenError):
        _jwt.decode(token, jwt_secret, algorithms = ["HS256"])


def test_link_token_expired_is_rejected(monkeypatch):
    admin = _seed_admin()
    # Negative TTL mints a token whose exp is already in the past.
    monkeypatch.setattr(authentication, "LINK_TOKEN_EXPIRE_SECONDS", -1)
    token = authentication.create_link_token(admin)
    monkeypatch.setattr(authentication, "LINK_TOKEN_EXPIRE_SECONDS", 600)
    assert authentication.exchange_link_token(token) is None


def test_link_token_tampered_signature_is_rejected():
    admin = _seed_admin()
    token = authentication.create_link_token(admin)
    payload_b64, sig_b64 = token.split(".", 1)
    # Flip the last signature character to a different value.
    flipped = "A" if sig_b64[-1] != "A" else "B"
    tampered = f"{payload_b64}.{sig_b64[:-1]}{flipped}"
    assert authentication.exchange_link_token(tampered) is None
    # The jti was never consumed, so a valid replay of the untampered token still
    # works exactly once afterwards.
    assert authentication.exchange_link_token(token) == admin


def test_link_token_tampered_payload_is_rejected():
    admin = _seed_admin()
    token = authentication.create_link_token(admin)
    _payload_b64, sig_b64 = token.split(".", 1)
    # Re-sign a different subject claim is impossible without the secret; a swapped
    # payload no longer matches the signature.
    forged_payload = authentication._b64url_encode(b'{"sub":"unsloth","jti":"x","exp":"z"}')
    forged = f"{forged_payload}.{sig_b64}"
    assert authentication.exchange_link_token(forged) is None


def test_link_token_unknown_subject_is_rejected():
    _seed_admin()
    # A token naming a user that does not exist has no derivable key -> rejected.
    forged_payload = authentication._b64url_encode(b'{"sub":"ghost","jti":"x","exp":"z"}')
    forged = f"{forged_payload}.{authentication._b64url_encode(b'deadbeef')}"
    assert authentication.exchange_link_token(forged) is None


def test_link_token_malformed_is_rejected():
    _seed_admin()
    for bad in ["", "no-dot", "a.b.c", ".", "x.", ".y"]:
        assert authentication.exchange_link_token(bad) is None


# ── /api/auth/link-exchange route ────────────────────────────────────


def _auth_client() -> TestClient:
    route_path = _BACKEND / "routes" / "auth.py"
    spec = importlib.util.spec_from_file_location("_link_auth_route", route_path)
    auth_route = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(auth_route)
    app = FastAPI()
    app.include_router(auth_route.router, prefix = "/api/auth")
    return TestClient(app)


def test_link_exchange_route_issues_jwt_once():
    admin = _seed_admin()
    token = authentication.create_link_token(admin)
    client = _auth_client()

    resp = client.post("/api/auth/link-exchange", json = {"link_token": token})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["must_change_password"] is False

    # The issued access token authenticates as the admin on the normal JWT path.
    subject = authentication._decode_subject_without_verification(body["access_token"])
    assert subject == admin

    # Replay is rejected (single-use consumed above).
    replay = client.post("/api/auth/link-exchange", json = {"link_token": token})
    assert replay.status_code == 401, replay.text


def test_link_exchange_route_rejects_garbage():
    _seed_admin()
    client = _auth_client()
    resp = client.post("/api/auth/link-exchange", json = {"link_token": "not-a-token"})
    assert resp.status_code == 401, resp.text
