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


def test_consume_link_token_is_single_use():
    # Storage-level single-use: the conditional DELETE (no DELETE ... RETURNING,
    # which would need SQLite >= 3.35) consumes a matching row exactly once.
    from datetime import datetime, timedelta, timezone

    admin = _seed_admin()
    exp = (datetime.now(timezone.utc) + timedelta(seconds = 600)).isoformat()

    storage.save_link_token("jti-a", admin, exp)
    assert storage.consume_link_token("jti-a", admin) is True
    # A replay finds no row and must not consume again.
    assert storage.consume_link_token("jti-a", admin) is False

    # A jti bound to one user cannot be consumed under a different username.
    storage.save_link_token("jti-b", admin, exp)
    assert storage.consume_link_token("jti-b", "not-the-admin") is False
    assert storage.consume_link_token("jti-b", admin) is True


def test_save_link_token_purges_expired_rows_on_mint():
    # The frontend is not yet wired to exchange link tokens, so consume_link_token
    # (the other purge site) may never run; without a purge on mint the table
    # would grow without bound across reruns. Minting reclaims stale rows in the
    # same transaction as the insert.
    from datetime import datetime, timedelta, timezone

    admin = _seed_admin()
    past = (datetime.now(timezone.utc) - timedelta(seconds = 5)).isoformat()
    future = (datetime.now(timezone.utc) + timedelta(seconds = 600)).isoformat()

    # An already-expired row lands on disk when minted (purge runs before insert).
    storage.save_link_token("stale", admin, past)
    conn = storage.get_connection()
    try:
        assert (
            conn.execute("SELECT 1 FROM link_tokens WHERE jti = ?", ("stale",)).fetchone()
            is not None
        )
    finally:
        conn.close()

    # The next mint reclaims the now-expired row; only the live token remains.
    storage.save_link_token("fresh", admin, future)
    conn = storage.get_connection()
    try:
        rows = {r["jti"] for r in conn.execute("SELECT jti FROM link_tokens")}
    finally:
        conn.close()
    assert rows == {"fresh"}


def test_password_change_deletes_outstanding_link_tokens():
    # A password change must invalidate outstanding link tokens atomically:
    # update_password() deletes the user's link_tokens rows in the SAME
    # transaction that rotates the JWT secret, closing the race where an in-flight
    # exchange read the old derived key before the rotation.
    admin = _seed_admin()
    token = authentication.create_link_token(admin)
    jti = authentication._decode_link_payload(token.split(".", 1)[0])["jti"]

    conn = storage.get_connection()
    try:
        assert (
            conn.execute("SELECT 1 FROM link_tokens WHERE jti = ?", (jti,)).fetchone() is not None
        )
    finally:
        conn.close()

    assert storage.update_password(admin, "new-human-password-456") is not None

    conn = storage.get_connection()
    try:
        assert conn.execute("SELECT 1 FROM link_tokens WHERE jti = ?", (jti,)).fetchone() is None
    finally:
        conn.close()
    # User-facing: the exchange is rejected (defense in depth, the key also rotated).
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
    # Flip a signature BYTE and re-encode canonically, so the tampering always
    # changes the decoded bytes (flipping the last base64 character only alters
    # pad bits about 1 in 16 times, which is what made this test flaky).
    raw_sig = bytearray(authentication._b64url_decode(sig_b64))
    raw_sig[0] ^= 0x01
    tampered = f"{payload_b64}.{authentication._b64url_encode(bytes(raw_sig))}"
    assert authentication.exchange_link_token(tampered) is None
    # The jti was never consumed, so a valid replay of the untampered token still
    # works exactly once afterwards.
    assert authentication.exchange_link_token(token) == admin


def test_link_token_non_canonical_signature_is_rejected():
    # base64url is not injective under a permissive decoder: the final character of
    # a 32-byte signature carries 2 unused pad bits, so 'A', 'B', 'C' and 'D' all
    # decode to the same trailing byte (RFC 4648 s3.5 puts the zero-pad-bits MUST on
    # encoders only), and urlsafe_b64decode also accepts extra "=" and silently
    # drops non-alphabet characters. Before the canonical re-encode check, rewriting
    # a canonical trailing 'A' as 'B' still passed compare_digest and exchanged
    # successfully -- roughly 1 token in 16. Mint until a token exhibits the
    # trailing-'A' case rather than relying on chance.
    admin = _seed_admin()
    token = None
    for _ in range(500):
        candidate = authentication.create_link_token(admin)
        if candidate.split(".", 1)[1].endswith("A"):
            token = candidate
            break
    assert token is not None, "no signature ending in 'A' in 500 mints"
    payload_b64, sig_b64 = token.split(".", 1)
    variants = [
        f"{payload_b64}.{sig_b64[:-1]}B",  # same bytes, different spelling
        f"{payload_b64}.{sig_b64[:-1]}C",
        f"{payload_b64}.{sig_b64[:-1]}D",
        f"{payload_b64}.{sig_b64}=",  # explicit padding
        f"{payload_b64}.{sig_b64}==",
        f"{payload_b64}.{sig_b64[:4]}*{sig_b64[4:]}",  # non-alphabet char, silently dropped
    ]
    for variant in variants:
        assert authentication.exchange_link_token(variant) is None, variant
    # None of the rejected variants consumed the jti: the real token still works once.
    assert authentication.exchange_link_token(token) == admin
    assert authentication.exchange_link_token(token) is None


def test_link_token_non_canonical_payload_is_rejected():
    # The signature covers the payload TEXT, so a re-spelled payload cannot verify;
    # assert it explicitly so the canonical decode stays in place.
    admin = _seed_admin()
    token = authentication.create_link_token(admin)
    payload_b64, sig_b64 = token.split(".", 1)
    assert authentication.exchange_link_token(f"{payload_b64}=.{sig_b64}") is None
    assert (
        authentication.exchange_link_token(f"{payload_b64[:4]}*{payload_b64[4:]}.{sig_b64}") is None
    )
    assert authentication.exchange_link_token(token) == admin


def test_b64url_decode_canonical_round_trips_and_rejects_variants():
    # Unit: only the exact canonical spelling of the bytes decodes.
    for size in (1, 2, 3, 16, 32, 48):
        raw = secrets.token_bytes(size)
        canonical = authentication._b64url_encode(raw)
        assert authentication._b64url_decode_canonical(canonical) == raw
        assert authentication._b64url_decode_canonical(canonical + "=") is None
        assert authentication._b64url_decode_canonical(canonical + "\n") is None


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


def _deeply_nested_token(depth: int = 1000) -> str:
    nested = ("[" * depth + "]" * depth).encode("ascii")
    return f"{authentication._b64url_encode(nested)}.{authentication._b64url_encode(b'x' * 32)}"


def test_link_token_deeply_nested_payload_is_rejected():
    # An unauthenticated caller can craft a canonical payload of ~1000 nested
    # arrays that still fits under LINK_TOKEN_MAX_LENGTH. On Python 3.10/3.11 the
    # json scanner raises RecursionError (not a ValueError) at that depth, which
    # used to escape the decoder and turn the request into a 500 that never
    # reached the failure counter, so it was repeatable without throttling.
    from models.auth import LINK_TOKEN_MAX_LENGTH

    _seed_admin()
    token = _deeply_nested_token()
    assert len(token) < LINK_TOKEN_MAX_LENGTH
    assert authentication.exchange_link_token(token) is None


def test_decode_link_payload_treats_recursion_error_as_a_bad_token(monkeypatch):
    # Version-independent form of the above: whatever depth the running
    # interpreter draws the line at, a RecursionError from the parser must be a
    # rejected token, never an exception that escapes to the route.
    import json as _json

    def _boom(*_a, **_k):
        raise RecursionError("maximum recursion depth exceeded while decoding a JSON object")

    monkeypatch.setattr(_json, "loads", _boom)
    assert authentication._decode_link_payload(authentication._b64url_encode(b"{}")) is None


# ── /api/auth/link-exchange route ────────────────────────────────────


def _load_auth_route():
    route_path = _BACKEND / "routes" / "auth.py"
    spec = importlib.util.spec_from_file_location("_link_auth_route", route_path)
    auth_route = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(auth_route)
    return auth_route


def _auth_client() -> TestClient:
    app = FastAPI()
    app.include_router(_load_auth_route().router, prefix = "/api/auth")
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


def test_link_exchange_route_is_sync_so_fastapi_offloads_it():
    # Every step of the handler is blocking SQLite work (token lookup, single-use
    # consume, refresh-token insert) that can wait out the connection busy timeout
    # while another writer holds the auth DB. FastAPI runs `async def` handlers on
    # the event loop, so that wait would stall every other request; a `def` handler
    # is dispatched to the threadpool instead (same reason /identity is sync).
    import inspect
    assert not inspect.iscoroutinefunction(_load_auth_route().link_exchange)


def test_link_exchange_route_rejects_garbage():
    _seed_admin()
    client = _auth_client()
    resp = client.post("/api/auth/link-exchange", json = {"link_token": "not-a-token"})
    assert resp.status_code == 401, resp.text


def test_link_exchange_rejects_when_password_rotates_mid_issuance(monkeypatch):
    # TOCTOU: a link token consumed just BEFORE a concurrent password change commits
    # would otherwise mint a session under the freshly rotated JWT secret and survive
    # the change (cf. Keycloak CVE-2026-1035 / Omni GHSA-5x9f-6vg5-qg4m: non-atomic
    # single-use enforcement undermining rotation). The route binds issuance to the
    # secret the token validated against and rejects if it rotated, revoking the
    # tokens it just minted.
    admin = _seed_admin()
    token = authentication.create_link_token(admin)
    auth_route = _load_auth_route()
    app = FastAPI()
    app.include_router(auth_route.router, prefix = "/api/auth")
    client = TestClient(app)

    minted: list = []
    _real_create_refresh_token = auth_route.create_refresh_token

    def _rotating_create_refresh_token(*, subject, **kwargs):
        tok = _real_create_refresh_token(subject = subject, **kwargs)
        minted.append(tok)
        # Simulate the concurrent password change committing after consumption but
        # before the route's secret recheck: this rotates the stored JWT secret.
        assert storage.update_password(admin, "concurrent-new-pw-789") is not None
        return tok

    monkeypatch.setattr(auth_route, "create_refresh_token", _rotating_create_refresh_token)

    resp = client.post("/api/auth/link-exchange", json = {"link_token": token})
    # Rejected rather than issuing a session that outlives the password change.
    assert resp.status_code == 401, resp.text
    # The refresh token minted mid-race was revoked, so it cannot be redeemed.
    assert len(minted) == 1
    assert storage.consume_refresh_token(minted[0]) is None


# ── rate-limiting failed exchanges ───────────────────────────────────


def test_link_exchange_rate_limits_repeated_failures():
    # Regression (Codex 3644647557, P2): /api/auth/link-exchange is unauthenticated
    # and each attempt performs a SQLite lookup + HMAC/base64 processing. Without a
    # limiter an attacker sprays invalid tokens and pins the event loop. Apply the
    # same per-IP failure bound as /login, checked BEFORE any storage work.
    admin = _seed_admin()
    auth_route = _load_auth_route()
    app = FastAPI()
    app.include_router(auth_route.router, prefix = "/api/auth")
    client = TestClient(app)

    threshold = auth_route._LOGIN_MAX_FAILS
    for _ in range(threshold):
        resp = client.post("/api/auth/link-exchange", json = {"link_token": "not-a-token"})
        assert resp.status_code == 401, resp.text

    # Now blocked: a VALID token is rejected with 429 too, proving the limiter runs
    # BEFORE exchange_link_token_with_secret rather than after the storage work.
    valid = authentication.create_link_token(admin)
    blocked = client.post("/api/auth/link-exchange", json = {"link_token": valid})
    assert blocked.status_code == 429, blocked.text
    assert blocked.headers.get("Retry-After")
    # The valid token was NOT consumed while blocked (the gate short-circuited before
    # the exchange), so it still works once the throttle is cleared below.


def test_link_exchange_success_clears_rate_limit_bucket():
    # A successful exchange resets the IP's failure throttle, exactly as a successful
    # /login does, so a legitimate click after a few earlier failures is not blocked.
    admin = _seed_admin()
    auth_route = _load_auth_route()
    app = FastAPI()
    app.include_router(auth_route.router, prefix = "/api/auth")
    client = TestClient(app)

    threshold = auth_route._LOGIN_MAX_FAILS
    for _ in range(threshold - 1):
        resp = client.post("/api/auth/link-exchange", json = {"link_token": "not-a-token"})
        assert resp.status_code == 401, resp.text

    ok = client.post(
        "/api/auth/link-exchange",
        json = {"link_token": authentication.create_link_token(admin)},
    )
    assert ok.status_code == 200, ok.text  # clears the bucket

    # Prior failures were reset: another full batch below the threshold stays 401,
    # never 429 (without the clear, the accumulated count would trip the limit).
    for _ in range(threshold - 1):
        resp = client.post("/api/auth/link-exchange", json = {"link_token": "not-a-token"})
        assert resp.status_code == 401, resp.text


# ── oversized-token DoS hardening ────────────────────────────────────


def test_link_token_schema_caps_length():
    # /api/auth/link-exchange is unauthenticated and public; a well-formed token is
    # only a few hundred bytes, so the schema bounds it. This rejects an oversized
    # token before exchange_link_token_with_secret() scans/decodes/HMACs it.
    from pydantic import ValidationError

    from models.auth import LINK_TOKEN_MAX_LENGTH, LinkTokenRequest

    ok = authentication.create_link_token(_seed_admin())
    assert len(ok) <= LINK_TOKEN_MAX_LENGTH
    assert LinkTokenRequest(link_token = ok).link_token == ok

    with pytest.raises(ValidationError):
        LinkTokenRequest(link_token = "x" * (LINK_TOKEN_MAX_LENGTH + 1))


def test_link_exchange_route_rejects_deeply_nested_payload():
    # Route level: a nesting bomb that fits under the length cap must come back as
    # a plain 401 (and count as a failure for the limiter), not a 500. The
    # TestClient re-raises server exceptions, so a RecursionError escaping the
    # decoder fails this test on the affected interpreters.
    _seed_admin()
    client = _auth_client()
    resp = client.post("/api/auth/link-exchange", json = {"link_token": _deeply_nested_token()})
    assert resp.status_code == 401, resp.text


def test_link_exchange_route_rejects_oversized_token():
    # The route enforces the cap: an oversized token is a 422 validation error, not
    # a 401, so the exchange path never runs on attacker-sized input.
    _seed_admin()
    from models.auth import LINK_TOKEN_MAX_LENGTH

    client = _auth_client()
    resp = client.post(
        "/api/auth/link-exchange",
        json = {"link_token": "x" * (LINK_TOKEN_MAX_LENGTH + 1)},
    )
    assert resp.status_code == 422, resp.text
