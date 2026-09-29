# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the server identity handshake (`GET /api/auth/identity`).

The endpoint lets a client confirm an endpoint is really this Unsloth install
before sending it a credential: the client sends a random nonce and checks the
returned HMAC against one computed from the install identity secret. A process
that cannot read this same-user secret cannot forge a matching proof.
"""

import base64
import hashlib
import hmac

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import storage

HOST = "127.0.0.1"  # the proof is bound to the connection address...
PORT = 8765  # ...and port


@pytest.fixture(autouse = True)
def isolated_auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_identity_secret_cache", None)
    yield


def test_identity_secret_is_persistent_and_cached():
    first = storage.get_or_create_identity_secret()
    assert isinstance(first, bytes) and len(first) == 32
    # Cached in-process.
    assert storage.get_or_create_identity_secret() == first
    # Persisted: a fresh process (cache cleared) reads the same stored value.
    storage._identity_secret_cache = None
    assert storage.get_or_create_identity_secret() == first


def test_compute_identity_proof_matches_manual_hmac():
    nonce = b"a-fixed-nonce-for-the-proof-test!"
    secret = storage.get_or_create_identity_secret()
    expected = hmac.new(
        secret, b"|".join([nonce, HOST.encode(), str(PORT).encode()]), hashlib.sha256
    ).hexdigest()
    assert storage.compute_identity_proof(nonce, HOST, PORT) == expected
    # Bound to nonce, host and port: changing any one yields a different proof.
    assert (
        storage.compute_identity_proof(b"a-different-nonce-entirely-here!!", HOST, PORT) != expected
    )
    assert storage.compute_identity_proof(nonce, "127.0.0.2", PORT) != expected
    assert storage.compute_identity_proof(nonce, HOST, PORT + 1) != expected


def test_proof_differs_when_secret_differs(tmp_path, monkeypatch):
    nonce = b"shared-nonce-across-two-installs!"
    proof_a = storage.compute_identity_proof(nonce, HOST, PORT)
    # A different install (different secret) can't reproduce the proof.
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "other_auth.db")
    monkeypatch.setattr(storage, "_identity_secret_cache", None)
    assert storage.compute_identity_proof(nonce, HOST, PORT) != proof_a


def _identity_client() -> TestClient:
    # routes.auth pulls the whole routes package (-> inference -> llama_cpp). Skip
    # if those heavy deps are missing; the proof crypto is covered above.
    try:
        from routes.auth import router
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"routes.auth not importable in this environment: {exc}")

    app = FastAPI()
    app.include_router(router, prefix = "/api/auth")
    # base_url host:port become scope["server"], which the route binds the proof to.
    return TestClient(app, base_url = f"http://{HOST}:{PORT}")


def test_identity_route_returns_matching_proof():
    client = _identity_client()
    nonce = b"route-level-nonce-for-the-server!"
    encoded = base64.urlsafe_b64encode(nonce).decode()
    response = client.get(f"/api/auth/identity?nonce={encoded}")
    assert response.status_code == 200
    assert response.json()["proof"] == storage.compute_identity_proof(nonce, HOST, PORT)


def test_identity_route_validates_nonce():
    client = _identity_client()
    # Decodes to < 16 bytes: too little entropy to be meaningful.
    short = base64.urlsafe_b64encode(b"tiny").decode()
    assert client.get(f"/api/auth/identity?nonce={short}").status_code == 400
    # Missing nonce: FastAPI request validation.
    assert client.get("/api/auth/identity").status_code == 422
