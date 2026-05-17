# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for MaxBodyMiddleware, SecurityHeadersMiddleware, and the /api/health auth gate."""

import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.testclient import TestClient


_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


@pytest.fixture(scope = "module")
def main_module():
    import main as _main  # noqa: F401

    return _main


# =====================================================================
# MaxBodyMiddleware
# =====================================================================


def _make_protected_app(max_bytes: int, main_module):
    app = FastAPI()
    app.add_middleware(
        main_module.MaxBodyMiddleware,
        max_bytes = max_bytes,
        protected_prefixes = ("/v1/chat/completions", "/api/train"),
    )

    @app.post("/v1/chat/completions")
    async def chat(payload: dict):
        return {"ok": True, "n": len(payload.get("text", ""))}

    @app.post("/api/other")
    async def other(payload: dict):
        return {"ok": True, "unprotected": True}

    @app.get("/api/train/status")
    async def status_get():
        return {"ok": True, "get": True}

    return app


class TestMaxBodyMiddleware:
    def test_small_protected_body_passes(self, main_module):
        app = _make_protected_app(1024, main_module)
        c = TestClient(app)
        r = c.post("/v1/chat/completions", json = {"text": "x" * 100})
        assert r.status_code == 200
        assert r.json()["n"] == 100

    def test_large_declared_content_length_rejected(self, main_module):
        app = _make_protected_app(1024, main_module)
        c = TestClient(app)
        r = c.post("/v1/chat/completions", json = {"text": "x" * 5000})
        assert r.status_code == 413
        assert "too large" in r.json()["detail"].lower()

    def test_unprotected_prefix_passes_large_body(self, main_module):
        app = _make_protected_app(1024, main_module)
        c = TestClient(app)
        r = c.post("/api/other", json = {"text": "x" * 5000})
        assert r.status_code == 200
        assert r.json()["unprotected"] is True

    def test_chunked_upload_over_cap_rejected(self, main_module):
        # Regression: declared-Content-Length-only check could be bypassed
        # by chunked transfer-encoding.
        app = _make_protected_app(1024, main_module)
        c = TestClient(app)

        def gen():
            yield b'{"text":"'
            yield b"x" * 800
            yield b'"}'
            yield b"\n" + b"y" * 500

        r = c.post(
            "/v1/chat/completions",
            content = gen(),
            headers = {"content-type": "application/json"},
        )
        assert r.status_code == 413
        assert "too large" in r.json()["detail"].lower()

    def test_chunked_upload_under_cap_passes(self, main_module):
        app = _make_protected_app(1024, main_module)
        c = TestClient(app)

        def gen():
            yield b'{"text":"'
            yield b"x" * 50
            yield b'"}'

        r = c.post(
            "/v1/chat/completions",
            content = gen(),
            headers = {"content-type": "application/json"},
        )
        assert r.status_code == 200
        assert r.json()["n"] == 50

    def test_get_not_subject_to_cap(self, main_module):
        app = _make_protected_app(1024, main_module)
        c = TestClient(app)
        r = c.get("/api/train/status")
        assert r.status_code == 200


# =====================================================================
# SecurityHeadersMiddleware / CSP
# =====================================================================


def _make_csp_app(main_module, attach_nonce: str | None = None):
    app = FastAPI()
    app.add_middleware(main_module.SecurityHeadersMiddleware)

    @app.get("/plain")
    async def plain():
        return {"ok": True}

    @app.get("/with-nonce")
    async def with_nonce():
        headers = {}
        if attach_nonce:
            headers[main_module._CSP_SCRIPT_NONCE_HEADER] = attach_nonce
        return Response(
            content = b"<html></html>",
            media_type = "text/html",
            headers = headers,
        )

    return app


class TestSecurityHeadersMiddleware:
    def test_csp_has_no_unsafe_inline_for_script_src(self, main_module):
        app = _make_csp_app(main_module)
        c = TestClient(app)
        r = c.get("/plain")
        assert r.status_code == 200
        csp = r.headers["content-security-policy"]
        # Parse per-directive so style-src unsafe-inline does not false-match.
        directives = {
            chunk.strip().split(" ", 1)[0]: chunk.strip()
            for chunk in csp.split(";")
            if chunk.strip()
        }
        assert "script-src" in directives
        assert "'unsafe-inline'" not in directives["script-src"]
        # style-src keeps unsafe-inline for Vite-injected styles.
        assert "'unsafe-inline'" in directives["style-src"]

    def test_default_security_headers_present(self, main_module):
        app = _make_csp_app(main_module)
        c = TestClient(app)
        r = c.get("/plain")
        assert r.headers["x-frame-options"] == "DENY"
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["referrer-policy"] == "no-referrer"
        assert "camera=()" in r.headers["permissions-policy"]
        assert r.headers["server"] == "unsloth-studio"

    def test_internal_nonce_header_is_spliced_into_csp_and_stripped(self, main_module):
        nonce = "test-nonce-abc"
        app = _make_csp_app(main_module, attach_nonce = nonce)
        c = TestClient(app)
        r = c.get("/with-nonce")
        csp = r.headers["content-security-policy"]
        assert f"'nonce-{nonce}'" in csp
        # Internal handoff header must not leak to clients.
        assert main_module._CSP_SCRIPT_NONCE_HEADER not in {
            k.lower() for k in r.headers.keys()
        }

    def test_build_csp_helper_shape(self, main_module):
        plain = main_module._build_csp()
        assert "script-src 'self';" in plain
        assert "'unsafe-inline'" not in plain.split("script-src", 1)[1].split(";", 1)[0]
        nonced = main_module._build_csp("XYZ")
        assert "script-src 'self' 'nonce-XYZ';" in nonced


# =====================================================================
# /api/health auth gate
# =====================================================================


@pytest.fixture
def health_app(tmp_path, monkeypatch):
    """Mount /api/health on a fresh app against an isolated auth db."""
    from auth import storage

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)

    import main as _main

    app = FastAPI()
    app.add_api_route("/api/health", _main.health_check, methods = ["GET"])

    import secrets as _secrets

    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = "human-password-123",
        jwt_secret = _secrets.token_urlsafe(64),
        must_change_password = False,
    )
    return app


class TestHealthAuthGate:
    def test_no_auth_returns_minimal_payload(self, health_app):
        c = TestClient(health_app)
        r = c.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert "timestamp" in body
        for forbidden in ("version", "device_type", "studio_root_id"):
            assert forbidden not in body

    def test_invalid_bearer_returns_minimal_payload(self, health_app):
        # Regression: calling the async dep without await made any Bearer header pass.
        c = TestClient(health_app)
        r = c.get(
            "/api/health",
            headers = {"Authorization": "Bearer not-a-real-token"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        for forbidden in ("version", "device_type", "studio_root_id"):
            assert forbidden not in body

    def test_valid_bearer_returns_full_payload(self, health_app):
        from auth import storage
        from auth.authentication import create_access_token

        token = create_access_token(storage.DEFAULT_ADMIN_USERNAME)
        c = TestClient(health_app)
        r = c.get(
            "/api/health",
            headers = {"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert "version" in body
        assert "device_type" in body
        assert "studio_root_id" in body
