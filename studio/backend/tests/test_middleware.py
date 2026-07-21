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


# MaxBodyMiddleware


def _make_protected_app(
    max_bytes: int,
    main_module,
    upload_passthrough_prefixes: tuple = (),
    upload_passthrough_max_bytes_getter = None,
):
    app = FastAPI()
    app.add_middleware(
        main_module.MaxBodyMiddleware,
        max_bytes_getter = lambda: max_bytes,
        protected_prefixes = ("/v1/chat/completions", "/api/settings", "/api/train"),
        upload_passthrough_prefixes = upload_passthrough_prefixes,
        upload_passthrough_max_bytes_getter = upload_passthrough_max_bytes_getter,
    )

    @app.post("/v1/chat/completions")
    async def chat(payload: dict):
        return {"ok": True, "n": len(payload.get("text", ""))}

    @app.post("/api/other")
    async def other(payload: dict):
        return {"ok": True, "unprotected": True}

    @app.put("/api/settings/upload-limit")
    async def update_upload_limit(payload: dict):
        return {"ok": True, "limit": payload.get("max_upload_size_mb")}

    @app.post("/api/train/upload")
    async def upload(request: Request):
        total = 0
        chunks = 0
        async for chunk in request.stream():
            if chunk:
                chunks += 1
                total += len(chunk)
        return {"ok": True, "chunks": chunks, "total": total}

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

    def test_settings_put_body_over_cap_rejected(self, main_module):
        app = _make_protected_app(1024, main_module)
        c = TestClient(app)
        r = c.put(
            "/api/settings/upload-limit",
            json = {"max_upload_size_mb": 500, "padding": "x" * 5000},
        )
        assert r.status_code == 413
        assert "too large" in r.json()["detail"].lower()

    def test_chunked_upload_over_cap_rejected(self, main_module):
        # Regression: declared-Content-Length-only check could be bypassed by
        # chunked transfer-encoding.
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

    def test_upload_passthrough_uses_dedicated_declared_cap(self, main_module):
        app = _make_protected_app(
            128,
            main_module,
            upload_passthrough_prefixes = ("/api/train/upload",),
            upload_passthrough_max_bytes_getter = lambda: 1024,
        )
        c = TestClient(app)
        r = c.post(
            "/api/train/upload",
            content = b"x" * 512,
            headers = {"content-type": "application/octet-stream"},
        )
        assert r.status_code == 200
        assert r.json()["total"] == 512

    def test_upload_passthrough_rejects_declared_body_over_dedicated_cap(
        self, main_module
    ):
        app = _make_protected_app(
            128,
            main_module,
            upload_passthrough_prefixes = ("/api/train/upload",),
            upload_passthrough_max_bytes_getter = lambda: 256,
        )
        c = TestClient(app)
        r = c.post(
            "/api/train/upload",
            content = b"x" * 512,
            headers = {"content-type": "application/octet-stream"},
        )
        assert r.status_code == 413
        assert "256" in r.json()["detail"]

    def test_upload_passthrough_requires_content_length(self, main_module):
        app = _make_protected_app(
            128,
            main_module,
            upload_passthrough_prefixes = ("/api/train/upload",),
            upload_passthrough_max_bytes_getter = lambda: 1024,
        )
        c = TestClient(app)

        def gen():
            yield b"x" * 64
            yield b"y" * 64

        r = c.post(
            "/api/train/upload",
            content = gen(),
            headers = {"content-type": "application/octet-stream"},
        )
        assert r.status_code == 411
        assert "Content-Length" in r.json()["detail"]


# SecurityHeadersMiddleware / CSP


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
        permissions_policy = r.headers["permissions-policy"]
        assert "camera=()" in permissions_policy
        assert "microphone=(self)" in permissions_policy
        assert "geolocation=()" in permissions_policy
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

    def test_img_and_media_allow_https_sources(self, main_module):
        # Model-card READMEs and citation favicons pull images/media from many
        # https origins (HF LFS/XET CDNs, shields/badge hosts, GitHub-hosted
        # assets, audio/video samples). img-src/media-src allow any https source
        # so they render; this mirrors the desktop CSP in tauri.conf.json.
        csp = main_module._build_csp()
        directives = {
            chunk.strip().split()[0]: chunk.strip().split()
            for chunk in csp.split(";")
            if chunk.strip()
        }
        for name in ("img-src", "media-src"):
            assert name in directives, f"missing {name} directive"
            # Tokenise and compare with `==` so CodeQL's URL-substring rule does
            # not read directive-string `in` membership as URL sanitisation.
            assert any(src == "https:" for src in directives[name])

    def test_headers_applied_to_streaming_response(self, main_module):
        # The ASGI middleware must set headers on streaming responses too.
        from fastapi.responses import StreamingResponse

        app = FastAPI()
        app.add_middleware(main_module.SecurityHeadersMiddleware)

        @app.get("/stream")
        async def stream():
            async def gen():
                yield b"a"
                yield b"b"

            return StreamingResponse(gen(), media_type = "text/plain")

        r = TestClient(app).get("/stream")
        assert r.status_code == 200
        assert r.text == "ab"
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["server"] == "unsloth-studio"
        assert "content-security-policy" in r.headers

    def test_artifact_preview_frame_omits_x_frame_options(self, main_module):
        app = FastAPI()
        app.add_middleware(main_module.SecurityHeadersMiddleware)

        @app.get(main_module._ARTIFACT_PREVIEW_FRAME_PATH)
        async def frame():
            return Response(content = b"<html></html>", media_type = "text/html")

        r = TestClient(app).get(main_module._ARTIFACT_PREVIEW_FRAME_PATH)
        assert r.status_code == 200
        assert "x-frame-options" not in {k.lower() for k in r.headers.keys()}
        assert r.headers["referrer-policy"] == "no-referrer"

    def test_response_start_with_tuple_headers_is_hardened(self, main_module):
        # An ASGI server may emit tuple-valued raw headers; the middleware must
        # coerce to a list and still inject security headers without crashing.
        import asyncio

        async def _inner_app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": ((b"content-type", b"text/plain"),),  # tuple, not list
                }
            )
            await send({"type": "http.response.body", "body": b"ok"})

        captured = {}

        async def _send(message):
            if message["type"] == "http.response.start":
                captured["headers"] = dict(message["headers"])

        async def _receive():
            return {"type": "http.request"}

        mw = main_module.SecurityHeadersMiddleware(_inner_app)
        asyncio.run(mw({"type": "http", "path": "/plain"}, _receive, _send))

        hdrs = captured["headers"]
        assert hdrs[b"server"] == b"unsloth-studio"
        assert b"content-security-policy" in hdrs
        assert hdrs[b"x-frame-options"] == b"DENY"

    def test_is_pure_asgi_not_basehttp_middleware(self, main_module):
        # Regression: as a BaseHTTPMiddleware this wrapped the SSE stream in its
        # own anyio task group, breaking disconnect detection (GPU stuck at 100%)
        # and raising cancel scope errors. Must stay pure ASGI.
        from starlette.middleware.base import BaseHTTPMiddleware

        cls = main_module.SecurityHeadersMiddleware
        assert not issubclass(cls, BaseHTTPMiddleware)
        assert not hasattr(cls, "dispatch")

    def test_forwards_receive_channel_unchanged(self, main_module):
        # Must forward the ASGI receive channel untouched so client disconnects
        # reach the streaming handler (BaseHTTPMiddleware swapped in its own).
        seen = {}

        async def inner_app(scope, receive, send):
            seen["receive"] = receive
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send(
                {"type": "http.response.body", "body": b"ok", "more_body": False}
            )

        mw = main_module.SecurityHeadersMiddleware(inner_app)
        sentinel_receive = object()  # forwarded verbatim, never wrapped/awaited
        sent = []

        async def send(message):
            sent.append(message)

        async def run():
            await mw(
                {"type": "http", "path": "/plain", "headers": []},
                sentinel_receive,
                send,
            )

        asyncio.run(run())
        assert seen["receive"] is sentinel_receive
        start = next(m for m in sent if m["type"] == "http.response.start")
        names = {n.lower() for n, _ in start["headers"]}
        assert b"content-security-policy" in names
        assert b"server" in names

    def test_streaming_response_survives_client_disconnect(self, main_module):
        # A StreamingResponse that polls is_disconnected() (like gguf_tool_stream)
        # must unwind cleanly on client disconnect: no cancel scope error, the
        # generator's finally runs, and security headers are still applied.
        from fastapi import FastAPI, Request
        from fastapi.responses import StreamingResponse

        state = {"cleaned_up": False}
        app = FastAPI()
        app.add_middleware(main_module.SecurityHeadersMiddleware)

        @app.get("/v1/chat/completions")
        async def stream(request: Request):
            async def gen():
                try:
                    for i in range(1000):
                        if await request.is_disconnected():
                            break
                        yield f"data: {i}\n\n".encode()
                        await asyncio.sleep(0.01)
                finally:
                    state["cleaned_up"] = True

            return StreamingResponse(gen(), media_type = "text/event-stream")

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "GET",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "root_path": "",
            "scheme": "http",
            "headers": [(b"host", b"testserver")],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 80),
        }

        async def run():
            body_started = asyncio.Event()
            calls = {"n": 0}

            async def receive():
                calls["n"] += 1
                if calls["n"] == 1:
                    return {"type": "http.request", "body": b"", "more_body": False}
                await body_started.wait()  # client clicks Stop after tokens stream
                return {"type": "http.disconnect"}

            sent = []

            async def send(message):
                sent.append(message)
                if message["type"] == "http.response.body" and message.get("body"):
                    body_started.set()

            # Must return without raising the anyio cancel-scope RuntimeError.
            await asyncio.wait_for(app(scope, receive, send), timeout = 5.0)
            return sent

        sent = asyncio.run(run())
        assert state["cleaned_up"] is True
        start = next(m for m in sent if m["type"] == "http.response.start")
        names = {n.lower() for n, _ in start["headers"]}
        assert b"content-security-policy" in names
        assert b"server" in names


# /api/health auth gate


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
    # Launcher / frontend bootstrap fields are unauth so the Tauri watchdog can
    # re-adopt a sibling backend and the SPA can detect chat-only mode before
    # any token exists. Version / device_type still require a bearer.
    LAUNCHER_BITS = (
        "service",
        "studio_root_id",
        "chat_only",
        "desktop_protocol_version",
        "desktop_manageability_version",
        "supports_desktop_auth",
        "supports_desktop_backend_ownership",
        "native_path_leases_supported",
    )
    FINGERPRINT_FIELDS = ("version", "studio_version", "device_type")

    def test_no_auth_exposes_launcher_bits(self, health_app):
        c = TestClient(health_app)
        r = c.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert "timestamp" in body
        for field in self.LAUNCHER_BITS:
            assert field in body, f"missing launcher bit: {field}"
        assert body["service"] == "Unsloth UI Backend"
        for forbidden in self.FINGERPRINT_FIELDS:
            assert forbidden not in body

    def test_invalid_bearer_returns_launcher_bits_only(self, health_app):
        # Regression: calling the async dep without await let any Bearer header pass.
        c = TestClient(health_app)
        r = c.get(
            "/api/health",
            headers = {"Authorization": "Bearer not-a-real-token"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        for field in self.LAUNCHER_BITS:
            assert field in body
        for forbidden in self.FINGERPRINT_FIELDS:
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
        for field in self.LAUNCHER_BITS + self.FINGERPRINT_FIELDS:
            assert field in body, f"missing: {field}"
