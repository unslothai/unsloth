# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for MaxBodyMiddleware, SecurityHeadersMiddleware, and the /api/health auth gate."""

import asyncio
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.testclient import TestClient
from starlette.middleware.gzip import GZipMiddleware


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
    request_max_bytes_getter = None,
    upload_passthrough_prefixes: tuple = (),
    upload_passthrough_max_bytes_getter = None,
    upload_passthrough_exact_paths: tuple = (),
):
    app = FastAPI()
    app.add_middleware(
        main_module.MaxBodyMiddleware,
        max_bytes_getter = lambda: max_bytes,
        protected_prefixes = (
            "/v1/chat/completions",
            "/api/inference",
            "/api/settings",
            "/api/train",
        ),
        request_max_bytes_getter = request_max_bytes_getter,
        upload_passthrough_prefixes = upload_passthrough_prefixes,
        upload_passthrough_max_bytes_getter = upload_passthrough_max_bytes_getter,
        upload_passthrough_exact_paths = upload_passthrough_exact_paths,
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

    @app.post("/api/inference/audio/transcribe/raw")
    async def transcribe_raw(request: Request):
        return {"ok": True, "total": len(await request.body())}

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

    def test_route_specific_cap_overrides_default(self, main_module):
        app = _make_protected_app(
            4096,
            main_module,
            request_max_bytes_getter = lambda path: 128 if path.endswith("/transcribe/raw") else 4096,
        )
        c = TestClient(app)

        rejected = c.post(
            "/api/inference/audio/transcribe/raw",
            content = b"x" * 129,
        )
        accepted = c.post(
            "/api/inference/audio/transcribe/raw",
            content = b"x" * 128,
        )

        assert rejected.status_code == 413
        assert accepted.status_code == 200
        assert accepted.json()["total"] == 128

    def test_stt_routes_use_audio_specific_caps(self, main_module):
        from utils.upload_limits import (
            STT_AUDIO_JSON_MAX_BYTES,
            STT_AUDIO_RAW_MAX_BYTES,
            upload_request_limit_bytes,
        )

        assert (
            main_module._get_request_body_max_bytes("/api/inference/audio/transcribe/raw")
            == STT_AUDIO_RAW_MAX_BYTES
        )
        assert (
            main_module._get_request_body_max_bytes("/api/inference/audio/transcribe")
            == STT_AUDIO_JSON_MAX_BYTES
        )
        # The OpenAI transcriptions route is multipart, so it gets headroom over the raw cap, on both mounts.
        for path in ("/v1/audio/transcriptions", "/api/inference/audio/transcriptions"):
            assert main_module._get_request_body_max_bytes(path) == upload_request_limit_bytes(
                STT_AUDIO_RAW_MAX_BYTES
            ), path
            assert path in main_module._BODY_UPLOAD_PASSTHROUGH_EXACT_PATHS, path
            assert main_module._get_upload_passthrough_request_max_bytes(path) == (
                upload_request_limit_bytes(STT_AUDIO_RAW_MAX_BYTES)
            ), path
            assert main_module._get_upload_passthrough_request_max_bytes(path + "/") == (
                upload_request_limit_bytes(STT_AUDIO_RAW_MAX_BYTES)
            ), path

    def test_auth_routes_are_body_capped(self, main_module):
        # The public /api/auth routes (login, refresh, link-exchange) take only small
        # JSON, so they sit behind their own small cap and are in the protected
        # prefixes -- bounding the buffered body before FastAPI reads it, well below
        # the upload-sized default.
        assert "/api/auth" in main_module._BODY_PROTECTED_PREFIXES
        assert (
            main_module._get_request_body_max_bytes("/api/auth/link-exchange")
            == main_module.AUTH_REQUEST_BODY_MAX_BYTES
        )
        assert (
            main_module.AUTH_REQUEST_BODY_MAX_BYTES < main_module.default_request_body_limit_bytes()
        )

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

    def test_diffusion_dataset_upload_in_body_passthrough(self, main_module):
        # The diffusion dataset upload route lives under the protected /api/train prefix, so it must be in the REAL passthrough allowlist with the
        # DB-aware + multipart-overhead cap, else MaxBodyMiddleware 413s near-limit batches. EXACT path, so its JSON sub-routes keep the small cap.
        from utils.upload_limits import (
            default_request_body_limit_bytes,
            upload_request_limit_bytes,
        )

        path = "/api/train/diffusion/dataset"
        assert path in main_module._BODY_UPLOAD_PASSTHROUGH_EXACT_PATHS
        assert not any(path.startswith(p) for p in main_module._BODY_UPLOAD_PASSTHROUGH_PREFIXES)
        cap = main_module._get_upload_passthrough_request_max_bytes(path)
        assert cap == upload_request_limit_bytes()  # DB-aware cap + multipart overhead
        assert cap > default_request_body_limit_bytes()  # not the plain default body cap

    def test_diffusion_dataset_json_subroutes_keep_default_cap(self, main_module):
        # The exact-path passthrough must NOT sweep in the JSON sub-routes under the same prefix: a prefix match would let a large
        # caption/import body bypass the default JSON cap and be buffered up to the far larger upload limit.
        from utils.upload_limits import default_request_body_limit_bytes
        for path in (
            "/api/train/diffusion/dataset/my-set/caption/img.png",
            "/api/train/diffusion/dataset/import-example",
        ):
            assert path not in main_module._BODY_UPLOAD_PASSTHROUGH_EXACT_PATHS, path
            assert not any(
                path.startswith(p) for p in main_module._BODY_UPLOAD_PASSTHROUGH_PREFIXES
            ), path
            assert main_module._get_upload_passthrough_request_max_bytes(path) == (
                default_request_body_limit_bytes()
            ), path

    def test_diffusion_dataset_trailing_slash_gets_upload_cap(self, main_module):
        # The trailing-slash variant reaches the middleware BEFORE the router's redirect_slashes 307, so it must resolve to the
        # same passthrough + upload cap. JSON sub-routes keep extra components after normalization, so they stay capped.
        from utils.upload_limits import (
            default_request_body_limit_bytes,
            upload_request_limit_bytes,
        )

        slashed = "/api/train/diffusion/dataset/"
        assert main_module._get_upload_passthrough_request_max_bytes(slashed) == (
            upload_request_limit_bytes()
        )
        # End to end through the middleware: a body over the default cap but under the upload cap passes on both path spellings.
        app = _make_protected_app(
            128,
            main_module,
            upload_passthrough_max_bytes_getter = lambda _p: 1024,
            upload_passthrough_exact_paths = ("/api/train/diffusion/dataset",),
        )

        @app.post("/api/train/diffusion/dataset")
        async def upload(request: Request):
            body = await request.body()
            return {"total": len(body)}

        c = TestClient(app)
        for path in ("/api/train/diffusion/dataset", "/api/train/diffusion/dataset/"):
            r = c.post(
                path,
                content = b"x" * 512,
                headers = {"content-type": "application/octet-stream"},
            )
            assert r.status_code == 200, path
            assert r.json()["total"] == 512, path
        # A slashed JSON sub-route is still NOT passthrough: over-cap body is rejected.
        r = c.post(
            "/api/train/diffusion/dataset/import-example/",
            content = b"x" * 512,
            headers = {"content-type": "application/octet-stream"},
        )
        assert r.status_code == 413
        assert (
            main_module._get_upload_passthrough_request_max_bytes(
                "/api/train/diffusion/dataset/import-example/"
            )
            == default_request_body_limit_bytes()
        )

    def test_v1_surface_is_body_protected(self, main_module):
        # /images/generations is mounted at both /api/inference and /v1, and every /v1 POST must be body-capped via the blanket
        # prefix or an unbounded prompt buffers outside the Unsloth request limit. Also confirms /v1 chat/completions stays protected.
        for path in (
            "/v1/images/generations",
            "/v1/audio/generate",
            "/v1/audio/speech",
            "/v1/audio/transcriptions",
            "/v1/embeddings",
            "/v1/responses",
            "/v1/messages",
            "/v1/chat/completions",
        ):
            assert any(path.startswith(p) for p in main_module._BODY_PROTECTED_PREFIXES), path

    def test_upload_passthrough_rejects_declared_body_over_dedicated_cap(self, main_module):
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

    def test_exact_path_passthrough_does_not_cover_subroutes(self, main_module):
        # The exact-path passthrough lifts the cap for the upload path itself, but a sibling sub-path under the same prefix stays capped.
        app = FastAPI()
        app.add_middleware(
            main_module.MaxBodyMiddleware,
            max_bytes_getter = lambda: 128,
            protected_prefixes = ("/api/train",),
            upload_passthrough_exact_paths = ("/api/train/ds",),
            upload_passthrough_max_bytes_getter = lambda path: 10_000,
        )

        @app.post("/api/train/ds")
        async def _upload(request: Request):
            total = 0
            async for chunk in request.stream():
                total += len(chunk)
            return {"ok": True, "total": total}

        @app.post("/api/train/ds/import-example")
        async def _import(payload: dict):
            return {"ok": True}

        c = TestClient(app)
        # The exact upload path takes the large cap: a 512-byte body passes.
        r = c.post(
            "/api/train/ds",
            content = b"x" * 512,
            headers = {"content-type": "application/octet-stream"},
        )
        assert r.status_code == 200 and r.json()["total"] == 512
        # The sibling JSON sub-route keeps the 128-byte default cap: a large body is 413'd.
        r = c.post("/api/train/ds/import-example", json = {"text": "x" * 5000})
        assert r.status_code == 413


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
        assert main_module._CSP_SCRIPT_NONCE_HEADER not in {k.lower() for k in r.headers.keys()}

    def test_build_csp_helper_shape(self, main_module):
        plain = main_module._build_csp()
        assert "script-src 'self';" in plain
        assert "'unsafe-inline'" not in plain.split("script-src", 1)[1].split(";", 1)[0]
        nonced = main_module._build_csp("XYZ")
        assert "script-src 'self' 'nonce-XYZ';" in nonced

    def test_docs_csp_never_widens_script_src(self, main_module):
        # The docs pages run vendored bundles off this origin, so the docs branch may relax
        # style/font/worker only. A third party in script-src here would reach the tokens
        # localStorage holds for the whole origin.
        docs = main_module._build_csp(docs = True)
        directives = {
            chunk.strip().split(" ", 1)[0]: chunk.strip()
            for chunk in docs.split(";")
            if chunk.strip()
        }
        assert directives["script-src"] == "script-src 'self'"
        assert "'unsafe-inline'" not in directives["script-src"]
        assert "cdn.jsdelivr.net" not in docs
        nonced = main_module._build_csp("XYZ", docs = True)
        assert "script-src 'self' 'nonce-XYZ';" in nonced

        assert "blob:" in directives["worker-src"]
        assert main_module._DOCS_FONT_CSS in directives["style-src"]
        assert main_module._DOCS_FONT_FILES in directives["font-src"]

        plain = main_module._build_csp()
        assert main_module._DOCS_FONT_CSS not in plain
        assert main_module._DOCS_FONT_FILES not in plain
        assert "worker-src 'self';" in plain
        assert "font-src 'self' data:;" in plain

    def test_docs_paths_get_the_relaxed_csp(self, main_module):
        assert "/docs" in main_module._DOCS_PATHS
        assert "/redoc" in main_module._DOCS_PATHS
        assert "/docs/oauth2-redirect" in main_module._DOCS_PATHS

    def test_middleware_relaxes_only_the_docs_paths(self, main_module):
        # _DOCS_PATHS matches scope["path"] exactly, so the trailing-slash twin stays strict.
        app = _make_csp_app(main_module)

        @app.get("/docs")
        async def docs():
            return {"ok": True}

        @app.get("/docs/")
        async def docs_slash():
            return {"ok": True}

        c = TestClient(app)
        relaxed = c.get("/docs").headers["content-security-policy"]
        assert main_module._DOCS_FONT_CSS in relaxed

        for path in ("/docs/", "/plain"):
            strict = c.get(path).headers["content-security-policy"]
            assert main_module._DOCS_FONT_CSS not in strict, path

    def test_docs_pages_load_no_third_party_script(self, main_module):
        # FastAPI's built-in docs pages point at cdn.jsdelivr.net. They are re-registered on
        # the same paths against assets/docs_ui so nothing off-origin executes where the
        # tokens live, and the built-ins must stay off or they would win the path.
        assert main_module.app.docs_url is None
        assert main_module.app.redoc_url is None
        assert main_module.app.swagger_ui_oauth2_redirect_url is None

        paths = {getattr(route, "path", None) for route in main_module.app.routes}
        assert {"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"} <= paths

        c = TestClient(main_module.app)
        for path in ("/docs", "/redoc", "/docs/oauth2-redirect"):
            body = c.get(path).text
            assert "cdn.jsdelivr.net" not in body, path
            assert "fastapi.tiangolo.com" not in body, path

    def test_docs_inline_script_runs_off_the_response_nonce(self, main_module):
        # Swagger's init is inline, so a strict script-src needs the nonce spliced into the
        # header to match the tag. A mismatch renders blank, which is what CDN-era /docs did.
        c = TestClient(main_module.app)
        for path in ("/docs", "/docs/oauth2-redirect"):
            r = c.get(path)
            csp = r.headers["content-security-policy"]
            nonce = re.search(r"'nonce-([^']+)'", csp)
            assert nonce, f"{path} served no nonce"
            assert f'<script nonce="{nonce.group(1)}">' in r.text, path
            # The hand-off header is internal and must not reach the client.
            assert main_module._CSP_SCRIPT_NONCE_HEADER not in {k.lower() for k in r.headers}

        # ReDoc has no inline script, so it gets no nonce to leak.
        assert (
            "nonce-"
            not in TestClient(main_module.app).get("/redoc").headers["content-security-policy"]
        )

    def test_docs_urls_follow_the_root_path(self, main_module):
        # Behind a path-stripping proxy the browser sees a prefix the server never does, so
        # every URL the pages emit has to carry it, as FastAPI's own docs routes do.
        c = TestClient(main_module.app, root_path = "/studio")
        docs = c.get("/docs").text
        assert "'/studio/openapi.json'" in docs
        assert "'/studio/docs/oauth2-redirect'" in docs
        for name in ("swagger-ui-bundle.js", "swagger-ui.css", "favicon-32x32.png"):
            assert f"/studio/docs-assets/{name}" in docs, name

        redoc = c.get("/redoc").text
        assert 'spec-url="/studio/openapi.json"' in redoc
        assert "/studio/docs-assets/redoc.standalone.js" in redoc

        # Unprefixed deployments, which is every default Unsloth, stay unprefixed.
        plain = TestClient(main_module.app).get("/docs").text
        assert "/studio/" not in plain
        assert "'/openapi.json'" in plain

    def test_swagger_nonce_survives_a_reflowed_upstream_template(self, main_module):
        # fastapi is unpinned, so the tag is matched by what follows it. A version that
        # reflows the page or drops the comment must still get the nonce, not a 500.
        reflowed = (
            "<html><body><script src='/docs-assets/swagger-ui-bundle.js'></script>\n"
            "<script>\n  const ui = SwaggerUIBundle({url: '/openapi.json'})\n</script>\n"
            "</body></html>"
        )
        r = main_module._nonced_docs_response(reflowed, tag = main_module._SWAGGER_INIT_TAG)
        nonce = r.headers[main_module._CSP_SCRIPT_NONCE_HEADER]
        body = r.body.decode()
        assert f'<script nonce="{nonce}">' in body
        # The bundle's own tag keeps its src and gains nothing.
        assert "<script src='/docs-assets/swagger-ui-bundle.js'></script>" in body

        with pytest.raises(RuntimeError):
            main_module._nonced_docs_response(
                "<html><body>no inline script</body></html>",
                tag = main_module._SWAGGER_INIT_TAG,
            )

    def test_docs_assets_are_served_from_this_origin(self, main_module):
        c = TestClient(main_module.app)
        for name in ("swagger-ui-bundle.js", "swagger-ui.css", "redoc.standalone.js"):
            r = c.get(f"{main_module._DOCS_ASSETS_URL}/{name}")
            assert r.status_code == 200, name
            assert len(r.content) > 10_000, name

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
            await send({"type": "http.response.body", "body": b"ok", "more_body": False})

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


class TestResearchPortMiddleware:
    def test_is_pure_asgi_and_forwards_receive_unchanged(self, main_module):
        from starlette.middleware.base import BaseHTTPMiddleware

        cls = main_module.ResearchPortMiddleware
        assert not issubclass(cls, BaseHTTPMiddleware)
        assert not hasattr(cls, "dispatch")

        seen = {}

        class Supervisor:
            def note_server_port(self, server):
                seen["server"] = server

        async def inner_app(scope, receive, send):
            seen["receive"] = receive
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok", "more_body": False})

        request_app = type("App", (), {})()
        request_app.state = type("State", (), {"research_supervisor": Supervisor()})()
        sentinel_receive = object()

        async def send(_message):
            return None

        asyncio.run(
            cls(inner_app)(
                {
                    "type": "http",
                    "path": "/api/research/runs/run-1/events",
                    "app": request_app,
                    "server": ("127.0.0.1", 4321),
                },
                sentinel_receive,
                send,
            )
        )

        assert seen["receive"] is sentinel_receive
        assert seen["server"] == ("127.0.0.1", 4321)


class TestFrontendAssets:
    def test_desktop_frontend_is_available_only_through_live_tunnel(self, tmp_path, main_module):
        (tmp_path / "index.html").write_text("<!doctype html><title>remote</title>")
        assets = tmp_path / "assets"
        assets.mkdir()
        (assets / "app.js").write_text("export {};", encoding = "utf-8")
        app = FastAPI()
        app.state.cloudflare_url = None
        assert main_module.setup_frontend(app, tmp_path, tunnel_only = True)
        client = TestClient(app)
        remote_client = TestClient(app, base_url = "https://remote.trycloudflare.com")
        headers = {"CF-Connecting-IP": "198.51.100.7"}
        assert client.get("/").status_code == 404
        assert client.get("/assets/app.js").status_code == 404
        assert remote_client.get("/", headers = headers).status_code == 404

        app.state.cloudflare_url = "https://remote.trycloudflare.com"
        assert remote_client.get("/", headers = headers).status_code == 200
        assert remote_client.get("/settings/api", headers = headers).status_code == 200
        assert remote_client.get("/assets/app.js", headers = headers).status_code == 200
        assert client.get("/", headers = headers).status_code == 404
        assert client.get("/settings/api", headers = headers).status_code == 404
        assert client.get("/").status_code == 404

    def test_hashed_assets_are_compressed_and_cached(self, tmp_path, main_module):
        content = b"export const value = 'responsive';\n" * 200
        (tmp_path / "page-abc123.js").write_bytes(content)
        app = FastAPI()
        assets_app = GZipMiddleware(
            main_module.ImmutableStaticFiles(directory = tmp_path),
            minimum_size = 1024,
            compresslevel = 6,
        )
        app.mount("/assets", assets_app, name = "assets")

        response = TestClient(app).get(
            "/assets/page-abc123.js",
            headers = {"Accept-Encoding": "gzip"},
        )

        assert response.status_code == 200
        assert response.content == content
        assert response.headers["content-encoding"] == "gzip"
        assert response.headers["cache-control"] == (main_module._IMMUTABLE_ASSET_CACHE_CONTROL)
        assert "accept-encoding" in response.headers["vary"].lower()

    def test_asset_revalidation_keeps_immutable_cache_header(self, tmp_path, main_module):
        (tmp_path / "page-abc123.js").write_text("export {};", encoding = "utf-8")
        app = FastAPI()
        app.mount(
            "/assets",
            main_module.ImmutableStaticFiles(directory = tmp_path),
            name = "assets",
        )
        client = TestClient(app)
        first = client.get("/assets/page-abc123.js")

        response = client.get(
            "/assets/page-abc123.js",
            headers = {"If-None-Match": first.headers["etag"]},
        )

        assert response.status_code == 304
        assert response.headers["cache-control"] == (main_module._IMMUTABLE_ASSET_CACHE_CONTROL)

    def test_range_request_is_not_compressed(self, tmp_path, main_module):
        content = b"export const value = 'responsive';\n" * 200
        (tmp_path / "page-abc123.js").write_bytes(content)
        app = FastAPI()
        assets_app = main_module._AssetGZipMiddleware(
            main_module.ImmutableStaticFiles(directory = tmp_path),
            minimum_size = 1024,
            compresslevel = 6,
        )
        app.mount("/assets", assets_app, name = "assets")

        response = TestClient(app).get(
            "/assets/page-abc123.js",
            headers = {"Accept-Encoding": "gzip", "Range": "bytes=0-99"},
        )

        assert response.status_code == 206
        assert response.headers.get("content-encoding") != "gzip"
        assert response.headers["content-range"] == f"bytes 0-99/{len(content)}"
        assert response.content == content[:100]
        assert response.headers["cache-control"] == (main_module._IMMUTABLE_ASSET_CACHE_CONTROL)


# /api/health auth gate


@pytest.fixture
def health_app(tmp_path, monkeypatch):
    """Mount /api/health on a fresh app against an isolated auth db."""
    from auth import storage

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)

    import main as _main

    # This fixture exercises bearer redaction, not hardware startup. Keep the
    # payload settled even on macOS while MLX self-repair holds the live verdict.
    # (chat_only, chat_only_reason, chat_only_detail): health_check reads all three, so a
    # two-tuple here raised IndexError once main added the detail field.
    monkeypatch.setattr(_main, "_hardware_snapshot", lambda: (False, None, None))
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
