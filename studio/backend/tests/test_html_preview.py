# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for the /api/preview/html route (interactive HTML preview)."""

import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


@pytest.fixture
def preview_app(tmp_path, monkeypatch):
    """Standalone app mounting only the html-preview router on a clean store."""
    from auth import storage
    from auth.authentication import create_access_token

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)

    import secrets as _secrets
    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = "human-password-123",
        jwt_secret = _secrets.token_urlsafe(64),
        must_change_password = False,
    )

    from routes.html_preview import router as html_preview_router
    from routes import html_preview as html_preview_module

    # Each test starts with an empty in-memory store.
    html_preview_module._PREVIEWS.clear()

    app = FastAPI()
    app.include_router(
        html_preview_router, prefix = "/api/preview/html", tags = ["html-preview"]
    )
    token = create_access_token(storage.DEFAULT_ADMIN_USERNAME)
    return app, token, html_preview_module


class TestPostHtmlPreview:
    def test_post_requires_auth(self, preview_app):
        app, _token, _mod = preview_app
        c = TestClient(app)
        r = c.post("/api/preview/html", json = {"source": "<h1>hi</h1>"})
        assert r.status_code in (401, 403)

    def test_post_returns_same_origin_url_and_ttl(self, preview_app):
        app, token, _mod = preview_app
        c = TestClient(app)
        r = c.post(
            "/api/preview/html",
            json = {"source": "<h1>hello</h1>"},
            headers = {"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["url"].startswith("/api/preview/html/")
        assert isinstance(body["expires_in_seconds"], int)
        assert body["expires_in_seconds"] > 0

    def test_post_rejects_oversize_body(self, preview_app):
        app, token, mod = preview_app
        c = TestClient(app)
        too_big = "x" * (mod.MAX_HTML_PREVIEW_BYTES + 1)
        r = c.post(
            "/api/preview/html",
            json = {"source": too_big},
            headers = {"Authorization": f"Bearer {token}"},
        )
        # Pydantic enforces the max_length so this is a 422 (validation),
        # NOT a 413 -- but either is acceptable as long as it does not get
        # stored. We assert non-2xx and an empty store.
        assert r.status_code >= 400
        assert mod._PREVIEWS == {}

    def test_post_returns_unguessable_tokens(self, preview_app):
        # Two POSTs of the same source must produce two distinct tokens.
        app, token, _mod = preview_app
        c = TestClient(app)
        urls = set()
        for _ in range(5):
            r = c.post(
                "/api/preview/html",
                json = {"source": "<p>same</p>"},
                headers = {"Authorization": f"Bearer {token}"},
            )
            assert r.status_code == 200
            urls.add(r.json()["url"])
        assert len(urls) == 5


class TestGetHtmlPreview:
    def _create(self, app, token, source):
        c = TestClient(app)
        r = c.post(
            "/api/preview/html",
            json = {"source": source},
            headers = {"Authorization": f"Bearer {token}"},
        )
        return r.json()["url"]

    def test_get_serves_stored_html_with_overriding_csp(self, preview_app):
        app, token, _mod = preview_app
        url = self._create(app, token, "<button onclick=\"alert('x')\">go</button>")
        c = TestClient(app)
        r = c.get(url)
        assert r.status_code == 200
        body = r.text
        # The doctype + base + body are present.
        assert "<!doctype html>" in body.lower()
        assert "<base target=\"_blank\">" in body
        assert "<button onclick=\"alert('x')\">go</button>" in body
        # The overriding CSP must permit inline script execution.
        csp = r.headers["content-security-policy"]
        directives = {
            chunk.strip().split(" ", 1)[0]: chunk.strip()
            for chunk in csp.split(";")
            if chunk.strip()
        }
        assert "default-src" in directives
        assert "'none'" in directives["default-src"]
        assert "'unsafe-inline'" in directives["script-src"]
        # Beacon paths are still closed.
        assert "'none'" in directives["connect-src"]
        assert "'none'" in directives["frame-src"]
        # Only same-origin embedders may iframe the preview.
        assert "frame-ancestors" in directives
        assert "'self'" in directives["frame-ancestors"]
        # X-Frame-Options is SAMEORIGIN so the host page can iframe us.
        assert r.headers["x-frame-options"].upper() == "SAMEORIGIN"
        # No caching of preview bodies.
        assert "no-store" in r.headers["cache-control"]

    def test_get_is_not_auth_gated(self, preview_app):
        # Browsers do not attach Authorization to iframe subresource loads.
        # The unguessable URL token IS the authorisation.
        app, token, _mod = preview_app
        url = self._create(app, token, "<p>nope</p>")
        c = TestClient(app)
        r = c.get(url)  # no Authorization header
        assert r.status_code == 200

    def test_get_unknown_token_is_404(self, preview_app):
        app, _token, _mod = preview_app
        c = TestClient(app)
        r = c.get("/api/preview/html/totally-not-a-real-token")
        assert r.status_code == 404

    def test_get_expired_token_is_404(self, preview_app):
        app, token, mod = preview_app
        url = self._create(app, token, "<p>aging</p>")
        # Force-age the stored entry past the TTL by rewinding monotonic.
        token_id = url.rsplit("/", 1)[-1]
        created, src = mod._PREVIEWS[token_id]
        mod._PREVIEWS[token_id] = (created - (mod.PREVIEW_TTL_SECONDS + 5), src)
        c = TestClient(app)
        r = c.get(url)
        assert r.status_code == 404
        # And the entry is swept on access.
        assert token_id not in mod._PREVIEWS


class TestEviction:
    def test_overflow_evicts_oldest_entries(self, preview_app):
        app, token, mod = preview_app
        c = TestClient(app)
        # Pin the cap low so the test is cheap.
        mod.MAX_LIVE_PREVIEWS = 4
        urls = []
        for i in range(6):
            r = c.post(
                "/api/preview/html",
                json = {"source": f"<p>{i}</p>"},
                headers = {"Authorization": f"Bearer {token}"},
            )
            urls.append(r.json()["url"])
            # Force monotonic progression so eviction order is deterministic.
            time.sleep(0.001)
        assert len(mod._PREVIEWS) == mod.MAX_LIVE_PREVIEWS
        # The two oldest tokens (urls[0], urls[1]) must have been evicted.
        for old in urls[:2]:
            token_id = old.rsplit("/", 1)[-1]
            assert token_id not in mod._PREVIEWS
        # Newer tokens are still present.
        for fresh in urls[-mod.MAX_LIVE_PREVIEWS:]:
            token_id = fresh.rsplit("/", 1)[-1]
            assert token_id in mod._PREVIEWS
