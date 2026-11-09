# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chat PDF export: real text for every script, no local file or network reads, auth required."""

import base64
import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pymupdf = pytest.importorskip("pymupdf")

from auth.authentication import get_current_subject
import routes.chat_export as chat_export


def _client(authenticated: bool = True) -> TestClient:
    app = FastAPI()
    app.include_router(chat_export.router, prefix = "/api/chat-export")
    if authenticated:
        app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    return TestClient(app)


def _text(pdf: bytes) -> str:
    with pymupdf.open("pdf", pdf) as document:
        return "".join(page.get_text() for page in document)


def test_renders_text_in_several_scripts_with_title_metadata():
    body = (
        '<section class="message user"><h3 class="role">User</h3>'
        "<div class='content'><p>Hello 你好 Привет</p></div></section>"
        "<pre><code>print(4)</code></pre>"
    )
    response = _client().post("/api/chat-export/pdf", json = {"title": "My chat", "html": body})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    text = _text(response.content)
    for needle in ("My chat", "Hello", "你好", "Привет", "print(4)"):
        assert needle in text
    with pymupdf.open("pdf", response.content) as document:
        assert document.metadata["title"] == "My chat"


def test_long_chats_span_pages():
    body = "".join(f"<p>line {index}</p>" for index in range(400))
    response = _client().post("/api/chat-export/pdf", json = {"title": "", "html": body})
    assert response.status_code == 200
    with pymupdf.open("pdf", response.content) as document:
        assert document.page_count > 1
    assert "line 399" in _text(response.content)


def test_only_data_images_embed(tmp_path):
    png = io.BytesIO()
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 4, 4), False)
    png.write(pixmap.tobytes("png"))
    local = tmp_path / "secret.png"
    local.write_bytes(png.getvalue())
    encoded = base64.b64encode(png.getvalue()).decode()

    def image_count(source: str) -> int:
        response = _client().post(
            "/api/chat-export/pdf", json = {"title": "x", "html": f'<img src="{source}">'}
        )
        assert response.status_code == 200
        with pymupdf.open("pdf", response.content) as document:
            return len(document[0].get_images())

    assert image_count(f"data:image/png;base64,{encoded}") == 1
    assert image_count(str(local)) == 0
    assert image_count(local.as_uri()) == 0


def test_page_cap_returns_413(monkeypatch):
    monkeypatch.setattr(chat_export, "MAX_EXPORT_PAGES", 1)
    body = "".join(f"<p>line {index}</p>" for index in range(400))
    response = _client().post("/api/chat-export/pdf", json = {"title": "x", "html": body})
    assert response.status_code == 413


def test_requires_authentication():
    response = _client(authenticated = False).post(
        "/api/chat-export/pdf", json = {"title": "x", "html": "<p>x</p>"}
    )
    assert response.status_code in (401, 403)


def test_registered_under_api_chat_export():
    import routes
    assert routes.chat_export_router is chat_export.router
