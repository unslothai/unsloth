# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/api/rag/extract-text: the composer's reader for documents the browser cannot open."""

import io
import os

import pytest
from fastapi import HTTPException, UploadFile

from routes import rag as rag_routes
from utils.paths import rag_uploads_root

MHT = (
    "MIME-Version: 1.0\r\n"
    'Content-Type: multipart/related; boundary="b"\r\n\r\n'
    "--b\r\nContent-Type: text/html; charset=utf-8\r\n\r\n"
    "<html><body><h1>Quarterly</h1><p>Revenue rose 12%.</p></body></html>\r\n"
    "--b--\r\n"
)


def _upload(name: str, data: bytes) -> UploadFile:
    return UploadFile(file = io.BytesIO(data), filename = name)


def _uploads_left() -> list[str]:
    root = rag_uploads_root()
    return os.listdir(root) if os.path.isdir(root) else []


def test_extracts_text_and_keeps_no_copy(rag_home):
    result = rag_routes.extract_text(_upload("report.mht", MHT.encode()), subject = "owner")
    assert result["filename"] == "report.mht"
    assert "Quarterly" in result["text"] and "Revenue rose 12%." in result["text"]
    assert result["truncated"] is False
    assert _uploads_left() == []


def test_works_without_sqlite_vec(rag_home, monkeypatch):
    from storage import rag_db

    monkeypatch.setattr(rag_db, "rag_available", lambda: False)
    result = rag_routes.extract_text(
        _upload("page.xhtml", b"<html><body><p>hello</p></body></html>"), subject = "owner"
    )
    assert result["text"] == "hello"


def test_unsupported_type_is_refused(rag_home):
    with pytest.raises(HTTPException) as exc:
        rag_routes.extract_text(_upload("model.gguf", b"GGUF"), subject = "owner")
    assert exc.value.status_code == 400
    assert _uploads_left() == []


@pytest.mark.parametrize("name", ["broken.doc", "broken.docm", "broken.epub"])
def test_damaged_document_is_422_and_cleaned_up(rag_home, name):
    with pytest.raises(HTTPException) as exc:
        rag_routes.extract_text(_upload(name, b"not a document"), subject = "owner")
    assert exc.value.status_code == 422
    assert name in exc.value.detail
    assert _uploads_left() == []


def test_text_is_capped_on_a_character_boundary(rag_home, monkeypatch):
    monkeypatch.setattr(rag_routes, "_EXTRACT_TEXT_MAX_BYTES", 7)
    html = "<html><body><p>ééééé</p></body></html>".encode()
    result = rag_routes.extract_text(_upload("page.xhtml", html), subject = "owner")
    assert result == {"filename": "page.xhtml", "text": "ééé", "truncated": True}


def test_cap_stops_at_the_page_that_crosses_it(rag_home, monkeypatch):
    from core.rag import parsers

    pages = [parsers._page(text, n) for n, text in enumerate(["first", "  ", "second", "third"], 1)]
    monkeypatch.setattr(parsers, "parse", lambda path: pages)
    monkeypatch.setattr(rag_routes, "_EXTRACT_TEXT_MAX_BYTES", 12)
    result = rag_routes.extract_text(_upload("deck.ppt", b"x"), subject = "owner")
    assert (result["text"], result["truncated"]) == ("first\n\nsecon", True)
    monkeypatch.setattr(rag_routes, "_EXTRACT_TEXT_MAX_BYTES", 1024)
    result = rag_routes.extract_text(_upload("deck.ppt", b"x"), subject = "owner")
    assert (result["text"], result["truncated"]) == ("first\n\nsecond\n\nthird", False)
