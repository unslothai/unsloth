# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import importlib
import io

import pytest
from fastapi import HTTPException, UploadFile

from core.rag import config
from routes.rag import _save_upload


@pytest.fixture(autouse = True)
def _restore_config(monkeypatch):
    yield
    monkeypatch.delenv("RAG_UPLOAD_EXTS", raising = False)
    importlib.reload(config)


def _reload(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("RAG_UPLOAD_EXTS", raising = False)
    else:
        monkeypatch.setenv("RAG_UPLOAD_EXTS", value)
    importlib.reload(config)


@pytest.mark.parametrize(
    "value, expected",
    [
        (
            None,
            config.SUPPORTED_UPLOAD_EXTS | config.SOURCE_TEXT_EXTS | config.DOCUMENT_UPLOAD_EXTS,
        ),
        ("", config.SUPPORTED_UPLOAD_EXTS | config.SOURCE_TEXT_EXTS | config.DOCUMENT_UPLOAD_EXTS),
        (" .MD , .Markdown ", {".md", ".markdown"}),
        ("md,pdf", {".md", ".pdf"}),
        (".md,.exe", {".md"}),
        (".py,.exe", {".py"}),
        (".xlsx,.epub", {".xlsx", ".epub"}),
        (
            ".exe",
            config.SUPPORTED_UPLOAD_EXTS | config.SOURCE_TEXT_EXTS | config.DOCUMENT_UPLOAD_EXTS,
        ),
    ],
)
def test_upload_exts_env(monkeypatch, value, expected):
    _reload(monkeypatch, value)
    assert config.UPLOAD_EXTS == expected
    assert config.UPLOAD_EXTS is not config.SUPPORTED_UPLOAD_EXTS


@pytest.mark.parametrize(
    "exts, filename, allowed",
    [
        (".md", "notes.md", True),
        (".md", "paper.pdf", False),
        (".pdf,.txt", "readme.txt", True),
        (".pdf,.txt", "readme.md", False),
    ],
)
def test_save_upload_respects_upload_exts_env(rag_home, monkeypatch, exts, filename, allowed):
    _reload(monkeypatch, exts)
    upload = UploadFile(file = io.BytesIO(b"content"), filename = filename)
    if allowed:
        _, out_name, content_hash = _save_upload(upload)
        assert out_name == filename
        assert content_hash
    else:
        with pytest.raises(HTTPException) as exc:
            _save_upload(upload)
        assert exc.value.status_code == 400
        assert "Unsupported file type" in exc.value.detail
