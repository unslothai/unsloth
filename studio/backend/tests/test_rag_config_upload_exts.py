# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RAG_UPLOAD_EXTS env override for linked-folder and upload extension gating."""

import importlib
import io

import pytest
from fastapi import HTTPException, UploadFile

from core.rag import config
from routes.rag import _save_upload


@pytest.fixture(autouse = True)
def _restore_rag_upload_exts_env(monkeypatch):
    yield
    monkeypatch.delenv("RAG_UPLOAD_EXTS", raising = False)
    importlib.reload(config)


def _reload_rag_config(monkeypatch, value: str | None):
    if value is None:
        monkeypatch.delenv("RAG_UPLOAD_EXTS", raising = False)
    else:
        monkeypatch.setenv("RAG_UPLOAD_EXTS", value)
    importlib.reload(config)


def test_upload_exts_default_when_env_unset(monkeypatch):
    _reload_rag_config(monkeypatch, None)
    assert config.UPLOAD_EXTS == {".pdf", ".txt", ".md", ".markdown", ".docx", ".html", ".htm"}


def test_upload_exts_from_env_normalizes_case_and_whitespace(monkeypatch):
    _reload_rag_config(monkeypatch, " .MD , .Markdown ")
    assert config.UPLOAD_EXTS == {".md", ".markdown"}


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
    _reload_rag_config(monkeypatch, exts)
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
