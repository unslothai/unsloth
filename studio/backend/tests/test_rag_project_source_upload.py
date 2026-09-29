# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Project sources upload: the path the create-project dialog drives."""

import io
import os

import pytest
from fastapi import UploadFile

from core.rag import ingestion, store
from routes.rag import _sanitize_filename, _save_upload
from storage import rag_db


def _wait(job_id, timeout = 30.0):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        status = ingestion.get_job_status(job_id)
        if status and status["status"] in ("completed", "failed"):
            return status
        time.sleep(0.05)
    raise AssertionError("ingestion did not finish in time")


def _ingest(project_id, filename, path):
    return ingestion.start_ingestion(
        store.project_scope(project_id), None, None, filename, path, project_id = project_id
    )


def test_project_document_persists_under_its_scope(rag_home, stub_embeddings, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("alpha bravo charlie " * 50, encoding = "utf-8")
    _, job_id = _ingest("P1", "notes.txt", str(path))
    assert _wait(job_id)["status"] == "completed"

    conn = rag_db.get_connection()
    try:
        docs = store.list_documents(conn, store.project_scope("P1"))
        assert [d["filename"] for d in docs] == ["notes.txt"]
        # Scoped: a sibling project cannot see it.
        assert store.list_documents(conn, store.project_scope("P2")) == []
        assert store.search_lexical(conn, store.project_scope("P1"), "bravo", 5)
    finally:
        conn.close()


@pytest.mark.parametrize(
    "raw",
    [
        "x" * 300 + ".txt",
        "y" * 512 + ".PDF",
        "../" * 80 + "deep.md",
    ],
)
def test_long_filenames_keep_their_extension(raw):
    # _save_upload gates on the extension, so trimming it would reject the file.
    out = _sanitize_filename(raw)
    assert len(out) <= 200
    assert os.path.splitext(out)[1].lower() == os.path.splitext(raw)[1].lower()


@pytest.mark.parametrize(
    "raw",
    [
        "../../etc/passwd.txt",
        "..\\..\\windows\\evil.txt",
        "/absolute/notes.txt",
        "C:\\Users\\me\\notes.txt",
    ],
)
def test_sanitized_filenames_carry_no_path(raw):
    out = _sanitize_filename(raw)
    assert "/" not in out and "\\" not in out


@pytest.mark.parametrize("raw", ["." * 300, "noext" * 100, "a" * 100 + "." + "e" * 250])
def test_sanitizer_degrades_safely(raw):
    assert 0 < len(_sanitize_filename(raw)) <= 200


@pytest.mark.parametrize(
    "raw",
    [
        "报告.pdf",
        "日本語ドキュメント.txt",
        "Отчёт 2026.pdf",
        "résumé.docx",
        "My Report.pdf",
        "Q3: Revenue.pdf",
        # macOS keeps a Finder "/" on disk as ":", so this is how a dropped
        # "P/L statement.pdf" reaches the sanitizer. Its first component is the name.
        "P:L statement.pdf",
        "C:notes.txt",
    ],
)
def test_sanitizer_keeps_the_name_the_user_gave(raw):
    assert _sanitize_filename(raw) == raw


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("a\x00b.pdf", "ab.pdf"),
        ("a\u202eb.pdf", "ab.pdf"),
        ("a\u200bb.pdf", "ab.pdf"),
        ("\ufeff报告.pdf", "报告.pdf"),
        ("a\nb.pdf", "a b.pdf"),
        ("a\u2028b.pdf", "a b.pdf"),
        # What a browser actually sends for a file picked on Windows.
        ("C:\\fakepath\\notes.txt", "notes.txt"),
    ],
)
def test_sanitizer_normalizes_what_a_label_cannot_carry(raw, expected):
    assert _sanitize_filename(raw) == expected


def test_uploaded_unicode_name_is_persisted_verbatim(rag_home, stub_embeddings):
    payload = b"alpha bravo charlie " * 50
    stored_path, filename, _ = _save_upload(
        UploadFile(file = io.BytesIO(payload), filename = "报告 2026.txt")
    )
    assert filename == "报告 2026.txt"

    _, job_id = _ingest("P9", filename, stored_path)
    assert _wait(job_id)["status"] == "completed"

    conn = rag_db.get_connection()
    try:
        docs = store.list_documents(conn, store.project_scope("P9"))
        assert [d["filename"] for d in docs] == ["报告 2026.txt"]
    finally:
        conn.close()
