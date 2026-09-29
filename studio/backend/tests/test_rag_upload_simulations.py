# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Upload races and filesystem aliases on real temporary files."""

import os
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from core.rag import ingestion, parsers, store
from storage import rag_db
from utils.paths import ensure_dir, rag_uploads_root
from .test_rag_upload_formats import EXTENSIONS, write_document


@pytest.mark.parametrize("extension", EXTENSIONS)
@pytest.mark.parametrize("fails", [False, True])
def test_concurrent_duplicates_share_one_worker(
    rag_home, stub_embeddings, monkeypatch, extension, fails
):
    uploads = ensure_dir(rag_uploads_root())
    source = write_document(uploads / f"source{extension}")
    copies = [source]
    for index in range(7):
        path = uploads / f"copy-{index}{extension}"
        path.write_bytes(source.read_bytes())
        copies.append(path)
    release, started = threading.Event(), threading.Event()
    calls = []
    parse = parsers.parse

    def blocked(path):
        calls.append(path)
        started.set()
        assert release.wait(15)
        if fails:
            raise ValueError("simulated parser failure")
        return parse(path)

    monkeypatch.setattr(parsers, "parse", blocked)
    scope = store.thread_scope("concurrent")

    def upload(path):
        return ingestion.start_ingestion(scope, None, "concurrent", path.name, str(path))

    try:
        with ThreadPoolExecutor(max_workers = 8) as pool:
            results = list(pool.map(upload, copies))
        assert len(set(results)) == 1
        assert started.wait(10)
        assert len(calls) == 1
        assert Path(calls[0]).exists()
    finally:
        release.set()
    document, job = results[0]
    events = list(ingestion.job_events(job))
    assert events[-1]["type"] == ("error" if fails else "complete")
    conn = rag_db.get_connection()
    try:
        assert len(store.list_documents(conn, scope)) == 1
        assert store.get_document(conn, document)["status"] == ("failed" if fails else "completed")
        if not fails:
            assert store.search_lexical(conn, scope, "zebramarker", 5)
    finally:
        conn.close()


def test_case_alias_cannot_delete_the_original_upload(rag_home, stub_embeddings):
    original = ensure_dir(rag_uploads_root()) / "report.txt"
    original.write_text("Revenue increased. Zebramarker concludes the report.", encoding = "utf-8")
    alias = original.with_name("REPORT.TXT")
    if not alias.exists() or not os.path.samefile(alias, original):
        pytest.skip("Filesystem is case-sensitive")
    scope = store.thread_scope("case-alias")
    document, job = ingestion.start_ingestion(
        scope, None, "case-alias", original.name, str(original), background = False
    )
    assert ingestion.get_job_status(job)["status"] == "completed"
    duplicate, _ = ingestion.start_ingestion(
        scope, None, "case-alias", alias.name, str(alias), background = False
    )
    assert duplicate == document
    assert original.exists()


def test_distinct_case_sensitive_upload_can_be_cleaned_up(rag_home):
    original = ensure_dir(rag_uploads_root()) / "report.txt"
    original.write_text("keep", encoding = "utf-8")
    duplicate = original.with_name("REPORT.TXT")
    if duplicate.exists():
        pytest.skip("Filesystem is case-insensitive")
    duplicate.write_text("remove", encoding = "utf-8")
    ingestion._remove_upload(str(duplicate), keep_path = str(original))
    assert original.read_text(encoding = "utf-8") == "keep"
    assert not duplicate.exists()


def test_unicode_filename_alias_keeps_the_original(rag_home):
    original = ensure_dir(rag_uploads_root()) / "café.txt"
    original.write_text("keep", encoding = "utf-8")
    alias = original.with_name(unicodedata.normalize("NFD", original.name))
    if not alias.exists() or not os.path.samefile(alias, original):
        pytest.skip("Filesystem distinguishes Unicode normalization forms")
    ingestion._remove_upload(str(alias), keep_path = str(original))
    assert original.read_text(encoding = "utf-8") == "keep"


def test_symlink_alias_keeps_the_original(rag_home):
    original = ensure_dir(rag_uploads_root()) / "original.txt"
    original.write_text("keep", encoding = "utf-8")
    alias = original.with_name("alias.txt")
    try:
        alias.symlink_to(original)
    except OSError:
        pytest.skip("Creating symlinks requires additional privileges")
    ingestion._remove_upload(str(alias), keep_path = str(original))
    assert original.read_text(encoding = "utf-8") == "keep"
