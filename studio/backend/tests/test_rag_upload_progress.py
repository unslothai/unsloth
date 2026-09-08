# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Uploads must follow real work and never report an in-flight duplicate as indexed."""

import runpy
import threading

import pytest

from core.rag import captioner, config, ingestion, parsers, store
from storage import rag_db
from .test_rag_upload_formats import EXTENSIONS, write_document


def test_captioning_is_opt_in(monkeypatch):
    monkeypatch.delenv("RAG_CAPTION_IMAGES", raising = False)
    assert runpy.run_path(config.__file__)["CAPTION_IMAGES"] is False
    monkeypatch.setenv("RAG_CAPTION_IMAGES", "1")
    assert runpy.run_path(config.__file__)["CAPTION_IMAGES"] is True


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("extension", EXTENSIONS)
def test_duplicate_upload_follows_original_job(
    rag_home, stub_embeddings, monkeypatch, fails, extension
):
    from utils.paths import ensure_dir, rag_uploads_root

    uploads = ensure_dir(rag_uploads_root())
    original = write_document(uploads / f"report{extension}")
    duplicate = uploads / f"copy{extension}"
    duplicate.write_bytes(original.read_bytes())
    started, release = threading.Event(), threading.Event()
    parse = parsers.parse

    def blocked_parse(path):
        started.set()
        assert release.wait(5)
        if fails:
            raise ValueError("Cannot read document")
        return parse(path)

    monkeypatch.setattr(parsers, "parse", blocked_parse)
    scope = store.thread_scope("upload-test")
    doc_id, job_id = ingestion.start_ingestion(
        scope, None, "upload-test", original.name, str(original)
    )
    try:
        assert started.wait(5)
        duplicate_id, duplicate_job = ingestion.start_ingestion(
            scope, None, "upload-test", duplicate.name, str(duplicate)
        )
        assert duplicate_id == doc_id
        assert duplicate_job == job_id
        assert ingestion.get_job_status(duplicate_job)["status"] == "running"
        assert not duplicate.exists()
        assert original.exists()
    finally:
        release.set()
        events = list(ingestion.job_events(job_id))
    assert events[-1]["type"] == ("error" if fails else "complete")
    assert ingestion.get_job_status(job_id)["status"] == ("failed" if fails else "completed")


def test_same_path_dedup_keeps_original_file(rag_home, stub_embeddings):
    from utils.paths import ensure_dir, rag_uploads_root

    original = ensure_dir(rag_uploads_root()) / "report.txt"
    original.write_text("Quarterly revenue increased substantially.")
    scope = store.thread_scope("same-path")
    doc_id, _ = ingestion.start_ingestion(
        scope, None, "same-path", original.name, str(original), background = False
    )
    repeated_id, _ = ingestion.start_ingestion(
        scope, None, "same-path", original.name, str(original), background = False
    )
    assert repeated_id == doc_id
    assert original.exists()


def test_vision_progress_includes_empty_responses(monkeypatch):
    monkeypatch.setattr(captioner, "_caption_one", lambda *args: None)
    monkeypatch.setattr(captioner, "_ocr_one", lambda *args: None)
    figures = [parsers.ParsedImage(b"png", page, page) for page in (1, 2, 3)]
    caption_progress, ocr_progress = [], []
    assert (
        captioner.caption_images(
            figures,
            endpoint = ("http://local", "local"),
            on_progress = lambda done, total: caption_progress.append((done, total)),
        )
        == {}
    )
    assert (
        captioner.ocr_pages(
            {1: b"png", 2: b"png"},
            endpoint = ("http://local", "local"),
            on_progress = lambda done, total: ocr_progress.append((done, total)),
        )
        == {}
    )
    assert caption_progress == [(1, 3), (2, 3), (3, 3)]
    assert ocr_progress == [(1, 2), (2, 2)]


def test_indexing_reports_batches_and_keeps_text(rag_home, stub_embeddings, monkeypatch, tmp_path):
    monkeypatch.setattr(ingestion, "_EMBED_BATCH", 1)
    monkeypatch.setattr(config, "CHUNK_TOKENS", 4)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 0)
    path = tmp_path / "report.txt"
    path.write_text("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima")
    scope = store.thread_scope("batch-progress")
    doc_id, job_id = ingestion.start_ingestion(
        scope, None, "batch-progress", path.name, str(path), background = False
    )
    events = list(ingestion.job_events(job_id))
    progress = [event["progress"] for event in events if event["type"] == "progress"]
    batches = [event for event in events if event.get("stage") == "embedding"]
    assert len(batches) >= 4
    assert progress == sorted(progress)
    assert events[-1]["type"] == "complete"
    conn = rag_db.get_connection()
    try:
        assert store.get_document(conn, doc_id)["num_chunks"] == 3
        assert store.search_lexical(conn, scope, "juliet", 5)
    finally:
        conn.close()


def test_figure_progress_arrives_before_render_and_after_every_tile(
    rag_home, stub_embeddings, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        parsers, "parse", lambda path: [parsers.Page("Revenue increased this quarter.", 1)]
    )
    monkeypatch.setattr(captioner, "vision_endpoint", lambda: ("http://local", "local"))
    monkeypatch.setattr(parsers, "pages_with_figures", lambda *args, **kwargs: [1])
    monkeypatch.setattr(captioner, "_caption_one", lambda *args: "chart: revenue up")
    events = []
    emit = ingestion._emit

    def record(job, event):
        if event is not None:
            events.append(event)
        emit(job, event)

    monkeypatch.setattr(ingestion, "_emit", record)

    def render(*args, **kwargs):
        assert events[-1]["stage"] == "captioning"
        return [parsers.ParsedImage(b"png", 1, 1)] * 5

    monkeypatch.setattr(parsers, "render_pdf_figure_tiles", render)
    path = tmp_path / "report.pdf"
    path.write_bytes(b"fixture")
    from core.rag import locators

    monkeypatch.setattr(locators, "pdf_regions_for_chunks", lambda *args: None)
    _, job_id = ingestion.start_ingestion(
        store.thread_scope("figures"),
        None,
        "figures",
        path.name,
        str(path),
        caption = True,
        ocr = False,
        background = False,
    )
    assert ingestion.get_job_status(job_id)["status"] == "completed"
    progress = [e["progress"] for e in events if e.get("stage") == "captioning"]
    assert len(progress) == 6
    assert all(a < b for a, b in zip(progress, progress[1:]))


def test_orphaned_duplicate_is_reindexed_instead_of_reported_complete(
    rag_home, stub_embeddings, tmp_path
):
    path = tmp_path / "orphan.txt"
    path.write_text("Revenue doubled this quarter.")
    scope = store.thread_scope("orphan")
    conn = rag_db.get_connection()
    try:
        original = store.create_document(
            conn,
            scope = scope,
            filename = path.name,
            sha256 = ingestion._sha256_file(str(path)),
            status = "pending",
            stored_path = str(path),
        )
    finally:
        conn.close()
    replacement, job = ingestion.start_ingestion(
        scope,
        None,
        "orphan",
        path.name,
        str(path),
        background = False,
    )
    assert replacement != original
    assert ingestion.get_job_status(job)["num_chunks"] > 0
    conn = rag_db.get_connection()
    try:
        assert store.get_document(conn, original) is None
        assert store.get_document(conn, replacement)["status"] == "completed"
    finally:
        conn.close()
