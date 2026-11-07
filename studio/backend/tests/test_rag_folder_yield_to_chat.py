# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Linked-folder indexing waits while a local chat generates (#16992)."""

from __future__ import annotations

import threading
from contextlib import closing
from pathlib import Path

import pytest

from core.rag import config, embeddings, folder_sync, ingestion
from state import active_generations
from storage import rag_db

pytestmark = pytest.mark.skipif(not rag_db.RAG_AVAILABLE, reason = "sqlite-vec is not installed")


def _folder(rag_home: Path, files: int = 1) -> dict:
    source = rag_home / "source"
    source.mkdir()
    for i in range(files):
        (source / f"doc{i}.txt").write_text(f"document number {i} words", encoding = "utf-8")
    return folder_sync.create_folder(
        scope_type = "knowledge_base", scope_id = "scope-1", path = str(source), name = "Docs"
    )


def _indexed(folder: dict) -> int:
    with closing(rag_db.get_connection()) as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM documents WHERE linked_folder_id=?", (folder["id"],)
        ).fetchone()[0]


def _start_sync(folder: dict) -> tuple[threading.Thread, list]:
    out: list = []
    job_id = folder_sync.request_sync(folder["id"])

    def run():
        folder_sync.reconcile_folder(job_id)
        out.append(folder_sync.get_job(job_id))

    thread = threading.Thread(target = run, daemon = True)
    thread.start()
    return thread, out


def test_indexing_waits_for_a_local_chat_then_resumes(rag_home, stub_embeddings):
    folder = _folder(rag_home, files = 3)
    with active_generations.ActiveGeneration(threading.Event(), kind = "chat"):
        thread, out = _start_sync(folder)
        # Without the yield the three documents index well inside this window.
        thread.join(timeout = 4)
        assert thread.is_alive()
        assert _indexed(folder) == 0
    thread.join(timeout = 30)

    assert out and out[0]["status"] == "completed"
    assert _indexed(folder) == 3


def test_a_chat_started_mid_document_pauses_it(rag_home, stub_embeddings, monkeypatch):
    folder = _folder(rag_home)
    real_encode, real_set_job = embeddings.encode, ingestion._set_job
    chat = active_generations.ActiveGeneration(threading.Event(), kind = "chat")
    started = threading.Event()
    stages_during_chat: list[str] = []

    def encode_while_a_chat_starts(*args, **kwargs):
        if not started.is_set():
            started.set()
            chat.__enter__()
            threading.Timer(1.0, chat.__exit__, (None, None, None)).start()
        return real_encode(*args, **kwargs)

    def record_set_job(conn, job_id, **kwargs):
        if started.is_set() and active_generations.count():
            stages_during_chat.append(kwargs.get("stage"))
        return real_set_job(conn, job_id, **kwargs)

    monkeypatch.setattr(embeddings, "encode", encode_while_a_chat_starts)
    monkeypatch.setattr(ingestion, "_set_job", record_set_job)
    thread, out = _start_sync(folder)
    thread.join(timeout = 30)

    assert started.is_set()
    assert out and out[0]["status"] == "completed"
    assert stages_during_chat == []
    assert _indexed(folder) == 1


@pytest.mark.parametrize("kwargs", [{"external": True}, {"kind": "embeddings"}])
def test_cloud_chats_and_embedding_requests_do_not_hold_indexing(rag_home, stub_embeddings, kwargs):
    folder = _folder(rag_home)
    with active_generations.ActiveGeneration(threading.Event(), **kwargs):
        thread, out = _start_sync(folder)
        thread.join(timeout = 30)
        assert out and out[0]["status"] == "completed"
    assert _indexed(folder) == 1


def test_switch_off_restores_indexing_during_chat(rag_home, stub_embeddings, monkeypatch):
    monkeypatch.setattr(config, "FOLDER_YIELD_TO_CHAT", False)
    folder = _folder(rag_home)
    with active_generations.ActiveGeneration(threading.Event(), kind = "chat"):
        thread, out = _start_sync(folder)
        thread.join(timeout = 30)
        assert out and out[0]["status"] == "completed"


def test_unlink_ends_the_wait(rag_home, stub_embeddings):
    folder = _folder(rag_home)
    with active_generations.ActiveGeneration(threading.Event(), kind = "chat"):
        thread, _ = _start_sync(folder)
        thread.join(timeout = 2)
        assert thread.is_alive()
        assert folder_sync.delete_folder(folder["id"]) is True
        thread.join(timeout = 30)
        assert not thread.is_alive()
    assert folder_sync.get_folder(folder["id"]) is None


def test_non_folder_work_never_waits():
    with active_generations.ActiveGeneration(threading.Event(), kind = "chat"):
        done = threading.Event()
        threading.Thread(
            target = lambda: (folder_sync.wait_while_chat_busy(), done.set()), daemon = True
        ).start()
        assert done.wait(timeout = 2)
