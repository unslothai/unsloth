# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A linked-folder sync while the embedding model's Settings transfer is still pending."""

from __future__ import annotations

import json
import time
from contextlib import closing

import pytest

from core.rag import config, embeddings, folder_sync
from storage import rag_db

requires_sqlite_vec = pytest.mark.skipif(
    not rag_db.RAG_AVAILABLE, reason = "sqlite-vec is not installed"
)


@pytest.fixture(autouse = True)
def _forget_pending_marker():
    # The resolution memo is process-wide: a pending marker left here would make later llama-server tests cache-only.
    from utils import embedding_model_settings as ems

    ems._resolved_gguf_memo.clear()
    ems._invalidate_cache()
    yield
    ems._resolved_gguf_memo.clear()
    ems._invalidate_cache()


def _run(folder_id: str, *, rebuild: bool = False) -> dict:
    job_id = folder_sync.request_sync(folder_id, rebuild = rebuild)
    folder_sync.reconcile_folder(job_id)
    return folder_sync.get_job(job_id)


def _run_periodic(folder_id: str) -> dict:
    """The background schedule's pass, which, unlike Sync changes, keeps recently failed files out."""
    folder_sync._enqueue_periodic()
    with closing(rag_db.get_connection()) as conn:
        job_id = conn.execute(
            "SELECT id FROM linked_folder_sync_jobs WHERE folder_id=? AND status='pending'",
            (folder_id,),
        ).fetchone()["id"]
    folder_sync.reconcile_folder(job_id)
    return folder_sync.get_job(job_id)


def _folder(rag_home, names):
    source = rag_home / "source"
    source.mkdir()
    for index, name in enumerate(names):
        (source / name).write_text(f"note {index} about lighthouses", encoding = "utf-8")
    row = folder_sync.create_folder(
        scope_type = "knowledge_base",
        scope_id = "scope-1",
        path = str(source),
        name = "Vault",
    )
    return source, row


def _folder_row(folder_id: str) -> dict:
    with closing(rag_db.get_connection()) as conn:
        return dict(
            conn.execute("SELECT * FROM linked_folders WHERE id=?", (folder_id,)).fetchone()
        )


def _mapped(folder_id: str) -> set[str]:
    with closing(rag_db.get_connection()) as conn:
        return {
            row["relative_path"]
            for row in conn.execute(
                "SELECT relative_path FROM linked_folder_files WHERE folder_id=?", (folder_id,)
            )
        }


@pytest.fixture
def pending_download(rag_home, monkeypatch):
    """Mark the active embedding model as activated before its transfer finished. ``state["landed"]``
    flips the transfer to complete; until then the loader raises as the real cache-only one does."""
    from utils.embedding_model_settings import (
        clear_stored_download_pending,
        set_rag_embedding_model,
    )

    model = config.effective_embedding_model()
    set_rag_embedding_model(model, backend = "sentence-transformers", download_pending = True)
    state = {"landed": False, "warms": 0}

    def warm(model_name = None):
        state["warms"] += 1
        name = model_name or config.effective_embedding_model()
        if not state["landed"]:
            raise embeddings.EmbeddingModelDownloadRequiredError(
                embeddings.download_required_message(name)
            )
        clear_stored_download_pending(name)

    monkeypatch.setattr(embeddings, "warm", warm)
    return state


def test_download_required_skips_the_loader_when_nothing_is_pending(rag_home, monkeypatch):
    def warm(model_name = None):
        raise AssertionError("no transfer is pending, so nothing should load")

    monkeypatch.setattr(embeddings, "warm", warm)
    assert embeddings.download_required() is False


def test_download_required_follows_the_pending_transfer(pending_download):
    assert embeddings.download_required() is True
    pending_download["landed"] = True
    assert embeddings.download_required() is False
    # The loader retired the marker, so later checks no longer load anything.
    warms = pending_download["warms"]
    assert embeddings.download_required() is False
    assert pending_download["warms"] == warms


def test_download_required_leaves_other_load_failures_to_the_documents(rag_home, monkeypatch):
    from utils.embedding_model_settings import set_rag_embedding_model

    set_rag_embedding_model(
        config.effective_embedding_model(), backend = "sentence-transformers", download_pending = True
    )

    def warm(model_name = None):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(embeddings, "warm", warm)
    assert embeddings.download_required() is False


def test_download_required_error_text_is_recognized():
    message = embeddings.download_required_message("unsloth/bge-small-en-v1.5")
    assert message == (
        "Embedding model 'unsloth/bge-small-en-v1.5' is not downloaded yet. "
        "Finish its Settings download before indexing documents."
    )
    assert embeddings.is_download_required_error(f"RuntimeError: {message}")
    assert not embeddings.is_download_required_error("Unsupported file type")
    assert not embeddings.is_download_required_error(None)


@requires_sqlite_vec
def test_sync_waits_for_a_pending_download_instead_of_failing_every_file(
    rag_home, stub_embeddings, pending_download, monkeypatch
):
    names = [f"note-{index:02d}.md" for index in range(12)]
    _, folder = _folder(rag_home, names)
    started = []
    real_start = folder_sync.ingestion.start_ingestion

    def start(*args, **kwargs):
        started.append(args[3])
        return real_start(*args, **kwargs)

    monkeypatch.setattr(folder_sync.ingestion, "start_ingestion", start)

    waiting = _run(folder["id"])

    assert waiting["status"] == "failed"
    assert waiting["failed"] == 0
    assert "is not downloaded yet" in waiting["error"]
    assert started == []
    row = _folder_row(folder["id"])
    assert row["status"] == "error"
    assert "is not downloaded yet" in row["last_error"]
    assert row["failed_files"] is None
    assert _mapped(folder["id"]) == set()

    # The periodic sync, not a Rebuild, indexes everything once the transfer lands.
    pending_download["landed"] = True
    synced = _run_periodic(folder["id"])

    assert synced["status"] == "completed", synced
    assert synced["added"] == len(names)
    assert _mapped(folder["id"]) == set(names)
    assert _folder_row(folder["id"])["failed_files"] is None


@requires_sqlite_vec
def test_rebuild_during_a_pending_download_keeps_the_index(
    rag_home, stub_embeddings, pending_download
):
    pending_download["landed"] = True
    _, folder = _folder(rag_home, ["a.md", "b.md"])
    assert _run(folder["id"])["status"] == "completed"

    from utils.embedding_model_settings import set_rag_embedding_model

    set_rag_embedding_model(
        config.effective_embedding_model(), backend = "sentence-transformers", download_pending = True
    )
    pending_download["landed"] = False

    rebuild = _run(folder["id"], rebuild = True)

    assert rebuild["status"] == "failed"
    assert "is not downloaded yet" in rebuild["error"]
    assert _mapped(folder["id"]) == {"a.md", "b.md"}


@requires_sqlite_vec
def test_a_download_required_failure_mid_sync_is_retried_by_the_next_sync(
    rag_home, stub_embeddings, monkeypatch
):
    names = ["a.md", "b.md", "c.md"]
    _, folder = _folder(rag_home, names)
    real_start = folder_sync.ingestion.start_ingestion
    model = config.effective_embedding_model()
    calls = {"pending": True}

    def start(*args, **kwargs):
        if calls["pending"] and args[3] != "a.md":
            # The transfer went pending after the sync's up-front check.
            raise RuntimeError(embeddings.download_required_message(model))
        if args[3] == "a.md" and calls["pending"]:
            raise RuntimeError("Unsupported or corrupt file")
        return real_start(*args, **kwargs)

    monkeypatch.setattr(folder_sync.ingestion, "start_ingestion", start)

    first = _run(folder["id"])

    assert first["status"] == "failed"
    assert first["failed"] == 3
    held_back = json.loads(_folder_row(folder["id"])["failed_files"] or "{}")
    # Only the genuinely bad file is held back from the next periodic sync.
    assert set(held_back) == {"a.md"}

    calls["pending"] = False
    second = _run_periodic(folder["id"])

    assert _mapped(folder["id"]) == {"b.md", "c.md"}
    assert second["failed"] == 1


@requires_sqlite_vec
def test_failures_saved_before_the_fix_for_a_pending_download_retry(rag_home, stub_embeddings):
    names = ["a.md", "b.md"]
    _, folder = _folder(rag_home, names)
    scan, _ = folder_sync._scan(folder["path"], None, {}, set())
    message = embeddings.download_required_message(config.effective_embedding_model())
    stored = {
        rel: {
            "sig": folder_sync._failure_signature(scan[rel]),
            "at": time.time(),
            "error": message,
        }
        for rel in names
    }
    with closing(rag_db.get_connection()) as conn:
        conn.execute(
            "UPDATE linked_folders SET failed_files=? WHERE id=?",
            (json.dumps(stored), folder["id"]),
        )
        conn.commit()

    result = _run_periodic(folder["id"])

    assert result["status"] == "completed", result
    assert _mapped(folder["id"]) == set(names)
