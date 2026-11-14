# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A linked folder's sync counts reach the folder's total, even when an earlier sync did most of it."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from core.rag import folder_sync
from routes.rag import _folder_job_view
from storage import rag_db

requires_sqlite_vec = pytest.mark.skipif(
    not rag_db.RAG_AVAILABLE, reason = "sqlite-vec is not installed"
)


def _folder(rag_home: Path):
    source = rag_home / "source"
    source.mkdir()
    row = folder_sync.create_folder(
        scope_type = "knowledge_base",
        scope_id = "scope-1",
        path = str(source),
        name = "Docs",
    )
    return source, row


def _run(folder_id: str) -> dict:
    job_id = folder_sync.request_sync(folder_id)
    folder_sync.reconcile_folder(job_id)
    return folder_sync.get_job(job_id)


@requires_sqlite_vec
def test_resync_counts_files_an_earlier_sync_already_indexed(
    rag_home, stub_embeddings, monkeypatch
):
    source, folder = _folder(rag_home)
    for name in ("a.txt", "b.txt", "c.txt"):
        (source / name).write_text(f"words in {name}", encoding = "utf-8")
    first = _folder_job_view(_run(folder["id"]))
    assert first["status"] == "completed"
    assert first["discoveredFiles"] == 3
    assert first["syncedFiles"] == 3

    for name in ("d.txt", "e.txt"):
        (source / name).write_text(f"fresh words in {name}", encoding = "utf-8")

    seen: list[dict] = []
    original = folder_sync._set_job

    def recording_set_job(job_id, **values):
        original(job_id, **values)
        seen.append(_folder_job_view(folder_sync.get_job(job_id)))

    monkeypatch.setattr(folder_sync, "_set_job", recording_set_job)
    second = _folder_job_view(_run(folder["id"]))

    assert second["status"] == "completed"
    assert second["discoveredFiles"] == 5
    # This pass only indexed the two new files, but the folder is fully synced.
    assert second["processedFiles"] == 2
    assert second["syncedFiles"] == 5

    ingesting = [view for view in seen if view["stage"] == "ingesting"]
    assert ingesting, "the pass never reported progress"
    # From the first ingesting update on, the three unchanged files already count as synced.
    assert ingesting[0]["syncedFiles"] == 3
    counts = [view["syncedFiles"] for view in ingesting]
    assert counts == sorted(counts)
    assert all(count <= view["discoveredFiles"] for count, view in zip(counts, ingesting))


def test_older_job_tables_gain_the_sync_count_columns(tmp_path):
    conn = sqlite3.connect(tmp_path / "old.db")
    try:
        conn.execute(
            "CREATE TABLE linked_folder_sync_jobs (id TEXT PRIMARY KEY, folder_id TEXT, "
            "status TEXT, discovered INTEGER NOT NULL DEFAULT 0, successor_kind TEXT)"
        )
        conn.execute(
            "INSERT INTO linked_folder_sync_jobs(id, folder_id, status) VALUES('j', 'f', 'running')"
        )
        rag_db.ensure_linked_folder_columns(conn)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(linked_folder_sync_jobs)")}
        assert {"up_to_date", "checked"} <= cols
        row = conn.execute("SELECT up_to_date, checked FROM linked_folder_sync_jobs").fetchone()
        assert row == (0, 0)
    finally:
        conn.close()


def test_job_view_without_the_new_columns_reports_zero_synced():
    view = _folder_job_view({"id": "j", "folder_id": "f", "kind": "sync", "status": "running"})
    assert view["syncedFiles"] == 0
