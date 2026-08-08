# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused linked-folder reconciliation tests; embeddings are always stubbed."""

from __future__ import annotations

import asyncio
import io
import os
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.rag import folder_sync, store
from storage import rag_db
from utils.paths import rag_db_path

requires_sqlite_vec = pytest.mark.skipif(
    not rag_db.RAG_AVAILABLE, reason = "sqlite-vec is not installed"
)


def _run(folder_id: str, *, rebuild: bool = False) -> dict:
    job_id = folder_sync.request_sync(folder_id, rebuild = rebuild)
    folder_sync.reconcile_folder(job_id)
    return folder_sync.get_job(job_id)


def _folder(rag_home: Path, scope_type: str = "knowledge_base"):
    source = rag_home / "source"
    source.mkdir()
    row = folder_sync.create_folder(
        scope_type = scope_type,
        scope_id = "scope-1",
        path = str(source),
        name = "Docs",
    )
    return source, row


@requires_sqlite_vec
def test_schema_is_idempotent_and_persists_folder_tables(rag_home):
    first = rag_db.get_connection()
    first.close()
    rag_db._schema_ready = False
    second = rag_db.get_connection()
    try:
        tables = {
            row["name"]
            for row in second.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'linked_folder%'"
            )
        }
        job_columns = {
            row["name"] for row in second.execute("PRAGMA table_info(linked_folder_sync_jobs)")
        }
    finally:
        second.close()
    assert {
        "linked_folders",
        "linked_folder_files",
        "linked_folder_sync_jobs",
        "linked_folder_retired_scopes",
    } <= tables
    assert "rebuild_requested" in job_columns


@requires_sqlite_vec
def test_schema_migrates_legacy_linked_folder_root_identity(rag_home):
    source = rag_home / "legacy-source"
    source.mkdir()
    db_path = rag_db_path()
    db_path.parent.mkdir(parents = True, exist_ok = True)
    legacy = sqlite3.connect(db_path)
    try:
        legacy.executescript(
            """
            CREATE TABLE linked_folders (
                id TEXT NOT NULL PRIMARY KEY,
                scope_type TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                scope TEXT NOT NULL,
                path TEXT NOT NULL,
                name TEXT NOT NULL,
                auto_sync INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'pending',
                last_error TEXT,
                last_scan_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(scope, path)
            );
            """
        )
        legacy.execute(
            "INSERT INTO linked_folders("
            "id, scope_type, scope_id, scope, path, name, created_at, updated_at) "
            "VALUES('legacy', 'project', 'p1', 'project:p1', ?, 'Legacy', 'now', 'now')",
            (str(source),),
        )
        legacy.commit()
    finally:
        legacy.close()

    conn = rag_db.get_connection()
    try:
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(linked_folders)").fetchall()
        }
        row = conn.execute(
            "SELECT root_device, root_inode FROM linked_folders WHERE id='legacy'"
        ).fetchone()
    finally:
        conn.close()
    assert {"root_device", "root_inode"} <= columns
    assert tuple(row) == (None, None)

    rag_db._schema_ready = False
    rerun = rag_db.get_connection()
    rerun.close()

    assert _run("legacy")["status"] == "completed"
    migrated = folder_sync.get_folder("legacy")
    source_stat = source.stat()
    assert (migrated["root_device"], migrated["root_inode"]) == (
        source_stat.st_dev,
        source_stat.st_ino,
    )


@requires_sqlite_vec
def test_reconcile_add_rename_delete_and_skip_unsupported_and_symlinks(rag_home, stub_embeddings):
    source, folder = _folder(rag_home)
    (source / "notes.txt").write_text("alpha original", encoding = "utf-8")
    (source / "ignored.exe").write_text("alpha ignored", encoding = "utf-8")
    outside = rag_home / "outside.txt"
    outside.write_text("alpha outside", encoding = "utf-8")
    try:
        (source / "escape.txt").symlink_to(outside)
    except OSError:
        pass

    first = _run(folder["id"])
    assert first["status"] == "completed"
    assert first["discovered"] == 1
    conn = rag_db.get_connection()
    try:
        mapping = dict(
            conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
        )
        document = store.get_document(conn, mapping["document_id"])
        assert mapping["relative_path"] == "notes.txt"
        assert os.path.realpath(document["stored_path"]) != os.path.realpath(source / "notes.txt")
        assert store.search_lexical(conn, folder["scope"], "original", 5)
    finally:
        conn.close()

    old_document_id = mapping["document_id"]
    (source / "notes.txt").rename(source / "renamed.txt")
    renamed = _run(folder["id"])
    assert renamed["renamed"] == 1
    conn = rag_db.get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
        ).fetchone()
        assert row["relative_path"] == "renamed.txt"
        assert row["document_id"] == old_document_id
    finally:
        conn.close()

    (source / "renamed.txt").unlink()
    deleted = _run(folder["id"])
    assert deleted["deleted"] == 1
    conn = rag_db.get_connection()
    try:
        assert store.get_document(conn, old_document_id) is None
        assert (
            conn.execute(
                "SELECT 1 FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
            is None
        )
    finally:
        conn.close()


@requires_sqlite_vec
def test_extension_changing_rename_reingests_with_the_new_parser(rag_home, stub_embeddings):
    source, folder = _folder(rag_home)
    original = source / "notes.html"
    original.write_text(
        "<p>visible alpha</p><script>hiddenscripttoken</script>",
        encoding = "utf-8",
    )
    assert _run(folder["id"])["status"] == "completed"
    conn = rag_db.get_connection()
    try:
        before = dict(
            conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
        )
        assert not store.search_lexical(conn, folder["scope"], "hiddenscripttoken", 5)
    finally:
        conn.close()

    original.rename(source / "notes.txt")
    result = _run(folder["id"])
    assert result["status"] == "completed"
    assert result["changed"] == 1
    conn = rag_db.get_connection()
    try:
        after = dict(
            conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
        )
        assert after["relative_path"] == "notes.txt"
        assert after["document_id"] != before["document_id"]
        assert store.get_document(conn, before["document_id"]) is None
        assert store.search_lexical(conn, folder["scope"], "hiddenscripttoken", 5)
    finally:
        conn.close()


@requires_sqlite_vec
def test_rename_reuse_verifies_content_before_reusing_the_document(rag_home, stub_embeddings):
    source, folder = _folder(rag_home)
    original = source / "original.txt"
    original.write_text("first words", encoding = "utf-8")
    assert _run(folder["id"])["status"] == "completed"
    conn = rag_db.get_connection()
    try:
        before = dict(
            conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
        )
    finally:
        conn.close()

    renamed = source / "renamed.txt"
    original.rename(renamed)
    renamed.write_text("other words", encoding = "utf-8")
    os.utime(renamed, ns = (renamed.stat().st_atime_ns, before["mtime_ns"]))
    assert renamed.stat().st_ino == before["inode"]
    assert renamed.stat().st_size == before["size_bytes"]

    result = _run(folder["id"])
    assert result["renamed"] == 0
    assert result["changed"] == 1
    conn = rag_db.get_connection()
    try:
        after = dict(
            conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
        )
        assert after["document_id"] != before["document_id"]
        assert store.search_lexical(conn, folder["scope"], "other", 5)
    finally:
        conn.close()


@requires_sqlite_vec
def test_edited_rename_failure_retains_the_prior_mapping(rag_home, stub_embeddings, monkeypatch):
    source, folder = _folder(rag_home)
    original = source / "original.txt"
    original.write_text("durable prior words", encoding = "utf-8")
    assert _run(folder["id"])["status"] == "completed"
    conn = rag_db.get_connection()
    try:
        before = dict(
            conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
        )
    finally:
        conn.close()

    renamed = source / "renamed.txt"
    original.rename(renamed)
    renamed.write_text("replacement content that fails", encoding = "utf-8")
    assert renamed.stat().st_ino == before["inode"]
    monkeypatch.setattr(
        folder_sync.ingestion,
        "start_ingestion",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("embed unavailable")),
    )

    result = _run(folder["id"])
    assert result["status"] == "failed"
    conn = rag_db.get_connection()
    try:
        current = dict(
            conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
        )
        assert current["relative_path"] == "original.txt"
        assert current["document_id"] == before["document_id"]
        assert store.search_lexical(conn, folder["scope"], "durable", 5)
    finally:
        conn.close()


@requires_sqlite_vec
def test_ambiguous_rename_failure_retains_all_prior_mappings(
    rag_home, stub_embeddings, monkeypatch
):
    source, folder = _folder(rag_home)
    for name, text in (("old-a.txt", "durable alpha"), ("old-b.txt", "durable bravo")):
        (source / name).write_text(text, encoding = "utf-8")
    assert _run(folder["id"])["status"] == "completed"
    conn = rag_db.get_connection()
    try:
        conn.execute(
            "UPDATE linked_folder_files SET device=0, inode=0, content_hash=NULL WHERE folder_id=?",
            (folder["id"],),
        )
        conn.commit()
    finally:
        conn.close()
    (source / "old-a.txt").rename(source / "renamed-a.txt")
    (source / "old-b.txt").rename(source / "renamed-b.txt")

    original_scan = folder_sync._scan

    def colliding_scan(*args, **kwargs):
        found, identity = original_scan(*args, **kwargs)
        for metadata in found.values():
            metadata["device"] = 0
            metadata["inode"] = 0
        return found, identity

    original_start = folder_sync.ingestion.start_ingestion

    def fail_one(*args, **kwargs):
        if args[3] == "renamed-b.txt":
            raise RuntimeError("embed unavailable")
        return original_start(*args, **kwargs)

    monkeypatch.setattr(folder_sync, "_scan", colliding_scan)
    monkeypatch.setattr(folder_sync.ingestion, "start_ingestion", fail_one)
    result = _run(folder["id"])

    assert result["status"] == "failed"
    conn = rag_db.get_connection()
    try:
        paths = {
            row["relative_path"]
            for row in conn.execute(
                "SELECT relative_path FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            )
        }
        assert paths == {"old-a.txt", "old-b.txt", "renamed-a.txt"}
        assert store.search_lexical(conn, folder["scope"], "alpha", 5)
        assert store.search_lexical(conn, folder["scope"], "bravo", 5)
    finally:
        conn.close()


@requires_sqlite_vec
def test_changed_file_failure_and_unavailable_scan_retain_prior_index(
    rag_home, stub_embeddings, monkeypatch
):
    source, folder = _folder(rag_home, scope_type = "project")
    path = source / "notes.txt"
    path.write_text("durable prior words", encoding = "utf-8")
    assert _run(folder["id"])["status"] == "completed"
    conn = rag_db.get_connection()
    try:
        prior = dict(
            conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
        )
    finally:
        conn.close()

    path.write_text("replacement content that fails", encoding = "utf-8")
    monkeypatch.setattr(
        folder_sync.ingestion,
        "start_ingestion",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("embed unavailable")),
    )
    failed = _run(folder["id"])
    assert failed["status"] == "failed"
    conn = rag_db.get_connection()
    try:
        current = conn.execute(
            "SELECT document_id FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
        ).fetchone()
        assert current["document_id"] == prior["document_id"]
        assert store.search_lexical(conn, folder["scope"], "durable", 5)
    finally:
        conn.close()

    source.rename(rag_home / "unavailable")
    unavailable = _run(folder["id"])
    assert unavailable["status"] == "failed"
    conn = rag_db.get_connection()
    try:
        assert (
            conn.execute(
                "SELECT document_id FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()["document_id"]
            == prior["document_id"]
        )
    finally:
        conn.close()


@requires_sqlite_vec
def test_replaced_empty_root_fails_and_retains_prior_index(rag_home, stub_embeddings):
    source, folder = _folder(rag_home)
    (source / "notes.txt").write_text("retained root document", encoding = "utf-8")
    assert _run(folder["id"])["status"] == "completed"
    conn = rag_db.get_connection()
    try:
        prior = dict(
            conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
        )
    finally:
        conn.close()

    source.rename(rag_home / "replaced-source")
    source.mkdir()
    result = _run(folder["id"])
    assert result["status"] == "failed"
    assert "identity changed" in result["error"]
    conn = rag_db.get_connection()
    try:
        current = conn.execute(
            "SELECT document_id FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
        ).fetchone()
        assert current["document_id"] == prior["document_id"]
        assert store.get_document(conn, prior["document_id"]) is not None
        assert store.search_lexical(conn, folder["scope"], "retained", 5)
    finally:
        conn.close()


@requires_sqlite_vec
def test_reauthorizing_same_path_refreshes_root_identity_and_retains_mappings(
    rag_home, stub_embeddings
):
    source, folder = _folder(rag_home)
    (source / "notes.txt").write_text("retained after remount", encoding = "utf-8")
    assert _run(folder["id"])["status"] == "completed"
    conn = rag_db.get_connection()
    try:
        mapping_before = dict(
            conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
        )
        conn.execute(
            "UPDATE linked_folders SET root_device=-1, root_inode=-1 WHERE id=?", (folder["id"],)
        )
        conn.commit()
    finally:
        conn.close()

    refreshed = {}
    reauthorized = threading.Event()

    def reauthorize():
        refreshed.update(
            folder_sync.create_folder(
                scope_type = "knowledge_base",
                scope_id = "scope-1",
                path = str(source),
            )
        )
        reauthorized.set()

    with folder_sync._folder_lock(folder["id"]):
        thread = threading.Thread(target = reauthorize)
        thread.start()
        assert not reauthorized.wait(0.1)
    thread.join(timeout = 1)

    source_stat = source.stat()
    assert reauthorized.is_set()
    assert refreshed["id"] == folder["id"]
    assert (refreshed["root_device"], refreshed["root_inode"]) == (
        source_stat.st_dev,
        source_stat.st_ino,
    )
    conn = rag_db.get_connection()
    try:
        mapping_after = dict(
            conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
        )
        assert mapping_after["document_id"] == mapping_before["document_id"]
        assert store.search_lexical(conn, folder["scope"], "remount", 5)
    finally:
        conn.close()


def test_directory_lease_contract_is_purpose_bound_and_testable(rag_home):
    source = rag_home / "source"
    source.mkdir()
    calls = []

    def verify(lease, **kwargs):
        calls.append((lease, kwargs))
        return SimpleNamespace(canonical_path = source)

    from routes.rag import _resolve_linked_folder_path

    assert _resolve_linked_folder_path("signed", verifier = verify) == str(source)
    assert calls == [
        (
            "signed",
            {
                "operation": "link-documents",
                "expected_kind": "document-folder",
                "expected_path_type": "directory",
            },
        )
    ]


def test_validate_folder_rejects_symlink_root(rag_home):
    source = rag_home / "source"
    source.mkdir()
    alias = rag_home / "alias"
    try:
        alias.symlink_to(source, target_is_directory = True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    with pytest.raises(ValueError, match = "Symbolic-link"):
        folder_sync.validate_folder_path(str(alias))


def test_snapshot_copy_is_bounded_to_the_validated_size():
    target = io.BytesIO()
    with pytest.raises(RuntimeError, match = "changed while it was copied"):
        folder_sync._copy_exact(io.BytesIO(b"validated-and-growing"), target, 9)
    assert target.getvalue() == b"validated"


@requires_sqlite_vec
def test_linked_folders_cannot_overlap_within_a_scope(rag_home):
    parent = rag_home / "parent"
    child = parent / "child"
    child.mkdir(parents = True)
    folder_sync.create_folder(scope_type = "project", scope_id = "one", path = str(parent))

    with pytest.raises(ValueError, match = "cannot overlap"):
        folder_sync.create_folder(scope_type = "project", scope_id = "one", path = str(child))

    other_scope = folder_sync.create_folder(scope_type = "project", scope_id = "two", path = str(child))
    assert other_scope["path"] == str(child)


@requires_sqlite_vec
def test_linked_folder_rejects_managed_rag_uploads_overlap(rag_home):
    from utils.paths import ensure_dir, rag_uploads_root

    uploads = ensure_dir(rag_uploads_root())
    uploads_child = uploads / "child"
    uploads_child.mkdir()
    for path in [uploads.parent, uploads, uploads_child]:
        with pytest.raises(ValueError, match = "managed RAG uploads"):
            folder_sync.create_folder(scope_type = "project", scope_id = "one", path = str(path))


def test_backend_routes_match_linked_folder_client_contract():
    from routes.rag import LinkFolderRequest, router

    paths = {route.path for route in router.routes}
    assert {
        "/linked-folders",
        "/knowledge-bases/{kb_id}/linked-folders",
        "/projects/{project_id}/linked-folders",
        "/linked-folders/{folder_id}/sync",
        "/linked-folders/{folder_id}/rebuild",
        "/linked-folder-jobs/{job_id}",
        "/linked-folder-jobs/{job_id}/events",
    } <= paths
    assert "path" not in LinkFolderRequest.model_fields
    assert LinkFolderRequest.model_fields["native_path_lease"].is_required()


@requires_sqlite_vec
def test_same_size_same_mtime_inode_replacement_is_reconciled(rag_home, stub_embeddings):
    source, folder = _folder(rag_home)
    path = source / "notes.txt"
    path.write_text("first text", encoding = "utf-8")
    assert _run(folder["id"])["status"] == "completed"
    conn = rag_db.get_connection()
    try:
        before = dict(
            conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
        )
    finally:
        conn.close()

    replacement = source / "replacement.txt"
    replacement.write_text("other text", encoding = "utf-8")
    os.utime(replacement, ns = (path.stat().st_atime_ns, path.stat().st_mtime_ns))
    replacement.replace(path)
    assert path.stat().st_size == before["size_bytes"]
    assert path.stat().st_mtime_ns == before["mtime_ns"]

    result = _run(folder["id"])
    assert result["changed"] == 1
    conn = rag_db.get_connection()
    try:
        after = dict(
            conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
        )
        assert after["inode"] != before["inode"]
        assert after["document_id"] != before["document_id"]
        assert store.search_lexical(conn, folder["scope"], "other", 5)
    finally:
        conn.close()


@requires_sqlite_vec
def test_content_identical_touch_updates_metadata_without_reembedding(
    rag_home, stub_embeddings, monkeypatch
):
    source, folder = _folder(rag_home)
    path = source / "notes.txt"
    path.write_text("stable searchable text", encoding = "utf-8")
    assert _run(folder["id"])["status"] == "completed"
    conn = rag_db.get_connection()
    try:
        before = dict(
            conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
        )
    finally:
        conn.close()
    assert before["content_hash"]

    os.utime(path, ns = (path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000))
    monkeypatch.setattr(
        folder_sync.ingestion,
        "start_ingestion",
        lambda *args, **kwargs: pytest.fail("content-identical touch was re-ingested"),
    )
    result = _run(folder["id"])
    assert result["status"] == "completed"
    assert result["changed"] == 0
    conn = rag_db.get_connection()
    try:
        after = dict(
            conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
        )
        assert after["document_id"] == before["document_id"]
        assert after["mtime_ns"] != before["mtime_ns"]
        assert store.search_lexical(conn, folder["scope"], "stable", 5)
    finally:
        conn.close()


@requires_sqlite_vec
def test_sync_requests_are_atomically_deduplicated_and_history_is_pruned(rag_home, monkeypatch):
    _, folder = _folder(rag_home)
    barrier = threading.Barrier(3)
    results = []

    def request():
        barrier.wait()
        results.append(folder_sync.request_sync(folder["id"]))

    threads = [threading.Thread(target = request) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()
    assert len(set(results)) == 1
    folder_sync._enqueue_periodic()

    conn = rag_db.get_connection()
    try:
        assert (
            conn.execute(
                "SELECT COUNT(*) AS n FROM linked_folder_sync_jobs "
                "WHERE folder_id=? AND status IN ('pending','running')",
                (folder["id"],),
            ).fetchone()["n"]
            == 1
        )
        conn.execute(
            "UPDATE linked_folder_sync_jobs SET status='completed', completed_at=created_at"
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(folder_sync.config, "FOLDER_JOB_HISTORY_LIMIT", 1)
    active = folder_sync.request_sync(folder["id"])
    conn = rag_db.get_connection()
    try:
        terminal_count = conn.execute(
            "SELECT COUNT(*) AS n FROM linked_folder_sync_jobs "
            "WHERE status IN ('completed','failed')"
        ).fetchone()["n"]
        assert terminal_count <= 1
        assert (
            conn.execute(
                "SELECT id FROM linked_folder_sync_jobs WHERE status='pending'"
            ).fetchone()["id"]
            == active
        )
    finally:
        conn.close()


@requires_sqlite_vec
def test_rebuild_requested_during_running_sync_queues_a_successor(rag_home):
    _, folder = _folder(rag_home)
    sync_job = folder_sync.request_sync(folder["id"])
    conn = rag_db.get_connection()
    try:
        conn.execute("UPDATE linked_folder_sync_jobs SET status='running' WHERE id=?", (sync_job,))
        conn.commit()
    finally:
        conn.close()

    assert folder_sync.request_sync(folder["id"], rebuild = True) == sync_job
    folder_sync.reconcile_folder(sync_job)

    conn = rag_db.get_connection()
    try:
        successor = conn.execute(
            "SELECT * FROM linked_folder_sync_jobs WHERE folder_id=? AND status='pending'",
            (folder["id"],),
        ).fetchone()
        assert successor is not None
        assert successor["id"] != sync_job
        assert successor["kind"] == "rebuild"
    finally:
        conn.close()


@requires_sqlite_vec
def test_requested_rebuild_promotes_an_intervening_pending_sync(rag_home):
    _, folder = _folder(rag_home)
    completed = folder_sync.request_sync(folder["id"])
    conn = rag_db.get_connection()
    try:
        conn.execute(
            "UPDATE linked_folder_sync_jobs "
            "SET status='completed', rebuild_requested=1 WHERE id=?",
            (completed,),
        )
        pending = "intervening-sync"
        conn.execute(
            "INSERT INTO linked_folder_sync_jobs"
            "(id, folder_id, kind, status, stage, created_at) "
            "VALUES(?,?,'sync','pending','queued',?)",
            (pending, folder["id"], folder_sync._now()),
        )
        conn.commit()
    finally:
        conn.close()

    folder_sync._queue_requested_rebuild(completed)
    conn = rag_db.get_connection()
    try:
        successor = conn.execute(
            "SELECT kind, rebuild_requested FROM linked_folder_sync_jobs WHERE id=?", (pending,)
        ).fetchone()
        assert tuple(successor) == ("rebuild", 0)
        assert (
            conn.execute(
                "SELECT rebuild_requested FROM linked_folder_sync_jobs WHERE id=?", (completed,)
            ).fetchone()["rebuild_requested"]
            == 0
        )
    finally:
        conn.close()


@requires_sqlite_vec
def test_pending_rebuild_promotion_clears_a_recovered_successor_flag(rag_home):
    _, folder = _folder(rag_home)
    job_id = folder_sync.request_sync(folder["id"])
    conn = rag_db.get_connection()
    try:
        conn.execute("UPDATE linked_folder_sync_jobs SET rebuild_requested=1 WHERE id=?", (job_id,))
        conn.commit()
    finally:
        conn.close()

    assert folder_sync.request_sync(folder["id"], rebuild = True) == job_id
    conn = rag_db.get_connection()
    try:
        job = conn.execute(
            "SELECT kind, rebuild_requested FROM linked_folder_sync_jobs WHERE id=?", (job_id,)
        ).fetchone()
        assert tuple(job) == ("rebuild", 0)
    finally:
        conn.close()


@requires_sqlite_vec
def test_startup_recovers_terminal_rebuild_handoff(rag_home):
    _, folder = _folder(rag_home)
    completed = folder_sync.request_sync(folder["id"])
    conn = rag_db.get_connection()
    try:
        conn.execute(
            "UPDATE linked_folder_sync_jobs SET status='completed', rebuild_requested=1, "
            "completed_at=created_at WHERE id=?",
            (completed,),
        )
        conn.commit()
    finally:
        conn.close()

    folder_sync._recover_startup_state()

    conn = rag_db.get_connection()
    try:
        successor = conn.execute(
            "SELECT * FROM linked_folder_sync_jobs WHERE folder_id=? AND status='pending'",
            (folder["id"],),
        ).fetchone()
        assert successor is not None
        assert successor["kind"] == "rebuild"
        assert (
            conn.execute(
                "SELECT rebuild_requested FROM linked_folder_sync_jobs WHERE id=?", (completed,)
            ).fetchone()["rebuild_requested"]
            == 0
        )
    finally:
        conn.close()


@requires_sqlite_vec
def test_linked_folder_ingestion_jobs_are_pruned_after_reconciliation(rag_home, stub_embeddings):
    source, folder = _folder(rag_home)
    (source / "notes.txt").write_text("temporary internal job", encoding = "utf-8")

    assert _run(folder["id"])["status"] == "completed"

    conn = rag_db.get_connection()
    try:
        assert conn.execute("SELECT COUNT(*) AS n FROM ingestion_jobs").fetchone()["n"] == 0
    finally:
        conn.close()


def test_document_and_folder_views_expose_management_and_scope_name():
    from routes.rag import _doc_view, _folder_view

    assert (
        _doc_view(
            {"id": "doc", "filename": "x", "status": "completed", "linked_folder_id": "folder"}
        )["managed"]
        is True
    )
    assert _doc_view({"id": "doc", "filename": "x", "status": "completed"})["managed"] is False
    assert (
        _folder_view(
            {
                "id": "folder",
                "name": "Docs",
                "scope_type": "project",
                "scope_id": "p1",
                "scope_name": "Project One",
                "status": "ready",
                "created_at": "now",
            }
        )["scopeName"]
        == "Project One"
    )


@requires_sqlite_vec
def test_global_folder_list_resolves_scope_name(rag_home, monkeypatch):
    from routes.rag import list_linked_folders
    from storage import studio_db

    monkeypatch.setattr(studio_db, "list_chat_projects", lambda **kwargs: [])

    conn = rag_db.get_connection()
    try:
        store.create_kb(conn, name = "Knowledge One", kb_id = "kb1")
    finally:
        conn.close()
    source = rag_home / "source"
    source.mkdir()
    folder_sync.create_folder(scope_type = "knowledge_base", scope_id = "kb1", path = str(source))
    result = list_linked_folders(scope_type = None, scope_id = None, subject = "test")
    assert result["linkedFolders"][0]["scopeName"] == "Knowledge One"


@requires_sqlite_vec
def test_unexpected_reconcile_failure_resets_folder_status(rag_home, monkeypatch):
    _, folder = _folder(rag_home)
    job_id = folder_sync.request_sync(folder["id"])
    conn = rag_db.get_connection()
    try:
        conn.execute("UPDATE linked_folders SET status='syncing' WHERE id=?", (folder["id"],))
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr(
        folder_sync,
        "_reconcile_folder",
        lambda value: (_ for _ in ()).throw(RuntimeError("unexpected")),
    )
    folder_sync.reconcile_folder(job_id)
    current = folder_sync.get_folder(folder["id"])
    assert current["status"] == "error"
    assert current["last_error"] == "unexpected"


@requires_sqlite_vec
def test_reconcile_errors_do_not_persist_native_paths(rag_home, monkeypatch):
    source, folder = _folder(rag_home)
    private_path = source / "private" / "notes.txt"
    monkeypatch.setattr(
        folder_sync,
        "_scan",
        lambda *args: (_ for _ in ()).throw(OSError(f"cannot read {private_path}")),
    )

    result = _run(folder["id"])
    current = folder_sync.get_folder(folder["id"])
    assert result["status"] == "failed"
    assert str(source) not in result["error"]
    assert str(source) not in current["last_error"]
    assert "<native_path>" in result["error"]


@requires_sqlite_vec
def test_shutdown_requeues_a_scan_before_it_mutates_mappings(rag_home, monkeypatch):
    source, folder = _folder(rag_home)
    (source / "notes.txt").write_text("pending shutdown", encoding = "utf-8")
    original_scan = folder_sync._scan

    def stop_after_scan(*args, **kwargs):
        result = original_scan(*args, **kwargs)
        folder_sync._stop.set()
        return result

    monkeypatch.setattr(folder_sync, "_scan", stop_after_scan)
    try:
        job_id = folder_sync.request_sync(folder["id"])
        folder_sync.reconcile_folder(job_id)
        result = folder_sync.get_job(job_id)
    finally:
        folder_sync._stop.clear()

    assert result["status"] == "pending"
    assert result["stage"] == "queued"
    conn = rag_db.get_connection()
    try:
        assert (
            conn.execute(
                "SELECT 1 FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
            is None
        )
    finally:
        conn.close()


def test_start_auto_sync_queues_replacement_for_retired_live_worker(monkeypatch):
    blocker = threading.Event()
    released = threading.Event()

    monkeypatch.setattr(folder_sync.rag_db, "RAG_AVAILABLE", True)
    monkeypatch.setattr(folder_sync.rag_db, "rag_available", lambda: True)
    monkeypatch.setattr(folder_sync, "_recover_startup_state", lambda: None)
    monkeypatch.setattr(folder_sync, "_enqueue_periodic", lambda: None)
    monkeypatch.setattr(folder_sync, "_next_job", lambda: None)

    def parked():
        blocker.wait()
        released.set()

    thread = threading.Thread(target = parked)
    thread.start()
    original_thread = folder_sync._thread
    original_thread_stop = folder_sync._thread_stop
    original_stop = folder_sync._stop.is_set()
    folder_sync._worker_lock.acquire()
    try:
        folder_sync._thread = thread
        folder_sync._thread_stop = threading.Event()
        folder_sync._thread_stop.set()
        folder_sync._stop.set()
        assert folder_sync.start_auto_sync() is True
        assert thread.is_alive()
        assert folder_sync._thread is not thread
        assert folder_sync._thread.is_alive()
    finally:
        blocker.set()
        thread.join(timeout = 1)
        folder_sync._worker_lock.release()
        folder_sync.stop_auto_sync(timeout = 1)
        if not original_stop:
            folder_sync._stop.clear()
        folder_sync._thread = original_thread
        folder_sync._thread_stop = original_thread_stop
    assert released.is_set()


def test_start_auto_sync_skips_runtime_unavailable_rag(monkeypatch):
    monkeypatch.setattr(folder_sync.rag_db, "RAG_AVAILABLE", True)
    monkeypatch.setattr(folder_sync.rag_db, "rag_available", lambda: False)
    monkeypatch.setattr(
        folder_sync.threading,
        "Thread",
        lambda *args, **kwargs: pytest.fail("worker must not be created"),
    )

    assert folder_sync.start_auto_sync() is False


@requires_sqlite_vec
def test_unlink_of_another_folder_does_not_wait_for_active_folder(rag_home):
    first_source = rag_home / "first"
    second_source = rag_home / "second"
    first_source.mkdir()
    second_source.mkdir()
    first = folder_sync.create_folder(scope_type = "project", scope_id = "one", path = str(first_source))
    second = folder_sync.create_folder(
        scope_type = "project", scope_id = "two", path = str(second_source)
    )
    deleted = threading.Event()

    with folder_sync._folder_lock(first["id"]):
        thread = threading.Thread(
            target = lambda: (folder_sync.delete_folder(second["id"]), deleted.set())
        )
        thread.start()
        thread.join(timeout = 1)

    assert deleted.is_set()
    assert folder_sync.get_folder(second["id"]) is None


def test_project_rag_cleanup_runs_off_the_event_loop(monkeypatch):
    from routes import chat_history

    project = {
        "id": "p1",
        "name": "Project",
        "createdAt": 1,
        "updatedAt": 1,
    }
    calls = []
    monkeypatch.setattr(chat_history, "get_chat_project", lambda project_id: project)
    monkeypatch.setattr(chat_history, "list_chat_threads", lambda **kwargs: [])
    monkeypatch.setattr(chat_history, "_cancel_active_research", lambda request, ids: None)
    monkeypatch.setattr(
        chat_history,
        "_retire_project_rag_sources",
        lambda project_id: (calls.append(("retire", threading.get_ident())), [dict(id = "f1")])[1],
    )
    monkeypatch.setattr(
        chat_history,
        "delete_chat_project",
        lambda *args, **kwargs: (calls.append(("delete", threading.get_ident())), project)[1],
    )
    monkeypatch.setattr(
        chat_history,
        "_delete_project_rag_sources",
        lambda project_id, folders: calls.append(("cleanup", threading.get_ident(), folders)),
    )
    event_loop_thread = threading.get_ident()
    result = asyncio.run(
        chat_history.delete_project("p1", SimpleNamespace(), current_subject = "test")
    )
    assert result.id == "p1"
    assert [call[0] for call in calls] == ["retire", "delete", "cleanup"]
    assert calls[0][1] != event_loop_thread
    assert calls[1][1] == event_loop_thread
    assert calls[2][1] != event_loop_thread
    assert calls[2][2] == [dict(id = "f1")]


def test_project_rag_retirement_failure_prevents_project_deletion(monkeypatch):
    from routes import chat_history

    deleted = []
    monkeypatch.setattr(chat_history, "get_chat_project", lambda project_id: dict(id = project_id))
    monkeypatch.setattr(chat_history, "list_chat_threads", lambda **kwargs: [])
    monkeypatch.setattr(chat_history, "_cancel_active_research", lambda request, ids: None)
    monkeypatch.setattr(
        chat_history,
        "_retire_project_rag_sources",
        lambda project_id: (_ for _ in ()).throw(sqlite3.OperationalError("database is busy")),
    )
    monkeypatch.setattr(
        chat_history,
        "delete_chat_project",
        lambda *args, **kwargs: deleted.append(True),
    )

    with pytest.raises(sqlite3.OperationalError, match = "database is busy"):
        asyncio.run(chat_history.delete_project("p1", SimpleNamespace(), current_subject = "test"))
    assert deleted == []


def test_project_deletion_skips_runtime_unavailable_rag(monkeypatch):
    from routes import chat_history

    project = {
        "id": "p1",
        "name": "Project",
        "createdAt": 1,
        "updatedAt": 1,
    }
    deleted = []
    monkeypatch.setattr(rag_db, "rag_available", lambda: False)
    monkeypatch.setattr(
        folder_sync,
        "retire_scope",
        lambda scope: pytest.fail("runtime-unavailable RAG must not retire a scope"),
    )
    monkeypatch.setattr(chat_history, "get_chat_project", lambda project_id: project)
    monkeypatch.setattr(chat_history, "list_chat_threads", lambda **kwargs: [])
    monkeypatch.setattr(chat_history, "_cancel_active_research", lambda request, ids: None)
    monkeypatch.setattr(
        chat_history,
        "delete_chat_project",
        lambda *args, **kwargs: (deleted.append(True), project)[1],
    )

    result = asyncio.run(
        chat_history.delete_project("p1", SimpleNamespace(), current_subject = "test")
    )
    assert result.id == "p1"
    assert deleted == [True]


@requires_sqlite_vec
def test_project_rag_cleanup_retires_every_folder_before_best_effort_deletion(
    rag_home, monkeypatch
):
    from routes import chat_history

    scope = store.project_scope("project")
    folders = []
    for name in ("first", "second"):
        source = rag_home / name
        source.mkdir()
        folders.append(
            folder_sync.create_folder(scope_type = "project", scope_id = "project", path = str(source))
        )
    failed_id = folders[0]["id"]
    original_delete = folder_sync.delete_folder

    def delete(folder_id, **kwargs):
        if folder_id == failed_id:
            raise sqlite3.OperationalError("database is busy")
        return original_delete(folder_id, **kwargs)

    monkeypatch.setattr(folder_sync, "delete_folder", delete)
    chat_history._delete_project_rag_sources("project")

    remaining = folder_sync.list_folders(scope)
    assert [folder["id"] for folder in remaining] == [failed_id]
    assert remaining[0]["auto_sync"] == 0
    assert remaining[0]["status"] == "retired"
    with pytest.raises(KeyError):
        folder_sync.request_sync(failed_id)
    with pytest.raises(KeyError):
        folder_sync.update_folder(failed_id, auto_sync = True)
    replacement = rag_home / "replacement"
    replacement.mkdir()
    with pytest.raises(ValueError, match = "scope no longer exists"):
        folder_sync.create_folder(scope_type = "project", scope_id = "project", path = str(replacement))


@requires_sqlite_vec
def test_kb_deletion_retires_scope_before_best_effort_folder_cleanup(
    rag_home, stub_embeddings, monkeypatch
):
    from routes import rag as rag_routes

    conn = rag_db.get_connection()
    try:
        store.create_kb(conn, name = "Knowledge", kb_id = "knowledge")
    finally:
        conn.close()
    folders = []
    for name in ("kb-first", "kb-second"):
        source = rag_home / name
        source.mkdir()
        folders.append(
            folder_sync.create_folder(
                scope_type = "knowledge_base", scope_id = "knowledge", path = str(source)
            )
        )
    (rag_home / "kb-first" / "notes.txt").write_text("managed snapshot", encoding = "utf-8")
    assert _run(folders[0]["id"])["status"] == "completed"
    conn = rag_db.get_connection()
    try:
        stored_path = conn.execute(
            "SELECT stored_path FROM documents WHERE linked_folder_id=?", (folders[0]["id"],)
        ).fetchone()["stored_path"]
    finally:
        conn.close()
    assert os.path.isfile(stored_path)
    failed_id = folders[0]["id"]
    original_delete = folder_sync.delete_folder

    def delete(folder_id, **kwargs):
        if folder_id == failed_id:
            raise sqlite3.OperationalError("database is busy")
        return original_delete(folder_id, **kwargs)

    monkeypatch.setattr(folder_sync, "delete_folder", delete)
    assert rag_routes.delete_knowledge_base("knowledge", subject = "test") == {"ok": True}

    remaining = folder_sync.list_folders(store.kb_scope("knowledge"))
    assert [folder["id"] for folder in remaining] == [failed_id]
    assert remaining[0]["status"] == "retired"
    assert not os.path.exists(stored_path)
    replacement = rag_home / "kb-replacement"
    replacement.mkdir()
    with pytest.raises(ValueError, match = "scope no longer exists"):
        folder_sync.create_folder(
            scope_type = "knowledge_base", scope_id = "knowledge", path = str(replacement)
        )


def test_preview_containment_is_component_aware(rag_home):
    from routes.rag import _is_managed_preview_path
    from utils.paths import ensure_dir, rag_uploads_root

    uploads = ensure_dir(rag_uploads_root())
    inside = uploads / "doc.txt"
    inside.write_text("inside", encoding = "utf-8")
    prefix_sibling = uploads.parent / f"{uploads.name}-evil" / "doc.txt"
    prefix_sibling.parent.mkdir()
    prefix_sibling.write_text("outside", encoding = "utf-8")

    assert _is_managed_preview_path(str(inside))
    assert not _is_managed_preview_path(str(prefix_sibling))
