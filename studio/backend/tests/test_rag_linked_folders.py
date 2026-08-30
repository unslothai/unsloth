# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused linked-folder reconciliation tests; embeddings are always stubbed."""

from __future__ import annotations

import asyncio
import io
import os
import sqlite3
import threading
import time
from contextlib import closing, contextmanager
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


def _connection(*, metadata: bool = False):
    connect = rag_db.get_metadata_connection if metadata else rag_db.get_connection
    return closing(connect())


def _row(sql: str, params = ()) -> dict | None:
    with _connection() as conn:
        row = conn.execute(sql, params).fetchone()
    return dict(row) if row else None


def _mapping(folder: dict) -> dict | None:
    return _row("SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],))


class _CommitFails:
    """A connection whose commit always fails, to exercise rollback ordering."""

    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def commit(self):
        raise sqlite3.OperationalError("database is locked")


@requires_sqlite_vec
def test_startup_preserves_foreign_leased_work_until_its_lease_expires(rag_home):
    from core.rag import ingestion, job_leases

    _, folder = _folder(rag_home)
    sync_job = folder_sync.request_sync(folder["id"])
    with _connection() as conn:
        document_id = store.create_document(
            conn,
            scope = folder["scope"],
            filename = "in-flight.txt",
            sha256 = "foreign",
            linked_folder_id = folder["id"],
            linked_relative_path = "in-flight.txt",
        )
        ingestion_job = ingestion._new_job(conn, document_id, folder["scope"])
        conn.execute("UPDATE linked_folder_sync_jobs SET status='running' WHERE id=?", (sync_job,))
        conn.execute(
            "UPDATE rag_job_leases SET owner_id='foreign', expires_at='9999-12-31' "
            "WHERE kind=? AND job_id=?",
            (job_leases.INGESTION, ingestion_job),
        )
        conn.execute(
            "INSERT INTO rag_job_leases(kind, job_id, owner_id, expires_at) "
            "VALUES(?,?,'foreign','9999-12-31')",
            (job_leases.FOLDER_SYNC, sync_job),
        )
        conn.commit()

    folder_sync._recover_startup_state()
    assert folder_sync.get_job(sync_job)["status"] == "running"
    assert folder_sync._claim_job(sync_job) is None
    with _connection() as conn:
        assert store.get_document(conn, document_id) is not None
        conn.execute(
            "DELETE FROM rag_job_leases WHERE kind=? AND job_id=?",
            (job_leases.INGESTION, ingestion_job),
        )
        conn.commit()

    folder_sync._recover_startup_state()
    with _connection() as conn:
        assert store.get_document(conn, document_id) is not None
        conn.execute("UPDATE rag_job_leases SET expires_at='2000-01-01'")
        conn.commit()

    folder_sync._recover_startup_state()
    assert folder_sync.get_job(sync_job)["status"] == "pending"
    with _connection() as conn:
        assert store.get_document(conn, document_id) is None


@requires_sqlite_vec
def test_failed_replacement_commit_keeps_the_prior_snapshot_readable(
    rag_home, stub_embeddings, monkeypatch
):
    """A rollback restores the old document, so its source must still be on disk."""
    source, folder = _folder(rag_home)
    path = source / "notes.txt"
    path.write_text("first searchable text", encoding = "utf-8")
    assert _run(folder["id"])["status"] == "completed"
    before = _mapping(folder)
    with _connection() as conn:
        stored_path = store.get_document(conn, before["document_id"])["stored_path"]

    path.write_text("second searchable text", encoding = "utf-8")
    with _connection() as conn:
        replacement = store.create_document(
            conn, scope = folder["scope"], filename = "notes.txt", sha256 = "second"
        )
    stat = path.stat()
    metadata = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }

    real_connection = folder_sync.rag_db.get_connection
    with monkeypatch.context() as patched:
        patched.setattr(
            folder_sync.rag_db, "get_connection", lambda: _CommitFails(real_connection())
        )
        with pytest.raises(sqlite3.OperationalError):
            folder_sync._install_mapping(folder, "notes.txt", metadata, replacement, "second")

    assert _mapping(folder)["document_id"] == before["document_id"]
    assert os.path.isfile(stored_path)
    with _connection() as conn:
        assert store.search_lexical(conn, folder["scope"], "first", 5)


@requires_sqlite_vec
def test_failed_deletion_commit_keeps_the_snapshot_readable(rag_home, stub_embeddings, monkeypatch):
    """An auto_sync=0 folder may not reconcile again for a long time."""
    source, folder = _folder(rag_home)
    path = source / "notes.txt"
    path.write_text("only searchable text", encoding = "utf-8")
    assert _run(folder["id"])["status"] == "completed"
    before = _mapping(folder)
    with _connection() as conn:
        stored_path = store.get_document(conn, before["document_id"])["stored_path"]

    real_connection = folder_sync.rag_db.get_connection
    with monkeypatch.context() as patched:
        patched.setattr(
            folder_sync.rag_db, "get_connection", lambda: _CommitFails(real_connection())
        )
        with pytest.raises(sqlite3.OperationalError):
            folder_sync._delete_mapping(folder["id"], "notes.txt")

    assert _mapping(folder)["document_id"] == before["document_id"]
    assert os.path.isfile(stored_path)
    with _connection() as conn:
        assert store.search_lexical(conn, folder["scope"], "searchable", 5)


@requires_sqlite_vec
def test_skipped_reconciliation_releases_a_claim_the_queue_already_activated(rag_home):
    """Otherwise the heartbeat renews it forever and the unlink never returns."""
    from core.rag import job_leases

    _, folder = _folder(rag_home)
    job_id = folder_sync.request_sync(folder["id"])
    assert folder_sync._claim_job(job_id) == (job_id, folder["id"])
    with _connection() as conn:
        assert job_leases.owned_by_this_process(conn, job_leases.FOLDER_SYNC, job_id)
        # delete_folder() fails the claimed job before reconciliation starts.
        conn.execute("UPDATE linked_folder_sync_jobs SET status='failed' WHERE id=?", (job_id,))
        conn.commit()

    folder_sync.reconcile_folder(job_id)

    assert (
        _row(
            "SELECT 1 FROM rag_job_leases WHERE kind=? AND job_id=?",
            (job_leases.FOLDER_SYNC, job_id),
        )
        is None
    )


@requires_sqlite_vec
def test_unlink_completes_when_the_queue_claimed_the_job_first(rag_home):
    _, folder = _folder(rag_home)
    job_id = folder_sync.request_sync(folder["id"])
    assert folder_sync._claim_job(job_id) == (job_id, folder["id"])

    unlinked = []
    worker = threading.Thread(
        target = lambda: unlinked.append(folder_sync.delete_folder(folder["id"])),
        daemon = True,
    )
    worker.start()
    time.sleep(0.1)
    folder_sync.reconcile_folder(job_id)
    worker.join(timeout = 10)

    assert not worker.is_alive(), "unlink is still waiting on a leaked claim"
    assert unlinked == [True]


def test_worker_retries_after_a_failed_queue_selection(monkeypatch):
    stop = threading.Event()
    attempts = []

    def flaky_next_job():
        attempts.append(1)
        if len(attempts) == 1:
            raise sqlite3.OperationalError("database is locked")
        stop.set()
        return None

    monkeypatch.setattr(folder_sync, "_recover_startup_state", lambda: None)
    monkeypatch.setattr(folder_sync, "_enqueue_periodic", lambda: None)
    monkeypatch.setattr(folder_sync, "_next_job", flaky_next_job)

    try:
        folder_sync._worker(stop)
    finally:
        del folder_sync._worker_state.stop_event

    assert len(attempts) == 2


@requires_sqlite_vec
def test_periodic_scheduling_reaps_orphans_a_survivor_never_saw_at_startup(rag_home):
    """The survivor's own startup pass ran before the other backend crashed."""
    from core.rag import ingestion, job_leases

    _, folder = _folder(rag_home)
    folder_sync._recover_startup_state()
    with _connection() as conn:
        document_id = store.create_document(
            conn,
            scope = folder["scope"],
            filename = "orphan.txt",
            sha256 = "orphan",
            linked_folder_id = folder["id"],
            linked_relative_path = "orphan.txt",
        )
        ingestion_job = ingestion._new_job(conn, document_id, folder["scope"])
        conn.execute(
            "UPDATE rag_job_leases SET owner_id='crashed', expires_at='2000-01-01' "
            "WHERE kind=? AND job_id=?",
            (job_leases.INGESTION, ingestion_job),
        )
        conn.commit()

    folder_sync._enqueue_periodic()

    with _connection() as conn:
        assert store.get_document(conn, document_id) is None
        assert (
            conn.execute("SELECT 1 FROM ingestion_jobs WHERE id=?", (ingestion_job,)).fetchone()
            is None
        )


@requires_sqlite_vec
def test_normal_scheduling_reclaims_an_expired_running_job(rag_home):
    from core.rag import job_leases

    _, folder = _folder(rag_home)
    job_id = folder_sync.request_sync(folder["id"])
    with _connection() as conn:
        conn.execute("UPDATE linked_folder_sync_jobs SET status='running' WHERE id=?", (job_id,))
        conn.execute(
            "INSERT INTO rag_job_leases(kind, job_id, owner_id, expires_at) "
            "VALUES(?,?,'foreign','2000-01-01')",
            (job_leases.FOLDER_SYNC, job_id),
        )
        conn.commit()

    try:
        assert folder_sync._next_job() == (job_id, folder["id"])
    finally:
        job_leases.release(job_leases.FOLDER_SYNC, job_id)


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
    with _connection() as conn:
        mapping = dict(
            conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
        )
        document = store.get_document(conn, mapping["document_id"])
        assert mapping["relative_path"] == "notes.txt"
        assert os.path.realpath(document["stored_path"]) != os.path.realpath(source / "notes.txt")
        assert store.search_lexical(conn, folder["scope"], "original", 5)
        assert conn.execute("SELECT COUNT(*) FROM ingestion_jobs").fetchone()[0] == 0
        from routes.rag import _doc_view

        assert _doc_view(document)["managed"] is True

    old_document_id = mapping["document_id"]
    (source / "notes.txt").rename(source / "renamed.txt")
    renamed = _run(folder["id"])
    assert renamed["renamed"] == 1
    with _connection() as conn:
        row = conn.execute(
            "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
        ).fetchone()
        assert row["relative_path"] == "renamed.txt"
        assert row["document_id"] == old_document_id

    (source / "renamed.txt").unlink()
    deleted = _run(folder["id"])
    assert deleted["deleted"] == 1
    with _connection() as conn:
        assert store.get_document(conn, old_document_id) is None
        assert (
            conn.execute(
                "SELECT 1 FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
            is None
        )


def test_scan_skips_revisited_directory_identity(rag_home, monkeypatch):
    source = rag_home / "cycle"
    loop = source / "loop"
    loop.mkdir(parents = True)
    (source / "notes.txt").write_text("alpha", encoding = "utf-8")
    original_scandir = os.scandir
    scanned = []

    def cycle_scandir(path):
        path = os.fspath(path)
        scanned.append(path)
        if path == str(loop):
            if scanned.count(path) > 1:
                raise AssertionError("revisited directory identity")
            return original_scandir(source)
        return original_scandir(path)

    monkeypatch.setattr(folder_sync.os, "scandir", cycle_scandir)

    found, _ = folder_sync._scan(str(source))

    assert set(found) == {"notes.txt"}
    assert scanned.count(str(loop)) == 1


@pytest.mark.parametrize("directory_identity", [(0, 0), (123, 456)])
def test_scan_keeps_distinct_directories_with_duplicate_identity(
    rag_home, monkeypatch, directory_identity
):
    source = rag_home / "duplicate-identities"
    first = source / "first"
    second = source / "second"
    first.mkdir(parents = True)
    second.mkdir()
    (first / "first.txt").write_text("first", encoding = "utf-8")
    (second / "second.txt").write_text("second", encoding = "utf-8")
    original_scandir = os.scandir

    class WeakIdentityEntry:
        def __init__(self, entry):
            self._entry = entry

        def __getattr__(self, name):
            return getattr(self._entry, name)

        def stat(self, *, follow_symlinks = True):
            if self._entry.is_dir(follow_symlinks = follow_symlinks):
                return SimpleNamespace(st_dev = directory_identity[0], st_ino = directory_identity[1])
            return self._entry.stat(follow_symlinks = follow_symlinks)

    class WeakIdentityScandir:
        def __init__(self, path):
            self._entries = original_scandir(path)

        def __enter__(self):
            self._entries.__enter__()
            return self

        def __exit__(self, *args):
            return self._entries.__exit__(*args)

        def __iter__(self):
            return (WeakIdentityEntry(entry) for entry in self._entries)

    monkeypatch.setattr(folder_sync, "_root_identity", lambda root: directory_identity)
    monkeypatch.setattr(folder_sync.os, "scandir", WeakIdentityScandir)

    found, _ = folder_sync._scan(str(source))

    assert set(found) == {"first/first.txt", "second/second.txt"}


@requires_sqlite_vec
def test_reconcile_pins_one_embedding_model_for_every_file(rag_home, stub_embeddings, monkeypatch):
    source, folder = _folder(rag_home)
    (source / "first.txt").write_text("first words", encoding = "utf-8")
    (source / "second.txt").write_text("second words", encoding = "utf-8")
    resolved = []
    models = []

    def effective_model():
        resolved.append(True)
        return "model/one" if len(resolved) == 1 else "model/two"

    original_start = folder_sync.ingestion.start_ingestion

    def capture_model(*args, **kwargs):
        models.append(kwargs.get("model_name"))
        return original_start(*args, **kwargs)

    monkeypatch.setattr(folder_sync.config, "effective_embedding_model", effective_model)
    monkeypatch.setattr(folder_sync.ingestion, "start_ingestion", capture_model)

    assert _run(folder["id"])["status"] == "completed"
    assert len(resolved) == 1
    assert models == ["model/one", "model/one"]


@requires_sqlite_vec
def test_reconcile_overlaps_the_next_parse_with_embedding(rag_home, stub_embeddings, monkeypatch):
    source, folder = _folder(rag_home)
    (source / "a-first.txt").write_text("first document words", encoding = "utf-8")
    (source / "b-second.txt").write_text("second document words", encoding = "utf-8")
    monkeypatch.setattr(folder_sync.config, "FOLDER_INGEST_WORKERS", 2)

    from core.rag import embeddings

    real_parse = folder_sync.ingestion.parsers.parse
    real_encode = embeddings.encode
    second_parsed = threading.Event()
    overlap_observed = []

    def observe_parse(path):
        pages = real_parse(path)
        if any("second document" in page.text for page in pages):
            second_parsed.set()
        return pages

    def observe_encode(
        texts,
        *,
        model_name = None,
        normalize = True,
    ):
        if any("first document" in text for text in texts):
            overlap_observed.append(second_parsed.wait(5))
        return real_encode(texts, model_name = model_name, normalize = normalize)

    monkeypatch.setattr(folder_sync.ingestion.parsers, "parse", observe_parse)
    monkeypatch.setattr(embeddings, "encode", observe_encode)

    result = _run(folder["id"])

    assert result["status"] == "completed"
    assert overlap_observed == [True]
    assert (
        _row(
            "SELECT COUNT(*) AS count FROM linked_folder_files WHERE folder_id=?",
            (folder["id"],),
        )["count"]
        == 2
    )


@requires_sqlite_vec
def test_reconcile_keeps_successful_sibling_when_parallel_ingest_fails(
    rag_home, stub_embeddings, monkeypatch
):
    source, folder = _folder(rag_home)
    (source / "a-good.txt").write_text("successful sibling words", encoding = "utf-8")
    (source / "b-fails.txt").write_text("failed sibling words", encoding = "utf-8")
    monkeypatch.setattr(folder_sync.config, "FOLDER_INGEST_WORKERS", 2)
    real_start = folder_sync.ingestion.start_ingestion

    def fail_one(*args, **kwargs):
        if args[3] == "b-fails.txt":
            raise RuntimeError("synthetic ingestion failure")
        return real_start(*args, **kwargs)

    monkeypatch.setattr(folder_sync.ingestion, "start_ingestion", fail_one)

    result = _run(folder["id"])

    assert result["status"] == "failed"
    assert result["failed"] == 1
    with _connection() as conn:
        paths = {
            row["relative_path"]
            for row in conn.execute(
                "SELECT relative_path FROM linked_folder_files WHERE folder_id=?",
                (folder["id"],),
            )
        }
        assert paths == {"a-good.txt"}
        assert store.search_lexical(conn, folder["scope"], "successful", 5)
        assert not store.search_lexical(conn, folder["scope"], "failed", 5)


@requires_sqlite_vec
def test_reconcile_retains_mapping_when_missing_file_reappears(
    rag_home, stub_embeddings, monkeypatch
):
    source, folder = _folder(rag_home)
    linked = source / "notes.txt"
    linked.write_text("durable words", encoding = "utf-8")
    assert _run(folder["id"])["status"] == "completed"
    before = _mapping(folder)
    linked.unlink()
    original_scan = folder_sync._scan

    def restore_after_scan(*args, **kwargs):
        result = original_scan(*args, **kwargs)
        linked.write_text("durable words", encoding = "utf-8")
        return result

    monkeypatch.setattr(folder_sync, "_scan", restore_after_scan)
    result = _run(folder["id"])

    assert result["status"] == "failed"
    assert result["deleted"] == 0
    with _connection() as conn:
        after = dict(
            conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
        )
        assert after["document_id"] == before["document_id"]
        assert store.search_lexical(conn, folder["scope"], "durable", 5)


@requires_sqlite_vec
def test_extension_changing_rename_reingests_with_the_new_parser(rag_home, stub_embeddings):
    source, folder = _folder(rag_home)
    original = source / "notes.html"
    original.write_text(
        "<p>visible alpha</p><script>hiddenscripttoken</script>",
        encoding = "utf-8",
    )
    assert _run(folder["id"])["status"] == "completed"
    with _connection() as conn:
        before = dict(
            conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
        )
        assert not store.search_lexical(conn, folder["scope"], "hiddenscripttoken", 5)

    original.rename(source / "notes.txt")
    result = _run(folder["id"])
    assert result["status"] == "completed"
    assert result["added"] == 1
    assert result["deleted"] == 1
    with _connection() as conn:
        after = dict(
            conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
        )
        assert after["relative_path"] == "notes.txt"
        assert after["document_id"] != before["document_id"]
        assert store.get_document(conn, before["document_id"]) is None
        assert store.search_lexical(conn, folder["scope"], "hiddenscripttoken", 5)


@requires_sqlite_vec
def test_rename_reuse_verifies_content_before_reusing_the_document(rag_home, stub_embeddings):
    source, folder = _folder(rag_home)
    original = source / "original.txt"
    original.write_text("first words", encoding = "utf-8")
    assert _run(folder["id"])["status"] == "completed"
    before = _mapping(folder)

    renamed = source / "renamed.txt"
    original.rename(renamed)
    renamed.write_text("other words", encoding = "utf-8")
    os.utime(renamed, ns = (renamed.stat().st_atime_ns, before["mtime_ns"]))
    assert renamed.stat().st_ino == before["inode"]
    assert renamed.stat().st_size == before["size_bytes"]

    result = _run(folder["id"])
    assert result["renamed"] == 0
    assert result["added"] == 1
    assert result["deleted"] == 1
    with _connection() as conn:
        after = dict(
            conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
        )
        assert after["document_id"] != before["document_id"]
        assert store.search_lexical(conn, folder["scope"], "other", 5)


@requires_sqlite_vec
def test_edited_rename_failure_retains_the_prior_mapping(rag_home, stub_embeddings, monkeypatch):
    source, folder = _folder(rag_home)
    original = source / "original.txt"
    original.write_text("durable prior words", encoding = "utf-8")
    assert _run(folder["id"])["status"] == "completed"
    before = _mapping(folder)

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
    with _connection() as conn:
        current = dict(
            conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
        )
        assert current["relative_path"] == "original.txt"
        assert current["document_id"] == before["document_id"]
        assert store.search_lexical(conn, folder["scope"], "durable", 5)


@requires_sqlite_vec
def test_rename_over_existing_path_failure_retains_both_prior_mappings(
    rag_home, stub_embeddings, monkeypatch
):
    source, folder = _folder(rag_home)
    old = source / "old.txt"
    destination = source / "destination.txt"
    old.write_text("durable source", encoding = "utf-8")
    destination.write_text("durable destination", encoding = "utf-8")
    assert _run(folder["id"])["status"] == "completed"
    os.replace(old, destination)
    monkeypatch.setattr(
        folder_sync.ingestion,
        "start_ingestion",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("embed unavailable")),
    )

    result = _run(folder["id"])

    assert result["status"] == "failed"
    with _connection() as conn:
        paths = {
            row["relative_path"]
            for row in conn.execute(
                "SELECT relative_path FROM linked_folder_files WHERE folder_id=?",
                (folder["id"],),
            )
        }
        assert paths == {"old.txt", "destination.txt"}
        assert store.search_lexical(conn, folder["scope"], "source", 5)
        assert store.search_lexical(conn, folder["scope"], "destination", 5)


@requires_sqlite_vec
def test_changed_file_failure_and_unavailable_scan_retain_prior_index(
    rag_home, stub_embeddings, monkeypatch
):
    source, folder = _folder(rag_home, scope_type = "project")
    path = source / "notes.txt"
    path.write_text("durable prior words", encoding = "utf-8")
    assert _run(folder["id"])["status"] == "completed"
    prior = _mapping(folder)

    path.write_text("replacement content that fails", encoding = "utf-8")
    monkeypatch.setattr(
        folder_sync.ingestion,
        "start_ingestion",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("embed unavailable")),
    )
    failed = _run(folder["id"])
    assert failed["status"] == "failed"
    with _connection() as conn:
        current = conn.execute(
            "SELECT document_id FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
        ).fetchone()
        assert current["document_id"] == prior["document_id"]
        assert store.search_lexical(conn, folder["scope"], "durable", 5)

    source.rename(rag_home / "unavailable")
    unavailable = _run(folder["id"])
    assert unavailable["status"] == "failed"
    with _connection() as conn:
        assert (
            conn.execute(
                "SELECT document_id FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()["document_id"]
            == prior["document_id"]
        )


@requires_sqlite_vec
def test_replaced_empty_root_fails_and_retains_prior_index(rag_home, stub_embeddings):
    source, folder = _folder(rag_home)
    (source / "notes.txt").write_text("retained root document", encoding = "utf-8")
    assert _run(folder["id"])["status"] == "completed"
    prior = _mapping(folder)

    source.rename(rag_home / "replaced-source")
    source.mkdir()
    result = _run(folder["id"])
    assert result["status"] == "failed"
    assert "identity changed" in result["error"]
    with _connection() as conn:
        current = conn.execute(
            "SELECT document_id FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
        ).fetchone()
        assert current["document_id"] == prior["document_id"]
        assert store.get_document(conn, prior["document_id"]) is not None
        assert store.search_lexical(conn, folder["scope"], "retained", 5)


@requires_sqlite_vec
def test_reauthorizing_same_path_refreshes_root_identity_and_retains_mappings(
    rag_home, stub_embeddings
):
    source, folder = _folder(rag_home)
    (source / "notes.txt").write_text("retained after remount", encoding = "utf-8")
    assert _run(folder["id"])["status"] == "completed"
    with _connection() as conn:
        mapping_before = dict(
            conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
        )
        conn.execute(
            "UPDATE linked_folders SET root_device=-1, root_inode=-1 WHERE id=?", (folder["id"],)
        )
        conn.commit()

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
    with _connection() as conn:
        mapping_after = dict(
            conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
        )
        assert mapping_after["document_id"] == mapping_before["document_id"]
        assert store.search_lexical(conn, folder["scope"], "remount", 5)


@pytest.mark.parametrize("source_name", ["source", "source "])
def test_directory_lease_contract_preserves_path_and_purpose(rag_home, source_name):
    source = rag_home / source_name
    source.mkdir()
    if source_name.endswith(" "):
        (rag_home / source_name.rstrip()).mkdir()
    calls = []
    identity = (source.stat().st_dev, source.stat().st_ino)

    def verify(lease, **kwargs):
        calls.append((lease, kwargs))
        return SimpleNamespace(
            canonical_path = source,
            device_id = identity[0],
            file_id = identity[1],
        )

    from routes.rag import _resolve_linked_folder_path

    assert _resolve_linked_folder_path("signed", verifier = verify) == (str(source), identity)
    assert calls[0][0] == "signed"
    assert calls[0][1] == {
        "operation": "link-documents",
        "expected_kind": "document-folder",
        "expected_path_type": "directory",
    }


@requires_sqlite_vec
def test_registration_rechecks_the_signed_folder_identity(rag_home, monkeypatch):
    source = rag_home / "identity-race"
    source.mkdir()
    signed_identity = (source.stat().st_dev, source.stat().st_ino)
    replacement_identity = (signed_identity[0], signed_identity[1] + 1)
    identities = iter((signed_identity, replacement_identity))
    monkeypatch.setattr(folder_sync, "_root_identity", lambda path: next(identities))

    with pytest.raises(ValueError, match = "changed after it was selected"):
        folder_sync.create_folder(
            scope_type = "project",
            scope_id = "identity-race",
            path = str(source),
            expected_identity = signed_identity,
        )

    assert folder_sync.list_folders(store.project_scope("identity-race")) == []


@requires_sqlite_vec
def test_large_windows_root_identity_round_trips_through_sqlite(rag_home, monkeypatch):
    source = rag_home / "large-identity"
    source.mkdir()
    identity = (1 << 63, 1 << 127)
    monkeypatch.setattr(folder_sync, "_root_identity", lambda path: identity)

    folder = folder_sync.create_folder(
        scope_type = "knowledge_base",
        scope_id = "large-identity",
        path = str(source),
        expected_identity = identity,
    )

    assert folder_sync._load_identity(folder["root_device"], folder["root_inode"]) == identity


@requires_sqlite_vec
def test_large_windows_file_identity_round_trips_through_sqlite(
    rag_home, stub_embeddings, monkeypatch
):
    """CPython 3.12+ reads st_ino from FILE_ID_INFO, so a ReFS file id is 128-bit."""
    source, folder = _folder(rag_home)
    (source / "notes.txt").write_text("large identity text", encoding = "utf-8")
    identity = (1 << 63, 1 << 127)
    real_scan, real_snapshot = folder_sync._scan, folder_sync._snapshot

    def scan(root, expected_identity = None):
        found, root_identity = real_scan(root, expected_identity)
        for metadata in found.values():
            metadata["scanned"], (metadata["device"], metadata["inode"]) = (
                (metadata["device"], metadata["inode"]),
                identity,
            )
        return found, root_identity

    monkeypatch.setattr(folder_sync, "_scan", scan)
    monkeypatch.setattr(
        folder_sync,
        "_snapshot",
        lambda root, metadata: real_snapshot(
            root, {**metadata, "device": metadata["scanned"][0], "inode": metadata["scanned"][1]}
        ),
    )

    assert _run(folder["id"])["status"] == "completed"
    row = _mapping(folder)
    assert folder_sync._load_identity(row["device"], row["inode"]) == identity
    # The same identity must still compare equal, so nothing re-embeds every sync.
    assert _run(folder["id"])["changed"] == 0


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


def _snapshot_metadata(document: Path, **overrides) -> dict:
    stats = os.stat(document)
    return {
        "path": str(document),
        "size_bytes": stats.st_size,
        "mtime_ns": stats.st_mtime_ns,
        "device": stats.st_dev,
        "inode": stats.st_ino,
    } | overrides


def test_snapshot_copies_the_source_byte_for_byte(rag_home):
    """CRLF runs and 0x1A bytes must survive: a text-mode read would eat both."""
    source = rag_home / "binary-source"
    source.mkdir()
    document = source / "notes.md"
    payload = b"first line\r\nsecond line\x1athird line\r\n"
    document.write_bytes(payload)

    snapshot = folder_sync._snapshot(str(source), _snapshot_metadata(document))

    try:
        assert Path(snapshot).read_bytes() == payload
    finally:
        folder_sync._remove_snapshot(snapshot)


def test_snapshot_accepts_a_source_the_scan_could_not_identify(rag_home):
    """os.scandir reports st_dev/st_ino as 0 on Windows, so every file failed there."""
    source = rag_home / "identity-less"
    source.mkdir()
    document = source / "notes.md"
    payload = b"windows scandir reports no identity\n"
    document.write_bytes(payload)
    metadata = _snapshot_metadata(document, device = 0, inode = 0)

    snapshot = folder_sync._snapshot(str(source), metadata)

    try:
        assert Path(snapshot).read_bytes() == payload
    finally:
        folder_sync._remove_snapshot(snapshot)


def test_snapshot_still_rejects_a_changed_source_without_an_identity(rag_home):
    source = rag_home / "identity-less-changed"
    source.mkdir()
    document = source / "notes.md"
    document.write_bytes(b"original")
    metadata = _snapshot_metadata(document, device = 0, inode = 0)
    document.write_bytes(b"replaced with a longer body")

    with pytest.raises(RuntimeError, match = "changed during reconciliation"):
        folder_sync._snapshot(str(source), metadata)


def test_snapshot_rejects_a_source_swapped_mid_copy_without_an_identity(rag_home, monkeypatch):
    """The post-copy check compares fstat to fstat, so it still works with no scan identity."""
    source = rag_home / "identity-less-swapped"
    source.mkdir()
    document = source / "notes.md"
    document.write_bytes(b"stable body")
    metadata = _snapshot_metadata(document, device = 0, inode = 0)
    real_copy = folder_sync._copy_exact

    def copy_then_touch(src, dst, size):
        real_copy(src, dst, size)
        with open(document, "ab") as handle:
            handle.write(b" appended")

    monkeypatch.setattr(folder_sync, "_copy_exact", copy_then_touch)

    with pytest.raises(RuntimeError, match = "changed while it was copied"):
        folder_sync._snapshot(str(source), metadata)


def test_snapshot_compares_the_identity_when_the_scan_recorded_one(rag_home):
    source = rag_home / "identity-kept"
    source.mkdir()
    document = source / "notes.md"
    document.write_bytes(b"same size body")
    metadata = _snapshot_metadata(document, inode = os.stat(document).st_ino + 1)

    with pytest.raises(RuntimeError, match = "changed during reconciliation"):
        folder_sync._snapshot(str(source), metadata)


def test_snapshot_ignores_a_path_recovered_identity_os_fstat_disagrees_with(rag_home):
    """Shared-folder and WebDAV drivers give os.lstat and os.fstat different ids for one file."""
    source = rag_home / "identity-unstable"
    source.mkdir()
    document = source / "notes.md"
    payload = b"the scan recovered an id os.fstat will not repeat\n"
    document.write_bytes(payload)
    metadata = _snapshot_metadata(
        document, inode = os.stat(document).st_ino + 1, identity_from_path = True
    )

    snapshot = folder_sync._snapshot(str(source), metadata)

    try:
        assert Path(snapshot).read_bytes() == payload
    finally:
        folder_sync._remove_snapshot(snapshot)


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
    before = _mapping(folder)

    replacement = source / "replacement.txt"
    replacement.write_text("other text", encoding = "utf-8")
    os.utime(replacement, ns = (path.stat().st_atime_ns, path.stat().st_mtime_ns))
    replacement.replace(path)
    assert path.stat().st_size == before["size_bytes"]
    assert path.stat().st_mtime_ns == before["mtime_ns"]

    result = _run(folder["id"])
    assert result["changed"] == 1
    with _connection() as conn:
        after = dict(
            conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
        )
        assert after["inode"] != before["inode"]
        assert after["document_id"] != before["document_id"]
        assert store.search_lexical(conn, folder["scope"], "other", 5)


class _IdentitylessStat:
    """DirEntry.stat as Windows returns it: FindFirstFileW carries no file index."""

    st_dev = 0
    st_ino = 0

    def __init__(self, stats):
        self._stats = stats

    def __getattr__(self, name):
        return getattr(self._stats, name)


class _IdentitylessEntry:
    def __init__(self, entry):
        self._entry = entry

    def stat(self, *, follow_symlinks = True):
        return _IdentitylessStat(self._entry.stat(follow_symlinks = follow_symlinks))

    def __getattr__(self, name):
        return getattr(self._entry, name)


@pytest.fixture
def windows_scandir(monkeypatch):
    """Strip the identity os.scandir cannot supply on Windows, leaving os.lstat intact."""
    real_scandir = os.scandir

    @contextmanager
    def scandir(directory):
        with real_scandir(directory) as entries:
            yield [_IdentitylessEntry(entry) for entry in entries]

    monkeypatch.setattr(folder_sync.os, "scandir", scandir)


@requires_sqlite_vec
def test_same_size_same_mtime_replacement_is_reconciled_without_a_scandir_identity(
    rag_home, stub_embeddings, windows_scandir
):
    """The scan must recover the identity itself, or Windows indexes go stale forever."""
    source, folder = _folder(rag_home)
    path = source / "notes.txt"
    path.write_text("first text", encoding = "utf-8")
    assert _run(folder["id"])["status"] == "completed"
    before = _mapping(folder)
    assert before["inode"] not in (None, 0, "0")

    replacement = source / "replacement.txt"
    replacement.write_text("other text", encoding = "utf-8")
    os.utime(replacement, ns = (path.stat().st_atime_ns, path.stat().st_mtime_ns))
    replacement.replace(path)
    assert path.stat().st_size == before["size_bytes"]
    assert path.stat().st_mtime_ns == before["mtime_ns"]

    assert _run(folder["id"])["changed"] == 1
    with _connection() as conn:
        after = _mapping(folder)
        assert after["inode"] != before["inode"]
        assert after["document_id"] != before["document_id"]
        assert store.search_lexical(conn, folder["scope"], "other", 5)


@requires_sqlite_vec
def test_content_identical_touch_updates_metadata_without_reembedding(
    rag_home, stub_embeddings, monkeypatch
):
    source, folder = _folder(rag_home)
    path = source / "notes.txt"
    path.write_text("stable searchable text", encoding = "utf-8")
    assert _run(folder["id"])["status"] == "completed"
    before = _mapping(folder)
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
    with _connection() as conn:
        after = dict(
            conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
        )
        assert after["document_id"] == before["document_id"]
        assert after["mtime_ns"] != before["mtime_ns"]
        assert store.search_lexical(conn, folder["scope"], "stable", 5)


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

    with _connection() as conn:
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

    monkeypatch.setattr(folder_sync.config, "FOLDER_JOB_HISTORY_LIMIT", 1)
    active = folder_sync.request_sync(folder["id"])
    with _connection() as conn:
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


@requires_sqlite_vec
def test_rebuild_requested_during_running_sync_queues_a_successor(rag_home):
    _, folder = _folder(rag_home)
    sync_job = folder_sync.request_sync(folder["id"])
    with _connection() as conn:
        conn.execute("UPDATE linked_folder_sync_jobs SET status='running' WHERE id=?", (sync_job,))
        conn.commit()

    assert folder_sync.request_sync(folder["id"], rebuild = True) == sync_job
    folder_sync.reconcile_folder(sync_job)

    with _connection() as conn:
        successor = conn.execute(
            "SELECT * FROM linked_folder_sync_jobs WHERE folder_id=? AND status='pending'",
            (folder["id"],),
        ).fetchone()
        assert successor is not None
        assert successor["id"] != sync_job
        assert successor["kind"] == "rebuild"


@requires_sqlite_vec
def test_requested_rebuild_promotes_an_intervening_pending_sync(rag_home):
    _, folder = _folder(rag_home)
    completed = folder_sync.request_sync(folder["id"])
    with _connection() as conn:
        conn.execute(
            "UPDATE linked_folder_sync_jobs SET status='completed', successor_kind='rebuild' "
            "WHERE id=?",
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

    folder_sync._queue_successor(completed)
    with _connection() as conn:
        successor = conn.execute(
            "SELECT kind, successor_kind FROM linked_folder_sync_jobs WHERE id=?", (pending,)
        ).fetchone()
        assert tuple(successor) == ("rebuild", None)
        assert (
            conn.execute(
                "SELECT successor_kind FROM linked_folder_sync_jobs WHERE id=?", (completed,)
            ).fetchone()["successor_kind"]
            is None
        )


@requires_sqlite_vec
def test_pending_rebuild_promotion_clears_a_recovered_successor_flag(rag_home):
    _, folder = _folder(rag_home)
    job_id = folder_sync.request_sync(folder["id"])
    with _connection() as conn:
        conn.execute(
            "UPDATE linked_folder_sync_jobs SET successor_kind='rebuild' WHERE id=?", (job_id,)
        )
        conn.commit()

    assert folder_sync.request_sync(folder["id"], rebuild = True) == job_id
    with _connection() as conn:
        job = conn.execute(
            "SELECT kind, successor_kind FROM linked_folder_sync_jobs WHERE id=?", (job_id,)
        ).fetchone()
        assert tuple(job) == ("rebuild", None)


@requires_sqlite_vec
def test_startup_recovers_terminal_rebuild_handoff(rag_home):
    _, folder = _folder(rag_home)
    completed = folder_sync.request_sync(folder["id"])
    with _connection() as conn:
        conn.execute(
            "UPDATE linked_folder_sync_jobs SET status='completed', successor_kind='rebuild', "
            "completed_at=created_at WHERE id=?",
            (completed,),
        )
        conn.commit()

    folder_sync._recover_startup_state()

    with _connection() as conn:
        successor = conn.execute(
            "SELECT * FROM linked_folder_sync_jobs WHERE folder_id=? AND status='pending'",
            (folder["id"],),
        ).fetchone()
        assert successor is not None
        assert successor["kind"] == "rebuild"
        assert (
            conn.execute(
                "SELECT successor_kind FROM linked_folder_sync_jobs WHERE id=?", (completed,)
            ).fetchone()["successor_kind"]
            is None
        )


@requires_sqlite_vec
def test_global_folder_list_resolves_scope_name(rag_home, monkeypatch):
    from routes.rag import list_linked_folders
    from storage import studio_db

    monkeypatch.setattr(studio_db, "list_chat_projects", lambda **kwargs: [])

    with _connection() as conn:
        store.create_kb(conn, name = "Knowledge One", kb_id = "kb1")
    source = rag_home / "source"
    source.mkdir()
    folder_sync.create_folder(scope_type = "knowledge_base", scope_id = "kb1", path = str(source))
    result = list_linked_folders(scope_type = None, scope_id = None, subject = "test")
    assert result["linkedFolders"][0]["scopeName"] == "Knowledge One"


@requires_sqlite_vec
def test_unexpected_reconcile_failure_resets_folder_status(rag_home, monkeypatch):
    _, folder = _folder(rag_home)
    job_id = folder_sync.request_sync(folder["id"])
    with _connection() as conn:
        conn.execute("UPDATE linked_folders SET status='syncing' WHERE id=?", (folder["id"],))
        conn.commit()
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
    with _connection() as conn:
        assert (
            conn.execute(
                "SELECT 1 FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchone()
            is None
        )


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


def test_start_auto_sync_launches_worker_after_transient_database_error(monkeypatch):
    created = []

    class FakeThread:
        def __init__(self, *args, **kwargs):
            self.started = False
            created.append(self)

        def start(self):
            self.started = True

        def is_alive(self):
            return self.started

    def unavailable_while_locked():
        raise sqlite3.OperationalError("database is locked")

    original_thread = folder_sync._thread
    original_thread_stop = folder_sync._thread_stop
    original_stop = folder_sync._stop.is_set()
    monkeypatch.setattr(folder_sync.rag_db, "rag_available", unavailable_while_locked)
    monkeypatch.setattr(folder_sync.threading, "Thread", FakeThread)
    folder_sync._thread = None
    folder_sync._thread_stop = None
    folder_sync._stop.clear()
    try:
        assert folder_sync.start_auto_sync() is True
        assert len(created) == 1
        assert created[0].started is True
    finally:
        if original_stop:
            folder_sync._stop.set()
        else:
            folder_sync._stop.clear()
        folder_sync._thread = original_thread
        folder_sync._thread_stop = original_thread_stop


def test_worker_reconciles_retired_scopes_before_scheduling(monkeypatch):
    stop = threading.Event()
    calls = []

    def project_exists(project_id):
        return project_id == "project"

    monkeypatch.setattr(folder_sync, "_recover_startup_state", lambda: calls.append("jobs"))
    monkeypatch.setattr(
        folder_sync,
        "reconcile_retired_scopes",
        lambda callback: calls.append(("scopes", callback is project_exists)),
    )

    def enqueue():
        calls.append("periodic")
        stop.set()

    monkeypatch.setattr(folder_sync, "_enqueue_periodic", enqueue)

    try:
        folder_sync._worker(stop, project_exists)
    finally:
        del folder_sync._worker_state.stop_event

    assert calls == ["jobs", ("scopes", True), "periodic"]


@requires_sqlite_vec
def test_unlink_does_not_wait_to_signal_a_locally_syncing_folder(rag_home):
    _, folder = _folder(rag_home)
    deleted = threading.Event()

    with folder_sync._folder_lock(folder["id"]):
        thread = threading.Thread(
            target = lambda: (folder_sync.delete_folder(folder["id"]), deleted.set())
        )
        thread.start()
        thread.join(timeout = 1)

    assert deleted.is_set()
    assert folder_sync.get_folder(folder["id"]) is None


@requires_sqlite_vec
def test_unlink_keeps_snapshot_cleanup_retryable(rag_home, stub_embeddings, monkeypatch):
    source, folder = _folder(rag_home)
    (source / "notes.txt").write_text("durable unlink", encoding = "utf-8")
    assert _run(folder["id"])["status"] == "completed"
    with _connection() as conn:
        document = conn.execute(
            "SELECT id, stored_path FROM documents WHERE linked_folder_id=?",
            (folder["id"],),
        ).fetchone()
    original_remove = folder_sync._remove_retired_snapshot
    monkeypatch.setattr(
        folder_sync,
        "_remove_retired_snapshot",
        lambda path: (_ for _ in ()).throw(OSError("snapshot is busy")),
    )

    with pytest.raises(OSError, match = "snapshot is busy"):
        folder_sync.delete_folder(folder["id"])

    assert folder_sync.get_folder(folder["id"])["status"] == "retired"
    with _connection() as conn:
        assert store.get_document(conn, document["id"]) is not None
    assert os.path.exists(document["stored_path"])

    monkeypatch.setattr(folder_sync, "_remove_retired_snapshot", original_remove)
    assert folder_sync.delete_folder(folder["id"]) is True
    assert folder_sync.get_folder(folder["id"]) is None
    assert not os.path.exists(document["stored_path"])


@requires_sqlite_vec
def test_unlink_waits_for_a_foreign_sync_lease(rag_home):
    from core.rag import job_leases

    _, folder = _folder(rag_home)
    job_id = folder_sync.request_sync(folder["id"])
    with _connection() as conn:
        conn.execute("UPDATE linked_folder_sync_jobs SET status='running' WHERE id=?", (job_id,))
        conn.execute(
            "INSERT INTO rag_job_leases(kind, job_id, owner_id, expires_at) "
            "VALUES(?,?,'foreign','9999-12-31')",
            (job_leases.FOLDER_SYNC, job_id),
        )
        conn.commit()

    deletion = threading.Thread(target = folder_sync.delete_folder, args = (folder["id"],))
    deletion.start()
    deadline = time.time() + 2
    while time.time() < deadline and folder_sync.get_folder(folder["id"])["status"] != "retired":
        time.sleep(0.01)
    assert deletion.is_alive()
    with _connection() as conn:
        conn.execute("UPDATE rag_job_leases SET expires_at='2000-01-01' WHERE job_id=?", (job_id,))
        conn.commit()
    deletion.join(2)
    assert not deletion.is_alive()
    assert folder_sync.get_folder(folder["id"]) is None


def test_project_delete_runs_before_best_effort_rag_cleanup(monkeypatch):
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

    def delete(*args, **kwargs):
        calls.append(("delete", threading.get_ident()))
        return project

    def cleanup(project_id):
        calls.append(("cleanup", threading.get_ident(), project_id))
        raise sqlite3.OperationalError("database is busy")

    monkeypatch.setattr(chat_history, "delete_chat_project", delete)
    monkeypatch.setattr(chat_history, "_delete_project_rag_sources", cleanup)

    event_loop_thread = threading.get_ident()
    result = asyncio.run(
        chat_history.delete_project("p1", SimpleNamespace(), current_subject = "test")
    )

    assert result.id == "p1"
    assert [call[0] for call in calls] == ["delete", "cleanup"]
    assert all(call[1] != event_loop_thread for call in calls)
    assert calls[1][2] == "p1"


def test_project_post_commit_file_failure_still_retires_rag(monkeypatch):
    from routes import chat_history

    project = {"id": "p1", "name": "Project", "createdAt": 1, "updatedAt": 1}
    owner_exists = True
    calls = []

    def get_project(project_id):
        return project if owner_exists else None

    def delete(*args, **kwargs):
        nonlocal owner_exists
        calls.append("delete")
        owner_exists = False
        raise OSError("workspace cleanup failed")

    def cleanup(project_id):
        calls.append(f"cleanup:{project_id}")

    monkeypatch.setattr(chat_history, "get_chat_project", get_project)
    monkeypatch.setattr(chat_history, "list_chat_threads", lambda **kwargs: [])
    monkeypatch.setattr(chat_history, "_cancel_active_research", lambda request, ids: None)
    monkeypatch.setattr(chat_history, "delete_chat_project", delete)
    monkeypatch.setattr(chat_history, "_delete_project_rag_sources", cleanup)

    with pytest.raises(OSError, match = "workspace cleanup failed"):
        asyncio.run(chat_history.delete_project("p1", SimpleNamespace(), current_subject = "test"))

    assert calls == ["delete", "cleanup:p1"]


@requires_sqlite_vec
def test_project_cleanup_persists_retirement_when_rag_is_unavailable(rag_home, monkeypatch):
    from routes import chat_history

    source = rag_home / "unavailable-project"
    source.mkdir()
    folder = folder_sync.create_folder(scope_type = "project", scope_id = "p1", path = str(source))
    monkeypatch.setattr(rag_db, "rag_available", lambda: False)

    chat_history._delete_project_rag_sources("p1")

    with _connection(metadata = True) as metadata:
        retired = metadata.execute(
            "SELECT purged_at FROM linked_folder_retired_scopes WHERE scope=?",
            (store.project_scope("p1"),),
        ).fetchone()
        persisted = metadata.execute(
            "SELECT auto_sync, status FROM linked_folders WHERE id=?", (folder["id"],)
        ).fetchone()
    assert retired is not None
    assert retired["purged_at"] is None
    assert dict(persisted) == {"auto_sync": 0, "status": "retired"}


@requires_sqlite_vec
def test_startup_retires_and_deletes_an_orphaned_project_scope(rag_home, stub_embeddings):
    scope = store.project_scope("deleted-project")
    source = rag_home / "deleted-project"
    source.mkdir()
    (source / "notes.txt").write_text("managed snapshot", encoding = "utf-8")
    folder = folder_sync.create_folder(
        scope_type = "project", scope_id = "deleted-project", path = str(source)
    )
    assert _run(folder["id"])["status"] == "completed"
    with _connection() as conn:
        document = conn.execute(
            "SELECT id, stored_path FROM documents WHERE scope=?", (scope,)
        ).fetchone()
    assert document is not None
    assert os.path.isfile(document["stored_path"])

    reconciled = folder_sync.reconcile_retired_scopes(lambda project_id: False)

    assert reconciled == {"retired": [scope], "deleted": [scope], "restored": []}
    assert folder_sync.get_folder(folder["id"]) is None
    assert folder_sync.scope_retired(scope) is True
    with _connection() as conn:
        assert conn.execute("SELECT 1 FROM documents WHERE scope=?", (scope,)).fetchone() is None
        tombstone = conn.execute(
            "SELECT purged_at FROM linked_folder_retired_scopes WHERE scope=?", (scope,)
        ).fetchone()
    assert tombstone["purged_at"] is not None
    assert not os.path.exists(document["stored_path"])


@requires_sqlite_vec
def test_retired_scope_waits_for_manual_ingestion_before_purging(
    rag_home, stub_embeddings, monkeypatch
):
    from core.rag import ingestion
    from utils.paths import ensure_dir, rag_uploads_root

    scope = store.project_scope("uploading-project")
    upload = ensure_dir(rag_uploads_root()) / "in-flight.txt"
    upload.write_text("durable in-flight words", encoding = "utf-8")
    parsing = threading.Event()
    release = threading.Event()
    original_parse = ingestion.parsers.parse

    def blocked_parse(path):
        parsing.set()
        assert release.wait(5)
        return original_parse(path)

    monkeypatch.setattr(ingestion.parsers, "parse", blocked_parse)
    document_id, job_id = ingestion.start_ingestion(
        scope, None, None, upload.name, str(upload), project_id = "uploading-project"
    )
    assert parsing.wait(5)
    folder_sync.retire_scope(scope)
    assert folder_sync.delete_retired_scope(scope) is False

    release.set()
    deadline = time.time() + 5
    while time.time() < deadline and ingestion.get_job_status(job_id)["status"] != "completed":
        time.sleep(0.05)
    while time.time() < deadline and not folder_sync.delete_retired_scope(scope):
        time.sleep(0.05)

    assert folder_sync.scope_retired(scope) is True
    with _connection() as conn:
        assert store.get_document(conn, document_id) is None
        assert conn.execute("SELECT 1 FROM ingestion_jobs WHERE id=?", (job_id,)).fetchone() is None
        tombstone = conn.execute(
            "SELECT purged_at FROM linked_folder_retired_scopes WHERE scope=?", (scope,)
        ).fetchone()
    assert tombstone["purged_at"] is not None
    assert not upload.exists()


@requires_sqlite_vec
def test_retired_scope_cleanup_keeps_retry_state_when_file_removal_fails(
    rag_home, stub_embeddings, monkeypatch
):
    scope = store.project_scope("retry-project")
    source = rag_home / "retry-project"
    source.mkdir()
    (source / "notes.txt").write_text("managed snapshot", encoding = "utf-8")
    folder = folder_sync.create_folder(
        scope_type = "project", scope_id = "retry-project", path = str(source)
    )
    assert _run(folder["id"])["status"] == "completed"
    folder_sync.retire_scope(scope)
    monkeypatch.setattr(
        folder_sync,
        "_remove_retired_snapshot",
        lambda path: (_ for _ in ()).throw(OSError("snapshot is busy")),
    )

    with pytest.raises(OSError, match = "snapshot is busy"):
        folder_sync.delete_retired_scope(scope)

    assert folder_sync.scope_retired(scope) is True
    assert folder_sync.get_folder(folder["id"])["status"] == "retired"
    with _connection() as conn:
        assert (
            conn.execute("SELECT 1 FROM documents WHERE scope=?", (scope,)).fetchone() is not None
        )
        tombstone = conn.execute(
            "SELECT purged_at FROM linked_folder_retired_scopes WHERE scope=?", (scope,)
        ).fetchone()
    assert tombstone["purged_at"] is None


@requires_sqlite_vec
def test_project_writer_contention_does_not_retire_a_live_project(rag_home, monkeypatch):
    from routes import chat_history
    from storage import studio_db

    project = studio_db.upsert_chat_project(
        {
            "id": "p1",
            "name": "Project",
            "createdAt": 1,
            "updatedAt": 1,
        }
    )
    source = rag_home / "locked-project-delete"
    source.mkdir()
    folder = folder_sync.create_folder(
        scope_type = "project", scope_id = project["id"], path = str(source)
    )
    blocker = studio_db.get_connection()
    original_get_connection = studio_db.get_connection

    def short_timeout_connection(*args, **kwargs):
        conn = original_get_connection()
        conn.execute("PRAGMA busy_timeout = 10")
        return conn

    blocker.execute("BEGIN IMMEDIATE")
    monkeypatch.setattr(studio_db, "get_connection", short_timeout_connection)
    try:
        with pytest.raises(sqlite3.OperationalError, match = "database is locked"):
            asyncio.run(
                chat_history.delete_project(
                    project["id"], SimpleNamespace(), current_subject = "test"
                )
            )
        assert folder_sync.scope_retired(store.project_scope(project["id"])) is False
        assert folder_sync.get_folder(folder["id"])["status"] == folder["status"]
    finally:
        blocker.rollback()
        blocker.close()


def test_project_upload_cleans_saved_file_when_scope_retires_after_save(rag_home, monkeypatch):
    from routes import rag as rag_routes
    from storage import studio_db
    from utils.paths import ensure_dir, rag_uploads_root

    project_id = "project"
    scope = store.project_scope(project_id)
    saved_path = ensure_dir(rag_uploads_root()) / "race.txt"
    retired = False

    def resolve_upload(*args, **kwargs):
        nonlocal retired
        saved_path.write_text("saved", encoding = "utf-8")
        retired = True
        return str(saved_path), "race.txt"

    monkeypatch.setattr(rag_routes.rag_db, "rag_available", lambda: True)
    monkeypatch.setattr(studio_db, "get_chat_project", lambda value: {"id": value})
    monkeypatch.setattr(
        rag_routes.folder_sync,
        "scope_retired",
        lambda value: value == scope and retired,
    )
    monkeypatch.setattr(rag_routes, "_resolve_document_upload", resolve_upload)
    monkeypatch.setattr(
        rag_routes.ingestion,
        "start_ingestion",
        lambda *args, **kwargs: pytest.fail("retired project upload must not ingest"),
    )

    with pytest.raises(Exception) as exc_info:
        asyncio.run(rag_routes.upload_project_document(project_id, subject = "test"))

    assert getattr(exc_info.value, "status_code", None) == 409
    assert not saved_path.exists()


@requires_sqlite_vec
@pytest.mark.parametrize("scope_type", ["knowledge_base", "project"])
def test_upload_rechecks_owner_after_saving_file(rag_home, monkeypatch, scope_type):
    from routes import rag as rag_routes
    from storage import studio_db
    from utils.paths import ensure_dir, rag_uploads_root

    owner_id = "owner"
    owner_exists = True
    saved_path = ensure_dir(rag_uploads_root()) / f"missing-{scope_type}.txt"

    def resolve_upload(*args, **kwargs):
        nonlocal owner_exists
        saved_path.write_text("saved", encoding = "utf-8")
        owner_exists = False
        return str(saved_path), saved_path.name

    monkeypatch.setattr(rag_routes.rag_db, "rag_available", lambda: True)
    monkeypatch.setattr(rag_routes.folder_sync, "scope_retired", lambda scope: False)
    monkeypatch.setattr(rag_routes, "_resolve_document_upload", resolve_upload)
    monkeypatch.setattr(
        rag_routes.ingestion,
        "start_ingestion",
        lambda *args, **kwargs: pytest.fail("an ownerless upload must not ingest"),
    )
    if scope_type == "knowledge_base":
        monkeypatch.setattr(
            rag_routes.store,
            "get_kb",
            lambda conn, value: {"id": value} if owner_exists else None,
        )
        upload = rag_routes.upload_kb_document(owner_id, subject = "test")
    else:
        monkeypatch.setattr(
            studio_db,
            "get_chat_project",
            lambda value: {"id": value} if owner_exists else None,
        )
        upload = rag_routes.upload_project_document(owner_id, subject = "test")

    with pytest.raises(Exception) as exc_info:
        asyncio.run(upload)

    assert getattr(exc_info.value, "status_code", None) == 404
    assert not saved_path.exists()


@pytest.mark.parametrize("scope_type", ["knowledge_base", "project"])
def test_linked_folder_rechecks_owner_after_resolving_lease(rag_home, monkeypatch, scope_type):
    from fastapi import HTTPException
    from routes import rag as rag_routes

    owner_exists = True

    def resolve_path(lease):
        nonlocal owner_exists
        owner_exists = False
        return str(rag_home), (rag_home.stat().st_dev, rag_home.stat().st_ino)

    def require_owner(kind, owner_id):
        if not owner_exists:
            raise HTTPException(status_code = 404, detail = "Owner not found")

    monkeypatch.setattr(rag_routes, "_resolve_linked_folder_path", resolve_path)
    monkeypatch.setattr(rag_routes, "_require_scope_owner", require_owner)
    monkeypatch.setattr(
        rag_routes.folder_sync,
        "create_folder_with_sync",
        lambda **kwargs: pytest.fail("an ownerless folder must not be linked"),
    )

    with pytest.raises(HTTPException) as exc_info:
        rag_routes._create_linked_folder(
            scope_type,
            "owner",
            rag_routes.LinkFolderRequest(nativePathLease = "lease"),
        )

    assert exc_info.value.status_code == 404


@requires_sqlite_vec
def test_project_rag_cleanup_atomically_removes_retired_scope(rag_home):
    from routes import chat_history

    scope = store.project_scope("project")
    folders = []
    for name in ("first", "second"):
        source = rag_home / name
        source.mkdir()
        folders.append(
            folder_sync.create_folder(scope_type = "project", scope_id = "project", path = str(source))
        )
    chat_history._delete_project_rag_sources("project")

    assert folder_sync.list_folders(scope) == []
    assert folder_sync.scope_retired(scope) is True
    replacement = rag_home / "replacement"
    replacement.mkdir()
    with pytest.raises(ValueError, match = "no longer exists"):
        folder_sync.create_folder(
            scope_type = "project",
            scope_id = "project",
            path = str(replacement),
        )


@requires_sqlite_vec
def test_kb_deletion_retries_retired_scope_cleanup_after_failure(
    rag_home, stub_embeddings, monkeypatch
):
    from routes import rag as rag_routes

    with _connection() as conn:
        store.create_kb(conn, name = "Knowledge", kb_id = "knowledge")
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
    with _connection() as conn:
        document = conn.execute(
            "SELECT id, stored_path FROM documents WHERE linked_folder_id=?", (folders[0]["id"],)
        ).fetchone()
        stored_path = document["stored_path"]
    assert os.path.isfile(stored_path)
    original_cleanup = folder_sync.delete_retired_scope
    monkeypatch.setattr(
        folder_sync,
        "delete_retired_scope",
        lambda scope, **kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("database is busy")),
    )
    assert rag_routes.delete_knowledge_base("knowledge", subject = "test") == {"ok": True}

    remaining = folder_sync.list_folders(store.kb_scope("knowledge"))
    assert {folder["id"] for folder in remaining} == {folder["id"] for folder in folders}
    assert {folder["status"] for folder in remaining} == {"retired"}
    assert folder_sync.scope_retired(store.kb_scope("knowledge")) is True
    assert os.path.exists(stored_path)
    assert rag_routes.list_kb_documents("knowledge", subject = "test") == {"documents": []}
    assert rag_routes.list_all_uploaded_documents(subject = "test") == {"documents": []}
    with pytest.raises(Exception) as exc_info:
        rag_routes.search(
            rag_routes.SearchRequest(query = "managed", kb_id = "knowledge", mode = "lexical"),
            subject = "test",
        )
    assert getattr(exc_info.value, "status_code", None) == 404
    with pytest.raises(Exception) as exc_info:
        rag_routes.preview_target(document["id"], subject = "test")
    assert getattr(exc_info.value, "status_code", None) == 404
    with _connection() as conn:
        assert (
            conn.execute(
                "SELECT 1 FROM documents WHERE scope=?", (store.kb_scope("knowledge"),)
            ).fetchone()
            is not None
        )
        assert store.all_chunks_for_scope(conn, store.kb_scope("knowledge")) == []
        assert store.scope_token_estimate(conn, store.kb_scope("knowledge")) == 0
    monkeypatch.setattr(folder_sync, "delete_retired_scope", original_cleanup)

    reconciled = folder_sync.reconcile_retired_scopes(lambda project_id: False)

    assert reconciled == {
        "retired": [],
        "deleted": [store.kb_scope("knowledge")],
        "restored": [],
    }
    assert folder_sync.list_folders(store.kb_scope("knowledge")) == []
    assert folder_sync.scope_retired(store.kb_scope("knowledge")) is True
    assert not os.path.exists(stored_path)


@requires_sqlite_vec
def test_kb_deletion_rolls_back_scope_before_any_folder_cleanup_on_failure(rag_home, monkeypatch):
    from routes import rag as rag_routes

    with _connection() as conn:
        store.create_kb(conn, name = "Knowledge", kb_id = "knowledge")
    source = rag_home / "failed-kb-delete"
    source.mkdir()
    folder = folder_sync.create_folder(
        scope_type = "knowledge_base",
        scope_id = "knowledge",
        path = str(source),
        auto_sync = False,
    )
    cleaned = []
    monkeypatch.setattr(
        rag_routes.store,
        "delete_kb",
        lambda *args, **kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("database is busy")),
    )
    monkeypatch.setattr(
        rag_routes.folder_sync,
        "delete_folder",
        lambda folder_id: cleaned.append(folder_id),
    )

    with pytest.raises(sqlite3.OperationalError, match = "database is busy"):
        rag_routes.delete_knowledge_base("knowledge", subject = "test")

    assert cleaned == []
    assert folder_sync.scope_retired(store.kb_scope("knowledge")) is False
    restored = folder_sync.get_folder(folder["id"])
    assert restored["auto_sync"] == folder["auto_sync"]
    assert restored["status"] == folder["status"]
    with _connection() as conn:
        assert store.get_kb(conn, "knowledge") is not None


@requires_sqlite_vec
def test_kb_writer_contention_cannot_commit_retirement_without_deletion(rag_home, monkeypatch):
    from routes import rag as rag_routes

    with _connection() as conn:
        store.create_kb(conn, name = "Knowledge", kb_id = "knowledge")
    source = rag_home / "locked-kb-delete"
    source.mkdir()
    folder = folder_sync.create_folder(
        scope_type = "knowledge_base", scope_id = "knowledge", path = str(source)
    )
    blocker = rag_db.get_metadata_connection()
    original_get_connection = rag_db.get_connection

    def short_timeout_connection(*args, **kwargs):
        conn = original_get_connection()
        conn.execute("PRAGMA busy_timeout = 10")
        return conn

    blocker.execute("BEGIN IMMEDIATE")
    monkeypatch.setattr(rag_db, "get_connection", short_timeout_connection)
    try:
        with pytest.raises(sqlite3.OperationalError, match = "database is locked"):
            rag_routes.delete_knowledge_base("knowledge", subject = "test")
        assert (
            blocker.execute("SELECT 1 FROM knowledge_bases WHERE id='knowledge'").fetchone()
            is not None
        )
        assert (
            blocker.execute(
                "SELECT 1 FROM linked_folder_retired_scopes WHERE scope=?",
                (store.kb_scope("knowledge"),),
            ).fetchone()
            is None
        )
        persisted = blocker.execute(
            "SELECT auto_sync, status FROM linked_folders WHERE id=?", (folder["id"],)
        ).fetchone()
        assert dict(persisted) == {"auto_sync": folder["auto_sync"], "status": folder["status"]}
    finally:
        blocker.rollback()
        blocker.close()


@requires_sqlite_vec
def test_kb_upload_rejects_retired_scope_before_saving(rag_home, monkeypatch):
    from routes import rag as rag_routes

    with _connection() as conn:
        store.create_kb(conn, name = "Knowledge", kb_id = "knowledge")
    folder_sync.retire_scope(store.kb_scope("knowledge"))
    monkeypatch.setattr(
        rag_routes,
        "_resolve_document_upload",
        lambda *args, **kwargs: pytest.fail("retired KB upload must be rejected before saving"),
    )

    with pytest.raises(Exception) as exc_info:
        asyncio.run(rag_routes.upload_kb_document("knowledge", subject = "test"))
    assert getattr(exc_info.value, "status_code", None) == 409


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


@requires_sqlite_vec
def test_unrelated_ingest_failure_still_removes_deleted_sources(
    rag_home, stub_embeddings, monkeypatch
):
    source, folder = _folder(rag_home)
    (source / "keep.txt").write_text("durable keeper", encoding = "utf-8")
    (source / "doomed.txt").write_text("removable words", encoding = "utf-8")
    assert _run(folder["id"])["status"] == "completed"

    (source / "doomed.txt").unlink()
    (source / "poison.txt").write_text("never indexes", encoding = "utf-8")
    real_start = folder_sync.ingestion.start_ingestion
    monkeypatch.setattr(
        folder_sync.ingestion,
        "start_ingestion",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("embed unavailable"))
        if args[3] == "poison.txt"
        else real_start(*args, **kwargs),
    )

    # each vanished path gets one grace pass before it is removed anyway
    assert _run(folder["id"])["deleted"] == 0
    result = _run(folder["id"])

    assert result["status"] == "failed"
    assert result["deleted"] == 1
    assert "poison.txt" in result["error"]
    with _connection() as conn:
        paths = {
            row["relative_path"]
            for row in conn.execute(
                "SELECT relative_path FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            )
        }
        assert paths == {"keep.txt"}
        assert not store.search_lexical(conn, folder["scope"], "removable", 5)


@requires_sqlite_vec
def test_sync_requested_during_a_running_sync_queues_a_successor(rag_home):
    _, folder = _folder(rag_home)
    running = folder_sync.request_sync(folder["id"])
    with _connection() as conn:
        conn.execute("UPDATE linked_folder_sync_jobs SET status='running' WHERE id=?", (running,))
        conn.commit()

    assert folder_sync.request_sync(folder["id"]) == running
    folder_sync.reconcile_folder(running)

    successor = _row(
        "SELECT id, kind FROM linked_folder_sync_jobs WHERE folder_id=? AND status='pending'",
        (folder["id"],),
    )
    assert successor is not None
    assert successor["id"] != running
    assert successor["kind"] == "sync"


@requires_sqlite_vec
def test_a_queued_rebuild_is_not_downgraded_by_a_later_sync_request(rag_home):
    _, folder = _folder(rag_home)
    running = folder_sync.request_sync(folder["id"])
    with _connection() as conn:
        conn.execute("UPDATE linked_folder_sync_jobs SET status='running' WHERE id=?", (running,))
        conn.commit()

    folder_sync.request_sync(folder["id"], rebuild = True)
    folder_sync.request_sync(folder["id"])

    assert (
        _row("SELECT successor_kind FROM linked_folder_sync_jobs WHERE id=?", (running,))[
            "successor_kind"
        ]
        == "rebuild"
    )


def test_failure_summary_names_the_files_and_caps_the_list():
    assert folder_sync._failure_summary([]) is None
    assert folder_sync._failure_summary(["b.txt", "a.txt"]) == (
        "2 file(s) could not be indexed (a.txt, b.txt)"
    )
    summary = folder_sync._failure_summary([f"f{index}.txt" for index in range(6)])
    assert summary.startswith("6 file(s) could not be indexed (f0.txt, f1.txt, f2.txt and 3 more)")


@requires_sqlite_vec
def test_a_rewritten_rename_retains_the_prior_document_until_it_reindexes(
    rag_home, stub_embeddings, monkeypatch
):
    source, folder = _folder(rag_home)
    original = source / "report.txt"
    original.write_text("durable travelling words", encoding = "utf-8")
    assert _run(folder["id"])["status"] == "completed"

    # an atomic re-save after a rename shares neither inode nor content with the original
    renamed = source / "report-final.txt"
    renamed.write_text("rewritten content that fails", encoding = "utf-8")
    original.unlink()
    monkeypatch.setattr(
        folder_sync.ingestion,
        "start_ingestion",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("embed unavailable")),
    )

    result = _run(folder["id"])

    assert result["status"] == "failed"
    assert result["deleted"] == 0
    with _connection() as conn:
        assert store.search_lexical(conn, folder["scope"], "travelling", 5)


@requires_sqlite_vec
def test_an_unreadable_file_stops_withholding_removals_after_one_pass(
    rag_home, stub_embeddings, monkeypatch
):
    source, folder = _folder(rag_home)
    (source / "keep.txt").write_text("durable keeper", encoding = "utf-8")
    (source / "doomed.txt").write_text("removable words", encoding = "utf-8")
    assert _run(folder["id"])["status"] == "completed"

    (source / "doomed.txt").unlink()
    (source / "unreadable.txt").write_text("cannot be copied", encoding = "utf-8")
    monkeypatch.setattr(
        folder_sync,
        "_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("source unreadable")),
    )

    assert _run(folder["id"])["deleted"] == 0
    assert _run(folder["id"])["deleted"] == 1
    with _connection() as conn:
        assert not store.search_lexical(conn, folder["scope"], "removable", 5)


@requires_sqlite_vec
def test_a_failing_file_that_keeps_changing_cannot_block_removals(
    rag_home, stub_embeddings, monkeypatch
):
    source, folder = _folder(rag_home)
    (source / "keep.txt").write_text("durable keeper", encoding = "utf-8")
    (source / "doomed.txt").write_text("removable words", encoding = "utf-8")
    assert _run(folder["id"])["status"] == "completed"

    (source / "doomed.txt").unlink()
    churn = source / "churn.txt"
    monkeypatch.setattr(
        folder_sync.ingestion,
        "start_ingestion",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("embed unavailable")),
    )
    churn.write_text("attempt one", encoding = "utf-8")
    assert _run(folder["id"])["deleted"] == 0
    churn.write_text("attempt two, a different size entirely", encoding = "utf-8")

    assert _run(folder["id"])["deleted"] == 1
    with _connection() as conn:
        assert not store.search_lexical(conn, folder["scope"], "removable", 5)


@requires_sqlite_vec
def test_job_events_emits_a_keepalive_while_a_job_is_quiet(rag_home, monkeypatch):
    _, folder = _folder(rag_home)
    job_id = folder_sync.request_sync(folder["id"])
    monkeypatch.setattr(folder_sync, "_JOB_EVENT_KEEPALIVE_S", 0.0)
    monkeypatch.setattr(folder_sync.time, "sleep", lambda _seconds: None)

    events = []
    for event in folder_sync.job_events(job_id):
        events.append(event)
        if len(events) == 3:
            break

    assert events[0] is not None
    assert events[1:] == [None, None]


@requires_sqlite_vec
def test_a_project_recreated_during_delete_keeps_its_rag_scope(rag_home, monkeypatch):
    from routes import chat_history

    scope = store.project_scope("p1")
    # the row delete has committed and another client has already created the id again
    monkeypatch.setattr(chat_history, "get_chat_project", lambda project_id: {"id": project_id})

    chat_history._delete_project_rag_sources("p1")

    assert folder_sync.scope_retired(scope) is False


@requires_sqlite_vec
def test_reconciliation_restores_a_scope_whose_project_came_back(rag_home):
    scope = store.project_scope("p1")
    source = rag_home / "recreated-project"
    source.mkdir()
    folder_sync.create_folder(scope_type = "project", scope_id = "p1", path = str(source))
    folder_sync.retire_scope(scope)
    assert folder_sync.scope_retired(scope) is True

    result = folder_sync.reconcile_retired_scopes(lambda project_id: project_id == "p1")

    assert result["restored"] == [scope]
    assert result["deleted"] == []
    assert folder_sync.scope_retired(scope) is False


@requires_sqlite_vec
def test_periodic_retirement_checks_ownership_under_the_scope_lock(rag_home):
    scope = store.project_scope("p1")
    source = rag_home / "recreated-under-lock"
    source.mkdir()
    folder = folder_sync.create_folder(scope_type = "project", scope_id = "p1", path = str(source))
    held = []

    def project_exists(project_id):
        # create_folder and upload admission take this lock, so holding it here is what
        # stops a project recreated mid-pass from having its new folders retired
        lock = folder_sync._scope_lock(scope)
        acquired = []
        probe = threading.Thread(target = lambda: acquired.append(lock.acquire(blocking = False)))
        probe.start()
        probe.join()
        held.append(not acquired[0])
        return True

    result = folder_sync.reconcile_retired_scopes(project_exists)

    assert held and all(held)
    assert result["retired"] == []
    assert folder_sync.scope_retired(scope) is False
    current = folder_sync.get_folder(folder["id"])
    assert current["status"] == folder["status"]
    assert current["auto_sync"] == folder["auto_sync"]


@requires_sqlite_vec
def test_retirement_leaves_a_folder_linked_after_the_ownership_check(rag_home):
    scope = store.project_scope("p1")
    source = rag_home / "before-check"
    source.mkdir()
    existing = folder_sync.create_folder(scope_type = "project", scope_id = "p1", path = str(source))
    checked_at = folder_sync.now_iso()
    # a second backend process links this one after the check and before the write
    later = rag_home / "after-check"
    later.mkdir()
    fresh = folder_sync.create_folder(scope_type = "project", scope_id = "p1", path = str(later))

    folder_sync.retire_scope(scope, checked_at)

    assert folder_sync.get_folder(existing["id"])["status"] == "retired"
    survivor = folder_sync.get_folder(fresh["id"])
    assert survivor["status"] == fresh["status"]
    assert survivor["auto_sync"] == fresh["auto_sync"]
    assert survivor["last_error"] is None
