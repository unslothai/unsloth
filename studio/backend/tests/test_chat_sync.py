# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""
Sandbox tests for the chat persistence layer (PR #4637).

Tests the SQLite storage functions and verifies:
- Schema creation
- CRUD operations on threads and messages
- FK cascade on thread deletion
- ON CONFLICT behavior (idempotent create, proper upsert)
- IntegrityError on orphaned message insert
- Python 3.9+ compatibility (from __future__ import annotations)
- Cross-platform path handling (Linux, macOS, Windows)
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import time

# Ensure the backend package root is on sys.path
_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

# Patch studio_db_path before importing storage module so tests use a temp DB
_tmpdir = tempfile.mkdtemp(prefix="chat_sync_test_")
_test_db = os.path.join(_tmpdir, "test_studio.db")


def _patched_studio_db_path():
    from pathlib import Path
    return Path(_test_db)


import utils.paths as _paths_mod
_paths_mod.studio_db_path = _patched_studio_db_path

# Reset schema state so each import gets a fresh schema
import storage.studio_db as sdb
sdb._schema_ready = False


def reset_db():
    """Delete and recreate the test DB."""
    if os.path.exists(_test_db):
        os.remove(_test_db)
    wal = _test_db + "-wal"
    shm = _test_db + "-shm"
    if os.path.exists(wal):
        os.remove(wal)
    if os.path.exists(shm):
        os.remove(shm)
    sdb._schema_ready = False


def test_schema_creation():
    """Tables and indexes are created on first connection."""
    reset_db()
    conn = sdb.get_connection()
    try:
        tables = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        assert "chat_threads" in tables, f"chat_threads not found in {tables}"
        assert "chat_messages" in tables, f"chat_messages not found in {tables}"
        assert "training_runs" in tables, f"training_runs not found in {tables}"

        # Verify FK pragma is ON
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1, f"foreign_keys should be 1, got {fk}"

        # Verify WAL mode
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal", f"journal_mode should be wal, got {mode}"
    finally:
        conn.close()
    print("  PASS: test_schema_creation")


def test_thread_crud():
    """Create, list, update, and delete a thread."""
    reset_db()
    now = int(time.time() * 1000)

    # Create
    sdb.create_chat_thread(
        thread_id="t1", title="Test Thread", model_type="base",
        model_id="llama-3", pair_id=None, created_at=now,
    )

    # List
    threads = sdb.list_chat_threads()
    assert len(threads) == 1, f"Expected 1 thread, got {len(threads)}"
    assert threads[0]["id"] == "t1"
    assert threads[0]["title"] == "Test Thread"
    assert threads[0]["archived"] == 0

    # List with filter
    threads_filtered = sdb.list_chat_threads(model_type="lora")
    assert len(threads_filtered) == 0

    # Update
    ok = sdb.update_chat_thread("t1", title="Updated Title")
    assert ok is True
    threads = sdb.list_chat_threads()
    assert threads[0]["title"] == "Updated Title"

    # Update non-existent
    ok = sdb.update_chat_thread("nonexistent", title="X")
    assert ok is False

    # Archive
    ok = sdb.update_chat_thread("t1", archived=1)
    assert ok is True
    threads = sdb.list_chat_threads()
    assert threads[0]["archived"] == 1

    # Delete
    ok = sdb.delete_chat_thread("t1")
    assert ok is True
    threads = sdb.list_chat_threads()
    assert len(threads) == 0

    # Delete non-existent
    ok = sdb.delete_chat_thread("t1")
    assert ok is False

    print("  PASS: test_thread_crud")


def test_message_upsert():
    """Create and update a message via upsert."""
    reset_db()
    now = int(time.time() * 1000)

    sdb.create_chat_thread(
        thread_id="t1", title="Thread", model_type="base",
        model_id="", pair_id=None, created_at=now,
    )

    # Insert message
    sdb.upsert_chat_message(
        message_id="m1", thread_id="t1", role="user",
        content='[{"type":"text","text":"Hello"}]',
        attachments=None, metadata=None, created_at=now,
    )

    msgs = sdb.get_chat_thread_messages("t1")
    assert len(msgs) == 1
    assert msgs[0]["content"] == '[{"type":"text","text":"Hello"}]'

    # Upsert (update content)
    sdb.upsert_chat_message(
        message_id="m1", thread_id="t1", role="user",
        content='[{"type":"text","text":"Hello updated"}]',
        attachments=None, metadata='{"key":"val"}', created_at=now,
    )

    msgs = sdb.get_chat_thread_messages("t1")
    assert len(msgs) == 1, f"Expected 1 message after upsert, got {len(msgs)}"
    assert msgs[0]["content"] == '[{"type":"text","text":"Hello updated"}]'
    assert msgs[0]["metadata"] == '{"key":"val"}'

    print("  PASS: test_message_upsert")


def test_fk_cascade_on_thread_delete():
    """Deleting a thread should CASCADE-delete its messages."""
    reset_db()
    now = int(time.time() * 1000)

    sdb.create_chat_thread(
        thread_id="t1", title="Thread", model_type="base",
        model_id="", pair_id=None, created_at=now,
    )
    sdb.upsert_chat_message(
        message_id="m1", thread_id="t1", role="user",
        content="[]", attachments=None, metadata=None, created_at=now,
    )
    sdb.upsert_chat_message(
        message_id="m2", thread_id="t1", role="assistant",
        content="[]", attachments=None, metadata=None, created_at=now + 1,
    )

    # Verify messages exist
    msgs = sdb.get_chat_thread_messages("t1")
    assert len(msgs) == 2

    # Delete thread
    sdb.delete_chat_thread("t1")

    # Messages should be gone (CASCADE)
    conn = sdb.get_connection()
    try:
        count = conn.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0]
        assert count == 0, f"Expected 0 messages after cascade delete, got {count}"
    finally:
        conn.close()

    print("  PASS: test_fk_cascade_on_thread_delete")


def test_orphan_message_integrity_error():
    """Inserting a message for a non-existent thread should raise IntegrityError."""
    reset_db()
    now = int(time.time() * 1000)

    try:
        sdb.upsert_chat_message(
            message_id="m1", thread_id="nonexistent", role="user",
            content="[]", attachments=None, metadata=None, created_at=now,
        )
        assert False, "Expected IntegrityError for orphan message"
    except sqlite3.IntegrityError:
        pass  # Expected

    print("  PASS: test_orphan_message_integrity_error")


def test_idempotent_thread_create():
    """Creating the same thread twice should be idempotent (no error, no update)."""
    reset_db()
    now = int(time.time() * 1000)

    sdb.create_chat_thread(
        thread_id="t1", title="Original", model_type="base",
        model_id="", pair_id=None, created_at=now,
    )
    # Second create with different title -- should be ignored
    sdb.create_chat_thread(
        thread_id="t1", title="Duplicate", model_type="base",
        model_id="", pair_id=None, created_at=now + 1000,
    )

    threads = sdb.list_chat_threads()
    assert len(threads) == 1
    assert threads[0]["title"] == "Original", "Duplicate create should not overwrite"

    print("  PASS: test_idempotent_thread_create")


def test_hydrate_all():
    """get_all_chat_data returns all threads and messages."""
    reset_db()
    now = int(time.time() * 1000)

    sdb.create_chat_thread(
        thread_id="t1", title="Thread 1", model_type="base",
        model_id="", pair_id=None, created_at=now,
    )
    sdb.create_chat_thread(
        thread_id="t2", title="Thread 2", model_type="lora",
        model_id="adapter-1", pair_id="p1", created_at=now + 1,
    )
    sdb.upsert_chat_message(
        message_id="m1", thread_id="t1", role="user",
        content='[{"type":"text","text":"msg1"}]',
        attachments=None, metadata=None, created_at=now,
    )
    sdb.upsert_chat_message(
        message_id="m2", thread_id="t2", role="assistant",
        content='[{"type":"text","text":"msg2"}]',
        attachments=None, metadata='{"timing":{"total":1.5}}', created_at=now + 2,
    )

    data = sdb.get_all_chat_data()
    assert len(data["threads"]) == 2
    assert len(data["messages"]) == 2
    # Threads ordered DESC by created_at
    assert data["threads"][0]["id"] == "t2"
    assert data["threads"][1]["id"] == "t1"
    # Messages ordered ASC by created_at
    assert data["messages"][0]["id"] == "m1"
    assert data["messages"][1]["id"] == "m2"

    print("  PASS: test_hydrate_all")


def test_update_only_allowed_fields():
    """update_chat_thread should only allow title and archived fields."""
    reset_db()
    now = int(time.time() * 1000)

    sdb.create_chat_thread(
        thread_id="t1", title="Thread", model_type="base",
        model_id="", pair_id=None, created_at=now,
    )

    # Try to inject a non-allowed field
    ok = sdb.update_chat_thread("t1", model_type="hacked")
    assert ok is False, "Non-allowed field should be silently dropped, returning False"

    threads = sdb.list_chat_threads()
    assert threads[0]["model_type"] == "base", "model_type should not be changed"

    print("  PASS: test_update_only_allowed_fields")


def test_on_conflict_do_update_preserves_row():
    """Verify ON CONFLICT DO UPDATE does not delete+reinsert (no CASCADE trigger)."""
    reset_db()
    now = int(time.time() * 1000)

    sdb.create_chat_thread(
        thread_id="t1", title="Thread", model_type="base",
        model_id="", pair_id=None, created_at=now,
    )
    sdb.upsert_chat_message(
        message_id="m1", thread_id="t1", role="user",
        content='["v1"]', attachments=None, metadata=None, created_at=now,
    )

    # Get rowid before upsert
    conn = sdb.get_connection()
    try:
        rowid_before = conn.execute(
            "SELECT rowid FROM chat_messages WHERE id = 'm1'"
        ).fetchone()[0]
    finally:
        conn.close()

    # Upsert same message with new content
    sdb.upsert_chat_message(
        message_id="m1", thread_id="t1", role="user",
        content='["v2"]', attachments=None, metadata=None, created_at=now,
    )

    # Rowid should be preserved (ON CONFLICT DO UPDATE, not INSERT OR REPLACE)
    conn = sdb.get_connection()
    try:
        rowid_after = conn.execute(
            "SELECT rowid FROM chat_messages WHERE id = 'm1'"
        ).fetchone()[0]
    finally:
        conn.close()

    assert rowid_before == rowid_after, (
        f"rowid changed from {rowid_before} to {rowid_after} -- "
        "ON CONFLICT DO UPDATE should preserve the row, not delete+reinsert"
    )

    print("  PASS: test_on_conflict_do_update_preserves_row")


def main():
    print("Running chat sync tests...")
    test_schema_creation()
    test_thread_crud()
    test_message_upsert()
    test_fk_cascade_on_thread_delete()
    test_orphan_message_integrity_error()
    test_idempotent_thread_create()
    test_hydrate_all()
    test_update_only_allowed_fields()
    test_on_conflict_do_update_preserves_row()
    print(f"\nAll 9 tests passed!")

    # Cleanup
    import shutil
    shutil.rmtree(_tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
