# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""rag_db.reuse_connections(): one connection per thread inside a block, with every caller's commit
and close keeping its meaning, and linked-folder sync using it without changing what is durable."""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import closing

import pytest

from storage import rag_db, sqlite_gate
from utils.paths import rag_db_path

pytestmark = pytest.mark.skipif(
    not rag_db.RAG_AVAILABLE or not hasattr(sqlite3.Connection, "enable_load_extension"),
    reason = "sqlite-vec cannot be loaded by this Python's sqlite3",
)


def _make_table() -> None:
    with closing(rag_db.get_connection()) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS reuse_probe(k TEXT PRIMARY KEY, v INTEGER)")
        conn.commit()


def _independent_rows() -> list[tuple]:
    conn = sqlite3.connect(str(rag_db_path()))
    try:
        return conn.execute("SELECT k, v FROM reuse_probe ORDER BY k").fetchall()
    finally:
        conn.close()


def _closed(conn) -> bool:
    try:
        conn.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        return True
    return False


def test_outside_a_block_every_open_is_new_and_close_closes(rag_home):
    first = rag_db.get_connection()
    second = rag_db.get_connection()
    assert first is not second
    assert type(first) is sqlite_gate.GatedConnection
    first.close()
    second.close()
    assert _closed(first) and _closed(second)


def test_block_hands_back_one_connection_and_close_discards_uncommitted(rag_home):
    _make_table()
    with rag_db.reuse_connections():
        conn = rag_db.get_connection()
        conn.execute("INSERT INTO reuse_probe VALUES('kept', 1)")
        conn.commit()
        conn.execute("INSERT INTO reuse_probe VALUES('dropped', 2)")
        assert conn.in_transaction
        conn.close()
        again = rag_db.get_connection()
        assert again is conn
        assert not again.in_transaction
        assert [tuple(row) for row in again.execute("SELECT k FROM reuse_probe ORDER BY k")] == [
            ("kept",)
        ]
        again.close()
        assert _independent_rows() == [("kept", 1)]
    assert _closed(conn)


def test_an_open_while_the_shared_one_is_held_gets_its_own(rag_home):
    _make_table()
    with rag_db.reuse_connections():
        outer = rag_db.get_connection()
        outer.execute("INSERT INTO reuse_probe VALUES('outer', 1)")
        inner = rag_db.get_connection()
        assert inner is not outer
        inner.close()
        assert _closed(inner)
        # The inner close must not end the outer caller's transaction.
        assert outer.in_transaction
        outer.commit()
        outer.close()
    assert _independent_rows() == [("outer", 1)]


def test_nested_blocks_share_and_only_the_outer_one_closes(rag_home):
    with rag_db.reuse_connections():
        conn = rag_db.get_connection()
        conn.close()
        with rag_db.reuse_connections():
            assert rag_db.get_connection() is conn
            conn.close()
        assert not _closed(conn)
    assert _closed(conn)


def test_checkout_restores_fresh_connection_settings(rag_home):
    with rag_db.reuse_connections():
        conn = rag_db.get_connection()
        conn.row_factory = None
        conn.execute("PRAGMA busy_timeout = 60000")
        conn.close()
        again = rag_db.get_connection()
        assert again is conn
        assert again.row_factory is sqlite3.Row
        assert again.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        again.close()


def test_a_dead_shared_connection_is_replaced(rag_home):
    with rag_db.reuse_connections():
        conn = rag_db.get_connection()
        conn.close()
        sqlite3.Connection.close(conn)
        fresh = rag_db.get_connection()
        assert fresh is not conn
        assert fresh.execute("SELECT 1").fetchone()[0] == 1
        fresh.close()


def test_another_database_path_is_not_shared(rag_home, monkeypatch):
    with rag_db.reuse_connections():
        conn = rag_db.get_connection()
        conn.close()
        other = rag_home / "other"
        other.mkdir()
        monkeypatch.setattr(rag_db, "rag_db_path", lambda: other / "rag.db")
        elsewhere = rag_db.get_connection()
        assert elsewhere is not conn
        elsewhere.close()
        assert _closed(elsewhere)


def test_blocks_are_per_thread(rag_home):
    seen = {}
    with rag_db.reuse_connections():
        mine = rag_db.get_connection()
        mine.close()

        def worker():
            conn = rag_db.get_connection()
            seen["other"] = conn
            conn.close()
            seen["closed"] = _closed(conn)

        thread = threading.Thread(target = worker)
        thread.start()
        thread.join()
        assert seen["other"] is not mine
        assert seen["closed"]


def test_an_idle_shared_connection_does_not_block_writers_or_checkpoints(rag_home):
    _make_table()
    with rag_db.reuse_connections():
        conn = rag_db.get_connection()
        conn.execute("SELECT k FROM reuse_probe").fetchall()
        conn.execute("INSERT INTO reuse_probe VALUES('a', 1)")
        conn.commit()
        conn.close()
        other = sqlite3.connect(str(rag_db_path()), timeout = 0)
        try:
            other.execute("BEGIN IMMEDIATE")
            other.execute("INSERT INTO reuse_probe VALUES('b', 2)")
            other.commit()
            busy, _, _ = other.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            assert busy == 0
        finally:
            other.close()


def test_retired_account_still_refuses_inside_a_block(rag_home, monkeypatch):
    from core.rag import account_db
    with rag_db.reuse_connections():
        account_db.get_connection().close()
        monkeypatch.setattr(account_db, "account_is_retired", lambda: True)
        with pytest.raises(RuntimeError, match = "Account is retired"):
            account_db.get_connection()


def _run(folder_id: str) -> dict:
    from core.rag import folder_sync

    job_id = folder_sync.request_sync(folder_id)
    folder_sync.reconcile_folder(job_id)
    return folder_sync.get_job(job_id)


def test_folder_sync_reuses_connections_and_commits_each_file(
    rag_home, stub_embeddings, monkeypatch
):
    from core.rag import folder_sync

    source = rag_home / "source"
    source.mkdir()
    for index in range(12):
        (source / f"f{index:02d}.txt").write_text(
            f"file {index} " + "words " * 30, encoding = "utf-8"
        )
    folder = folder_sync.create_folder(
        scope_type = "knowledge_base",
        scope_id = "scope-1",
        path = str(source),
        name = "Docs",
    )
    opens = []
    connect = sqlite_gate.connect

    def counting_connect(*args, **kwargs):
        opens.append(1)
        return connect(*args, **kwargs)

    monkeypatch.setattr(rag_db.sqlite_gate, "connect", counting_connect)
    assert _run(folder["id"])["status"] == "completed"
    attach_opens = len(opens)
    # About fourteen fresh opens per file before; three per file now (the worker's block and two in
    # finish_pending) plus the lease heartbeat and a fixed per-pass overhead.
    assert attach_opens < 12 * 8

    # Metadata-only changes: each file's row must be committed before the next file starts.
    for path in source.iterdir():
        os.utime(path, ns = (10**18, 10**18))
    durable = []
    update = folder_sync._update_mapping_metadata

    def observed_update(folder_id, rel, metadata, content_hash):
        update(folder_id, rel, metadata, content_hash)
        conn = sqlite3.connect(str(rag_db_path()))
        try:
            durable.append(
                conn.execute(
                    "SELECT mtime_ns FROM linked_folder_files WHERE folder_id=? AND relative_path=?",
                    (folder_id, rel),
                ).fetchone()[0]
            )
        finally:
            conn.close()

    monkeypatch.setattr(folder_sync, "_update_mapping_metadata", observed_update)
    opens.clear()
    assert _run(folder["id"])["status"] == "completed"
    assert durable == [10**18] * 12
    # Four fresh opens per file before (68 here); three now, since the stat and snapshot removal
    # between them hold no connection.
    assert len(opens) < 20 + 12 * 4 - 6


def test_release_reused_closes_only_an_idle_shared_connection(rag_home):
    rag_db.release_reused()  # outside a block: nothing to do
    with rag_db.reuse_connections():
        held = rag_db.get_connection()
        rag_db.release_reused()
        assert not _closed(held)
        held.close()
        rag_db.release_reused()
        assert _closed(held)
        fresh = rag_db.get_connection()
        assert fresh is not held
        fresh.close()
        assert rag_db.get_connection() is fresh
        fresh.close()
    assert _closed(fresh)


def test_a_stale_close_does_not_end_another_callers_work(rag_home):
    _make_table()
    with rag_db.reuse_connections():
        first = rag_db.get_connection()
        first.close()
        first.close()
        second = rag_db.get_connection()
        assert second is first
        second.close()
        rag_db.release_reused()
        third = rag_db.get_connection()
        assert third is not first
        third.execute("INSERT INTO reuse_probe VALUES('live', 1)")
        first.close()
        assert third.in_transaction
        third.commit()
        third.close()
    assert _independent_rows() == [("live", 1)]


def test_folder_sync_holds_no_handle_while_waiting_statting_or_copying(
    rag_home, stub_embeddings, monkeypatch
):
    from core.rag import folder_sync

    source = rag_home / "source"
    source.mkdir()
    for index in range(6):
        (source / f"f{index}.txt").write_text(f"file {index} " + "words " * 20, encoding = "utf-8")
    folder = folder_sync.create_folder(
        scope_type = "knowledge_base",
        scope_id = "scope-1",
        path = str(source),
        name = "Docs",
    )
    created: list[tuple[int, sqlite3.Connection]] = []
    connect = sqlite_gate.connect

    def tracking_connect(*args, **kwargs):
        conn = connect(*args, **kwargs)
        created.append((threading.get_ident(), conn))
        return conn

    def open_on_this_thread() -> int:
        keepers = set(map(id, rag_db._wal_keepers.values()))
        me = threading.get_ident()
        return sum(
            1
            for owner, conn in created
            if owner == me and id(conn) not in keepers and not _closed(conn)
        )

    # Probes sit inside the file work (after each helper's release), counting this thread's open handles.
    seen = {
        "wait": [],
        "reappeared": [],
        "snapshot_call": [],
        "snapshot": [],
        "identity": [],
        "uploads": [],
    }

    def probe(name, real):
        def wrapper(*args, **kwargs):
            seen[name].append(open_on_this_thread())
            return real(*args, **kwargs)

        monkeypatch.setattr(folder_sync, real.__name__, wrapper)

    probe("wait", folder_sync.wait)
    probe("reappeared", folder_sync.PurePosixPath)
    # A wrapper around _snapshot runs before the helper's own release, as a slow copy start would.
    probe("snapshot_call", folder_sync._snapshot)
    probe("snapshot", folder_sync._copy_exact)
    probe("identity", folder_sync._root_identity)
    probe("uploads", folder_sync.rag_uploads_root)
    monkeypatch.setattr(rag_db.sqlite_gate, "connect", tracking_connect)
    assert _run(folder["id"])["status"] == "completed"
    for path in source.iterdir():
        os.utime(path, ns = (10**18, 10**18))
    assert _run(folder["id"])["status"] == "completed"
    for index in range(3):
        (source / f"f{index}.txt").unlink()
    assert _run(folder["id"])["status"] == "completed"
    assert seen["wait"] and set(seen["wait"]) == {0}
    assert len(seen["reappeared"]) == 3 and set(seen["reappeared"]) == {0}
    assert len(seen["snapshot_call"]) == 12 and set(seen["snapshot_call"]) == {0}
    assert len(seen["snapshot"]) == 12 and set(seen["snapshot"]) == {0}
    assert len(seen["identity"]) >= 6 + 6 + 3 * 2 and set(seen["identity"]) == {0}
    # twelve snapshot targets, six touch-pass removals, three removals after _delete_mapping commits
    assert len(seen["uploads"]) >= 12 + 6 + 3 and set(seen["uploads"]) == {0}
