# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RAG operations retain their WAL files without blocking writes or storage teardown."""

from contextlib import closing
from pathlib import Path
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from storage import rag_db

# Some macOS Pythons build sqlite3 without extension loading, so the import alone is not enough.
requires_sqlite_vec = pytest.mark.skipif(
    not rag_db.RAG_AVAILABLE or not hasattr(sqlite3.Connection, "enable_load_extension"),
    reason = "sqlite-vec cannot be loaded here",
)


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "rag.db"
    rag_db.close_wal_keeper()
    rag_db.start_wal_keepers()
    monkeypatch.setattr(rag_db, "rag_db_path", lambda: path)
    monkeypatch.setattr(rag_db, "_schema_ready", set())
    yield path
    rag_db.close_wal_keeper()


def _sidecars(path):
    return Path(str(path) + "-wal"), Path(str(path) + "-shm")


@requires_sqlite_vec
def test_rag_operations_retain_sidecars_until_keeper_closes(db):
    with closing(rag_db.get_connection()) as conn:
        conn.execute("CREATE TABLE keeper_probe (value INTEGER)")
        conn.execute("INSERT INTO keeper_probe VALUES (42)")
        conn.commit()
    wal, shm = _sidecars(db)
    identities = [(path.stat().st_dev, path.stat().st_ino) for path in (wal, shm)]

    for _ in range(6):
        with closing(rag_db.get_connection()) as conn:
            assert conn.execute("SELECT value FROM keeper_probe").fetchone()[0] == 42
        assert [(path.stat().st_dev, path.stat().st_ino) for path in (wal, shm)] == identities

    rag_db.close_wal_keeper()
    assert not wal.exists()
    assert not shm.exists()
    with closing(sqlite3.connect(str(db))) as conn:
        assert conn.execute("SELECT value FROM keeper_probe").fetchone()[0] == 42


@requires_sqlite_vec
def test_idle_keeper_does_not_block_a_writer_or_checkpoint(db):
    with closing(rag_db.get_connection()) as conn:
        conn.execute("CREATE TABLE keeper_probe (value INTEGER)")
        conn.commit()

    with closing(sqlite3.connect(str(db), timeout = 1)) as conn:
        conn.execute("INSERT INTO keeper_probe VALUES (7)")
        conn.commit()
        assert conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone() == (0, 0, 0)
    assert _sidecars(db)[0].stat().st_size == 0
    with closing(rag_db.get_connection()) as conn:
        assert conn.execute("SELECT value FROM keeper_probe").fetchone()[0] == 7


@requires_sqlite_vec
def test_metadata_first_does_not_suppress_a_later_wal_keeper(db):
    with closing(rag_db.get_metadata_connection()) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert db.resolve() not in rag_db._wal_keepers

    with closing(rag_db.get_connection()) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert db.resolve() in rag_db._wal_keepers
    assert all(path.exists() for path in _sidecars(db))


@pytest.mark.parametrize(
    "with_vectors",
    [False, pytest.param(True, marks = requires_sqlite_vec)],
    ids = ["plain-metadata", "vec0-schema"],
)
def test_metadata_keeper_does_not_require_vector_extension(db, monkeypatch, with_vectors):
    connection = rag_db.get_connection() if with_vectors else sqlite3.connect(str(db))
    with closing(connection) as conn:
        if with_vectors:
            rag_db.ensure_vec(conn, 3)
        else:
            conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE metadata_probe (value INTEGER)")
        conn.execute("INSERT INTO metadata_probe VALUES (3)")
        conn.commit()
    rag_db.close_wal_keeper()
    rag_db.start_wal_keepers()

    def unavailable(_conn):
        pytest.fail("A metadata-only keeper must not load sqlite-vec")

    monkeypatch.setattr(rag_db, "sqlite_vec", SimpleNamespace(load = unavailable))
    monkeypatch.setattr(rag_db, "RAG_AVAILABLE", False)
    with closing(rag_db.get_metadata_connection()) as conn:
        assert conn.execute("SELECT value FROM metadata_probe").fetchone()[0] == 3
        if with_vectors:
            assert (
                conn.execute("SELECT 1 FROM sqlite_schema WHERE name='chunks_vec'").fetchone()
                is not None
            )
    assert db.resolve() in rag_db._wal_keepers
    assert all(path.exists() for path in _sidecars(db))


@requires_sqlite_vec
def test_shutdown_prevents_late_keeper_reopen_and_restart_enables_it(db):
    rag_db.get_connection().close()
    assert db.resolve() in rag_db._wal_keepers
    rag_db.close_wal_keeper()

    # A worker finishing after the bounded shutdown join may still record metadata.
    with closing(rag_db.get_metadata_connection()) as conn:
        conn.execute("SELECT count(*) FROM documents").fetchone()
    assert not rag_db._wal_keepers
    assert not any(path.exists() for path in _sidecars(db))

    rag_db.start_wal_keepers()
    rag_db.get_metadata_connection().close()
    assert db.resolve() in rag_db._wal_keepers
    assert all(path.exists() for path in _sidecars(db))


@requires_sqlite_vec
def test_concurrent_first_opens_share_one_keeper(db):
    ready = threading.Barrier(8)

    def read():
        ready.wait(timeout = 10)
        with closing(rag_db.get_connection()) as conn:
            return conn.execute("SELECT count(*) FROM documents").fetchone()[0]

    with ThreadPoolExecutor(max_workers = 8) as pool:
        futures = [pool.submit(read) for _ in range(8)]
        assert [future.result(timeout = 30) for future in futures] == [0] * 8
    assert set(rag_db._wal_keepers) == {db.resolve()}
    keeper = rag_db._wal_keepers[db.resolve()]
    rag_db.get_connection().close()
    assert rag_db._wal_keepers[db.resolve()] is keeper
    assert all(path.exists() for path in _sidecars(db))
