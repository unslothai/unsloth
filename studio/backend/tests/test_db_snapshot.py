# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sqlite3

from storage.db_snapshot import create_snapshot, restore_snapshot_if_needed


def _database(path, value):
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE metadata (value TEXT)")
        conn.execute("INSERT INTO metadata VALUES (?)", (value,))


def _value(path):
    with sqlite3.connect(path) as conn:
        return conn.execute("SELECT value FROM metadata").fetchone()[0]


def test_missing_snapshot_is_noop(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_DB_BACKUP", str(tmp_path / "missing.db"))
    assert not restore_snapshot_if_needed(tmp_path / "live.db")


def test_valid_snapshot_restores_but_never_overwrites_local(monkeypatch, tmp_path):
    snapshot = tmp_path / "persistent" / "studio.db"
    snapshot.parent.mkdir()
    _database(snapshot, "snapshot")
    live = tmp_path / "runtime" / "studio.db"
    monkeypatch.setenv("UNSLOTH_STUDIO_DB_BACKUP", str(snapshot))
    assert restore_snapshot_if_needed(live)
    assert _value(live) == "snapshot"
    with sqlite3.connect(live) as conn:
        conn.execute("UPDATE metadata SET value = 'local'")
    assert not restore_snapshot_if_needed(live)
    assert _value(live) == "local"


def test_wal_database_has_consistent_atomic_snapshot(monkeypatch, tmp_path):
    live = tmp_path / "runtime" / "studio.db"
    live.parent.mkdir()
    destination = tmp_path / "persistent" / "studio.db"
    monkeypatch.setenv("UNSLOTH_STUDIO_DB_BACKUP", str(destination))
    with sqlite3.connect(live) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE metadata (value TEXT)")
        conn.execute("INSERT INTO metadata VALUES ('wal-visible')")
        conn.commit()
        assert create_snapshot(live)
    assert _value(destination) == "wal-visible"
    assert not list(destination.parent.glob(".studio.db.snapshot-*"))


def test_corrupt_snapshot_falls_back_without_creating_live(monkeypatch, tmp_path):
    snapshot = tmp_path / "corrupt.db"
    snapshot.write_bytes(b"not sqlite")
    live = tmp_path / "live.db"
    monkeypatch.setenv("UNSLOTH_STUDIO_DB_BACKUP", str(snapshot))
    assert not restore_snapshot_if_needed(live)
    assert not live.exists()
