# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os
import sqlite3

import storage.studio_db as studio_db


def _write_directly(key: str, value_json: str) -> None:
    """A write from another connection, as another worker or process would make it."""
    conn = sqlite3.connect(str(studio_db.studio_db_path()))
    try:
        conn.execute(
            "INSERT INTO app_settings(key, value_json, updated_at) VALUES(?, ?, 'now') "
            "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
            (key, value_json),
        )
        conn.commit()
    finally:
        conn.close()


def test_settings_reads_reuse_one_connection(monkeypatch):
    studio_db.upsert_app_settings({"reader_probe": 1})
    opened = []
    real = studio_db.connect_studio_db
    monkeypatch.setattr(
        studio_db, "connect_studio_db", lambda *a, **k: opened.append(1) or real(*a, **k)
    )
    studio_db.close_settings_readers()
    for _ in range(20):
        assert studio_db.get_app_setting("reader_probe") == 1
        assert studio_db.get_app_settings(["reader_probe", "missing"]) == {"reader_probe": 1}
    assert len(opened) == 1


def test_a_write_from_another_connection_is_seen_by_the_next_read():
    studio_db.upsert_app_settings({"reader_probe": "old"})
    assert studio_db.get_app_setting("reader_probe") == "old"
    _write_directly("reader_probe", '"new"')
    assert studio_db.get_app_setting("reader_probe") == "new"
    assert studio_db.get_app_settings(["reader_probe"]) == {"reader_probe": "new"}


def test_a_replaced_database_file_is_reopened_not_read_stale(tmp_path):
    studio_db.upsert_app_settings({"reader_probe": "before"})
    assert studio_db.get_app_setting("reader_probe") == "before"
    path = studio_db.studio_db_path()
    replacement = tmp_path / "replacement.db"
    conn = sqlite3.connect(str(path))
    try:
        backup = sqlite3.connect(str(replacement))
        conn.backup(backup)
        backup.execute(
            "UPDATE app_settings SET value_json = ? WHERE key = ?", ('"after"', "reader_probe")
        )
        backup.commit()
        backup.close()
    finally:
        conn.close()
    studio_db.close_wal_keeper()  # a restore lets go first, as the real one does
    os.replace(replacement, path)
    assert studio_db.get_app_setting("reader_probe") == "after"


def test_the_reader_is_closed_with_the_wal_keeper():
    assert studio_db.get_app_setting("reader_probe", "fallback") == "fallback"
    assert studio_db._settings_readers
    studio_db.close_wal_keeper()
    assert not studio_db._settings_readers


def test_a_missing_key_returns_the_fallback():
    assert studio_db.get_app_setting("never_written", {"d": 1}) == {"d": 1}
    assert studio_db.get_app_settings([]) == {}


def test_a_changed_file_identity_reopens_the_reader(monkeypatch):
    studio_db.upsert_app_settings({"reader_probe": 2})
    assert studio_db.get_app_setting("reader_probe") == 2
    first = studio_db._settings_readers[studio_db.studio_db_path()].conn
    real = studio_db._db_file_identity
    monkeypatch.setattr(studio_db, "_db_file_identity", lambda path: (real(path)[0], -1))
    assert studio_db.get_app_setting("reader_probe") == 2
    second = studio_db._settings_readers[studio_db.studio_db_path()].conn
    assert second is not first


def test_a_warm_reader_still_refuses_a_retired_account(monkeypatch):
    import pytest

    from core.training import account_jobs
    from utils.paths import storage_roots

    studio_db.upsert_app_settings({"reader_probe": 3})
    assert studio_db.get_app_setting("reader_probe") == 3
    monkeypatch.setattr(storage_roots, "is_owner_context", lambda: False)
    monkeypatch.setattr(account_jobs, "account_is_retired", lambda: True)
    for read in (
        lambda: studio_db.get_app_setting("reader_probe"),
        lambda: studio_db.get_app_settings(["reader_probe"]),
        studio_db.list_scan_folders,
    ):
        with pytest.raises(storage_roots.RetiredAccountError):
            read()


def test_a_read_never_queues_behind_another_on_the_shared_connection(monkeypatch):
    studio_db.upsert_app_settings({"reader_probe": 4})
    assert studio_db.get_app_setting("reader_probe") == 4
    reader = studio_db._settings_readers[studio_db.studio_db_path()]
    opened = []
    real = studio_db.connect_studio_db
    monkeypatch.setattr(
        studio_db, "connect_studio_db", lambda *a, **k: opened.append(1) or real(*a, **k)
    )
    assert reader.busy.acquire(blocking = False)
    try:
        assert studio_db.get_app_setting("reader_probe") == 4
    finally:
        reader.busy.release()
    assert opened == [1]
    assert studio_db.get_app_setting("reader_probe") == 4
    assert opened == [1]


def test_a_reader_closed_mid_lookup_falls_back_to_a_fresh_read(monkeypatch):
    studio_db.upsert_app_settings({"reader_probe": 5})
    assert studio_db.get_app_setting("reader_probe") == 5
    reader = studio_db._settings_readers[studio_db.studio_db_path()]
    reader.conn.close()  # as close_settings_readers would, after this caller looked the reader up
    monkeypatch.setattr(studio_db, "_settings_reader", lambda db_path: reader)
    assert studio_db.get_app_setting("reader_probe") == 5


def test_a_substituted_factory_is_read_through_and_never_cached(monkeypatch):
    studio_db.upsert_app_settings({"reader_probe": 6})
    studio_db.close_settings_readers()
    real = studio_db.get_connection
    monkeypatch.setattr(studio_db, "get_connection", lambda *a, **k: real(*a, **k))
    assert studio_db.get_app_setting("reader_probe") == 6
    assert not studio_db._settings_readers


def test_a_failed_reader_is_dropped_not_reused():
    studio_db.upsert_app_settings({"reader_probe": 7})
    assert studio_db.get_app_setting("reader_probe") == 7
    path = studio_db.studio_db_path()
    broken = studio_db._settings_readers[path]
    broken.conn.close()
    assert studio_db.get_app_setting("reader_probe") == 7
    assert studio_db._settings_readers.get(path) is not broken


def test_a_schema_reset_closes_the_readers():
    assert studio_db.get_app_setting("reader_probe", "fallback") == "fallback"
    assert studio_db._settings_readers
    studio_db.reset_schema_state_for_tests()
    assert not studio_db._settings_readers


def test_a_forked_child_never_reuses_the_parent_reader():
    import os

    import pytest

    if not hasattr(os, "fork"):
        pytest.skip("no fork on this platform")
    assert studio_db.get_app_setting("reader_probe", "fallback") == "fallback"
    parent = dict(studio_db._settings_readers)
    assert parent
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - runs in the child
        ok = not studio_db._settings_readers and not studio_db._settings_reader_lock.locked()
        os.write(write, b"1" if ok else b"0")
        os._exit(0)
    os.close(write)
    assert os.read(read, 1) == b"1"
    os.close(read)
    os.waitpid(pid, 0)
    assert studio_db._settings_readers == parent


def test_a_reader_opening_during_a_close_is_never_published(monkeypatch):
    studio_db.close_settings_readers()
    real = studio_db.connect_studio_db

    def open_then_close_all(*a, **k):
        conn = real(*a, **k)
        studio_db.close_settings_readers()  # the WAL keeper closes while this reader opens
        return conn

    monkeypatch.setattr(studio_db, "connect_studio_db", open_then_close_all)
    assert studio_db.get_app_setting("reader_probe", "fallback") == "fallback"
    assert not studio_db._settings_readers


def test_a_file_replaced_while_the_reader_opens_is_not_cached(monkeypatch):
    studio_db.upsert_app_settings({"reader_probe": "x"})
    studio_db.close_settings_readers()
    real = studio_db._db_file_identity
    seen = []

    def moving(path):
        seen.append(1)
        identity = real(path)
        return identity if len(seen) == 1 else (identity[0], -len(seen))

    monkeypatch.setattr(studio_db, "_db_file_identity", moving)
    assert studio_db.get_app_setting("reader_probe") == "x"
    assert not studio_db._settings_readers
