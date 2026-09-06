# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Concurrent first opens of auth.db add the account columns exactly once.

The backend pushes auth reads to a threadpool, so several connections can be
the first to see the columns missing. The migration takes the write lock and
re-reads the table before altering it; a loser never raises."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from auth import policy, storage


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth" / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    policy.invalidate_account_cache()
    yield
    policy.invalidate_account_cache()


def _open_and_close():
    conn = storage.get_connection()
    try:
        return {row[1] for row in conn.execute("PRAGMA table_info(auth_user)")}
    finally:
        conn.close()


@pytest.mark.parametrize("round_", range(6))
def test_concurrent_first_opens_never_raise(fresh_db, round_):
    with ThreadPoolExecutor(max_workers = 8) as pool:
        columns = list(pool.map(lambda _i: _open_and_close(), range(8)))
    expected = {name for name, _decl in storage._ACCOUNT_COLUMNS}
    for seen in columns:
        assert expected <= seen


def test_legacy_table_is_migrated_once_under_contention(fresh_db):
    import sqlite3

    storage.DB_PATH.parent.mkdir(parents = True)
    conn = sqlite3.connect(storage.DB_PATH)
    conn.execute(
        "CREATE TABLE auth_user (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, "
        "password_salt TEXT NOT NULL, password_hash TEXT NOT NULL, jwt_secret TEXT NOT NULL, "
        "must_change_password INTEGER NOT NULL DEFAULT 0)"
    )
    conn.execute(
        "INSERT INTO auth_user (username, password_salt, password_hash, jwt_secret) VALUES ('unsloth', 's', 'h', 'j')"
    )
    conn.commit()
    conn.close()
    with ThreadPoolExecutor(max_workers = 8) as pool:
        list(pool.map(lambda _i: _open_and_close(), range(16)))
    record = storage.get_user_record("unsloth")
    assert record["account_id"] == "owner" and record["role"] == "owner"
    assert record["jwt_secret"] == "j"
