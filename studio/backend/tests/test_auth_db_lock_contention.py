# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A busy auth.db must not log the user out.

On a Colab session a large GGUF download saturated the disk, /api/auth/refresh hit "database is
locked" after the busy timeout, the route answered a plain 500 after the old token was already
consumed, and the UI signed the user out. These pin the three links: auth.db commits skip the WAL
fsync, the refresh rotation is one transaction (a failure leaves the old token valid), and a busy
database answers a retryable 503.
"""

import hashlib
import inspect
import logging
import secrets
import sqlite3
import threading
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.auth as auth_routes
from auth import policy, storage
from auth.authentication import new_refresh_token_value
from storage import sqlite_gate

FULL, NORMAL = 2, 1


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth" / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    policy.invalidate_account_cache()
    yield
    policy.invalidate_account_cache()


@pytest.fixture
def owner(fresh_db):
    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = "human-password-123",
        jwt_secret = secrets.token_urlsafe(64),
        must_change_password = False,
    )
    return storage.DEFAULT_ADMIN_USERNAME


@pytest.fixture
def client(owner):
    app = FastAPI()
    app.include_router(auth_routes.router, prefix = "/api/auth")
    with TestClient(app) as http:
        yield http


def _refresh(http, token):
    return http.post("/api/auth/refresh", json = {"refresh_token": token})


def _issue(username):
    token, expires_at = new_refresh_token_value()
    storage.save_refresh_token(token, username, expires_at)
    return token


def _synchronous(conn) -> int:
    return int(conn.execute("PRAGMA synchronous").fetchone()[0])


def _live_tokens() -> int:
    conn = storage.get_connection()
    try:
        return conn.execute("SELECT COUNT(*) FROM refresh_tokens").fetchone()[0]
    finally:
        conn.close()


def test_auth_connections_use_normal_sync_under_wal(owner):
    # owner already opened once, so this also covers the schema-cache early return.
    for _ in range(2):
        conn = storage.get_connection()
        try:
            assert str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"
            assert _synchronous(conn) == NORMAL
        finally:
            conn.close()


def test_rollback_journal_keeps_full_sync(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "rollback.db"))
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
        sqlite_gate.apply_wal_synchronous(conn)
        assert _synchronous(conn) == FULL
    finally:
        conn.close()


@pytest.mark.parametrize(
    "answer",
    [None, (None,), ("",), (123,), sqlite3.OperationalError("pragma unavailable")],
)
def test_unreadable_journal_mode_is_not_treated_as_wal(answer):
    class Connection:
        def __init__(self):
            self.executed = []

        def execute(self, sql, *args):
            self.executed.append(sql)
            if isinstance(answer, Exception):
                raise answer
            return type("Cursor", (), {"fetchone": lambda _self: answer})()

    conn = Connection()
    sqlite_gate.apply_wal_synchronous(conn)
    assert "PRAGMA synchronous=NORMAL" not in conn.executed


def test_busy_auth_db_answers_retryable_503_and_keeps_the_token(client, owner, monkeypatch):
    token = _issue(owner)
    monkeypatch.setattr(storage, "_BUSY_TIMEOUT_MS", 50)
    holder = sqlite3.connect(str(storage.DB_PATH), isolation_level = None)
    try:
        holder.execute("BEGIN IMMEDIATE")
        busy = _refresh(client, token)
    finally:
        holder.execute("ROLLBACK")
        holder.close()

    assert busy.status_code == 503
    assert busy.headers["Retry-After"] == "2"
    assert busy.json()["detail"] == "Unsloth Studio is busy. Try again in a moment."

    ok = _refresh(client, token)
    assert ok.status_code == 200
    body = ok.json()
    assert body["access_token"] and body["refresh_token"] != token
    assert body["account_id"] == "owner"
    assert body["must_change_password"] is False

    assert _refresh(client, body["refresh_token"]).status_code == 200
    assert _refresh(client, token).status_code == 401


def _last_used(key_id):
    with sqlite3.connect(str(storage.DB_PATH)) as conn:
        return conn.execute("SELECT last_used_at FROM api_keys WHERE id = ?", (key_id,)).fetchone()[
            0
        ]


def test_busy_auth_db_still_validates_an_api_key_without_recording_use(owner, monkeypatch):
    raw, row = storage.create_api_key(owner, name = "cli")
    assert storage.validate_api_key_account(raw) is not None
    used = _last_used(row["id"])
    assert used is not None

    monkeypatch.setattr(storage, "_BUSY_TIMEOUT_MS", 50)
    holder = sqlite3.connect(str(storage.DB_PATH), isolation_level = None)
    try:
        holder.execute("BEGIN IMMEDIATE")
        verified = storage.validate_api_key_account(raw)
    finally:
        holder.execute("ROLLBACK")
        holder.close()

    assert verified is not None
    assert verified[0]["username"] == owner
    assert _last_used(row["id"]) == used


def test_a_non_busy_error_on_the_api_key_touch_still_raises(owner, monkeypatch):
    raw, _row = storage.create_api_key(owner, name = "cli")
    real = storage.get_connection

    class _Broken:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, *args):
            if sql == "BEGIN IMMEDIATE":
                raise sqlite3.OperationalError("disk I/O error")
            return self._conn.execute(sql, *args)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    monkeypatch.setattr(storage, "get_connection", lambda *a, **k: _Broken(real(*a, **k)))
    with pytest.raises(sqlite3.OperationalError, match = "disk I/O error"):
        storage.validate_api_key_account(raw)


def test_failed_replacement_insert_leaves_the_old_token_valid(client, owner):
    token = _issue(owner)
    conn = storage.get_connection()
    try:
        conn.execute(
            "CREATE TRIGGER reject_insert BEFORE INSERT ON refresh_tokens "
            "BEGIN SELECT RAISE(ABORT, 'insert rejected'); END"
        )
        conn.commit()
    finally:
        conn.close()

    new_token, expires_at = new_refresh_token_value()
    with pytest.raises(sqlite3.IntegrityError):
        storage.rotate_refresh_token(token, new_token, expires_at)
    assert _live_tokens() == 1

    conn = storage.get_connection()
    try:
        conn.execute("DROP TRIGGER reject_insert")
        conn.commit()
    finally:
        conn.close()
    assert _refresh(client, token).status_code == 200


def test_refresh_token_stays_single_use_and_logout_revokes_the_rotation(client, owner):
    token = _issue(owner)
    first = _refresh(client, token)
    assert first.status_code == 200
    assert _refresh(client, token).status_code == 401

    storage.revoke_user_refresh_tokens(owner)
    assert _live_tokens() == 0
    assert _refresh(client, first.json()["refresh_token"]).status_code == 401


def test_non_busy_operational_error_is_not_mapped_to_503(owner, monkeypatch):
    def broken(*_args):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(storage, "rotate_refresh_token", broken)
    app = FastAPI()
    app.include_router(auth_routes.router, prefix = "/api/auth")
    with TestClient(app, raise_server_exceptions = False) as http:
        response = _refresh(http, "t")
    assert response.status_code == 500


def _wal() -> Path:
    return Path(f"{storage.DB_PATH}-wal")


def _short_lived_write(username, n = 1):
    for _ in range(n):
        _issue(username)


def test_keeper_holds_the_wal_until_released(owner):
    assert not _wal().exists(), "without a keeper the last close checkpoints and removes it"
    assert storage.open_wal_keeper() is True
    _short_lived_write(owner, 20)
    assert _wal().is_file()

    storage.close_wal_keeper()
    assert storage._wal_keeper is None
    assert not _wal().exists()


def test_closing_a_connection_under_the_keeper_does_not_checkpoint(owner):
    """The checkpoint is what fsyncs inside sqlite_gate's process-wide lock."""
    assert storage.open_wal_keeper() is True
    try:
        _short_lived_write(owner)
        main_file = hashlib.sha256(storage.DB_PATH.read_bytes()).hexdigest()
        wal_size = _wal().stat().st_size

        _short_lived_write(owner)

        assert _wal().stat().st_size >= wal_size
        assert hashlib.sha256(storage.DB_PATH.read_bytes()).hexdigest() == main_file
    finally:
        storage.close_wal_keeper()


def test_opening_again_replaces_the_keeper_of_an_earlier_db(owner, tmp_path, monkeypatch):
    assert storage.open_wal_keeper() is True
    stale = storage._wal_keeper[1]

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "other" / "auth.db")
    assert storage.open_wal_keeper() is True
    assert storage._wal_keeper[0] == str(storage.DB_PATH.resolve())
    with pytest.raises(sqlite3.ProgrammingError, match = "closed database"):
        stale.execute("SELECT 1")
    storage.close_wal_keeper()
    assert storage._wal_keeper is None


@pytest.mark.parametrize("then", ["close", "reopen"])
def test_a_keeper_opened_on_an_exited_thread_is_released(owner, then):
    opened = threading.Thread(target = storage.open_wal_keeper)
    opened.start()
    opened.join()
    stale = storage._wal_keeper[1]

    if then == "reopen":
        assert storage.open_wal_keeper() is True
        assert storage._wal_keeper[1] is not stale
    storage.close_wal_keeper()
    with pytest.raises(sqlite3.ProgrammingError, match = "closed database"):
        stale.execute("SELECT 1")
    assert not _wal().exists()


def test_keeper_declines_a_rollback_journal(owner, monkeypatch, caplog):
    real = storage.get_connection

    def refused_wal(**kwargs):
        conn = real(**kwargs)
        conn.execute("PRAGMA journal_mode=DELETE")
        return conn

    monkeypatch.setattr(storage, "get_connection", refused_wal)
    with caplog.at_level(logging.INFO, logger = storage.logger.name):
        assert storage.open_wal_keeper() is False
    assert storage._wal_keeper is None
    assert "not WAL" in caplog.text


def test_the_lifespan_holds_the_auth_keeper_across_every_writer():
    """Reads the source rather than running the lifespan, which imports the whole stack."""
    import main

    source = inspect.getsource(main.lifespan)
    served = source.index("yield")
    opened = source.index("storage.open_wal_keeper()")
    assert opened < source.index("startup_reconciliation_accounts()") < served
    assert opened < source.index("storage.ensure_default_admin()") < served
    assert (
        served
        < source.index("await run_lifespan_shutdown(")
        < source.index("storage.close_wal_keeper()")
    )


def test_a_cross_origin_client_can_read_the_refresh_retry_after():
    # tauri://localhost and tunnel pages are cross-origin: an unexposed header reads as null there.
    import main
    exposed = [
        header
        for middleware in main.app.user_middleware
        for header in middleware.kwargs.get("expose_headers", ())
    ]
    assert "Retry-After" in exposed
