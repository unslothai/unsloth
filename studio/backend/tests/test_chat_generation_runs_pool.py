# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The reused connection behind the durable-run write path.

Opening a connection costs ~50x what the query costs, so this module keeps one per thread per
database. These are the properties that make that safe to do; each fails if the reuse in
``chat_generation_runs_db._connect`` is removed or mis-keyed.
"""

import sqlite3
import threading

import pytest

from storage import chat_generation_runs_db as runs_db
from storage import studio_db
from utils.account_context import OWNER, AccountContext, run_as
from utils.paths import storage_roots as roots

ALICE = AccountContext("11111111111111111111111111111111", "alice")
BOB = AccountContext("22222222222222222222222222222222", "bob")


@pytest.fixture(autouse = True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    studio_db.close_wal_keeper()
    runs_db.reset_connection_pool_for_tests()
    studio_db.reset_schema_state_for_tests()
    runs_db.reset_schema_state_for_tests()
    yield
    runs_db.reset_connection_pool_for_tests()


def test_one_thread_reuses_a_single_connection_for_the_same_database():
    first = runs_db._connect()
    underlying = first._conn
    first.close()
    second = runs_db._connect()
    try:
        assert second._conn is underlying
    finally:
        second.close()


def test_a_borrowed_connection_is_usable_after_being_returned():
    conn = runs_db._connect()
    conn.close()
    again = runs_db._connect()
    try:
        assert again.execute("SELECT 1").fetchone()[0] == 1
    finally:
        again.close()


def test_a_nested_borrow_gets_its_own_connection():
    """Two live borrows on one thread must not share a handle: BEGIN IMMEDIATE on a connection
    already inside a transaction raises, and one caller's rollback would discard the other's work."""
    outer = runs_db._connect()
    inner = runs_db._connect()
    try:
        assert getattr(inner, "_conn", inner) is not outer._conn
    finally:
        inner.close()
        outer.close()


def test_an_unfinished_transaction_is_rolled_back_before_reuse():
    conn = runs_db._connect()
    conn.execute("BEGIN IMMEDIATE")
    assert conn.in_transaction
    conn.close()
    nxt = runs_db._connect()
    try:
        # A handle still inside a transaction would fail here, or worse, silently adopt it.
        assert not nxt.in_transaction
        nxt.execute("BEGIN IMMEDIATE")
        nxt.rollback()
    finally:
        nxt.close()


def test_two_accounts_never_share_a_cached_connection():
    """Two accounts resolve to two databases, so a handle cached under one account id must never be
    handed to the other."""
    paths = {}
    handles = {}
    for account in (ALICE, BOB):

        def grab(account = account):
            conn = runs_db._connect()
            paths[account.account_id] = roots.studio_db_path().resolve()
            handles[account.account_id] = conn._conn
            conn.close()

        run_as(account, grab)

    assert paths[ALICE.account_id] != paths[BOB.account_id]
    assert handles[ALICE.account_id] is not handles[BOB.account_id]


def test_switching_account_does_not_write_to_the_previous_database():
    """The sharpest form of the isolation risk: a stale handle would send Bob's row to Alice."""

    def write(marker):
        conn = runs_db._connect()
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS pool_probe (marker TEXT)")
            conn.execute("INSERT INTO pool_probe (marker) VALUES (?)", (marker,))
            conn.commit()
        finally:
            conn.close()

    def read():
        conn = runs_db._connect()
        try:
            return [row[0] for row in conn.execute("SELECT marker FROM pool_probe").fetchall()]
        finally:
            conn.close()

    run_as(ALICE, lambda: write("alice"))
    run_as(BOB, lambda: write("bob"))
    assert run_as(ALICE, read) == ["alice"]
    assert run_as(BOB, read) == ["bob"]


def test_each_thread_gets_its_own_connection():
    """sqlite connections here are created with check_same_thread = True, so a handle shared
    across threads would raise ProgrammingError on use."""
    seen = {}

    def grab(name):
        conn = runs_db._connect()
        seen[name] = getattr(conn, "_conn", conn)
        conn.execute("SELECT 1").fetchone()
        conn.close()

    threads = [threading.Thread(target = grab, args = (name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert seen["a"] is not seen["b"]


def test_rebinding_schema_ready_drops_the_pooled_connection(monkeypatch):
    """conftest's _isolate_studio_home rebinds _schema_ready per test. An account id alone cannot
    see a home that moved beneath it, so the pool keys on that object's identity too."""
    conn = runs_db._connect()
    underlying = conn._conn
    conn.close()
    monkeypatch.setattr(runs_db, "_schema_ready", set())
    fresh = runs_db._connect()
    try:
        assert fresh._conn is not underlying
    finally:
        fresh.close()


def test_resetting_schema_state_drops_the_pooled_connection():
    """reset_schema_state_for_tests is the explicit hook, used by tests that swap homes directly."""
    conn = runs_db._connect()
    underlying = conn._conn
    conn.close()
    runs_db.reset_schema_state_for_tests()
    with pytest.raises(sqlite3.ProgrammingError):
        underlying.execute("SELECT 1")


def test_append_events_still_persists_through_a_reused_connection():
    studio_db.upsert_chat_thread(
        {"id": "t", "title": "Chat", "modelType": "base", "modelId": "local", "createdAt": 1}
    )
    studio_db.upsert_chat_message(
        {
            "id": "u",
            "threadId": "t",
            "role": "user",
            "content": [{"type": "text", "text": "x"}],
            "createdAt": 2,
        }
    )
    runs_db.create_run(
        run_id = "r",
        owner_subject = "alice",
        thread_id = "t",
        user_message_id = "u",
        assistant_message_id = "a",
        request_payload = {"model": "local", "messages": [], "stream": True},
    )
    token = runs_db.get_worker_token("r")
    runs_db.mark_running("r", token)
    chunk = ("chunk", {"choices": [{"delta": {"content": "hi "}, "index": 0}]})
    for _ in range(5):
        runs_db.append_events("r", token, [chunk] * 3)
    assert len(runs_db.list_events("r", 0)) >= 15


def test_an_idle_connection_on_another_thread_is_closed_by_a_global_discard():
    """Account retirement renames the account directory from a request thread, while the SSE loop
    parks its connections on a 32 thread pool of its own. Windows refuses the rename while any file
    underneath is open, so leaving those for their owners to close eventually is not enough."""
    parked = {}

    def park():
        conn = runs_db._connect()
        parked["conn"] = conn._conn
        conn.close()

    worker = threading.Thread(target = park)
    worker.start()
    worker.join()
    assert parked["conn"].execute("SELECT 1").fetchone()[0] == 1

    runs_db._discard_all_pooled()
    with pytest.raises(sqlite3.ProgrammingError):
        parked["conn"].execute("SELECT 1")


def test_closing_the_keeper_drops_the_pool_even_when_there_was_no_keeper():
    """journal_mode=WAL declines on filesystems without shared memory, so those installs never have
    a keeper. Retirement still calls close_wal_keeper_for and still needs the handle released."""
    studio_db.close_wal_keeper()
    conn = runs_db._connect()
    underlying = conn._conn
    conn.close()
    assert studio_db._wal_keepers == {}, "no keeper should be held for this test to mean anything"

    studio_db.close_wal_keeper_for(studio_db.studio_db_path())
    with pytest.raises(sqlite3.ProgrammingError):
        underlying.execute("SELECT 1")


def test_a_connection_in_use_during_a_global_discard_is_closed_on_return():
    """Yanking a handle mid query would fail that caller, so the borrower closes it instead of
    parking it back into a pool that has moved on."""
    borrowed = runs_db._connect()
    underlying = borrowed._conn
    runs_db._discard_all_pooled()
    assert underlying.execute("SELECT 1").fetchone()[0] == 1, "an in-use handle must survive"
    borrowed.close()
    with pytest.raises(sqlite3.ProgrammingError):
        underlying.execute("SELECT 1")
