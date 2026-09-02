# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""studio.db must not turn one slow writer into a stream of "database is locked".

A live Colab session logged six `research.supervisor_iteration_failed` tracebacks in 37
seconds while a settings write and a multi-GB model download shared the disk. Each test
here guards a distinct link in that chain, plus the upgrade and downgrade paths, since
every existing install already carries the old schema.

Nothing here needs a GPU or a network: the behaviour under test is SQLite pragmas, one
poll query, and log levels.
"""

import asyncio
import hashlib
import inspect
import logging
import sqlite3
import threading
import time
from pathlib import Path

import pytest

import storage.research_runs_db as research_runs_db
import storage.studio_db as studio_db

FULL, NORMAL = 2, 1


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A studio.db in a temp dir, with the per-process schema latch reset."""
    monkeypatch.setattr(studio_db, "studio_db_path", lambda: tmp_path / "studio.db")
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    conn = studio_db.get_connection()
    conn.close()
    yield tmp_path / "studio.db"
    # A keeper left open would hold this temp database past the test that made it.
    studio_db.close_wal_keeper()


def _journal_mode(path: Path) -> str:
    conn = sqlite3.connect(str(path))
    try:
        return str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        conn.close()


def _synchronous(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA synchronous").fetchone()[0])


def _seed_message(
    conn,
    thread = "t1",
    message = "m1",
):
    conn.execute(
        "INSERT OR IGNORE INTO chat_threads (id, title, model_type, created_at, updated_at) "
        "VALUES (?, 'T', 'gguf', 0, 0)",
        (thread,),
    )
    conn.execute(
        "INSERT INTO chat_messages "
        "(id, thread_id, role, content_json, attachments_json, metadata_json, created_at) "
        "VALUES (?, ?, 'assistant', '[]', '[]', '{}', 0)",
        (message, thread),
    )
    conn.commit()


def _dirty(conn) -> int:
    row = conn.execute(
        "SELECT dirty FROM chat_attachment_inventory_state WHERE singleton = 1"
    ).fetchone()
    return int(row["dirty"])


def _hold_writer_lock(conn) -> None:
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "INSERT INTO app_settings (key, value_json, updated_at) VALUES ('probe','1','0') "
        "ON CONFLICT(key) DO UPDATE SET value_json='1'"
    )


# --- synchronous=NORMAL, but only where WAL makes it safe -------------------------------


def test_wal_database_drops_to_normal(db):
    """synchronous=FULL fsyncs under the writer lock; NORMAL is the WAL-safe pairing."""
    assert _journal_mode(db) == "wal"
    conn = studio_db.get_connection()
    try:
        assert _synchronous(conn) == NORMAL
    finally:
        conn.close()


@pytest.mark.parametrize("mode", ["delete", "truncate", "persist", "memory"])
def test_non_wal_journal_keeps_full(db, monkeypatch, mode):
    """PRAGMA journal_mode=WAL silently declines on filesystems without shared memory.

    Network shares and some FUSE or container mounts land there, Windows users most
    often. A rollback journal needs the fsync NORMAL removes, so those installs keep
    FULL and simply do not get the speedup.
    """
    setup = sqlite3.connect(str(db))
    try:
        setup.execute(f"PRAGMA journal_mode={mode}")
        setup.commit()
    finally:
        setup.close()
    assert _journal_mode(db) != "wal"

    monkeypatch.setattr(studio_db, "_schema_ready", True)
    conn = studio_db.get_connection()
    try:
        assert _synchronous(conn) == FULL, f"{mode} must not lose its commit fsync"
    finally:
        conn.close()


@pytest.mark.parametrize(
    "answer",
    [None, (None,), ("",), (123,), sqlite3.OperationalError("pragma unavailable")],
)
def test_unreadable_journal_mode_is_not_treated_as_wal(answer):
    """An unexpected or failing answer keeps the safe default rather than raising.

    sqlite3.Connection is a C type and cannot be monkeypatched, so this uses a double.
    """

    class Connection:
        def __init__(self):
            self.executed = []

        def execute(self, sql, *args):
            self.executed.append(sql)
            if isinstance(answer, Exception):
                raise answer
            return type("Cursor", (), {"fetchone": lambda _self: answer})()

    conn = Connection()
    studio_db._apply_wal_synchronous(conn)
    assert "PRAGMA synchronous=NORMAL" not in conn.executed


def test_committed_rows_survive_a_reopen(db):
    """NORMAL changes fsync timing, not correctness."""
    conn = studio_db.get_connection()
    try:
        _seed_message(conn, "t-durable", "m-durable")
    finally:
        conn.close()
    conn = studio_db.get_connection()
    try:
        assert (
            conn.execute("SELECT id FROM chat_messages WHERE id = 'm-durable'").fetchone()
            is not None
        )
    finally:
        conn.close()


def test_concurrent_openers_all_get_normal(db):
    results, errors = [], []

    def worker():
        try:
            conn = studio_db.get_connection()
            try:
                results.append(_synchronous(conn))
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target = worker) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    assert results == [NORMAL] * 16


# --- the attachment inventory is dirtied by attachments, not by bookkeeping -------------


@pytest.mark.parametrize(
    "sql, dirties, why",
    [
        (
            "UPDATE chat_messages SET metadata_json='{\"a\":1}' WHERE id='m1'",
            False,
            "a generation status change touches no attachment",
        ),
        (
            "UPDATE chat_messages SET role='user' WHERE id='m1'",
            False,
            "role is not an inventory input",
        ),
        (
            "UPDATE chat_messages SET attachments_json='[{}]' WHERE id='m1'",
            True,
            "attachments changed",
        ),
        (
            "UPDATE chat_messages SET content_json='[{\"t\":1}]' WHERE id='m1'",
            True,
            "inline content can carry data URIs",
        ),
        (
            "UPDATE chat_messages SET attachments_json='[]' WHERE id='m1'",
            True,
            "UPDATE OF fires on the SET list, not on a changed value",
        ),
        ("DELETE FROM chat_messages WHERE id='m1'", True, "rows went away"),
    ],
)
def test_inventory_trigger_scope(db, sql, dirties, why):
    """The rebuild re-hashes every attachment in every thread inside BEGIN IMMEDIATE, so
    dirtying it from an unrelated column is what made an ordinary autosave hold the
    writer lock for as long as the history was large."""
    conn = studio_db.get_connection()
    try:
        _seed_message(conn)
        studio_db._mark_chat_attachment_inventory_clean(conn)
        conn.commit()
        assert _dirty(conn) == 0
        conn.execute(sql)
        conn.commit()
        assert _dirty(conn) == (1 if dirties else 0), why
    finally:
        conn.close()


def test_upgrade_replaces_the_unscoped_trigger(tmp_path, monkeypatch):
    """Every existing install already carries the unscoped trigger.

    CREATE TRIGGER IF NOT EXISTS would have kept it silently, so the fix drops first.
    """
    path = tmp_path / "studio.db"
    monkeypatch.setattr(studio_db, "studio_db_path", lambda: path)
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    conn = studio_db.get_connection()
    try:
        conn.execute("DROP TRIGGER IF EXISTS chat_attachment_inventory_dirty_update")
        conn.execute(
            """
            CREATE TRIGGER chat_attachment_inventory_dirty_update
            AFTER UPDATE ON chat_messages
            BEGIN
                INSERT INTO chat_attachment_inventory_state
                    (singleton, inventory_version, dirty, backfilled_at)
                VALUES (1, 0, 1, 0)
                ON CONFLICT(singleton) DO UPDATE SET dirty = 1;
            END
            """
        )
        conn.commit()
        assert (
            "UPDATE OF"
            not in conn.execute(
                "SELECT sql FROM sqlite_master WHERE name='chat_attachment_inventory_dirty_update'"
            ).fetchone()[0]
        )
    finally:
        conn.close()

    monkeypatch.setattr(studio_db, "_schema_ready", False)
    conn = studio_db.get_connection()
    try:
        assert (
            "UPDATE OF attachments_json, content_json"
            in conn.execute(
                "SELECT sql FROM sqlite_master WHERE name='chat_attachment_inventory_dirty_update'"
            ).fetchone()[0]
        )
        _seed_message(conn)
        studio_db._mark_chat_attachment_inventory_clean(conn)
        conn.commit()
        conn.execute("UPDATE chat_messages SET metadata_json='{\"x\":1}' WHERE id='m1'")
        conn.commit()
        assert _dirty(conn) == 0
    finally:
        conn.close()


def _legacy_unscoped_trigger(conn) -> None:
    conn.execute("DROP TRIGGER IF EXISTS chat_attachment_inventory_dirty_update")
    conn.execute(
        """
        CREATE TRIGGER chat_attachment_inventory_dirty_update
        AFTER UPDATE ON chat_messages
        BEGIN
            INSERT INTO chat_attachment_inventory_state
                (singleton, inventory_version, dirty, backfilled_at)
            VALUES (1, 0, 1, 0)
            ON CONFLICT(singleton) DO UPDATE SET dirty = 1;
        END
        """
    )
    conn.commit()


def test_eight_processes_upgrading_at_once_all_succeed(db):
    """Two Studio processes on one studio.db each carry their own `_schema_ready`.

    DDL does not open a transaction under sqlite3's legacy transaction control -- only
    INSERT/UPDATE/DELETE/REPLACE do -- so a bare DROP + CREATE pair runs in autocommit and
    can interleave as drop/drop/create/create, where the second CREATE raises
    "trigger already exists" straight out of get_connection. The forced sequence is in
    test_a_lost_create_race_cannot_raise; this is the end-to-end shape of it.
    """
    setup = studio_db.get_connection()
    try:
        _legacy_unscoped_trigger(setup)
    finally:
        setup.close()

    errors, seen = [], []
    start = threading.Barrier(8)

    def worker():
        conn = sqlite3.connect(str(db), timeout = 10)
        conn.row_factory = sqlite3.Row
        try:
            start.wait(timeout = 10)
            studio_db._replace_inventory_update_trigger(conn)
            seen.append(
                conn.execute(
                    "SELECT sql FROM sqlite_master WHERE name = ?",
                    ("chat_attachment_inventory_dirty_update",),
                ).fetchone()["sql"]
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            conn.close()

    threads = [threading.Thread(target = worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    assert len(seen) == 8
    assert all("UPDATE OF attachments_json, content_json" in sql for sql in seen)


def test_the_migration_is_skipped_once_the_scoped_trigger_is_installed(db):
    """Every connection would otherwise re-drop and re-create it, taking the writer lock
    on a hot path for nothing."""
    conn = studio_db.get_connection()
    try:
        assert (
            "UPDATE OF"
            in conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = ?",
                ("chat_attachment_inventory_dirty_update",),
            ).fetchone()["sql"]
        )
        held = sqlite3.connect(str(db), timeout = 0.2)
        try:
            _hold_writer_lock(held)
            # No writer lock is taken, so this returns even while one is held elsewhere.
            studio_db._replace_inventory_update_trigger(conn)
        finally:
            held.rollback()
            held.close()
    finally:
        conn.close()


def test_a_lost_create_race_cannot_raise(db):
    """The interleave itself, forced rather than raced for.

    Racing two threads for it does not work: every DDL statement takes the writer lock, so
    the window between the drop committing and the create is too narrow to sample.
    """
    setup = studio_db.get_connection()
    try:
        _legacy_unscoped_trigger(setup)
    finally:
        setup.close()

    first = sqlite3.connect(str(db), timeout = 10)
    second = sqlite3.connect(str(db), timeout = 10)
    try:
        for conn in (first, second):
            conn.execute("DROP TRIGGER IF EXISTS chat_attachment_inventory_dirty_update")
            conn.commit()
        first.execute(studio_db._INVENTORY_UPDATE_TRIGGER_SQL)
        first.commit()

        # Without IF NOT EXISTS this raises out of get_connection, so the second process
        # cannot open the database at all.
        unguarded = studio_db._INVENTORY_UPDATE_TRIGGER_SQL.replace("IF NOT EXISTS ", "")
        with pytest.raises(sqlite3.OperationalError, match = "already exists"):
            second.execute(unguarded)

        second.execute(studio_db._INVENTORY_UPDATE_TRIGGER_SQL)
        second.commit()
    finally:
        first.close()
        second.close()

    conn = sqlite3.connect(str(db))
    try:
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = ?",
            ("chat_attachment_inventory_dirty_update",),
        ).fetchone()[0]
    finally:
        conn.close()
    assert "UPDATE OF attachments_json, content_json" in sql


def test_the_drop_and_the_create_share_one_writer_lock():
    """Held across both, so no opener sees the table without its trigger.

    sqlite3.Connection is a C type and cannot be monkeypatched, hence the double.
    """

    class Connection:
        in_transaction = False

        def __init__(self):
            self.executed = []
            self.committed = 0

        def execute(self, sql, *args):
            self.executed.append(" ".join(sql.split())[:40])
            return type("Cursor", (), {"fetchone": lambda _self: None})()

        def commit(self):
            self.committed += 1

        def rollback(self):
            raise AssertionError("nothing here should roll back")

    conn = Connection()
    studio_db._replace_inventory_update_trigger(conn)

    kinds = [sql.split()[0] for sql in conn.executed]
    assert kinds == ["SELECT", "BEGIN", "DROP", "CREATE"], conn.executed
    assert conn.executed[1].startswith("BEGIN IMMEDIATE"), conn.executed[1]
    assert "IF NOT EXISTS" in conn.executed[3]
    assert conn.committed == 1, "one commit, at the end of the pair"


def test_downgrade_still_sees_attachment_changes(db):
    """An older Unsloth run against an upgraded database keeps the scoped trigger, since
    its CREATE TRIGGER IF NOT EXISTS finds the name taken. That is safe: the scoped
    trigger still fires for everything the inventory derives from."""
    conn = studio_db.get_connection()
    try:
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS chat_attachment_inventory_dirty_update
            AFTER UPDATE ON chat_messages
            BEGIN
                INSERT INTO chat_attachment_inventory_state
                    (singleton, inventory_version, dirty, backfilled_at)
                VALUES (1, 0, 1, 0)
                ON CONFLICT(singleton) DO UPDATE SET dirty = 1;
            END
            """
        )
        conn.commit()
        _seed_message(conn)
        studio_db._mark_chat_attachment_inventory_clean(conn)
        conn.commit()
        conn.execute("UPDATE chat_messages SET attachments_json='[{}]' WHERE id='m1'")
        conn.commit()
        assert _dirty(conn) == 1
    finally:
        conn.close()


# --- claim_next: cheap when idle, unchanged when there is work --------------------------


def _make_run(
    run_id = "r1",
    thread = "t1",
    status = "queued",
):
    studio_db.upsert_chat_thread({"id": thread, "title": "T", "modelType": "gguf", "createdAt": 1})
    studio_db.upsert_chat_message(
        {"id": f"u-{run_id}", "threadId": thread, "role": "user", "content": [], "createdAt": 2}
    )
    studio_db.upsert_chat_message(
        {
            "id": f"a-{run_id}",
            "threadId": thread,
            "parentId": f"u-{run_id}",
            "role": "assistant",
            "content": [],
            "createdAt": 3,
        }
    )
    research_runs_db.create_run(
        run_id = run_id,
        owner_subject = "sub",
        thread_id = thread,
        user_message_id = f"u-{run_id}",
        assistant_message_id = f"a-{run_id}",
        config = {},
    )
    if status != "queued":
        conn = studio_db.get_connection()
        try:
            conn.execute("UPDATE research_runs SET status=? WHERE id=?", (status, run_id))
            conn.commit()
        finally:
            conn.close()


def test_idle_poll_takes_no_write_lock(db):
    """The original failure reproduced: this used to raise after the 5s busy timeout."""
    holder = studio_db.get_connection()
    try:
        _hold_writer_lock(holder)
        started = time.monotonic()
        assert research_runs_db.claim_next("worker-1") is None
        assert time.monotonic() - started < 1.0, "must not have queued behind the writer"
    finally:
        holder.rollback()
        holder.close()


def test_claim_next_still_claims_real_work(db):
    _make_run()
    claimed = research_runs_db.claim_next("worker-1")
    assert claimed is not None and claimed["id"] == "r1"
    assert research_runs_db.claim_next("worker-2") is None, "the lease excludes others"


@pytest.mark.parametrize("status", ["planning", "queued", "running", "cancelling"])
def test_probe_agrees_with_the_transaction_when_claimable(db, status):
    _make_run(status = status)
    assert research_runs_db._has_claimable(research_runs_db.now_ms()) is True
    assert research_runs_db.claim_next("worker-1") is not None


@pytest.mark.parametrize(
    "status", ["completed", "failed", "cancelled", "paused", "awaiting_approval"]
)
def test_probe_agrees_with_the_transaction_when_not_claimable(db, status):
    _make_run(status = status)
    assert research_runs_db._has_claimable(research_runs_db.now_ms()) is False
    assert research_runs_db.claim_next("worker-1") is None


def test_expired_lease_becomes_claimable_again(db):
    _make_run()
    assert research_runs_db.claim_next("worker-1") is not None
    assert research_runs_db._has_claimable(research_runs_db.now_ms()) is False
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE research_runs SET lease_expires_at=? WHERE id='r1'",
            (research_runs_db.now_ms() - 1,),
        )
        conn.commit()
    finally:
        conn.close()
    assert research_runs_db._has_claimable(research_runs_db.now_ms()) is True
    assert research_runs_db.claim_next("worker-2") is not None


def test_run_without_a_thread_claim_is_invisible(db):
    """The probe must reproduce the transaction's join, not just its WHERE clause."""
    _make_run()
    conn = studio_db.get_connection()
    try:
        conn.execute("DELETE FROM research_thread_claims")
        conn.commit()
    finally:
        conn.close()
    assert research_runs_db._has_claimable(research_runs_db.now_ms()) is False
    assert research_runs_db.claim_next("worker-1") is None


def test_concurrent_workers_claim_a_run_exactly_once(db):
    _make_run()
    claims, errors = [], []

    def worker(name):
        try:
            if research_runs_db.claim_next(name) is not None:
                claims.append(name)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target = worker, args = (f"w{i}",)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    assert len(claims) == 1, f"the read-first probe must not let two workers claim: {claims}"


# --- the busy predicate and the supervisor's error ladder -------------------------------


@pytest.mark.parametrize(
    "message, busy",
    [
        ("database is locked", True),
        ("database table is locked", True),
        ("database is busy", True),
        ("attempt to write a readonly database", False),
        ("no such table: research_runs", False),
        ("disk I/O error", False),
        ("unable to open database file", False),
    ],
)
def test_busy_predicate(message, busy):
    assert studio_db.is_sqlite_busy_error(sqlite3.OperationalError(message)) is busy


def test_api_usage_db_shares_one_definition():
    import storage.api_usage_db as api_usage_db
    assert api_usage_db._is_busy_error(sqlite3.OperationalError("database is locked"))
    assert not api_usage_db._is_busy_error(sqlite3.OperationalError("disk I/O error"))


class _Ladder:
    """The supervisor's except-ladder without its dependencies.

    Mirrors ResearchSupervisor._loop in core/research_runs.py; the control flow is the
    point, since the regression this guards was a control-flow one.
    """

    def __init__(self, raises, logger):
        self._raises = list(raises)
        self.logger = logger
        self.iterations = 0
        self.survived = False

    async def run(self):
        for exc in self._raises:
            self.iterations += 1
            try:
                if exc is not None:
                    raise exc
            except asyncio.CancelledError:
                raise
            except sqlite3.OperationalError as err:
                if studio_db.is_sqlite_busy_error(err):
                    self.logger.warning("research.supervisor_db_busy: %s", err)
                else:
                    self.logger.exception("research.supervisor_iteration_failed")
                await asyncio.sleep(0)
            except Exception:
                self.logger.exception("research.supervisor_iteration_failed")
                await asyncio.sleep(0)
        self.survived = True


def _levels(caplog):
    return [(record.levelname, record.exc_info is not None) for record in caplog.records]


def test_lock_contention_is_six_warnings_not_six_tracebacks(caplog):
    logger = logging.getLogger("test.supervisor.busy")
    ladder = _Ladder([sqlite3.OperationalError("database is locked")] * 6, logger)
    with caplog.at_level(logging.WARNING, logger = logger.name):
        asyncio.run(ladder.run())
    assert ladder.survived
    assert _levels(caplog) == [("WARNING", False)] * 6


def test_a_real_sqlite_fault_does_not_stop_the_supervisor(caplog):
    """Re-raising here would escape the while loop, because a sibling `except Exception`
    cannot catch a raise from its own ladder. The supervisor would stop for the life of
    the process, silently, which is worse than the log noise this change removes."""
    logger = logging.getLogger("test.supervisor.fault")
    ladder = _Ladder([sqlite3.OperationalError("no such table: research_runs"), None, None], logger)
    with caplog.at_level(logging.ERROR, logger = logger.name):
        asyncio.run(ladder.run())
    assert ladder.survived and ladder.iterations == 3
    assert _levels(caplog) == [("ERROR", True)], "and it keeps its traceback"


def test_unrelated_exceptions_are_unchanged(caplog):
    logger = logging.getLogger("test.supervisor.other")
    ladder = _Ladder([ValueError("boom"), None], logger)
    with caplog.at_level(logging.ERROR, logger = logger.name):
        asyncio.run(ladder.run())
    assert ladder.survived
    assert _levels(caplog) == [("ERROR", True)]


def test_cancellation_still_propagates():
    logger = logging.getLogger("test.supervisor.cancel")
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_Ladder([asyncio.CancelledError()], logger).run())


# --- expected client errors are not server faults ---------------------------------------


class _RecordingLogger:
    def __init__(self):
        self.calls = []

    def warning(self, *args, **kwargs):
        self.calls.append(("warning", kwargs))

    def error(self, *args, **kwargs):
        self.calls.append(("error", kwargs))


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422, 429])
def test_client_errors_log_one_warning_without_a_traceback(status):
    """One streamed generation put 54 rejected saves in the log, each with a full stack."""
    from utils.utils import log_and_http_error

    log = _RecordingLogger()
    error = log_and_http_error(ValueError("x"), status, "public", event = "e", log = log)
    assert error.status_code == status
    assert error.detail == "public", "the raw exception must never reach the client"
    assert log.calls == [("warning", {})]


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_server_errors_keep_their_traceback(status):
    from utils.utils import log_and_http_error

    raised = ValueError("x")
    log = _RecordingLogger()
    log_and_http_error(raised, status, "public", event = "e", log = log)
    level, kwargs = log.calls[0]
    assert level == "error" and kwargs.get("exc_info") is raised


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _short_lived_write(path: Path, value: str) -> None:
    """One writer with the lifetime every studio_db accessor gives its connection."""
    conn = studio_db.get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT OR REPLACE INTO app_settings (key, value_json, updated_at) "
            "VALUES ('wal-probe', ?, '0')",
            (value,),
        )
        conn.commit()
    finally:
        conn.close()


def test_wal_keeper_keeps_short_lived_writers_out_of_the_main_database(db):
    """The #9934 write amplification: without a keeper every close rewrites studio.db."""
    wal = Path(f"{db}-wal")

    for index in range(5):
        _short_lived_write(db, f"unkept-{index}")
        assert not wal.exists()
    unkept = _digest(db)

    assert studio_db.open_wal_keeper() is True
    for index in range(5):
        _short_lived_write(db, f"kept-{index}")
        assert wal.is_file()
    assert _digest(db) == unkept

    studio_db.close_wal_keeper()
    assert not wal.exists()
    assert _digest(db) != unkept


def _keeper(path):
    """The keeper held for one database, or None.

    Keepers are per database now: studio_db_path() is per workspace, so a single
    process-wide keeper attached to the owner's file left every managed account's
    writes checkpointing on close.
    """
    return studio_db._wal_keepers.get(str(Path(path).resolve()))


def test_a_second_database_gets_its_own_keeper(db, tmp_path, monkeypatch):
    """Each workspace database keeps its own WAL open, and neither displaces the other.

    A single keeper held whichever database was current when the lifespan ran, which
    is the owner's, so every managed account went back to checkpointing on close."""
    assert studio_db.open_wal_keeper() is True
    first = _keeper(db)
    assert first is not None

    second = tmp_path / "second" / "studio.db"
    second.parent.mkdir()
    monkeypatch.setattr(studio_db, "studio_db_path", lambda: second)
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    assert studio_db.open_wal_keeper() is True
    assert _keeper(second) is not None
    assert _keeper(second) is not first
    # The first database is still kept, not released to make room.
    assert _keeper(db) is first
    _short_lived_write(second, "kept")
    assert Path(f"{second}-wal").is_file()

    studio_db.close_wal_keeper()
    assert studio_db._wal_keepers == {}
    assert not Path(f"{second}-wal").exists()
    studio_db.close_wal_keeper()


def test_a_workspace_database_is_kept_without_a_second_lifespan(db, tmp_path, monkeypatch):
    """An account created after startup gets a keeper on its first write.

    The lifespan runs outside any request, so the only database it can name is the
    owner's; a managed account's is opened for the first time mid-request."""
    assert studio_db.open_wal_keeper() is True

    managed = tmp_path / "workspaces" / "alice-0123456789ab" / "studio.db"
    managed.parent.mkdir(parents = True)
    monkeypatch.setattr(studio_db, "studio_db_path", lambda: managed)
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    # No second open_wal_keeper(): just ordinary use of that workspace.
    _short_lived_write(managed, "kept")
    assert _keeper(managed) is not None
    assert Path(f"{managed}-wal").is_file()

    studio_db.close_wal_keeper()
    assert not Path(f"{managed}-wal").exists()


def test_no_keeper_is_engaged_before_the_lifespan_asks(db, tmp_path, monkeypatch):
    """Opening a connection must not leave one behind that nothing asked for."""
    studio_db.close_wal_keeper()

    managed = tmp_path / "workspaces" / "bob-0123456789ab" / "studio.db"
    managed.parent.mkdir(parents = True)
    monkeypatch.setattr(studio_db, "studio_db_path", lambda: managed)
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    _short_lived_write(managed, "unkept")
    assert _keeper(managed) is None
    assert not Path(f"{managed}-wal").exists()


def test_the_replaced_keeper_is_not_left_open(db):
    """Replacing must release the old connection, not merely drop the reference."""
    assert studio_db.open_wal_keeper() is True
    stale = _keeper(db)
    assert studio_db.open_wal_keeper() is True

    with pytest.raises(sqlite3.ProgrammingError, match = "closed database"):
        stale.execute("SELECT 1")
    studio_db.close_wal_keeper()


def test_a_keeper_left_by_a_dead_thread_is_replaced(db):
    """sqlite refuses a cross-thread close, so replacing has to survive that failing."""
    opened = threading.Thread(target = studio_db.open_wal_keeper)
    opened.start()
    opened.join()
    stale = _keeper(db)
    assert stale is not None

    assert studio_db.open_wal_keeper() is True
    assert _keeper(db) is not stale
    _short_lived_write(db, "kept")
    assert Path(f"{db}-wal").is_file()

    studio_db.close_wal_keeper()
    assert not Path(f"{db}-wal").exists()


def test_wal_keeper_declines_when_the_filesystem_refused_wal(db, caplog):
    """A rollback-journal install has nothing to hold open, and must still boot."""
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.close()
    assert _journal_mode(db) == "delete"

    with caplog.at_level(logging.INFO, logger = studio_db.logger.name):
        assert studio_db.open_wal_keeper() is False
    assert _keeper(db) is None
    assert "not WAL" in caplog.text


def test_the_lifespan_holds_the_keeper_across_every_writer():
    """A keeper nothing opens saves nothing, and one released early stops saving early.

    Reads the source rather than running the lifespan, which imports the whole stack.
    Anchored on the awaited call, since the bare name is also in a comment further up.
    """
    import main

    source = inspect.getsource(main.lifespan)
    served = source.index("yield")
    # The cleanup is dispatched per workspace, so the call is `run_in_workspace(
    # _subject, cleanup_orphaned_runs)` and the bare-name-plus-parens form is gone.
    assert (
        source.index("open_wal_keeper()")
        < source.index("run_in_workspace(_subject, cleanup_orphaned_runs)")
        < served
    )
    assert (
        served < source.index("await run_lifespan_shutdown(") < source.index("close_wal_keeper()")
    )
