# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""studio.db must not turn one slow writer into a stream of "database is locked".

A live Colab session logged six `research.supervisor_iteration_failed` tracebacks in 37
seconds while a settings write and a multi-GB model download shared the disk. Each guards a
distinct link in that chain.
"""

import sqlite3

import pytest

import storage.research_runs_db as research_runs_db
import storage.studio_db as studio_db


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A studio.db in a temp dir, with the per-process schema latch reset."""
    monkeypatch.setattr(studio_db, "studio_db_path", lambda: tmp_path / "studio.db")
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    conn = studio_db.get_connection()
    conn.close()
    return tmp_path / "studio.db"


def test_connections_use_wal_with_normal_sync(db):
    """synchronous=FULL fsyncs under the writer lock; NORMAL is the WAL-safe pairing."""
    conn = studio_db.get_connection()
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        # 1 == NORMAL. FULL (2) is what made a single commit block for 37s on a busy disk.
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1
    finally:
        conn.close()


def test_claim_next_takes_no_write_lock_when_idle(db):
    """The supervisor polls twice a second forever; an empty poll must stay a read.

    Holding the writer lock elsewhere and calling claim_next reproduces the original
    failure exactly: it used to raise OperationalError after the 5s busy timeout.
    """
    holder = studio_db.get_connection()
    try:
        holder.execute("BEGIN IMMEDIATE")
        holder.execute(
            "INSERT INTO app_settings (key, value_json, updated_at) VALUES ('probe', '1', '0') "
            "ON CONFLICT(key) DO UPDATE SET value_json = '1'"
        )
        # No commit: the writer lock stays held for the duration of this call.
        assert research_runs_db.claim_next("worker-1") is None
    finally:
        holder.rollback()
        holder.close()


def test_metadata_only_update_does_not_dirty_attachment_inventory(db):
    """A generation status change must not schedule a full attachment rebuild.

    The rebuild re-hashes every attachment in every thread inside BEGIN IMMEDIATE, so
    dirtying it from an unrelated column is what made an ordinary autosave hold the lock.
    """
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "INSERT INTO chat_threads (id, title, model_type, created_at, updated_at) "
            "VALUES ('t1', 'T', 'gguf', 0, 0)"
        )
        conn.execute(
            "INSERT INTO chat_messages "
            "(id, thread_id, role, content_json, attachments_json, metadata_json, created_at) "
            "VALUES ('m1', 't1', 'assistant', '[]', '[]', '{}', 0)"
        )
        conn.commit()
        studio_db._mark_chat_attachment_inventory_clean(conn)
        conn.commit()

        def dirty() -> int:
            row = conn.execute(
                "SELECT dirty FROM chat_attachment_inventory_state WHERE singleton = 1"
            ).fetchone()
            return int(row["dirty"])

        assert dirty() == 0
        # What chat_generation_runs_db does on every status transition.
        conn.execute("UPDATE chat_messages SET metadata_json = '{\"a\":1}' WHERE id = 'm1'")
        conn.commit()
        assert dirty() == 0, "a metadata-only update must not dirty the inventory"

        # A real attachment change still must.
        conn.execute("UPDATE chat_messages SET attachments_json = '[{}]' WHERE id = 'm1'")
        conn.commit()
        assert dirty() == 1
    finally:
        conn.close()


class _RecordingLogger:
    """The module logger is structlog, so caplog cannot see it; log_and_http_error
    accepts an explicit one, which is what the routes pass anyway."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def warning(self, *args, **kwargs):
        self.calls.append(("warning", kwargs))

    def error(self, *args, **kwargs):
        self.calls.append(("error", kwargs))


def test_client_errors_log_without_a_traceback():
    """A 409 is the caller's business, not a server fault: one warning line, no stack.

    Logging it at error with exc_info put 54 full tracebacks in a single session's log.
    """
    from utils.utils import log_and_http_error

    error = ValueError("server-managed generation messages cannot be edited")

    log = _RecordingLogger()
    exc = log_and_http_error(error, 409, "Conflict", event="chat.conflict", log=log)
    assert exc.status_code == 409
    assert exc.detail == "Conflict", "the raw exception text must not reach the client"
    assert log.calls == [("warning", {})], "4xx: one warning, no exc_info"

    log = _RecordingLogger()
    log_and_http_error(error, 500, "Boom", event="server.broke", log=log)
    level, kwargs = log.calls[0]
    assert level == "error" and kwargs.get("exc_info") is error, "5xx keeps its traceback"


def test_claim_next_still_claims_real_work(db):
    """The read-first probe is an optimisation, not a suppression: a claimable run must
    still be claimed, or the supervisor would quietly stop doing its job."""
    # Through the modules' own APIs, so the fixture cannot drift from the real schema.
    studio_db.upsert_chat_thread(
        {"id": "t1", "title": "T", "modelType": "gguf", "createdAt": 1}
    )
    studio_db.upsert_chat_message(
        {"id": "u1", "threadId": "t1", "role": "user", "content": [], "createdAt": 2}
    )
    studio_db.upsert_chat_message(
        {
            "id": "a1",
            "threadId": "t1",
            "parentId": "u1",
            "role": "assistant",
            "content": [],
            "createdAt": 3,
        }
    )
    research_runs_db.create_run(
        run_id="r1",
        owner_subject="sub",
        thread_id="t1",
        user_message_id="u1",
        assistant_message_id="a1",
        config={},
    )

    assert research_runs_db._has_claimable(research_runs_db.now_ms()) is True
    claimed = research_runs_db.claim_next("worker-1")
    assert claimed is not None and claimed["id"] == "r1"
    # Claimed once: the lease is now held, so the next poll is idle again.
    assert research_runs_db.claim_next("worker-2") is None
