# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from storage import chat_generation_runs_db as runs
from storage import research_runs_db as research
from storage import studio_db as db

PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aS1sAAAAASUVORK5CYII="


@pytest.fixture(autouse = True)
def studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    db.close_wal_keeper()
    db.reset_schema_state_for_tests()
    runs.reset_schema_state_for_tests()
    for name in ("history", "new"):
        db.upsert_chat_thread({"id": name, "modelType": "gguf", "createdAt": 1})
    db.upsert_chat_message(
        {
            "id": "image",
            "threadId": "history",
            "role": "user",
            "content": [{"type": "image", "image": PNG}],
            "createdAt": 1,
        }
    )
    db.upsert_chat_message(
        {
            "id": "user",
            "threadId": "new",
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
            "createdAt": 1,
        }
    )
    yield
    db.close_wal_keeper()


def _query(statement, parameters = ()):
    conn = db.get_connection()
    try:
        return conn.execute(statement, parameters).fetchall()
    finally:
        conn.close()


def test_every_foreign_key_child_column_leads_an_index():
    conn = db.get_connection()
    try:
        tables = [
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        ]
        missing = []
        for table in tables:
            leading = {
                conn.execute(f"PRAGMA index_info('{index[1]}')").fetchone()[2]
                for index in conn.execute(f"PRAGMA index_list('{table}')").fetchall()
            }
            for fk in conn.execute(f"PRAGMA foreign_key_list('{table}')").fetchall():
                if fk[3] not in leading:
                    missing.append(f"{table}.{fk[3]}")
    finally:
        conn.close()
    assert missing == [], f"cascading deletes full-scan these tables: {missing}"


def test_schema_is_idempotent():
    before = _query("SELECT type, name, sql FROM sqlite_master ORDER BY type, name")
    conn = db.get_connection()
    try:
        db._ensure_schema(conn)
        conn.commit()
    finally:
        conn.close()
    assert _query("SELECT type, name, sql FROM sqlite_master ORDER BY type, name") == before


def test_lease_sweep_reads_only_active_runs():
    runs.reconcile_runs(stale_after_ms = 60_000)
    plan = _query(
        "EXPLAIN QUERY PLAN SELECT id, status, cancel_requested FROM chat_generation_runs "
        "WHERE status IN ('queued','running','cancelling') "
        "AND COALESCE(progress_at, started_at, created_at) <= ? ORDER BY created_at, id",
        (0,),
    )
    detail = " ".join(str(row[3]) for row in plan)
    assert "idx_chat_generation_runs_active" in detail
    assert "TEMP B-TREE" not in detail


def test_research_placeholder_keeps_a_clean_inventory_clean(monkeypatch):
    before = db.list_chat_attachments_page()[0]
    calls = []
    real = db._rebuild_chat_attachment_inventory
    monkeypatch.setattr(
        db, "_rebuild_chat_attachment_inventory", lambda conn: calls.append(True) or real(conn)
    )
    research.create_run(
        run_id = "run-1",
        owner_subject = "alice",
        thread_id = "new",
        user_message_id = "user",
        assistant_message_id = "assistant",
        config = {"model": "local-model", "inferenceRequest": {"model": "local-model"}},
    )
    state = _query("SELECT dirty FROM chat_attachment_inventory_state WHERE singleton=1")
    assert state[0][0] == 0
    db.sync_chat_messages("new", db.list_chat_messages("new"))
    assert db.list_chat_attachments_page()[0] == before
    assert calls == [], "starting Deep Research caused a full history rebuild"


def test_research_placeholder_keeps_a_dirty_inventory_dirty():
    db.list_chat_attachments_page()
    conn = db.get_connection()
    try:
        conn.execute("UPDATE chat_attachment_inventory_state SET dirty=1 WHERE singleton=1")
        conn.commit()
    finally:
        conn.close()
    research.create_run(
        run_id = "run-1",
        owner_subject = "alice",
        thread_id = "new",
        user_message_id = "user",
        assistant_message_id = "assistant",
        config = {"model": "local-model", "inferenceRequest": {"model": "local-model"}},
    )
    state = _query("SELECT dirty FROM chat_attachment_inventory_state WHERE singleton=1")
    assert state[0][0] == 1
    assert [item["messageId"] for item in db.list_chat_attachments_page()[0]] == ["image"]
