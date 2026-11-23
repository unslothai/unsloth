# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Starting a generation must not make autosave reindex unrelated attachments."""

import json

import pytest

from storage import chat_generation_runs_db as runs
from storage import studio_db as db
from utils.account_context import AccountContext, run_as

PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aS1sAAAAASUVORK5CYII="


@pytest.fixture(autouse = True)
def chat_home(tmp_path, monkeypatch):
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


def create():
    return runs.create_run(
        run_id = "run",
        owner_subject = "test",
        thread_id = "new",
        user_message_id = "user",
        assistant_message_id = "assistant",
        request_payload = {},
    )


def inventory():
    return db.list_chat_attachments_page()[0]


def state():
    conn = db.get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM chat_attachment_inventory_state WHERE singleton=1"
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


def sql(statement, parameters = ()):
    conn = db.get_connection()
    try:
        conn.execute(statement, parameters)
        conn.commit()
    finally:
        conn.close()


def spy_rebuild(monkeypatch):
    calls = []
    real = db._rebuild_chat_attachment_inventory

    def rebuild(conn):
        calls.append(True)
        return real(conn)

    monkeypatch.setattr(db, "_rebuild_chat_attachment_inventory", rebuild)
    return calls


def test_new_empty_assistant_does_not_rebuild_history_on_next_autosave(monkeypatch):
    before = inventory()
    calls = spy_rebuild(monkeypatch)
    create()
    assistant = db.get_chat_message("new", "assistant")
    assistant["metadata"]["generationStatus"] = "queued"
    db.upsert_chat_message(assistant)
    assert inventory() == before
    assert calls == [], "starting a generation caused a full history rebuild"


def test_existing_dirty_inventory_is_not_marked_clean():
    inventory()
    attachment = [
        {
            "id": "legacy-file",
            "name": "notes.txt",
            "type": "file",
            "content": [{"type": "text", "text": "legacy text"}],
        }
    ]
    sql(
        "INSERT INTO chat_messages (id,thread_id,role,content_json,attachments_json,created_at) "
        "VALUES ('legacy','history','user','[]',?,2)",
        (json.dumps(attachment),),
    )
    assert state()["dirty"] == 1
    create()
    assert state()["dirty"] == 1
    assert {item["messageId"] for item in inventory()} == {"image", "legacy"}


def test_missing_inventory_state_still_backfills():
    sql("DELETE FROM chat_attachment_inventory_state")
    sql("DELETE FROM chat_attachment_inventory")
    create()
    assert state()["dirty"] == 1
    assert [item["messageId"] for item in inventory()] == ["image"]


def test_old_inventory_version_is_not_advanced(monkeypatch):
    sql("UPDATE chat_attachment_inventory_state SET inventory_version=0, dirty=0")
    calls = spy_rebuild(monkeypatch)
    create()
    assert state()["inventory_version"] == 0
    assert [item["messageId"] for item in inventory()] == ["image"]
    assert len(calls) == 1, "an old inventory version must still be rebuilt"


def test_binding_existing_placeholder_keeps_its_attachments(monkeypatch):
    attachments = [
        {"id": "draft-file", "name": "draft.txt", "content": [{"type": "text", "text": "draft"}]}
    ]
    db.upsert_chat_message(
        {
            "id": "assistant",
            "threadId": "new",
            "parentId": "user",
            "role": "assistant",
            "content": [],
            "attachments": attachments,
            "createdAt": 2,
        }
    )
    before = inventory()
    calls = spy_rebuild(monkeypatch)
    create()
    assert db.get_chat_message("new", "assistant")["attachments"] == attachments
    assert inventory() == before
    assert calls == []


def test_real_attachment_changes_still_invalidate_the_inventory():
    create()
    inventory()
    sql("UPDATE chat_messages SET content_json='[]' WHERE id='image'")
    assert state()["dirty"] == 1
    assert inventory() == []


@pytest.mark.parametrize("already_dirty", [False, True])
def test_failed_generation_rolls_back_placeholder_and_inventory_state(monkeypatch, already_dirty):
    before = inventory()
    if already_dirty:
        sql("UPDATE chat_messages SET content_json='[]' WHERE id='image'")
    before_state = state()

    def fail_event_write(*args, **kwargs):
        raise RuntimeError("event write failed")

    monkeypatch.setattr(runs, "_append_events_locked", fail_event_write)
    with pytest.raises(RuntimeError, match = "event write failed"):
        create()
    assert state() == before_state
    assert db.get_chat_message("new", "assistant") is None
    assert runs.get_run("run") is None
    assert inventory() == ([] if already_dirty else before)


def test_creating_one_accounts_placeholder_does_not_clear_another_accounts_dirty_index():
    alice = AccountContext("11111111111111111111111111111111", "alice")
    bob = AccountContext("22222222222222222222222222222222", "bob")

    def seed_account(name):
        for thread in ("history", "new"):
            db.upsert_chat_thread({"id": thread, "modelType": "gguf", "createdAt": 1})
        db.upsert_chat_message(
            {"id": "user", "threadId": "new", "role": "user", "content": [], "createdAt": 1}
        )
        db.upsert_chat_message(
            {
                "id": "file",
                "threadId": "history",
                "role": "user",
                "content": [],
                "attachments": [{"id": name, "name": f"{name}.txt"}],
                "createdAt": 2,
            }
        )

    run_as(alice, seed_account, "alice-file")
    run_as(bob, seed_account, "bob-file")
    alice_before = run_as(alice, inventory)
    run_as(
        bob,
        sql,
        "UPDATE chat_messages SET attachments_json=? WHERE id='file'",
        (json.dumps([{"id": "bob-replacement", "name": "replacement.txt"}]),),
    )
    assert run_as(bob, state)["dirty"] == 1
    run_as(alice, create)
    assert run_as(alice, state)["dirty"] == 0
    assert run_as(alice, inventory) == alice_before
    assert run_as(bob, state)["dirty"] == 1
    run_as(bob, create)
    assert run_as(bob, state)["dirty"] == 1
    assert [item["id"] for item in run_as(bob, inventory)] == ["bob-replacement"]
    assert run_as(alice, inventory) == alice_before
