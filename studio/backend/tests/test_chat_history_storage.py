# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import threading

import pytest

from storage import studio_db


def _reset_studio_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)


def _thread(thread_id: str = "thread-1") -> dict:
    return {
        "id": thread_id,
        "title": "Test Chat",
        "modelType": "base",
        "modelId": "test-model",
        "pairId": None,
        "archived": False,
        "createdAt": 1_700_000_000_000,
    }


def _message(
    message_id: str,
    created_at: int,
    content: str,
    thread_id: str = "thread-1",
) -> dict:
    return {
        "id": message_id,
        "threadId": thread_id,
        "parentId": None,
        "role": "user",
        "content": [{"type": "text", "text": content}],
        "createdAt": created_at,
    }


def test_sync_chat_messages_upserts_without_pruning(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    studio_db.sync_chat_messages(
        "thread-1",
        [
            _message("msg-1", 1, "keep me"),
            _message("msg-2", 2, "old text"),
        ],
        prune_missing = True,
    )

    messages = studio_db.sync_chat_messages(
        "thread-1",
        [_message("msg-2", 2, "updated text")],
    )

    by_id = {message["id"]: message for message in messages}
    assert set(by_id) == {"msg-1", "msg-2"}
    assert by_id["msg-2"]["content"] == [{"type": "text", "text": "updated text"}]


def test_sync_chat_messages_prunes_when_requested(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    studio_db.sync_chat_messages(
        "thread-1",
        [
            _message("msg-1", 1, "delete me"),
            _message("msg-2", 2, "keep me"),
        ],
    )

    messages = studio_db.sync_chat_messages(
        "thread-1",
        [_message("msg-2", 2, "keep me")],
        prune_missing = True,
    )

    assert [message["id"] for message in messages] == ["msg-2"]


def test_upsert_chat_message_rejects_cross_thread_id_conflict(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread("thread-1"))
    studio_db.upsert_chat_thread(_thread("thread-2"))
    studio_db.upsert_chat_message(_message("msg-1", 1, "original", "thread-1"))

    with pytest.raises(studio_db.ChatMessageConflictError):
        studio_db.upsert_chat_message(_message("msg-1", 2, "moved", "thread-2"))

    assert [m["id"] for m in studio_db.list_chat_messages("thread-1")] == ["msg-1"]
    assert studio_db.list_chat_messages("thread-2") == []


def test_sync_chat_messages_detects_conflict_before_prune(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread("thread-1"))
    studio_db.upsert_chat_thread(_thread("thread-2"))
    studio_db.sync_chat_messages(
        "thread-1",
        [_message("keep-me", 1, "keep", "thread-1")],
    )
    studio_db.upsert_chat_message(_message("conflict", 2, "other", "thread-2"))

    with pytest.raises(studio_db.ChatMessageConflictError):
        studio_db.sync_chat_messages(
            "thread-1",
            [_message("conflict", 3, "bad", "thread-1")],
            prune_missing = True,
        )

    assert [m["id"] for m in studio_db.list_chat_messages("thread-1")] == ["keep-me"]
    assert [m["id"] for m in studio_db.list_chat_messages("thread-2")] == ["conflict"]


def test_settings_merge_atomic_under_concurrency(tmp_path, monkeypatch):
    """Two threads writing distinct keys must not drop each other's update."""
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_settings_merge({"inferenceParams": {}})

    barrier = threading.Barrier(2)

    def writer(key: str, value: float) -> None:
        barrier.wait()
        studio_db.upsert_chat_settings_merge({"inferenceParams": {key: value}})

    t1 = threading.Thread(target = writer, args = ("temperature", 0.7))
    t2 = threading.Thread(target = writer, args = ("topP", 0.9))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    merged = studio_db.list_chat_settings()["inferenceParams"]
    assert merged.get("temperature") == 0.7
    assert merged.get("topP") == 0.9


def test_settings_merge_preserves_nested_keys(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_settings_merge(
        {"inferenceParams": {"temperature": 0.5, "topP": 0.8}}
    )
    studio_db.upsert_chat_settings_merge({"inferenceParams": {"temperature": 0.9}})

    params = studio_db.list_chat_settings()["inferenceParams"]
    assert params == {"temperature": 0.9, "topP": 0.8}


def test_settings_merge_quarantines_corrupt_json_and_rejects_partial_patch(
    tmp_path,
    monkeypatch,
):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_settings_merge(
        {"inferenceParams": {"temperature": 0.5, "topP": 0.8}}
    )
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE chat_settings SET value_json = ? WHERE key = ?",
            ('{"temperature": 0.5', "inferenceParams"),
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(studio_db.CorruptSettingsError):
        studio_db.upsert_chat_settings_merge({"inferenceParams": {"temperature": 0.9}})

    conn = studio_db.get_connection()
    try:
        quarantined = conn.execute(
            "SELECT key, value_json, reason FROM chat_settings_quarantine"
        ).fetchall()
        remaining = conn.execute(
            "SELECT key FROM chat_settings WHERE key = ?",
            ("inferenceParams",),
        ).fetchall()
    finally:
        conn.close()
    assert [row["key"] for row in quarantined] == ["inferenceParams"]
    assert quarantined[0]["reason"] == "json_decode_error"
    assert remaining == []


def test_settings_merge_replaces_corrupt_scalar_after_quarantine(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_settings_merge({"autoTitle": False})
    conn = studio_db.get_connection()
    try:
        conn.execute(
            "UPDATE chat_settings SET value_json = ? WHERE key = ?",
            ("not-json", "autoTitle"),
        )
        conn.commit()
    finally:
        conn.close()

    settings = studio_db.upsert_chat_settings_merge({"autoTitle": True})

    assert settings["autoTitle"] is True
    conn = studio_db.get_connection()
    try:
        quarantined = conn.execute(
            "SELECT key, reason FROM chat_settings_quarantine"
        ).fetchall()
    finally:
        conn.close()
    assert [(row["key"], row["reason"]) for row in quarantined] == [
        ("autoTitle", "json_decode_error")
    ]


def test_list_chat_messages_for_threads_chunks_over_900_ids(tmp_path, monkeypatch):
    """SQLite host-parameter limit is 999 on older builds; chunk at 900."""
    _reset_studio_db(tmp_path, monkeypatch)
    n = 901
    for i in range(n):
        studio_db.upsert_chat_thread(
            {
                "id": f"t-{i}",
                "title": "T",
                "modelType": "base",
                "modelId": "m",
                "pairId": None,
                "archived": False,
                "createdAt": 1_700_000_000_000 + i,
            }
        )
        studio_db.upsert_chat_message(
            {
                "id": f"m-{i}",
                "threadId": f"t-{i}",
                "parentId": None,
                "role": "user",
                "content": [{"type": "text", "text": "hi"}],
                "createdAt": 1_700_000_000_000 + i,
            }
        )
    out = studio_db.list_chat_messages_for_threads([f"t-{i}" for i in range(n)])
    assert len(out) == n
    assert {m["threadId"] for m in out} == {f"t-{i}" for i in range(n)}


# ---------------------------------------------------------------------------
# Legacy Dexie import ledger
# ---------------------------------------------------------------------------


def test_legacy_imports_empty_by_default(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    assert studio_db.list_chat_legacy_imports() == []


def test_legacy_imports_records_and_lists(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    accepted, inserted = studio_db.upsert_chat_legacy_imports(
        ["legacy-a", "legacy-b", "legacy-c"],
    )
    assert accepted == 3
    assert inserted == 3
    assert set(studio_db.list_chat_legacy_imports()) == {
        "legacy-a",
        "legacy-b",
        "legacy-c",
    }


def test_legacy_imports_is_idempotent(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    accepted1, inserted1 = studio_db.upsert_chat_legacy_imports(
        ["legacy-a", "legacy-b"],
    )
    accepted2, inserted2 = studio_db.upsert_chat_legacy_imports(
        ["legacy-b", "legacy-c"],
    )
    assert (accepted1, inserted1) == (2, 2)
    # legacy-b is already in the ledger, only legacy-c is genuinely new.
    assert (accepted2, inserted2) == (2, 1)
    assert set(studio_db.list_chat_legacy_imports()) == {
        "legacy-a",
        "legacy-b",
        "legacy-c",
    }


def test_legacy_imports_dedups_input(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    accepted, inserted = studio_db.upsert_chat_legacy_imports(
        ["x", "x", "y", "x"],
    )
    # accepted is the deduped non-empty input size; inserted is the rows
    # actually new in the ledger after ON CONFLICT DO NOTHING.
    assert accepted == 2
    assert inserted == 2
    assert set(studio_db.list_chat_legacy_imports()) == {"x", "y"}


def test_legacy_imports_ignores_empty(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    assert studio_db.upsert_chat_legacy_imports([]) == (0, 0)
    assert studio_db.upsert_chat_legacy_imports(["", None]) == (0, 0)  # type: ignore[list-item]
    assert studio_db.list_chat_legacy_imports() == []
