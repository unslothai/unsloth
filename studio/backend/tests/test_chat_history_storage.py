# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import threading

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


def _message(message_id: str, created_at: int, content: str) -> dict:
    return {
        "id": message_id,
        "threadId": "thread-1",
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


def test_settings_merge_atomic_under_concurrency(tmp_path, monkeypatch):
    """Two threads writing distinct keys must not drop each other's update."""
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_settings_merge({"inferenceParams": {}})

    barrier = threading.Barrier(2)

    def writer(key: str, value: float) -> None:
        barrier.wait()
        studio_db.upsert_chat_settings_merge(
            {"inferenceParams": {key: value}}
        )

    t1 = threading.Thread(target = writer, args = ("temperature", 0.7))
    t2 = threading.Thread(target = writer, args = ("topP", 0.9))
    t1.start(); t2.start()
    t1.join(); t2.join()

    merged = studio_db.list_chat_settings()["inferenceParams"]
    assert merged.get("temperature") == 0.7
    assert merged.get("topP") == 0.9


def test_settings_merge_preserves_nested_keys(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_settings_merge(
        {"inferenceParams": {"temperature": 0.5, "topP": 0.8}}
    )
    studio_db.upsert_chat_settings_merge(
        {"inferenceParams": {"temperature": 0.9}}
    )

    params = studio_db.list_chat_settings()["inferenceParams"]
    assert params == {"temperature": 0.9, "topP": 0.8}


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
