# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The per-thread settings snapshot that makes a chat keep its own modes.

Covers the contract PATCH /api/chat/threads/{id} accepts and the storage rules the
snapshot depends on: writers that rebuild a thread record must not clear it, the thread
listing must not carry it, and a fork must inherit it.
"""

import sqlite3
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from routes.chat_history import ChatThreadSettings  # noqa: E402
from storage import studio_db  # noqa: E402
from utils.paths import studio_db_path  # noqa: E402

SETTINGS = {
    "toolsEnabled": True,
    "codeToolsEnabled": False,
    "deepResearchEnabled": False,
    "permissionMode": "auto",
    "ragEnabled": True,
    "ragSource": {"type": "kb", "kbId": "notes"},
    "ragMode": "dense",
    "ragTopK": 12,
    "ragAutoInject": "on",
    "ragAutoInjectMinScore": 0.42,
    "reasoningEffort": "high",
}


def _reset_studio_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)


def _thread(thread_id: str = "thread-1", **extra) -> dict:
    return {
        "id": thread_id,
        "title": "Test Chat",
        "modelType": "base",
        "modelId": "test-model",
        "pairId": None,
        "archived": False,
        "createdAt": 1_700_000_000_000,
        **extra,
    }


def test_thread_settings_round_trip():
    settings = ChatThreadSettings.model_validate(SETTINGS)
    assert settings.model_dump(exclude_unset = True) == SETTINGS


@pytest.mark.parametrize(
    "field, value",
    [
        # full access disables the sandbox, so it stays session-only per thread too.
        ("permissionMode", "full"),
        ("ragTopK", 51),
        ("ragAutoInjectMinScore", 1.5),
        ("ragMode", "vector"),
        # global-only settings must not reach a thread and silently become per-chat.
        ("gpuMemoryMode", "manual"),
        ("showCanvasMenuItem", True),
    ],
)
def test_thread_settings_rejects_out_of_contract(field, value):
    with pytest.raises(ValidationError):
        ChatThreadSettings.model_validate({field: value})


def test_thread_settings_survive_a_record_rewrite(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread(settings = SETTINGS))

    # the title autosave and every other writer rebuild the record without the snapshot.
    studio_db.upsert_chat_thread(_thread(title = "Renamed"))

    stored = studio_db.get_chat_thread("thread-1")
    assert stored["title"] == "Renamed"
    assert stored["settings"] == SETTINGS


def test_thread_settings_patch_replaces_the_snapshot(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread(settings = SETTINGS))

    # ragSource is a discriminated union, so the write replaces: no kb id survives the switch.
    replacement = {"toolsEnabled": False, "ragSource": {"type": "thread"}}
    updated = studio_db.update_chat_thread("thread-1", {"settings": replacement})

    assert updated["settings"] == replacement
    assert studio_db.get_chat_thread("thread-1")["settings"] == replacement


def test_thread_listing_leaves_out_the_snapshot(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread(settings = SETTINGS))

    listed = studio_db.list_chat_threads()

    assert [row["id"] for row in listed] == ["thread-1"]
    assert "settings" not in listed[0]


def test_thread_without_a_snapshot_reads_back_null(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())

    assert studio_db.get_chat_thread("thread-1")["settings"] is None


def test_fork_inherits_the_snapshot(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread(settings = SETTINGS))
    studio_db.upsert_chat_message(
        {
            "id": "message-1",
            "threadId": "thread-1",
            "parentId": None,
            "role": "user",
            "content": "hello",
            "createdAt": 1_700_000_000_001,
        }
    )

    forked = studio_db.fork_chat_thread(
        source_thread_id = "thread-1",
        branch_message_id = "message-1",
        new_thread_id = "thread-2",
        new_title = "Fork",
        created_at = 1_700_000_000_002,
        id_factory = lambda: "message-2",
    )

    assert forked is not None
    assert studio_db.get_chat_thread("thread-2")["settings"] == SETTINGS


def test_settings_column_is_added_to_an_existing_database(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    # a database created before this change has no settings_json, and the CREATE TABLE in
    # _ensure_schema never runs for it, so only the ALTER keeps thread writes working.
    db_path = Path(studio_db_path())
    db_path.parent.mkdir(parents = True, exist_ok = True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE chat_threads (
                id TEXT NOT NULL PRIMARY KEY,
                title TEXT NOT NULL,
                model_type TEXT NOT NULL,
                model_id TEXT,
                pair_id TEXT,
                archived INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE chat_messages (
                id TEXT NOT NULL PRIMARY KEY,
                thread_id TEXT NOT NULL,
                parent_id TEXT,
                role TEXT NOT NULL,
                content_json TEXT NOT NULL,
                attachments_json TEXT,
                metadata_json TEXT,
                created_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO chat_threads (id, title, model_type, created_at) VALUES (?, ?, ?, ?)",
            ("thread-1", "Old", "base", 1_700_000_000_000),
        )
        conn.commit()
    finally:
        conn.close()

    assert studio_db.get_chat_thread("thread-1")["settings"] is None
    studio_db.upsert_chat_thread(_thread(settings = SETTINGS))
    assert studio_db.get_chat_thread("thread-1")["settings"] == SETTINGS
