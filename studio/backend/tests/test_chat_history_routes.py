# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import os
import sys

import pytest
from fastapi import HTTPException

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from routes import chat_history


def _message(message_id: str, thread_id: str) -> chat_history.ChatMessage:
    return chat_history.ChatMessage(
        id = message_id,
        threadId = thread_id,
        parentId = None,
        role = "user",
        content = [{"type": "text", "text": "hello"}],
        createdAt = 1_700_000_000_000,
    )


def test_replace_thread_messages_rejects_body_thread_mismatch(monkeypatch):
    called = False

    def fake_get_chat_thread(thread_id: str):
        return {"id": thread_id}

    def fake_sync_chat_messages(*args, **kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(chat_history, "get_chat_thread", fake_get_chat_thread)
    monkeypatch.setattr(chat_history, "sync_chat_messages", fake_sync_chat_messages)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            chat_history.replace_thread_messages(
                "thread-1",
                chat_history.ChatMessageSyncRequest(
                    messages = [_message("msg-1", "thread-2")],
                    pruneMissing = True,
                ),
                current_subject = "test-user",
            )
        )

    assert exc_info.value.status_code == 400
    assert "Message threadId mismatch" in str(exc_info.value.detail)
    assert called is False


# ---------------------------------------------------------------------------
# /api/chat/import-ledger
# ---------------------------------------------------------------------------


def test_get_import_ledger_round_trips_through_storage(monkeypatch):
    seen: list[str] = []

    def fake_list():
        return list(seen)

    monkeypatch.setattr(chat_history, "list_chat_legacy_imports", fake_list)

    response = asyncio.run(chat_history.get_import_ledger(current_subject = "test-user"))
    assert response.threadIds == []

    seen.extend(["legacy-a", "legacy-b"])
    response = asyncio.run(chat_history.get_import_ledger(current_subject = "test-user"))
    assert response.threadIds == ["legacy-a", "legacy-b"]


def test_record_import_ledger_returns_accepted_and_inserted(monkeypatch):
    captured: list[list[str]] = []

    def fake_upsert(thread_ids):
        captured.append(list(thread_ids))
        # Pretend two of the three were already in the ledger.
        return (len(thread_ids), max(0, len(thread_ids) - 2))

    monkeypatch.setattr(chat_history, "upsert_chat_legacy_imports", fake_upsert)

    response = asyncio.run(
        chat_history.record_import_ledger(
            payload = chat_history.ChatImportLedgerRecordRequest(
                threadIds = ["a", "b", "c"],
            ),
            current_subject = "test-user",
        )
    )
    assert response.accepted == 3
    assert response.inserted == 1
    assert captured == [["a", "b", "c"]]


def test_record_import_ledger_rejects_oversize_payload():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        chat_history.ChatImportLedgerRecordRequest(
            threadIds = [f"id-{i}" for i in range(10_001)],
        )


# ---------------------------------------------------------------------------
# /api/chat/threads/{id}/fork
# ---------------------------------------------------------------------------


def test_fork_thread_404_when_source_missing(monkeypatch):
    monkeypatch.setattr(chat_history, "get_chat_thread", lambda _id: None)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            chat_history.fork_thread(
                thread_id = "missing",
                payload = chat_history.ChatForkRequest(
                    messageId = "m1",
                    newThreadId = "new",
                    createdAt = 1,
                ),
                current_subject = "test-user",
            )
        )
    assert exc.value.status_code == 404


def test_fork_thread_404_when_branch_message_missing(monkeypatch):
    monkeypatch.setattr(
        chat_history, "get_chat_thread", lambda _id: {"id": _id, "title": "T"}
    )
    monkeypatch.setattr(chat_history, "get_chat_message", lambda _t, _m: None)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            chat_history.fork_thread(
                thread_id = "src",
                payload = chat_history.ChatForkRequest(
                    messageId = "missing",
                    newThreadId = "new",
                    createdAt = 1,
                ),
                current_subject = "test-user",
            )
        )
    assert exc.value.status_code == 404


def test_fork_thread_happy_path(monkeypatch):
    source = {
        "id": "src",
        "title": "Original",
        "modelType": "base",
        "modelId": "m",
        "pairId": None,
        "archived": False,
        "createdAt": 1,
        "openaiCodeExecContainerId": None,
        "anthropicCodeExecContainerId": None,
        "forkedFromThreadId": None,
        "forkedFromMessageId": None,
    }
    forked = {
        **source,
        "id": "new",
        "title": "fork · Original",
        "createdAt": 2,
        "forkedFromThreadId": "src",
        "forkedFromMessageId": "m1",
    }
    monkeypatch.setattr(chat_history, "get_chat_thread", lambda _id: source)
    monkeypatch.setattr(
        chat_history,
        "get_chat_message",
        lambda _t, _m: {
            "id": _m,
            "threadId": _t,
            "role": "user",
            "content": [],
            "createdAt": 1,
        },
    )
    monkeypatch.setattr(chat_history, "fork_chat_thread", lambda **_: forked)
    monkeypatch.setattr(
        chat_history,
        "list_chat_messages",
        lambda _id: [
            {
                "id": "n1",
                "threadId": "new",
                "parentId": None,
                "role": "user",
                "content": [],
                "createdAt": 1,
            }
        ],
    )
    response = asyncio.run(
        chat_history.fork_thread(
            thread_id = "src",
            payload = chat_history.ChatForkRequest(
                messageId = "m1",
                newThreadId = "new",
                createdAt = 2,
            ),
            current_subject = "test-user",
        )
    )
    assert response.thread.id == "new"
    assert response.thread.title == "fork · Original"
    assert response.thread.forkedFromThreadId == "src"
    assert response.thread.forkedFromMessageId == "m1"
    assert len(response.messages) == 1
    assert response.containerSnapshotWarning is None


def test_fork_thread_warns_when_parent_had_container(monkeypatch):
    source = {
        "id": "src",
        "title": "T",
        "modelType": "base",
        "modelId": "",
        "pairId": None,
        "archived": False,
        "createdAt": 1,
        "openaiCodeExecContainerId": "cnt_123",
        "anthropicCodeExecContainerId": None,
        "forkedFromThreadId": None,
        "forkedFromMessageId": None,
    }
    monkeypatch.setattr(chat_history, "get_chat_thread", lambda _id: source)
    monkeypatch.setattr(
        chat_history,
        "get_chat_message",
        lambda _t, _m: {
            "id": _m,
            "threadId": _t,
            "role": "user",
            "content": [],
            "createdAt": 1,
        },
    )
    monkeypatch.setattr(
        chat_history,
        "fork_chat_thread",
        lambda **_: {
            **source,
            "id": "new",
            "title": "fork · T",
            "forkedFromThreadId": "src",
            "forkedFromMessageId": "m1",
            "openaiCodeExecContainerId": None,
        },
    )
    monkeypatch.setattr(chat_history, "list_chat_messages", lambda _id: [])
    response = asyncio.run(
        chat_history.fork_thread(
            thread_id = "src",
            payload = chat_history.ChatForkRequest(
                messageId = "m1",
                newThreadId = "new",
                createdAt = 2,
            ),
            current_subject = "test-user",
        )
    )
    assert response.containerSnapshotWarning is not None
    assert "fresh" in response.containerSnapshotWarning.lower()


def test_get_fork_count(monkeypatch):
    monkeypatch.setattr(chat_history, "count_forks_for_message", lambda _t, _m: 3)
    response = asyncio.run(
        chat_history.get_fork_count(
            thread_id = "t",
            message_id = "m",
            current_subject = "test-user",
        )
    )
    assert response.count == 3
