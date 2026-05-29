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
# /api/chat/settings
# ---------------------------------------------------------------------------


def test_chat_settings_payload_accepts_fast_mode_presets():
    payload = chat_history.ChatSettingsPayload.model_validate(
        {
            "inferenceParams": {"fastMode": False},
            "customPresets": [
                {
                    "name": "Fast Opus",
                    "params": {
                        "temperature": 0.6,
                        "topP": 0.95,
                        "topK": 20,
                        "minP": 0.01,
                        "repetitionPenalty": 1.0,
                        "presencePenalty": 0.0,
                        "maxTokens": 8192,
                        "systemPrompt": "",
                        "trustRemoteCode": False,
                        "fastMode": True,
                    },
                },
            ],
        }
    )

    dumped = payload.model_dump(exclude_unset = True)
    assert dumped["inferenceParams"]["fastMode"] is False
    assert dumped["customPresets"][0]["params"]["fastMode"] is True


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
