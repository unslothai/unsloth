# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A thread whose rows never recorded a context fit answers "no boundary" without loading its transcript."""

import pytest

from core.inference import llama_cpp
from routes import inference as inference_routes
from storage import studio_db


def _seed(thread_id, metadata):
    studio_db.upsert_chat_thread(
        {"id": thread_id, "title": "Chat", "modelType": "base", "modelId": "local", "createdAt": 1}
    )
    studio_db.upsert_chat_message(
        {
            "id": f"{thread_id}-u",
            "threadId": thread_id,
            "role": "user",
            "content": [{"type": "text", "text": "question"}],
            "createdAt": 2,
        }
    )
    studio_db.upsert_chat_message(
        {
            "id": f"{thread_id}-a",
            "threadId": thread_id,
            "parentId": f"{thread_id}-u",
            "role": "assistant",
            "content": [{"type": "text", "text": "answer"}],
            "metadata": metadata,
            "createdAt": 3,
        }
    )


BRANCH = [
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "answer"},
    {"role": "user", "content": "next"},
]


@pytest.fixture
def counted_reads(monkeypatch):
    calls = []
    real = studio_db.list_chat_messages

    def _counting(thread_id):
        calls.append(thread_id)
        return real(thread_id)

    monkeypatch.setattr(studio_db, "list_chat_messages", _counting)
    return calls


def test_gate_reads_only_threads_with_a_recorded_fit():
    _seed("plain", {"custom": {"generationStatus": "completed"}})
    _seed("top", {"contextTruncation": {"fits": True, "boundary_messages": 2}})
    _seed("custom", {"custom": {"contextTruncation": {"fits": True, "boundary_messages": 2}}})
    assert studio_db.chat_thread_has_context_truncation("plain") is False
    assert studio_db.chat_thread_has_context_truncation("top") is True
    assert studio_db.chat_thread_has_context_truncation("custom") is True
    assert studio_db.chat_thread_has_context_truncation("missing") is False


def test_no_recorded_fit_skips_the_transcript_read(counted_reads):
    _seed("plain", {"custom": {"generationStatus": "completed"}, "note": "contextTruncatio n"})
    assert llama_cpp._sticky_compaction_details("plain", BRANCH) == (0, False, False)
    assert inference_routes._thread_has_checkpoint("plain", BRANCH) is False
    assert counted_reads == []


def test_recorded_fit_still_reads_the_transcript(counted_reads):
    _seed(
        "top", {"contextTruncation": {"fits": True, "boundary_messages": 2, "dropped_messages": 2}}
    )
    llama_cpp._sticky_compaction_details("top", BRANCH)
    inference_routes._thread_has_checkpoint("top", BRANCH)
    assert counted_reads == ["top", "top"]
