# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Turns an external provider's fit evicts stay reachable through search_conversation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from starlette.requests import Request

import routes.inference as ri
from core.inference import tools as tools_mod
from core.rag import conversation_archive
from models.inference import ChatCompletionRequest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_external_auto_compaction import _proxy  # noqa: E402

THREAD = "thread-external-archive"
_FILLER = "word " * 400
_DOCUMENT = "[PDF: referto.pdf]\nReparto di cardiologia, telefono 055 794 3321."


def _chat() -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": "Answer briefly."}]
    messages.append({"role": "user", "content": f"Read this document.\n\n{_DOCUMENT}"})
    messages.append({"role": "assistant", "content": "I have read the cardiology report."})
    for index in range(10):
        messages.append({"role": "user", "content": f"question {index} {_FILLER}"})
        messages.append({"role": "assistant", "content": f"answer {index} {_FILLER}"})
    messages.append({"role": "user", "content": "Was there a phone number on the report?"})
    return messages


def _persist(thread_id: str, messages: list[dict]) -> None:
    from storage import studio_db
    studio_db.upsert_chat_thread(
        {
            "id": thread_id,
            "title": "t",
            "modelType": "base",
            "modelId": "a-model",
            "createdAt": 1,
        }
    )
    for index, message in enumerate(m for m in messages if m["role"] != "system"):
        studio_db.upsert_chat_message(
            {
                "id": f"{thread_id}-{index}",
                "threadId": thread_id,
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
                "createdAt": index + 2,
            }
        )


def _payload(thread_id: str | None) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages = [{"role": "user", "content": "latest"}],
        provider_type = "openrouter",
        model = "a-model",
        thread_id = thread_id,
        context_overflow = "truncate_oldest",
        compaction_threshold = 3_000,
        context_window = 4_000,
        max_tokens = 512,
    )


@pytest.fixture
def archive_ready(rag_home, rag_conn, stub_embeddings):
    return rag_conn


def test_an_external_fit_archives_the_turns_it_evicts(archive_ready):
    messages = _chat()
    _persist(THREAD, messages)

    fitted, truncation, _ = ri._fit_external_context(messages, _payload(THREAD))

    assert truncation and truncation["dropped_messages"] > 0
    assert all(_DOCUMENT not in str(m.get("content")) for m in fitted)
    assert conversation_archive.has_archive(THREAD)


def test_an_attached_document_from_an_evicted_turn_is_searchable(archive_ready):
    messages = _chat()
    _persist(THREAD, messages)

    ri._fit_external_context(messages, _payload(THREAD))
    result = tools_mod.execute_tool(
        "search_conversation",
        {"query": "cardiology report phone number"},
        thread_id = THREAD,
        timeout = None,
    )

    assert "055 794 3321" in result


def test_a_fit_without_a_thread_archives_nothing(archive_ready):
    fitted, truncation, _ = ri._fit_external_context(_chat(), _payload(None))

    assert truncation and truncation["dropped_messages"] > 0
    assert not conversation_archive.has_archive(THREAD)


def test_a_chat_that_fits_archives_nothing(archive_ready):
    messages = [
        {"role": "user", "content": f"Read this document.\n\n{_DOCUMENT}"},
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "Thanks."},
    ]
    _persist(THREAD, messages)

    fitted, truncation, _ = ri._fit_external_context(messages, _payload(THREAD))

    assert truncation is None
    assert fitted is messages
    assert not conversation_archive.has_archive(THREAD)


def test_an_archive_failure_still_returns_the_fitted_chat(archive_ready, monkeypatch):
    messages = _chat()
    _persist(THREAD, messages)

    def broken(*args, **kwargs):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(conversation_archive, "archive_turns", broken)

    fitted, truncation, _ = ri._fit_external_context(messages, _payload(THREAD))

    assert truncation and truncation["dropped_messages"] > 0
    assert fitted[-1]["content"] == "Was there a phone number on the report?"


def test_an_external_tool_chat_with_an_archive_is_told_about_compaction(monkeypatch):
    monkeypatch.setattr(ri, "_thread_has_conversation_archive", lambda thread_id: True)

    async def connected(self) -> bool:
        return False

    # The tool loop polls for a disconnect and would stop before reaching the provider.
    monkeypatch.setattr(Request, "is_disconnected", connected)
    _, sent = _proxy(
        "openrouter",
        [{"role": "user", "content": "Was there a phone number on the report?"}],
        thread_id = THREAD,
        enable_tools = True,
        enabled_tools = ["python"],
        permission_mode = "off",
    )

    names = {tool["function"]["name"] for tool in sent["tools"]}
    system = " ".join(str(m.get("content")) for m in sent["messages"] if m.get("role") == "system")
    assert "search_conversation" in names
    assert "call search_conversation before answering" in system


def test_an_external_tool_chat_without_an_archive_gets_no_compaction_nudge(monkeypatch):
    monkeypatch.setattr(ri, "_thread_has_conversation_archive", lambda thread_id: False)

    async def connected(self) -> bool:
        return False

    monkeypatch.setattr(Request, "is_disconnected", connected)
    _, sent = _proxy(
        "openrouter",
        [{"role": "user", "content": "hello"}],
        thread_id = THREAD,
        enable_tools = True,
        enabled_tools = ["python"],
        permission_mode = "off",
    )

    system = " ".join(str(m.get("content")) for m in sent["messages"] if m.get("role") == "system")
    assert "search_conversation" not in system
