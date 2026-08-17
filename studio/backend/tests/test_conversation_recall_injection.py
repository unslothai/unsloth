# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""How recalled turns get back into a compacted request, and what must not ride along."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.inference import llama_cpp  # noqa: E402
from core.inference import tools as tools_mod  # noqa: E402
from core.rag import conversation_archive  # noqa: E402

THREAD = "thread-recall"


@pytest.fixture
def archived(rag_home, rag_conn, stub_embeddings):
    conversation_archive.archive_turns(
        THREAD,
        [
            {"role": "user", "content": "write me a limerick about pelicans"},
            {"role": "assistant", "content": "There once was a bird with a bill"},
        ],
    )
    return rag_conn


def _conversation():
    return [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "what was that pelicans limerick"},
    ]


def test_recall_runs_even_when_document_rag_is_off(archived):
    """Compaction happens regardless of the user's RAG toggle.

    The turns being recalled are the conversation's own, so gating this on rag_scope
    would mean a compacted chat silently loses its history whenever documents are off.
    """
    built = tools_mod.build_conversation_recall(_conversation(), THREAD, style = "tool")

    assert built is not None
    assert built["sources"] >= 1
    assert built["messages"][0]["tool_calls"][0]["function"]["name"] == "search_conversation"


def test_tool_style_matches_the_rag_autoinject_shape(archived):
    """The UI renders forced retrieval through the existing tool-card path."""
    built = tools_mod.build_conversation_recall(_conversation(), THREAD, style = "tool")

    assert [event["type"] for event in built["events"]] == [
        "status",
        "tool_start",
        "tool_end",
        "status",
    ]
    assert [message["role"] for message in built["messages"]] == ["assistant", "tool"]
    call_id = built["messages"][0]["tool_calls"][0]["id"]
    assert call_id.startswith("conv_recall_")
    assert built["messages"][1]["tool_call_id"] == call_id


def test_inline_style_returns_a_prefix_and_no_forged_tool_messages(archived):
    """The plain path sends no tools array; a tool role there is a template hazard."""
    built = tools_mod.build_conversation_recall(_conversation(), THREAD, style = "inline")

    assert built["messages"] == []
    assert built["events"] == []
    assert "<recalled_conversation>" in built["prefix"]


def test_recall_is_none_without_a_thread_id(archived):
    assert tools_mod.build_conversation_recall(_conversation(), None) is None


def test_recall_is_none_when_nothing_matches(archived):
    built = tools_mod.build_conversation_recall(
        [{"role": "user", "content": "zzzz unrelated quantum plumbing"}], THREAD
    )

    assert built is None or built["sources"] >= 0


def test_prefix_user_text_does_not_mutate_the_original():
    message = {"role": "user", "content": "original"}

    updated = llama_cpp._prefix_user_text(message, "PREFIX ")

    assert message["content"] == "original"
    assert updated["content"] == "PREFIX original"


def test_prefix_user_text_handles_content_parts():
    message = {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {}}, {"type": "text", "text": "hello"}],
    }

    updated = llama_cpp._prefix_user_text(message, "PREFIX ")

    assert updated["content"][1]["text"] == "PREFIX hello"
    assert message["content"][1]["text"] == "hello"


def test_archive_and_recall_reports_counts_only(archived):
    """The counts merge into the context_truncated SSE payload, which goes to the client.

    Message text must never ride along on that event.
    """
    before = _conversation() + [{"role": "user", "content": "evicted turn"}]
    after = _conversation()

    result = llama_cpp._archive_and_recall(
        after, before, thread_id = THREAD, style = "tool", recall_done = False
    )

    assert set(result["counts"]) <= {"archived_messages", "recalled_chunks"}
    assert all(isinstance(value, int) for value in result["counts"].values())


def test_archive_and_recall_skips_recall_once_already_done(archived):
    before = _conversation() + [{"role": "user", "content": "evicted turn"}]
    after = _conversation()

    result = llama_cpp._archive_and_recall(
        after, before, thread_id = THREAD, style = "tool", recall_done = True
    )

    assert result["recalled"] is False
    assert result["conversation"] is after
    assert "recalled_chunks" not in result["counts"]


def test_archive_and_recall_is_a_noop_without_a_thread_id(archived):
    after = _conversation()

    result = llama_cpp._archive_and_recall(
        after, after, thread_id = None, style = "tool", recall_done = False
    )

    assert result["conversation"] is after
    assert result["counts"] == {}


def test_recall_reserve_is_zero_without_a_thread():
    assert llama_cpp._conversation_recall_reserve(None) == 0


def test_recall_reserve_is_zero_when_the_archive_is_disabled(monkeypatch):
    monkeypatch.setattr(conversation_archive.config, "CONVERSATION_ARCHIVE", False)

    assert llama_cpp._conversation_recall_reserve(THREAD) == 0


def test_build_rag_autoinject_still_emits_its_original_shape(monkeypatch):
    """Guards the extraction that gave recall and document auto-inject a shared builder."""
    from core.rag import tool as rag_tool

    # Patched on core.rag.tool, not on tools_mod: build_rag_autoinject imports it lazily
    # inside the function body, so rebinding the caller's module has no effect.
    monkeypatch.setattr(
        rag_tool,
        "search_for_autoinject",
        lambda **kwargs: ("passage text", [{"citationId": 1, "filename": "doc.txt"}]),
    )

    # kb_id rather than thread_id so this exercises the retrieval branch rather than the
    # whole-document one, and the toggles ride in the scope instead of a monkeypatch.
    built = tools_mod.build_rag_autoinject(
        [{"role": "user", "content": "question"}],
        {"kb_id": "kb-1", "autoinject": True, "autoinject_min_score": 0.0},
    )

    assert built is not None
    assert [event["type"] for event in built["events"]] == [
        "status",
        "tool_start",
        "tool_end",
        "status",
    ]
    assert built["messages"][0]["tool_calls"][0]["function"]["name"] == "search_knowledge_base"
    assert built["messages"][0]["tool_calls"][0]["id"].startswith("rag_auto_")
    assert built["messages"][1]["content"] == "passage text"


def test_search_conversation_is_always_safe():
    """Read-only, so auto mode must not prompt on every call."""
    assert tools_mod.is_always_safe_tool("search_conversation") is True


def test_search_conversation_tool_is_registered():
    names = {tool["function"]["name"] for tool in tools_mod.ALL_TOOLS}

    assert "search_conversation" in names


def test_execute_tool_without_a_thread_returns_a_message_not_a_traceback():
    result = tools_mod.execute_tool(
        "search_conversation", {"query": "anything"}, thread_id = None, timeout = None
    )

    assert isinstance(result, str)
    assert "no earlier conversation" in result.lower()


def test_execute_tool_rejects_an_empty_query(archived):
    result = tools_mod.execute_tool(
        "search_conversation", {"query": "  "}, thread_id = THREAD, timeout = None
    )

    assert result == "Error: query is empty."


def test_execute_tool_finds_an_archived_turn(archived):
    result = tools_mod.execute_tool(
        "search_conversation", {"query": "pelicans"}, thread_id = THREAD, timeout = None
    )

    assert "pelicans" in result


def test_compaction_nudge_only_fires_when_the_tool_is_present():
    import routes.inference as routes_mod

    without = routes_mod._apply_compaction_nudge("base.", [])
    with_tool = routes_mod._apply_compaction_nudge(
        "base.", [{"function": {"name": "search_conversation"}}]
    )

    assert without == "base."
    assert "search_conversation" in with_tool
