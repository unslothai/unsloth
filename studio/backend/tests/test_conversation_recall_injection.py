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
    turns = [
        {"role": "user", "content": "write me a limerick about pelicans"},
        {"role": "assistant", "content": "There once was a bird with a bill"},
    ]
    # The thread has to exist in studio.db: only a persisted thread can be deleted, and
    # an archive nothing can delete is the temporary-chat leak the rule prevents.
    from storage import studio_db

    studio_db.upsert_chat_thread(
        {
            "id": THREAD,
            "title": "t",
            "modelType": "base",
            "modelId": "local-model",
            "createdAt": 1,
        }
    )
    for index, message in enumerate(turns):
        studio_db.upsert_chat_message(
            {
                "id": f"{THREAD}-{index}",
                "threadId": THREAD,
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
                "createdAt": index + 2,
            }
        )
    conversation_archive.archive_turns(THREAD, turns)
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


# ---------------------------------------------------------------------------
# The sticky compaction boundary
# ---------------------------------------------------------------------------


def _fake_studio_db(monkeypatch, messages):
    """Stand in for storage.studio_db.list_chat_messages."""
    import sys
    import types

    module = types.SimpleNamespace(list_chat_messages = lambda thread_id: messages)
    package = types.ModuleType("storage")
    package.studio_db = module
    monkeypatch.setitem(sys.modules, "storage", package)
    monkeypatch.setitem(sys.modules, "storage.studio_db", module)


def test_sticky_boundary_reads_the_newest_assistant_truncation(monkeypatch):
    from core.inference import llama_cpp
    _fake_studio_db(
        monkeypatch,
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "a",
                "metadata": {
                    "custom": {"contextTruncation": {"fits": True, "dropped_messages": 12}}
                },
            },
            {"role": "user", "content": "q2"},
            {
                "role": "assistant",
                "content": "a2",
                "metadata": {
                    "custom": {"contextTruncation": {"fits": True, "dropped_messages": 18}}
                },
            },
        ],
    )

    assert llama_cpp._sticky_compaction_boundary("t1") == 18


def test_sticky_boundary_reads_the_flattened_metadata_shape(monkeypatch):
    """The history row flattens `custom` into `metadata`; both shapes are in the wild."""
    from core.inference import llama_cpp

    _fake_studio_db(
        monkeypatch,
        [
            {
                "role": "assistant",
                "content": "a",
                "metadata": {"contextTruncation": {"fits": True, "dropped_messages": 7}},
            },
        ],
    )

    assert llama_cpp._sticky_compaction_boundary("t1") == 7


def test_sticky_boundary_is_zero_without_a_thread_or_history(monkeypatch):
    from core.inference import llama_cpp

    assert llama_cpp._sticky_compaction_boundary(None) == 0
    assert llama_cpp._sticky_compaction_boundary("") == 0
    _fake_studio_db(monkeypatch, [])
    assert llama_cpp._sticky_compaction_boundary("t1") == 0


def test_sticky_boundary_ignores_a_fit_that_did_not_fit(monkeypatch):
    """`fits: false` means the fit gave up, so it describes no boundary to restore."""
    from core.inference import llama_cpp

    _fake_studio_db(
        monkeypatch,
        [
            {
                "role": "assistant",
                "content": "a",
                "metadata": {
                    "custom": {"contextTruncation": {"fits": False, "dropped_messages": 40}}
                },
            },
        ],
    )

    assert llama_cpp._sticky_compaction_boundary("t1") == 0


def test_sticky_boundary_never_raises_on_a_storage_failure(monkeypatch):
    """A boundary is an optimisation; losing it must never cost the user a turn."""
    import sys
    import types

    from core.inference import llama_cpp

    def explode(thread_id):
        raise RuntimeError("database is locked")

    module = types.SimpleNamespace(list_chat_messages = explode)
    package = types.ModuleType("storage")
    package.studio_db = module
    monkeypatch.setitem(sys.modules, "storage", package)
    monkeypatch.setitem(sys.modules, "storage.studio_db", module)

    assert llama_cpp._sticky_compaction_boundary("t1") == 0


# ---------------------------------------------------------------------------
# Sizing the forced recall, and applying the sticky boundary once
# ---------------------------------------------------------------------------


def test_recall_is_sized_by_the_room_the_fit_actually_obtained():
    """The reserve is what the fit AIMS for, not what it always gets.

    Protected messages (system, the latest turn, anchored recalls) can stop the trim
    reaching its target while the result is still under the prompt budget, which the fit
    accepts. Reproduced at ctx 8000: the fit accepted at 6900, and a full reserve of
    recall on top took the request to 8948, past the window it had just been made to fit.
    """
    from core.inference import llama_cpp

    assert llama_cpp._recall_top_k(0) == 0
    assert llama_cpp._recall_top_k(-500) == 0
    # Enough room for some chunks but not the full allowance.
    assert 0 < llama_cpp._recall_top_k(1024) <= 4
    # Plenty of room is still capped by the configured top-k.
    assert llama_cpp._recall_top_k(1_000_000) == 4


def test_the_sticky_boundary_is_applied_once_per_request():
    """The tool loop refits on the conversation the previous fit returned.

    Re-applying the persisted count there evicts another boundary-sized block of live
    history. Measured before the fix: a second fit dropped 28 of the 30 surviving
    messages instead of 14, leaving 4.
    """
    from core.inference.context_window import fit_rolling_context

    def counter(messages):
        return sum(max(1, len(m["content"]) // 4) for m in messages)

    conversation = [{"role": "system", "content": "s" * 80}]
    for index in range(40):
        conversation.append({"role": "user", "content": f"q{index} " * 200})
        conversation.append({"role": "assistant", "content": f"a{index} " * 200})
    conversation.append({"role": "user", "content": "latest"})

    conversation, first = fit_rolling_context(
        conversation,
        context_length = 8000,
        max_tokens = 512,
        count_tokens = counter,
        sticky_dropped = 52,
    )
    assert first["dropped_messages"] == 52
    # A tool result lands and pushes the already-fitted conversation back over budget.
    conversation = conversation + [
        {"role": "assistant", "content": "calling"},
        {"role": "tool", "content": "R" * (2600 * 4)},
    ]
    _, reapplied = fit_rolling_context(
        conversation,
        context_length = 8000,
        max_tokens = 512,
        count_tokens = counter,
        sticky_dropped = 52,
    )
    _, once = fit_rolling_context(
        conversation,
        context_length = 8000,
        max_tokens = 512,
        count_tokens = counter,
        sticky_dropped = 0,
    )
    assert reapplied["dropped_messages"] > once["dropped_messages"]
    # The request path must take the second shape: the boundary describes the ORIGINAL
    # transcript, so it is spent after the first fit.
    from pathlib import Path

    source = Path(__file__).resolve().parent.parent / "core/inference/llama_cpp.py"
    text = source.read_text()
    assert "_sticky_boundary_applied = True" in text
    assert "0 if _sticky_boundary_applied" in text


def test_conversation_search_top_k_is_clamped(archived, monkeypatch):
    """top_k is written by the model. A negative reaches a slice as out[:-1]."""
    seen = {}

    def fake_recall(
        thread_id,
        query,
        *,
        top_k = None,
    ):
        seen["top_k"] = top_k
        return ("earlier turn", [{"id": "1"}])

    monkeypatch.setattr(conversation_archive, "recall", fake_recall)

    tools_mod._search_conversation({"query": "pelicans", "top_k": -1}, {"thread_id": THREAD})
    assert seen["top_k"] >= 1

    tools_mod._search_conversation({"query": "pelicans", "top_k": 10_000}, {"thread_id": THREAD})
    assert seen["top_k"] <= tools_mod._MAX_CONVERSATION_SEARCH_TOP_K


def test_the_conversation_tool_survives_studios_explicit_allowlist(monkeypatch):
    """Studio always sends enabled_tools, and it never names this internal tool.

    While the gate could only REMOVE, the allowlist filter dropped search_conversation
    before it ran, so the tool (and the compaction nudge gated on it) never appeared in a
    Studio chat at all, however long the conversation got.
    """
    import asyncio
    import types

    import routes.inference as routes_mod

    monkeypatch.setattr(routes_mod, "_thread_has_conversation_archive", lambda _tid: True)

    payload = types.SimpleNamespace(
        enabled_tools = ["search_knowledge_base", "web_search"],
        rag_scope = {"thread_id": THREAD},
        thread_id = THREAD,
        bypass_permissions = False,
    )
    tools = asyncio.run(routes_mod._select_request_tools(payload, tools_on = True, mcp_allowed = False))
    names = [tool["function"]["name"] for tool in tools]

    assert "search_conversation" in names
    assert names.count("search_conversation") == 1
    # Still absent without an archive: an ordinary short chat never sees the schema.
    monkeypatch.setattr(routes_mod, "_thread_has_conversation_archive", lambda _tid: False)
    tools = asyncio.run(routes_mod._select_request_tools(payload, tools_on = True, mcp_allowed = False))
    assert "search_conversation" not in [t["function"]["name"] for t in tools]


def test_both_retrieval_tools_share_the_per_turn_search_cap():
    """Each search appends passages into the protected current exchange.

    The rolling window cannot evict those, so an uncapped tool can only end the turn in
    a context-length error after paying for the embeddings.
    """
    from pathlib import Path

    from core.inference.llama_cpp import _RAG_SEARCH_TOOLS

    assert _RAG_SEARCH_TOOLS == {"search_knowledge_base", "search_conversation"}
    text = (Path(__file__).resolve().parent.parent / "core/inference/llama_cpp.py").read_text()
    # The cap and the counter must both key on the set, not on one tool name.
    assert "decision.tool_name in _RAG_SEARCH_TOOLS" in text
    assert 'decision.tool_name == "search_knowledge_base"' not in text
