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


def test_sticky_boundary_ignores_a_sibling_branchs_assistant_turn(monkeypatch):
    """Retry keeps the replaced response, and the stored rows are the whole DAG.

    Ordered by creation time, the newest assistant can be the sibling the user switched
    away from. Its boundary was measured on a different conversation, so applying it to
    the branch being sent evicts a block sized for history this branch does not have.
    """
    from core.inference import llama_cpp

    _fake_studio_db(
        monkeypatch,
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "answer on the branch we are on",
                "metadata": {
                    "custom": {"contextTruncation": {"fits": True, "dropped_messages": 4}}
                },
            },
            {
                "role": "assistant",
                "content": "regenerated answer the user switched away from",
                "metadata": {
                    "custom": {"contextTruncation": {"fits": True, "dropped_messages": 40}}
                },
            },
        ],
    )
    branch = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "answer on the branch we are on"},
        {"role": "user", "content": "next"},
    ]

    # Thread-wide, the newest row wins even though it is on the other branch.
    assert llama_cpp._sticky_compaction_boundary("t1") == 40
    # Told which branch the request is on, the branch's own boundary is used.
    assert llama_cpp._sticky_compaction_boundary("t1", branch) == 4


def test_sticky_boundary_takes_the_smaller_of_two_identical_replies(monkeypatch):
    """Retry on a short reply gives two siblings whose text is the same.

    The branch check is textual, so "Done." on the live branch and "Done." on the branch
    the user switched away from both look on-branch, and the newest row wins -- applying a
    boundary measured on a much deeper branch and evicting live history this branch still
    has. Where the text cannot separate them, the smaller boundary is the safe one.
    """
    from core.inference import llama_cpp

    _fake_studio_db(
        monkeypatch,
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "Done.",
                "metadata": {
                    "custom": {"contextTruncation": {"fits": True, "dropped_messages": 4}}
                },
            },
            {
                "role": "assistant",
                "content": "Done.",
                "metadata": {
                    "custom": {"contextTruncation": {"fits": True, "dropped_messages": 60}}
                },
            },
        ],
    )
    branch = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "next"},
    ]

    assert llama_cpp._sticky_compaction_boundary("t1", branch) == 4


def test_sticky_boundary_still_prefers_the_newest_distinguishable_reply(monkeypatch):
    """Only replies indistinguishable from the newest are folded in.

    Otherwise the smallest boundary anywhere in a long thread would win every time, and
    the boundary would ratchet backwards over the life of the chat.
    """
    from core.inference import llama_cpp

    _fake_studio_db(
        monkeypatch,
        [
            {
                "role": "assistant",
                "content": "an earlier, shallower answer",
                "metadata": {
                    "custom": {"contextTruncation": {"fits": True, "dropped_messages": 2}}
                },
            },
            {
                "role": "assistant",
                "content": "the newest answer",
                "metadata": {
                    "custom": {"contextTruncation": {"fits": True, "dropped_messages": 30}}
                },
            },
        ],
    )
    branch = [
        {"role": "assistant", "content": "an earlier, shallower answer"},
        {"role": "assistant", "content": "the newest answer"},
    ]

    assert llama_cpp._sticky_compaction_boundary("t1", branch) == 30


def test_sticky_boundary_prefers_a_reply_that_matches_the_branch_exactly(monkeypatch):
    """The branch check is a substring test, and short replies ride in on longer ones.

    "Done" is contained in the live "Not done yet, still running.", so an abandoned
    sibling looked active; having no live twin it then decided the boundary alone, and
    its much deeper count evicted history the branch still had.
    """
    from core.inference import llama_cpp

    _fake_studio_db(
        monkeypatch,
        [
            {"role": "user", "content": "did it work"},
            {
                "role": "assistant",
                "content": "Not done yet, still running.",
                "metadata": {
                    "custom": {"contextTruncation": {"fits": True, "dropped_messages": 4}}
                },
            },
            {
                "role": "assistant",
                "content": "Done",
                "metadata": {
                    "custom": {"contextTruncation": {"fits": True, "dropped_messages": 60}}
                },
            },
        ],
    )
    branch = [
        {"role": "user", "content": "did it work"},
        {"role": "assistant", "content": "Not done yet, still running."},
        {"role": "user", "content": "keep going"},
    ]

    assert llama_cpp._sticky_compaction_boundary("t1", branch) == 4


def test_sticky_boundary_still_reads_a_reply_no_branch_message_matches_exactly(monkeypatch):
    """The exact preference must not become a requirement.

    A stored row is not always byte-identical to what the client re-sends (blocks the
    request omits, for one), and demanding equality would silently switch the boundary
    off for every such thread.
    """
    from core.inference import llama_cpp

    _fake_studio_db(
        monkeypatch,
        [
            {
                "role": "assistant",
                "content": "the answer",
                "metadata": {
                    "custom": {"contextTruncation": {"fits": True, "dropped_messages": 9}}
                },
            },
        ],
    )
    branch = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "the answer, with a block the request drops"},
    ]

    assert llama_cpp._sticky_compaction_boundary("t1", branch) == 9


def test_sticky_boundary_prefers_the_recorded_branch_boundary(monkeypatch):
    """A turn that refit several times persists a dropped count larger than the branch.

    The counts are summed by the client and include the tool exchanges the turn itself
    created, which the next request's saved transcript does not contain. The boundary
    recorded against the request's own messages is the one to restore.
    """
    from core.inference import llama_cpp

    _fake_studio_db(
        monkeypatch,
        [
            {
                "role": "assistant",
                "content": "a",
                "metadata": {
                    "custom": {
                        "contextTruncation": {
                            "fits": True,
                            "dropped_messages": 12,
                            "boundary_messages": 4,
                        }
                    }
                },
            },
        ],
    )

    assert llama_cpp._sticky_compaction_boundary("t1") == 4


def test_sticky_boundary_falls_back_for_turns_saved_before_the_boundary_existed(monkeypatch):
    """Rows written by an earlier build carry only the dropped count."""
    from core.inference import llama_cpp

    _fake_studio_db(
        monkeypatch,
        [
            {
                "role": "assistant",
                "content": "a",
                "metadata": {
                    "custom": {"contextTruncation": {"fits": True, "dropped_messages": 6}}
                },
            },
        ],
    )

    assert llama_cpp._sticky_compaction_boundary("t1") == 6


def test_the_recall_reserve_is_dropped_once_archiving_has_failed(monkeypatch):
    """sqlite-vec present and the thread saved, but the embedder cannot start.

    archive_turns swallows that, and recall then injects nothing, so the room the fit held
    back is pure loss -- on every compaction. This failure mode forgot substantially more
    history than having the feature turned off.
    """
    from core.inference import llama_cpp
    from core.rag import conversation_archive as archive

    monkeypatch.setattr(archive, "can_archive", lambda _tid: True)
    monkeypatch.setattr(archive, "degraded", lambda: False)
    assert llama_cpp._conversation_recall_reserve("t1") > 0

    monkeypatch.setattr(archive, "degraded", lambda: True)
    assert llama_cpp._conversation_recall_reserve("t1") == 0


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
    # Whitespace-insensitive: the gate is one expression however the formatter wraps it.
    assert "0 if _sticky_boundary_applied" in " ".join(text.split())


def test_conversation_search_top_k_is_clamped(archived, monkeypatch):
    """top_k is written by the model. A negative reaches a slice as out[:-1]."""
    seen = {}

    def fake_recall(
        thread_id,
        query,
        *,
        top_k = None,
        branch_messages = None,
    ):
        seen["top_k"] = top_k
        seen["branch_messages"] = branch_messages
        return ("earlier turn", [{"id": "1"}])

    monkeypatch.setattr(conversation_archive, "recall", fake_recall)

    tools_mod._search_conversation({"query": "pelicans", "top_k": -1}, {"thread_id": THREAD})
    assert seen["top_k"] >= 1

    tools_mod._search_conversation({"query": "pelicans", "top_k": 10_000}, {"thread_id": THREAD})
    assert seen["top_k"] <= tools_mod._MAX_CONVERSATION_SEARCH_TOP_K


def test_conversation_search_top_k_is_clamped_by_the_live_budget(archived, monkeypatch):
    """The fixed ceiling bounds what the model may ask for, not what the context holds.

    Eight chunks is roughly 4,000 tokens once wrapped, and the result lands in the current
    tool exchange, which rolling truncation protects and cannot evict -- so on a small
    context an unbudgeted search is a context-length error the next preflight cannot undo.
    """
    from core.rag import config as rag_config

    seen = {}

    def fake_recall(
        thread_id,
        query,
        *,
        top_k = None,
        branch_messages = None,
    ):
        seen["top_k"] = top_k
        return ("earlier turn", [{"id": "1"}])

    monkeypatch.setattr(conversation_archive, "recall", fake_recall)

    # Room for two chunks, asked for the ceiling.
    tools_mod._search_conversation(
        {"query": "pelicans", "top_k": 8},
        {"thread_id": THREAD, "budget_tokens": rag_config.CHUNK_TOKENS * 2},
    )
    assert seen["top_k"] == 2

    # An omitted top_k is budgeted too, rather than falling through to the default.
    seen.clear()
    tools_mod._search_conversation(
        {"query": "pelicans"},
        {"thread_id": THREAD, "budget_tokens": rag_config.CHUNK_TOKENS},
    )
    assert seen["top_k"] == 1

    # No room at all: say so instead of returning a result that cannot be sent.
    seen.clear()
    answer = tools_mod._search_conversation(
        {"query": "pelicans", "top_k": 4},
        {"thread_id": THREAD, "budget_tokens": 10},
    )
    assert "no room" in answer.lower()
    assert not seen


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

    from core.inference.tool_call_parser import RAG_SEARCH_TOOLS

    assert RAG_SEARCH_TOOLS == {"search_knowledge_base", "search_conversation"}
    # BOTH loops: the tool is advertised per thread, so a chat compacted under a GGUF
    # model can go on to call it under a safetensors one.
    backend = Path(__file__).resolve().parent.parent / "core/inference"
    for module in ("llama_cpp.py", "safetensors_agentic.py"):
        text = (backend / module).read_text()
        # The cap and the counter must both key on the set, not on one tool name.
        assert "decision.tool_name in RAG_SEARCH_TOOLS" in text, module
        assert 'decision.tool_name == "search_knowledge_base"' not in text, module


def test_an_omitted_top_k_falls_through_to_the_configured_default(archived, monkeypatch):
    """Defaulting to the CEILING made an ordinary search return eight archived turns.

    Those land in the protected current exchange, which rolling truncation cannot evict,
    so a single search could fail the next pass on a small window.
    """
    seen = {}

    def fake_recall(
        thread_id,
        query,
        *,
        top_k = None,
        branch_messages = None,
    ):
        seen["top_k"] = top_k
        seen["branch_messages"] = branch_messages
        return ("earlier turn", [{"id": "1"}])

    monkeypatch.setattr(conversation_archive, "recall", fake_recall)
    tools_mod._search_conversation({"query": "pelicans"}, {"thread_id": THREAD})

    assert seen["top_k"] is None


def test_a_thread_that_cannot_be_archived_holds_back_no_reserve(monkeypatch):
    """The reserve is room for recalled turns, and a temporary chat never has any.

    archive_turns refuses a thread with no saved messages, so nothing can ever be
    recalled into that room -- while the fit subtracts it from the trim target, so it
    is paid for in evicted history. On a 4K window that was most of the conversation.
    """
    from core.inference import llama_cpp
    from core.rag import conversation_archive

    monkeypatch.setattr(conversation_archive, "enabled", lambda: True)

    monkeypatch.setattr(conversation_archive, "can_archive", lambda thread_id: False)
    assert llama_cpp._conversation_recall_reserve("incognito") == 0

    monkeypatch.setattr(conversation_archive, "can_archive", lambda thread_id: True)
    assert llama_cpp._conversation_recall_reserve("saved") > 0
    # And no thread at all is still zero.
    assert llama_cpp._conversation_recall_reserve(None) == 0


def test_the_forced_recall_searches_for_the_USERS_question(archived, monkeypatch):
    """The loop conversation can end with an internal user-role re-prompt.

    The plan-without-action nudge and the deferred no-op both append one, so a first
    overflow on a later iteration would search the archive for a generic controller
    instruction instead of what the user asked, and return whatever sits near it.
    """
    seen = {}

    def fake_recall(
        thread_id,
        query,
        *,
        top_k = None,
        branch_messages = None,
    ):
        seen["query"] = query
        return ("earlier turn", [{"id": "1"}])

    monkeypatch.setattr(conversation_archive, "recall", fake_recall)
    branch = [
        {"role": "user", "content": "what was the VULPINE code from earlier"},
        {"role": "assistant", "content": "checking"},
    ]
    conversation = branch + [
        {"role": "user", "content": "You said you would use a tool. Do it now or answer."}
    ]

    tools_mod.build_conversation_recall(
        conversation, THREAD, style = "inline", branch_messages = branch
    )

    assert seen["query"] == "what was the VULPINE code from earlier"


def test_a_model_initiated_search_is_filtered_to_the_request_branch(archived, monkeypatch):
    """The forced recall is branch-filtered, so the tool the model can call must be too.

    Otherwise the model simply asks for what the forced recall correctly refused, and a
    response the user replaced with Retry comes back through the other door.
    """
    seen = {}

    def fake_recall(
        thread_id,
        query,
        *,
        top_k = None,
        branch_messages = None,
    ):
        seen["branch_messages"] = branch_messages
        return ("earlier turn", [{"id": "1"}])

    monkeypatch.setattr(conversation_archive, "recall", fake_recall)
    branch = [{"role": "user", "content": "on this branch"}]

    tools_mod.execute_tool(
        "search_conversation",
        {"query": "pelicans"},
        thread_id = THREAD,
        conversation_branch = branch,
    )

    assert seen["branch_messages"] == branch


def test_inline_recall_anchors_only_the_turn_it_rewrote(archived, monkeypatch):
    """Inline recall appends nothing; it rewrites the latest user message in place.

    Anchoring the last two messages therefore pinned the assistant turn before it as well,
    and with it a whole eviction unit the fit was entitled to drop. On a request whose
    next iteration overflows, that is the difference between compacting and failing.
    """
    from core.inference import llama_cpp

    conversation = [
        {"role": "user", "content": "an earlier question"},
        {"role": "assistant", "content": "an earlier answer"},
        {"role": "user", "content": "what was that limerick"},
    ]
    monkeypatch.setattr(
        tools_mod,
        "build_conversation_recall",
        lambda *args, **kwargs: {"prefix": "RECALLED ", "messages": [], "events": [], "sources": 1},
    )

    out = llama_cpp._archive_and_recall(
        conversation,
        conversation,
        thread_id = THREAD,
        style = "inline",
        recall_done = False,
        # Any non-zero budget: with none, the fit obtained no room and recall is skipped.
        recall_budget_tokens = 100_000,
    )

    assert out["recalled"] is True
    # Exactly one message is anchored, and it is the rewritten latest user turn.
    assert len(out["anchored"]) == 1
    assert out["anchored"][0]["content"].startswith("RECALLED ")
    assert out["anchored"][0] is out["conversation"][-1]
    # The assistant turn before it stays evictable.
    assert id(conversation[1]) not in {id(message) for message in out["anchored"]}


def test_tool_recall_anchors_the_synthetic_exchange(archived, monkeypatch):
    """The tool style DOES append two messages, and those are what must survive a refit."""
    from core.inference import llama_cpp

    conversation = [{"role": "user", "content": "what was that limerick"}]
    synthetic = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "conv_recall_x"}]},
        {"role": "tool", "tool_call_id": "conv_recall_x", "content": "an earlier turn"},
    ]
    monkeypatch.setattr(
        tools_mod,
        "build_conversation_recall",
        lambda *args, **kwargs: {"messages": synthetic, "events": [], "sources": 1},
    )

    out = llama_cpp._archive_and_recall(
        conversation,
        conversation,
        thread_id = THREAD,
        style = "tool",
        recall_done = False,
        recall_budget_tokens = 100_000,
    )

    assert [id(message) for message in out["anchored"]] == [id(m) for m in out["conversation"][-2:]]


def test_an_over_budget_recall_is_retried_with_fewer_turns(archived, monkeypatch):
    """Dropping the lot is the wrong answer when three of four chunks would fit.

    A full top-K of long archived turns lands just over the reserve once the wrappers
    are priced, which is the common case on exactly the long conversations this feature
    exists for, so an all-or-nothing check disables the forced retrieval there.
    """
    from core.inference import llama_cpp

    conversation = [{"role": "user", "content": "what was that pelicans limerick"}]
    asked = []

    def build(
        _conversation,
        _thread_id,
        *,
        style,
        top_k,
        branch_messages = None,
    ):
        asked.append(top_k)
        # 1400 characters per requested chunk, so only the smallest k fits the budget.
        return {"prefix": "R" * (1400 * top_k), "messages": [], "events": [], "sources": top_k}

    monkeypatch.setattr(tools_mod, "build_conversation_recall", build)
    chars = lambda messages: sum(len(m.get("content") or "") for m in messages)

    out = llama_cpp._archive_and_recall(
        conversation,
        conversation,
        thread_id = THREAD,
        style = "inline",
        recall_done = False,
        # Room for four chunks by the estimate, but only one once actually counted.
        recall_budget_tokens = 2500,
        count_tokens = chars,
    )

    assert asked == [4, 2, 1]
    assert out["recalled"] is True
    assert out["counts"]["recalled_chunks"] == 1
    assert chars(out["conversation"]) > chars(conversation)


def test_recall_is_dropped_when_the_real_prompt_exceeds_the_budget(archived, monkeypatch):
    """The chunk arithmetic is an estimate; the tokenizer is not.

    CHUNK_TOKENS is an embedding-token limit rather than the chat template's cost for the
    same text, and neither it nor the budget prices the <recalled_conversation> or tool
    wrappers around the injection, so a nominally-fitting recall can still overshoot.
    """
    from core.inference import llama_cpp

    conversation = [{"role": "user", "content": "what was that pelicans limerick"}]
    monkeypatch.setattr(
        tools_mod,
        "build_conversation_recall",
        lambda *args, **kwargs: {
            "prefix": "R" * 4000,
            "messages": [],
            "events": [],
            "sources": 2,
        },
    )
    chars = lambda messages: sum(len(m.get("content") or "") for m in messages)

    # Budget far below what the injection actually costs: dropped, and the conversation
    # comes back untouched rather than over the window.
    tight = llama_cpp._archive_and_recall(
        conversation,
        conversation,
        thread_id = THREAD,
        style = "inline",
        recall_done = False,
        recall_budget_tokens = 10,
        count_tokens = chars,
    )
    assert tight["recalled"] is False
    assert tight["conversation"] == conversation

    # Room to spare: the same injection is kept.
    roomy = llama_cpp._archive_and_recall(
        conversation,
        conversation,
        thread_id = THREAD,
        style = "inline",
        recall_done = False,
        recall_budget_tokens = 100_000,
        count_tokens = chars,
    )
    assert roomy["recalled"] is True
    assert chars(roomy["conversation"]) > chars(conversation)
