# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The archive behind rolling-context compaction: what it keeps, and what it must not touch."""

import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag import config, conversation_archive, retrieval, store  # noqa: E402
from storage import rag_db  # noqa: E402

THREAD = "thread-abc"


def _assistant_call(
    name,
    arguments,
    *,
    id = "c1",
    content = "",
):
    """An assistant turn whose only content is one function tool call."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"id": id, "function": {"name": name, "arguments": arguments}}],
    }


def _turn(question, answer):
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def _save_thread(
    thread_id,
    turns,
    *,
    append = False,
):
    """Persist a transcript by appending; replacing the rows would leave only the newest turn saved."""
    from storage import studio_db

    studio_db.upsert_chat_thread(
        {
            "id": thread_id,
            "title": "t",
            "modelType": "base",
            "modelId": "local-model",
            "createdAt": 1,
        }
    )
    existing = len(studio_db.list_chat_messages(thread_id) or []) if append else 0
    rows = [
        {
            "id": f"{thread_id}-{existing + index}",
            "threadId": thread_id,
            "role": message["role"],
            "content": [{"type": "text", "text": message["content"]}],
            "createdAt": existing + index + 2,
        }
        for index, message in enumerate(turns)
    ]
    if append:
        for row in rows:
            studio_db.upsert_chat_message(row)
        return
    # A rewind via prune_missing sync: deleting would tombstone the id, and recreating it raises.
    studio_db.sync_chat_messages(thread_id, rows, prune_missing = True)


def _archive(
    messages,
    thread_id = THREAD,
    *,
    persist = True,
):
    """Archiving needs the thread persisted in studio.db; persist=False exercises the refusal path."""
    if persist:
        _save_thread(thread_id, messages, append = True)
    return conversation_archive.archive_turns(thread_id, messages)


@pytest.fixture
def conn(rag_home, rag_conn, stub_embeddings):
    return rag_conn


def _tool_part(
    *,
    type = "tool-call",
    toolCallId = "c1",
    toolName = "terminal",
    command = "ls",
    result = "main.py readme.md",
):
    """One stored tool-invocation content part, with per-test overrides."""
    return {
        "type": type,
        "toolCallId": toolCallId,
        "toolName": toolName,
        "args": {"command": command},
        "result": result,
    }


def test_evicted_turns_are_archived_under_the_conversation_scope(conn):
    written = _archive(_turn("what is a duck", "a waterfowl"))

    scope = store.conversation_archive_scope(THREAD)
    assert written == 1
    documents = store.list_documents(conn, scope)
    assert len(documents) == 1
    assert "earlier turn" in documents[0]["filename"]


def test_re_archiving_the_same_turns_writes_nothing(conn):
    """Re-evicted turns are archived again on every request, so a repeat must write nothing."""
    turn = _turn("what is a duck", "a waterfowl")
    first = _archive(turn)
    second = _archive(turn, persist = False)

    scope = store.conversation_archive_scope(THREAD)
    assert (first, second) == (1, 0)
    assert len(store.list_documents(conn, scope)) == 1


def test_archive_accumulates_across_compaction_epochs(conn):
    """Recall must cover every earlier compaction, not just the latest; the archive is cumulative."""
    _archive(_turn("tell me about pelicans", "they have large bills"))
    _archive(_turn("tell me about otters", "they use tools"))
    _archive(_turn("tell me about pangolins", "they have scales"))

    scope = store.conversation_archive_scope(THREAD)
    assert len(store.list_documents(conn, scope)) == 3

    found = conversation_archive.recall(THREAD, "pelicans")

    assert found is not None
    text, _sources = found
    assert "pelicans" in text


def test_whole_document_context_never_sees_archived_turns(conn):
    """Thread documents are injected whole each request, so archived turns must sit in a separate scope."""
    _archive(_turn("secret archived question", "secret archived answer"))

    thread_scope = store.thread_scope(THREAD)
    archive_scope = store.conversation_archive_scope(THREAD)
    assert thread_scope != archive_scope
    assert store.list_documents(conn, thread_scope) == []


def test_archive_skips_instructions_and_its_own_injections(conn):
    written = _archive(
        [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "conv_recall_1"}]},
            {"role": "tool", "tool_call_id": "conv_recall_1", "content": "recalled text"},
        ],
    )

    assert written == 0
    assert store.list_documents(conn, store.conversation_archive_scope(THREAD)) == []


def test_recall_returns_the_gold_turn(conn):
    _archive(_turn("how do I bake sourdough", "start a starter"))
    _archive(_turn("what is the capital of Peru", "Lima"))

    found = conversation_archive.recall(THREAD, "sourdough")

    assert found is not None
    text, sources = found
    assert "sourdough" in text
    assert sources


def test_recall_degrades_to_lexical_when_dense_retrieval_raises(monkeypatch, conn):
    """No embedder must mean weaker recall, never a broken chat."""
    _archive(_turn("how do I bake sourdough", "start a starter"))

    real_hybrid = retrieval.retrieve_hybrid

    def only_lexical_works(
        conn_,
        scope,
        query,
        *,
        k = None,
        model_name = None,
        mode = "hybrid",
        lexical_query = None,
    ):
        if mode != "lexical":
            raise RuntimeError("no embedding backend available")
        return real_hybrid(
            conn_, scope, query, k = k, model_name = model_name, mode = mode, lexical_query = lexical_query
        )

    monkeypatch.setattr(retrieval, "retrieve_hybrid", only_lexical_works)
    found = conversation_archive.recall(THREAD, "sourdough")

    assert found is not None
    assert "sourdough" in found[0]


def test_archive_is_a_noop_when_rag_is_unavailable(monkeypatch, conn):
    monkeypatch.setattr(rag_db, "RAG_AVAILABLE", False)

    assert conversation_archive.archive_turns(THREAD, _turn("q", "a")) == 0
    assert conversation_archive.recall(THREAD, "q") is None
    assert conversation_archive.has_archive(THREAD) is False


def test_archive_is_a_noop_when_disabled(monkeypatch, conn):
    monkeypatch.setattr(conversation_archive.config, "CONVERSATION_ARCHIVE", False)

    assert conversation_archive.archive_turns(THREAD, _turn("q", "a")) == 0
    assert conversation_archive.recall(THREAD, "q") is None


def test_archived_turns_are_hidden_from_the_documents_list(conn):
    _archive(_turn("what is a duck", "a waterfowl"))

    listed = store.list_all_documents(conn)

    assert all(not d["scope"].startswith(store.CONVERSATION_ARCHIVE_PREFIX) for d in listed)


def test_the_transcript_is_never_mutated(conn):
    messages = _turn("what is a duck", "a waterfowl")
    original = copy.deepcopy(messages)

    _archive(messages)
    conversation_archive.recall(THREAD, "duck")

    assert messages == original


def test_has_archive_reports_whether_anything_was_kept(conn):
    assert conversation_archive.has_archive(THREAD) is False
    _archive(_turn("what is a duck", "a waterfowl"))
    assert conversation_archive.has_archive(THREAD) is True


def test_render_turn_keeps_both_sides_together():
    rendered = conversation_archive.render_turn(_turn("what is a duck", "a waterfowl"))

    assert "user: what is a duck" in rendered
    assert "assistant: a waterfowl" in rendered


def test_render_turn_truncates_a_huge_tool_result():
    rendered = conversation_archive.render_turn(
        [{"role": "tool", "tool_call_id": "c1", "content": "x" * 20000}]
    )

    assert len(rendered) < 20000
    assert rendered.endswith("...")


def test_delete_for_thread_drops_the_archive(conn):
    _archive(_turn("what is a duck", "a waterfowl"))

    removed = conversation_archive.delete_for_thread(THREAD)

    assert removed == 1
    assert store.list_documents(conn, store.conversation_archive_scope(THREAD)) == []


def test_recall_finds_a_rare_token_buried_in_boilerplate(conn):
    """Lexical first, since recall is exact matching; hybrid alone lost a rare token in boilerplate."""
    for index in range(1, 9):
        code = " Internal tracking code: VULPINE-9134-QK." if index == 1 else ""
        _archive(
            [
                {
                    "role": "user",
                    "content": f"Here is section {index} of the climate change article."
                    f"{code} Reply with one short sentence naming its main topic.",
                },
                {
                    "role": "assistant",
                    "content": f"Section {index} is about climate change impacts.",
                },
            ],
            thread_id = "needle-thread",
        )

    found = conversation_archive.recall(
        "needle-thread",
        "Earlier I gave you section 1 and it carried an internal tracking code. "
        "What was that exact tracking code?",
    )

    assert found is not None
    assert "VULPINE-9134-QK" in found[0]


def test_recall_does_not_resurrect_a_turn_the_user_rolled_back_past(conn, monkeypatch):
    """Archive is append-only, so recall must filter to the active branch or it returns abandoned turns."""
    kept = [
        {"role": "user", "content": "section one, code KEEPME-1111"},
        {"role": "assistant", "content": "noted section one"},
    ]
    rolled_back = [
        {"role": "user", "content": "section two, code GONEAWAY-2222"},
        {"role": "assistant", "content": "noted section two"},
    ]
    _archive(kept, thread_id = "branch-thread")
    _archive(rolled_back, thread_id = "branch-thread")
    _save_thread("branch-thread", kept)

    survived = conversation_archive.recall("branch-thread", "KEEPME-1111")
    abandoned = conversation_archive.recall("branch-thread", "GONEAWAY-2222")

    assert survived is not None and "KEEPME-1111" in survived[0]
    # Recall has no relevance floor; only the rolled-back content must never come back.
    assert abandoned is None or "GONEAWAY-2222" not in abandoned[0]
    assert "GONEAWAY-2222" not in (survived[0] or "")


def test_recall_filters_to_the_ACTIVE_branch_not_the_whole_stored_thread(conn):
    """Retry keeps the replaced reply as a stored sibling, so only the request's branch separates them."""
    live = [
        {"role": "user", "content": "what is the code, code KEEPME-1111"},
        {"role": "assistant", "content": "the code is KEEPME-1111"},
    ]
    retried_away = [
        {"role": "user", "content": "what is the code, code KEEPME-1111"},
        {"role": "assistant", "content": "the code is SIBLING-3333"},
    ]
    _archive(live, thread_id = "retry-thread")
    _archive(retried_away, thread_id = "retry-thread")
    _save_thread("retry-thread", live)
    _save_thread("retry-thread", retried_away, append = True)

    thread_wide = conversation_archive.recall("retry-thread", "SIBLING-3333")
    assert thread_wide is not None and "SIBLING-3333" in thread_wide[0]

    on_branch = conversation_archive.recall("retry-thread", "SIBLING-3333", branch_messages = live)
    assert on_branch is None or "SIBLING-3333" not in on_branch[0]
    survived = conversation_archive.recall("retry-thread", "KEEPME-1111", branch_messages = live)
    assert survived is not None and "KEEPME-1111" in survived[0]


def test_the_reply_that_FOLLOWS_a_forced_recall_is_still_archived(conn):
    """Keep the reply after a forced recall: rejecting the group over our own injection drops the answer."""
    evicted = [
        {"role": "user", "content": "what was the passphrase"},
        _assistant_call(
            "search_conversation",
            '{"query": "pass"}',
            id = "conv_recall_1",
            content = None,
        ),
        {"role": "tool", "tool_call_id": "conv_recall_1", "content": "<chunk>RETRIEVED</chunk>"},
        {"role": "assistant", "content": "The passphrase you set earlier was SWORDFISH-42."},
    ]
    _save_thread("recall-turn-thread", evicted, append = True)

    assert conversation_archive.archive_turns("recall-turn-thread", evicted) == 2

    found = conversation_archive.recall(
        "recall-turn-thread", "SWORDFISH-42", branch_messages = evicted
    )
    assert found is not None
    assert "SWORDFISH-42" in found[0]
    assert "RETRIEVED" not in found[0]
    assert (
        conversation_archive.recall("recall-turn-thread", "RETRIEVED", branch_messages = evicted)
        is None
        or "RETRIEVED"
        not in conversation_archive.recall(
            "recall-turn-thread", "RETRIEVED", branch_messages = evicted
        )[0]
    )


def test_deleting_a_thread_works_without_sqlite_vec(conn, monkeypatch):
    """Deleting a thread must work even when sqlite-vec stops loading, or archived turns stay on disk."""
    turns = _turn("what is the passphrase", "the passphrase is VECGONE-2020")
    _save_thread("vecless-thread", turns, append = True)
    assert conversation_archive.archive_turns("vecless-thread", turns) == 1

    def no_vec():
        raise rag_db.RagExtensionUnavailable("vec0 will not load")

    monkeypatch.setattr(rag_db, "get_connection", no_vec)
    removed = conversation_archive.delete_for_thread("vecless-thread")
    monkeypatch.undo()

    assert removed == 1
    assert conversation_archive.has_archive("vecless-thread") is False
    assert conversation_archive.recall("vecless-thread", "VECGONE-2020") is None


def test_a_turns_CHUNKS_must_all_sit_in_the_same_place_on_the_branch(conn):
    """A turn's chunks must all sit in one place on the branch; checking them independently mixes parts."""
    rows = [
        {
            "text": "user: How do I deploy?\nassistant: Run the deploy script from the release branch."
        },
        {"text": "assistant: Never deploy on a Friday afternoon."},
    ]
    edited_with_a_later_echo = [
        {"role": "user", "content": "How do I deploy?"},
        {
            "role": "assistant",
            "content": "Run the deploy script from the release branch. Any day is fine.",
        },
        {"role": "user", "content": "What was that old rule of thumb?"},
        {"role": "assistant", "content": "Never deploy on a Friday afternoon."},
    ]
    intact = [
        {"role": "user", "content": "How do I deploy?"},
        {
            "role": "assistant",
            "content": (
                "Run the deploy script from the release branch.\n"
                "Never deploy on a Friday afternoon."
            ),
        },
        {"role": "user", "content": "Thanks"},
    ]

    texts = conversation_archive.branch_message_texts(edited_with_a_later_echo)
    assert all(conversation_archive._on_live_branch(row["text"], texts) for row in rows)
    assert conversation_archive._document_matches_one_run(rows, texts) is False
    assert (
        conversation_archive._document_matches_one_run(
            rows, conversation_archive.branch_message_texts(intact)
        )
        is True
    )


def test_a_turns_CHUNKS_cannot_spill_into_the_message_after_the_turn(conn):
    """A turn's run is bounded by its messages, not its lines, so a tail can't match past the turn."""
    rows = [
        {
            "text": "user: how do I deploy\nassistant: Run the deploy script from the release branch."
        },
        {"text": "Never deploy on a Friday afternoon."},
    ]
    edited = [
        {"role": "user", "content": "how do I deploy"},
        {"role": "assistant", "content": "Run the deploy script from the release branch."},
        {"role": "user", "content": "Never deploy on a Friday afternoon."},
    ]
    intact = [
        {"role": "user", "content": "how do I deploy"},
        {
            "role": "assistant",
            "content": (
                "Run the deploy script from the release branch.\n"
                "Never deploy on a Friday afternoon."
            ),
        },
        {"role": "user", "content": "thanks"},
    ]

    assert (
        conversation_archive._document_matches_one_run(
            rows, conversation_archive.branch_message_texts(edited)
        )
        is False
    )
    assert (
        conversation_archive._document_matches_one_run(
            rows, conversation_archive.branch_message_texts(intact)
        )
        is True
    )


def test_a_search_the_MODEL_asked_for_is_not_archived_as_new_history():
    """A model's search_conversation call is removed by name, since its id looks ordinary."""
    from core.rag import conversation_archive as archive

    recalled = [
        _assistant_call("search_conversation", '{"query":"pass"}', id = "call_0"),
        {"role": "tool", "tool_call_id": "call_0", "content": "<chunk>RETRIEVEDPASSAGE</chunk>"},
        {"role": "assistant", "content": "It was ZQXVARA123."},
    ]

    rendered = archive.render_turn(archive._archivable(recalled))
    assert "RETRIEVEDPASSAGE" not in rendered
    assert "ZQXVARA123" in rendered

    mixed = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_0", "function": {"name": "search_conversation", "arguments": "{}"}},
                {"id": "call_1", "function": {"name": "terminal", "arguments": '{"cmd":"ls"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "<chunk>RETRIEVEDPASSAGE</chunk>"},
        {"role": "tool", "tool_call_id": "call_1", "content": "total 12"},
    ]

    mixed_rendered = archive.render_turn(archive._archivable(mixed))
    assert "RETRIEVEDPASSAGE" not in mixed_rendered
    assert "terminal" in mixed_rendered
    assert "total 12" in mixed_rendered


def test_a_folded_retrieval_result_is_still_kept_out_of_the_archive():
    """Folded retrieval results lose their ids; the cut must keep the question merged after them."""
    from core.inference.anthropic_compat import fold_tool_results_into_user
    from core.rag import conversation_archive as archive
    from routes.inference import _coalesce_consecutive_user_turns

    recalled = [
        {"role": "user", "content": "what did we say?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "function": {"name": "search_conversation", "arguments": '{"query":"pass"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "name": "search_conversation",
            "content": "<chunk>RETRIEVEDPASSAGE</chunk>",
        },
        {"role": "user", "content": "ASKEDAFTERWARDS?"},
        {"role": "assistant", "content": "It was ZQXVARA123."},
    ]
    folded = _coalesce_consecutive_user_turns(fold_tool_results_into_user(recalled))

    rendered = archive.render_turn(archive._archivable(folded))
    assert "RETRIEVEDPASSAGE" not in rendered
    assert "ASKEDAFTERWARDS?" in rendered
    assert "ZQXVARA123" in rendered

    recalled[2]["content"] = [{"type": "text", "text": "<chunk>RETRIEVEDPASSAGE</chunk>"}]
    listed = _coalesce_consecutive_user_turns(fold_tool_results_into_user(recalled))
    rendered = archive.render_turn(archive._archivable(listed))
    assert "RETRIEVEDPASSAGE" not in rendered
    assert "ASKEDAFTERWARDS?" in rendered

    with_image = [
        {"role": "user", "content": "what did we say?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_0", "function": {"name": "search_conversation", "arguments": "{}"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "name": "search_conversation",
            "content": "<chunk>RETRIEVEDPASSAGE</chunk>",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "ASKEDWITHIMAGE?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        },
    ]
    folded_image = _coalesce_consecutive_user_turns(fold_tool_results_into_user(with_image))
    assert isinstance(folded_image[-1]["content"], list), folded_image[-1]
    kept_image = archive._archivable(folded_image)
    dumped = json.dumps(kept_image)
    assert "RETRIEVEDPASSAGE" not in dumped
    assert "ASKEDWITHIMAGE?" in dumped
    assert "image_url" in dumped

    # A user typing the fold's JSON shape keeps their words: no retrieval call, nothing to strip.
    typed = [
        {
            "role": "user",
            "content": "why this shape?\n\n"
            + json.dumps(
                {
                    "tool_response": {
                        "tool": "search_conversation",
                        "content": "MYOWNWORDS",
                        "tool_call_id": "call_0",
                    }
                },
                indent = 2,
            ),
        },
        {"role": "assistant", "content": "because of the fold."},
    ]
    assert "MYOWNWORDS" in json.dumps(archive._archivable(typed))

    kept = fold_tool_results_into_user(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "terminal", "arguments": '{"cmd":"ls"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "terminal", "content": "total 12"},
        ]
    )
    assert "total 12" in archive.render_turn(archive._archivable(kept))


def test_swapping_the_tool_retires_the_archived_call():
    """The tool name is part of the turn's meaning, so a probe must include it, not just the arguments."""
    from core.rag import conversation_archive as archive

    archived = archive.render_turn(
        [
            _assistant_call("terminal", '{"cmd":"ls -la /srv"}'),
            {"role": "tool", "tool_call_id": "c1", "content": "total 12"},
        ]
    )

    def _branch(tool):
        return [
            archive._normalise(archive._probe_text(message))
            for message in [
                _assistant_call(tool, '{"cmd":"ls -la /srv"}'),
                {"role": "tool", "tool_call_id": "c1", "content": "total 12"},
            ]
        ]

    assert archive._on_live_branch(archived, _branch("terminal")) is True
    assert archive._on_live_branch(archived, _branch("python")) is False


def test_the_tool_call_exemption_ends_where_the_call_does():
    """The exemption from exact anchors belongs to the call only; text after it is matched exactly."""
    from core.rag import conversation_archive as archive

    probes = [("search_conversation", True), ("old answer", False)]

    def _eligible(message):
        found = archive._scan_probes(probes, [message], 0, 1)
        if found is None:
            return False
        position, cursor, opened_at, partial, _opened_index = found
        return not opened_at and (partial or cursor >= len([message][position]))

    assert _eligible('{"tool":"search_conversation"}\nold answer') is True
    assert _eligible('{"tool":"search_conversation"}\nold answer, correction: new answer') is False


def test_a_line_inserted_INTO_an_archived_turn_retires_it():
    """A line inserted between two archived lines is still an edit, and retires the turn."""
    from core.rag import conversation_archive as archive

    probes = [("A: drain traffic", False), ("B: flip the flag", False)]

    assert archive._scan_probes(probes, ["A: drain traffic\nB: flip the flag"], 0, 1) is not None
    assert (
        archive._scan_probes(
            probes, ["A: drain traffic\ncorrection: hold on\nB: flip the flag"], 0, 1
        )
        is None
    )
    assert (
        archive._scan_probes(probes, ["A: drain traffic\nuser: B: flip the flag"], 0, 1) is not None
    )


def test_a_pasted_transcript_cannot_widen_a_turns_run(conn):
    """Pasted transcript labels must not widen a turn's run; the real size is recorded at archive time."""
    rows = [
        {"text": "user: look at this log\nassistant: Here is what it says:\nuser: hello there"},
        {"text": "The fix is to restart the worker."},
    ]
    edited = conversation_archive.branch_message_texts(
        [
            {"role": "user", "content": "look at this log"},
            {"role": "assistant", "content": "Here is what it says:\nuser: hello there"},
            {"role": "user", "content": "The fix is to restart the worker."},
        ]
    )
    intact = conversation_archive.branch_message_texts(
        [
            {"role": "user", "content": "look at this log"},
            {
                "role": "assistant",
                "content": (
                    "Here is what it says:\nuser: hello there\nThe fix is to restart the worker."
                ),
            },
        ]
    )

    assert conversation_archive._document_matches_one_run(rows, edited, 2) is False
    assert conversation_archive._document_matches_one_run(rows, intact, 2) is True


def test_an_archived_turn_records_how_many_messages_it_came_from(conn):
    """The count has to reach the database, or the bound falls back on every recall."""
    from core.rag import store

    thread_id = "sized-archive"
    _save_thread(thread_id, [{"role": "user", "content": "hi"}])
    written = conversation_archive.archive_turns(
        thread_id,
        [
            {"role": "user", "content": "what is the deploy code"},
            {"role": "assistant", "content": "the deploy code is 5150"},
        ],
    )
    assert written == 1

    row = conn.execute(
        "SELECT archive_messages FROM documents WHERE scope = ?",
        (store.conversation_archive_scope(thread_id),),
    ).fetchone()
    assert row["archive_messages"] == 2


def test_one_turn_archived_twice_at_once_is_stored_once(conn, monkeypatch):
    """The hash check is not atomic with the insert, so concurrent compactions can both write a turn."""
    import threading

    from core.rag import embeddings, store
    from storage import studio_db

    thread_id = "concurrent-archive"
    studio_db.upsert_chat_thread(
        {
            "id": thread_id,
            "title": "t",
            "modelType": "base",
            "modelId": "m",
            "createdAt": 1,
        }
    )
    studio_db.sync_chat_messages(
        thread_id,
        [
            {
                "id": "m1",
                "threadId": thread_id,
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
                "createdAt": 2,
            }
        ],
    )

    barrier = threading.Barrier(2)
    real_encode = embeddings.encode_with_identity

    def slow_encode(texts, **kwargs):
        barrier.wait(timeout = 10)
        return real_encode(texts, **kwargs)

    monkeypatch.setattr(embeddings, "encode_with_identity", slow_encode)

    evicted = [
        {"role": "user", "content": "what is the deploy code"},
        {"role": "assistant", "content": "the deploy code is 5150"},
    ]
    workers = [
        threading.Thread(
            target = lambda: conversation_archive.archive_turns(
                thread_id, [dict(message) for message in evicted]
            )
        )
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    rows = conn.execute(
        "SELECT COUNT(*) AS c FROM documents WHERE scope = ?",
        (store.conversation_archive_scope(thread_id),),
    ).fetchone()
    assert rows["c"] == 1


def test_a_LATER_turn_cannot_supply_a_line_the_edit_removed(conn):
    """The branch check must stay inside its own turn, or a later message can satisfy a missing line."""
    archived = conversation_archive.render_turn(
        [{"role": "user", "content": "Should I deploy?"}, {"role": "assistant", "content": "No"}]
    )
    edited_away = [
        {"role": "user", "content": "Should I deploy?"},
        {"role": "assistant", "content": "Yes"},
        {"role": "user", "content": "Is the staging queue busy?"},
        {"role": "assistant", "content": "No"},
    ]
    still_there = [
        {"role": "user", "content": "Should I deploy?"},
        {"role": "assistant", "content": "No"},
        {"role": "user", "content": "Is the staging queue busy?"},
        {"role": "assistant", "content": "Yes"},
    ]

    assert (
        conversation_archive._on_live_branch(
            archived, conversation_archive.branch_message_texts(edited_away)
        )
        is False
    )
    assert (
        conversation_archive._on_live_branch(
            archived, conversation_archive.branch_message_texts(still_there)
        )
        is True
    )


def test_a_turn_whose_lines_were_REORDERED_is_no_longer_on_the_branch(conn):
    """Line membership alone accepts a rearranged turn; probes must match in order, not merely occur."""
    original = [{"role": "assistant", "content": "REORDER-A first\nREORDER-B second"}]
    archived = conversation_archive.render_turn(original)

    same = conversation_archive.branch_message_texts(original)
    swapped = conversation_archive.branch_message_texts(
        [{"role": "assistant", "content": "REORDER-B second\nREORDER-A first"}]
    )

    assert conversation_archive._on_live_branch(archived, same) is True
    assert conversation_archive._on_live_branch(archived, swapped) is False


def test_a_tool_turn_with_BOTH_text_and_a_call_stays_on_its_branch(conn):
    """Both transcript shapes must order a tool turn as call, text, then result, or matching fails."""
    request_shape = [
        {"role": "user", "content": "check the log"},
        _assistant_call("terminal", '{"cmd":"cat log"}', content = "I will read it now"),
        {"role": "tool", "tool_call_id": "c1", "content": "log contents here"},
    ]
    archived = conversation_archive.render_turn(request_shape)
    stored_shape = [
        {"role": "user", "content": [{"type": "text", "text": "check the log"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I will read it now"},
                {
                    "type": "tool-call",
                    "toolName": "terminal",
                    "args": {"cmd": "cat log"},
                    "result": "log contents here",
                },
            ],
        },
    ]

    assert conversation_archive._on_live_branch(
        archived, conversation_archive.branch_message_texts(request_shape)
    )
    assert conversation_archive._on_live_branch(
        archived, conversation_archive.branch_message_texts(stored_shape)
    )


def test_editing_ONE_chunk_of_a_long_turn_retires_the_whole_turn(conn):
    """Editing one chunk of a long turn retires the whole turn; the archived unit is the turn."""
    # The head spans several chunks, so the first chunk passes a per-chunk check unchanged.
    head = "opening CHUNKSPLIT-7373 marker. " + ("unchanged opening sentence. " * 300)
    turn = [
        {"role": "user", "content": "explain the deploy process"},
        {"role": "assistant", "content": head + ("original ending sentence. " * 300)},
    ]
    _save_thread("chunk-thread", turn, append = True)
    assert conversation_archive.archive_turns("chunk-thread", turn) == 1
    scope = store.conversation_archive_scope("chunk-thread")
    document_ids = {
        row["document_id"]
        for row in conversation_archive.rag_db.get_connection()
        .execute(
            "SELECT document_id FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE d.scope = ?",
            (scope,),
        )
        .fetchall()
    }
    chunk_count = (
        conversation_archive.rag_db.get_connection()
        .execute(
            "SELECT COUNT(*) AS n FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE d.scope = ?",
            (scope,),
        )
        .fetchone()["n"]
    )
    assert len(document_ids) == 1 and chunk_count > 1

    rewritten = [
        turn[0],
        {"role": "assistant", "content": head + ("a completely different ending. " * 300)},
    ]
    first_chunk = (
        conversation_archive.rag_db.get_connection()
        .execute(
            "SELECT c.text FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE d.scope = ? ORDER BY c.chunk_index ASC LIMIT 1",
            (scope,),
        )
        .fetchone()["text"]
    )
    assert conversation_archive._on_live_branch(
        first_chunk, conversation_archive.branch_message_texts(rewritten)
    )

    found = conversation_archive.recall(
        "chunk-thread", "CHUNKSPLIT-7373", branch_messages = rewritten
    )

    assert found is None


def test_a_long_multi_line_tool_result_stays_on_its_branch(conn):
    """Capped tool results are one multi-line string; the marker on the last line broke the branch check."""
    body = "opening line TOOLWALL-6060\n" + ("filler output line\n" * 400) + "trailing line"
    group = [
        _assistant_call("terminal", '{"cmd": "cat log"}'),
        {"role": "tool", "tool_call_id": "c1", "content": body},
    ]
    text = conversation_archive.render_turn(group)
    assert text.endswith("...")
    assert len(text.splitlines()) > 100

    transcript = conversation_archive.branch_message_texts(group)

    assert conversation_archive._on_live_branch(text, transcript) is True
    edited = conversation_archive.branch_message_texts(
        [
            group[0],
            {**group[1], "content": body.replace("opening line", "rewritten line")},
        ]
    )
    assert conversation_archive._on_live_branch(text, edited) is False


def test_recall_widens_past_a_wall_of_abandoned_branch_hits(conn):
    """A fixed over-fetch fills with abandoned-branch hits, so recall must widen until live matches show."""
    live = _turn(
        "where is the marker",
        "the marker is WIDEN-5150 and the rest of this answer is about unrelated matters "
        "such as scheduling, packaging, release notes and the weather in three cities",
    )
    abandoned = [
        _turn(
            f"where is the marker attempt {index}",
            f"WIDEN-5150 WIDEN-5150 WIDEN-5150 attempt {index} discarded",
        )
        for index in range(40)
    ]

    _save_thread("wall-thread", live, append = True)
    conversation_archive.archive_turns("wall-thread", live)
    for turn in abandoned:
        _save_thread("wall-thread", turn, append = True)
        conversation_archive.archive_turns("wall-thread", turn)

    found = conversation_archive.recall("wall-thread", "WIDEN-5150", branch_messages = live)

    assert found is not None
    assert "the marker is WIDEN-5150" in found[0]
    assert "discarded" not in found[0]


def test_the_branch_transcript_carries_request_shaped_tool_calls(conn):
    """Tool arguments live in tool_calls, not content, so a branch built from content misses them."""
    branch = [
        _assistant_call("terminal", '{"cmd": "ls TOOLARG-7777"}', id = "call_1"),
        {"role": "tool", "tool_call_id": "call_1", "content": "TOOLARG-7777 listed"},
    ]
    _archive(branch, thread_id = "tool-branch-thread")
    _save_thread("tool-branch-thread", [{"role": "user", "content": "unrelated"}])

    found = conversation_archive.recall(
        "tool-branch-thread", "TOOLARG-7777", branch_messages = branch
    )

    assert found is not None and "TOOLARG-7777" in found[0]


def test_a_thread_that_was_never_persisted_is_never_archived(conn):
    """An incognito chat must never be archived: the request carries its thread_id and no incognito flag."""
    written = _archive(
        [
            {"role": "user", "content": "temporary section, code EPHEMERAL-4444"},
            {"role": "assistant", "content": "noted"},
        ],
        thread_id = "temporary-thread",
        persist = False,
    )

    assert written == 0
    assert conversation_archive.has_archive("temporary-thread") is False
    assert conversation_archive.recall("temporary-thread", "EPHEMERAL-4444") is None
    assert conversation_archive.can_archive("temporary-thread") is False


def test_a_thread_deleted_mid_ingest_does_not_leave_its_turns_behind(conn, monkeypatch):
    """Deleting a chat mid-compaction must not resurrect its archive; the commit can undo the sweep."""
    from storage import studio_db

    turns = _turn("what is the code", "the code is DELETED-9999")
    _save_thread("doomed-thread", turns, append = True)

    original = conversation_archive.embeddings.encode_with_identity

    def delete_the_thread_mid_ingest(*args, **kwargs):
        studio_db.delete_chat_threads(["doomed-thread"])
        conversation_archive.delete_for_thread("doomed-thread")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        conversation_archive.embeddings, "encode_with_identity", delete_the_thread_mid_ingest
    )

    written = conversation_archive.archive_turns("doomed-thread", turns)

    assert written == 0
    assert conversation_archive.has_archive("doomed-thread") is False
    assert conversation_archive.recall("doomed-thread", "DELETED-9999") is None


def test_recall_is_unfiltered_when_the_thread_has_no_saved_transcript(conn):
    """An empty transcript is not evidence the turns are gone, so recall must not be disabled."""
    _archive(
        [
            {"role": "user", "content": "unsaved section, code ORPHAN-3333"},
            {"role": "assistant", "content": "noted"},
        ],
        thread_id = "unsaved-thread",
    )
    from storage import studio_db

    studio_db.delete_chat_threads(["unsaved-thread"])

    found = conversation_archive.recall("unsaved-thread", "ORPHAN-3333")

    assert found is not None
    assert "ORPHAN-3333" in found[0]


def test_editing_only_the_assistant_half_retires_the_archived_turn(conn):
    """Match the whole turn, not its first line; an unchanged user line must not vouch for the answer."""
    original = [
        {"role": "user", "content": "what is the launch code"},
        {"role": "assistant", "content": "the launch code is STALEANSWER-9999"},
    ]
    _archive(original, thread_id = "edited-thread")
    _save_thread(
        "edited-thread",
        [
            {"role": "user", "content": "what is the launch code"},
            {"role": "assistant", "content": "the launch code is FRESHANSWER-1111"},
        ],
    )

    found = conversation_archive.recall("edited-thread", "STALEANSWER-9999")

    assert found is None or "STALEANSWER-9999" not in found[0]


def test_a_failed_chunk_write_leaves_the_turn_retryable(conn, monkeypatch):
    """A failed chunk write must not leave a completed but empty document, which would never be retried."""
    turns = _turn("what is a quokka", "a small marsupial")

    def explode(*args, **kwargs):
        raise RuntimeError("disk full")

    # Restored by re-setting: undo() would also revert the shared stub_embeddings fixture.
    real_add_chunks = store.add_chunks
    monkeypatch.setattr(store, "add_chunks", explode)
    assert _archive(turns, thread_id = "retry-thread") == 0
    monkeypatch.setattr(store, "add_chunks", real_add_chunks)

    assert _archive(turns, thread_id = "retry-thread", persist = False) == 1
    found = conversation_archive.recall("retry-thread", "quokka")
    assert found is not None and "quokka" in found[0]


def test_archived_tool_turns_keep_what_the_call_actually_did(conn):
    """ "assistant called terminal" cannot answer "what did you run earlier?"."""
    rendered = conversation_archive.render_turn(
        [
            {
                "role": "assistant",
                "content": "running the migration now",
                "tool_calls": [
                    {
                        "function": {
                            "name": "terminal",
                            "arguments": '{"command": "alembic upgrade head"}',
                        }
                    }
                ],
            },
            {"role": "tool", "content": "ok"},
        ]
    )

    assert "terminal" in rendered
    assert "alembic upgrade head" in rendered
    assert "running the migration now" in rendered


def test_an_archived_tool_turn_survives_the_branch_filter(conn):
    """Requiring every render_turn label in the transcript left no archived tool turn able to match."""
    from storage import studio_db

    turn = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command": "alembic upgrade head"}',
                    }
                }
            ],
        },
        {"role": "tool", "content": "migration applied cleanly"},
    ]
    studio_db.upsert_chat_thread(
        {
            "id": "tool-thread",
            "title": "t",
            "modelType": "base",
            "modelId": "local-model",
            "createdAt": 1,
        }
    )
    studio_db.upsert_chat_message(
        {
            "id": "tool-thread-0",
            "threadId": "tool-thread",
            "role": "assistant",
            "content": [
                _tool_part(command = "alembic upgrade head", result = "migration applied cleanly")
            ],
            "createdAt": 2,
        }
    )
    conversation_archive.archive_turns("tool-thread", turn)

    found = conversation_archive.recall("tool-thread", "alembic upgrade head")

    assert found is not None
    assert "alembic" in found[0]


def test_an_edit_past_the_probe_cutoff_still_retires_the_turn(conn):
    """A prefix probe cannot see an edit past its cut-off; the tail of a long answer must be matched."""
    head = "the deployment steps are as follows and here is the full detail " * 4
    original = [
        {"role": "user", "content": "how do I deploy"},
        {"role": "assistant", "content": head + "finally run OLDSTEP-7777"},
    ]
    _archive(original, thread_id = "tail-edit-thread")
    _save_thread(
        "tail-edit-thread",
        [
            {"role": "user", "content": "how do I deploy"},
            {"role": "assistant", "content": head + "finally run NEWSTEP-8888"},
        ],
    )

    found = conversation_archive.recall("tail-edit-thread", "OLDSTEP-7777")

    assert found is None or "OLDSTEP-7777" not in found[0]


def test_an_embedder_download_still_pending_is_logged_once_without_a_traceback(
    conn, monkeypatch, caplog
):
    """Each fit must not log the same traceback again, and recall still answers lexically."""
    import logging

    from core.rag import embeddings

    _archive(_turn("how do I bake sourdough", "start a starter"))

    def pending(*_args, **_kwargs):
        raise embeddings.EmbeddingModelDownloadRequiredError(
            "Embedding model 'unsloth/Qwen3-Embedding-0.6B' is not downloaded yet."
        )

    monkeypatch.setattr(embeddings, "token_counter", pending)
    monkeypatch.setattr(embeddings, "encode_with_identity", pending)
    monkeypatch.setattr(conversation_archive, "_EMBEDDER_PENDING_LOGGED", set(), raising = False)
    monkeypatch.setattr(conversation_archive, "_INGEST_FAILED", False)
    turn = _turn("what is the deploy code", "the deploy code is 5150")
    caplog.set_level(logging.INFO, logger = conversation_archive.logger.name)

    for _ in range(3):
        assert _archive([dict(m) for m in turn]) == 0
        assert conversation_archive.degraded() is True
        found = conversation_archive.recall(THREAD, "sourdough")
        assert found is not None and "sourdough" in found[0]

    records = [r for r in caplog.records if r.name == conversation_archive.logger.name]
    assert not [r for r in records if r.levelno >= logging.WARNING or r.exc_info]
    assert [r.getMessage() for r in records if "embedder_pending" in r.getMessage()] == [
        "conversation_archive.embedder_pending: "
        "Embedding model 'unsloth/Qwen3-Embedding-0.6B' is not downloaded yet."
    ]


def test_a_failed_archive_marks_the_feature_degraded(conn, monkeypatch):
    """And a later success clears it, so one bad moment is not permanent."""
    from core.rag import embeddings

    thread_id = "degraded-archive"
    _save_thread(thread_id, [{"role": "user", "content": "hi"}])
    turn = [
        {"role": "user", "content": "what is the deploy code"},
        {"role": "assistant", "content": "the deploy code is 5150"},
    ]

    assert conversation_archive.degraded() is False

    real = embeddings.encode_with_identity

    def no_embedder(*_args, **_kwargs):
        raise RuntimeError("no embedding model could be started")

    monkeypatch.setattr(embeddings, "encode_with_identity", no_embedder)
    assert conversation_archive.archive_turns(thread_id, [dict(m) for m in turn]) == 0
    assert conversation_archive.degraded() is True

    monkeypatch.setattr(embeddings, "encode_with_identity", real)
    assert conversation_archive.archive_turns(thread_id, [dict(m) for m in turn]) == 1
    assert conversation_archive.degraded() is False


def test_the_late_archive_cleanup_spares_a_recreated_thread(conn):
    """The late archive sweep after DELETE must spare a chat recreated under the same id."""
    from routes import chat_history
    from storage import studio_db

    thread_id = "recreated-thread"
    turns = _turn("what is the code", "the code is 5150")
    _save_thread(thread_id, turns, append = True)
    assert conversation_archive.archive_turns(thread_id, turns) == 1

    chat_history._remove_thread_rag_data([thread_id])

    assert conversation_archive.has_archive(thread_id) is True

    studio_db.delete_chat_threads([thread_id])
    chat_history._remove_thread_rag_data([thread_id])
    assert conversation_archive.has_archive(thread_id) is False


def test_an_answer_edited_by_appending_to_it_retires_the_archived_copy(conn):
    """Appending a correction to an answer must retire the archived copy, since its probes still match."""
    rows = [{"text": "user: should I deploy on Friday\nassistant: No"}]
    edited = conversation_archive.branch_message_texts(
        [
            {"role": "user", "content": "should I deploy on Friday"},
            {"role": "assistant", "content": "No, correction: yes, the freeze lifted"},
        ]
    )
    intact = conversation_archive.branch_message_texts(
        [
            {"role": "user", "content": "should I deploy on Friday"},
            {"role": "assistant", "content": "No"},
        ]
    )

    assert conversation_archive._document_matches_one_run(rows, edited, 2) is False
    assert conversation_archive._document_matches_one_run(rows, intact, 2) is True


def test_a_truncated_tool_result_may_still_end_mid_message(conn):
    """A truncated tool result is a prefix by design; the probe need not end where the message does."""
    marker = conversation_archive._TRUNCATION_MARKER
    rows = [{"text": "user: run it\ntool result: " + "x" * 100 + marker}]
    live = conversation_archive.branch_message_texts(
        [
            {"role": "user", "content": "run it"},
            {"role": "tool", "content": "x" * 400},
        ]
    )

    assert conversation_archive._document_matches_one_run(rows, live, 2) is True


def test_an_answer_edited_by_prepending_to_it_retires_the_archived_copy(conn):
    """Prepending a correction keeps the old text as a suffix; an end-only check still calls it live."""
    rows = [{"text": "user: should I deploy on Friday\nassistant: No"}]

    def _live(answer):
        return conversation_archive.branch_message_texts(
            [
                {"role": "user", "content": "should I deploy on Friday"},
                {"role": "assistant", "content": answer},
            ]
        )

    assert conversation_archive._document_matches_one_run(rows, _live("Correction: no"), 2) is False
    assert conversation_archive._document_matches_one_run(rows, _live("No"), 2) is True


def test_a_tool_exchange_archived_mid_request_is_recallable(conn):
    """A tool exchange archived mid-request is not in the client's messages, so it must stay recallable."""
    thread_id = "toolrun-thread"
    request_branch = [{"role": "user", "content": "find the deploy code in the repo"}]
    _save_thread(thread_id, request_branch, append = True)

    tool_exchange = [
        _assistant_call("grep", '{"q": "deploy"}'),
        {"role": "tool", "tool_call_id": "c1", "content": "config/deploy.yml: token ZQX-5150"},
    ]
    assert conversation_archive.archive_turns(thread_id, tool_exchange) == 1

    assert (
        conversation_archive.recall(thread_id, "ZQX-5150", top_k = 4, branch_messages = request_branch)
        is None
    )
    assert (
        conversation_archive.recall(
            thread_id,
            "ZQX-5150",
            top_k = 4,
            branch_messages = request_branch + tool_exchange,
        )
        is not None
    )


def test_an_edit_to_any_message_of_a_turn_retires_the_archived_copy(conn):
    """Every message of a turn must be anchored, not just the last, or the question stays editable."""
    rows = [{"text": "user: should I deploy on Friday\nassistant: No"}]

    def _live(question, answer = "No"):
        return conversation_archive.branch_message_texts(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )

    match = conversation_archive._document_matches_one_run
    assert match(rows, _live("Actually, should I deploy on Friday"), 2) is False
    assert match(rows, _live("should I deploy on Friday or wait"), 2) is False
    assert match(rows, _live("should I deploy on Friday", "Correction: no"), 2) is False
    assert match(rows, _live("should I deploy on Friday", "No, correction: yes"), 2) is False
    assert match(rows, _live("should I deploy on Friday"), 2) is True


def test_a_tool_call_message_is_exempt_from_the_character_anchors(conn):
    """A stored tool call cannot line up exactly with live text, so it is exempt from character anchors."""
    live = conversation_archive.branch_message_texts(
        [
            {
                "role": "assistant",
                "content": [
                    _tool_part(
                        command = "alembic upgrade head",
                        result = "migration applied cleanly",
                    )
                ],
            }
        ]
    )
    rows = [
        {
            "text": (
                'assistant called terminal: {"command": "alembic upgrade head"}\n'
                "tool result: migration applied cleanly"
            )
        }
    ]

    assert conversation_archive._document_matches_one_run(rows, live, 1) is True


def test_a_turn_is_re_embedded_when_the_embedder_changes(conn, monkeypatch):
    """A turn archived under an old embedder must be re-embedded, or dense search never finds it again."""
    from core.rag import embeddings, store

    thread_id = "identity-thread"
    turn = _turn("what is the deploy code", "the deploy code is 5150")
    _save_thread(thread_id, turn, append = True)

    identity = {"name": "st:model-a"}
    real = embeddings.encode_with_identity
    monkeypatch.setattr(
        embeddings,
        "encode_with_identity",
        lambda texts, **kwargs: (real(texts, **kwargs)[0], identity["name"]),
    )
    monkeypatch.setattr(embeddings, "embedding_identity", lambda *_a, **_k: identity["name"])

    assert conversation_archive.archive_turns(thread_id, [dict(m) for m in turn]) == 1
    identity["name"] = "st:model-b"
    assert conversation_archive.archive_turns(thread_id, [dict(m) for m in turn]) == 1

    rows = conn.execute(
        "SELECT embedding_model FROM documents WHERE scope = ?",
        (store.conversation_archive_scope(thread_id),),
    ).fetchall()
    assert [row["embedding_model"] for row in rows] == ["st:model-b"]

    assert conversation_archive.archive_turns(thread_id, [dict(m) for m in turn]) == 0


def test_a_first_compaction_embeds_its_turns_in_one_pass(conn, monkeypatch):
    """Embed a first compaction's turns in one pass; per-turn jobs serialise and delay the reply."""
    from core.rag import embeddings

    thread_id = "batch-thread"
    _save_thread(thread_id, _turn("hello", "hi"), append = True)

    calls = []
    real = embeddings.encode_with_identity

    def counted(texts, **kwargs):
        calls.append(len(texts))
        return real(texts, **kwargs)

    monkeypatch.setattr(embeddings, "encode_with_identity", counted)

    evicted = []
    for index in range(12):
        evicted.append({"role": "user", "content": f"question number {index} about the deploy"})
        evicted.append({"role": "assistant", "content": f"answer number {index}, code {index}"})

    assert conversation_archive.archive_turns(thread_id, evicted) == 12
    assert len(calls) == 1
    assert calls[0] >= 12


VARIABLE = "ZQXVARA123"


# Filler must VARY: the defect is an IDF collapse, and one repeated distractor hides it.
_DISTRACTORS = [
    (
        "What is a good default value for a retry budget?",
        "Three attempts with backoff is a common default.",
    ),
    ("Change the log level to debug for now.", "Log level is debug."),
    ("Remind me to update the deployment notes later.", "I will remind you."),
    ("Is it better to set a timeout per request or per session?", "Per request is usually safer."),
    (
        "Correction to my earlier note about the changelog wording.",
        "Noted, the changelog wording is corrected.",
    ),
    ("Which branch should the release notes land on?", "The release branch."),
]


def _revisions(
    count,
    thread_id = THREAD,
    *,
    distractors = 3,
):
    """Fixed values keep failures reproducible; filler never names the variable, so it can only compete."""
    values = [f"10000{index}" for index in range(count)]
    filler = 0
    for value in values:
        _archive(
            _turn(f"Set {VARIABLE} to {value}.", f"Understood. {VARIABLE} is {value}."), thread_id
        )
        for _ in range(distractors):
            question, answer = _DISTRACTORS[filler % len(_DISTRACTORS)]
            filler += 1
            _archive(_turn(f"{question} (note {filler})", answer), thread_id)
    return values


def test_every_recall_slot_goes_to_the_subject_of_the_question(conn):
    """Only turns naming the subject are candidates; BM25 decides among equal-scoring assignments."""
    values = _revisions(8)

    found = conversation_archive.recall(
        THREAD, f"What is the current value of {VARIABLE}?", top_k = 4
    )

    assert found is not None
    text, sources = found
    assert len(sources) == 4
    assert all(VARIABLE.lower() in source["text"].lower() for source in sources)
    assert any(value in text for value in values)


def test_the_newest_revision_is_recalled_when_there_is_room(conn):
    """With a slot per revision the newest must be there, and must be read last."""
    values = _revisions(4)

    found = conversation_archive.recall(
        THREAD, f"What is the current value of {VARIABLE}?", top_k = 4
    )

    assert found is not None
    text, _sources = found
    assert values[-1] in text
    assert max(text.index(value) for value in values if value in text) == text.index(values[-1])


def test_the_questions_filler_cannot_outrank_the_subject(conn):
    """One slot, and it must go to the turn about the thing asked about."""
    for index in range(6):
        _archive(_turn(f"Set {VARIABLE} to 42{index}.", f"Understood. {VARIABLE} is 42{index}."))
    _archive(
        _turn(
            "What is a good default value for a retry budget?",
            "Three attempts with backoff is a common default value.",
        )
    )

    found = conversation_archive.recall(
        THREAD, f"What is the current value of {VARIABLE}?", top_k = 1
    )

    assert found is not None
    assert VARIABLE.lower() in found[0].lower()


def test_the_archive_query_requires_the_rare_token_and_drops_filler():
    """The conjunctive pass first, the stopword-stripped OR second, and never nothing."""
    focused = store.conversation_match_queries(f"What is the current value of {VARIABLE}?")

    assert focused[0] == f'"{VARIABLE.lower()}"'
    assert '"current"' in focused[1] and '"value"' in focused[1]
    assert '"what"' not in focused[1] and '"the"' not in focused[1]
    filler = store.conversation_match_queries("what about it")
    assert filler and '"about"' in filler[0]
    assert store.conversation_match_queries("!!!") == []


def test_recalled_turns_are_presented_oldest_first(conn):
    """The model answers with the last assignment it reads, so the order IS the answer."""
    _archive(
        _turn(f"Set {VARIABLE} to 111111. " + "Some padding about the topic. " * 40, "Understood.")
    )
    _archive(_turn(f"{VARIABLE} 222222", "Understood."))

    found = conversation_archive.recall(
        THREAD, f"What is the current value of {VARIABLE}?", top_k = 2
    )

    assert found is not None
    text, sources = found
    assert text.index("111111") < text.index("222222")
    assert "supersedes" in text
    assert sources[0]["citationId"] == 1


def test_each_archived_turn_records_its_position(conn):
    _archive(_turn("first", "a"))
    _archive(_turn("second", "b"))
    _archive(_turn("third", "c"))

    scope = store.conversation_archive_scope(THREAD)
    ordinals = [
        row["archive_ordinal"]
        for row in conn.execute(
            "SELECT archive_ordinal FROM documents WHERE scope=? ORDER BY archive_ordinal",
            (scope,),
        ).fetchall()
    ]
    assert ordinals == [0, 1, 2]


def test_re_embedding_a_turn_keeps_its_place(conn, monkeypatch):
    """Re-embedding walks the whole archive, so a fresh ordinal here would renumber the
    entire conversation into the order its vectors were rebuilt."""
    from core.rag import embeddings

    # Patch embedding_identity too: archive_turns short-circuits on the expected identity.
    identity = {"name": "st:model-a"}
    real = embeddings.encode_with_identity
    monkeypatch.setattr(
        embeddings,
        "encode_with_identity",
        lambda texts, **kwargs: (real(texts, **kwargs)[0], identity["name"]),
    )
    monkeypatch.setattr(embeddings, "embedding_identity", lambda *_a, **_k: identity["name"])

    first = _turn("the oldest turn", "a")
    assert _archive([dict(message) for message in first]) == 1
    assert _archive(_turn("a later turn", "b")) == 1
    identity["name"] = "st:model-b"
    assert _archive([dict(message) for message in first]) == 1

    scope = store.conversation_archive_scope(THREAD)
    rows = conn.execute(
        "SELECT filename, archive_ordinal FROM documents WHERE scope=? ORDER BY archive_ordinal",
        (scope,),
    ).fetchall()
    assert [row["archive_ordinal"] for row in rows] == [0, 1]


def test_an_archive_written_before_ordinals_still_recalls_in_order(conn):
    """NULL ordinals predate the column, so they sort first rather than not at all."""
    _archive(_turn("the pelican turn", "older"))
    _archive(_turn("the pelican answer", "newer"))
    scope = store.conversation_archive_scope(THREAD)
    oldest = conn.execute(
        "SELECT id FROM documents WHERE scope=? ORDER BY archive_ordinal", (scope,)
    ).fetchone()["id"]
    conn.execute("UPDATE documents SET archive_ordinal=NULL WHERE id=?", (oldest,))
    conn.commit()

    found = conversation_archive.recall(THREAD, "pelican", top_k = 2)

    assert found is not None
    text, _sources = found
    assert text.index("older") < text.index("newer")
    assert 'turn="1"' not in text
    assert 'turn="2"' in text


def test_asking_what_it_was_originally_still_returns_the_first_assignment(conn):
    """The guard against a fix that just prefers the newest thing it can find."""
    values = _revisions(8)

    found = conversation_archive.recall(
        THREAD, f"What was {VARIABLE} set to at the very start?", top_k = 4
    )

    assert found is not None
    text, sources = found
    assert values[0] in text
    assert values[0] in sources[0]["text"]


def test_relevance_order_is_restored_when_the_knobs_are_off(conn, monkeypatch):
    """The off setting has to reproduce the previous build, not approximate it."""
    from core.rag import config

    monkeypatch.setattr(config, "CONVERSATION_QUERY_FOCUS", False)
    monkeypatch.setattr(config, "CONVERSATION_RECALL_ORDER", "relevance")
    _archive(
        _turn(f"Set {VARIABLE} to 111111. " + "Some padding about the topic. " * 40, "Understood.")
    )
    _archive(_turn(f"{VARIABLE} 222222", "Understood."))

    found = conversation_archive.recall(
        THREAD, f"What is the current value of {VARIABLE}?", top_k = 2
    )

    assert found is not None
    text, _sources = found
    assert "111111" in text and "222222" in text
    assert "turn=" not in text
    assert "supersedes" not in text
    assert "oldest first" not in text


def test_a_ubiquitous_identifier_cannot_crowd_out_the_newest_revision(conn):
    """A conjunctive filter must not also rank or fill fetch: FTS5 floors IDF for ubiquitous terms."""
    for index in range(19):
        _archive(
            _turn(
                f"lets discuss {VARIABLE} aspect number {index}",
                f"{VARIABLE} is a config knob, remark {index} about how {VARIABLE} behaves",
            )
        )
    _archive(_turn(f"please update {VARIABLE}", f"the current value of {VARIABLE} is now 991234"))

    found = conversation_archive.recall(THREAD, f"what is the current value of {VARIABLE}?")

    assert found is not None
    text, _sources = found
    assert "991234" in text


def test_a_question_about_two_variables_recalls_both_current_values(conn):
    """Two identifiers must not both be required: the current-value turn names only one of them."""
    other = "ZQXVARB456"
    for index in range(6):
        _archive(
            _turn(
                f"How does {VARIABLE} compare with {other} in scenario {index}?",
                f"In scenario {index}, {VARIABLE} and {other} trade off differently.",
            )
        )
    _archive(_turn(f"please update {VARIABLE}", f"the current value of {VARIABLE} is now 700001"))
    _archive(_turn(f"please update {other}", f"the current value of {other} is now 800002"))

    found = conversation_archive.recall(
        THREAD, f"What is the current value of {VARIABLE} and of {other}?", top_k = 4
    )

    assert found is not None
    text, sources = found
    assert "700001" in text and "800002" in text
    assert all(
        VARIABLE.lower() in source["text"].lower() or other.lower() in source["text"].lower()
        for source in sources
    )


def test_the_newest_revision_survives_a_strict_pass_that_hit_its_cap(conn, monkeypatch):
    """Past the candidate cap, absence from the identifier pass proves nothing about the subject."""
    monkeypatch.setattr(conversation_archive, "_BRANCH_FILTER_MAX_CANDIDATES", 16)
    for index in range(19):
        _archive(
            _turn(
                f"lets discuss {VARIABLE} aspect number {index}",
                f"{VARIABLE} is a config knob, remark {index} about how {VARIABLE} behaves",
            )
        )
    _archive(_turn(f"please update {VARIABLE}", f"the current value of {VARIABLE} is now 991234"))

    found = conversation_archive.recall(THREAD, f"what is the current value of {VARIABLE}?")

    assert found is not None
    assert "991234" in found[0]


_CONTENT_WORD_TURNS = [
    ("What is the current value of the retry budget?", "Three attempts, by default."),
    ("Is the timeout value still 30 seconds?", "Yes, that is the current setting."),
    ("What value should the batch size take?", "Whatever the current GPU allows."),
    ("Remind me of the current log level.", "Debug, as of the last change."),
    ("Does the cache TTL have a sensible value?", "The current one is an hour."),
    ("What is the current default for max tokens?", "The value is 2048."),
    ("Is that value configurable at runtime?", "Yes, the current build reads it live."),
    ("What is the current branch protection rule?", "One review, no stale value."),
]


def test_the_ranking_pass_is_widened_until_it_has_eligible_chunks_to_order(conn):
    """The content-word pass ranks only eligible chunks, so its window must reach them rather than fetch."""
    for index in range(20):
        _archive(
            _turn(
                f"lets discuss {VARIABLE} aspect number {index}",
                f"{VARIABLE} is a config knob, remark {index} about how {VARIABLE} behaves",
            )
        )
    _archive(
        _turn(
            f"please update {VARIABLE}",
            f"the current value of {VARIABLE} is now 991234. I bumped it after the load test "
            "showed the old setting was too low for the nightly job, so please redeploy the "
            "workers before the next run and keep an eye on the queue depth for the first hour",
        )
    )
    for index in range(20):
        question, answer = _CONTENT_WORD_TURNS[index % len(_CONTENT_WORD_TURNS)]
        _archive(_turn(f"{question} (note {index})", answer))

    found = conversation_archive.recall(THREAD, f"what is the current value of {VARIABLE}?")

    assert found is not None
    text, sources = found
    assert "991234" in text
    assert all(VARIABLE.lower() in source["text"].lower() for source in sources)


def test_the_archive_query_keeps_the_negation_that_carries_the_question(conn):
    """ "What did I say NOT to delete" is only that question while `not` survives."""
    assert '"not"' in store.conversation_match_queries("What did I say not to delete?")[0]

    _archive(_turn("Please do not delete the staging bucket, ever.", "Understood."))
    for question, answer in (
        ("delete the old build artifacts in dist", "Removed the dist folder."),
        ("can you delete the unused import in main.py", "Import removed."),
        ("delete the stale feature branch", "Branch deleted."),
        ("I want you to delete the temp uploads folder", "Temp uploads cleared."),
        ("delete every log older than a week", "Old logs cleared."),
        ("please delete the duplicated test file", "Duplicate test removed."),
        ("delete the leftover docker volumes", "Volumes pruned."),
        ("delete the commented out block in config", "Block removed."),
    ):
        _archive(_turn(question, answer))

    found = conversation_archive.recall(THREAD, "What did I say not to delete?", top_k = 4)

    assert found is not None
    assert "staging bucket" in found[0]


def test_re_embedding_a_turn_archived_before_ordinals_leaves_it_unnumbered(conn, monkeypatch):
    """A NULL ordinal predates the column. Allocating one on re-embed would move the
    conversation's OLDEST turn behind every numbered one, and the renderer would then
    read it as the later, superseding statement."""
    from core.rag import embeddings

    identity = {"name": "st:model-a"}
    real = embeddings.encode_with_identity
    monkeypatch.setattr(
        embeddings,
        "encode_with_identity",
        lambda texts, **kwargs: (real(texts, **kwargs)[0], identity["name"]),
    )
    monkeypatch.setattr(embeddings, "embedding_identity", lambda *_a, **_k: identity["name"])

    oldest = _turn("the pelican turn", "the oldest statement about pelicans")
    assert _archive([dict(message) for message in oldest]) == 1
    scope = store.conversation_archive_scope(THREAD)
    conn.execute("UPDATE documents SET archive_ordinal=NULL WHERE scope=?", (scope,))
    conn.commit()
    assert _archive(_turn("newest pelican question", "the newest statement about pelicans")) == 1

    identity["name"] = "st:model-b"
    assert _archive([dict(message) for message in oldest]) == 1

    # Re-embed keeps the original timestamp, so the unnumbered turn is still the oldest row.
    ordinals = [
        row["archive_ordinal"]
        for row in conn.execute(
            "SELECT archive_ordinal FROM documents WHERE scope=? ORDER BY created_at", (scope,)
        ).fetchall()
    ]
    assert ordinals == [None, 1]
    text, _sources = conversation_archive.recall(THREAD, "pelicans", top_k = 4)
    assert text.index("oldest statement") < text.index("newest statement")


def test_a_re_embed_that_stops_partway_does_not_reorder_a_legacy_archive(conn, monkeypatch):
    """A re-embed that stops partway must not re-stamp legacy rows, or the oldest turns are quoted last."""
    from core.rag import embeddings

    identity = {"name": "st:model-a"}
    real = embeddings.encode_with_identity
    monkeypatch.setattr(
        embeddings,
        "encode_with_identity",
        lambda texts, **kwargs: (real(texts, **kwargs)[0], identity["name"]),
    )
    monkeypatch.setattr(embeddings, "embedding_identity", lambda *_a, **_k: identity["name"])

    turns = [_turn(f"turn {n} about pelicans", f"STATEMENT{n} about pelicans") for n in range(1, 6)]
    history = [dict(message) for turn in turns for message in turn]
    _save_thread(THREAD, history, append = True)
    assert conversation_archive.archive_turns(THREAD, [dict(m) for m in history]) == 5
    scope = store.conversation_archive_scope(THREAD)
    conn.execute("UPDATE documents SET archive_ordinal=NULL WHERE scope=?", (scope,))
    conn.commit()

    identity["name"] = "st:model-b"
    real_add = store.add_chunks
    calls = {"n": 0}

    def add_chunks_until_the_disk_fills(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("database or disk is full")
        return real_add(*args, **kwargs)

    monkeypatch.setattr(store, "add_chunks", add_chunks_until_the_disk_fills)
    conversation_archive.archive_turns(THREAD, [dict(m) for m in history])
    monkeypatch.setattr(store, "add_chunks", real_add)
    models = [
        row["embedding_model"]
        for row in conn.execute(
            "SELECT embedding_model FROM documents WHERE scope=? ORDER BY created_at", (scope,)
        ).fetchall()
    ]
    assert sorted(models) == ["st:model-a"] * 3 + ["st:model-b"] * 2

    text, sources = conversation_archive.recall(THREAD, "pelicans", top_k = 5)
    assert "supersedes" in text
    quoted = [source["text"].split("STATEMENT")[1][0] for source in sources]
    assert quoted == ["1", "2", "3", "4", "5"], quoted


def test_a_legacy_archive_written_in_one_clock_tick_is_still_ordered(conn, monkeypatch):
    """A clock tie with no ordinal makes a stable sort quote turns in relevance order, not oldest first."""
    from core.rag import embeddings

    monkeypatch.setattr(store, "_now", lambda: "2026-01-01T00:00:00+00:00")

    identity = {"name": "st:model-a"}
    real = embeddings.encode_with_identity
    monkeypatch.setattr(
        embeddings,
        "encode_with_identity",
        lambda texts, **kwargs: (real(texts, **kwargs)[0], identity["name"]),
    )
    monkeypatch.setattr(embeddings, "embedding_identity", lambda *_a, **_k: identity["name"])

    turns = [_turn(f"turn {n} about pelicans", f"STATEMENT{n} about pelicans") for n in range(1, 6)]
    history = [dict(message) for turn in turns for message in turn]
    _save_thread(THREAD, history, append = True)
    assert conversation_archive.archive_turns(THREAD, [dict(m) for m in history]) == 5
    scope = store.conversation_archive_scope(THREAD)
    conn.execute("UPDATE documents SET archive_ordinal=NULL WHERE scope=?", (scope,))
    conn.commit()
    stamps = {
        row["created_at"]
        for row in conn.execute("SELECT created_at FROM documents WHERE scope=?", (scope,))
    }
    assert stamps == {"2026-01-01T00:00:00+00:00"}, stamps

    identity["name"] = "st:model-b"
    real_add = store.add_chunks
    calls = {"n": 0}

    def add_chunks_until_the_disk_fills(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("database or disk is full")
        return real_add(*args, **kwargs)

    monkeypatch.setattr(store, "add_chunks", add_chunks_until_the_disk_fills)
    conversation_archive.archive_turns(THREAD, [dict(m) for m in history])
    monkeypatch.setattr(store, "add_chunks", real_add)

    _text, sources = conversation_archive.recall(THREAD, "pelicans", top_k = 5)
    quoted = [source["text"].split("STATEMENT")[1][0] for source in sources]
    assert quoted == ["1", "2", "3", "4", "5"], quoted


def test_two_turns_stamped_alike_are_quoted_whole_and_not_interleaved(conn, monkeypatch):
    """Tied turns must group by document, since chunk_index is only a position within one document."""
    monkeypatch.setattr(config, "CHUNK_TOKENS", 30)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 0)
    monkeypatch.setattr(store, "_now", lambda: "2026-01-01T00:00:00+00:00")

    def _long_turn(tag):
        return _turn(
            f"turn {tag} about pelicans",
            f"{tag}HEAD pelicans at the opening "
            + " ".join(f"w{index}" for index in range(25))
            + f" {tag}TAIL pelicans at the closing "
            + " ".join(f"z{index}" for index in range(25)),
        )

    history = [dict(message) for tag in ("AAA", "BBB") for message in _long_turn(tag)]
    _save_thread(THREAD, history, append = True)
    assert conversation_archive.archive_turns(THREAD, [dict(m) for m in history]) == 2
    scope = store.conversation_archive_scope(THREAD)
    conn.execute("UPDATE documents SET archive_ordinal=NULL WHERE scope=?", (scope,))
    conn.commit()
    assert {
        row["created_at"]
        for row in conn.execute("SELECT created_at FROM documents WHERE scope=?", (scope,))
    } == {"2026-01-01T00:00:00+00:00"}
    per_document = [
        row["n"]
        for row in conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE scope=? GROUP BY document_id", (scope,)
        )
    ]
    assert min(per_document) > 1, per_document

    _text, sources = conversation_archive.recall(THREAD, "pelicans", top_k = 8)

    documents = [source["documentId"] for source in sources]
    runs = [
        document
        for index, document in enumerate(documents)
        if index == 0 or documents[index - 1] != document
    ]
    assert len(runs) == len(set(documents)) == 2, documents
    for document in runs:
        indexes = [s["chunkIndex"] for s in sources if s["documentId"] == document]
        assert indexes == sorted(indexes), (document, indexes)
    assert "AAAHEAD" in sources[0]["text"], sources[0]["text"]


def test_the_sql_candidate_order_agrees_with_the_python_recall_order(conn):
    """SQL and Python order turns separately; the SQL LIMIT picks candidates, so drift silently
    drops turns."""
    import types

    scope = store.conversation_archive_scope(THREAD)
    plan = [
        # Legacy rows tie on created_at, then one separates by clock, then two numbered rows tie
        (None, "2026-01-01T00:00:00+00:00", 3),
        (None, "2026-01-01T00:00:00+00:00", 2),
        (None, "2026-01-02T00:00:00+00:00", 1),
        (7, "2026-01-03T00:00:00+00:00", 2),
        (8, "2026-01-03T00:00:00+00:00", 2),
    ]
    for position, (ordinal, created, count) in enumerate(plan):
        # Descending ids against ascending conversation order: id order is exactly wrong.
        document_id = f"{len(plan) - position:04d}-turn"
        store.create_document(
            conn,
            scope = scope,
            thread_id = THREAD,
            filename = "earlier turn",
            sha256 = f"h{position}",
            status = "completed",
            embedding_model = "m",
            archive_messages = 2,
            archive_ordinal = ordinal,
            document_id = document_id,
            created_at = created,
            commit = False,
        )
        store.add_chunks(
            conn,
            scope,
            document_id,
            [
                types.SimpleNamespace(
                    chunk_index = index,
                    text = "ZQXAGREE statement " + "word " * (index + position),
                    page_number = None,
                    source_page_index = None,
                    token_count = 5,
                    char_count = 20,
                )
                for index in range(count)
            ],
            [[0.0] * 4] * count,
        )
    conn.commit()

    def _tiers(**direction):
        hits = store.search_lexical(conn, scope, "ZQXAGREE", 500, **direction)
        grouped: list = []
        for chunk_id, score in hits:
            if grouped and grouped[-1][0] == score:
                grouped[-1][1].append(chunk_id)
            else:
                grouped.append((score, [chunk_id]))
        return grouped

    oldest = _tiers(oldest_first = True)
    newest = _tiers(newest_first = True)
    every_id = [chunk_id for _score, tier in oldest for chunk_id in tier]
    assert len(every_id) == sum(count for _o, _c, count in plan)
    rows = store.chunks_by_id(conn, every_id)
    assert max(len(tier) for _score, tier in oldest) > 1, oldest

    for score, tier in oldest:
        expected = sorted(
            tier, key = lambda chunk_id: conversation_archive._conversation_order(rows[chunk_id])
        )
        assert tier == expected, (score, tier, expected)
    assert [score for score, _ in newest] == [score for score, _ in oldest]
    for (_score, forward), (_same, backward) in zip(oldest, newest):
        assert backward == list(reversed(forward)), (forward, backward)


def test_a_rewritten_turn_keeps_the_insertion_order_it_was_archived_in(conn, monkeypatch):
    """A re-embedded row must keep its archive position; a fresh rowid would move it to the end."""
    from core.rag import embeddings

    monkeypatch.setattr(store, "_now", lambda: "2026-01-01T00:00:00+00:00")
    identity = {"name": "st:model-a"}
    real = embeddings.encode_with_identity
    monkeypatch.setattr(
        embeddings,
        "encode_with_identity",
        lambda texts, **kwargs: (real(texts, **kwargs)[0], identity["name"]),
    )
    monkeypatch.setattr(embeddings, "embedding_identity", lambda *_a, **_k: identity["name"])

    turns = [_turn(f"turn {n} about pelicans", f"STATEMENT{n} about pelicans") for n in range(1, 4)]
    history = [dict(message) for turn in turns for message in turn]
    _save_thread(THREAD, history, append = True)
    assert conversation_archive.archive_turns(THREAD, [dict(m) for m in history]) == 3
    scope = store.conversation_archive_scope(THREAD)
    before = [
        (row["rowid"], row["id"])
        for row in conn.execute(
            "SELECT rowid, id FROM documents WHERE scope=? ORDER BY rowid", (scope,)
        )
    ]

    identity["name"] = "st:model-b"
    conversation_archive.archive_turns(THREAD, [dict(m) for m in history])

    after = [
        (row["rowid"], row["id"])
        for row in conn.execute(
            "SELECT rowid, id FROM documents WHERE scope=? ORDER BY rowid", (scope,)
        )
    ]
    assert [rowid for rowid, _ in after] == [rowid for rowid, _ in before]
    assert [document_id for _, document_id in after] != [document_id for _, document_id in before]


def test_merging_two_recall_queries_still_lists_legacy_turns_first(conn):
    """The merge key has to agree with `_conversation_order`, or the merged block
    contradicts its own "oldest first" header on an upgraded archive."""
    _archive(_turn("the pelican turn", "OLDLEGACY statement about pelicans"))
    _archive(_turn("more pelican talk", "NEWNUMBERED statement about pelicans"))
    scope = store.conversation_archive_scope(THREAD)
    oldest = conn.execute(
        "SELECT id FROM documents WHERE scope=? ORDER BY created_at", (scope,)
    ).fetchone()["id"]
    conn.execute("UPDATE documents SET archive_ordinal=NULL WHERE id=?", (oldest,))
    conn.commit()

    merged = conversation_archive.recall(THREAD, "pelican", top_k = 4, extra_queries = ["statement"])

    assert merged is not None
    text, _sources = merged
    assert text.index("OLDLEGACY") < text.index("NEWNUMBERED")


def test_merging_two_recall_queries_keeps_one_turns_chunks_in_order(conn, monkeypatch):
    """Both queries hit the same long turn, so every source carries the same `turn` and a
    stable sort would quote it in query order -- tail first."""
    monkeypatch.setattr(config, "CHUNK_TOKENS", 30)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 0)
    body = (
        "ALPHAHEAD the opening of the turn "
        + " ".join(f"w{index}" for index in range(25))
        + " OMEGATAIL the closing of the turn "
        + " ".join(f"z{index}" for index in range(25))
    )
    _archive(_turn("a very long turn", body))

    merged = conversation_archive.recall(THREAD, "ALPHAHEAD", top_k = 2, extra_queries = ["OMEGATAIL"])

    assert merged is not None
    text, _sources = merged
    assert text.index("ALPHAHEAD") < text.index("OMEGATAIL")


def test_merging_two_recall_queries_keeps_legacy_turns_in_the_order_they_were_said(
    tmp_path, monkeypatch
):
    """Merged recall must tie-break legacy rows by created_at, as _conversation_order does."""
    from core.rag import conversation_archive

    merged = [
        {"turn": None, "createdAt": "2026-01-02T00:00:00Z", "chunkIndex": 0, "text": "later"},
        {"turn": None, "createdAt": "2026-01-01T00:00:00Z", "chunkIndex": 0, "text": "earlier"},
        {"turn": 3, "createdAt": "2026-01-03T00:00:00Z", "chunkIndex": 0, "text": "numbered"},
    ]
    merged.sort(
        key = lambda source: conversation_archive._order_key(
            source.get("turn"),
            source.get("createdAt"),
            source.get("documentRowid"),
            source.get("chunkIndex"),
        )
    )

    assert [m["text"] for m in merged] == ["earlier", "later", "numbered"]


def test_both_recall_paths_order_by_the_same_key():
    """The single-query path reads snake_case columns and the merge reads camelCase keys;
    they are only the same key while both call `_order_key`.
    """
    from core.rag import conversation_archive
    for ordinal in (None, 0, 4):
        for created in ("", "2026-01-01T00:00:00Z"):
            for rowid in (None, 0, 12):
                for index in (None, 0, 3):
                    row = {
                        "archive_ordinal": ordinal,
                        "created_at": created,
                        "document_rowid": rowid,
                        "chunk_index": index,
                    }
                    source = {
                        "turn": ordinal,
                        "createdAt": created,
                        "documentRowid": rowid,
                        "chunkIndex": index,
                    }
                    assert conversation_archive._conversation_order(row) == (
                        conversation_archive._order_key(
                            source.get("turn"),
                            source.get("createdAt"),
                            source.get("documentRowid"),
                            source.get("chunkIndex"),
                        )
                    ), row


def test_recall_sources_carry_the_fields_the_merge_orders_by():
    """Recall sources must carry the sort keys (createdAt, documentRowid), since nothing renders them."""
    from types import SimpleNamespace

    from core.rag import tool

    rows = {
        "c1": {
            "document_id": "d1",
            "filename": "chat",
            "text": "hello",
            "archive_ordinal": None,
            "chunk_index": 2,
            "created_at": "2026-01-01T00:00:00Z",
            "document_rowid": 41,
        },
    }
    hits = [SimpleNamespace(chunk_id = "c1", score = 0.5)]

    _, sources = tool.format_conversation_recall(rows, hits)

    assert sources[0]["createdAt"] == "2026-01-01T00:00:00Z"
    assert sources[0]["chunkIndex"] == 2
    assert sources[0]["documentRowid"] == 41
    assert sources[0]["turn"] is None


def test_the_forced_floor_filters_candidates_rather_than_deleting_results(conn, monkeypatch):
    """A score floor must filter candidates before the top-k slice, or weak hits take slots and vanish."""
    from core.rag import config

    for index in range(8):
        _archive(_turn(f"pelican note {index}", f"statement about pelican {index}"))
    real = conversation_archive._candidates

    def weak_first(*args, **kwargs):
        hits = real(*args, **kwargs)
        for position, hit in enumerate(hits):
            hit.dense_score = 0.1 if position < 4 else 0.9
        return hits

    monkeypatch.setattr(conversation_archive, "_candidates", weak_first)
    monkeypatch.setattr(config, "CONVERSATION_FORCED_MIN_SCORE", 0.5)

    forced = conversation_archive.recall(THREAD, "pelican", top_k = 4, forced = True)

    assert forced is not None, "the floor deleted the result instead of filtering candidates"
    assert len(forced[1]) == 4


def test_a_floor_nothing_clears_still_returns_nothing(conn, monkeypatch):
    """The filter must not turn into "return the weak ones anyway"."""
    from core.rag import config

    for index in range(8):
        _archive(_turn(f"pelican note {index}", f"statement about pelican {index}"))
    real = conversation_archive._candidates

    def all_weak(*args, **kwargs):
        hits = real(*args, **kwargs)
        for hit in hits:
            hit.dense_score = 0.1
        return hits

    monkeypatch.setattr(conversation_archive, "_candidates", all_weak)
    monkeypatch.setattr(config, "CONVERSATION_FORCED_MIN_SCORE", 0.5)

    assert conversation_archive.recall(THREAD, "pelican", top_k = 4, forced = True) is None
    assert conversation_archive.recall(THREAD, "pelican", top_k = 4) is not None


def test_the_newest_revision_survives_a_tied_run_LONGER_than_the_cap(conn):
    """A tied run past the candidate cap keeps the oldest rows, so the newest turn is unreachable."""
    count = conversation_archive._BRANCH_FILTER_MAX_CANDIDATES + 40
    for index in range(count - 1):
        _archive(_turn(f"note {index:03d} about ZQXVARA123", "noted"))
    _archive(_turn("set ZQXVARA123 to 9999", "done"))

    found = conversation_archive.recall(THREAD, "what is ZQXVARA123 currently", top_k = 4)

    assert found is not None
    assert "9999" in found[0]
    oldest = conversation_archive.recall(THREAD, "what was ZQXVARA123 originally", top_k = 4)
    assert oldest is not None
    assert "note 000" in oldest[0]


def test_a_re_embedded_oldest_turn_is_still_reachable_past_the_cap(conn, monkeypatch):
    """Both halves of the window must order by the chunk ordinal, since a re-embed moves rowids."""
    from core.rag import embeddings

    identity = {"name": "st:model-a"}
    real = embeddings.encode_with_identity
    monkeypatch.setattr(
        embeddings,
        "encode_with_identity",
        lambda texts, **kwargs: (real(texts, **kwargs)[0], identity["name"]),
    )
    monkeypatch.setattr(embeddings, "embedding_identity", lambda *_a, **_k: identity["name"])

    oldest_turn = _turn("note 000 about ZQXVARA123", "noted")
    _archive([dict(message) for message in oldest_turn])
    count = conversation_archive._BRANCH_FILTER_MAX_CANDIDATES + 40
    for index in range(1, count - 1):
        _archive(_turn(f"note {index:03d} about ZQXVARA123", "noted"))
    _archive(_turn("set ZQXVARA123 to 9999", "done"))

    identity["name"] = "st:model-b"
    _archive([dict(message) for message in oldest_turn])

    oldest = conversation_archive.recall(THREAD, "what was ZQXVARA123 originally", top_k = 4)
    assert oldest is not None
    assert "note 000" in oldest[0]


def test_the_newest_revision_survives_a_tie_and_the_oldest_one_still_does(conn):
    """A tied score is not an order: truncation must take from both ends, or the stale oldest value wins."""
    values = _revisions(8)

    found = conversation_archive.recall(THREAD, f"what is the current value of {VARIABLE}", top_k = 4)

    assert found is not None
    text, _sources = found
    assert values[-1] in text, (
        "the newest revision was dropped by a tie-break that prefers whatever the index "
        "emitted first"
    )
    assert values[0] in text, "the oldest revision must not be dropped either"


def test_an_overlapping_anchor_query_does_not_shrink_the_recall(conn):
    """Overlapping queries must not shrink the recall: cut after dedup, not per query's share."""
    for index in range(6):
        _archive(_turn(f"pelican note {index}", f"statement about pelican {index}"))

    alone = conversation_archive.recall(THREAD, "pelican", top_k = 4)
    merged = conversation_archive.recall(THREAD, "pelican", top_k = 4, extra_queries = ["statement"])

    assert alone is not None and merged is not None
    assert len(alone[1]) == 4
    assert (
        len(merged[1]) == 4
    ), f"the anchor cost slots to its overlap with the latest query: {len(merged[1])} of 4"


def test_a_shouted_question_filters_as_well_as_a_typed_one(conn):
    """A shouted question with no lower case would make every word an identifier; the rule needs
    contrast."""
    for index in range(6):
        _archive(_turn(f"Set {VARIABLE} to 42{index}.", f"Understood. {VARIABLE} is 42{index}."))
    _archive(
        _turn(
            "What is a good default value for a retry budget?",
            "Three attempts with backoff is a common default value.",
        )
    )
    question = f"What is the current value of {VARIABLE}?"

    assert store.conversation_match_queries(question.upper()) == (
        store.conversation_match_queries(question)
    )
    found = conversation_archive.recall(THREAD, question.upper(), top_k = 1)

    assert found is not None
    assert (
        VARIABLE.lower() in found[0].lower()
    ), "the shouted question filtered nothing and spent its only slot on filler"


def test_a_numeric_subject_is_still_an_identifier_when_the_question_is_shouted(conn):
    """Shape is 'contains a digit', so numeric subjects keep filtering in a shouted question."""
    for index in range(6):
        _archive(_turn(f"Set 9134 to 42{index}.", f"Understood. 9134 is 42{index}."))
    _archive(
        _turn(
            "What is a good default value for a retry budget?",
            "Three attempts with backoff is a common default value.",
        )
    )
    question = "What is the current value of 9134?"

    assert store.conversation_match_queries(question)[0] == '"9134"'
    assert store.conversation_match_queries(question.upper())[0] == '"9134"'
    found = conversation_archive.recall(THREAD, question.upper(), top_k = 1)

    assert found is not None
    assert "9134" in found[0]


def test_turning_the_query_focus_off_restores_the_old_order_on_a_tied_archive(conn, monkeypatch):
    """The tie-break reorders candidates, so it must sit behind the rollback knob too."""
    from core.rag import config

    monkeypatch.setattr(config, "CONVERSATION_QUERY_FOCUS", False)
    monkeypatch.setattr(config, "CONVERSATION_RECALL_ORDER", "relevance")
    values = _revisions(8, distractors = 0)

    found = conversation_archive.recall(THREAD, f"what is the current value of {VARIABLE}", top_k = 4)

    assert found is not None
    returned = [value for value in values if value in found[0]]
    assert returned == values[:4], f"the knob did not restore the previous selection: {returned}"


def test_a_turn_repeated_later_is_archived_again_at_its_own_position(conn):
    """A repeated turn said later is archived again at its own position, not treated as a duplicate."""
    written = [
        _archive(_turn("set ZQXVARA123 to 1", "ok")),
        _archive(_turn("set ZQXVARA123 to 2", "ok")),
        _archive(_turn("set ZQXVARA123 to 1", "ok")),
    ]

    assert written == [1, 1, 1]
    scope = store.conversation_archive_scope(THREAD)
    assert len(store.list_documents(conn, scope)) == 3
    found = conversation_archive.recall(THREAD, "ZQXVARA123", top_k = 4)
    assert found is not None
    turns = [source.get("turn") for source in found[1]]
    assert turns == sorted(turns)
    assert "set ZQXVARA123 to 1" in found[1][-1]["text"]


def test_a_repeat_still_in_the_prompt_is_not_archived_early(conn):
    """A repeat still in the prompt is not archived early; written when its own turn is evicted."""
    repeat = _turn("set ZQXVARA123 to 1", "ok")
    middle = _turn("tell me about ZQXVARA123 pelicans", "sure")
    tail = _turn("and now something else about ZQXVARA123", "fine")
    conversation = repeat + middle + list(repeat) + tail
    _save_thread(THREAD, conversation)

    live = list(repeat) + tail
    conversation_archive.archive_turns(THREAD, repeat, live = live)
    conversation_archive.archive_turns(THREAD, repeat, live = live)

    scope = store.conversation_archive_scope(THREAD)
    assert len(store.list_documents(conn, scope)) == 1

    found = conversation_archive.recall(THREAD, "ZQXVARA123", top_k = 4)
    assert found is not None
    texts = [source["text"] for source in found[1]]
    assert len(texts) == len(set(texts))

    conversation_archive.archive_turns(THREAD, conversation[:6], live = tail)
    ordinals = [
        row["archive_ordinal"]
        for row in conn.execute(
            "SELECT archive_ordinal FROM documents WHERE scope=? ORDER BY archive_ordinal",
            (scope,),
        ).fetchall()
    ]
    assert ordinals == [0, 1, 2]


def test_a_re_embed_does_not_swallow_a_repeat_evicted_later(conn, monkeypatch):
    """A re-embed replaces one copy's vectors and must not swallow a repeat that is evicted later."""
    from core.rag import embeddings

    identity = {"name": "st:model-a"}
    real = embeddings.encode_with_identity
    monkeypatch.setattr(
        embeddings,
        "encode_with_identity",
        lambda texts, **kwargs: (real(texts, **kwargs)[0], identity["name"]),
    )
    monkeypatch.setattr(embeddings, "embedding_identity", lambda *_a, **_k: identity["name"])

    repeat = _turn("set ZQXVARA123 to 1", "ok")
    middle = _turn("set ZQXVARA123 to 2", "ok")
    tail = _turn("and now something else about ZQXVARA123", "fine")
    conversation = repeat + middle + list(repeat) + tail
    _save_thread(THREAD, conversation)

    live = list(repeat) + tail
    conversation_archive.archive_turns(THREAD, repeat + middle, live = live)

    identity["name"] = "st:model-b"
    conversation_archive.archive_turns(THREAD, list(repeat), live = tail)

    scope = store.conversation_archive_scope(THREAD)
    ordinals = [
        row["archive_ordinal"]
        for row in conn.execute(
            "SELECT archive_ordinal FROM documents WHERE scope=? ORDER BY archive_ordinal",
            (scope,),
        ).fetchall()
    ]
    assert ordinals == [0, 1, 2]

    found = conversation_archive.recall(THREAD, "ZQXVARA123", top_k = 4)
    assert found is not None
    assert "set ZQXVARA123 to 1" in found[1][-1]["text"]


def test_two_evicted_copies_of_a_thrice_said_turn_are_both_archived(conn):
    """A set of live texts cannot count. Three identical turns with two evicted and one
    still in the prompt looked entirely live, so only one copy was ever written and the
    archive was a turn short of what was actually said."""
    repeat = _turn("set ZQXVARA123 to 1", "ok")
    mid = _turn("set ZQXVARA123 to 2", "ok")
    other = _turn("something else about ZQXVARA123", "fine")
    conversation = repeat + mid + list(repeat) + other + list(repeat)
    _save_thread(THREAD, conversation)

    live = other + list(repeat)
    conversation_archive.archive_turns(THREAD, repeat + mid + list(repeat), live = live)

    scope = store.conversation_archive_scope(THREAD)
    ordinals = [
        row["archive_ordinal"]
        for row in conn.execute(
            "SELECT archive_ordinal FROM documents WHERE scope=? ORDER BY archive_ordinal",
            (scope,),
        ).fetchall()
    ]
    assert ordinals == [0, 1, 2]


def test_a_rewound_repeat_moves_the_survivor_to_the_seat_it_still_has(conn, monkeypatch):
    """A rewound repeat must restamp the survivor onto a seat it still has, or the header misleads."""
    from core.rag import embeddings

    identity = {"name": "st:model-a"}
    real = embeddings.encode_with_identity
    monkeypatch.setattr(
        embeddings,
        "encode_with_identity",
        lambda texts, **kwargs: (real(texts, **kwargs)[0], identity["name"]),
    )
    monkeypatch.setattr(embeddings, "embedding_identity", lambda *_a, **_k: identity["name"])

    repeat = _turn("set ZQXVARA123 to 1", "ok")
    mid = _turn("set ZQXVARA123 to 2", "ok")
    whole = repeat + mid + list(repeat)
    _save_thread(THREAD, whole)
    conversation_archive.archive_turns(THREAD, whole, live = [])

    _save_thread(THREAD, mid + list(repeat))
    identity["name"] = "st:model-b"
    conversation_archive.archive_turns(THREAD, mid + list(repeat), live = [])

    found = conversation_archive.recall(THREAD, "ZQXVARA123", top_k = 4)
    assert found is not None
    assert found[0].index("set ZQXVARA123 to 2") < found[0].rindex("set ZQXVARA123 to 1")


def test_the_write_budget_is_every_seat_when_the_caller_says_nothing(conn):
    """`live` is optional, and without it the budget is the old count. Direct callers
    (every other test here, and any caller outside the fit) must not change behaviour."""
    from core.rag import conversation_archive as archive

    group = [{"role": "user", "content": "a"}]
    one_live = [{"role": "user", "content": "a"}]

    assert archive._write_budget([["a"], ["a"]], [0, 1], None, group) == 2
    assert (
        archive._write_budget([["a"], ["a"]], [0, 1], archive._live_positions(one_live), group) == 1
    )
    assert archive._write_budget([["a"], ["a"]], [], archive._live_positions(one_live), group) == 1
    assert (
        archive._write_budget(
            [["a"], ["a"], ["a"]], [0, 1, 2], archive._live_positions(one_live), group
        )
        == 2
    )


def test_an_out_of_order_eviction_still_numbers_turns_in_conversation_order(conn):
    """Archive order is not conversation order: a pinned or newest-group turn is evicted after later
    ones."""
    conversation = (
        _turn("the standing instruction about pelicans", "Understood.")
        + _turn("the middle turn about pelicans", "Noted.")
        + _turn("the final turn about pelicans", "Noted again.")
    )
    _save_thread(THREAD, conversation)

    conversation_archive.archive_turns(THREAD, conversation[2:6])
    conversation_archive.archive_turns(THREAD, conversation[0:2])

    scope = store.conversation_archive_scope(THREAD)
    ordinals = [
        row["archive_ordinal"]
        for row in conn.execute(
            "SELECT archive_ordinal FROM documents WHERE scope=? ORDER BY archive_ordinal",
            (scope,),
        ).fetchall()
    ]
    assert ordinals == [0, 1, 2]
    found = conversation_archive.recall(THREAD, "pelicans", top_k = 4)
    assert found is not None
    text = found[0]
    assert text.index("standing instruction") < text.index("middle turn") < text.index("final turn")


def test_two_turns_that_start_the_same_do_not_take_each_others_places(conn):
    """Turns sharing a first message must match on the whole turn, or each takes the other's seat."""
    first = _turn("continue ZQXVARA123", "the first continuation, about ducks")
    second = _turn("continue ZQXVARA123", "the second continuation, about geese")

    written = [_archive(first), _archive(second)]
    scope = store.conversation_archive_scope(THREAD)
    ordinals = sorted(
        row["archive_ordinal"]
        for row in conn.execute(
            "SELECT archive_ordinal FROM documents WHERE scope=?", (scope,)
        ).fetchall()
    )
    again = [_archive(first, persist = False), _archive(second, persist = False)]

    assert written == [1, 1]
    assert ordinals == [0, 1]
    assert again == [0, 0]
    assert len(store.list_documents(conn, scope)) == 2


def test_an_archive_numbered_by_the_old_allocator_converges_on_the_next_compaction(conn):
    """The re-stamp migration must run on the cheap pre-check path too, not only the locked branch."""
    conversation = _turn("alpha about pelicans", "first") + _turn("beta about pelicans", "second")
    _save_thread(THREAD, conversation)
    conversation_archive.archive_turns(THREAD, conversation)
    scope = store.conversation_archive_scope(THREAD)

    def ordinals():
        return [
            row["archive_ordinal"]
            for row in conn.execute(
                "SELECT archive_ordinal FROM documents WHERE scope=? ORDER BY created_at",
                (scope,),
            ).fetchall()
        ]

    assert ordinals() == [0, 1]

    conn.execute("UPDATE documents SET archive_ordinal=NULL WHERE scope=?", (scope,))
    conn.commit()
    conversation_archive.archive_turns(THREAD, conversation)
    assert ordinals() == [0, 1]

    conn.execute(
        "UPDATE documents SET archive_ordinal=(CASE WHEN archive_ordinal=0 THEN 1 ELSE 0 END) "
        "WHERE scope=?",
        (scope,),
    )
    conn.commit()
    conversation_archive.archive_turns(THREAD, conversation)
    assert ordinals() == [0, 1]
    text, _sources = conversation_archive.recall(THREAD, "pelicans", top_k = 4)
    assert text.index("alpha") < text.index("beta")


def _persist_agent_thread():
    """A thread whose newest turn is a tool exchange, stored the way the UI stores one."""
    from storage import studio_db

    studio_db.upsert_chat_thread(
        {"id": THREAD, "title": "t", "modelType": "base", "modelId": "local-model", "createdAt": 1}
    )
    rows = [
        ("user", [{"type": "text", "text": "what is the capital of peru"}]),
        ("assistant", [{"type": "text", "text": "Lima."}]),
        ("user", [{"type": "text", "text": "list the files in the repo"}]),
        (
            "assistant",
            [
                _tool_part(type = "tool-call"),
                {"type": "text", "text": "the repo has two files."},
            ],
        ),
    ]
    for index, (role, content) in enumerate(rows):
        studio_db.upsert_chat_message(
            {
                "id": f"{THREAD}-{index}",
                "threadId": THREAD,
                "role": role,
                "content": content,
                "createdAt": index + 2,
            }
        )
    return [
        _assistant_call("terminal", '{"command": "ls"}'),
        {"role": "tool", "tool_call_id": "c1", "content": "main.py readme.md"},
        {"role": "assistant", "content": "the repo has two files."},
    ]


def test_an_archived_tool_exchange_is_still_reachable_by_a_query(conn):
    """A stored tool call's row order differs from the wire order, so probes must follow the wire."""
    tool_turn = _persist_agent_thread()
    conversation_archive.archive_turns(THREAD, tool_turn)

    found = conversation_archive.recall(THREAD, "terminal ls repo files", top_k = 4)

    assert found is not None
    assert any("terminal" in source["text"] for source in found[1])


def test_a_tool_exchange_is_numbered_where_the_conversation_put_it(conn):
    """Stored rows never carry tool_calls, so group_turns must still give a tool exchange its own group."""
    tool_turn = _persist_agent_thread()
    conversation_archive.archive_turns(
        THREAD,
        [
            {"role": "user", "content": "what is the capital of peru"},
            {"role": "assistant", "content": "Lima."},
        ],
    )
    conversation_archive.archive_turns(THREAD, tool_turn)
    conversation_archive.archive_turns(
        THREAD,
        [
            {"role": "user", "content": "list the files in the repo"},
        ],
    )

    scope = store.conversation_archive_scope(THREAD)
    ordinals = [
        row["archive_ordinal"]
        for row in conn.execute(
            "SELECT archive_ordinal FROM documents WHERE scope=? ORDER BY archive_ordinal",
            (scope,),
        ).fetchall()
    ]

    assert ordinals == [0, 1, 2]
    text, _sources = conversation_archive.recall(THREAD, "repo files peru ls", top_k = 4)
    assert text.index("capital of peru") < text.index("list the files")
    assert text.index("list the files") < text.index("called terminal")


def test_an_anchor_query_cannot_cost_the_newest_revision_its_slot(conn):
    """The refill must keep retrieval rank, not chronological order, so the newest revision survives."""
    values = _revisions(8, distractors = 0)

    alone = conversation_archive.recall(THREAD, f"{VARIABLE}", top_k = 4)
    merged = conversation_archive.recall(THREAD, f"{VARIABLE}", top_k = 4, extra_queries = ["timeout"])

    assert alone is not None and merged is not None
    assert values[-1] in alone[0]
    assert values[-1] in merged[0], "the anchor cost the newest revision its slot"


def test_an_orphan_user_row_does_not_lend_its_seat_to_a_later_turn(conn):
    """A turn missing its reply must not prefix-match anywhere, since zip stops at the shorter side."""
    from storage import studio_db

    studio_db.upsert_chat_thread(
        {"id": THREAD, "title": "t", "modelType": "base", "modelId": "local-model", "createdAt": 1}
    )
    rows = [
        ("user", "set ZQXVARA123 to 1"),
        ("user", "set ZQXVARA123 to 1"),
        ("assistant", "done, ZQXVARA123 is 1"),
    ]
    for index, (role, text) in enumerate(rows):
        studio_db.upsert_chat_message(
            {
                "id": f"{THREAD}-{index}",
                "threadId": THREAD,
                "role": role,
                "content": [{"type": "text", "text": text}],
                "createdAt": index + 2,
            }
        )
    answered = [
        {"role": "user", "content": "set ZQXVARA123 to 1"},
        {"role": "assistant", "content": "done, ZQXVARA123 is 1"},
    ]

    positions = conversation_archive._transcript_positions(THREAD)
    assert conversation_archive._occurrences(positions, answered) == [1]

    written = [
        conversation_archive.archive_turns(THREAD, answered),
        conversation_archive.archive_turns(THREAD, answered),
    ]
    scope = store.conversation_archive_scope(THREAD)

    assert written == [1, 0]
    assert len(store.list_documents(conn, scope)) == 1


def test_a_retried_turn_is_numbered_on_the_branch_the_user_is_on(conn):
    """Stored rows form a tree; flattening them glues a retry's abandoned reply onto the wrong turn."""
    from storage import studio_db

    studio_db.upsert_chat_thread(
        {"id": THREAD, "title": "t", "modelType": "base", "modelId": "local-model", "createdAt": 1}
    )
    rows = [
        ("m0", None, "user", "turn 1 about ZQXVARA123"),
        ("m1", "m0", "assistant", "answer 1"),
        ("m2", "m1", "user", "turn 2 about ZQXVARA123"),
        ("m3", "m2", "assistant", "answer 2 attempt one"),
        ("m4", "m2", "assistant", "answer 2 attempt two"),
        ("m5", "m4", "user", "turn 3 about ZQXVARA123"),
        ("m6", "m5", "assistant", "answer 3"),
    ]
    for index, (identifier, parent, role, text) in enumerate(rows):
        studio_db.upsert_chat_message(
            {
                "id": identifier,
                "threadId": THREAD,
                "parentId": parent,
                "role": role,
                "content": [{"type": "text", "text": text}],
                "createdAt": index + 2,
            }
        )

    live = [
        {"role": "user", "content": "turn 1 about ZQXVARA123"},
        {"role": "assistant", "content": "answer 1"},
        {"role": "user", "content": "turn 2 about ZQXVARA123"},
        {"role": "assistant", "content": "answer 2 attempt two"},
        {"role": "user", "content": "turn 3 about ZQXVARA123"},
        {"role": "assistant", "content": "answer 3"},
    ]
    positions = conversation_archive._transcript_positions(THREAD)

    assert len(positions) == 3, positions
    assert conversation_archive._occurrences(positions, live[2:4]) == [1]

    conversation_archive.archive_turns(THREAD, live)
    scope = store.conversation_archive_scope(THREAD)
    ordinals = sorted(
        row["archive_ordinal"]
        for row in conn.execute(
            "SELECT archive_ordinal FROM documents WHERE scope=?", (scope,)
        ).fetchall()
    )
    assert ordinals == [0, 1, 2]


def test_a_rewind_retires_the_copy_the_conversation_no_longer_holds(conn):
    """Identical copies outnumber occurrences after a rewind, so surplus copies must be retired."""
    first = _turn("set ZQXVARA123 to 1", "ok")
    second = _turn("set ZQXVARA123 to 2", "ok")
    repeat = _turn("set ZQXVARA123 to 1", "ok")
    for group in (first, second, repeat):
        _archive(group)
    scope = store.conversation_archive_scope(THREAD)
    assert len(store.list_documents(conn, scope)) == 3

    _save_thread(THREAD, first + second)
    conversation_archive.archive_turns(THREAD, first)

    ordinals = sorted(
        row["archive_ordinal"]
        for row in conn.execute(
            "SELECT archive_ordinal FROM documents WHERE scope=?", (scope,)
        ).fetchall()
    )
    assert ordinals == [0, 1]
    found = conversation_archive.recall(THREAD, "ZQXVARA123", top_k = 4)
    assert found is not None
    assert len(found[1]) == len({source["text"] for source in found[1]})


def test_an_incidental_number_does_not_take_over_the_filter(conn):
    """A bare number must not become an identifier: 'answer in 2 sentences' must not filter on '2'."""
    assert store.conversation_match_queries("answer in 2 sentences") == [
        '"answer" OR "2" OR "sentences"'
    ]
    assert store.conversation_match_queries("which python, 3.11 or 3.12") == [
        '"python" OR "3" OR "11" OR "12"'
    ]
    assert store.conversation_match_queries("what about v2 of the plan")[0] == '"v2"'
    assert store.conversation_match_queries("we talked about 2024 revenue")[0] == '"2024"'
    assert store.conversation_match_queries("What is the current value of 9134?")[0] == '"9134"'


def test_a_re_embed_after_a_rewind_retires_the_surplus_copy_too(conn, monkeypatch):
    """A re-embed after a rewind replaces one copy and returns, so the surplus copy is never retired."""
    from core.rag import embeddings

    identity = {"name": "st:model-a"}
    real = embeddings.encode_with_identity
    monkeypatch.setattr(
        embeddings,
        "encode_with_identity",
        lambda texts, **kwargs: (real(texts, **kwargs)[0], identity["name"]),
    )
    monkeypatch.setattr(embeddings, "embedding_identity", lambda *_a, **_k: identity["name"])

    first = _turn("set ZQXVARA123 to 1", "ok")
    second = _turn("set ZQXVARA123 to 2", "ok")
    repeat = _turn("set ZQXVARA123 to 1", "ok")
    for group in (first, second, repeat):
        _archive(group)
    scope = store.conversation_archive_scope(THREAD)
    assert len(store.list_documents(conn, scope)) == 3

    _save_thread(THREAD, first + second)
    identity["name"] = "st:model-b"
    conversation_archive.archive_turns(THREAD, first)

    ordinals = sorted(
        row["archive_ordinal"]
        for row in conn.execute(
            "SELECT archive_ordinal FROM documents WHERE scope=?", (scope,)
        ).fetchall()
    )
    assert ordinals == [0, 1]
    found = conversation_archive.recall(THREAD, "ZQXVARA123", top_k = 4)
    assert found is not None
    assert len(found[1]) == len({source["text"] for source in found[1]})


def test_text_said_before_a_tool_call_rides_on_the_call_message():
    """Text before the first tool call rides on the call message; text after the calls stays last."""
    call = _tool_part(type = "tool-call")
    before = conversation_archive._as_wire(
        [{"role": "assistant", "content": [{"type": "text", "text": "Let me check."}, call]}]
    )
    after = conversation_archive._as_wire(
        [{"role": "assistant", "content": [call, {"type": "text", "text": "Two files."}]}]
    )

    assert [message["role"] for message in before] == ["assistant", "tool"]
    assert "Let me check." in conversation_archive._normalise(
        conversation_archive._probe_text(before[0])
    )
    assert [message["role"] for message in after] == ["assistant", "tool", "assistant"]
    assert (
        conversation_archive._normalise(conversation_archive._probe_text(after[2])) == "Two files."
    )


def test_turns_differing_only_in_case_do_not_share_a_seat(conn):
    """Matching is case-sensitive, so 'Set key Foo' and 'Set key FOO' each keep their own seat."""
    lower = _turn("set key Foo", "done")
    upper = _turn("set key FOO", "done")
    _save_thread(THREAD, lower + upper)

    positions = conversation_archive._transcript_positions(THREAD)
    assert conversation_archive._occurrences(positions, lower) == [0]
    assert conversation_archive._occurrences(positions, upper) == [1]


def test_the_deleted_conversation_goes_even_when_its_id_comes_back(conn):
    """The scope is keyed by thread id, so a delete must cut turns at the moment it is accepted."""
    from datetime import datetime, timezone

    from routes import chat_history

    thread_id = "recreated-with-cutoff"
    old_turns = _turn("what is the code", "the code is 5150")
    _save_thread(thread_id, old_turns, append = True)
    assert conversation_archive.archive_turns(thread_id, old_turns) == 1

    cutoff = datetime.now(timezone.utc).isoformat()

    fresh = _turn("what is the new code", "the new code is 8080")
    _save_thread(thread_id, old_turns + fresh, append = True)
    assert conversation_archive.archive_turns(thread_id, fresh) == 1

    chat_history._remove_thread_rag_data([thread_id], cutoff = cutoff)

    scope = store.conversation_archive_scope(thread_id)
    remaining = " ".join(
        row["text"]
        for row in conn.execute(
            "SELECT c.text FROM chunks c JOIN documents d ON d.id=c.document_id WHERE d.scope=?",
            (scope,),
        ).fetchall()
    )
    assert "5150" not in remaining
    assert "8080" in remaining


def _branch_switch_thread():
    """Branch A, a later sibling branch B, and the user back on A. Returns A's messages."""
    from storage import studio_db

    studio_db.upsert_chat_thread(
        {"id": THREAD, "title": "t", "modelType": "base", "modelId": "local-model", "createdAt": 1}
    )
    rows = [
        ("m0", None, "user", "turn 1 about ZQXVARA123", 2),
        ("m1", "m0", "assistant", "answer 1", 3),
        ("m2", "m1", "user", "turn 2 on A about ZQXVARA123", 4),
        ("m3", "m2", "assistant", "answer 2 on A", 5),
        ("m4", "m3", "user", "turn 3 on A about ZQXVARA123", 6),
        ("m5", "m4", "assistant", "answer 3 on A", 7),
        ("m6", "m1", "user", "turn 2 on B about ZQXVARA123", 8),
        ("m7", "m6", "assistant", "answer 2 on B", 9),
    ]
    for identifier, parent, role, text, created in rows:
        studio_db.upsert_chat_message(
            {
                "id": identifier,
                "threadId": THREAD,
                "parentId": parent,
                "role": role,
                "content": [{"type": "text", "text": text}],
                "createdAt": created,
            }
        )
    return [
        {"role": "user", "content": "turn 1 about ZQXVARA123"},
        {"role": "assistant", "content": "answer 1"},
        {"role": "user", "content": "turn 2 on A about ZQXVARA123"},
        {"role": "assistant", "content": "answer 2 on A"},
        {"role": "user", "content": "turn 3 on A about ZQXVARA123"},
        {"role": "assistant", "content": "answer 3 on A"},
    ]


def test_positions_follow_the_request_branch_not_the_newest_stored_row(conn):
    """Seed the walk from the request's branch, not the newest stored row, which may be abandoned."""
    live = _branch_switch_thread()

    positions = conversation_archive._transcript_positions(THREAD, branch = live)

    assert len(positions) == 3, positions
    assert conversation_archive._occurrences(positions, live[2:4]) == [1]
    assert conversation_archive._occurrences(positions, live[4:6]) == [2]


def test_the_branch_seed_falls_back_when_nothing_matches(conn):
    """No branch, or one matching nothing, must fall back to the newest row, not empty every seat."""
    _branch_switch_thread()

    seeded = conversation_archive._transcript_positions(THREAD)
    unmatched = conversation_archive._transcript_positions(
        THREAD, branch = [{"role": "user", "content": "nothing in this thread says this"}]
    )

    assert len(seeded) == 2, seeded
    assert unmatched == seeded


def test_two_sequential_tool_rounds_replay_as_two_exchanges():
    """A row can hold several tool rounds; replay each as its own exchange, not one merged group."""

    def _call(index, command, result):
        return _tool_part(toolCallId = f"c{index}", command = command, result = result)

    wire = conversation_archive._as_wire(
        [
            {
                "role": "assistant",
                "content": [
                    _call(1, "ls", "a.py"),
                    {"type": "text", "text": "Now the tests."},
                    _call(2, "pytest", "2 passed"),
                    {"type": "text", "text": "All green."},
                ],
            }
        ]
    )

    assert [message["role"] for message in wire] == [
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert [message.get("tool_call_id") for message in wire if message["role"] == "tool"] == [
        "c1",
        "c2",
    ]
    second = conversation_archive._normalise(conversation_archive._probe_text(wire[2]))
    assert "pytest" in second and "Now the tests." in second
    assert (
        conversation_archive._normalise(conversation_archive._probe_text(wire[4])) == "All green."
    )


def test_an_in_flight_tool_group_does_not_take_the_live_user_turn_s_number(conn):
    """Seats count transcript positions, so an in-flight tool group cannot take the user turn's number."""
    user_turn = _turn("run the deploy", "deploying now")
    _save_thread(THREAD, user_turn, append = True)

    in_flight = [
        _assistant_call("terminal", '{"command": "deploy"}'),
        {"role": "tool", "tool_call_id": "c1", "content": "deploy failed: port in use"},
    ]
    conversation_archive.archive_turns(THREAD, in_flight)
    conversation_archive.archive_turns(THREAD, user_turn)

    scope = store.conversation_archive_scope(THREAD)
    numbered = {
        row["filename"]: row["archive_ordinal"]
        for row in conn.execute(
            "SELECT filename, archive_ordinal FROM documents WHERE scope=?", (scope,)
        ).fetchall()
    }

    assert len(set(numbered.values())) == len(numbered), numbered
    assert numbered["earlier turn (user + assistant)"] < numbered["earlier turn (assistant + tool)"]


def test_an_answer_corrected_only_in_case_retires_the_archived_copy(conn):
    """Comparison must be case-sensitive, so a case-only correction still retires the archived copy."""
    rows = [{"text": "user: set the key\nassistant: Foo"}]
    corrected = conversation_archive.branch_message_texts(
        [{"role": "user", "content": "set the key"}, {"role": "assistant", "content": "foo"}]
    )
    intact = conversation_archive.branch_message_texts(
        [{"role": "user", "content": "set the key"}, {"role": "assistant", "content": "Foo"}]
    )

    assert conversation_archive._document_matches_one_run(rows, corrected, 2) is False
    assert conversation_archive._document_matches_one_run(rows, intact, 2) is True


def test_a_turn_that_opens_on_whitespace_is_still_on_its_branch(conn):
    """A turn opening with whitespace stays on its branch; probes must be stripped, not rstripped."""
    for content in ("   hello there", "\n  def f():\n    pass"):
        turn = [{"role": "user", "content": content}, {"role": "assistant", "content": "ok"}]
        rendered = conversation_archive.render_turn(turn)
        text = rendered[1] if isinstance(rendered, tuple) else rendered
        live = conversation_archive.branch_message_texts(turn)

        assert conversation_archive._document_matches_one_run([{"text": text}], live, 2) is True


def test_a_tool_result_cut_exactly_on_a_line_stays_on_its_branch(conn):
    """Strip must not drop a truncation marker on its own line, or the over-cap tool result is retired."""
    for length, where in ((7, "on a line boundary"), (8, "mid line")):
        body = "\n".join("y" * length for _ in range(900))
        turn = [{"role": "user", "content": "run it"}, {"role": "tool", "content": body}]
        rendered = conversation_archive.render_turn(turn)
        text = rendered[1] if isinstance(rendered, tuple) else rendered
        live = conversation_archive.branch_message_texts(turn)

        assert (
            conversation_archive._document_matches_one_run([{"text": text}], live, 2) is True
        ), f"cut {where}"


def test_an_empty_tool_result_still_produces_a_tool_message():
    """Only undefined and null results are absent; an empty string or {} still produces a tool message."""

    def _row(result, *, present = True):
        call = {
            "type": "tool-call",
            "toolCallId": "c1",
            "toolName": "terminal",
            "args": {"command": "true"},
        }
        if present:
            call["result"] = result
        return [{"role": "assistant", "content": [call]}]

    emitted = {
        result: [
            message["content"]
            for message in conversation_archive._as_wire(_row(result))
            if message["role"] == "tool"
        ]
        for result in ("", "ok")
    }
    # Byte for byte JSON.stringify output (no space after the colon, unlike json.dumps).
    assert emitted[""] == ['{"result":""}']
    assert emitted["ok"] == ["ok"]

    for container, expected in (({}, "{}"), ([], "[]"), ({"a": 1}, '{"a":1}')):
        wire = conversation_archive._as_wire(_row(container))
        assert [message["content"] for message in wire if message["role"] == "tool"] == [expected]

    for row in (_row(None), _row(None, present = False)):
        assert [
            message for message in conversation_archive._as_wire(row) if message["role"] == "tool"
        ] == []

    wire = [
        _assistant_call("terminal", '{"command":"true"}'),
        {"role": "tool", "tool_call_id": "c1", "content": '{"result":""}'},
    ]
    rendered = conversation_archive.render_turn(wire)
    document = rendered[1] if isinstance(rendered, tuple) else rendered
    reconstructed = conversation_archive.branch_message_texts(
        conversation_archive._as_wire(_row(""))
    )

    assert conversation_archive._on_live_branch(document, reconstructed) is True


def test_a_bare_identifier_query_also_reaches_past_the_cap(conn):
    """A query that is only an identifier shapes to one expression, so it must still reach past the cap."""
    count = conversation_archive._BRANCH_FILTER_MAX_CANDIDATES + 40
    for index in range(count - 1):
        _archive(_turn(f"note {index:03d} about ZQXVARA123", "noted"))
    _archive(_turn("set ZQXVARA123 to 9999", "done"))

    assert len(store.conversation_match_queries("ZQXVARA123")) == 1
    found = conversation_archive.recall(THREAD, "ZQXVARA123", top_k = 4)

    assert found is not None
    assert "9999" in found[0]
    assert "note 000" in conversation_archive.recall(THREAD, "ZQXVARA123", top_k = 256)[0]


def test_a_persisted_tool_call_followed_by_its_answer_stays_on_its_branch(conn):
    """A tool call, its result, then the answer: bucketing the answer in the middle broke the match."""
    stored = [
        {
            "role": "assistant",
            "content": [
                _tool_part(command = "cat deploy.yml", result = "token ZQX-5150"),
                {"type": "text", "text": "The deploy token is ZQX-5150."},
            ],
        }
    ]
    wire = [
        _assistant_call("terminal", '{"command": "cat deploy.yml"}'),
        {"role": "tool", "tool_call_id": "c1", "content": "token ZQX-5150"},
        {"role": "assistant", "content": "The deploy token is ZQX-5150."},
    ]
    rendered = conversation_archive.render_turn(wire)
    text = rendered[1] if isinstance(rendered, tuple) else rendered

    assert (
        conversation_archive._document_matches_one_run(
            [{"text": text}], conversation_archive.branch_message_texts(stored), 3
        )
        is True
    )
    assert (
        conversation_archive._document_matches_one_run(
            [{"text": text}], conversation_archive.branch_message_texts(wire), 3
        )
        is True
    )


def test_a_provider_side_builtin_is_replayed_the_way_the_frontend_replays_it():
    """A builtin card is omitted without a native part, else replayed as a call with no tool message."""
    marked = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool-call",
                    "toolCallId": "s1",
                    "toolName": "web_search",
                    "args": {"query": "ZQX rate", "_server_tool": True},
                    "result": "search hits",
                },
                {"type": "text", "text": "The ZQX rate is 5150."},
            ],
        }
    ]
    native = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool-call",
                    "toolCallId": "s2",
                    "toolName": "code_execution",
                    "args": {"google": {"native_part": {"code": "print(1)"}}},
                    "result": "1",
                },
                {"type": "text", "text": "Done."},
            ],
        }
    ]
    homonym = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool-call",
                    "toolCallId": "u1",
                    "toolName": "web_search",
                    "args": {"query": "ZQX rate"},
                    "result": "hits",
                }
            ],
        }
    ]

    assert [m["role"] for m in conversation_archive._as_wire(marked)] == ["assistant"]
    assert [m["role"] for m in conversation_archive._as_wire(native)] == [
        "assistant",
        "assistant",
    ]
    assert [m["role"] for m in conversation_archive._as_wire(homonym)] == ["assistant", "tool"]


def test_a_sandbox_result_is_replayed_as_the_text_the_model_saw():
    """Python and terminal results are replayed as their result.text alone, not the whole wrapper."""

    def _row(tool_name, result):
        return [
            {
                "role": "assistant",
                "content": [_tool_part(toolName = tool_name, result = result)],
            }
        ]

    def _tool_content(rows):
        return [m["content"] for m in conversation_archive._as_wire(rows) if m["role"] == "tool"]

    sandbox = {
        "text": "token ZQX-5150",
        "images": [],
        "sessionId": "project-7",
        "files": [{"name": "out.csv", "size": 12}],
    }
    mcp_image = {
        "text": "chart rendered",
        "images": [{"data": "AAAA", "mimeType": "image/png"}],
    }

    assert _tool_content(_row("terminal", sandbox)) == ["token ZQX-5150"]
    # edit_file is wrapped the same way when it created a file.
    assert _tool_content(_row("edit_file", sandbox)) == ["token ZQX-5150"]
    assert _tool_content(_row("python", mcp_image)) == ["chart rendered"]
    assert _tool_content(_row("terminal", {**sandbox, "text": ""})) == ['{"result":""}']
    assert _tool_content(_row("lookup", sandbox)) == [
        '{"text":"token ZQX-5150","images":[],"sessionId":"project-7",'
        '"files":[{"name":"out.csv","size":12}]}'
    ]


_IMAGE_ID = "aabbccddeeff"
_IMAGE_ENTRY = {
    "id": _IMAGE_ID,
    "title": "Ragdoll ZQXVARA123",
    "domain": "example.com",
    "source": "https://example.com/ragdoll.jpg",
    "subject": "ragdoll",
}
_IMAGE_ENTRY_NO_SUBJECT = {key: value for key, value in _IMAGE_ENTRY.items() if key != "subject"}
_SEARCH_TEXT = (
    "The ZQXVARA123 ragdoll weighs 6 kg.\n\n---\n\n"
    "ragdoll:\n- [[img:%s]] Ragdoll ZQXVARA123 \u2014 example.com" % _IMAGE_ID
)
_SEARCH_TEXT_REPLAYED = (
    "The ZQXVARA123 ragdoll weighs 6 kg.\n\n---\n\n"
    "ragdoll:\n-  Ragdoll ZQXVARA123 \u2014 example.com"
)


_ANSWER = "A ZQXVARA123 ragdoll weighs 6 kg."


def _image_search_row(
    result,
    tool_name = "web_search",
    answer = _ANSWER,
):
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool-call",
                    "toolCallId": "w1",
                    "toolName": tool_name,
                    "args": {"query": "ragdoll ZQXVARA123"},
                    "result": result,
                },
                {"type": "text", "text": answer},
            ],
        }
    ]


def _wire_tool_content(rows):
    return [m["content"] for m in conversation_archive._as_wire(rows) if m["role"] == "tool"]


def test_a_web_search_result_is_replayed_without_its_image_tokens():
    """A {text, webImages} search result is replayed as its text, without image tokens."""
    result = {"text": _SEARCH_TEXT, "webImages": [_IMAGE_ENTRY]}

    assert _wire_tool_content(_image_search_row(result)) == [_SEARCH_TEXT_REPLAYED]
    assert _wire_tool_content(
        _image_search_row({"text": _SEARCH_TEXT, "webImages": [_IMAGE_ENTRY_NO_SUBJECT]})
    ) == [_SEARCH_TEXT_REPLAYED]
    assert _wire_tool_content(
        _image_search_row({"text": "[[img:%s]]" % _IMAGE_ID, "webImages": [_IMAGE_ENTRY]})
    ) == ['{"result":""}']
    assert _wire_tool_content(_image_search_row(result, "lookup")) == [_SEARCH_TEXT_REPLAYED]
    assert _wire_tool_content(
        _image_search_row(
            {
                "text": "The ragdoll:\n\n[[img:%s]]\n\nIt weighs 6 kg." % _IMAGE_ID,
                "webImages": [_IMAGE_ENTRY],
            }
        )
    ) == ["The ragdoll:\n\nIt weighs 6 kg."]
    assert _wire_tool_content(
        _image_search_row(
            {
                "text": _SEARCH_TEXT,
                "webImages": [{**_IMAGE_ENTRY, "source": "HTTPS://example.com/cat.jpg"}],
            }
        )
    ) == [_SEARCH_TEXT_REPLAYED]
    assert _wire_tool_content(
        _image_search_row(
            {
                "text": "see [[img:%s]] and [[img:not-an-id]]" % _IMAGE_ID,
                "webImages": [_IMAGE_ENTRY],
            }
        )
    ) == ["see  and [[img:not-an-id]]"]


def test_an_envelope_that_is_not_the_search_shape_is_still_serialised_whole():
    """Unwrapping on text alone would drop other fields; any entry failing its check goes out as JSON."""
    rejected = [
        [],
        "not a list",
        [{**_IMAGE_ENTRY, "source": "javascript:alert(1)"}],
        # `httpſ` is `https` to a Unicode case fold, not to JavaScript's `/i`.
        [{**_IMAGE_ENTRY, "source": "httpſ://example.com/cat.jpg"}],
        [{**_IMAGE_ENTRY, "id": "nope"}],
        [{**_IMAGE_ENTRY, "title": None}],
        [{**_IMAGE_ENTRY, "domain": 7}],
        [{**_IMAGE_ENTRY, "subject": None}],
        [_IMAGE_ENTRY, {**_IMAGE_ENTRY, "id": "nope"}],
    ]
    for entries in rejected:
        result = {"text": _SEARCH_TEXT, "webImages": entries}
        assert _wire_tool_content(_image_search_row(result)) == [
            json.dumps(result, ensure_ascii = False, separators = (",", ":"))
        ], entries


def test_a_result_that_is_two_wrappers_at_once_is_still_stripped():
    """Being a wrapper and losing the tokens are two questions in the serializer."""
    both = {
        "text": _SEARCH_TEXT,
        "images": [{"data": "AAAA", "mimeType": "image/png"}],
        "webImages": [_IMAGE_ENTRY],
    }
    assert _wire_tool_content(_image_search_row(both)) == [_SEARCH_TEXT_REPLAYED]
    sandboxed = {
        "text": _SEARCH_TEXT,
        "images": [],
        "sessionId": "project-7",
        "webImages": [_IMAGE_ENTRY],
    }
    assert _wire_tool_content(_image_search_row(sandboxed, "terminal")) == [_SEARCH_TEXT_REPLAYED]
    assert _wire_tool_content(
        _image_search_row({"text": _SEARCH_TEXT, "images": both["images"]})
    ) == [_SEARCH_TEXT]
    nulled = {"text": _SEARCH_TEXT, "images": both["images"], "sessionId": None}
    assert _wire_tool_content(_image_search_row(nulled)) == [
        json.dumps(nulled, ensure_ascii = False, separators = (",", ":"))
    ]


def _persist_image_search_turn(answer = _ANSWER):
    """The stored rows for one web_search turn, in the shape assistant-ui saves."""
    from storage import studio_db

    studio_db.upsert_chat_thread(
        {"id": THREAD, "title": "t", "modelType": "base", "modelId": "local-model", "createdAt": 1}
    )
    row = _image_search_row({"text": _SEARCH_TEXT, "webImages": [_IMAGE_ENTRY]}, answer = answer)
    rows = [
        ("u0", None, "user", [{"type": "text", "text": "how heavy is a ZQXVARA123 ragdoll"}]),
        ("a0", "u0", "assistant", row[0]["content"]),
    ]
    for index, (identifier, parent, role, content) in enumerate(rows):
        studio_db.upsert_chat_message(
            {
                "id": identifier,
                "threadId": THREAD,
                "parentId": parent,
                "role": role,
                "content": content,
                "createdAt": index + 2,
            }
        )


_IMAGE_SEARCH_WIRE = [
    {"role": "user", "content": "how heavy is a ZQXVARA123 ragdoll"},
    _assistant_call("web_search", '{"query":"ragdoll ZQXVARA123"}', id = "w1"),
    {"role": "tool", "tool_call_id": "w1", "content": _SEARCH_TEXT_REPLAYED},
    {"role": "assistant", "content": "A ZQXVARA123 ragdoll weighs 6 kg."},
]


def test_a_turn_that_returned_pictures_still_finds_its_transcript_seat(conn):
    """Transcript seats are matched on stored rows, so the image envelope must be unwrapped there too."""
    _persist_image_search_turn()

    positions = conversation_archive._transcript_positions(THREAD)

    assert len(positions) == 2, positions
    assert conversation_archive._occurrences(positions, _IMAGE_SEARCH_WIRE[1:]) == [1]


def test_a_recalled_turn_that_returned_pictures_survives_the_branch_filter(conn):
    """Recall without a branch falls back to stored rows, which must unwrap pictures the same way."""
    _persist_image_search_turn()
    conversation_archive.archive_turns(THREAD, _IMAGE_SEARCH_WIRE)

    with_branch = conversation_archive.recall(
        THREAD, "ZQXVARA123 ragdoll", branch_messages = _IMAGE_SEARCH_WIRE
    )
    without_branch = conversation_archive.recall(THREAD, "ZQXVARA123 ragdoll")

    assert with_branch is not None and "6 kg" in with_branch[0]
    assert without_branch is not None, "the stored rows rejected a turn that is on branch"
    assert "6 kg" in without_branch[0]


def test_a_reply_that_shows_the_picture_still_finds_its_transcript_seat(conn):
    """Replay strips the image token from a reply that shows a picture; the seat must still match."""
    answer = "%s\n\n[[img:%s]]" % (_ANSWER, _IMAGE_ID)
    _persist_image_search_turn(answer = answer)

    positions = conversation_archive._transcript_positions(THREAD)

    assert conversation_archive._occurrences(positions, _IMAGE_SEARCH_WIRE[1:]) == [1], positions


def test_an_audio_reply_is_replayed_as_the_sentinel_the_request_carried():
    """Inline audio is replayed as the sentinel the request carried, not the whole wav."""
    row = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": '<audio-player src="data:audio/wav;base64,QUJD" />'}
            ],
        }
    ]

    assert conversation_archive._probe_text(conversation_archive._as_wire(row)[0]) == (
        '<audio-player src="[audio]" />'
    )
    assert (
        conversation_archive._probe_text(
            conversation_archive._as_wire(row, sanitise_assistant = False)[0]
        )
        == '<audio-player src="data:audio/wav;base64,QUJD" />'
    )
    user = [{"role": "user", "content": [{"type": "text", "text": "[[img:%s]]" % _IMAGE_ID}]}]
    assert conversation_archive._probe_text(conversation_archive._as_wire(user)[0]) == (
        "[[img:%s]]" % _IMAGE_ID
    )


def test_the_branch_seed_scores_an_in_order_run_not_a_set(conn):
    """The branch seed must score an in-order run, not a set, so a sibling with the same texts
    cannot win."""
    from storage import studio_db

    studio_db.upsert_chat_thread(
        {"id": THREAD, "title": "t", "modelType": "base", "modelId": "local-model", "createdAt": 1}
    )
    rows = [
        ("m0", None, "user", "A", 2),
        ("m1", "m0", "assistant", "a1", 3),
        ("m2", "m1", "user", "B", 4),
        ("m3", "m2", "assistant", "b1", 5),
        ("m4", "m3", "user", "B", 6),
        ("m5", "m4", "assistant", "b1", 7),
        ("n0", "m1", "user", "B", 8),
        ("n1", "n0", "assistant", "b1", 9),
    ]
    for identifier, parent, role, text, created in rows:
        studio_db.upsert_chat_message(
            {
                "id": identifier,
                "threadId": THREAD,
                "parentId": parent,
                "role": role,
                "content": [{"type": "text", "text": text}],
                "createdAt": created,
            }
        )

    branch = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "B"},
        {"role": "assistant", "content": "b1"},
        {"role": "user", "content": "B"},
        {"role": "assistant", "content": "b1"},
    ]
    positions = conversation_archive._transcript_positions(THREAD, branch = branch)

    assert len(positions) == 3, positions
    assert conversation_archive._occurrences(positions, branch[2:4]) == [1, 2]


def test_a_batch_mixing_a_search_with_an_ordinary_tool_keeps_its_transcript_span(conn):
    """archive_messages must bound the branch check by the transcript span, not the stripped group."""
    group = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "search_conversation", "arguments": '{"q":"x"}'}},
                {"id": "c2", "function": {"name": "terminal", "arguments": '{"command":"ls"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "earlier turns about the repo"},
        {"role": "tool", "tool_call_id": "c2", "content": "main.py readme.md"},
        {"role": "assistant", "content": "The repo has two files."},
    ]
    archivable = conversation_archive._archivable(group)
    assert len(archivable) == 3 and len(group) == 4

    rendered = conversation_archive.render_turn(archivable)
    text = rendered[1] if isinstance(rendered, tuple) else rendered
    live = conversation_archive.branch_message_texts(group)

    assert (
        conversation_archive._document_matches_one_run([{"text": text}], live, len(group)) is True
    )

    _archive(group)
    scope = store.conversation_archive_scope(THREAD)
    spans = [
        row["archive_messages"]
        for row in conn.execute(
            "SELECT archive_messages FROM documents WHERE scope=?", (scope,)
        ).fetchall()
    ]
    assert spans == [4], spans


def test_a_system_prompt_does_not_stall_the_branch_seed(conn):
    """A system prompt not in the stored chain must not stall the branch seed; skip it."""
    live = _branch_switch_thread()
    with_system = [{"role": "system", "content": "You are a helpful assistant."}] + live

    plain = conversation_archive._transcript_positions(THREAD, branch = live)
    seeded = conversation_archive._transcript_positions(THREAD, branch = with_system)

    assert len(plain) == 3, plain
    assert seeded == plain


def test_the_branch_seed_reaches_a_leaf_older_than_the_retry_pile(conn):
    """The branch seed must reach a leaf older than many retries, so its candidate cap cannot be small."""
    from storage import studio_db

    live = _branch_switch_thread()
    for index in range(40):
        studio_db.upsert_chat_message(
            {
                "id": f"r{index}",
                "threadId": THREAD,
                "parentId": "m1",
                "role": "user",
                "content": [{"type": "text", "text": f"abandoned retry {index}"}],
                "createdAt": 100 + index,
            }
        )

    positions = conversation_archive._transcript_positions(THREAD, branch = live)

    assert len(positions) == 3, positions
    assert conversation_archive._occurrences(positions, live[4:6]) == [2]


def test_an_unfinished_local_tool_call_is_not_replayed_at_all():
    """A cancelled local tool call is dropped entirely, not replayed as a call with its result missing."""
    row = {
        "role": "assistant",
        "content": [
            {
                "type": "tool-call",
                "toolCallId": "c1",
                "toolName": "terminal",
                "provenance": {"source": "local"},
            },
            {"type": "text", "text": "cancelled, moving on"},
        ],
    }

    wire = conversation_archive._as_wire([row])

    assert [message.get("role") for message in wire] == ["assistant"]
    assert "tool_calls" not in wire[0]
    assert not any(
        isinstance(part, dict) and part.get("type") == "tool-call" for part in wire[0]["content"]
    ), "an unreplayable call was rebuilt into the wire form"


def test_two_completed_local_tool_calls_replay_as_two_rounds():
    """Each completed local tool pair is its own group, or the request's exchanges merge into one."""
    row = {
        "role": "assistant",
        "content": [
            {
                "type": "tool-call",
                "toolCallId": "c1",
                "toolName": "terminal",
                "provenance": {"source": "local"},
                "result": "one",
            },
            {
                "type": "tool-call",
                "toolCallId": "c2",
                "toolName": "terminal",
                "provenance": {"source": "local"},
                "result": "two",
            },
        ],
    }

    wire = conversation_archive._as_wire([row])

    assert [message.get("role") for message in wire] == [
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert [message.get("tool_call_id") for message in wire if message["role"] == "tool"] == [
        "c1",
        "c2",
    ]


def test_a_new_local_tool_round_starts_a_new_group():
    """`startsNewCodexToolRound`: same round batches, a different round flushes."""

    def _call(identifier, round_id):
        return {
            "type": "tool-call",
            "toolCallId": identifier,
            "toolName": "terminal",
            "provenance": {"source": "local", "round_id": round_id},
            "result": identifier,
        }

    wire = conversation_archive._as_wire(
        [{"role": "assistant", "content": [_call("c1", 1), _call("c2", 1), _call("c3", 2)]}]
    )

    assert [message.get("role") for message in wire] == [
        "assistant",
        "tool",
        "tool",
        "assistant",
        "tool",
    ]
    assert [call["id"] for call in wire[0]["tool_calls"]] == ["c1", "c2"]
    assert [call["id"] for call in wire[3]["tool_calls"]] == ["c3"]


def test_a_topped_up_copy_keeps_the_transcript_span(conn):
    """The top-up re-embed must record the transcript span, not the group length, or recall drops it."""
    group = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "search_conversation", "arguments": '{"q":"x"}'}},
                {"id": "c2", "function": {"name": "terminal", "arguments": '{"command":"ls"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "earlier turns about the repo"},
        {"role": "tool", "tool_call_id": "c2", "content": "main.py readme.md"},
        {"role": "assistant", "content": "The repo has two files."},
    ]
    archivable = conversation_archive._archivable(group)
    assert len(archivable) == 3 and len(group) == 4

    _archive(group)
    scope = store.conversation_archive_scope(THREAD)
    digest = [
        row["sha256"]
        for row in conn.execute("SELECT sha256 FROM documents WHERE scope=?", (scope,)).fetchall()
    ][0]

    assert (
        conversation_archive._write_copy(
            conn,
            scope = scope,
            thread_id = THREAD,
            roles = "assistant",
            digest = digest,
            identity = "test-embedder",
            group = archivable,
            span = len(group),
            chunks = [],
            vectors = [],
            seats = [0, 1],
        )
        is True
    )
    conn.commit()

    spans = [
        row["archive_messages"]
        for row in conn.execute(
            "SELECT archive_messages FROM documents WHERE scope=?", (scope,)
        ).fetchall()
    ]
    assert spans == [4, 4], spans


def test_a_tool_turn_with_a_preamble_still_gets_its_seat():
    """A tool turn with preamble text must still get its seat; the second JSON spelling breaks the match."""
    user = {"role": "user", "content": "what files are here"}
    row = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me check"},
            _tool_part(result = "main.py"),
        ],
    }
    positions = [
        [
            conversation_archive._normalise_cased(conversation_archive._probe_text(message))
            for message in conversation_archive._as_wire([record])
        ]
        for record in (user, row)
    ]
    live = [
        _assistant_call("terminal", '{"command": "ls"}', content = "Let me check"),
        {"role": "tool", "tool_call_id": "c1", "content": "main.py"},
    ]

    assert conversation_archive._occurrences(positions, live) == [1]


def test_the_same_text_over_a_longer_span_widens_the_stored_window(conn):
    """Same text over a longer span must widen the stored window, or the turn is filtered out of recall."""
    short = [
        _assistant_call("terminal", '{"command":"ls"}', id = "c2"),
        {"role": "tool", "tool_call_id": "c2", "content": "main.py readme.md"},
        {"role": "assistant", "content": "The repo has two files."},
    ]
    long = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "search_conversation", "arguments": '{"q":"x"}'}},
                {"id": "c2", "function": {"name": "terminal", "arguments": '{"command":"ls"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "earlier turns about the repo"},
        {"role": "tool", "tool_call_id": "c2", "content": "main.py readme.md"},
        {"role": "assistant", "content": "The repo has two files."},
    ]
    assert conversation_archive.render_turn(
        conversation_archive._archivable(short)
    ) == conversation_archive.render_turn(conversation_archive._archivable(long))

    _archive(short)
    _archive(long)

    scope = store.conversation_archive_scope(THREAD)
    rows = conn.execute(
        "SELECT archive_messages, sha256 FROM documents WHERE scope=?", (scope,)
    ).fetchall()
    assert [row["archive_messages"] for row in rows] == [4], "the window stayed at the shorter span"

    text = conversation_archive.render_turn(conversation_archive._archivable(long))
    live = conversation_archive.branch_message_texts(long)
    assert conversation_archive._document_matches_one_run([{"text": text}], live, 3) is False
    assert conversation_archive._document_matches_one_run([{"text": text}], live, 4) is True


def test_a_reasoning_turn_is_reconstructed_without_its_thinking():
    """Reasoning is not content: replay leaves thinking out, as the wire carries only the answer."""
    row = {
        "role": "assistant",
        "content": [
            {"type": "reasoning", "text": "The user wants the file list. I should run ls."},
            {"type": "text", "text": "There are two files."},
        ],
    }
    live = [
        {
            "role": "assistant",
            "content": "There are two files.",
            "reasoning_content": "The user wants the file list. I should run ls.",
        }
    ]

    stored = [conversation_archive._probe_text(m) for m in conversation_archive._as_wire([row])]

    assert stored == [conversation_archive._probe_text(live[0])] == ["There are two files."]
    assert conversation_archive._occurrences([stored], live) == [0]


def test_a_unicode_tool_result_is_serialised_the_way_javascript_serialises_it():
    """Tool JSON serialises like JSON.stringify: non-ASCII stays as-is, where json.dumps escapes it."""
    assert (
        conversation_archive._tool_result_content({"ville": "Montréal"}, "terminal")
        == '{"ville":"Montréal"}'
    )


def test_a_long_tool_exchange_stays_on_branch_across_a_chunk_boundary():
    """Chunk overlap can start in an earlier message, so resuming at the last message misses it."""
    from core.rag import config
    from core.rag.chunking import chunk_pages
    from core.rag.parsers import Page

    def _matches(lines: int, count) -> bool:
        group = [
            {
                "role": "user",
                "content": "\n".join(
                    f"line {i} of the question about the repo" for i in range(lines)
                ),
            },
            _assistant_call("grep", '{"pattern":"foo"}'),
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "\n".join(
                    f"result row {i} with some matching text here" for i in range(200)
                ),
            },
            {"role": "assistant", "content": "That is the whole match list."},
        ]
        text = conversation_archive.render_turn(group)
        chunks = chunk_pages(
            [Page(text = text, page_number = None, char_count = len(text))],
            max_tokens = config.CHUNK_TOKENS,
            overlap = config.CHUNK_OVERLAP,
            count = count,
        )
        assert len(chunks) > 1, "this test needs a turn that really crosses a chunk boundary"
        return conversation_archive._document_matches_one_run(
            [{"text": chunk.text} for chunk in chunks],
            conversation_archive.branch_message_texts(group),
            len(group),
        )

    for count in (lambda t: max(1, len(t) // 4), lambda t: max(1, len(t.split()))):
        missed = [lines for lines in range(1, 90) if not _matches(lines, count)]
        assert not missed, f"unedited turns retired as off-branch at question lengths {missed}"


def test_one_pass_holding_both_spans_widens_the_window_too(conn):
    """The locked duplicate check must widen the stored window, or the longer turn is unsearchable."""
    short = [
        _assistant_call("terminal", '{"command":"ls"}', id = "c2"),
        {"role": "tool", "tool_call_id": "c2", "content": "main.py readme.md"},
        {"role": "assistant", "content": "The repo has two files."},
    ]
    long = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "search_conversation", "arguments": '{"q":"x"}'}},
                {"id": "c2", "function": {"name": "terminal", "arguments": '{"command":"ls"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "earlier turns about the repo"},
        {"role": "tool", "tool_call_id": "c2", "content": "main.py readme.md"},
        {"role": "assistant", "content": "The repo has two files."},
    ]

    _archive(short + [{"role": "user", "content": "and again please"}] + long)

    scope = store.conversation_archive_scope(THREAD)
    spans = [
        row["archive_messages"]
        for row in conn.execute(
            "SELECT archive_messages FROM documents WHERE scope=? AND archive_messages >= 3",
            (scope,),
        ).fetchall()
    ]
    assert spans == [4], spans


def test_a_repeat_that_came_back_into_the_prompt_keeps_one_copy(conn):
    """A repeat returning to the prompt must keep one copy, not one per seat, or recall repeats itself."""
    import hashlib

    repeat = _turn("set ZQXVARA123 to 1", "ok")
    middle = _turn("tell me about ZQXVARA123 pelicans", "sure")
    tail = _turn("and now something else about ZQXVARA123", "fine")
    conversation = repeat + middle + list(repeat) + tail
    _save_thread(THREAD, conversation)
    scope = store.conversation_archive_scope(THREAD)
    digest = hashlib.sha256(
        conversation_archive.render_turn(repeat).encode("utf-8", "ignore")
    ).hexdigest()

    conversation_archive.archive_turns(THREAD, conversation[:6], live = tail)
    assert len(store.documents_by_hash(conn, scope, digest)) == 2

    conversation_archive.archive_turns(THREAD, repeat, live = list(repeat) + tail)

    assert len(store.documents_by_hash(conn, scope, digest)) == 1
    found = conversation_archive.recall(THREAD, "ZQXVARA123", top_k = 4)
    assert found is not None
    texts = [source["text"] for source in found[1]]
    assert len(texts) == len(set(texts)), texts
