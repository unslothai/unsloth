# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The archive behind rolling-context compaction: what it keeps, and what it must not touch."""

import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag import config, conversation_archive, retrieval, store  # noqa: E402
from storage import rag_db  # noqa: E402

THREAD = "thread-abc"


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
    """Persist a transcript the way the chat history route does.

    ``append`` matters: replacing the rows on every archive call would leave only the
    newest turn saved and the branch filter would reject everything archived earlier.
    Tests that rewind a thread pass append=False to replace.
    """
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
    # A rewind, through the same prune_missing sync the PUT route uses. Deleting the
    # thread instead would tombstone its id, and recreating a tombstoned id raises.
    studio_db.sync_chat_messages(thread_id, rows, prune_missing = True)


def _archive(
    messages,
    thread_id = THREAD,
    *,
    persist = True,
):
    """The module opens its own connection to the same temp DB the fixture points at.

    Archiving requires the thread to exist in studio.db, so the default persists a
    matching transcript: only a persisted thread can be deleted, and an unreachable
    archive is the temporary-chat leak. ``persist=False`` exercises the refusal.
    """
    if persist:
        _save_thread(thread_id, messages, append = True)
    return conversation_archive.archive_turns(thread_id, messages)


@pytest.fixture
def conn(rag_home, rag_conn, stub_embeddings):
    return rag_conn


def test_evicted_turns_are_archived_under_the_conversation_scope(conn):
    written = _archive(_turn("what is a duck", "a waterfowl"))

    scope = store.conversation_archive_scope(THREAD)
    assert written == 1
    documents = store.list_documents(conn, scope)
    assert len(documents) == 1
    assert "earlier turn" in documents[0]["filename"]


def test_re_archiving_the_same_turns_writes_nothing(conn):
    """The same turns are evicted again on every later request, so repeats must be free.

    `persist = False` on the repeat because that is what the scenario actually is: the
    transcript is written once and the SAME turns are handed to the archive again on the
    next request. Appending them to the thread a second time would describe a different
    situation, a user who said the same thing twice, which is now a real distinction:
    ordinals come from the transcript, and a genuine repeat is stored as its own turn.
    """
    turn = _turn("what is a duck", "a waterfowl")
    first = _archive(turn)
    second = _archive(turn, persist = False)

    scope = store.conversation_archive_scope(THREAD)
    assert (first, second) == (1, 0)
    assert len(store.list_documents(conn, scope)) == 1


def test_archive_accumulates_across_compaction_epochs(conn):
    """Compaction N must still find what compaction 1 evicted.

    Latest-compaction-only measured 0.058 against 0.450 cumulative, so this is the
    property the feature rests on.
    """
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
    """The hazard that decided the scope layout.

    Thread-attached documents are injected in full on every request, so an archive in the
    thread scope would re-inject the evicted history each turn and undo itself.
    """
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
    """Lexical first, because recalling a conversation is mostly exact matching.

    Measured on a 30-turn walkthrough with identical wrapper text per turn: the needle
    chunk ranked 3rd lexically at any k, was never returned by dense retrieval, and RRF
    pushed it to 16th. Hybrid alone lost the answer, so this pins the ordering.
    """
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
    """Editing an earlier message rewinds the thread onto a new branch.

    The archive is append-only and still holds the abandoned continuation, so without a
    branch check a recall returns a turn that never happened here. Verified live.
    """
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
    # The saved thread now holds only the surviving branch.
    _save_thread("branch-thread", kept)

    survived = conversation_archive.recall("branch-thread", "KEEPME-1111")
    abandoned = conversation_archive.recall("branch-thread", "GONEAWAY-2222")

    assert survived is not None and "KEEPME-1111" in survived[0]
    # Recall has no relevance floor, so this query can still return the surviving turn.
    # What matters is that the rolled-back content itself never comes back.
    assert abandoned is None or "GONEAWAY-2222" not in abandoned[0]
    assert "GONEAWAY-2222" not in (survived[0] or "")


def test_recall_filters_to_the_ACTIVE_branch_not_the_whole_stored_thread(conn):
    """Retry keeps the replaced response as a sibling, and siblings are stored rows.

    Editing prunes the abandoned rows, so thread-wide filtering suffices there. Retry
    keeps both responses on purpose, so the thread-wide blob still contains the one the
    user replaced, and only the branch the request was sent on separates them.
    """
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
    # Both branches remain stored, which is exactly what Retry leaves behind.
    _save_thread("retry-thread", live)
    _save_thread("retry-thread", retried_away, append = True)

    # Thread-wide filtering cannot reject the sibling: its text is genuinely in the DAG.
    thread_wide = conversation_archive.recall("retry-thread", "SIBLING-3333")
    assert thread_wide is not None and "SIBLING-3333" in thread_wide[0]

    # Told which branch this is, it is rejected while the live answer still comes back.
    on_branch = conversation_archive.recall("retry-thread", "SIBLING-3333", branch_messages = live)
    assert on_branch is None or "SIBLING-3333" not in on_branch[0]
    survived = conversation_archive.recall("retry-thread", "KEEPME-1111", branch_messages = live)
    assert survived is not None and "KEEPME-1111" in survived[0]


def test_the_reply_that_FOLLOWS_a_forced_recall_is_still_archived(conn):
    """group_turns keeps a tool call, its result and the reply after it in one group.

    Rejecting the whole group over our own injection threw away the model's answer, so a
    later search found what was asked and never what was said, on compaction turns.
    """
    evicted = [
        {"role": "user", "content": "what was the passphrase"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "conv_recall_1",
                    "function": {"name": "search_conversation", "arguments": '{"query": "pass"}'},
                }
            ],
        },
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
    # The retrieved passage itself is still kept out, or the archive feeds on itself.
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
    """An archive is only written while vec0 loads, but it can stop loading afterwards.

    A venv change is enough (common on macOS), and a delete that silently does nothing
    leaves the turns on disk, ready to answer again once the extension loads.
    """
    turns = _turn("what is the passphrase", "the passphrase is VECGONE-2020")
    _save_thread("vecless-thread", turns, append = True)
    assert conversation_archive.archive_turns("vecless-thread", turns) == 1

    def no_vec():
        raise rag_db.RagExtensionUnavailable("vec0 will not load")

    monkeypatch.setattr(rag_db, "get_connection", no_vec)
    removed = conversation_archive.delete_for_thread("vecless-thread")
    monkeypatch.undo()

    assert removed == 1
    # And with the extension back, nothing of that conversation is left to find.
    assert conversation_archive.has_archive("vecless-thread") is False
    assert conversation_archive.recall("vecless-thread", "VECGONE-2020") is None


def test_a_turns_CHUNKS_must_all_sit_in_the_same_place_on_the_branch(conn):
    """The chunks of one turn are consecutive slices of one rendering.

    Validating them independently reassembles a turn from parts that never sat together:
    head on the current answer, tail on a later message repeating what the edit removed.
    """
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

    # Each chunk on its own still matches somewhere: that is exactly the trap.
    texts = conversation_archive.branch_message_texts(edited_with_a_later_echo)
    assert all(conversation_archive._on_live_branch(row["text"], texts) for row in rows)
    # As one document sharing a run, it is correctly rejected.
    assert conversation_archive._document_matches_one_run(rows, texts) is False
    # And a turn that really is still there is still accepted.
    assert (
        conversation_archive._document_matches_one_run(
            rows, conversation_archive.branch_message_texts(intact)
        )
        is True
    )


def test_a_turns_CHUNKS_cannot_spill_into_the_message_after_the_turn(conn):
    """The run is bounded by the MESSAGES the turn was rendered from, not by its lines.

    A turn is two or three messages however long it is, so bounding by line count let a
    long answer's tail be satisfied well outside the turn. Here the edit removes the end
    of the answer and the next message repeats it, the shape of a short correction.
    Chunks are shaped as the chunker produces them, so a continuation starts mid-message.
    """
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
    # The same two chunks, with the answer still whole, are still accepted.
    assert (
        conversation_archive._document_matches_one_run(
            rows, conversation_archive.branch_message_texts(intact)
        )
        is True
    )


def test_a_search_the_MODEL_asked_for_is_not_archived_as_new_history():
    """`_is_injected` knows this feature's own ids, and the model's searches carry none.

    A model-emitted `search_conversation` gets an ordinary `call_N` id from the parser, so
    both the call and the passages it retrieved were indexed as fresh conversation. A
    second search then archived the first one's output inside its own, one nesting level
    per distinct search, each copy competing for the four recall slots.

    Removed by NAME, and only the retrieval parts: the reply that follows a search is real
    conversation, and an assistant message can carry an ordinary call beside the search.
    """
    from core.rag import conversation_archive as archive

    recalled = [
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
        {"role": "tool", "tool_call_id": "call_0", "content": "<chunk>RETRIEVEDPASSAGE</chunk>"},
        {"role": "assistant", "content": "It was ZQXVARA123."},
    ]

    rendered = archive.render_turn(archive._archivable(recalled))
    assert "RETRIEVEDPASSAGE" not in rendered
    assert "ZQXVARA123" in rendered

    # A retrieval call beside an ordinary one loses only the retrieval half.
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


def test_swapping_the_tool_retires_the_archived_call():
    """Which tool ran is part of what the turn says, so it has to be part of the probe.

    The label `render_turn` writes ("assistant called terminal: <args>") was stripped
    whole, name included, leaving only the arguments to match. A retry that kept the
    arguments and changed the tool therefore left the archived pre-edit turn eligible, and
    it could be recalled as though the old call had happened on this branch. `_probe_text`
    renders a live call as "<name> <arguments>", so the name is there to be required.
    """
    from core.rag import conversation_archive as archive

    archived = archive.render_turn(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "terminal", "arguments": '{"cmd":"ls -la /srv"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "total 12"},
        ]
    )

    def _branch(tool):
        return [
            archive._normalise(archive._probe_text(message))
            for message in [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": tool, "arguments": '{"cmd":"ls -la /srv"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "total 12"},
            ]
        ]

    assert archive._on_live_branch(archived, _branch("terminal")) is True
    assert archive._on_live_branch(archived, _branch("python")) is False


def test_the_tool_call_exemption_ends_where_the_call_does():
    """The exemption belongs to the CALL, not to the rest of the message.

    A stored tool call cannot line up character for character with the live text, because
    the store keeps arguments as an object and offers both JSON spellings, so the cursor
    after one is not exact and the anchors have to be relaxed. Left set for the remainder
    of the message, an assistant turn carrying both a call and text stayed matched after a
    correction was appended to that text, and the pre-edit turn was still recallable.
    Once an ordinary text probe has matched, the cursor is exact again.
    """
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
    """An edit that adds a line BETWEEN two archived lines is still an edit.

    The two anchors either side of this covered an edit that prepends to a message the run
    stepped into and one that appends to a message it is leaving. A correction dropped
    between two archived lines matched both probes with the new line sitting unexamined in
    the gap, so the pre-edit turn stayed recallable and could be quoted back as current.

    A label `render_turn` wrote is still allowed in that gap, or a pasted chat log carrying
    its own "user:" lines would retire turns nobody touched.
    """
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
    """Counting role labels counts lines the USER wrote, not just the renderer's.

    A pasted chat log carries lines that look exactly like `render_turn`'s, each widening
    the run by a message, enough for the message after an edited turn to supply what the
    edit removed. The turn's real size is recorded when it is archived.
    """
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

    # Two messages, whatever the rendered text looks like.
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
    """Two generations compacting the same thread both clear the hash check.

    The embedding pass sits between the check and the insert, and `(scope, sha256)` is a
    plain index, so both wrote and the duplicate took two of the few recall slots.
    """
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
        # Both passes are inside the window: neither has inserted yet.
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
    """The branch check has to stay inside the turn it is checking.

    Against one flattened transcript, any later message repeating the words satisfies a
    missing line, so a short answer could survive being edited away.
    """
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
    # And the turn that IS still there is still found, or the check is just "no".
    assert (
        conversation_archive._on_live_branch(
            archived, conversation_archive.branch_message_texts(still_there)
        )
        is True
    )


def test_a_turn_whose_lines_were_REORDERED_is_no_longer_on_the_branch(conn):
    """Independent line membership accepts a turn that was merely rearranged.

    Every probe still occurs somewhere, so the pre-edit ordering would be served back.
    """
    original = [{"role": "assistant", "content": "REORDER-A first\nREORDER-B second"}]
    archived = conversation_archive.render_turn(original)

    same = conversation_archive.branch_message_texts(original)
    swapped = conversation_archive.branch_message_texts(
        [{"role": "assistant", "content": "REORDER-B second\nREORDER-A first"}]
    )

    assert conversation_archive._on_live_branch(archived, same) is True
    assert conversation_archive._on_live_branch(archived, swapped) is False


def test_a_tool_turn_with_BOTH_text_and_a_call_stays_on_its_branch(conn):
    """Ordered matching only works if both sides agree on the order.

    render_turn writes a tool call before any assistant text on the same message and the
    result after it, so both transcript shapes (`tool_calls`, `tool-call` parts) must lay
    a turn out the same way.
    """
    request_shape = [
        {"role": "user", "content": "check the log"},
        {
            "role": "assistant",
            "content": "I will read it now",
            "tool_calls": [
                {"id": "c1", "function": {"name": "terminal", "arguments": '{"cmd":"cat log"}'}}
            ],
        },
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
    # The stored shape keeps arguments as an object, so its spacing is not the model's.
    assert conversation_archive._on_live_branch(
        archived, conversation_archive.branch_message_texts(stored_shape)
    )


def test_editing_ONE_chunk_of_a_long_turn_retires_the_whole_turn(conn):
    """A turn longer than CHUNK_TOKENS is stored as several chunks of one document.

    Per-chunk checking retires only the chunks carrying an edit, leaving untouched
    earlier chunks of the same retired turn eligible. The archived unit is the turn.
    """
    # The head spans several chunks, so the FIRST chunk lies entirely inside the part
    # that is never edited and passes a per-chunk check unchanged.
    head = "opening CHUNKSPLIT-7373 marker. " + ("unchanged opening sentence. " * 300)
    turn = [
        {"role": "user", "content": "explain the deploy process"},
        {"role": "assistant", "content": head + ("original ending sentence. " * 300)},
    ]
    _save_thread("chunk-thread", turn, append = True)
    assert conversation_archive.archive_turns("chunk-thread", turn) == 1
    # More than one chunk, or this test is not testing anything.
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

    # Same turn with its TAIL rewritten; the head holding the marker is untouched.
    rewritten = [
        turn[0],
        {"role": "assistant", "content": head + ("a completely different ending. " * 300)},
    ]
    # The first chunk really is still on the branch: this is what per-chunk admitted.
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
    """render_turn caps a tool result and marks the cut with a truncation marker.

    That result is ONE string of many newlines, so only its first line carries the "tool
    result:" label while the marker is on its last, and nothing in a real transcript ends
    in the marker: every archived tool turn over the cap was rejected as rolled back.
    """
    body = "opening line TOOLWALL-6060\n" + ("filler output line\n" * 400) + "trailing line"
    group = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "terminal", "arguments": '{"cmd": "cat log"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": body},
    ]
    text = conversation_archive.render_turn(group)
    assert text.endswith("...")
    assert len(text.splitlines()) > 100

    transcript = conversation_archive.branch_message_texts(group)

    assert conversation_archive._on_live_branch(text, transcript) is True
    # And an edit to the part that WAS archived still retires the copy.
    edited = conversation_archive.branch_message_texts(
        [
            group[0],
            {**group[1], "content": body.replace("opening line", "rewritten line")},
        ]
    )
    assert conversation_archive._on_live_branch(text, edited) is False


def test_recall_widens_past_a_wall_of_abandoned_branch_hits(conn):
    """One over-fetch is not enough when the abandoned branch is long.

    Rewinding a compacted continuation leaves enough stale turns to fill any fixed
    candidate window, so a single fetch rejects the whole page and never looks at the
    live match just below it, reporting nothing while the answer is in the archive.
    """
    # The live answer mentions the marker once, in a turn that is mostly other words.
    live = _turn(
        "where is the marker",
        "the marker is WIDEN-5150 and the rest of this answer is about unrelated matters "
        "such as scheduling, packaging, release notes and the weather in three cities",
    )
    # The abandoned branch repeats it, so these outrank the live turn lexically and fill
    # any fixed candidate window ahead of it.
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
        # Stored as siblings, as Retry leaves them: on the thread, not on the branch.
        _save_thread("wall-thread", turn, append = True)
        conversation_archive.archive_turns("wall-thread", turn)

    found = conversation_archive.recall("wall-thread", "WIDEN-5150", branch_messages = live)

    assert found is not None
    assert "the marker is WIDEN-5150" in found[0]
    assert "discarded" not in found[0]


def test_the_branch_transcript_carries_request_shaped_tool_calls(conn):
    """A tool turn's arguments live in `tool_calls`, not in content, on the wire.

    render_turn indexes them, so a branch blob built from content alone misses every
    archived tool turn and filters the whole exchange out as rolled back.
    """
    branch = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "terminal", "arguments": '{"cmd": "ls TOOLARG-7777"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "TOOLARG-7777 listed"},
    ]
    _archive(branch, thread_id = "tool-branch-thread")
    _save_thread("tool-branch-thread", [{"role": "user", "content": "unrelated"}])

    found = conversation_archive.recall(
        "tool-branch-thread", "TOOLARG-7777", branch_messages = branch
    )

    assert found is not None and "TOOLARG-7777" in found[0]


def test_a_thread_that_was_never_persisted_is_never_archived(conn):
    """The temporary-chat guarantee.

    An incognito chat is never written to studio.db, yet the frontend still sends its
    thread_id and the request carries no incognito flag. Archiving it would persist the
    one conversation the user asked not to keep, where no deletion flow could reach it.
    """
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
    # Same predicate the fit asks before reserving, so an unarchivable chat never pays
    # for room to recall into.
    assert conversation_archive.can_archive("temporary-thread") is False


def test_a_thread_deleted_mid_ingest_does_not_leave_its_turns_behind(conn, monkeypatch):
    """Deleting a chat while it is compacting must not resurrect the archive.

    Cancellation is cooperative and the embedding pass between the liveness check and the
    commit does not observe it, so a sweep in that window is undone by the commit, into a
    scope no later delete can reach now the thread is gone.
    """
    from storage import studio_db

    turns = _turn("what is the code", "the code is DELETED-9999")
    _save_thread("doomed-thread", turns, append = True)

    original = conversation_archive.embeddings.encode_with_identity

    def delete_the_thread_mid_ingest(*args, **kwargs):
        # The interleaving that matters: rows first, sweep last, both before this commit.
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
    """A thread archived earlier whose saved rows are gone still answers.

    An empty transcript is absence of evidence, not evidence the turns are gone, so the
    branch check must not silently disable recall for an existing archive.
    """
    _archive(
        [
            {"role": "user", "content": "unsaved section, code ORPHAN-3333"},
            {"role": "assistant", "content": "noted"},
        ],
        thread_id = "unsaved-thread",
    )
    # Drop the saved rows, leaving the archive behind.
    from storage import studio_db

    studio_db.delete_chat_threads(["unsaved-thread"])

    found = conversation_archive.recall("unsaved-thread", "ORPHAN-3333")

    assert found is not None
    assert "ORPHAN-3333" in found[0]


def test_editing_only_the_assistant_half_retires_the_archived_turn(conn):
    """A turn is archived as a unit, so it lives or dies as a unit.

    Matching on the first surviving line kept serving the old answer: the unchanged user
    line vouched for the whole turn.
    """
    original = [
        {"role": "user", "content": "what is the launch code"},
        {"role": "assistant", "content": "the launch code is STALEANSWER-9999"},
    ]
    _archive(original, thread_id = "edited-thread")
    # The user keeps their question and rewrites the answer.
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
    """An empty 'completed' document would make the turn unarchivable forever.

    `document_by_hash` skips whatever it finds, and the row says completed, so nothing
    would ever retry it.
    """
    turns = _turn("what is a quokka", "a small marsupial")

    def explode(*args, **kwargs):
        raise RuntimeError("disk full")

    # Restored by re-setting, not monkeypatch.undo(): fixtures share this instance, so
    # undo() would also revert stub_embeddings and fail the retry for another reason.
    real_add_chunks = store.add_chunks
    monkeypatch.setattr(store, "add_chunks", explode)
    assert _archive(turns, thread_id = "retry-thread") == 0
    monkeypatch.setattr(store, "add_chunks", real_add_chunks)

    # The retry succeeds, which it cannot do if a completed husk was left behind.
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
    """The two previous fixes together could make tool turns permanently unrecallable.

    render_turn archives a tool turn as "assistant called X: args" and "tool result: ...",
    while assistant-ui persists a structured tool-call part the flattener used to drop, so
    with every archived line required in the transcript no tool turn could match.
    """
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
    # Persisted the way the UI stores it: a structured part, not text.
    studio_db.upsert_chat_message(
        {
            "id": "tool-thread-0",
            "threadId": "tool-thread",
            "role": "assistant",
            "content": [
                {
                    "type": "tool-call",
                    "toolCallId": "c1",
                    "toolName": "terminal",
                    "args": {"command": "alembic upgrade head"},
                    "result": "migration applied cleanly",
                }
            ],
            "createdAt": 2,
        }
    )
    conversation_archive.archive_turns("tool-thread", turn)

    found = conversation_archive.recall("tool-thread", "alembic upgrade head")

    assert found is not None
    assert "alembic" in found[0]


def test_an_edit_past_the_probe_cutoff_still_retires_the_turn(conn):
    """A prefix probe cannot see a change after its cut-off.

    Rewriting a long answer's tail left the archived copy matching on its first 160
    characters, so the stale text stayed eligible.
    """
    head = "the deployment steps are as follows and here is the full detail " * 4
    original = [
        {"role": "user", "content": "how do I deploy"},
        {"role": "assistant", "content": head + "finally run OLDSTEP-7777"},
    ]
    _archive(original, thread_id = "tail-edit-thread")
    # Same opening, different ending.
    _save_thread(
        "tail-edit-thread",
        [
            {"role": "user", "content": "how do I deploy"},
            {"role": "assistant", "content": head + "finally run NEWSTEP-8888"},
        ],
    )

    found = conversation_archive.recall("tail-edit-thread", "OLDSTEP-7777")

    assert found is None or "OLDSTEP-7777" not in found[0]


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
    """DELETE removes the rows, then awaits the sandbox pass, and only then sweeps here.

    Another tab can POST the same id in that window and its generation can archive turns
    under it before the sweep runs. The sandbox pass re-checks for exactly that; this had
    not, so the recreated chat silently lost its memory.
    """
    from routes import chat_history
    from storage import studio_db

    thread_id = "recreated-thread"
    turns = _turn("what is the code", "the code is 5150")
    _save_thread(thread_id, turns, append = True)
    assert conversation_archive.archive_turns(thread_id, turns) == 1

    chat_history._remove_conversation_archives([thread_id])

    assert conversation_archive.has_archive(thread_id) is True

    # And a thread that really is gone still has its archive dropped.
    studio_db.delete_chat_threads([thread_id])
    chat_history._remove_conversation_archives([thread_id])
    assert conversation_archive.has_archive(thread_id) is False


def test_an_answer_edited_by_appending_to_it_retires_the_archived_copy(conn):
    """Keeping the old text and adding to it left every probe matching.

    "No" becoming "No, correction: yes" is the ordinary way a person fixes an answer, and
    the pre-edit copy stayed eligible: a later search could return "No" as the answer with
    the correction nowhere in it.
    """
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
    """render_turn cuts long tool results, so that probe is a prefix by design.

    Demanding that the turn end where the live message does would retire every turn that
    carried one.
    """
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
    """The other side of the same edit: the old text is kept as a suffix.

    "No" becoming "Correction: no" leaves the probe matching and ending exactly where the
    live message does, so an end-only check still called the pre-edit copy live.
    """
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
    """What the branch filter costs when the branch is the client's messages alone.

    The exchange was created by this request, evicted by a later refit, and archived. The
    client never sent it, so filtering against those messages calls it an abandoned branch
    and refuses it, and the model loses a tool result it still needs to answer.
    """
    thread_id = "toolrun-thread"
    request_branch = [{"role": "user", "content": "find the deploy code in the repo"}]
    _save_thread(thread_id, request_branch, append = True)

    tool_exchange = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "grep", "arguments": '{"q": "deploy"}'}}
            ],
        },
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
    """Anchoring only the final message left the question editable underneath it.

    The turn is checked as a whole, so every message it claims has to be accounted for
    from its first character to its last, wherever the edit landed.
    """
    rows = [{"text": "user: should I deploy on Friday\nassistant: No"}]

    def _live(question, answer = "No"):
        return conversation_archive.branch_message_texts(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )

    match = conversation_archive._document_matches_one_run
    # The question, either side.
    assert match(rows, _live("Actually, should I deploy on Friday"), 2) is False
    assert match(rows, _live("should I deploy on Friday or wait"), 2) is False
    # The answer, either side.
    assert match(rows, _live("should I deploy on Friday", "Correction: no"), 2) is False
    assert match(rows, _live("should I deploy on Friday", "No, correction: yes"), 2) is False
    # And the turn as it was archived is still live.
    assert match(rows, _live("should I deploy on Friday"), 2) is True


def test_a_tool_call_message_is_exempt_from_the_character_anchors(conn):
    """The store keeps a call as a structured part, so nothing can line up exactly.

    The live text carries the tool name and BOTH spellings of the arguments, spaced and
    compact, while the archived copy has one line of one of them. Demanding coverage there
    would retire every tool turn in the archive.
    """
    live = conversation_archive.branch_message_texts(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool-call",
                        "toolCallId": "c1",
                        "toolName": "terminal",
                        "args": {"command": "alembic upgrade head"},
                        "result": "migration applied cleanly",
                    }
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
    """Dense search only reads documents whose embedder matches the query's.

    Hashed and skipped, a turn archived under the previous model stayed invisible to every
    paraphrased search for good, however often the client re-presented it.
    """
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
    # Same turn, same hash, under an embedder the query side no longer asks for.
    identity["name"] = "st:model-b"
    assert conversation_archive.archive_turns(thread_id, [dict(m) for m in turn]) == 1

    rows = conn.execute(
        "SELECT embedding_model FROM documents WHERE scope = ?",
        (store.conversation_archive_scope(thread_id),),
    ).fetchall()
    # Replaced, not duplicated.
    assert [row["embedding_model"] for row in rows] == ["st:model-b"]

    # And re-presenting it under the SAME embedder is still a no-op.
    assert conversation_archive.archive_turns(thread_id, [dict(m) for m in turn]) == 0


def test_a_first_compaction_embeds_its_turns_in_one_pass(conn, monkeypatch):
    """Per group, a long chat's first compaction ran dozens of jobs back to back.

    Both backends serialise them, so the reply could not start until the last one landed.
    """
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


# --- The subject of a conversation must not be the least findable thing in its archive ---
#
# Measured on the pre-fix build with `scripts/fact_update/retrieval_probe.py`: a variable
# revised seven times was archived 8/8 and retrieved 0/5, because BM25 gives almost no
# weight to a term present in half the chunks. `zqxvara123` scored 0.16; `value`, from the
# question itself, scored 4.755.

VARIABLE = "ZQXVARA123"


# Filler that VARIES. The defect is an IDF collapse, so it only reproduces while the
# question's incidental word ("value") is rare: a fixture repeating one distractor made
# "value" as common as the variable and passed against the unfixed build.
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
    """A variable assigned, then revised, with filler that shares the vocabulary.

    Values are fixed rather than random so a failure is reproducible, and the filler never
    names the variable, so it can compete for slots without ever being a correct answer.
    """
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
    """Pre-fix this returns four distractors and not one turn about the variable.

    What the fix guarantees is the CANDIDATE SET: only turns naming the thing asked
    about. Which of eight equally-scoring assignments wins a slot is BM25's business and
    is not claimed here -- see `test_the_newest_revision_is_recalled_when_there_is_room`
    for the part that is.
    """
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
    # Chronological presentation is the point: the LAST assignment read is the current one.
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
    # An all-function-word question must still search for something: an empty expression
    # makes search_lexical return [] and the recall silently vanishes.
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

    # `embedding_identity` too, not only the encode: `archive_turns` short-circuits on the
    # expected identity, so patching the encode alone never reaches the re-embed path.
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
    assert 'turn="1"' not in text  # the NULL one claims no position
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
    # First block, because the original assignment is the oldest turn in the set.
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
    # The claim is the RENDERING, not the order: nothing the previous build did not emit.
    assert "111111" in text and "222222" in text
    assert "turn=" not in text
    assert "supersedes" not in text
    assert "oldest first" not in text


def test_a_ubiquitous_identifier_cannot_crowd_out_the_newest_revision(conn):
    """The conjunctive pass FILTERS; it must not also rank, and must not fill `fetch`.

    FTS5 floors the BM25 IDF of a term present in more than half the index at 1e-6, so in
    an archive that is all about one variable the identifier orders nothing. Cutting that
    pass off at `fetch` therefore dropped the turn stating the current value and answered
    with the four oldest turns instead -- worse than the OR query it replaced.
    """
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
    """Two identifiers must not become a requirement to name BOTH.

    The turn that answers "what are A and B now" names one of them; the turns naming both
    are the older comparisons. Requiring the conjunction made every comparison eligible
    and both assignments ineligible, so the four slots went to the four oldest turns and
    neither current value came back -- the same lost answer as the ubiquitous-identifier
    case above, reached through the filter rather than through the ranking.
    """
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
    # And the filter still did its job: every slot names something that was asked about.
    assert all(
        VARIABLE.lower() in source["text"].lower() or other.lower() in source["text"].lower()
        for source in sources
    )


def test_the_newest_revision_survives_a_strict_pass_that_hit_its_cap(conn, monkeypatch):
    """Eligibility is a property of a chunk, not of the capped pass's top rows.

    The identifier pass is bounded at `_BRANCH_FILTER_MAX_CANDIDATES`, and by the same
    IDF floor its order within that bound carries no information. Past the bound the turn
    stating the current value can be the one left out, and reading absence from that list
    as "does not name the subject" ranked it behind every capped row, where the fetch
    window then dropped it. The cap is patched down here so the archive stays small; at
    the shipped 256 the same loss was measured on a 301-chunk archive, which a long
    thread reaches at roughly 60 turns once any of them carry a pasted block
    (CHUNK_TOKENS is 500).
    """
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


# Ordinary turns of a long engineering chat: each uses the question's content words the
# way a person would and none names the variable, so none can be a correct answer.
# "current" and "value" are plain English, which is the point.
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
    """The content-word pass ranks the ELIGIBLE chunks, so its window must reach them.

    Fetched at `fetch` it need not reach any: `fetch` ordinary turns using "current" or
    "value" about other things fill it end to end, none of them names the variable, and
    the membership probe -- which only classifies ids already in that window -- then has
    nothing to promote. The merged list falls back to the identifier pass's order, which
    the IDF floor makes uninformative (every turn merely discussing the variable scores
    the same, and the one stating its value scores WORSE, because it names the variable
    once where they name it three times). At 41 turns the assignment ranked 21st of 21
    and the recall answered with the four oldest turns instead.
    """
    for index in range(20):
        _archive(
            _turn(
                f"lets discuss {VARIABLE} aspect number {index}",
                f"{VARIABLE} is a config knob, remark {index} about how {VARIABLE} behaves",
            )
        )
    # A normal-length answer, not a one-liner: length normalisation is what puts it below
    # the short distractors in the content-word pass.
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
    # And the filter still holds: no slot goes to a turn that never names the subject.
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
    # What an upgraded database looks like: written before the column, never backfilled.
    conn.execute("UPDATE documents SET archive_ordinal=NULL WHERE scope=?", (scope,))
    conn.commit()
    assert _archive(_turn("newest pelican question", "the newest statement about pelicans")) == 1

    identity["name"] = "st:model-b"
    assert _archive([dict(message) for message in oldest]) == 1

    # By created_at, because the re-embed keeps the turn's original timestamp: the
    # unnumbered turn is still the oldest row in the scope.
    ordinals = [
        row["archive_ordinal"]
        for row in conn.execute(
            "SELECT archive_ordinal FROM documents WHERE scope=? ORDER BY created_at", (scope,)
        ).fetchall()
    ]
    # 1, not 0: the ordinal is the turn's POSITION, and the newest question is the second
    # turn. The old allocator said 0 by counting from MAX over a column holding one NULL.
    assert ordinals == [None, 1]
    text, _sources = conversation_archive.recall(THREAD, "pelicans", top_k = 4)
    assert text.index("oldest statement") < text.index("newest statement")


def test_a_re_embed_that_stops_partway_does_not_reorder_a_legacy_archive(conn, monkeypatch):
    """An archive with no ordinals is ordered by `created_at` and by nothing else, so a
    re-embed that re-stamps the rows it reaches puts them AFTER the rows it never got to.

    `archive_turns` is built to survive a pass that dies partway -- it logs, sets
    `_INGEST_FAILED` and leaves whatever it wrote searchable -- and a locked database or
    a full disk is an ordinary way to get there. The turns it managed to rewrite then
    carry the newest timestamps in the scope, so the oldest statements in the
    conversation are quoted LAST, under a header that says the list is oldest first and
    that the later turn supersedes the earlier one. The model is told the first answer
    is the current one.
    """
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
    # What an upgraded database looks like: archived before the column, never backfilled.
    conn.execute("UPDATE documents SET archive_ordinal=NULL WHERE scope=?", (scope,))
    conn.commit()

    # The embedder changes, so the whole archive is rewritten, and the rewrite dies on the
    # third turn.
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
    # The pass really did stop partway: two rows re-embedded, three left on the old model.
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
    """Every pre-ordinal row has `turn` None, so ordering them only by chunk index leaves
    them in whatever order the two queries happened to return them: the anchor's hits
    first, then the follow-up's. `_conversation_order` breaks exactly this tie with
    `created_at`, and the merged path has to agree with it or the block contradicts its own
    oldest-first header on an upgraded database."""
    from core.rag import conversation_archive

    merged = [
        {"turn": None, "createdAt": "2026-01-02T00:00:00Z", "chunkIndex": 0, "text": "later"},
        {"turn": None, "createdAt": "2026-01-01T00:00:00Z", "chunkIndex": 0, "text": "earlier"},
        {"turn": 3, "createdAt": "2026-01-03T00:00:00Z", "chunkIndex": 0, "text": "numbered"},
    ]
    merged.sort(
        key = lambda source: (
            source.get("turn") is not None,
            source.get("turn") or 0,
            source.get("createdAt") or "",
            source.get("chunkIndex") or 0,
        )
    )

    assert [m["text"] for m in merged] == ["earlier", "later", "numbered"]


def test_recall_sources_carry_the_fields_the_merge_orders_by():
    """The sort above is only as good as the field it reads, and nothing RENDERS
    `createdAt` or `chunkIndex`, so an unused-looking key is exactly the sort of thing a
    later cleanup deletes. This pins the producer."""
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
        },
    }
    hits = [SimpleNamespace(chunk_id = "c1", score = 0.5)]

    _, sources = tool.format_conversation_recall(rows, hits)

    assert sources[0]["createdAt"] == "2026-01-01T00:00:00Z"
    assert sources[0]["chunkIndex"] == 2
    assert sources[0]["turn"] is None


def test_the_forced_floor_filters_candidates_rather_than_deleting_results(conn, monkeypatch):
    """A floor applied after the top-k slice is a deletion, not a filter.

    Weak hits took the slots and were then removed, so at a floor of 0.5 with four weak
    candidates on top the forced recall returned nothing where the unforced one returned 4.
    Only reachable when an operator raises RAG_CONVERSATION_FORCED_MIN_SCORE off its 0.0
    default.
    """
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
    # And the tool-initiated path is untouched by the floor.
    assert conversation_archive.recall(THREAD, "pelican", top_k = 4) is not None


def test_the_newest_revision_survives_a_tied_run_LONGER_than_the_cap(conn):
    """Reordering a window cannot reach a turn that never entered it.

    Every hit on the conversation's own identifier ties at the IDF floor, and SQLite
    returns a fully tied run in rowid order, so the candidate cap took the OLDEST rows.
    Past that many chunks on the subject, the newest assignment was unreachable at any k:
    the ends-first ordering ran inside the cap and could only pick both ends of the window
    it was handed. The existing tie tests patch the cap down instead of exceeding it,
    which is why this went unnoticed.
    """
    count = conversation_archive._BRANCH_FILTER_MAX_CANDIDATES + 40
    for index in range(count - 1):
        _archive(_turn(f"note {index:03d} about ZQXVARA123", "noted"))
    _archive(_turn("set ZQXVARA123 to 9999", "done"))

    found = conversation_archive.recall(THREAD, "what is ZQXVARA123 currently", top_k = 4)

    assert found is not None
    assert "9999" in found[0]
    # The oldest end is still reachable: a fix that just returns the newest turns fails.
    oldest = conversation_archive.recall(THREAD, "what was ZQXVARA123 originally", top_k = 4)
    assert oldest is not None
    assert "note 000" in oldest[0]


def test_a_re_embedded_oldest_turn_is_still_reachable_past_the_cap(conn, monkeypatch):
    """Both halves have to be ordered by the ordinal, not just the newest one.

    A re-embed deletes and reinserts a chunk while keeping its ordinal, so rowid and
    conversation order diverge. Fetched by rowid, the oldest turn moved to the BACK of the
    front window while the newest-first leg deliberately skips it, and it was in neither
    half: measured, the ordinal-0 chunk's rowid went from 1 to 297 and "note 000" stopped
    being recallable at all.
    """
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

    # Only the oldest turn is re-embedded, which reinserts its chunk at the newest rowid.
    identity["name"] = "st:model-b"
    _archive([dict(message) for message in oldest_turn])

    oldest = conversation_archive.recall(THREAD, "what was ZQXVARA123 originally", top_k = 4)
    assert oldest is not None
    assert "note 000" in oldest[0]


def test_the_newest_revision_survives_a_tie_and_the_oldest_one_still_does(conn):
    """A tie in the score is not an order, and truncating it silently picked the past.

    FTS5 floors the IDF of a term that appears in more than half the index at 1e-6, and in
    a per-thread archive the identifier the whole conversation is about is exactly such a
    term. When the revisions share nothing else with the question, every one of them comes
    back with the SAME bm25, so `hits[:limit]` kept whichever rows SQLite happened to emit
    first, which is the oldest. Measured on eight revisions at top_k 4: one distinct score
    across all eight, and the recall returned revisions 1 to 4 with the current value
    absent.

    That is worse than a miss. `format_conversation_recall` tells the model that a later
    turn supersedes an earlier one, so the stale value is handed over as the authoritative
    one.

    Taking from both ends of an equal-score run is what keeps this test and the
    original-assignment guard true at the same time: preferring the newest outright fails
    that guard, which exists precisely to stop a fix that just returns the latest thing it
    can find.
    """
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
    """Two queries must never return LESS than either of them alone.

    The anchor is a second query drawn from the same thread as the first, so overlap is
    the normal case, not the corner. Each query's slice was cut to its share BEFORE the
    dedup, so every chunk they agreed on consumed a slot and left it empty: measured on
    six turns matching both queries at top_k 4, each query alone returned 4 sources and
    the pair returned 2, with four eligible chunks sitting unread. Adding the anchor
    that exists to RESCUE a thin message made the recall smaller than not adding it.
    """
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
    """Capitals only mean "identifier" where there is lower case to contrast with.

    In a line with no lower case at all the rule fires on every word, so the focused pass
    ORs in "what" and "the" and filters nothing, leaving the permissive BM25 ranking to
    hand the slot to whatever filler shares a content word. Measured on the same fixture
    as `test_the_questions_filler_cannot_outrank_the_subject`, at top_k 1: typed normally
    the slot went to the variable, shouted it went to the retry-budget turn.
    """
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
    """Making the capitals rule need contrast must not cost numbers their shape.

    A purely numeric subject qualified only through the capitals rule, because "9134"
    upper-cased is itself, and the shape rule demanded a letter as well. So in a shouted
    question it stopped being an identifier altogether, the focused pass was dropped, and
    measured at top_k 1 the slot went to a turn about a retry budget rather than the
    number asked about. Shape is now "contains a digit", which for any ordinary-case
    query is the answer the capitals rule already gave.
    """
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
    """The rollback knob says the candidate set is identical to before, so it must be.

    The tie-break reorders CANDIDATES, which is selection, not presentation, so leaving it
    outside the knob meant an operator who turned the feature off still got the new
    behaviour out of an archive whose scores are tied. Measured before this gate: the
    knobs-off recall returned the both-ends set rather than the four the previous build
    returned.
    """
    from core.rag import config

    monkeypatch.setattr(config, "CONVERSATION_QUERY_FOCUS", False)
    monkeypatch.setattr(config, "CONVERSATION_RECALL_ORDER", "relevance")
    values = _revisions(8, distractors = 0)

    found = conversation_archive.recall(THREAD, f"what is the current value of {VARIABLE}", top_k = 4)

    assert found is not None
    returned = [value for value in values if value in found[0]]
    assert returned == values[:4], f"the knob did not restore the previous selection: {returned}"


def test_a_turn_repeated_later_is_archived_again_at_its_own_position(conn):
    """Saying the same thing twice is two turns, and the second one is usually the point.

    The archive is idempotent by content hash, which is what makes re-archiving an
    eviction free, but hash alone treated a genuine repeat as a duplicate: "set X to 1",
    "set X to 2", "set X to 1" stored TWO documents for three turns, and the third turn,
    the one holding the current value, was never indexed at all, so no query could reach
    it. The recall then quoted turns 1 and 2 under a header stating that the higher turn
    number was said later and supersedes the earlier one, which told the model X was 2.
    """
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
    # Rendered oldest first with the repeat LAST, which makes the header's rule true.
    assert turns == sorted(turns)
    assert "set ZQXVARA123 to 1" in found[1][-1]["text"]


def test_a_repeat_still_in_the_prompt_is_not_archived_early(conn):
    """A turn said twice with only the older one evicted is ONE archived turn, for now.

    Seats come from the persisted transcript, so both occurrences count, and the same
    evicted group is handed back on every later request while the sticky boundary holds.
    The second pass then saw one stored copy against two seats, decided it was short, and
    wrote a document for the occurrence still sitting in the prompt. Both were recallable,
    so identical text took two of the four recall slots and one of them repeated what the
    model could already read.

    Bounded by what the fit KEPT instead. The copy is written when its own turn is
    evicted, at its own ordinal, which is the second half of this test.
    """
    repeat = _turn("set ZQXVARA123 to 1", "ok")
    middle = _turn("tell me about ZQXVARA123 pelicans", "sure")
    tail = _turn("and now something else about ZQXVARA123", "fine")
    conversation = repeat + middle + list(repeat) + tail
    _save_thread(THREAD, conversation)

    # Only the OLDEST copy has crossed the boundary; the newer one is still in the prompt.
    live = list(repeat) + tail
    conversation_archive.archive_turns(THREAD, repeat, live = live)
    conversation_archive.archive_turns(THREAD, repeat, live = live)

    scope = store.conversation_archive_scope(THREAD)
    assert len(store.list_documents(conn, scope)) == 1

    found = conversation_archive.recall(THREAD, "ZQXVARA123", top_k = 4)
    assert found is not None
    texts = [source["text"] for source in found[1]]
    assert len(texts) == len(set(texts))

    # Once the newer copy is evicted too, it is archived at the position it was said.
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
    """A REPLACEMENT is not an addition, and the two used to be confused.

    When the embedder identity changes between the first copy of a repeated turn being
    archived and the second occurrence being evicted, the re-embed branch swaps that one
    copy's vectors and keeps the count where it was, so the newly evicted occurrence was
    never written. With a contradicting turn in between, the chronological block then
    presents the contradiction as the conversation's last word, on the very response the
    eviction triggered.
    """
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

    # The first copy and the contradiction are evicted while the repeat is still live.
    live = list(repeat) + tail
    conversation_archive.archive_turns(THREAD, repeat + middle, live = live)

    # The embedder changes, and only then is the repeat evicted. Handed over ALONE, as four
    # of the five call sites do: they pass the already-fitted list.
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
    # The last word on the identifier is 1, rendered last under the supersedes header.
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

    # The last occurrence and `other` are still in the prompt; the first two are not.
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
    """Retiring a copy without restamping leaves the survivor on a seat that is gone.

    Identical turns at ordinals 0 and 2 with the FIRST rewound away left the survivor on
    0, so a contradiction at 1 rendered after it and the header, which says the higher
    turn number was said later and supersedes, handed the model the superseded value.
    """
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

    # The user rewinds away the FIRST occurrence, and the embedder changes.
    _save_thread(THREAD, mid + list(repeat))
    identity["name"] = "st:model-b"
    conversation_archive.archive_turns(THREAD, mid + list(repeat), live = [])

    found = conversation_archive.recall(THREAD, "ZQXVARA123", top_k = 4)
    assert found is not None
    # Rendered oldest first, so the conversation's last word has to come last.
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
    # Live occurrences are COUNTED, not tested for membership: three seats with one copy
    # still in the prompt owe two writes, not one.
    assert (
        archive._write_budget(
            [["a"], ["a"], ["a"]], [0, 1, 2], archive._live_positions(one_live), group
        )
        == 2
    )


def test_an_out_of_order_eviction_still_numbers_turns_in_conversation_order(conn):
    """Eviction is not strictly oldest-first, so archive time is not conversation order.

    `truncate_oldest_messages` always protects the newest user group, and a pinned
    instruction is held until it stops being pinned, so a LATER turn is routinely archived
    before an EARLIER one. Numbering by arrival recorded the oldest turn as the newest,
    and `format_conversation_recall` says outright that the higher number was said later
    and supersedes the earlier one, so the block asserted the reverse of what happened.
    """
    conversation = (
        _turn("the standing instruction about pelicans", "Understood.")
        + _turn("the middle turn about pelicans", "Noted.")
        + _turn("the final turn about pelicans", "Noted again.")
    )
    _save_thread(THREAD, conversation)

    # Archived out of order: the later turns first, the instruction only afterwards.
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
    """A turn is matched by the whole turn, not by the line it opens with.

    Repeated "continue" prompts, the same question re-asked, a regenerated reply: all of
    them produce two DIFFERENT turns sharing a first message. Matched on the head alone
    both claimed both seats, so both were stamped with the same ordinal, and because each
    then believed it had a second occurrence still to fill, the next compaction wrote both
    of them again. Measured: 4 documents for 2 turns, the recall spending four slots on
    two turns' content, and the older answer quoted under the higher turn number, which
    the header presents to the model as the one that supersedes.
    """
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
    # And re-evicting the same two turns stays free.
    again = [_archive(first, persist = False), _archive(second, persist = False)]

    assert written == [1, 1]
    assert ordinals == [0, 1]
    assert again == [0, 0]
    assert len(store.list_documents(conn, scope)) == 2


def test_an_archive_numbered_by_the_old_allocator_converges_on_the_next_compaction(conn):
    """The migration has to actually run, and it ran on a path that could not be reached.

    The cheap pre-check ahead of the embedding pass fires on exactly the condition the
    write-locked branch does, so re-stamping only in the latter was dead code outside a
    race. Measured before this: an archive forced back to NULL ordinals still read
    NULL, NULL after a full re-compaction, and one forced into archive-time order 1, 0
    stayed 1, 0, with the recall rendering the second turn first under a header saying the
    higher number supersedes.
    """
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

    # An archive written before the column existed.
    conn.execute("UPDATE documents SET archive_ordinal=NULL WHERE scope=?", (scope,))
    conn.commit()
    conversation_archive.archive_turns(THREAD, conversation)
    assert ordinals() == [0, 1]

    # And one numbered in the order the turns happened to be archived.
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
                {
                    "type": "tool-call",
                    "toolCallId": "c1",
                    "toolName": "terminal",
                    "args": {"command": "ls"},
                    "result": "main.py readme.md",
                },
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
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "terminal", "arguments": '{"command": "ls"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "main.py readme.md"},
        {"role": "assistant", "content": "the repo has two files."},
    ]


def test_an_archived_tool_exchange_is_still_reachable_by_a_query(conn):
    """The persisted shape of a tool call is not the wire shape, and both readers assumed it was.

    The store keeps a call as one `tool-call` content PART carrying its result, while the
    request carries three messages: the call, the result, the reply. `_probe_text` renders
    the stored row as call/reply/result and the archived copy as call/result/reply, and
    the branch check matches IN ORDER, so every archived agent turn failed it. Measured:
    the exchange was indexed and then filtered out of every recall, so no query could
    return it. That is archived content the model can never get back, on every tool-using
    turn.
    """
    tool_turn = _persist_agent_thread()
    conversation_archive.archive_turns(THREAD, tool_turn)

    found = conversation_archive.recall(THREAD, "terminal ls repo files", top_k = 4)

    assert found is not None
    assert any("terminal" in source["text"] for source in found[1])


def test_a_tool_exchange_is_numbered_where_the_conversation_put_it(conn):
    """`group_turns` splits on tool_calls, which a persisted row never carries.

    So the whole exchange folded into the preceding user group, the evicted tool group
    found no seat of its own, and it fell back to MAX + 1 -- the same number its own
    opening question takes from the transcript. Measured: two documents at ordinal 1, with
    created_at breaking the tie in favour of the ANSWER, under a header telling the model
    the later turn supersedes.
    """
    tool_turn = _persist_agent_thread()
    # The order eviction really produces: the oldest turn, then the tool groups, then the
    # user turn that opened them once it stops being the newest.
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
    """The refill has to keep RETRIEVAL rank, which the chronological sort throws away.

    Every candidate in a tied archive carries the same score, and the score a source
    carries is rounded for display on top of that, so sorting the refill by score alone
    left the list in the order it arrived: chronological. The refill then spent its slots
    on the oldest turns, and adding the anchor that exists to rescue a thin message made
    the recall worse than not adding it. Measured on eight revisions at top_k 4: the single
    query returned the newest, the same query plus an anchor did not.
    """
    values = _revisions(8, distractors = 0)

    alone = conversation_archive.recall(THREAD, f"{VARIABLE}", top_k = 4)
    merged = conversation_archive.recall(THREAD, f"{VARIABLE}", top_k = 4, extra_queries = ["timeout"])

    assert alone is not None and merged is not None
    assert values[-1] in alone[0]
    assert values[-1] in merged[0], "the anchor cost the newest revision its slot"


def test_an_orphan_user_row_does_not_lend_its_seat_to_a_later_turn(conn):
    """A position SHORTER than the turn may only match the trailing one.

    `zip` stops at the shorter side, so a persisted turn missing its reply prefix-matched
    anywhere in the transcript. A thread carrying an orphan user row -- an assistant reply
    deleted from the thread, or a reload before the reply was appended -- therefore handed
    the later, answered turn two seats, and the next compaction wrote a second copy of it.
    Measured: seats [0, 1] where only [1] is real, two documents with one sha, and the
    recall quoting the same turn twice.
    """
    from storage import studio_db

    studio_db.upsert_chat_thread(
        {"id": THREAD, "title": "t", "modelType": "base", "modelId": "local-model", "createdAt": 1}
    )
    rows = [
        ("user", "set ZQXVARA123 to 1"),  # orphan: its reply is gone
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
    """The stored rows are a tree, and reading them as a list numbers an abandoned sibling.

    Retry leaves the replaced reply in place, so a flat read drops it between two live
    turns and the grouper glues it onto whichever turn precedes it. The regenerated turn
    then matches no position and takes MAX + 1, which the cumulative archive has already
    pushed past every live turn: measured, a regenerated turn 2 came back numbered 5 out of
    4 live turns, colliding with live turn 3 under the header that says the higher number
    supersedes.
    """
    from storage import studio_db

    studio_db.upsert_chat_thread(
        {"id": THREAD, "title": "t", "modelType": "base", "modelId": "local-model", "createdAt": 1}
    )
    rows = [
        ("m0", None, "user", "turn 1 about ZQXVARA123"),
        ("m1", "m0", "assistant", "answer 1"),
        ("m2", "m1", "user", "turn 2 about ZQXVARA123"),
        ("m3", "m2", "assistant", "answer 2 attempt one"),  # abandoned sibling
        ("m4", "m2", "assistant", "answer 2 attempt two"),  # the live reply
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
    """A repeat that is rewound away leaves more copies than occurrences.

    Both copies are byte-identical, so the branch filter validates each against the single
    surviving occurrence and `recall` dedups on chunk id, which differs. Measured: a recall
    slot went on quoting one turn twice, and the surplus kept an ordinal that a genuinely
    later turn had since taken.
    """
    first = _turn("set ZQXVARA123 to 1", "ok")
    second = _turn("set ZQXVARA123 to 2", "ok")
    repeat = _turn("set ZQXVARA123 to 1", "ok")
    for group in (first, second, repeat):
        _archive(group)
    scope = store.conversation_archive_scope(THREAD)
    assert len(store.list_documents(conn, scope)) == 3

    # Rewind past the repeat, through the same sync the PUT route uses.
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
    """A bare number needs length to be a name, which is the bar the capitals rule had.

    Treating any digit-bearing token as an identifier made "answer in 2 sentences" filter
    the archive on "2". Measured on an archive whose filler mentions small numbers in
    ordinary prose, at top_k 1, the focused pass returned the staging-environments turn
    where both the previous build and the rollback knob returned the billing turn.
    """
    assert store.conversation_match_queries("answer in 2 sentences") == [
        '"answer" OR "2" OR "sentences"'
    ]
    assert store.conversation_match_queries("which python, 3.11 or 3.12") == [
        '"python" OR "3" OR "11" OR "12"'
    ]
    # A name is still a name, at any length, and a long number still qualifies.
    assert store.conversation_match_queries("what about v2 of the plan")[0] == '"v2"'
    assert store.conversation_match_queries("we talked about 2024 revenue")[0] == '"2024"'
    assert store.conversation_match_queries("What is the current value of 9134?")[0] == '"9134"'


def test_a_re_embed_after_a_rewind_retires_the_surplus_copy_too(conn, monkeypatch):
    """The re-embed path never reaches the branch where surplus copies are retired.

    A repeated turn archived twice and then rewound leaves one copy more than the
    conversation holds. Replacing one copy's vectors under a new embedder writes a fresh
    document and returns, so the surplus is never looked at: measured, three documents for
    two turns after the rewind, with the recall quoting the repeated turn twice and the
    surplus still holding a position a later turn had taken.
    """
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

    # Rewind past the repeat, and the embedder changes underneath the thread.
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
    """A persisted assistant row holds the whole turn, in generation order.

    Text BEFORE the first `tool-call` part is what the model said on its way to calling,
    and the live wire form carries that on the call message itself. Emitting it after the
    synthesized results instead gave the archived copy three messages reading
    call/result/text against the live two reading call+text/result, so `_occurrences`
    matched nothing and the turn took a fallback ordinal. Text AFTER the calls is the
    reply that followed the result and still belongs last, which is why this splits by
    POSITION and not by part type.
    """
    call = {
        "type": "tool-call",
        "toolCallId": "c1",
        "toolName": "terminal",
        "args": {"command": "ls"},
        "result": "main.py readme.md",
    }
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
    """`Set key Foo` and `Set key FOO` hash differently, so each keeps its own document.

    Folding case when matching the transcript handed BOTH seats to BOTH of them, and a
    turn that believes it has two occurrences to fill is written twice at the next
    compaction: four documents for two turns, each stamped at both ordinals, and no way
    to tell which spelling was said later.
    """
    lower = _turn("set key Foo", "done")
    upper = _turn("set key FOO", "done")
    _save_thread(THREAD, lower + upper)

    positions = conversation_archive._transcript_positions(THREAD)
    assert conversation_archive._occurrences(positions, lower) == [0]
    assert conversation_archive._occurrences(positions, upper) == [1]


def test_the_deleted_conversation_goes_even_when_its_id_comes_back(conn):
    """Sparing a recreated id spared the DELETED conversation along with it.

    The scope is keyed by thread id alone, so the pre-delete turns stayed under a live id
    with nothing left to sweep them: the endpoint reported success while the conversation
    the user asked to delete remained recallable in the new chat. Cutting at the instant
    the delete was accepted takes the old turns and leaves the new ones.
    """
    from datetime import datetime, timezone

    from routes import chat_history

    thread_id = "recreated-with-cutoff"
    old_turns = _turn("what is the code", "the code is 5150")
    _save_thread(thread_id, old_turns, append = True)
    assert conversation_archive.archive_turns(thread_id, old_turns) == 1

    cutoff = datetime.now(timezone.utc).isoformat()

    # The stale tab recreates the id and its generation archives a turn of its own.
    fresh = _turn("what is the new code", "the new code is 8080")
    _save_thread(thread_id, old_turns + fresh, append = True)
    assert conversation_archive.archive_turns(thread_id, fresh) == 1

    chat_history._remove_conversation_archives([thread_id], cutoff = cutoff)

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
        # The user rewinds to m1 and continues on branch B, so B's rows are the newest in
        # the thread. Then the branch picker takes them back to A.
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
    """The newest stored row is not the branch the request is on.

    Switching to a sibling branch, continuing there, and then switching BACK leaves the
    abandoned branch holding the greatest created_at. Seeding the ancestry walk from the
    last stored row therefore read the branch the user had left: turns being evicted from
    the request's own branch matched no position, took MAX + 1 over a cumulative archive
    the other branch had already pushed up, and `format_conversation_recall` presented an
    older statement as the one that supersedes.
    """
    live = _branch_switch_thread()

    positions = conversation_archive._transcript_positions(THREAD, branch = live)

    assert len(positions) == 3, positions
    assert conversation_archive._occurrences(positions, live[2:4]) == [1]
    assert conversation_archive._occurrences(positions, live[4:6]) == [2]


def test_the_branch_seed_falls_back_when_nothing_matches(conn):
    """No branch, or a branch that matches nothing, has to leave today's behaviour alone.

    An API-only caller passes none, and a zero-match seed must not collapse `positions`:
    an empty chain empties every seat and sends every turn to MAX + 1, which is strictly
    worse than reading the newest row.
    """
    _branch_switch_thread()

    seeded = conversation_archive._transcript_positions(THREAD)
    unmatched = conversation_archive._transcript_positions(
        THREAD, branch = [{"role": "user", "content": "nothing in this thread says this"}]
    )

    # Branch B is the newest stored row, so both fall back to it: two turns, not three.
    assert len(seeded) == 2, seeded
    assert unmatched == seeded


def test_two_sequential_tool_rounds_replay_as_two_exchanges():
    """One persisted row can hold a whole agent turn, rounds and all.

    `chat-adapter.ts` flushes the pending calls whenever text arrives, so a row reading
    call, text, call goes out as call/result, then text riding on the second call
    message, then its result. Collecting every call into one message and appending every
    result after it rebuilt a different order: `group_turns` glued exchanges that were
    separate on the wire, and the later calls matched no position and took an invented
    ordinal.
    """

    def _call(index, command, result):
        return {
            "type": "tool-call",
            "toolCallId": f"c{index}",
            "toolName": "terminal",
            "args": {"command": command},
            "result": result,
        }

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
    # The text before the second call rides ON it, exactly as the flush builds it.
    second = conversation_archive._normalise(conversation_archive._probe_text(wire[2]))
    assert "pytest" in second and "Now the tests." in second
    assert (
        conversation_archive._normalise(conversation_archive._probe_text(wire[4])) == "All green."
    )


def test_an_in_flight_tool_group_does_not_take_the_live_user_turn_s_number(conn):
    """Seats count TRANSCRIPT positions; the archive counter counts what was archived.

    The newest user group is protected from eviction, so during a long tool loop it sits
    in the transcript and not in the archive. A tool group evicted before its assistant
    row is persisted matches no seat and took the archive's next number, which the user
    turn later claimed from the transcript: both documents landed on the same ordinal,
    and since created_at breaks the tie the tool answer rendered ahead of the prompt that
    caused it, under the header saying a higher number was said later.
    """
    user_turn = _turn("run the deploy", "deploying now")
    _save_thread(THREAD, user_turn, append = True)

    in_flight = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "terminal", "arguments": '{"command": "deploy"}'}}
            ],
        },
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
    """Lowercasing the comparison made a case-only correction invisible.

    `Foo` corrected to `foo` is a real edit, and the pre-edit copy stayed eligible: a
    later search could answer with the spelling the user had just fixed. Same for a block
    re-indented and nothing else, which is the ordinary way YAML and Python get corrected.
    """
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
    """The guard on the tighter comparison, which `rstrip()` would have broken.

    `render_turn` strips the whole message, so keeping a probe's LEADING whitespace makes
    a live turn beginning with a space or a newline start its run at a non-zero offset and
    `_document_matches_one_run` retires it. Pasted code is the common shape here, so the
    loss would land on exactly the turns worth recalling.
    """
    for content in ("   hello there", "\n  def f():\n    pass"):
        turn = [{"role": "user", "content": content}, {"role": "assistant", "content": "ok"}]
        rendered = conversation_archive.render_turn(turn)
        text = rendered[1] if isinstance(rendered, tuple) else rendered
        live = conversation_archive.branch_message_texts(turn)

        assert conversation_archive._document_matches_one_run([{"text": text}], live, 2) is True


def test_a_tool_result_cut_exactly_on_a_line_stays_on_its_branch(conn):
    """`render_turn`'s cut can land on a newline, and the marker is then its own line.

    Stripped, that line is empty and was dropped, taking the truncation flag with it. The
    last real probe was read as complete, `_document_matches_one_run` demanded the live
    message end where the probe did, and an unedited over-cap tool result was retired:
    measured on a 900-line result, no query could return it.
    """
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
    """Only an ABSENT result is absent.

    `serializeToolResultPart` skips exactly `undefined` and `null`, and emits a `tool`
    message for everything else: a `{"result": ""}` sentinel for an empty string, since
    the ChatMessage validator rejects an empty `tool` content, and JSON for containers.
    Treating "" / {} / [] as nothing dropped a message the wire carries, so the
    reconstructed run was shorter than the archived one and branch validation could filter
    the turn out of every recall.
    """

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
    # Byte for byte what `JSON.stringify({ result: "" })` produces: no space after the
    # colon, which `json.dumps` adds, and every comparison downstream is exact.
    assert emitted[""] == ['{"result":""}']
    assert emitted["ok"] == ["ok"]

    for container, expected in (({}, "{}"), ([], "[]"), ({"a": 1}, '{"a":1}')):
        wire = conversation_archive._as_wire(_row(container))
        assert [message["content"] for message in wire if message["role"] == "tool"] == [expected]

    # A result that is genuinely not there stays not there, matching the serializer.
    for row in (_row(None), _row(None, present = False)):
        assert [
            message for message in conversation_archive._as_wire(row) if message["role"] == "tool"
        ] == []

    # And the reconstructed row matches the document archived from the request, which is
    # the point: emitting the message and still failing the match only looks fixed.
    wire = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "terminal", "arguments": '{"command":"true"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"result":""}'},
    ]
    rendered = conversation_archive.render_turn(wire)
    document = rendered[1] if isinstance(rendered, tuple) else rendered
    reconstructed = conversation_archive.branch_message_texts(
        conversation_archive._as_wire(_row(""))
    )

    assert conversation_archive._on_live_branch(document, reconstructed) is True


def test_a_bare_identifier_query_also_reaches_past_the_cap(conn):
    """A query that is ONLY an identifier shapes to ONE expression.

    Its focused and permissive spellings coincide, and the single-expression path returned
    the plain one-ended fetch, so the query most likely to tie on the IDF floor got the
    oldest rows and the newest assignment was never a candidate. The two-expression case
    was already covered, which is why this went unnoticed.
    """
    count = conversation_archive._BRANCH_FILTER_MAX_CANDIDATES + 40
    for index in range(count - 1):
        _archive(_turn(f"note {index:03d} about ZQXVARA123", "noted"))
    _archive(_turn("set ZQXVARA123 to 9999", "done"))

    assert len(store.conversation_match_queries("ZQXVARA123")) == 1
    found = conversation_archive.recall(THREAD, "ZQXVARA123", top_k = 4)

    assert found is not None
    assert "9999" in found[0]
    # The oldest end stays reachable, the invariant the ends-first ordering exists for.
    assert "note 000" in conversation_archive.recall(THREAD, "ZQXVARA123", top_k = 256)[0]


def test_a_persisted_tool_call_followed_by_its_answer_stays_on_its_branch(conn):
    """The ordinary agent turn: call, result, then the model's final answer.

    Bucketing the whole message as call, text, result put the answer in the middle, so
    `_scan_probes` advanced past it to find the result and could not find it again. The
    document rendered from the request said call, result, answer, and an unchanged evicted
    tool exchange was classified off-branch: measured end to end, recall came back with
    the user's question alone and the document holding the answer was filtered out.
    """
    stored = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool-call",
                    "toolCallId": "c1",
                    "toolName": "terminal",
                    "args": {"command": "cat deploy.yml"},
                    "result": "token ZQX-5150",
                },
                {"type": "text", "text": "The deploy token is ZQX-5150."},
            ],
        }
    ]
    wire = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {"name": "terminal", "arguments": '{"command": "cat deploy.yml"}'},
                }
            ],
        },
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
    # The wire-shaped branch, which every live caller supplies, is unchanged.
    assert (
        conversation_archive._document_matches_one_run(
            [{"text": text}], conversation_archive.branch_message_texts(wire), 3
        )
        is True
    )


def test_a_provider_side_builtin_is_replayed_the_way_the_frontend_replays_it():
    """The frontend drops a builtin card from the history it sends.

    A `web_search` card with the server marker and no native part is omitted entirely,
    call and result, and one WITH a native part replays as a call carrying no `tool`
    message, since its result travels in the provider's own part. Reconstructing either as
    an ordinary local call inserted an exchange the request never carried, so the turn
    matched nothing and took a fallback ordinal.
    """
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
    # A USER function that merely shares the name is untouched: the name never decides.
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
    """`python` and `terminal` results are wrapped on every call.

    The replay adapter sends `result.text` alone rather than feeding the model a session
    id and file metadata, so serialising the whole wrapper reconstructed a tool message
    that can never equal the archived one.
    """

    def _row(tool_name, result):
        return [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool-call",
                        "toolCallId": "c1",
                        "toolName": tool_name,
                        "args": {"command": "ls"},
                        "result": result,
                    }
                ],
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
    assert _tool_content(_row("python", mcp_image)) == ["chart rendered"]
    # An empty wrapper text still takes the sentinel, as the adapter does.
    assert _tool_content(_row("terminal", {**sandbox, "text": ""})) == ['{"result":""}']
    # Someone else's result with text and a session is NOT unwrapped, or its other fields
    # would be dropped. The name gates it, as on the frontend.
    assert _tool_content(_row("lookup", sandbox)) == [
        '{"text":"token ZQX-5150","images":[],"sessionId":"project-7",'
        '"files":[{"name":"out.csv","size":12}]}'
    ]


def test_the_branch_seed_scores_an_in_order_run_not_a_set(conn):
    """Sets lose repetition and ordering, and leaves are tried newest-first.

    A newer abandoned sibling holding the same distinct texts scored identically to the
    request's own branch and won the tie, so a turn was handed the seat belonging to its
    earlier twin and two distinct turns claimed one seat. A multiset would fix the repeat
    case and not the reordered one.
    """
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
        # The abandoned sibling is NEWER and holds the same distinct texts, once each.
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
    # Both repeats keep their own seat instead of collapsing onto the first.
    assert conversation_archive._occurrences(positions, branch[2:4]) == [1, 2]


def test_a_batch_mixing_a_search_with_an_ordinary_tool_keeps_its_transcript_span(conn):
    """`archive_messages` bounds the branch check, so it has to be the TRANSCRIPT span.

    An assistant batch that called `search_conversation` alongside an ordinary tool has
    its retrieval call and result stripped before the document is written, so the archived
    copy is three messages where the live turn is four. Bounded by the shorter figure, the
    perfectly valid ordinary-tool exchange and the answer that followed were rejected as
    off-branch and could never be recalled.
    """
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
    """Unsloth prepends chat and project instructions to every outbound request.

    That synthetic `system` message is not part of the stored chain, so a strict cursor
    stalled on it: no leaf could advance past `wanted[0]`, every one scored zero, and the
    seed fell back to the newest row -- the abandoned branch it exists to avoid. Measured
    with an ordinary system prompt in front, the seed picked branch B over the request's
    own A on a thread it had just got right.
    """
    live = _branch_switch_thread()
    with_system = [{"role": "system", "content": "You are a helpful assistant."}] + live

    plain = conversation_archive._transcript_positions(THREAD, branch = live)
    seeded = conversation_archive._transcript_positions(THREAD, branch = with_system)

    assert len(plain) == 3, plain
    assert seeded == plain


def test_the_branch_seed_reaches_a_leaf_older_than_the_retry_pile(conn):
    """The branch a user goes BACK to is older than every retry made since.

    Capping the candidate leaves at a small number therefore excluded the one branch this
    exists to find: past that many retries the request's own branch could not be selected
    however well it matched, and the walk fell back to the newest abandoned leaf.
    """
    from storage import studio_db

    live = _branch_switch_thread()
    # A pile of newer abandoned retries, each its own leaf off the very first reply.
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
    """A cancelled card is not a call with a missing result, it is not a call.

    `chat-adapter.ts` drops the whole thing (`if (!toolResult &&
    !canReplayToolCallWithoutRoleTool(part)) continue`), so keeping the call and merely
    omitting its `tool` message reconstructs an assistant `tool_calls` message the request
    never carried. That shifts every group after it, which is what decides ordinals and
    whether an archived turn passes the live-branch check.
    """
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
    """`shouldFlushCompletedLocalToolPair` makes each completed local pair its own group.

    Batched into one parallel call message, `group_turns` sees one exchange where the
    request sent two, so the second call matches no position and takes an invented
    ordinal.
    """
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
    """The top-up path writes the same document, so it needs the same span.

    `_archivable` strips a retrieval call and its result, so a group of three can cover
    four transcript messages. The primary write records the transcript figure; the
    re-embed top-up recorded `len(group)`, and a copy written short is bounded by the
    smaller run in `_document_matches_one_run` and filtered out of every recall.
    """
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
    """ "Let me check" ahead of a tool call is the ordinary agent turn, not an edge case.

    `_probe_text` offers BOTH JSON spellings of the stored arguments, and that second
    spelling lands between the arguments and whatever followed them, so the live render
    (name/args/text) is no longer contiguous inside the stored one (name/args/args/text).
    The turn matched no transcript position and took a fallback ordinal past the whole
    transcript, where the recall header presents it as superseding genuinely later
    instructions.
    """
    user = {"role": "user", "content": "what files are here"}
    row = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me check"},
            {
                "type": "tool-call",
                "toolCallId": "c1",
                "toolName": "terminal",
                "args": {"command": "ls"},
                "result": "main.py",
            },
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
        {
            "role": "assistant",
            "content": "Let me check",
            "tool_calls": [
                {"id": "c1", "function": {"name": "terminal", "arguments": '{"command": "ls"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "main.py"},
    ]

    assert conversation_archive._occurrences(positions, live) == [1]


def test_the_same_text_over_a_longer_span_widens_the_stored_window(conn):
    """One document, two spans: the window has to fit the LONGER of them.

    The digest is the rendered text, and `_archivable` strips a retrieval call and its
    result, so a three-message tool exchange and a four-message batch containing that same
    exchange render identically. Archived shortest first, the second was skipped as a
    duplicate and inherited a window of three, which `_document_matches_one_run` then used
    to bound a four-message live run: the turn was rejected as off-branch and no query
    could return it.
    """
    short = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c2", "function": {"name": "terminal", "arguments": '{"command":"ls"}'}}
            ],
        },
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

    # And the longer turn now validates against its own four-message run.
    text = conversation_archive.render_turn(conversation_archive._archivable(long))
    live = conversation_archive.branch_message_texts(long)
    assert conversation_archive._document_matches_one_run([{"text": text}], live, 3) is False
    assert conversation_archive._document_matches_one_run([{"text": text}], live, 4) is True


def test_a_reasoning_turn_is_reconstructed_without_its_thinking():
    """Reasoning is not content, and the wire form never carries it as content.

    The serializer puts it in `reasoning_content` or drops it, so forwarding the stored
    part list rendered a reasoning model's thinking inline where the request sends only
    the answer. That turn matched no transcript seat and took an ordinal past genuinely
    later turns, under a header that calls the higher number the latest word.
    """
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
    """`JSON.stringify` leaves non-ASCII alone; `json.dumps` escapes it by default.

    The reconstructed `tool` message then reads `Montr\\u00e9al` where the archived wire
    text carries the character itself, and every comparison downstream is exact, so a
    multilingual tool exchange loses its seat and is filtered out of recall.
    """
    assert (
        conversation_archive._tool_result_content({"ville": "Montréal"}, "terminal")
        == '{"ville":"Montréal"}'
    )


def test_a_long_tool_exchange_stays_on_branch_across_a_chunk_boundary():
    """A chunk's overlap can carry a whole short message into the next chunk.

    `CHUNK_OVERLAP` repeats the previous chunk's tail, and where that tail begins just
    after a short line -- an assistant tool call sitting in front of a long tool result --
    the repeat starts in an EARLIER message than the one the previous chunk finished in.
    Resuming the scan strictly at the finishing message could never match it, so an
    unedited document was retired as off-branch and that turn became unsearchable. Swept
    over question lengths, 7 of 89 failed before the fix.
    """
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
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "grep", "arguments": '{"pattern":"foo"}'}}
                ],
            },
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
    """The window has to grow on the LOCKED duplicate path as well.

    Both turns can arrive in one compaction: the pre-check runs before either is written,
    so it clears both, and the shorter one is written first. The longer one then meets the
    re-check under the write lock, which rolls back and moves on without touching the
    span. The stored window stays at three, the four-message occurrence is bounded by it,
    and the turn is unsearchable -- the same failure as the unlocked path, one lock down.
    """
    short = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c2", "function": {"name": "terminal", "arguments": '{"command":"ls"}'}}
            ],
        },
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

    # One call, both turns, shortest first.
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
    """A rewind, or a bigger window, can put an evicted occurrence back in the prompt.

    The already-archived path then re-stamped a copy per transcript SEAT while the
    live-aware budget had dropped to one, so two byte-identical documents survived. Both
    pass the branch filter and `recall` dedups on chunk id, which differs, so a recall slot
    went on text the model could already read: measured, three passages of which two were
    distinct.
    """
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

    # Both occurrences evicted: two copies, one per seat.
    conversation_archive.archive_turns(THREAD, conversation[:6], live = tail)
    assert len(store.documents_by_hash(conn, scope, digest)) == 2

    # The newer occurrence is live again, so only one of them is still evicted.
    conversation_archive.archive_turns(THREAD, repeat, live = list(repeat) + tail)

    assert len(store.documents_by_hash(conn, scope, digest)) == 1
    found = conversation_archive.recall(THREAD, "ZQXVARA123", top_k = 4)
    assert found is not None
    texts = [source["text"] for source in found[1]]
    assert len(texts) == len(set(texts)), texts
