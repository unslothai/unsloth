# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The archive behind rolling-context compaction: what it keeps, and what it must not touch."""

import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag import conversation_archive, retrieval, store  # noqa: E402
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
    """The same turns are evicted again on every later request, so repeats must be free."""
    turn = _turn("what is a duck", "a waterfowl")
    first = _archive(turn)
    second = _archive(turn)

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
    ):
        if mode != "lexical":
            raise RuntimeError("no embedding backend available")
        return real_hybrid(conn_, scope, query, k = k, model_name = model_name, mode = mode)

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
