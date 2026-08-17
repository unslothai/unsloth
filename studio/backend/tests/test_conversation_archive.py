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

    ``append`` matters: a real thread grows, and replacing its rows on every archive call
    would leave only the newest turn saved, so the live-branch filter would reject
    everything archived earlier. Tests that rewind a thread pass append=False to replace.
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

    Archiving now requires the thread to exist in studio.db, so the default persists a
    matching transcript. That is not scaffolding: only a persisted thread can ever be
    deleted, and an archive no delete can reach is exactly the temporary-chat leak the
    rule exists to prevent. ``persist=False`` exercises the refusal.
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

    An archive holding only the latest compaction's evictions measured 0.058 against
    0.450 for this cumulative one, so this is the property the feature rests on.
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

    Thread-attached documents are injected in full on every request. If the archive lived
    in the thread scope, compaction would re-inject the entire evicted history each turn
    and undo itself.
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

    Measured on a real 30-turn document walkthrough where every turn shared the same
    wrapper text: the chunk holding the needle ranked 3rd lexically at any k, was never
    returned by dense retrieval, and RRF fusion pushed it to 16th. Hybrid alone lost the
    answer outright, so this pins the ordering rather than the plumbing.
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

    The archive is append-only and still holds what the abandoned continuation produced,
    so without a branch check the right question pulls back a turn that, on this branch,
    never happened. Verified against the live build before the check existed.
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
    # Recall has no relevance floor, so a query for the abandoned turn can still come back
    # with the surviving one. The property that matters is that the rolled-back content
    # itself is never returned.
    assert abandoned is None or "GONEAWAY-2222" not in abandoned[0]
    assert "GONEAWAY-2222" not in (survived[0] or "")


def test_recall_filters_to_the_ACTIVE_branch_not_the_whole_stored_thread(conn):
    """Retry keeps the replaced response as a sibling, and siblings are stored rows.

    Rewinding by editing prunes the abandoned rows, so filtering against the whole thread
    is enough there. Retry and regenerate do not: both responses are kept on purpose,
    which is what the branch arrows navigate between, and the thread-wide blob therefore
    still contains a response the user replaced. Only the branch the request was sent on
    can tell the two apart.
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

    # Told which branch the request is on, it is rejected, and the live answer still comes
    # back rather than the filter simply refusing everything.
    on_branch = conversation_archive.recall("retry-thread", "SIBLING-3333", branch_messages = live)
    assert on_branch is None or "SIBLING-3333" not in on_branch[0]
    survived = conversation_archive.recall("retry-thread", "KEEPME-1111", branch_messages = live)
    assert survived is not None and "KEEPME-1111" in survived[0]


def test_the_reply_that_FOLLOWS_a_forced_recall_is_still_archived(conn):
    """group_turns keeps a tool call, its result and the reply after it in one group.

    Rejecting that whole group because our own injection is in it threw away the model's
    actual answer: the question was archived from its own group and the answer was not,
    so a later search could find what was asked and never what was said -- and this
    happens on compaction turns specifically, which is where the feature is used.
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

    A venv change is enough (the common macOS case). A delete that quietly does nothing
    then leaves a deleted conversation's turns on disk, ready to answer again the day the
    extension loads once more.
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

    Validating them independently lets a turn be reassembled out of parts that never sat
    together: the head matching the question and its current answer, the tail matching
    some later message that repeats the passage the edit removed.
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

    A turn is two or three messages however long it is, so bounding the run by the line
    count let the tail of a long answer be satisfied by a message well outside the turn.
    Here the edit removes the end of the answer and the very next message repeats it,
    which is the shape a short correction takes in an ordinary chat.

    The chunks are shaped the way the chunker really produces them: only the rendered
    text carries the role labels, so a continuation chunk starts mid-message.
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


def test_a_pasted_transcript_cannot_widen_a_turns_run(conn):
    """Counting role labels counts lines the USER wrote, not just the renderer's.

    A pasted chat log carries lines that look exactly like the ones `render_turn` writes,
    and every one of them widened the run by a message -- enough for the message after an
    edited turn to supply the passage the edit removed. The turn's real size is recorded
    when it is archived.
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

    The slow part is the embedding pass that sits between the check and the insert, and
    `(scope, sha256)` carries a plain index rather than a unique one, so both wrote. The
    turn was then stored twice and its copies took two of the few recall slots.
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

    Against one flattened transcript, a missing line is satisfied by any later message
    that repeats the words. Short answers repeat constantly, so an archived question and
    the answer it got could survive that answer being edited away.
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

    Every probe still occurs somewhere in the transcript, so the pre-edit ordering stays
    eligible and would be served back as what happened.
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
    result after it, so the transcript builders have to lay a turn out the same way, in
    both the request shape (`tool_calls`) and the stored shape (`tool-call` parts).
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

    Checking the branch per chunk means editing the second half of a long answer retires
    only the chunks carrying the edit, and an untouched earlier chunk of the same retired
    turn stays eligible on its own. The unit that was archived is the turn.
    """
    # The head is several chunks long on its own, so the FIRST chunk lies entirely
    # inside the part that is never edited and passes a per-chunk check unchanged.
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

    # The branch now carries the same turn with its TAIL rewritten. The head, which is
    # the chunk holding the marker, is untouched.
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

    That result is ONE appended string containing many newlines, so only its first line
    carries the "tool result:" label. Every continuation line took the full-string path,
    including the last one, which is the only line the marker is on, and nothing in a
    real transcript ends in that marker: every archived tool turn over the cap was
    rejected as rolled back.
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

    Rewinding or retrying a continuation that had already been compacted leaves enough
    stale turns to fill any fixed candidate window. Filtering after a single fetch then
    rejects the whole page while the live match sitting just below it is never looked at,
    so recall reports nothing although the answer is in the archive.
    """
    # The live answer mentions the marker once, in a turn that is mostly other words.
    live = _turn(
        "where is the marker",
        "the marker is WIDEN-5150 and the rest of this answer is about unrelated matters "
        "such as scheduling, packaging, release notes and the weather in three cities",
    )
    # The abandoned branch repeats it, so every one of these outranks the live turn
    # lexically and fills any fixed candidate window ahead of it.
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
        # Stored as siblings: Retry keeps them, so they are on the thread but not the branch.
        _save_thread("wall-thread", turn, append = True)
        conversation_archive.archive_turns("wall-thread", turn)

    found = conversation_archive.recall("wall-thread", "WIDEN-5150", branch_messages = live)

    assert found is not None
    assert "the marker is WIDEN-5150" in found[0]
    assert "discarded" not in found[0]


def test_the_branch_transcript_carries_request_shaped_tool_calls(conn):
    """A tool turn's arguments live in `tool_calls`, not in content, on the wire.

    render_turn indexes those arguments, so a branch blob built from content alone would
    miss every archived tool turn and filter out the whole exchange as rolled back.
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

    An incognito chat is never written to studio.db and is documented to vanish on
    reload, but the frontend still sends its thread_id and the request carries no
    incognito flag. Archiving it would persist the one conversation the user asked not
    to keep, in a scope no deletion flow could reach.
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
    # Same predicate the fit asks before holding a recall reserve back, so a chat that
    # is never archived never pays for the room to recall into either.
    assert conversation_archive.can_archive("temporary-thread") is False


def test_a_thread_deleted_mid_ingest_does_not_leave_its_turns_behind(conn, monkeypatch):
    """Deleting a chat while it is compacting must not resurrect the archive.

    Deletion cancels the generation, but cancellation is cooperative and the embedding
    pass between the liveness check and the commit does not observe it. A delete that
    sweeps the scope in that window is undone by the commit, and the rows it puts back
    are in a scope no later delete can reach, since the thread is gone.
    """
    from storage import studio_db

    turns = _turn("what is the code", "the code is DELETED-9999")
    _save_thread("doomed-thread", turns, append = True)

    original = conversation_archive.embeddings.encode_with_identity

    def delete_the_thread_mid_ingest(*args, **kwargs):
        # Exactly the interleaving that matters: rows first, archive sweep last, both
        # completing before this ingest commits.
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

    An empty transcript is absence of evidence, not evidence that the turns are gone, so
    the branch check must not silently disable recall for an archive that already exists.
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

    Matching on the first line that is still present kept serving the old answer after
    the user edited it: the unchanged user line vouched for the whole turn.
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

    # Restored by re-setting, not by monkeypatch.undo(): fixtures requesting monkeypatch
    # share this function's instance, so undo() would also revert stub_embeddings and the
    # retry would fail for an unrelated reason.
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

    render_turn archives a tool turn as "assistant called X: args" and "tool result: ..."
    while assistant-ui persists the call as a structured tool-call content part, which
    the transcript flattener used to drop. With every archived line required to appear in
    that transcript, no archived tool turn could ever match.
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

    Rewriting the tail of a long answer left the archived copy matching on its first 160
    characters, so the stale text stayed eligible for recall.
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
