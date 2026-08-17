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

    transcript = conversation_archive.branch_transcript(group)

    assert conversation_archive._on_live_branch(text, transcript) is True
    # And an edit to the part that WAS archived still retires the copy.
    edited = conversation_archive.branch_transcript(
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
