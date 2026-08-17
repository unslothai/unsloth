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
