# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.rag import conversation_archive, store  # noqa: E402

THREAD = "thread-session"


def _save_thread(thread_id, messages):
    from storage import studio_db

    studio_db.upsert_chat_thread(
        {"id": thread_id, "title": "t", "modelType": "base", "modelId": "m", "createdAt": 1}
    )
    existing = len(studio_db.list_chat_messages(thread_id) or [])
    for index, message in enumerate(messages):
        studio_db.upsert_chat_message(
            {
                "id": f"{thread_id}-{existing + index}",
                "threadId": thread_id,
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
                "createdAt": existing + index + 2,
            }
        )


def _archive(messages):
    _save_thread(THREAD, messages)
    return conversation_archive.archive_turns(THREAD, messages)


@pytest.fixture
def conn(rag_home, rag_conn, stub_embeddings):
    return rag_conn


class _AllStarts:
    """An index that names every message, which is the unpruned search."""

    def __init__(self, texts):
        self.texts = texts

    def containing(self, probe):
        return list(range(len(self.texts)))


def _random_case(rng):
    words = ["alpha", "beta", "gamma", "delta", "run", "fix", "TICKET-7", "the", "cache"]
    pool = [" ".join(rng.choice(words) for _ in range(rng.randint(1, 4))) for _ in range(6)]
    messages = [
        {"role": rng.choice(["user", "assistant"]), "content": rng.choice(pool)}
        for _ in range(rng.randint(1, 6))
    ]
    transcript = [rng.choice(pool) for _ in range(rng.randint(1, 30))]
    if rng.random() < 0.5:
        # Plant the turn so a real match is possible, sometimes edited.
        at = rng.randint(0, len(transcript))
        planted = [m["content"] for m in messages]
        if rng.random() < 0.3:
            planted[0] = planted[0] + " edited"
        transcript[at:at] = planted
    lines = conversation_archive.render_turn(messages).splitlines()
    if len(lines) > 1 and rng.random() < 0.6:
        # Chunked like a long turn: consecutive slices that may overlap by a line.
        cut = rng.randint(1, len(lines) - 1)
        back = rng.randint(0, 1)
        rows = [{"text": "\n".join(lines[:cut])}, {"text": "\n".join(lines[cut - back :])}]
    else:
        rows = [{"text": "\n".join(lines)}]
    count = rng.choice([None, len(messages), len(messages) + 1])
    return rows, transcript, count


def test_pruned_branch_match_equals_the_exhaustive_search():
    rng = random.Random(1234)
    matched = 0
    for _ in range(20000):
        rows, transcript, count = _random_case(rng)
        exhaustive = conversation_archive._document_matches_one_run(
            rows, transcript, count, _AllStarts(transcript)
        )
        index = conversation_archive._TranscriptIndex(transcript)
        assert (
            conversation_archive._document_matches_one_run(rows, transcript, count, index)
            == exhaustive
        )
        assert conversation_archive._document_matches_one_run(rows, transcript, count) == exhaustive
        matched += exhaustive
    assert matched > 200, "the random cases must exercise real matches"


def test_transcript_index_finds_every_holder_once():
    texts = ["abc abc", "", "xabc", "ab", "c abc\nabc"]
    index = conversation_archive._TranscriptIndex(texts)
    assert index.containing("abc") == [0, 2, 4]
    assert index.containing("ab") == [0, 2, 3, 4]
    assert index.containing("zzz") == []


def _seed(n = 40):
    turns = []
    for i in range(n):
        turns += [
            {"role": "user", "content": f"how do we handle TICKET-{i % 5} in module m{i}"},
            {"role": "assistant", "content": f"we retry the cache for m{i} twice"},
        ]
    _archive(turns)
    return turns


def _compaction(turns, query, session):
    from core.inference import tools

    branch = turns + [{"role": "user", "content": query}]

    def attempts():
        return [
            tools.build_conversation_recall(
                branch[-3:], THREAD, style = "inline", top_k = k, branch_messages = branch
            )
            for k in (4, 2, 1)
        ]

    if not session:
        return attempts()
    with conversation_archive.recall_session():
        return attempts()


@pytest.mark.parametrize(
    "query", ["TICKET-3 retry cache", "module m7", "continue", "nothing matches xyzzy"]
)
def test_a_session_returns_what_separate_recalls_return(conn, query):
    turns = _seed()
    assert _compaction(turns, query, session = True) == _compaction(turns, query, session = False)


def test_retries_in_a_session_share_lexical_passes_and_branch_checks(conn, monkeypatch):
    turns = _seed()
    calls = {"fts": 0, "texts": 0, "match": 0}
    real_fts, real_texts, real_match = (
        store.search_lexical,
        conversation_archive.branch_message_texts,
        conversation_archive._document_matches_one_run,
    )

    def count(key, fn):
        def wrapped(*a, **kw):
            calls[key] += 1
            return fn(*a, **kw)

        return wrapped

    monkeypatch.setattr(store, "search_lexical", count("fts", real_fts))
    monkeypatch.setattr(conversation_archive, "branch_message_texts", count("texts", real_texts))
    monkeypatch.setattr(
        conversation_archive, "_document_matches_one_run", count("match", real_match)
    )

    _compaction(turns, "TICKET-3 retry cache", session = False)
    separate = dict(calls)
    calls.update(fts = 0, texts = 0, match = 0)
    _compaction(turns, "TICKET-3 retry cache", session = True)

    assert separate["texts"] == 3 and calls["texts"] == 1
    assert calls["fts"] < separate["fts"]
    assert calls["match"] < separate["match"]


def test_an_archive_write_inside_a_session_is_seen_by_the_next_recall(conn, monkeypatch):
    from core.rag import retrieval

    # No hybrid top-up, so only the memoized focused lexical passes can find the new turn.
    monkeypatch.setattr(retrieval, "retrieve_hybrid", lambda *a, **kw: [])
    turns = _seed(6)
    branch = list(turns)
    with conversation_archive.recall_session():
        before = conversation_archive.recall(THREAD, "zanzibar", branch_messages = branch)
        assert not before or "zanzibar" not in before[0]
        new = [
            {"role": "user", "content": "what about zanzibar"},
            {"role": "assistant", "content": "zanzibar ships friday"},
        ]
        _archive(new)
        branch += new
        found = conversation_archive.recall(THREAD, "zanzibar", branch_messages = branch)
    assert found and "zanzibar" in found[0]
