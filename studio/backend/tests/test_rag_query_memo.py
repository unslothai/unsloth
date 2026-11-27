# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Query vectors are encoded once per recall or compaction, and archive widening reaches past the hybrid legs."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.inference import llama_cpp  # noqa: E402
from core.rag import conversation_archive, embeddings, retrieval, store  # noqa: E402

THREAD = "thread-memo"


def _save(thread_id, messages):
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


def _archive(turn):
    _save(THREAD, turn)
    conversation_archive.archive_turns(THREAD, turn)


def _turn(question, answer):
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


@pytest.fixture
def encodes(rag_home, rag_conn, stub_embeddings, monkeypatch):
    calls = []
    real = embeddings.encode

    def counting(texts, **kwargs):
        calls.append(list(texts))
        return real(texts, **kwargs)

    monkeypatch.setattr(embeddings, "encode", counting)
    return calls


def test_retrieve_dense_encodes_once_inside_a_memo_and_again_outside(encodes, rag_conn):
    scope = store.conversation_archive_scope(THREAD)
    with retrieval.query_vector_memo():
        with retrieval.query_vector_memo():
            retrieval.retrieve_dense(rag_conn, scope, "pelicans", 5)
        retrieval.retrieve_dense(rag_conn, scope, "pelicans", 5)
    retrieval.retrieve_dense(rag_conn, scope, "pelicans", 5)
    assert encodes.count(["pelicans"]) == 2


def test_a_compaction_encodes_its_query_once_across_budget_retries(encodes):
    _archive(_turn("write me a limerick about pelicans", "There once was a bird with a bill"))
    query = "what was that pelicans limerick"
    conversation = [{"role": "system", "content": "sys"}, {"role": "user", "content": query}]
    del encodes[:]

    result = llama_cpp._archive_and_recall(
        conversation,
        conversation,
        thread_id = THREAD,
        style = "inline",
        recall_done = False,
        recall_budget_tokens = 4000,
        # Every attempt overflows, so all of 4 -> 2 -> 1 run.
        count_tokens = lambda conv: 10**7 if "recalled" in repr(conv) else 0,
    )

    assert result["recalled"] is False
    assert encodes.count([query]) == 1


def test_separate_recalls_do_not_share_vectors(encodes):
    _archive(_turn("write me a limerick about pelicans", "There once was a bird with a bill"))
    del encodes[:]
    conversation_archive.recall(THREAD, "pelicans limerick")
    conversation_archive.recall(THREAD, "pelicans limerick")
    assert encodes.count(["pelicans limerick"]) == 2


def test_a_recall_lexical_fills_never_encodes(encodes):
    for i in range(20):
        _archive(_turn(f"question {i} about zx81tape", f"answer {i} zx81tape"))
    del encodes[:]
    found = conversation_archive.recall(THREAD, "zx81tape", top_k = 4)
    assert found is not None
    assert not any(texts == ["zx81tape"] for texts in encodes)


def test_widening_reaches_a_live_turn_ranked_past_the_dense_leg(
    rag_home, rag_conn, stub_embeddings, monkeypatch
):
    """40 abandoned-branch turns outrank the live one by cosine; widening to 64 must reach it."""
    dim = stub_embeddings

    def vec(text):
        v = [0.0] * dim
        if "flamingo" in text:
            v[0] = 1.0
        elif "decoy" in text:
            v[0], v[1] = 1.0, 0.05 + 0.001 * (len(text) % 7)
        elif "zebra" in text:
            v[0], v[1] = 0.6, 0.8
        else:
            v[2] = 1.0
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v]

    monkeypatch.setattr(embeddings, "encode", lambda texts, **_k: [vec(t) for t in texts])
    for i in range(40):
        _archive(_turn(f"decoy question number {i}", f"decoy reply {i}"))
    live = _turn("tell me about zebra stripes", "zebra stripes deter flies")
    _archive(live)

    found = conversation_archive.recall(THREAD, "flamingo", top_k = 4, branch_messages = live)

    assert found is not None
    assert "zebra" in found[0]
