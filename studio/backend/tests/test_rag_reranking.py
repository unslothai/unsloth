# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import time
from dataclasses import replace

import pytest

from core.rag import config, reranking, retrieval, store, tool
from core.rag.chunking import Chunk


@pytest.fixture(autouse = True)
def reset_reranker(monkeypatch):
    monkeypatch.setattr(reranking, "enabled", lambda: True)
    monkeypatch.setattr(config, "RERANK_CANDIDATES", 3)
    monkeypatch.setattr(reranking, "_retry_after", 0.0)


@pytest.fixture
def candidates(rag_conn, monkeypatch):
    hits = []
    for i, text in enumerate(
        ("Returns are discussed here.", "Shipping takes a week.", "Returns close after 30 days.")
    ):
        doc = f"d{i}"
        store.create_document(
            rag_conn, scope = "kb_test", filename = f"{doc}.txt", sha256 = doc, document_id = doc
        )
        store.add_chunks(
            rag_conn,
            "kb_test",
            doc,
            [
                Chunk(
                    text = text,
                    token_count = 8,
                    page_number = i + 1,
                    source_page_index = 0,
                    chunk_index = 0,
                    page_char_start = 0,
                    page_char_end = len(text),
                )
            ],
            [[1.0, 0.0]],
        )
        hits.append(retrieval.Hit(f"{doc}:0", 1.0 / (i + 1), dense_score = 0.9 - i * 0.1))
    monkeypatch.setattr(
        retrieval, "retrieve_hybrid", lambda conn, scope, query, **kw: hits[: kw["k"]]
    )
    return hits


def test_reranks_wider_pool_without_changing_retrieval_scores(rag_conn, candidates, monkeypatch):
    seen = []

    def score(query, passages):
        seen.extend(passages)
        return [0.1, 0.2, 0.9]

    monkeypatch.setattr(reranking, "score", score)
    hits = retrieval.retrieve_ranked(rag_conn, "kb_test", "Return deadline?", k = 1)
    assert len(seen) == 3
    assert hits[0].chunk_id == "d2:0"
    assert hits[0].score == candidates[2].score
    assert hits[0].dense_score == candidates[2].dense_score
    assert hits[0].rerank_score == 0.9
    assert all(hit.rerank_score is None for hit in candidates)


def test_similarity_floor_is_applied_before_reranking(rag_conn, candidates, monkeypatch):
    monkeypatch.setattr(reranking, "score", lambda query, passages: [0.1, 0.9])
    hits = retrieval.retrieve_ranked(rag_conn, "kb_test", "q", k = 1, min_score = 0.75)
    assert [h.chunk_id for h in hits] == ["d1:0"]


def test_equal_scores_keep_retrieval_order(rag_conn, candidates, monkeypatch):
    monkeypatch.setattr(reranking, "score", lambda query, passages: [0.5] * len(passages))
    assert retrieval.retrieve_ranked(rag_conn, "kb_test", "q", k = 2) == [
        replace(h, rerank_score = 0.5) for h in candidates[:2]
    ]


@pytest.mark.parametrize("enabled", [False, True])
def test_disabled_or_unavailable_preserves_original_result(
    rag_conn, candidates, monkeypatch, enabled
):
    monkeypatch.setattr(reranking, "enabled", lambda: enabled)
    monkeypatch.setattr(
        reranking, "score", lambda *args: None if enabled else pytest.fail("reranker called")
    )
    assert retrieval.retrieve_ranked(rag_conn, "kb_test", "q", k = 2) == candidates[:2]


def test_oversized_result_limit_bypasses_reranker(rag_conn, candidates, monkeypatch):
    monkeypatch.setattr(reranking, "score", lambda *args: pytest.fail("reranker called"))
    assert retrieval.retrieve_ranked(rag_conn, "kb_test", "q", k = 4) == candidates


def test_conversation_search_preserves_retrieval_order(rag_conn, candidates, monkeypatch):
    monkeypatch.setattr(reranking, "score", lambda *args: pytest.fail("reranker called"))
    _, sources = tool.search_knowledge_base_with_sources(
        query = "q",
        scope_conversation_id = "thread",
        top_k = 1,
    )
    assert sources[0]["chunkId"] == candidates[0].chunk_id


def test_real_lexical_retrieval_keeps_scopes_separate(rag_conn, candidates, monkeypatch):
    monkeypatch.setattr(
        retrieval,
        "retrieve_hybrid",
        lambda conn, scope, query, **kw: retrieval.retrieve_lexical(conn, scope, query, kw["k"]),
    )
    seen = []

    def score(query, passages):
        seen.extend(passages)
        return [float("30 days" in text) for text in passages]

    monkeypatch.setattr(reranking, "score", score)
    hits = retrieval.retrieve_ranked(rag_conn, "kb_test", "Returns", k = 1, mode = "lexical")
    assert hits[0].chunk_id == "d2:0"
    assert len(seen) == 2
    assert retrieval.retrieve_ranked(rag_conn, "kb_other", "Returns", k = 1, mode = "lexical") == []
    assert len(seen) == 2


@pytest.mark.parametrize("mode", ["hybrid", "dense", "lexical"])
def test_search_tool_preserves_citations(rag_conn, candidates, monkeypatch, mode):
    monkeypatch.setattr(reranking, "score", lambda *args: [0.1, 0.2, 0.9])
    text, sources = tool.search_knowledge_base_with_sources(
        query = "Return deadline?",
        scope_kb_id = "test",
        top_k = 1,
        mode = mode,
    )
    assert '<chunk id="1" source="d2.txt" page="3">' in text
    assert "30 days" in text
    assert sources[0]["chunkId"] == "d2:0"
    assert sources[0]["citationId"] == 1
    assert sources[0]["rerankScore"] == 0.9


def test_autoinject_uses_reranking_after_similarity_gate(rag_conn, candidates, monkeypatch):
    monkeypatch.setattr(reranking, "score", lambda *args: [0.1, 0.9])
    text, sources = tool.search_for_autoinject(
        query = "q",
        scope_kb_id = "test",
        top_k = 1,
        min_dense_score = 0.75,
    )
    assert sources[0]["chunkId"] == "d1:0"
    assert "Shipping" in text


def test_autoinject_lexical_only_hits_cannot_displace_dense_evidence(
    rag_conn, candidates, monkeypatch
):
    candidates[1].dense_score = None
    monkeypatch.setattr(
        reranking,
        "score",
        lambda query, passages: [0.99 if "Shipping" in passage else 0.5 for passage in passages],
    )
    result = tool.search_for_autoinject(
        query = "q",
        scope_kb_id = "test",
        top_k = 1,
        min_dense_score = 0.75,
    )
    assert result is not None
    assert result[1][0]["chunkId"] == candidates[0].chunk_id


def test_autoinject_eligibility_comes_from_unreranked_top_k(rag_conn, candidates, monkeypatch):
    candidates[0].dense_score = None
    candidates[1].dense_score = candidates[2].dense_score = 0.8
    monkeypatch.setattr(reranking, "score", lambda *args: pytest.fail("reranker called"))
    assert (
        tool.search_for_autoinject(
            query = "q",
            scope_kb_id = "test",
            top_k = 1,
            min_dense_score = 0.75,
        )
        is None
    )


def test_search_route_returns_reranked_results(rag_conn, candidates, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.authentication import get_current_subject
    from routes.rag import router

    monkeypatch.setattr(reranking, "score", lambda *args: [0.1, 0.2, 0.9])
    app = FastAPI()
    app.include_router(router, prefix = "/api/rag")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    response = TestClient(app).post(
        "/api/rag/search",
        json = {"query": "q", "thread_id": "t", "top_k": 1, "mode": "dense"},
    )
    assert response.status_code == 200, response.text
    [result] = response.json()["results"]
    assert result["chunkId"] == "d2:0"
    assert result["score"] == candidates[2].score
    assert result["rerankScore"] == 0.9


def test_failed_reranking_does_not_backfill_filtered_baseline(rag_conn, candidates, monkeypatch):
    candidates[0].dense_score = 0.1
    monkeypatch.setattr(reranking, "score", lambda *args: None)
    assert retrieval.retrieve_ranked(rag_conn, "kb_test", "q", k = 1, min_score = 0.5) == []


def test_missing_row_falls_back(rag_conn, candidates, monkeypatch):
    monkeypatch.setattr(store, "chunks_by_id", lambda *args: {})
    monkeypatch.setattr(reranking, "score", lambda *args: pytest.fail("reranker called"))
    assert retrieval.retrieve_ranked(rag_conn, "kb_test", "q", k = 1) == candidates[:1]


@pytest.mark.parametrize(
    "settings, expected",
    [
        ({}, False),
        ({"systemone_enabled": True}, False),
        ({"systemone_rag_rerank": True}, False),
        ({"systemone_enabled": True, "systemone_rag_rerank": True}, True),
    ],
)
def test_reranking_needs_the_decision_api_and_its_switch(monkeypatch, settings, expected):
    from utils import systemone_settings

    monkeypatch.undo()
    monkeypatch.delenv("UNSLOTH_SYSTEMONE_DISABLE", raising = False)
    monkeypatch.setattr(systemone_settings, "_owner_setting", settings.get)
    assert reranking.enabled() is expected


def test_disabled_never_touches_the_model(monkeypatch):
    from core.systemone import laya_runtime

    monkeypatch.setattr(reranking, "enabled", lambda: False)
    monkeypatch.setattr(laya_runtime, "score_noul", lambda *args: pytest.fail("model used"))
    assert reranking.score("q", ["p"]) is None


def test_passages_are_scored_as_question_and_passage(monkeypatch):
    from core.systemone import laya_runtime

    seen = []

    def score_noul(question, states):
        seen.append((question, states))
        return [0.25, 0.75]

    monkeypatch.setattr(laya_runtime, "score_noul", score_noul)
    assert reranking.score("q", ["one", "two"]) == [0.25, 0.75]
    [(question, states)] = seen
    assert question == reranking._QUESTION
    assert states == ["Question:\nq\n\nPassage:\none", "Question:\nq\n\nPassage:\ntwo"]


def test_model_not_ready_keeps_retrieval_order_without_backoff(monkeypatch):
    from core.systemone import laya_runtime

    monkeypatch.setattr(laya_runtime, "score_noul", lambda *args: None)
    assert reranking.score("q", ["p"]) is None
    assert reranking._retry_after == 0.0


def test_long_input_skips_without_poisoning_next_request(monkeypatch):
    from core.systemone import laya_runtime

    def oversized(*args):
        raise ValueError("context window")

    monkeypatch.setattr(laya_runtime, "score_noul", oversized)
    assert reranking.score("long", ["passage"]) is None
    assert reranking._retry_after == 0.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, "bad"])
def test_invalid_model_output_falls_back_and_backs_off(monkeypatch, value):
    from core.systemone import laya_runtime

    monkeypatch.setattr(laya_runtime, "score_noul", lambda *args: [value])
    assert reranking.score("q", ["p"]) is None
    assert reranking._retry_after > time.monotonic()
    monkeypatch.setattr(laya_runtime, "score_noul", lambda *args: pytest.fail("model used"))
    assert reranking.score("q", ["p"]) is None


def test_model_failure_backs_off(monkeypatch):
    from core.systemone import laya_runtime

    def boom(*args):
        raise RuntimeError("model broke")

    monkeypatch.setattr(laya_runtime, "score_noul", boom)
    assert reranking.score("q", ["p"]) is None
    assert reranking._retry_after > time.monotonic()


@pytest.fixture
def resident(monkeypatch):
    from core.systemone import catalog, laya_runtime

    checkpoint = catalog.CHECKPOINTS["laya-multilingual"]
    monkeypatch.setattr(catalog, "default_checkpoint", lambda: checkpoint)
    for name, value in (("_agent", None), ("_loaded", None), ("_loader", None), ("_loading", None), ("_failure", None)):
        monkeypatch.setattr(laya_runtime, name, value)
    return laya_runtime, checkpoint


def test_unloaded_model_starts_loading_only_when_already_downloaded(monkeypatch, resident):
    laya_runtime, checkpoint = resident
    started = []
    monkeypatch.setattr(laya_runtime, "package_available", lambda: True)
    monkeypatch.setattr(laya_runtime, "_ensure_loading", lambda c: started.append(c))
    monkeypatch.setattr(laya_runtime, "is_cached", lambda c: False)
    assert laya_runtime.score_noul(reranking._QUESTION, ["s"]) is None
    assert started == []
    monkeypatch.setattr(laya_runtime, "is_cached", lambda c: True)
    assert laya_runtime.score_noul(reranking._QUESTION, ["s"]) is None
    assert started == [checkpoint]


def test_unloaded_model_is_not_installed_from_a_search(monkeypatch, resident):
    laya_runtime, _ = resident
    monkeypatch.setattr(laya_runtime, "package_available", lambda: False)
    monkeypatch.setattr(laya_runtime, "_ensure_loading", lambda c: pytest.fail("load started"))
    assert laya_runtime.score_noul(reranking._QUESTION, ["s"]) is None


def test_load_backoff_does_not_break_search(monkeypatch, resident):
    laya_runtime, _ = resident

    def unavailable(checkpoint):
        raise laya_runtime.Unavailable(503, "model_unavailable", "failed")

    monkeypatch.setattr(laya_runtime, "package_available", lambda: True)
    monkeypatch.setattr(laya_runtime, "is_cached", lambda c: True)
    monkeypatch.setattr(laya_runtime, "_ensure_loading", unavailable)
    assert laya_runtime.score_noul(reranking._QUESTION, ["s"]) is None


def test_search_does_not_queue_behind_a_decision_request(monkeypatch, resident):
    laya_runtime, checkpoint = resident
    monkeypatch.setattr(laya_runtime, "_agent", object())
    monkeypatch.setattr(laya_runtime, "_loaded", checkpoint)
    monkeypatch.setattr(laya_runtime, "_forward", lambda *args: pytest.fail("model called"))
    with laya_runtime._run_lock:
        assert laya_runtime.score_noul(reranking._QUESTION, ["s"]) is None


def test_other_resident_checkpoint_is_neither_used_nor_evicted(monkeypatch, resident):
    from core.systemone import catalog

    laya_runtime, _ = resident
    monkeypatch.setattr(laya_runtime, "_agent", object())
    monkeypatch.setattr(laya_runtime, "_loaded", catalog.CHECKPOINTS["laya-english"])
    monkeypatch.setattr(laya_runtime, "package_available", lambda: True)
    monkeypatch.setattr(laya_runtime, "is_cached", lambda c: True)
    monkeypatch.setattr(laya_runtime, "_ensure_loading", lambda c: pytest.fail("evicted"))
    monkeypatch.setattr(laya_runtime, "_forward", lambda *args: pytest.fail("model called"))
    assert laya_runtime.score_noul(reranking._QUESTION, ["s"]) is None


def test_real_scores_match_laya_predict_and_rank_the_answer_first(monkeypatch, resident):
    path = os.environ.get("SYSTEMONE_TEST_LAYA")
    if not path:
        pytest.skip("set SYSTEMONE_TEST_LAYA to a downloaded convaiinnovations/laya snapshot")
    laya = pytest.importorskip("laya")
    laya_runtime, checkpoint = resident
    agent = laya.load(path, subfolder = "multilingual", device = "cpu")
    monkeypatch.setattr(laya_runtime, "_agent", agent)
    monkeypatch.setattr(laya_runtime, "_loaded", checkpoint)
    query = "How many days do customers have to return an item?"
    passages = [
        "Shipping takes five to seven business days.",
        "Customers may return any unused item within 30 days of delivery.",
    ] + [f"Warehouse report {i}: pallets and shelving. " * (i + 1) for i in range(9)]
    states = [f"Question:\n{query}\n\nPassage:\n{p}" for p in passages]
    started = time.perf_counter()
    scores = laya_runtime.score_noul(reranking._QUESTION, states)
    elapsed = time.perf_counter() - started
    single = [agent.predict(s, {"r": reranking._QUESTION})["answers"]["r"]["noul"] for s in states]
    assert scores == pytest.approx(single, abs = 2e-3)
    assert max(range(len(scores)), key = scores.__getitem__) == 1
    print(f"scored {len(states)} passages in {elapsed:.2f}s")
    with pytest.raises(ValueError):
        laya_runtime.score_noul(reranking._QUESTION, ["word " * 5000])
