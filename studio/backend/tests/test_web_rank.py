"""Unit tests for the ephemeral web-RAG used by deep research auto-read.

These run the *real* Studio RAG store + hybrid retrieval + formatter against a temporary
rag.db (so the ingest -> retrieve -> render reuse chain is exercised end to end) with a fake
deterministic embedding so no model is downloaded. They also assert the ephemeral scope is
deleted, i.e. an auto-read leaves nothing behind in the store."""

import numpy as np
import pytest

from core.rag import web_rank


@pytest.fixture
def rag_home(tmp_path, monkeypatch):
    """Point rag.db at a throwaway file and rebuild its schema there."""
    from storage import rag_db

    db_file = tmp_path / "rag.db"
    monkeypatch.setattr(rag_db, "rag_db_path", lambda: db_file)
    monkeypatch.setattr(rag_db, "_schema_ready", False, raising = False)
    return db_file


@pytest.fixture(autouse = True)
def fake_embeddings(monkeypatch):
    """Token counter = word count; embedding = 3-d bag over 'lora'/'license' (+ tiny bias),
    so relevance is deterministic and independent of any downloaded model."""
    from core.rag import embeddings as rag_embeddings

    monkeypatch.setattr(
        rag_embeddings,
        "token_counter",
        lambda model_name = None: (lambda text: max(1, len(text.split()))),
    )

    def encode(texts, *, model_name = None, normalize = True):
        rows = []
        for text in texts:
            low = text.lower()
            vec = np.array(
                [float(low.count("lora")), float(low.count("license")), 0.001],
                dtype = "float32",
            )
            norm = np.linalg.norm(vec)
            rows.append(vec / norm if (normalize and norm) else vec)
        return np.stack(rows)

    monkeypatch.setattr(rag_embeddings, "encode", encode)


def _scope_rows(db_file):
    """Count leftover ephemeral documents/chunks in the store."""
    import sqlite3

    conn = sqlite3.connect(str(db_file))
    try:
        docs = conn.execute(
            "SELECT count(*) FROM documents WHERE scope LIKE 'research_scrape_%'"
        ).fetchone()[0]
        chunks = conn.execute(
            "SELECT count(*) FROM chunks WHERE scope LIKE 'research_scrape_%'"
        ).fetchone()[0]
        return docs, chunks
    finally:
        conn.close()


def test_retrieves_relevant_passages_as_chunks(rag_home):
    pages = [
        {"text": "LoRA is a low-rank adapter method for fine tuning.", "title": "LoRA", "url": "https://a"},
        {"text": "The Apache license governs redistribution terms.", "title": "License", "url": "https://b"},
    ]
    rendered, sources = web_rank.retrieve_web_chunks(pages, "what is lora", top_n = 5, min_score = 0.0)

    assert "<chunk" in rendered
    assert "LoRA" in rendered
    assert sources and sources[0]["citationId"] == 1
    # source attribution is the page title, via Studio's formatter
    assert 'source="LoRA"' in rendered


def test_min_score_floor_drops_irrelevant(rag_home):
    pages = [
        {"text": "LoRA adapters reduce trainable parameters for fine tuning.", "url": "https://a"},
        {"text": "Completely separate cooking recipe with onions and garlic.", "url": "https://b"},
    ]
    rendered, _ = web_rank.retrieve_web_chunks(pages, "lora fine tuning", top_n = 5, min_score = 0.5)
    assert "cooking" not in rendered.lower()
    assert "lora" in rendered.lower()


def test_char_budget_caps_kept_chunks(rag_home):
    # ~2000 words -> several ~500-word chunks; a tight budget keeps a bounded subset.
    pages = [{"text": " ".join(["lora"] * 2000), "url": "https://a"}]
    full, _ = web_rank.retrieve_web_chunks(pages, "lora", top_n = 10, min_score = 0.0)
    capped, _ = web_rank.retrieve_web_chunks(
        pages, "lora", top_n = 10, min_score = 0.0, char_budget = 3000
    )
    assert full.count("<chunk id") >= 2
    assert 1 <= capped.count("<chunk id") < full.count("<chunk id")


def test_empty_and_invalid_inputs_return_empty(rag_home):
    assert web_rank.retrieve_web_chunks([], "lora", top_n = 5, min_score = 0.1) == ("", [])
    assert web_rank.retrieve_web_chunks([{"text": " "}], "lora", top_n = 5, min_score = 0.1) == ("", [])
    assert web_rank.retrieve_web_chunks([{"text": "lora"}], "", top_n = 5, min_score = 0.1) == ("", [])
    assert web_rank.retrieve_web_chunks([{"text": "lora"}], "lora", top_n = 0, min_score = 0.1) == ("", [])


def test_ephemeral_scope_is_cleaned_up(rag_home):
    pages = [{"text": "LoRA low-rank adaptation fine tuning.", "title": "LoRA", "url": "https://a"}]
    rendered, _ = web_rank.retrieve_web_chunks(pages, "lora", top_n = 5, min_score = 0.0)
    assert "<chunk" in rendered
    # nothing from the auto-read is left in the store
    assert _scope_rows(rag_home) == (0, 0)
