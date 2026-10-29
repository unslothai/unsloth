# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#9485: picking a knowledge base in a project chat must not drop the project's sources."""

import asyncio
import json
import math

import pytest

from core.rag import store, tool
from core.rag.chunking import Chunk

VOCAB = ["alpha", "bravo", "company", "customer", "thread"]


def _embed(text):
    v = [float(text.lower().count(w)) for w in VOCAB]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


@pytest.fixture
def bow_embeddings(monkeypatch):
    from core.rag import embeddings
    monkeypatch.setattr(
        embeddings,
        "encode",
        lambda texts, *, model_name = None, normalize = True: [_embed(t) for t in texts],
    )
    monkeypatch.setattr(embeddings, "dim", lambda model_name = None: len(VOCAB))


def _add_doc(conn, scope, doc_id, filename, text):
    store.create_document(conn, scope = scope, filename = filename, sha256 = doc_id, document_id = doc_id)
    chunk = Chunk(
        text = text,
        token_count = len(text.split()),
        page_number = None,
        source_page_index = 0,
        chunk_index = 0,
        page_char_start = 0,
        page_char_end = len(text),
    )
    store.add_chunks(conn, scope, doc_id, [chunk], [_embed(text)])
    conn.execute("UPDATE documents SET status='completed', num_chunks=1 WHERE id=?", (doc_id,))
    conn.commit()


def test_scope_resolution():
    assert tool._resolve_scope("K", "T", "P") == ["kb_K", "project_P"]
    assert tool._resolve_scope("K", "T", None) == "kb_K"
    assert tool._resolve_scope(None, "T", "P") == ["project_P", "thread_T"]
    # The conversation archive stays exclusive.
    assert tool._resolve_scope("K", "T", "P", "C") == store.conversation_archive_scope("C")


def _seed(conn):
    _add_doc(conn, "kb_K", "dk", "company.pdf", "alpha bravo company policy")
    _add_doc(conn, "project_P", "dp", "customer.pdf", "alpha bravo customer notes")
    _add_doc(conn, "thread_T", "dt", "thread.pdf", "alpha bravo thread upload")


def test_search_with_kb_also_reads_project(rag_conn, bow_embeddings):
    _seed(rag_conn)
    _, sources = tool.search_knowledge_base_with_sources(
        query = "alpha bravo", scope_kb_id = "K", scope_thread_id = "T", scope_project_id = "P"
    )
    assert {s["filename"] for s in sources} == {"company.pdf", "customer.pdf"}


def test_autoinject_with_kb_also_reads_project(rag_conn, bow_embeddings, monkeypatch):
    from core.inference import tools
    from storage import rag_db

    monkeypatch.setattr(rag_db, "RAG_AVAILABLE", True, raising = False)
    monkeypatch.setattr(rag_db, "rag_available", lambda: True, raising = False)
    _seed(rag_conn)
    out = tools.build_rag_autoinject(
        [{"role": "user", "content": "alpha bravo"}],
        {
            "kb_id": "K",
            "thread_id": "T",
            "project_id": "P",
            "autoinject": True,
            "autoinject_min_score": 0.0,
        },
    )
    blob = json.dumps(out)
    assert "company.pdf" in blob and "customer.pdf" in blob and "thread.pdf" not in blob


def test_roster_with_kb_also_names_project(rag_conn):
    from routes import inference

    _seed(rag_conn)
    tools = [{"type": "function", "function": {"name": "search_knowledge_base"}}]
    out = asyncio.run(
        inference._apply_rag_nudge(
            "", tools, rag_scope = {"kb_id": "K", "thread_id": "T", "project_id": "P"}
        )
    )
    assert '"company.pdf"' in out and '"customer.pdf"' in out and '"thread.pdf"' not in out
