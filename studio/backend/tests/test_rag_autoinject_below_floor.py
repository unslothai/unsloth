# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import math

import pytest

from core.inference import tools
from core.rag import store
from core.rag.chunking import Chunk

HANDBOOK = "Full-time employees receive 20 paid vacation days per year."
COSINE = 0.69


def _embed(text):
    if text == HANDBOOK:
        return [COSINE, math.sqrt(1 - COSINE * COSINE)]
    return [1.0, 0.0]


@pytest.fixture
def handbook(rag_conn, monkeypatch):
    from core.rag import embeddings

    monkeypatch.setattr(
        embeddings,
        "encode",
        lambda texts, *, model_name = None, normalize = True: [_embed(t) for t in texts],
    )
    monkeypatch.setattr(embeddings, "dim", lambda model_name = None: 2)

    def add(scope):
        store.create_document(
            rag_conn, scope = scope, filename = "handbook.md", sha256 = "h", document_id = "d"
        )
        chunk = Chunk(
            text = HANDBOOK,
            token_count = len(HANDBOOK.split()),
            page_number = None,
            source_page_index = 0,
            chunk_index = 0,
            page_char_start = 0,
            page_char_end = len(HANDBOOK),
        )
        store.add_chunks(rag_conn, scope, "d", [chunk], [_embed(HANDBOOK)])
        store.set_document_status(rag_conn, "d", "completed", num_chunks = 1)

    return add


def _ask(text, scope):
    return tools.build_rag_autoinject([{"role": "user", "content": text}], scope)


def test_handbook_answer_below_floor_still_reaches_the_model(handbook):
    handbook(store.kb_scope("K"))
    out = _ask(
        "How many vacation days do I get?",
        {"kb_id": "K", "thread_id": "T", "autoinject": True, "autoinject_min_score": 0.7},
    )
    assert out is not None
    assert "20 paid vacation days" in json.dumps(out)


def test_project_sources_below_floor_inject_nothing(handbook):
    handbook(store.project_scope("P"))
    out = _ask(
        "How many vacation days do I get?",
        {"project_id": "P", "thread_id": "T", "autoinject": True, "autoinject_min_score": 0.7},
    )
    assert out is None


def test_nudge_below_floor_injects_nothing(handbook):
    handbook(store.kb_scope("K"))
    assert (
        _ask("ok, thanks!", {"kb_id": "K", "autoinject": True, "autoinject_min_score": 0.7}) is None
    )
