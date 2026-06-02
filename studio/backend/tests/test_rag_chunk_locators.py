# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import queue as queue_module
import uuid

import pytest
import storage.studio_db as studio_db
from core.rag.chunking import chunk_pages, chunk_pages_with_spans
from core.rag.ingestion import (
    _JobState,
    _insert_chunks,
    _pump,
    _replace_document_pages,
)
from core.rag.parsers import ParsedPage


def _uid() -> str:
    return str(uuid.uuid4())


def _token_count(text: str) -> int:
    return max(1, len(text.split()))


def test_standard_chunking_records_page_local_char_and_line_spans():
    pages = [
        ParsedPage(
            text = "alpha first line\nbeta target line\ngamma final line",
            page_number = 7,
        )
    ]

    chunks = chunk_pages(
        pages,
        max_tokens = 3,
        overlap_tokens = 0,
        token_counter = _token_count,
        separators = ("\n", " ", ""),
    )

    target = next(chunk for chunk in chunks if "beta" in chunk.text)
    assert target.page_number == 7
    assert target.source_page_index == 0
    assert target.page_char_start == pages[0].text.index("beta target line")
    assert target.page_char_end == target.page_char_start + len("beta target line")
    assert target.line_start == 2
    assert target.line_end == 2


def test_late_chunking_maps_global_span_back_to_source_page():
    pages = [
        ParsedPage(text = "page one alpha", page_number = 1),
        ParsedPage(text = "page two beta target", page_number = 2),
    ]

    _full_doc, chunks, spans = chunk_pages_with_spans(
        pages,
        max_tokens = 4,
        overlap_tokens = 0,
        token_counter = _token_count,
        separators = ("\n\n", " ", ""),
    )

    target = next(chunk for chunk in chunks if "beta" in chunk.text)
    assert spans[chunks.index(target)][0] >= len(pages[0].text)
    assert target.page_number == 2
    assert target.source_page_index == 1
    assert target.page_char_start is not None
    assert target.page_char_end is not None
    assert pages[1].text[target.page_char_start : target.page_char_end].strip()


def test_image_chunk_persistence_keeps_page_focus_and_null_text_locators(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    from core.rag import ingestion

    captured_points: list[dict] = []
    monkeypatch.setattr(
        ingestion.vector_store,
        "upsert_chunks",
        lambda _scope, points: captured_points.extend(points),
    )

    kb_id = _uid()
    doc_id = _uid()
    with studio_db.get_connection() as conn:
        conn.execute(
            """
            INSERT INTO rag_knowledge_bases
            (id, name, embedding_model, owner_user_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (kb_id, "KB", "embedder", "alice", 1_700_000_000),
        )
        conn.execute(
            """
            INSERT INTO rag_documents
            (id, kb_id, thread_id, filename, content_type, stored_path, status,
             num_chunks, byte_size, created_at)
            VALUES (?, ?, NULL, ?, ?, ?, 'completed', 1, 10, ?)
            """,
            (doc_id, kb_id, "image.pdf", "application/pdf", "image.pdf", 1_700_000_001),
        )

    _insert_chunks(
        doc_id,
        "kb_scope",
        0,
        [
            {
                "text": "",
                "token_count": 0,
                "page_number": 3,
                "kind": "image",
                "image_path": str(tmp_path / "img.png"),
            }
        ],
        [[0.1, 0.2]],
    )

    with studio_db.get_connection() as conn:
        row = conn.execute(
            """
            SELECT page_number, source_page_index, page_char_start,
                   page_char_end, line_start, line_end
            FROM rag_chunks WHERE document_id = ?
            """,
            (doc_id,),
        ).fetchone()
    assert row["page_number"] == 3
    assert row["source_page_index"] is None
    assert row["page_char_start"] is None
    assert row["page_char_end"] is None
    assert row["line_start"] is None
    assert row["line_end"] is None
    assert captured_points[0]["payload"]["page_number"] == 3
    assert captured_points[0]["payload"]["page_char_start"] is None


def test_replace_document_pages_replaces_existing_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    kb_id = _uid()
    doc_id = _uid()
    with studio_db.get_connection() as conn:
        conn.execute(
            """
            INSERT INTO rag_knowledge_bases
            (id, name, embedding_model, owner_user_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (kb_id, "KB", "embedder", "alice", 1_700_000_000),
        )
        conn.execute(
            """
            INSERT INTO rag_documents
            (id, kb_id, thread_id, filename, content_type, stored_path, status,
             num_chunks, byte_size, created_at)
            VALUES (?, ?, NULL, ?, ?, ?, 'completed', 1, 10, ?)
            """,
            (doc_id, kb_id, "doc.pdf", "application/pdf", "doc.pdf", 1_700_000_001),
        )

    _replace_document_pages(
        doc_id,
        [
            {
                "page_index": 0,
                "page_number": 1,
                "text": "old page",
                "char_count": 8,
                "line_count": 1,
            }
        ],
    )
    _replace_document_pages(
        doc_id,
        [
            {
                "page_index": 1,
                "page_number": 2,
                "text": "new\npage",
                "char_count": 8,
                "line_count": 2,
            }
        ],
    )

    with studio_db.get_connection() as conn:
        rows = conn.execute(
            """
            SELECT page_index, page_number, text, char_count, line_count
            FROM rag_document_pages WHERE document_id = ?
            """,
            (doc_id,),
        ).fetchall()

    assert [dict(row) for row in rows] == [
        {
            "page_index": 1,
            "page_number": 2,
            "text": "new\npage",
            "char_count": 8,
            "line_count": 2,
        }
    ]


class _OneMessageQueue:
    def __init__(self, message: dict) -> None:
        self.message = message
        self.used = False

    def get(self, timeout: float) -> dict:
        if self.used:
            raise queue_module.Empty
        self.used = True
        return self.message


class _FinishedWorker:
    def join(self, timeout: float | None = None) -> None:
        return None

    def is_alive(self) -> bool:
        return False


@pytest.mark.parametrize(
    "pages",
    [
        [
            {
                "page_index": 0,
                "page_number": 1,
                "text": "orphan page",
                "char_count": 11,
                "line_count": 1,
            }
        ],
        [],
    ],
)
def test_document_pages_missing_document_fails_pump_cleanly(
    tmp_path,
    monkeypatch,
    pages,
):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    with studio_db.get_connection():
        pass

    state = _JobState("job-missing-doc", "missing-doc", "kb_scope")
    queue = _OneMessageQueue(
        {
            "type": "document_pages",
            "pages": pages,
        }
    )

    _pump(state, _FinishedWorker(), queue)

    assert state.status == "failed"
    assert state.error is not None
    assert "document was removed before ingestion finished" in state.error
