# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import uuid

import storage.studio_db as studio_db


def _uid() -> str:
    return str(uuid.uuid4())


def test_locator_schema_is_additive_and_nullable(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    with studio_db.get_connection() as conn:
        chunk_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(rag_chunks)")
        }
        assert {
            "source_page_index",
            "page_char_start",
            "page_char_end",
            "line_start",
            "line_end",
        }.issubset(chunk_cols)

        page_cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(rag_document_pages)")
        }
        assert {
            "document_id",
            "page_index",
            "page_number",
            "text",
            "char_count",
            "line_count",
        }.issubset(page_cols)

        kb_id = _uid()
        doc_id = _uid()
        chunk_id = _uid()
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
            (doc_id, kb_id, "old.pdf", "application/pdf", "old.pdf", 1_700_000_001),
        )
        conn.execute(
            """
            INSERT INTO rag_chunks
            (id, document_id, chunk_index, text, token_count, page_number)
            VALUES (?, ?, 0, ?, 3, 1)
            """,
            (chunk_id, doc_id, "legacy chunk"),
        )

        row = conn.execute(
            """
            SELECT source_page_index, page_char_start, page_char_end,
                   line_start, line_end
            FROM rag_chunks WHERE id = ?
            """,
            (chunk_id,),
        ).fetchone()
        assert dict(row) == {
            "source_page_index": None,
            "page_char_start": None,
            "page_char_end": None,
            "line_start": None,
            "line_end": None,
        }
