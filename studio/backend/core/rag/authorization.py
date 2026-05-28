# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Subject-scoped authorization for RAG document preview routes.

Used by `/api/rag/documents/{document_id}/file` and
`/api/rag/documents/{document_id}/preview-target` to enforce that the
current authenticated subject is allowed to see a given document and
chunk. Existence and authorization failures collapse to a single 404
so the API does not leak document IDs to a non-owner.
"""

from __future__ import annotations

import sqlite3

from fastapi import HTTPException

from storage.studio_db import get_connection

_NOT_FOUND_DETAIL = "Document not found"


def document_for_subject_or_404(
    document_id: str,
    current_subject: str,
) -> sqlite3.Row:
    """Return the `rag_documents` row if `current_subject` may access it.

    Authorization rules:

    - KB documents: the document's KB must have
      `rag_knowledge_bases.owner_user_id == current_subject`. A KB with a
      NULL owner is not accessible through this helper (legacy pre-auth
      rows must be migrated or accessed via admin tooling).

    - Thread documents: thread-scoped RAG documents are gated by an
      explicit single-user invariant for Studio's current release. The
      `chat_threads` table does not yet carry an `owner_user_id` column,
      so we cannot bind a thread to a specific subject in the schema.
      The helper still requires (a) an authenticated subject (enforced
      by the route's `Depends(get_current_subject)`) and (b) that the
      referenced thread actually exists in `chat_threads`. A missing
      thread row collapses to 404 so a non-existent thread cannot
      silently grant access through a dangling `thread_id`.
      # TODO(thread-owner): once `chat_threads.owner_user_id` exists,
      # join through it the same way KB documents do and drop the
      # single-user invariant. Update the test
      # `tests/test_rag_authorization.py::test_thread_doc_other_user_404`
      # to assert per-user isolation rather than thread existence.

    Both not-found and not-authorized raise `HTTPException(404)` with the
    same detail string. Callers must NOT distinguish the two cases in
    their response, to avoid leaking document existence to a non-owner.

    Returns the document row so the caller can read `stored_path`,
    `filename`, `content_type`, etc. without re-querying.
    """
    if not document_id or not current_subject:
        raise HTTPException(status_code = 404, detail = _NOT_FOUND_DETAIL)

    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM rag_documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code = 404, detail = _NOT_FOUND_DETAIL)

        kb_id = row["kb_id"]
        thread_id = row["thread_id"]

        if kb_id is not None:
            owner_row = conn.execute(
                "SELECT owner_user_id FROM rag_knowledge_bases WHERE id = ?",
                (kb_id,),
            ).fetchone()
            if owner_row is None:
                raise HTTPException(status_code = 404, detail = _NOT_FOUND_DETAIL)
            owner = owner_row["owner_user_id"]
            if owner is None or owner != current_subject:
                raise HTTPException(status_code = 404, detail = _NOT_FOUND_DETAIL)
            return row

        if thread_id is not None:
            # Single-user invariant (see TODO above). We require the
            # thread row to exist; an unknown thread_id is treated as
            # not-found, not as silent grant.
            thread_row = conn.execute(
                "SELECT id FROM chat_threads WHERE id = ?",
                (thread_id,),
            ).fetchone()
            if thread_row is None:
                raise HTTPException(status_code = 404, detail = _NOT_FOUND_DETAIL)
            return row

        # Documents must belong to either a KB or a thread (DB CHECK
        # constraint enforces XOR on insert); a row that satisfies
        # neither is corrupt — treat as 404.
        raise HTTPException(status_code = 404, detail = _NOT_FOUND_DETAIL)


def chunk_belongs_to_document(chunk_id: str, document_id: str) -> bool:
    """True iff `chunk_id` exists in `rag_chunks` for `document_id`.

    Used by `/preview-target?chunk_id=...` after the caller has
    already established subject authorization for `document_id`. Does
    NOT perform authorization itself: callers MUST call
    `document_for_subject_or_404(document_id, ...)` first, otherwise a
    valid `chunk_id` from another subject's document would leak via a
    `True` return.
    """
    if not chunk_id or not document_id:
        return False

    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM rag_chunks WHERE id = ? AND document_id = ?",
            (chunk_id, document_id),
        ).fetchone()
    return row is not None
