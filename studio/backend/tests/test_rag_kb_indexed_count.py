# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


def test_list_knowledge_bases_counts_only_completed_documents(rag_home, rag_conn):
    from core.rag import store
    from routes import rag as rag_routes

    kb_id = store.create_kb(rag_conn, name = "notes")
    empty_id = store.create_kb(rag_conn, name = "empty")
    statuses = [
        ("completed", 3),
        ("completed", 1),
        ("completed", 0),
        ("running", None),
        ("pending", None),
        ("failed", None),
    ]
    for i, (status, chunks) in enumerate(statuses):
        doc_id = store.create_document(
            rag_conn,
            scope = store.kb_scope(kb_id),
            kb_id = kb_id,
            filename = f"doc{i}.txt",
            sha256 = f"sha{i}",
            status = status,
        )
        if chunks is not None:
            store.set_document_status(rag_conn, doc_id, status, num_chunks = chunks)

    rows = {
        kb["id"]: kb for kb in rag_routes.list_knowledge_bases(subject = "tester")["knowledgeBases"]
    }
    counts = lambda kb: (kb["documentCount"], kb["indexedCount"], kb["indexingCount"])
    # The zero-chunk row and the failed row are neither embedded nor still embedding.
    assert counts(rows[kb_id]) == (6, 2, 2)
    assert counts(rows[empty_id]) == (0, 0, 0)
