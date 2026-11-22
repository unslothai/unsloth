# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sqlite3

import pytest


@pytest.mark.skipif(
    not hasattr(sqlite3.Connection, "enable_load_extension"),
    reason = "this interpreter's sqlite3 cannot load sqlite-vec",
)
def test_list_knowledge_bases_counts_only_completed_documents(rag_home, rag_conn):
    from core.rag import store
    from routes import rag as rag_routes

    kb_id = store.create_kb(rag_conn, name = "notes")
    empty_id = store.create_kb(rag_conn, name = "empty")
    for i, status in enumerate(["completed", "completed", "running", "pending", "failed"]):
        store.create_document(
            rag_conn,
            scope = store.kb_scope(kb_id),
            kb_id = kb_id,
            filename = f"doc{i}.txt",
            sha256 = f"sha{i}",
            status = status,
        )

    rows = {
        kb["id"]: kb for kb in rag_routes.list_knowledge_bases(subject = "tester")["knowledgeBases"]
    }
    assert (rows[kb_id]["documentCount"], rows[kb_id]["indexedCount"]) == (5, 2)
    assert (rows[empty_id]["documentCount"], rows[empty_id]["indexedCount"]) == (0, 0)
