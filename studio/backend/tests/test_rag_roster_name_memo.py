# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The roster's cleaned document names are cached, with the same text as a fresh cleanup."""

import asyncio

import pytest

from core.rag import store

TOOLS = [{"type": "function", "function": {"name": "search_knowledge_base"}}]

NAMES = [
    "syllabus.pdf",
    "  spaced \t\n name  .pdf",
    'quote " and back\\slash.txt',
    "‮evil‬.pdf",
    "⁦isolate⁩.md",
    "line\r\nbreak.pdf",
    "x" * 300 + ".pdf",
    "\U0001f600" * 130,
    "研究資料.pdf",
    "",
    "   ",
]


@pytest.fixture
def roster_name():
    from routes import inference

    inference._roster_name.cache_clear()
    yield inference._roster_name
    inference._roster_name.cache_clear()


def test_cached_names_match_a_fresh_cleanup(roster_name):
    for raw in [*NAMES, None, 7, 7.0, True, b"bytes.pdf"]:
        assert roster_name(raw) == roster_name(raw) == roster_name.__wrapped__(raw)
    assert [roster_name(1), roster_name(1.0), roster_name(True)] == ["1", "1.0", "True"]


def test_each_filename_is_cleaned_once_across_turns(roster_name, rag_conn):
    from routes import inference

    for i in range(60):
        store.create_document(
            rag_conn,
            scope = "project_p1",
            filename = f"doc {i:03d}.pdf",
            sha256 = f"d{i}",
            document_id = f"d{i}",
            status = "completed",
        )
    rag_conn.execute("UPDATE documents SET num_chunks = 3")
    rag_conn.commit()

    def nudge():
        return asyncio.run(inference._apply_rag_nudge("", TOOLS, rag_scope = {"project_id": "p1"}))

    first = nudge()
    assert "doc 059.pdf" in first and "more" in first
    assert roster_name.cache_info().misses == 60
    assert nudge() == first
    assert roster_name.cache_info().misses == 60
