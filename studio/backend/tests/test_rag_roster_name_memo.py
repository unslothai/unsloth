# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The roster's cleaned document names are memoized, with the same text as a fresh cleanup."""

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
def fresh_memo(monkeypatch):
    from routes import inference
    monkeypatch.setattr(inference, "_ROSTER_NAME_MEMO", {})
    return inference


def test_memoized_names_match_a_fresh_cleanup(fresh_memo):
    inference = fresh_memo
    for raw in [*NAMES, None, 7, b"bytes.pdf"]:
        first = inference._roster_name(raw)
        again = inference._roster_name(raw)
        assert first == again == inference._roster_name_uncached(raw)
    assert set(inference._ROSTER_NAME_MEMO) == set(NAMES)


def test_each_filename_is_cleaned_once_across_turns(fresh_memo, rag_conn, monkeypatch):
    inference = fresh_memo
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
    calls = []
    real = inference._roster_name_uncached
    monkeypatch.setattr(
        inference, "_roster_name_uncached", lambda raw: calls.append(raw) or real(raw)
    )

    def nudge():
        return asyncio.run(inference._apply_rag_nudge("", TOOLS, rag_scope = {"project_id": "p1"}))

    first = nudge()
    assert "doc 059.pdf" in first and "more" in first
    assert sorted(calls) == [f"doc {i:03d}.pdf" for i in range(60)]
    calls.clear()
    assert nudge() == first
    assert calls == []


def test_the_memo_stays_bounded(fresh_memo, monkeypatch):
    inference = fresh_memo
    monkeypatch.setattr(inference, "_ROSTER_NAME_MEMO_MAX", 8)
    for i in range(50):
        assert inference._roster_name(f"f{i}.pdf") == f"f{i}.pdf"
        assert 0 < len(inference._ROSTER_NAME_MEMO) <= 8
