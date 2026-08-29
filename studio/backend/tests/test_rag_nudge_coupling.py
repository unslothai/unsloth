# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for #9947: Project Sources RAG vs web_search coupling."""

import asyncio

import pytest

from core.inference import tools as inf_tools
from routes import inference

TOOLS = [{"type": "function", "function": {"name": "search_knowledge_base"}}]
TOOLS_WITH_WEB = TOOLS + [{"type": "function", "function": {"name": "web_search"}}]
RAG_SCOPE = {"project_id": "p1"}


def _rag_nudge(*, nudge: str, tools: list[dict], rag_scope) -> str:
    return asyncio.run(inference._apply_rag_nudge(nudge, tools, rag_scope = rag_scope))


@pytest.fixture(autouse = True)
def _empty_roster(monkeypatch):
    async def _no_roster(_rag_scope):
        return ""

    monkeypatch.setattr(inference, "_rag_roster_sentence", _no_roster)


def test_rag_only_nudge_includes_closed_corpus_guidance():
    out = _rag_nudge(nudge = "", tools = TOOLS, rag_scope = RAG_SCOPE)
    assert inference._RAG_GROUNDING_NUDGE in out
    assert inference._RAG_CLOSED_CORPUS_NUDGE in out
    assert inference._RAG_WEB_SEARCH_PRIORITY_NUDGE not in out
    assert "call search_knowledge_base before answering" in out


def test_rag_and_web_nudge_discourages_automatic_web_fallback():
    out = _rag_nudge(nudge = "", tools = TOOLS_WITH_WEB, rag_scope = RAG_SCOPE)
    assert inference._RAG_GROUNDING_NUDGE in out
    assert inference._RAG_WEB_SEARCH_PRIORITY_NUDGE in out
    assert inference._RAG_CLOSED_CORPUS_NUDGE not in out


def test_rag_nudge_unchanged_without_scope():
    assert _rag_nudge(nudge = "keep", tools = TOOLS, rag_scope = None) == "keep"
    assert _rag_nudge(nudge = "keep", tools = [], rag_scope = RAG_SCOPE) == "keep"


def test_build_rag_autoinject_project_scope_runs_when_autoinject_off(monkeypatch):
    import storage.rag_db as rag_db
    from core.rag import tool as rag_tool

    captured = {}

    def fake_search(**kw):
        captured.update(kw)
        return (
            "project hit",
            [
                {
                    "citationId": 1,
                    "chunkId": "pj:0",
                    "documentId": "pj",
                    "filename": "project.txt",
                    "page": None,
                    "text": "fourteen minutes",
                    "score": 0.95,
                }
            ],
        )

    monkeypatch.setattr(rag_db, "RAG_AVAILABLE", True)
    monkeypatch.setattr(rag_tool, "search_for_autoinject", fake_search)
    result = inf_tools.build_rag_autoinject(
        [{"role": "user", "content": "how many minutes"}],
        {"project_id": "p1", "autoinject": False},
    )
    assert result is not None
    assert "project hit" in result["messages"][-1]["content"]
    assert captured.get("scope_project_id") == "p1"
