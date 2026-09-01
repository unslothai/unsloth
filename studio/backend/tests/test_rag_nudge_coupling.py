# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for #9947: Project Sources RAG vs web_search coupling."""

import asyncio

import pytest

from core.inference import tools as inf_tools
from routes import inference

TOOLS = [{"type": "function", "function": {"name": "search_knowledge_base"}}]
TOOLS_WITH_WEB = TOOLS + [{"type": "function", "function": {"name": "web_search"}}]
TOOLS_WITH_RESEARCH = TOOLS + [{"type": "function", "function": {"name": "deep_research"}}]

TOOLS_WITH_MCP_BROWSER = TOOLS + [
    {"type": "function", "function": {"name": "mcp__browser__browser_navigate"}}
]

TOOLS_WITH_MCP_FILESYSTEM = TOOLS + [
    {
        "type": "function",
        "function": {
            "name": "mcp__filesystem__read_file",
            "description": "Read a local file",
        },
    }
]
RAG_SCOPE = {"project_id": "p1"}


def _rag_nudge(*, nudge: str, tools: list[dict], rag_scope, max_tool_calls = None) -> str:
    return asyncio.run(
        inference._apply_rag_nudge(
            nudge,
            tools,
            rag_scope = rag_scope,
            max_tool_calls = max_tool_calls,
        )
    )


@pytest.fixture(autouse = True)
def _empty_roster(monkeypatch):
    async def _no_roster(_rag_scope, *, max_bytes = inference._RAG_ROSTER_MAX_BYTES):
        return ""

    monkeypatch.setattr(inference, "_rag_roster_sentence", _no_roster)


def test_rag_only_nudge_includes_closed_corpus_guidance():
    out = _rag_nudge(nudge = "", tools = TOOLS, rag_scope = RAG_SCOPE)
    assert inference._RAG_GROUNDING_NUDGE in out
    assert inference._RAG_CLOSED_CORPUS_NUDGE in out
    assert inference._RAG_WEB_SEARCH_PRIORITY_NUDGE not in out
    assert "call search_knowledge_base before answering" in out


# The three scopes the composer can emit. A thread attachment and a selected knowledge
# base are closed corpora in the same sense a project is, so the guidance must not be
# gated on project_id.
ALL_SCOPES = [{"project_id": "p1"}, {"thread_id": "t1"}, {"kb_id": "k1"}]


@pytest.mark.parametrize("rag_scope", ALL_SCOPES)
def test_rag_and_web_nudge_discourages_automatic_web_fallback(rag_scope):
    """#9947 symptom (b) reached thread and KB scopes too, not just projects."""
    out = _rag_nudge(nudge = "", tools = TOOLS_WITH_WEB, rag_scope = rag_scope)
    assert inference._RAG_GROUNDING_NUDGE in out
    assert inference._RAG_WEB_SEARCH_PRIORITY_NUDGE in out
    assert inference._RAG_CLOSED_CORPUS_NUDGE not in out


def test_deep_research_skips_closed_corpus_nudge():
    out = _rag_nudge(nudge = "", tools = TOOLS_WITH_RESEARCH, rag_scope = RAG_SCOPE)
    assert inference._RAG_GROUNDING_NUDGE in out
    assert inference._RAG_CLOSED_CORPUS_NUDGE not in out
    assert inference._RAG_WEB_SEARCH_PRIORITY_NUDGE not in out


def test_closed_corpus_nudge_allows_explicit_mcp_browser_requests():
    out = _rag_nudge(nudge = "", tools = TOOLS_WITH_MCP_BROWSER, rag_scope = RAG_SCOPE)
    assert inference._RAG_GROUNDING_NUDGE in out
    assert inference._RAG_CLOSED_CORPUS_NUDGE in out
    assert "explicitly requests another enabled tool" in out
    assert "Do not search the public internet" not in out
    assert inference._RAG_WEB_SEARCH_PRIORITY_NUDGE not in out


def test_non_web_mcp_keeps_closed_corpus_nudge():
    out = _rag_nudge(nudge = "", tools = TOOLS_WITH_MCP_FILESYSTEM, rag_scope = RAG_SCOPE)
    assert inference._RAG_GROUNDING_NUDGE in out
    assert inference._RAG_CLOSED_CORPUS_NUDGE in out
    assert inference._RAG_WEB_SEARCH_PRIORITY_NUDGE not in out


def test_external_provider_routes_apply_rag_nudge_before_streaming():
    import inspect

    src = inspect.getsource(inference._proxy_to_external_provider)
    codex_branch = src.index('if provider_type == "openai_codex":')
    codex_return = src.index("return StreamingResponse(", codex_branch)
    codex_rag_nudge = src.index("_apply_rag_nudge", codex_branch)
    external_tool_loop = src.index("stream_with_studio_tools", codex_return)
    external_rag_nudge = src.index("_apply_rag_nudge", codex_return)
    assert codex_rag_nudge < codex_return
    assert external_rag_nudge < external_tool_loop


def test_rag_nudge_unchanged_without_scope():
    assert _rag_nudge(nudge = "keep", tools = TOOLS, rag_scope = None) == "keep"
    assert _rag_nudge(nudge = "keep", tools = [], rag_scope = RAG_SCOPE) == "keep"


def test_rag_nudge_unchanged_when_tool_budget_is_zero():
    assert _rag_nudge(nudge = "keep", tools = TOOLS, rag_scope = RAG_SCOPE, max_tool_calls = 0) == "keep"


def _fake_search_factory(captured):
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

    return fake_search


@pytest.fixture
def _stub_retrieval(monkeypatch):
    """Retrieval that records its scope, so a test can assert it never ran."""
    import storage.rag_db as rag_db
    from core.rag import tool as rag_tool

    captured: dict = {}
    monkeypatch.setattr(rag_db, "RAG_AVAILABLE", True)
    monkeypatch.setattr(rag_tool, "search_for_autoinject", _fake_search_factory(captured))
    monkeypatch.setattr(rag_tool, "whole_document_context", lambda **_kw: None)
    return captured


CONVERSATION = [{"role": "user", "content": "how many minutes"}]


def test_build_rag_autoinject_project_scope_runs_without_an_explicit_flag(_stub_retrieval):
    """#9947: with the Search pill off, a project chat must still pre-retrieve."""
    result = inf_tools.build_rag_autoinject(CONVERSATION, {"project_id": "p1"})
    assert result is not None
    assert "project hit" in result["messages"][-1]["content"]
    assert _stub_retrieval.get("scope_project_id") == "p1"


def test_build_rag_autoinject_project_scope_honors_explicit_off(_stub_retrieval):
    """Auto-retrieve is a separate control from the Search pill, and its own UI says
    "On and Off force it either way". A run persisted by an older build carries the same
    explicit flag, so the override must not reach back and turn it on."""
    result = inf_tools.build_rag_autoinject(
        CONVERSATION,
        {"project_id": "p1", "autoinject": False},
    )
    assert result is None
    assert _stub_retrieval == {}


@pytest.mark.parametrize("value", ["off", "false", "no", "0", "", "  OFF  ", 0])
def test_build_rag_autoinject_treats_legacy_off_scalars_as_off(_stub_retrieval, value):
    """A cached bundle, or a run stored before the field became a boolean, can send the
    mode string. `"off"` is a non-empty string, so bare truthiness read it as ON."""
    result = inf_tools.build_rag_autoinject(
        CONVERSATION,
        {"project_id": "p1", "autoinject": value},
    )
    assert result is None
    assert _stub_retrieval == {}


@pytest.mark.parametrize("value", ["on", "auto", "true", "1", 1, True])
def test_build_rag_autoinject_treats_legacy_on_scalars_as_on(_stub_retrieval, value):
    result = inf_tools.build_rag_autoinject(
        CONVERSATION,
        {"project_id": "p1", "autoinject": value},
    )
    assert result is not None
    assert _stub_retrieval.get("scope_project_id") == "p1"


def test_build_rag_autoinject_leaves_thread_scopes_to_the_caller(_stub_retrieval):
    """The project override is deliberately project-only: a thread attachment keeps the
    caller's flag so the whole-doc fallback runs instead of one combined search."""
    result = inf_tools.build_rag_autoinject(
        CONVERSATION,
        {"thread_id": "t1", "project_id": "p1", "autoinject": False, "whole_doc": False},
    )
    assert result is None
    assert _stub_retrieval == {}


def test_conditional_nudge_is_charged_to_the_roster_budget(monkeypatch):
    """`_RAG_ROSTER_MAX_BYTES` bounds what an unevictable instruction costs a small local
    model, so text added beside the roster has to shrink the roster, not the window."""
    seen = {}

    async def _record(_rag_scope, *, max_bytes = inference._RAG_ROSTER_MAX_BYTES):
        seen["max_bytes"] = max_bytes
        return ""

    monkeypatch.setattr(inference, "_rag_roster_sentence", _record)

    _rag_nudge(nudge = "", tools = TOOLS, rag_scope = RAG_SCOPE)
    closed_cost = len((" " + inference._RAG_CLOSED_CORPUS_NUDGE).encode("utf-8"))
    assert seen["max_bytes"] == inference._RAG_ROSTER_MAX_BYTES - closed_cost

    _rag_nudge(nudge = "", tools = TOOLS_WITH_WEB, rag_scope = RAG_SCOPE)
    web_cost = len((" " + inference._RAG_WEB_SEARCH_PRIORITY_NUDGE).encode("utf-8"))
    assert seen["max_bytes"] == inference._RAG_ROSTER_MAX_BYTES - web_cost

    _rag_nudge(nudge = "", tools = TOOLS_WITH_RESEARCH, rag_scope = RAG_SCOPE)
    assert seen["max_bytes"] == inference._RAG_ROSTER_MAX_BYTES
