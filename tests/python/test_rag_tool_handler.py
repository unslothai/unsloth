"""Unit tests for the `search_knowledge_base` tool handler (Phase 4)."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_BACKEND = REPO_ROOT / "studio" / "backend"
if str(STUDIO_BACKEND) not in sys.path:
    sys.path.insert(0, str(STUDIO_BACKEND))


def _make_hit(chunk_id: str):
    """Minimal stand-in for retrieval.Hit — just needs .chunk_id."""

    class _Hit:
        pass

    h = _Hit()
    h.chunk_id = chunk_id
    h.score = 1.0
    h.kind = "text"
    h.document_id = None
    h.chunk_index = 0
    return h


def test_empty_query_returns_error():
    from core.rag.tool import search_knowledge_base

    result = search_knowledge_base(query = "", scope_thread_id = "t-1")
    assert result.startswith("Error:")
    assert "empty" in result.lower()


def test_missing_scope_returns_user_facing_hint():
    from core.rag.tool import search_knowledge_base

    result = search_knowledge_base(
        query = "anything",
        scope_kb_id = None,
        scope_thread_id = None,
    )
    assert "No knowledge base" in result
    assert "thread documents" in result


def test_kb_takes_precedence_over_thread():
    """When both kb_id and thread_id are passed, kb_id wins."""
    from core.rag import tool

    captured = {}

    def _stub_retrieve(scope, query, k):
        captured["scope"] = scope
        return []

    with patch.object(
        tool.__import__("core.rag.retrieval", fromlist = ["retrieve_hybrid"]),
        "retrieve_hybrid",
        _stub_retrieve,
    ):
        result = tool.search_knowledge_base(
            query = "x",
            scope_kb_id = "kb-abc",
            scope_thread_id = "thread-xyz",
        )

    assert captured["scope"].startswith("kb_")
    assert "kb-abc" in captured["scope"]
    assert "thread" not in captured["scope"].split("kb_")[1]


def test_thread_scope_when_only_thread_set():
    from core.rag import tool

    captured = {}

    def _stub_retrieve(scope, query, k):
        captured["scope"] = scope
        return []

    with patch.object(
        tool.__import__("core.rag.retrieval", fromlist = ["retrieve_hybrid"]),
        "retrieve_hybrid",
        _stub_retrieve,
    ):
        tool.search_knowledge_base(
            query = "x",
            scope_thread_id = "thread-xyz",
        )

    assert captured["scope"].startswith("thread_")


def test_empty_results_message_is_user_facing():
    from core.rag.tool import _format_hits_for_llm

    result = _format_hits_for_llm([])
    assert "No matching chunks" in result


def test_format_hits_produces_numbered_citations():
    from core.rag.tool import _format_hits_for_llm

    hits = [
        {"filename": "alpha.pdf", "page_number": 3, "text": "first body"},
        {"filename": "beta.md", "page_number": None, "text": "second body"},
    ]
    result = _format_hits_for_llm(hits)
    assert "[1] alpha.pdf (page 3): first body" in result
    assert "[2] beta.md: second body" in result
    # Each hit on its own paragraph so the LLM can cite cleanly.
    assert "\n\n" in result


def test_format_hits_handles_unknown_source():
    from core.rag.tool import _format_hits_for_llm

    hits = [{"filename": None, "page_number": None, "text": "orphan"}]
    result = _format_hits_for_llm(hits)
    assert "[1] unknown source: orphan" in result


def test_tool_spec_shape_is_openai_compatible():
    from core.rag.tool import SEARCH_KNOWLEDGE_BASE_TOOL

    assert SEARCH_KNOWLEDGE_BASE_TOOL["type"] == "function"
    fn = SEARCH_KNOWLEDGE_BASE_TOOL["function"]
    assert fn["name"] == "search_knowledge_base"
    assert "query" in fn["parameters"]["required"]
    assert "top_k" in fn["parameters"]["properties"]
    # Description should hint at when to call so the LLM picks it up
    # appropriately. Don't lock the exact wording.
    assert "documents" in fn["description"].lower()


def test_execute_tool_dispatches_to_search_knowledge_base():
    """tools.execute_tool should route 'search_knowledge_base' correctly."""
    from core.inference import tools

    called = {}

    def _stub(
        *,
        query,
        top_k = None,
        scope_kb_id = None,
        scope_thread_id = None,
        enable_rerank = False,
        reranker_model = None,
        default_top_k = 5,
        min_score = 0.0,
    ):
        called["query"] = query
        called["top_k"] = top_k
        called["scope_kb_id"] = scope_kb_id
        called["scope_thread_id"] = scope_thread_id
        called["enable_rerank"] = enable_rerank
        called["default_top_k"] = default_top_k
        called["min_score"] = min_score
        return "stub-result"

    with patch("core.rag.tool.search_knowledge_base", _stub):
        result = tools.execute_tool(
            "search_knowledge_base",
            {"query": "hello", "top_k": 7},
            tool_context = {
                "rag_scope": {
                    "kb_id": "kb-1",
                    "enable_rerank": True,
                    "default_top_k": 3,
                    "min_score": 0.35,
                }
            },
        )
    assert result == "stub-result"
    assert called["query"] == "hello"
    assert called["top_k"] == 7
    assert called["scope_kb_id"] == "kb-1"
    assert called["scope_thread_id"] is None
    assert called["enable_rerank"] is True
    assert called["default_top_k"] == 3
    assert called["min_score"] == 0.35


def test_execute_tool_handles_missing_tool_context():
    """tool_context=None should still dispatch without crashing."""
    from core.inference import tools

    def _stub(*, query, **_kwargs):
        return f"got: {query}"

    with patch("core.rag.tool.search_knowledge_base", _stub):
        result = tools.execute_tool(
            "search_knowledge_base",
            {"query": "ping"},
            tool_context = None,
        )
    assert result == "got: ping"


def test_all_tools_includes_rag():
    from core.inference.tools import ALL_TOOLS

    names = [t["function"]["name"] for t in ALL_TOOLS]
    assert "search_knowledge_base" in names
    assert "web_search" in names  # regression — we shouldn't have removed the others
    assert "python" in names
