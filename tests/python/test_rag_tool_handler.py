"""Unit tests for the `search_knowledge_base` tool handler (Phase 4)."""

import sys
from pathlib import Path
from unittest.mock import patch

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

    def _stub_retrieve(scope, query, *args, **kwargs):
        captured["scope"] = scope
        return []

    with patch("core.rag.retrieval.retrieve_hybrid", _stub_retrieve):
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

    def _stub_retrieve(scope, query, *args, **kwargs):
        captured["scope"] = scope
        return []

    with patch("core.rag.retrieval.retrieve_hybrid", _stub_retrieve):
        tool.search_knowledge_base(
            query = "x",
            scope_thread_id = "thread-xyz",
        )

    assert captured["scope"].startswith("thread_")


def test_empty_results_message_is_user_facing():
    from core.rag.tool import _format_hits_for_llm

    result = _format_hits_for_llm([])
    assert "No matching chunks" in result


def test_format_hits_produces_fenced_chunks():
    from core.rag.tool import _format_hits_for_llm

    hits = [
        {
            "filename": "alpha.pdf",
            "page_number": 3,
            "text": "first body",
            "score": 0.78,
            "chunk_index": 12,
            "token_count": 42,
        },
        {
            "filename": "beta.md",
            "page_number": None,
            "text": "second body",
            "score": 0.61,
        },
    ]
    result = _format_hits_for_llm(hits)
    assert (
        '<chunk id="1" source="alpha.pdf" page="3" chunk_index="12" tokens="42">'
        in result
    )
    assert 'chunk_index="12"' in result
    assert 'tokens="42"' in result
    assert "first body\n</chunk>" in result
    assert '<chunk id="2" source="beta.md">' in result
    # Blank line between blocks so the model can scan them.
    assert "</chunk>\n\n<chunk" in result


def test_format_hits_handles_unknown_source():
    from core.rag.tool import _format_hits_for_llm

    hits = [{"filename": None, "page_number": None, "text": "orphan"}]
    result = _format_hits_for_llm(hits)
    assert '<chunk id="1" source="unknown">' in result
    assert "\norphan\n</chunk>" in result


def test_format_hits_offsets_ids_by_start_id():
    from core.rag.tool import _format_hits_for_llm

    hits = [
        {"filename": "a.pdf", "text": "first"},
        {"filename": "b.pdf", "text": "second"},
    ]
    result = _format_hits_for_llm(hits, start_id = 5)
    assert '<chunk id="6"' in result
    assert '<chunk id="7"' in result
    assert '<chunk id="1"' not in result


def test_format_hits_emits_image_url_for_image_kind():
    from core.rag.tool import _format_hits_for_llm

    hits = [
        {
            "filename": "paper.pdf",
            "text": "Figure 1 shows a bar chart of X over time.",
            "kind": "image",
            "image_path": "/abs/path/images/doc-123/img-0007.png",
            "document_id": "doc-123",
            "page_number": 5,
        }
    ]
    result = _format_hits_for_llm(hits)
    assert 'kind="image"' in result
    assert 'image_url="/api/rag/images/doc-123/img-0007.png"' in result
    assert "Figure 1 shows a bar chart" in result


def test_format_hits_escapes_xml_in_source():
    from core.rag.tool import _format_hits_for_llm

    hits = [{"filename": 'weird"name<.pdf', "text": "body"}]
    result = _format_hits_for_llm(hits)
    assert 'source="weird&quot;name&lt;.pdf"' in result


def test_tool_spec_shape_is_openai_compatible():
    from core.rag.tool import SEARCH_KNOWLEDGE_BASE_TOOL

    assert SEARCH_KNOWLEDGE_BASE_TOOL["type"] == "function"
    fn = SEARCH_KNOWLEDGE_BASE_TOOL["function"]
    assert fn["name"] == "search_knowledge_base"
    assert "query" in fn["parameters"]["required"]
    assert "top_k" in fn["parameters"]["properties"]
    # Description hints when to call; don't lock the exact wording.
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
        **kwargs,
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
    assert "web_search" in names  # regression: others must stay
    assert "python" in names
