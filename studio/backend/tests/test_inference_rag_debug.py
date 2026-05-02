# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import sys
import types as _types
from pathlib import Path


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


_unsloth_stub = _types.ModuleType("unsloth")
_unsloth_stub.FastLanguageModel = object
_unsloth_stub.FastVisionModel = object
sys.modules.setdefault("unsloth", _unsloth_stub)

_chat_templates_stub = _types.ModuleType("unsloth.chat_templates")
_chat_templates_stub.get_chat_template = lambda tokenizer, chat_template = None: tokenizer
sys.modules.setdefault("unsloth.chat_templates", _chat_templates_stub)

_transformers_stub = _types.ModuleType("transformers")
_transformers_stub.TextStreamer = object
sys.modules.setdefault("transformers", _transformers_stub)

_peft_stub = _types.ModuleType("peft")
_peft_stub.PeftModel = object
_peft_stub.PeftModelForCausalLM = object
sys.modules.setdefault("peft", _peft_stub)

_structlog_stub = _types.ModuleType("structlog")
sys.modules.setdefault("structlog", _structlog_stub)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)


from core.inference.inference import InferenceBackend


class _FakeWikiManager:
    def __init__(self, payload: dict):
        self.payload = payload
        self.last_query = ""

    def retrieve_context(self, query: str) -> dict:
        self.last_query = query
        return dict(self.payload)


def _make_backend(wiki_manager) -> InferenceBackend:
    backend = InferenceBackend.__new__(InferenceBackend)
    backend.wiki_manager = wiki_manager
    return backend


def test_get_rag_context_debug_includes_ranking_mode_and_selected_pages():
    payload = {
        "ranking_mode": "llm_rerank",
        "context_blocks": [
            {
                "page": "sources/alpha.md",
                "score": 0.91,
                "content": "alpha source block",
            },
            {
                "page": "analysis/alpha-summary.md",
                "score": 0.82,
                "content": "alpha analysis block",
            },
        ],
    }
    wiki_manager = _FakeWikiManager(payload)
    backend = _make_backend(wiki_manager)

    context, debug = backend._get_rag_context("alpha query", return_debug = True)

    assert "PAGE: sources/alpha.md" in context
    assert "PAGE: analysis/alpha-summary.md" in context
    assert debug["query"] == "alpha query"
    assert debug["ranking_mode"] == "llm_rerank"
    assert debug["selected_pages"] == [
        "sources/alpha.md",
        "analysis/alpha-summary.md",
    ]
    assert wiki_manager.last_query == "alpha query"


def test_get_rag_context_debug_handles_missing_wiki_manager():
    backend = _make_backend(None)

    context, debug = backend._get_rag_context("alpha query", return_debug = True)

    assert context == ""
    assert debug == {
        "query": "alpha query",
        "ranking_mode": "unknown",
        "selected_pages": [],
    }
