# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Inference API routes for model loading and text generation.
"""

import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Any, Optional
import json
import structlog
from loggers import get_logger
import asyncio
import threading


import re as _re

# Model size extraction (shared with core/inference/llama_cpp.py)
from utils.models import extract_model_size_b as _extract_model_size_b


def _friendly_error(exc: Exception) -> str:
    """Extract a user-friendly message from known llama-server errors."""
    msg = str(exc)
    m = _re.search(
        r"request \((\d+) tokens?\) exceeds the available context size \((\d+) tokens?\)",
        msg,
    )
    if m:
        return (
            f"Message too long: {m.group(1)} tokens exceeds the {m.group(2)}-token "
            f"context window. Try increasing the Context Length in Model settings, "
            f"or shorten the conversation."
        )
    if "Lost connection to llama-server" in msg:
        return "Lost connection to the model server. It may have crashed -- try reloading the model."
    return "An internal error occurred"


# Add backend directory to path
backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Import backend functions
try:
    from core.inference import get_inference_backend
    from core.inference.llama_cpp import LlamaCppBackend
    from utils.models import ModelConfig
    from utils.inference import load_inference_config
    from utils.models.model_config import load_model_defaults
except ImportError:
    parent_backend = backend_path.parent / "backend"
    if str(parent_backend) not in sys.path:
        sys.path.insert(0, str(parent_backend))
    from core.inference import get_inference_backend
    from core.inference.llama_cpp import LlamaCppBackend
    from utils.models import ModelConfig
    from utils.inference import load_inference_config
    from utils.models.model_config import load_model_defaults

from models.inference import (
    LoadRequest,
    UnloadRequest,
    GenerateRequest,
    LoadResponse,
    UnloadResponse,
    InferenceStatusResponse,
    ChatCompletionRequest,
    ChatCompletionChunk,
    ChatCompletion,
    ChunkChoice,
    ChoiceDelta,
    CompletionChoice,
    CompletionMessage,
    CompletionUsage,
    ValidateModelRequest,
    ValidateModelResponse,
    RagContextDebugRequest,
    RagContextDebugResponse,
    RagContextSnippet,
    WikiArchiveRequest,
    WikiArchiveResponse,
    WikiIngestRequest,
    WikiIngestResponse,
    WikiEnrichRequest,
    WikiEnrichResponse,
    WikiRetryFallbackRequest,
    WikiRetryFallbackResponse,
    WikiQueryRequest,
    WikiQueryResponse,
    WikiLintResponse,
)
from auth.authentication import get_current_subject

import io
import wave
import base64
import numpy as np
from datetime import date as _date
import datetime as _datetime

from core.wiki.manager import WikiManager
from core.wiki.ingestor import WikiIngestor

router = APIRouter()

# Appended to tool-use nudge to discourage plan-without-action
_TOOL_ACTION_NUDGE = (
    " IMPORTANT: Always call tools directly -- never write code yourself."
    " Never describe what you plan to do -- just call the tool immediately."
    " For any code request, call the python tool. For any factual question, call web_search."
    " Do NOT output code blocks -- use the python tool instead."
)

# Regex for stripping leaked tool-call XML from assistant messages/stream
_TOOL_XML_RE = _re.compile(
    r"<tool_call>.*?</tool_call>|<function=\w+>.*?</function>",
    _re.DOTALL,
)
logger = get_logger(__name__)

_WIKI_VAULT_ROOT = Path(os.getenv("UNSLOTH_WIKI_VAULT", "/tmp/unsloth_wiki"))
_ROUTE_WIKI_MANAGER: Optional[WikiManager] = None
_ROUTE_WIKI_INGESTOR: Optional[WikiIngestor] = None
_RAG_MAX_PAGES = int(os.getenv("UNSLOTH_WIKI_RAG_MAX_PAGES", "3"))
_RAG_MAX_CHARS_PER_PAGE = int(os.getenv("UNSLOTH_WIKI_RAG_MAX_CHARS_PER_PAGE", "900"))
_RAG_MAX_TOTAL_CHARS = int(os.getenv("UNSLOTH_WIKI_RAG_MAX_TOTAL_CHARS", "4500"))
_RAG_LOG_INJECTED_CONTEXT = (
    os.getenv("UNSLOTH_WIKI_LOG_INJECTED_CONTEXT", "true").strip().lower()
    not in {"0", "false", "no", "off"}
)
try:
    _RAG_LOG_INJECTED_CONTEXT_MAX_CHARS = max(
        0,
        int(os.getenv("UNSLOTH_WIKI_LOG_INJECTED_CONTEXT_MAX_CHARS", "12000")),
    )
except ValueError:
    _RAG_LOG_INJECTED_CONTEXT_MAX_CHARS = 12000
_WIKI_LLM_MAX_TOKENS = int(os.getenv("UNSLOTH_WIKI_LLM_MAX_TOKENS", "1200"))
_WIKI_AUTO_LINT_EVERY_QUERY = int(os.getenv("UNSLOTH_WIKI_AUTO_LINT_EVERY", "10"))
try:
    _WIKI_AUTO_RETRY_FALLBACK_MAX_PAGES = max(
        0,
        int(os.getenv("UNSLOTH_WIKI_AUTO_RETRY_FALLBACK_ANALYSES_MAX_PAGES", "24")),
    )
except ValueError:
    _WIKI_AUTO_RETRY_FALLBACK_MAX_PAGES = 24
_WIKI_QUERY_RUN_COUNT = 0
_LAST_RAG_DEBUG: dict[str, Any] = {}
_CHAT_HISTORY_FLUSH_SECONDS = max(
    0,
    int(os.getenv("UNSLOTH_WIKI_CHAT_HISTORY_FLUSH_SECONDS", "600")),
)
_CHAT_HISTORY_PENDING_BLOCKS: list[str] = []
_CHAT_HISTORY_BUFFER_STARTED_AT: Optional[_datetime.datetime] = None
_CHAT_HISTORY_LOCK = threading.Lock()


def _loggable_rag_context(context: str) -> str:
    if _RAG_LOG_INJECTED_CONTEXT_MAX_CHARS <= 0:
        return context
    if len(context) <= _RAG_LOG_INJECTED_CONTEXT_MAX_CHARS:
        return context
    return (
        context[:_RAG_LOG_INJECTED_CONTEXT_MAX_CHARS].rstrip()
        + "\n...[truncated by UNSLOTH_WIKI_LOG_INJECTED_CONTEXT_MAX_CHARS]"
    )


def _wiki_llm_available() -> bool:
    try:
        if get_llama_cpp_backend().is_loaded:
            return True
    except Exception:
        pass
    try:
        backend = get_inference_backend()
        return bool(getattr(backend, "active_model_name", None))
    except Exception:
        return False


def _route_wiki_llm_stub(prompt: str) -> str:
    """Best-effort wiki LLM function using whichever model backend is active."""
    wants_structured_json = (
        "Return strict JSON with keys:" in prompt
        or "JSON repair assistant" in prompt
    )

    temp = 0.0 if wants_structured_json else 0.2
    top_p = 1.0 if wants_structured_json else 0.9
    top_k = 1 if wants_structured_json else 20
    min_p = 0.0

    try:
        llama_backend = get_llama_cpp_backend()
        if llama_backend.is_loaded:
            chunks = llama_backend.generate_chat_completion(
                messages = [{"role": "user", "content": prompt}],
                temperature = temp,
                top_p = top_p,
                top_k = top_k,
                min_p = min_p,
                max_tokens = _WIKI_LLM_MAX_TOKENS,
                repetition_penalty = 1.0,
                presence_penalty = 0.0,
                enable_thinking = False,
            )
            final = ""
            for chunk in chunks:
                if isinstance(chunk, str):
                    final = chunk
            if final.strip():
                return final.strip()
    except Exception as exc:
        logger.warning(f"GGUF wiki LLM call failed, falling back: {exc}")

    try:
        backend = get_inference_backend()
        if backend.active_model_name:
            out = ""
            for token in backend.generate_chat_response(
                messages = [{"role": "user", "content": prompt}],
                system_prompt = "",
                temperature = temp,
                top_p = top_p,
                top_k = (top_k if wants_structured_json else 40),
                min_p = min_p,
                max_new_tokens = _WIKI_LLM_MAX_TOKENS,
                repetition_penalty = 1.0,
                cancel_event = None,
            ):
                out += token
            if out.strip():
                return out.strip()
    except Exception as exc:
        logger.warning(f"Transformer wiki LLM call failed, falling back: {exc}")

    # Final fallback keeps ingestion resilient when no model is loaded.
    return prompt


def _get_route_wiki_components() -> tuple[WikiManager, WikiIngestor]:
    global _ROUTE_WIKI_MANAGER, _ROUTE_WIKI_INGESTOR
    if _ROUTE_WIKI_MANAGER is None or _ROUTE_WIKI_INGESTOR is None:
        _ROUTE_WIKI_MANAGER = WikiManager.create(_WIKI_VAULT_ROOT, _route_wiki_llm_stub)
        _ROUTE_WIKI_INGESTOR = WikiIngestor(_ROUTE_WIKI_MANAGER, _WIKI_VAULT_ROOT / "raw")
    return _ROUTE_WIKI_MANAGER, _ROUTE_WIKI_INGESTOR


def _ingest_pending_raw_files(max_files: int = 8) -> list[dict[str, Any]]:
    manager, ingestor = _get_route_wiki_components()
    raw_dir = _WIKI_VAULT_ROOT / "raw"
    sources_dir = _WIKI_VAULT_ROOT / "wiki" / "sources"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sources_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(
        [p for p in raw_dir.iterdir() if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    ingested = 0
    results: list[dict[str, Any]] = []
    for path in candidates:
        if ingested >= max_files:
            break
        if path.name.lower() in {".ds_store", "thumbs.db"} or path.name.startswith(".") or path.name.startswith("._"):
            continue
        if path.suffix.lower() not in {
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".pdf",
            ".txt",
            ".md",
            ".png",
            ".jpg",
            ".jpeg",
        }:
            continue

        slug = manager.engine._slug(path.stem)
        source_page = sources_dir / f"{slug}.md"
        if source_page.exists():
            continue

        result = ingestor.ingest_file(path, contributor="Unsloth Studio")
        if result:
            ingested += 1
            results.append({"source_path": str(path), "result": result})

    return results


def _extract_source_ref(source_page: Path) -> Optional[str]:
    text = source_page.read_text(encoding="utf-8", errors="ignore")
    match = _re.search(r"(?mi)^source_ref:\s*(.+?)\s*$", text)
    if not match:
        return None
    source_ref = match.group(1).strip()
    return source_ref or None


def _archive_stale_wiki_pages(
    *,
    dry_run: bool,
    keep_recent_chat: int,
    keep_recent_per_source: int,
) -> dict[str, Any]:
    sources_dir = _WIKI_VAULT_ROOT / "wiki" / "sources"
    archive_sources_dir = _WIKI_VAULT_ROOT / "wiki" / ".archive" / "sources"
    raw_dir = _WIKI_VAULT_ROOT / "raw"
    archive_raw_dir = raw_dir / ".archive"

    report: dict[str, Any] = {
        "dry_run": dry_run,
        "archive_dir": str(archive_sources_dir),
        "moved_count": 0,
        "moved_sources": [],
        "moved_raw": [],
        "errors": [],
    }

    if not sources_dir.exists():
        return report

    source_pages = sorted(
        [p for p in sources_dir.glob("*.md") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    to_archive: dict[Path, Optional[str]] = {}

    chat_pages = [p for p in source_pages if p.name.lower().startswith("chat-history-")]
    for p in chat_pages[keep_recent_chat:]:
        to_archive[p] = _extract_source_ref(p)

    grouped: dict[str, list[tuple[Path, Optional[str]]]] = {}
    non_chat_pages = [p for p in source_pages if p not in to_archive]
    for p in non_chat_pages:
        source_ref = _extract_source_ref(p)
        source_name = Path(source_ref).stem if source_ref else p.stem
        canonical = _re.sub(r"[^a-z0-9]+", "-", source_name.lower()).strip("-")
        canonical = canonical or p.stem.lower()
        grouped.setdefault(canonical, []).append((p, source_ref))

    for entries in grouped.values():
        entries.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
        for stale_page, source_ref in entries[keep_recent_per_source:]:
            to_archive[stale_page] = source_ref

    for stale_page, source_ref in sorted(
        to_archive.items(), key=lambda x: x[0].stat().st_mtime
    ):
        try:
            target_name = stale_page.name
            target = archive_sources_dir / target_name
            if target.exists():
                stamp = int(stale_page.stat().st_mtime)
                target = archive_sources_dir / f"{stale_page.stem}--{stamp}{stale_page.suffix}"

            if not dry_run:
                archive_sources_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(stale_page), str(target))

            report["moved_sources"].append(str(target))

            if source_ref:
                raw_path = Path(source_ref)
                if raw_path.exists() and raw_dir in raw_path.parents:
                    raw_target = archive_raw_dir / raw_path.name
                    if raw_target.exists():
                        stamp = int(raw_path.stat().st_mtime)
                        raw_target = archive_raw_dir / f"{raw_path.stem}--{stamp}{raw_path.suffix}"
                    if not dry_run:
                        archive_raw_dir.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(raw_path), str(raw_target))
                    report["moved_raw"].append(str(raw_target))
        except Exception as exc:
            report["errors"].append(f"{stale_page}: {exc}")

    report["moved_count"] = len(report["moved_sources"])
    return report


def _to_rag_debug_response(payload: dict[str, Any]) -> RagContextDebugResponse:
    source = str(payload.get("source", "live-query"))
    if source not in {"live-query", "last-request"}:
        source = "live-query"

    selected = [
        RagContextSnippet(
            page=str(item.get("page", "unknown")),
            score=float(item.get("score", 0.0)),
            snippet=str(item.get("snippet", "")),
        )
        for item in payload.get("selected", [])
    ]

    applied_limits = payload.get("applied_limits", {})
    if not isinstance(applied_limits, dict):
        applied_limits = {}

    return RagContextDebugResponse(
        query=str(payload.get("query", "")),
        source=source,
        wants_history=bool(payload.get("wants_history", False)),
        context=str(payload.get("context", "")),
        context_characters=int(payload.get("context_characters", 0)),
        pages_considered=int(payload.get("pages_considered", 0)),
        selected=selected,
        applied_limits={
            "max_pages": int(applied_limits.get("max_pages", _RAG_MAX_PAGES)),
            "max_chars_per_page": int(
                applied_limits.get("max_chars_per_page", _RAG_MAX_CHARS_PER_PAGE)
            ),
            "max_total_chars": int(
                applied_limits.get("max_total_chars", _RAG_MAX_TOTAL_CHARS)
            ),
        },
        generated_at=str(
            payload.get("generated_at", _datetime.datetime.now().isoformat(timespec="seconds"))
        ),
    )


def _get_route_rag_context(
    query: str,
    *,
    return_debug: bool = False,
    max_pages_override: Optional[int] = None,
    max_chars_per_page_override: Optional[int] = None,
    max_total_chars_override: Optional[int] = None,
    debug_source: str = "live-query",
) -> str | tuple[str, dict[str, Any]]:
    manager, _ = _get_route_wiki_components()
    query_lower = query.lower()
    max_pages = max_pages_override or _RAG_MAX_PAGES
    max_chars_per_page = max_chars_per_page_override or _RAG_MAX_CHARS_PER_PAGE
    max_total_chars = max_total_chars_override or _RAG_MAX_TOTAL_CHARS

    max_pages = max(1, min(max_pages, 32))
    max_chars_per_page = max(200, min(max_chars_per_page, 12000))
    max_total_chars = max(500, min(max_total_chars, 30000))

    wants_history = any(
        token in query_lower
        for token in (
            "conversation history",
            "chat history",
            "earlier conversation",
            "remember",
            "token",
            "previous message",
        )
    )

    result = manager.retrieve_context(
        query,
        max_pages=max(max_pages * 4, 12),
        max_chars_per_page=max(max_chars_per_page * 6, 6000),
    )
    blocks: list[dict] = result.get("context_blocks", [])

    query_terms = [
        t
        for t in _re.findall(r"[a-zA-Z0-9]{3,}", query_lower)
        if t
        not in {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "what",
            "when",
            "where",
            "who",
            "why",
            "how",
            "from",
            "into",
            "about",
            "tell",
            "please",
            "using",
            "wiki",
            "context",
            "only",
        }
    ]

    def _block_hit_score(block: dict) -> int:
        page = str(block.get("page", "")).lower()
        content = str(block.get("content", "")).lower()
        sample = content[:4000]
        if not query_terms:
            return 0
        score = 0
        for term in query_terms:
            score += page.count(term) * 3
            score += sample.count(term)
        return score

    if wants_history:
        chat_blocks = [b for b in blocks if "chat-history" in str(b.get("page", "")).lower()]
        non_chat = [b for b in blocks if "chat-history" not in str(b.get("page", "")).lower()]
        blocks = (chat_blocks + non_chat)[:max_pages]

        sources_dir = _WIKI_VAULT_ROOT / "wiki" / "sources"
        if sources_dir.exists():
            history_files = sorted(
                [p for p in sources_dir.glob("chat-history-*.md")],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            hint_terms = [
                t for t in _re.findall(r"[A-Za-z]{2,}-[A-Za-z0-9]{2,}", query) if len(t) >= 5
            ]

            selected: list[dict] = []
            for p in history_files:
                text = p.read_text(encoding="utf-8", errors="ignore")
                if hint_terms and not any(h.lower() in text.lower() for h in hint_terms):
                    continue
                selected.append(
                    {
                        "page": f"sources/{p.stem}.md",
                        "score": 1.0,
                        "content": text,
                    }
                )
                if len(selected) >= max_pages:
                    break

            if not selected:
                for p in history_files[:max_pages]:
                    selected.append(
                        {
                            "page": f"sources/{p.stem}.md",
                            "score": 1.0,
                            "content": p.read_text(encoding="utf-8", errors="ignore"),
                        }
                    )

            if selected:
                blocks = selected
    else:
        rerank_enabled = bool(manager.engine.cfg.ranking_llm_rerank_enabled)
        if rerank_enabled:
            # Preserve planner-selected order when LLM reranking is enabled.
            blocks = blocks[:max_pages]
        else:
            chat_blocks = [b for b in blocks if "chat-history" in str(b.get("page", "")).lower()]
            non_chat_blocks = [b for b in blocks if "chat-history" not in str(b.get("page", "")).lower()]

            best_non_chat_hit = max((_block_hit_score(b) for b in non_chat_blocks), default = 0)
            candidate_chat_hits = [b for b in chat_blocks if _block_hit_score(b) > 0]
            candidate_chat_hits.sort(key = _block_hit_score, reverse = True)

            # For non-history queries, still prefer non-chat pages by default, but if
            # there is no lexical hit at all, keep matching chat-history pages instead
            # of dropping them unconditionally.
            blocks = list(non_chat_blocks)
            if best_non_chat_hit <= 0 and candidate_chat_hits:
                blocks = candidate_chat_hits + blocks

            if "resume" in query_lower or ".pdf" in query_lower:
                resume_blocks = [b for b in blocks if "resume" in str(b.get("page", "")).lower()]
                other_blocks = [b for b in blocks if "resume" not in str(b.get("page", "")).lower()]
                blocks = (resume_blocks + other_blocks)[:max_pages]
            else:
                blocks = blocks[:max_pages]

    if not blocks and not wants_history:
        sources_dir = _WIKI_VAULT_ROOT / "wiki" / "sources"
        if sources_dir.exists() and ("resume" in query_lower or ".pdf" in query_lower or "document" in query_lower):
            source_candidates = sorted(
                [p for p in sources_dir.glob("*.md") if "chat-history" not in p.name.lower()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            resume_first = [p for p in source_candidates if "resume" in p.name.lower()]
            ordered = resume_first + [p for p in source_candidates if p not in resume_first]
            for p in ordered[:max_pages]:
                blocks.append(
                    {
                        "page": f"sources/{p.stem}.md",
                        "score": 1.0,
                        "content": p.read_text(encoding="utf-8", errors="ignore"),
                    }
                )

    def _select_snippet(content: str) -> str:
        content = str(content)
        if len(content) <= max_chars_per_page:
            return content

        terms = [t for t in _re.findall(r"[a-zA-Z0-9]{4,}", query_lower) if t not in {"using", "wiki", "context", "only", "from", "what", "which", "that"}]
        lowered = content.lower()
        for term in terms:
            idx = lowered.find(term)
            if idx >= 0:
                half = max_chars_per_page // 2
                start = max(0, idx - half)
                end = min(len(content), start + max_chars_per_page)
                start = max(0, end - max_chars_per_page)
                return content[start:end]

        return content[:max_chars_per_page]

    context_parts = []
    for block in blocks:
        page = block.get("page", "unknown")
        score = float(block.get("score", 0.0))
        content = _select_snippet(block.get("content", ""))
        context_parts.append(
            f"PAGE: {page}\n"
            f"SCORE: {score:.4f}\n"
            f"CONTENT:\n{content}"
        )
    context = "\n\n---\n\n".join(context_parts)
    if len(context) > max_total_chars:
        context = context[:max_total_chars].rstrip() + "\n..."

    debug_payload: dict[str, Any] = {
        "query": query,
        "source": debug_source,
        "wants_history": wants_history,
        "context": context,
        "context_characters": len(context),
        "pages_considered": len(blocks),
        "selected": [
            {
                "page": str(block.get("page", "unknown")),
                "score": float(block.get("score", 0.0)),
                "snippet": _select_snippet(block.get("content", "")),
            }
            for block in blocks
        ],
        "applied_limits": {
            "max_pages": max_pages,
            "max_chars_per_page": max_chars_per_page,
            "max_total_chars": max_total_chars,
        },
        "generated_at": _datetime.datetime.now().isoformat(timespec="seconds"),
    }

    if return_debug:
        return context, debug_payload
    return context


def _save_chat_history_to_route_wiki(messages: list[dict]) -> None:
    if not messages:
        return

    timestamp = _datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"## Chat Snapshot - {timestamp}\n"]
    for msg in messages:
        role = str(msg.get("role", "unknown")).capitalize()
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append(f"### {role}\n{content}\n")
    block = "\n".join(lines).strip()

    global _CHAT_HISTORY_BUFFER_STARTED_AT
    now = _datetime.datetime.now()
    with _CHAT_HISTORY_LOCK:
        _CHAT_HISTORY_PENDING_BLOCKS.append(block)
        if _CHAT_HISTORY_BUFFER_STARTED_AT is None:
            _CHAT_HISTORY_BUFFER_STARTED_AT = now

        should_flush = _CHAT_HISTORY_FLUSH_SECONDS == 0
        if (
            _CHAT_HISTORY_BUFFER_STARTED_AT is not None
            and not should_flush
            and (now - _CHAT_HISTORY_BUFFER_STARTED_AT).total_seconds()
            >= _CHAT_HISTORY_FLUSH_SECONDS
        ):
            should_flush = True

        if not should_flush:
            return

        filename = (
            f"chat_history_{_datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.md"
        )
        file_path = _WIKI_VAULT_ROOT / "raw" / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = "# Chat History Batch\n\n" + "\n\n---\n\n".join(_CHAT_HISTORY_PENDING_BLOCKS)

        try:
            file_path.write_text(payload, encoding="utf-8")
            _, ingestor = _get_route_wiki_components()
            ingestor.ingest_file(file_path, contributor="Unsloth Studio")
            _CHAT_HISTORY_PENDING_BLOCKS.clear()
            _CHAT_HISTORY_BUFFER_STARTED_AT = None
        except Exception as exc:
            logger.warning("Failed to flush buffered chat history: %s", exc)


# GGUF inference backend (llama-server)
_llama_cpp_backend = LlamaCppBackend()


def get_llama_cpp_backend() -> LlamaCppBackend:
    return _llama_cpp_backend


@router.post("/rag/debug/context", response_model = RagContextDebugResponse)
async def debug_rag_context(
    payload: RagContextDebugRequest,
    current_subject: str = Depends(get_current_subject),
):
    """Preview the exact wiki context pages/snippets selected for a query."""
    if payload.include_pending_raw:
        _ingest_pending_raw_files()

    _, debug_payload = _get_route_rag_context(
        payload.query,
        return_debug=True,
        max_pages_override=payload.max_pages,
        max_chars_per_page_override=payload.max_chars_per_page,
        max_total_chars_override=payload.max_total_chars,
        debug_source="live-query",
    )
    return _to_rag_debug_response(debug_payload)


@router.get("/rag/debug/last", response_model = RagContextDebugResponse)
async def debug_rag_last_request(
    current_subject: str = Depends(get_current_subject),
):
    """Return the most recent RAG context used by /chat/completions."""
    if not _LAST_RAG_DEBUG:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail = "No RAG debug record available yet.",
        )
    return _to_rag_debug_response(_LAST_RAG_DEBUG)


@router.post("/wiki/archive/stale", response_model = WikiArchiveResponse)
async def archive_stale_wiki_sources(
    payload: WikiArchiveRequest,
    current_subject: str = Depends(get_current_subject),
):
    """Archive older duplicate/wiki-history source pages to reduce noisy retrieval."""
    report = _archive_stale_wiki_pages(
        dry_run=payload.dry_run,
        keep_recent_chat=payload.keep_recent_chat,
        keep_recent_per_source=payload.keep_recent_per_source,
    )

    if report["moved_count"] and not payload.dry_run:
        manager, _ = _get_route_wiki_components()
        manager.engine._rebuild_index()

    return WikiArchiveResponse(
        dry_run=bool(report["dry_run"]),
        archive_dir=str(report["archive_dir"]),
        moved_count=int(report["moved_count"]),
        moved_sources=[str(x) for x in report["moved_sources"]],
        moved_raw=[str(x) for x in report["moved_raw"]],
        errors=[str(x) for x in report["errors"]],
    )


@router.post("/wiki/ingest", response_model = WikiIngestResponse)
async def wiki_ingest(
    payload: WikiIngestRequest,
    current_subject: str = Depends(get_current_subject),
):
    """Ingest a specific file or pending raw files into the maintained wiki."""
    _, ingestor = _get_route_wiki_components()

    if payload.source_path:
        source_path = Path(payload.source_path).expanduser()
        if not source_path.exists() or not source_path.is_file():
            raise HTTPException(
                status_code = status.HTTP_400_BAD_REQUEST,
                detail = f"File not found: {source_path}",
            )
        result = ingestor.ingest_file(source_path, contributor = "Unsloth Studio")
        if not result:
            raise HTTPException(
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail = f"Failed to ingest file: {source_path}",
            )
        return WikiIngestResponse(
            status = "ok",
            processed_files = 1,
            results = [{"source_path": str(source_path), "result": result}],
        )

    results = _ingest_pending_raw_files(max_files = payload.max_pending_raw_files)
    return WikiIngestResponse(
        status = "ok",
        processed_files = len(results),
        results = results,
    )


@router.post("/wiki/enrich", response_model = WikiEnrichResponse)
async def wiki_enrich(
    payload: WikiEnrichRequest,
    current_subject: str = Depends(get_current_subject),
):
    """Enrich wiki analysis pages by prepending an index-driven Enrichment section."""
    manager, _ = _get_route_wiki_components()

    # Keep manual enrichment behavior aligned with scheduled maintenance:
    # retry fallback pages first, then run enrichment.
    if _WIKI_AUTO_RETRY_FALLBACK_MAX_PAGES > 0:
        try:
            retry_report = manager.retry_fallback_analysis_pages(
                dry_run=payload.dry_run,
                max_analysis_pages=payload.max_analysis_pages,
            )
            logger.info(
                "Fallback-retry before /wiki/enrich: scanned=%d fallback_found=%d regenerated=%d still_fallback=%d",
                int(retry_report.get("scanned_pages", 0)),
                int(retry_report.get("fallback_pages_found", 0)),
                int(retry_report.get("regenerated_pages", 0)),
                int(retry_report.get("fallback_still", 0)),
            )
        except Exception as exc:
            logger.warning("Fallback-retry before /wiki/enrich failed: %s", exc)

    try:
        report = manager.enrich_analysis_pages(
            dry_run=payload.dry_run,
            max_analysis_pages=payload.max_analysis_pages,
            fill_gaps_from_web=payload.fill_gaps_from_web,
            max_web_gap_queries=payload.max_web_gap_queries,
        )
    except Exception as exc:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail = f"Wiki enrichment failed: {exc}",
        )

    return WikiEnrichResponse(
        status = str(report.get("status", "ok")),
        dry_run = bool(report.get("dry_run", payload.dry_run)),
        scanned_pages = int(report.get("scanned_pages", 0)),
        updated_pages = int(report.get("updated_pages", 0)),
        changes = [dict(item) for item in report.get("changes", [])],
        web_gap_fill = dict(report.get("web_gap_fill", {})),
    )


@router.post("/wiki/retry-fallback", response_model = WikiRetryFallbackResponse)
async def wiki_retry_fallback(
    payload: WikiRetryFallbackRequest,
    current_subject: str = Depends(get_current_subject),
):
    """Retry analysis pages that were previously generated via extractive fallback."""
    manager, _ = _get_route_wiki_components()

    try:
        report = manager.retry_fallback_analysis_pages(
            dry_run=payload.dry_run,
            max_analysis_pages=payload.max_analysis_pages,
        )
    except Exception as exc:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail = f"Wiki fallback retry failed: {exc}",
        )

    return WikiRetryFallbackResponse(
        status = str(report.get("status", "ok")),
        dry_run = bool(report.get("dry_run", payload.dry_run)),
        scanned_pages = int(report.get("scanned_pages", 0)),
        fallback_pages_found = int(report.get("fallback_pages_found", 0)),
        retried_pages = int(report.get("retried_pages", 0)),
        regenerated_pages = int(report.get("regenerated_pages", 0)),
        fallback_still = int(report.get("fallback_still", 0)),
        skipped_no_question = int(report.get("skipped_no_question", 0)),
        errors = [str(item) for item in report.get("errors", [])],
        results = [dict(item) for item in report.get("results", [])],
    )


@router.post("/wiki/query", response_model = WikiQueryResponse)
async def wiki_query(
    payload: WikiQueryRequest,
    current_subject: str = Depends(get_current_subject),
):
    """Query the maintained wiki and optionally file answers into wiki/analysis."""
    if not _wiki_llm_available():
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = "No active model loaded for wiki query synthesis. Load a model first.",
        )

    manager, _ = _get_route_wiki_components()

    try:
        result = manager.engine.query(payload.question, save_answer = payload.save_answer)
    except Exception as exc:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail = f"Wiki query failed: {exc}",
        )

    global _WIKI_QUERY_RUN_COUNT
    _WIKI_QUERY_RUN_COUNT += 1
    if _WIKI_AUTO_LINT_EVERY_QUERY > 0 and _WIKI_QUERY_RUN_COUNT % _WIKI_AUTO_LINT_EVERY_QUERY == 0:
        try:
            lint_report = manager.engine.lint()
            logger.info(
                "Auto lint after wiki query #%d: orphans=%d stale=%d broken=%d",
                _WIKI_QUERY_RUN_COUNT,
                len(lint_report.get("orphans", [])),
                len(lint_report.get("stale_pages", [])),
                len(lint_report.get("broken_links", [])),
            )
        except Exception as exc:
            logger.warning("Auto lint after query #%d failed: %s", _WIKI_QUERY_RUN_COUNT, exc)

        if _WIKI_AUTO_RETRY_FALLBACK_MAX_PAGES > 0:
            try:
                retry_report = manager.retry_fallback_analysis_pages(
                    dry_run=False,
                    max_analysis_pages=_WIKI_AUTO_RETRY_FALLBACK_MAX_PAGES,
                )
                logger.info(
                    "Auto fallback-retry after wiki query #%d: scanned=%d fallback_found=%d regenerated=%d still_fallback=%d",
                    _WIKI_QUERY_RUN_COUNT,
                    int(retry_report.get("scanned_pages", 0)),
                    int(retry_report.get("fallback_pages_found", 0)),
                    int(retry_report.get("regenerated_pages", 0)),
                    int(retry_report.get("fallback_still", 0)),
                )
            except Exception as exc:
                logger.warning("Auto fallback-retry after query #%d failed: %s", _WIKI_QUERY_RUN_COUNT, exc)

        try:
            enrich_report = manager.enrich_analysis_pages(dry_run=False)
            logger.info(
                "Auto enrichment after wiki query #%d: scanned=%d updated=%d",
                _WIKI_QUERY_RUN_COUNT,
                int(enrich_report.get("scanned_pages", 0)),
                int(enrich_report.get("updated_pages", 0)),
            )
        except Exception as exc:
            logger.warning("Auto enrichment after query #%d failed: %s", _WIKI_QUERY_RUN_COUNT, exc)

    return WikiQueryResponse(
        status = str(result.get("status", "ok")),
        answer = str(result.get("answer", "")),
        answer_page = result.get("answer_page"),
        context_pages = [str(p) for p in result.get("context_pages", [])],
    )


@router.get("/wiki/lint", response_model = WikiLintResponse)
async def wiki_lint(
    current_subject: str = Depends(get_current_subject),
):
    """Run a wiki health-check report (orphans, stale pages, links, gaps)."""
    manager, _ = _get_route_wiki_components()

    # Keep manual lint behavior aligned with scheduled maintenance.
    if _WIKI_AUTO_RETRY_FALLBACK_MAX_PAGES > 0:
        try:
            retry_report = manager.retry_fallback_analysis_pages(
                dry_run=False,
                max_analysis_pages=_WIKI_AUTO_RETRY_FALLBACK_MAX_PAGES,
            )
            logger.info(
                "Fallback-retry before /wiki/lint: scanned=%d fallback_found=%d regenerated=%d still_fallback=%d",
                int(retry_report.get("scanned_pages", 0)),
                int(retry_report.get("fallback_pages_found", 0)),
                int(retry_report.get("regenerated_pages", 0)),
                int(retry_report.get("fallback_still", 0)),
            )
        except Exception as exc:
            logger.warning("Fallback-retry before /wiki/lint failed: %s", exc)

    report = manager.engine.lint()

    return WikiLintResponse(
        status = str(report.get("status", "ok")),
        orphans = [str(x) for x in report.get("orphans", [])],
        stale_pages = [dict(x) for x in report.get("stale_pages", [])],
        broken_links = [dict(x) for x in report.get("broken_links", [])],
        missing_concepts = [str(x) for x in report.get("missing_concepts", [])],
        low_coverage_sources = [str(x) for x in report.get("low_coverage_sources", [])],
        total_pages = int(report.get("total_pages", 0)),
    )


@router.post("/load", response_model = LoadResponse)
async def load_model(
    request: LoadRequest,
    current_subject: str = Depends(get_current_subject),
):
    """
    Load a model for inference.

    The model_path should be a clean identifier from GET /models/list.
    Returns inference configuration parameters (temperature, top_p, top_k, min_p)
    from the model's YAML config, falling back to default.yaml for missing values.

    GGUF models are loaded via llama-server (llama.cpp) instead of Unsloth.
    """
    try:
        # Version switching is handled automatically by the subprocess-based
        # inference backend — no need for ensure_transformers_version() here.

        # ── Already-loaded check: skip reload if the exact model is active ──
        backend = get_inference_backend()
        llama_backend = get_llama_cpp_backend()

        if request.gguf_variant:
            if (
                llama_backend.is_loaded
                and llama_backend.hf_variant
                and llama_backend.hf_variant.lower() == request.gguf_variant.lower()
                and llama_backend.model_identifier
                and llama_backend.model_identifier.lower() == request.model_path.lower()
            ):
                logger.info(
                    f"Model already loaded (GGUF): {request.model_path} variant={request.gguf_variant}, skipping reload"
                )
                inference_config = load_inference_config(llama_backend.model_identifier)
                from utils.models import is_audio_input_type

                _gguf_audio = (
                    llama_backend._audio_type
                    if hasattr(llama_backend, "_audio_type")
                    else None
                )
                _gguf_is_audio = getattr(llama_backend, "_is_audio", False)
                return LoadResponse(
                    status = "already_loaded",
                    model = llama_backend.model_identifier,
                    display_name = llama_backend.model_identifier,
                    is_vision = llama_backend._is_vision,
                    is_lora = False,
                    is_gguf = True,
                    is_audio = _gguf_is_audio,
                    audio_type = _gguf_audio,
                    has_audio_input = is_audio_input_type(_gguf_audio)
                    if _gguf_audio
                    else False,
                    inference = inference_config,
                    context_length = llama_backend.context_length,
                    max_context_length = llama_backend.max_context_length,
                    native_context_length = llama_backend.native_context_length,
                    supports_reasoning = llama_backend.supports_reasoning,
                    reasoning_always_on = llama_backend.reasoning_always_on,
                    chat_template = llama_backend.chat_template,
                    speculative_type = llama_backend.speculative_type,
                )
        else:
            if (
                backend.active_model_name
                and backend.active_model_name.lower() == request.model_path.lower()
            ):
                logger.info(
                    f"Model already loaded (Unsloth): {request.model_path}, skipping reload"
                )
                inference_config = load_inference_config(backend.active_model_name)
                _model_info = backend.models.get(backend.active_model_name, {})
                _chat_template = None
                try:
                    _tpl_info = _model_info.get("chat_template_info", {})
                    _chat_template = _tpl_info.get("template")
                except Exception as e:
                    logger.warning(
                        f"Could not retrieve chat template for {backend.active_model_name}: {e}"
                    )
                return LoadResponse(
                    status = "already_loaded",
                    model = backend.active_model_name,
                    display_name = backend.active_model_name,
                    is_vision = _model_info.get("is_vision", False),
                    is_lora = _model_info.get("is_lora", False),
                    is_gguf = False,
                    is_audio = _model_info.get("is_audio", False),
                    audio_type = _model_info.get("audio_type"),
                    has_audio_input = _model_info.get("has_audio_input", False),
                    inference = inference_config,
                    chat_template = _chat_template,
                )

        # Create config using clean factory method
        # is_lora is auto-detected from adapter_config.json on disk/HF
        config = ModelConfig.from_identifier(
            model_id = request.model_path,
            hf_token = request.hf_token,
            gguf_variant = request.gguf_variant,
        )

        if not config:
            raise HTTPException(
                status_code = 400,
                detail = f"Invalid model identifier: {request.model_path}",
            )

        # Normalize gpu_ids: empty list means auto-selection, same as None
        effective_gpu_ids = request.gpu_ids if request.gpu_ids else None

        # ── GGUF path: load via llama-server ──────────────────────
        if config.is_gguf:
            if effective_gpu_ids is not None:
                raise HTTPException(
                    status_code = 400,
                    detail = "gpu_ids is not supported for GGUF models yet.",
                )

            llama_backend = get_llama_cpp_backend()
            unsloth_backend = get_inference_backend()

            # Unload any active Unsloth model first to free VRAM
            if unsloth_backend.active_model_name:
                logger.info(
                    f"Unloading Unsloth model '{unsloth_backend.active_model_name}' before loading GGUF"
                )
                unsloth_backend.unload_model(unsloth_backend.active_model_name)

            # Route to HF mode or local mode based on config
            # Run in a thread so the event loop stays free for progress
            # polling and other requests during the (potentially long)
            # GGUF download + llama-server startup.
            if config.gguf_hf_repo:
                # HF mode: download via huggingface_hub then start llama-server
                success = await asyncio.to_thread(
                    llama_backend.load_model,
                    hf_repo = config.gguf_hf_repo,
                    hf_variant = config.gguf_variant,
                    hf_token = request.hf_token,
                    model_identifier = config.identifier,
                    is_vision = config.is_vision,
                    n_ctx = request.max_seq_length,
                    chat_template_override = request.chat_template_override,
                    cache_type_kv = request.cache_type_kv,
                    speculative_type = request.speculative_type,
                )
            else:
                # Local mode: llama-server loads via -m <path>
                success = await asyncio.to_thread(
                    llama_backend.load_model,
                    gguf_path = config.gguf_file,
                    mmproj_path = config.gguf_mmproj_file,
                    model_identifier = config.identifier,
                    is_vision = config.is_vision,
                    n_ctx = request.max_seq_length,
                    chat_template_override = request.chat_template_override,
                    cache_type_kv = request.cache_type_kv,
                    speculative_type = request.speculative_type,
                )

            if not success:
                raise HTTPException(
                    status_code = 500,
                    detail = f"Failed to load GGUF model: {config.display_name}",
                )

            logger.info(f"Loaded GGUF model via llama-server: {config.identifier}")

            # Detect TTS audio by probing the loaded model's vocabulary
            from utils.models import is_audio_input_type

            _gguf_audio = llama_backend.detect_audio_type()
            _gguf_is_audio = _gguf_audio in ("snac", "bicodec", "dac")
            llama_backend._is_audio = _gguf_is_audio
            llama_backend._audio_type = _gguf_audio
            if _gguf_is_audio:
                logger.info(f"GGUF model detected as audio: audio_type={_gguf_audio}")
                await asyncio.to_thread(llama_backend.init_audio_codec, _gguf_audio)

            inference_config = load_inference_config(config.identifier)

            return LoadResponse(
                status = "loaded",
                model = config.identifier,
                display_name = config.display_name,
                is_vision = config.is_vision,
                is_lora = False,
                is_gguf = True,
                is_audio = _gguf_is_audio,
                audio_type = _gguf_audio,
                has_audio_input = is_audio_input_type(_gguf_audio),
                inference = inference_config,
                context_length = llama_backend.context_length,
                max_context_length = llama_backend.max_context_length,
                native_context_length = llama_backend.native_context_length,
                supports_reasoning = llama_backend.supports_reasoning,
                reasoning_always_on = llama_backend.reasoning_always_on,
                supports_tools = llama_backend.supports_tools,
                cache_type_kv = llama_backend.cache_type_kv,
                chat_template = llama_backend.chat_template,
                speculative_type = llama_backend.speculative_type,
            )

        # ── Standard path: load via Unsloth/transformers ──────────
        backend = get_inference_backend()

        # Unload any active GGUF model first
        llama_backend = get_llama_cpp_backend()
        if llama_backend.is_loaded:
            logger.info("Unloading GGUF model before loading Unsloth model")
            llama_backend.unload_model()

        # Shut down any export subprocess to free VRAM
        try:
            from core.export import get_export_backend

            exp_backend = get_export_backend()
            if exp_backend.current_checkpoint:
                logger.info(
                    "Shutting down export subprocess to free GPU memory for inference"
                )
                exp_backend._shutdown_subprocess()
                exp_backend.current_checkpoint = None
                exp_backend.is_vision = False
                exp_backend.is_peft = False
        except Exception as e:
            logger.warning("Could not shut down export subprocess: %s", e)

        # Auto-detect quantization for LoRA adapters from adapter_config.json
        # The training pipeline patches this file with "unsloth_training_method"
        # which is 'qlora' or 'lora'. Only LoRA (16-bit) needs load_in_4bit=False.
        load_in_4bit = request.load_in_4bit
        if config.is_lora and config.path:
            import json
            from pathlib import Path

            adapter_cfg_path = Path(config.path) / "adapter_config.json"
            if adapter_cfg_path.exists():
                try:
                    with open(adapter_cfg_path) as f:
                        adapter_cfg = json.load(f)
                    training_method = adapter_cfg.get("unsloth_training_method")
                    if training_method == "lora" and load_in_4bit:
                        logger.info(
                            f"adapter_config.json says unsloth_training_method='lora' — "
                            f"setting load_in_4bit=False to match 16-bit training"
                        )
                        load_in_4bit = False
                    elif training_method == "qlora" and not load_in_4bit:
                        logger.info(
                            f"adapter_config.json says unsloth_training_method='qlora' — "
                            f"setting load_in_4bit=True to match QLoRA training"
                        )
                        load_in_4bit = True
                    elif training_method:
                        logger.info(
                            f"Training method: {training_method}, load_in_4bit={load_in_4bit}"
                        )
                    else:
                        # No unsloth_training_method — fallback to base model name
                        if (
                            config.base_model
                            and "-bnb-4bit" not in config.base_model.lower()
                            and load_in_4bit
                        ):
                            logger.info(
                                f"No unsloth_training_method in adapter_config.json. "
                                f"Base model '{config.base_model}' has no -bnb-4bit suffix — "
                                f"setting load_in_4bit=False"
                            )
                            load_in_4bit = False
                except Exception as e:
                    logger.warning(f"Could not read adapter_config.json: {e}")

        # Load the model in a thread so the event loop stays free
        # for download progress polling and other requests.
        success = await asyncio.to_thread(
            backend.load_model,
            config = config,
            max_seq_length = request.max_seq_length,
            load_in_4bit = load_in_4bit,
            hf_token = request.hf_token,
            trust_remote_code = request.trust_remote_code,
            gpu_ids = effective_gpu_ids,
        )

        if not success:
            # Check if YAML says this model needs trust_remote_code
            if not request.trust_remote_code:
                model_defaults = load_model_defaults(config.identifier)
                yaml_trust = model_defaults.get("inference", {}).get(
                    "trust_remote_code", False
                )
                if yaml_trust:
                    raise HTTPException(
                        status_code = 400,
                        detail = (
                            f"Model '{config.display_name}' requires trust_remote_code to be enabled. "
                            f"Please enable 'Trust remote code' in Chat Settings and try again."
                        ),
                    )
            raise HTTPException(
                status_code = 500, detail = f"Failed to load model: {config.display_name}"
            )

        logger.info(f"Loaded model: {config.identifier}")

        # Load inference configuration parameters
        inference_config = load_inference_config(config.identifier)

        # Get chat template from tokenizer
        _chat_template = None
        try:
            _model_info = backend.models.get(config.identifier, {})
            _tpl_info = _model_info.get("chat_template_info", {})
            _chat_template = _tpl_info.get("template")
        except Exception:
            pass

        return LoadResponse(
            status = "loaded",
            model = config.identifier,
            display_name = config.display_name,
            is_vision = config.is_vision,
            is_lora = config.is_lora,
            is_gguf = False,
            is_audio = config.is_audio,
            audio_type = config.audio_type,
            has_audio_input = config.has_audio_input,
            inference = inference_config,
            chat_template = _chat_template,
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("Rejected inference GPU selection: %s", e)
        raise HTTPException(status_code = 400, detail = str(e))
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info = True)
        msg = str(e)
        # Surface a friendlier message for models that Unsloth cannot load
        not_supported_hints = [
            "No config file found",
            "not yet supported",
            "is not supported",
            "does not support",
        ]
        if any(h.lower() in msg.lower() for h in not_supported_hints):
            msg = f"This model is not supported yet. Try a different model. (Original error: {msg})"
        raise HTTPException(status_code = 500, detail = f"Failed to load model: {msg}")


@router.post("/validate", response_model = ValidateModelResponse)
async def validate_model(
    request: ValidateModelRequest,
    current_subject: str = Depends(get_current_subject),
):
    """
    Lightweight validation endpoint for model identifiers.

    This checks that ModelConfig.from_identifier() can resolve the given
    model_path, but it does NOT actually load model weights into GPU memory.
    """
    try:
        config = ModelConfig.from_identifier(
            model_id = request.model_path,
            hf_token = request.hf_token,
            gguf_variant = request.gguf_variant,
        )

        if not config:
            raise HTTPException(
                status_code = 400,
                detail = f"Invalid model identifier: {request.model_path}",
            )

        return ValidateModelResponse(
            valid = True,
            message = "Model identifier is valid.",
            identifier = config.identifier,
            display_name = getattr(config, "display_name", config.identifier),
            is_gguf = getattr(config, "is_gguf", False),
            is_lora = getattr(config, "is_lora", False),
            is_vision = getattr(config, "is_vision", False),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error validating model identifier '{request.model_path}': {e}",
            exc_info = True,
        )
        raise HTTPException(
            status_code = 400,
            detail = f"Invalid model: {str(e)}",
        )


@router.post("/unload", response_model = UnloadResponse)
async def unload_model(
    request: UnloadRequest,
    current_subject: str = Depends(get_current_subject),
):
    """
    Unload a model from memory.
    Routes to the correct backend (llama-server for GGUF, Unsloth otherwise).
    """
    try:
        # Check if the GGUF backend has this model loaded or is loading it
        llama_backend = get_llama_cpp_backend()
        if llama_backend.is_active and (
            llama_backend.model_identifier == request.model_path
            or not llama_backend.is_loaded
        ):
            llama_backend.unload_model()
            logger.info(f"Unloaded GGUF model: {request.model_path}")
            return UnloadResponse(status = "unloaded", model = request.model_path)

        # Otherwise, unload from Unsloth backend
        backend = get_inference_backend()
        backend.unload_model(request.model_path)
        logger.info(f"Unloaded model: {request.model_path}")
        return UnloadResponse(status = "unloaded", model = request.model_path)

    except Exception as e:
        logger.error(f"Error unloading model: {e}", exc_info = True)
        raise HTTPException(status_code = 500, detail = f"Failed to unload model: {str(e)}")


@router.post("/generate/stream")
async def generate_stream(
    request: GenerateRequest,
    current_subject: str = Depends(get_current_subject),
):
    """
    Generate a chat response with Server-Sent Events (SSE) streaming.

    For vision models, provide image_base64 with the base64-encoded image.
    """
    backend = get_inference_backend()

    if not backend.active_model_name:
        raise HTTPException(
            status_code = 400, detail = "No model loaded. Call POST /inference/load first."
        )

    # Decode image if provided (for vision models)
    image = None
    if request.image_base64:
        try:
            import base64
            from PIL import Image
            from io import BytesIO

            # Check if current model supports vision
            model_info = backend.models.get(backend.active_model_name, {})
            if not model_info.get("is_vision"):
                raise HTTPException(
                    status_code = 400,
                    detail = "Image provided but current model is text-only. Load a vision model.",
                )

            image_data = base64.b64decode(request.image_base64)
            image = Image.open(BytesIO(image_data))
            image = backend.resize_image(image)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code = 400, detail = f"Failed to decode image: {str(e)}"
            )

    async def stream():
        try:
            for chunk in backend.generate_chat_response(
                messages = request.messages,
                system_prompt = request.system_prompt,
                image = image,
                temperature = request.temperature,
                top_p = request.top_p,
                top_k = request.top_k,
                max_new_tokens = request.max_new_tokens,
                repetition_penalty = request.repetition_penalty,
            ):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            backend.reset_generation_state()
            logger.error(f"Error during generation: {e}", exc_info = True)
            yield f"data: {json.dumps({'error': _friendly_error(e)})}\n\n"

    return StreamingResponse(
        stream(),
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/status", response_model = InferenceStatusResponse)
async def get_status(
    current_subject: str = Depends(get_current_subject),
):
    """
    Get current inference backend status.
    Reports whichever backend (Unsloth or llama-server) is currently active.
    """
    try:
        llama_backend = get_llama_cpp_backend()

        # If a GGUF model is loaded via llama-server, report that
        if llama_backend.is_loaded:
            _model_id = llama_backend.model_identifier
            _inference_cfg = load_inference_config(_model_id) if _model_id else None
            return InferenceStatusResponse(
                active_model = _model_id,
                is_vision = llama_backend.is_vision,
                is_gguf = True,
                gguf_variant = llama_backend.hf_variant,
                is_audio = getattr(llama_backend, "_is_audio", False),
                audio_type = getattr(llama_backend, "_audio_type", None),
                loading = [],
                loaded = [_model_id],
                inference = _inference_cfg,
                supports_reasoning = llama_backend.supports_reasoning,
                reasoning_always_on = llama_backend.reasoning_always_on,
                supports_tools = llama_backend.supports_tools,
                context_length = llama_backend.context_length,
                max_context_length = llama_backend.max_context_length,
                native_context_length = llama_backend.native_context_length,
                speculative_type = llama_backend.speculative_type,
            )

        # Otherwise, report Unsloth backend status
        backend = get_inference_backend()

        is_vision = False
        is_audio = False
        audio_type = None
        has_audio_input = False
        if backend.active_model_name:
            model_info = backend.models.get(backend.active_model_name, {})
            is_vision = model_info.get("is_vision", False)
            is_audio = model_info.get("is_audio", False)
            audio_type = model_info.get("audio_type")
            has_audio_input = model_info.get("has_audio_input", False)

        # gpt-oss safetensors models support reasoning via harmony channels
        supports_reasoning = False
        if backend.active_model_name and hasattr(backend, "_is_gpt_oss_model"):
            supports_reasoning = backend._is_gpt_oss_model()

        return InferenceStatusResponse(
            active_model = backend.active_model_name,
            is_vision = is_vision,
            is_gguf = False,
            is_audio = is_audio,
            audio_type = audio_type,
            has_audio_input = has_audio_input,
            loading = list(getattr(backend, "loading_models", set())),
            loaded = list(backend.models.keys()),
            supports_reasoning = supports_reasoning,
        )

    except Exception as e:
        logger.error(f"Error getting status: {e}", exc_info = True)
        raise HTTPException(status_code = 500, detail = f"Failed to get status: {str(e)}")


# =====================================================================
# Audio (TTS) Generation  (/audio/generate)
# =====================================================================


@router.post("/audio/generate")
async def generate_audio(
    payload: ChatCompletionRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """
    Generate audio (TTS) from the latest user message.
    Returns a JSON response with base64-encoded WAV audio.
    Works with both GGUF (llama-server) and Unsloth/transformers backends.
    """
    import base64

    # Extract text from the last user message
    _, chat_messages, _ = _extract_content_parts(payload.messages)
    if not chat_messages:
        raise HTTPException(status_code = 400, detail = "No messages provided.")
    last_user_msg = next(
        (m for m in reversed(chat_messages) if m["role"] == "user"), None
    )
    if not last_user_msg:
        raise HTTPException(status_code = 400, detail = "No user message found.")
    text = last_user_msg["content"]

    # Pick backend — both return (wav_bytes, sample_rate)
    llama_backend = get_llama_cpp_backend()
    if llama_backend.is_loaded and getattr(llama_backend, "_is_audio", False):
        model_name = llama_backend.model_identifier
        gen = lambda: llama_backend.generate_audio_response(
            text = text,
            audio_type = llama_backend._audio_type,
            temperature = payload.temperature,
            top_p = payload.top_p,
            top_k = payload.top_k,
            min_p = payload.min_p,
            max_new_tokens = payload.max_tokens or 2048,
            repetition_penalty = payload.repetition_penalty,
        )
    else:
        backend = get_inference_backend()
        if not backend.active_model_name:
            raise HTTPException(status_code = 400, detail = "No model loaded.")
        model_info = backend.models.get(backend.active_model_name, {})
        if not model_info.get("is_audio"):
            raise HTTPException(
                status_code = 400, detail = "Active model is not an audio model."
            )
        model_name = backend.active_model_name
        gen = lambda: backend.generate_audio_response(
            text = text,
            temperature = payload.temperature,
            top_p = payload.top_p,
            top_k = payload.top_k,
            min_p = payload.min_p,
            max_new_tokens = payload.max_tokens or 2048,
            repetition_penalty = payload.repetition_penalty,
            use_adapter = payload.use_adapter,
        )

    try:
        wav_bytes, sample_rate = await asyncio.get_event_loop().run_in_executor(
            None, gen
        )
    except Exception as e:
        logger.error(f"Audio generation error: {e}", exc_info = True)
        raise HTTPException(status_code = 500, detail = str(e))

    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    return JSONResponse(
        content = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.audio",
            "model": model_name,
            "audio": {"data": audio_b64, "format": "wav", "sample_rate": sample_rate},
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f'[Generated audio from: "{text[:100]}"]',
                    },
                    "finish_reason": "stop",
                }
            ],
        }
    )


# =====================================================================
# OpenAI-Compatible Chat Completions  (/chat/completions)
# =====================================================================


def _decode_audio_base64(b64: str) -> np.ndarray:
    """Decode base64 audio (any format) → float32 numpy array at 16kHz."""
    import torch
    import torchaudio
    import tempfile
    import os
    from utils.paths import ensure_dir, tmp_root

    raw = base64.b64decode(b64)
    # torchaudio.load needs a file path or file-like object with format hint
    # Write to a temp file so torchaudio can auto-detect the format
    with tempfile.NamedTemporaryFile(
        suffix = ".audio",
        delete = False,
        dir = str(ensure_dir(tmp_root())),
    ) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        waveform, sr = torchaudio.load(tmp_path)
    finally:
        os.unlink(tmp_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim = 0, keepdim = True)

    # Resample to 16kHz if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq = sr, new_freq = 16000)
        waveform = resampler(waveform)

    return waveform.squeeze(0).numpy()


def _extract_content_parts(
    messages: list,
) -> tuple[str, list[dict], "Optional[str]"]:
    """
    Parse OpenAI-format messages into components the inference backend expects.

    Handles both plain-string ``content`` and multimodal content-part arrays
    (``[{type: "text", ...}, {type: "image_url", ...}]``).

    Returns:
        system_prompt:  The system message text (empty string if none provided).
        chat_messages:  Non-system messages with content flattened to strings.
        image_base64:   Base64 data of the *first* image found, or ``None``.
    """
    system_prompt = ""
    chat_messages: list[dict] = []
    first_image_b64: Optional[str] = None

    for msg in messages:
        # ── System messages → extract as system_prompt ────────
        if msg.role == "system":
            if isinstance(msg.content, str):
                system_prompt = msg.content
            elif isinstance(msg.content, list):
                # Unlikely but handle: join text parts
                system_prompt = "\n".join(
                    p.text for p in msg.content if p.type == "text"
                )
            continue

        # ── User / assistant messages ─────────────────────────
        if isinstance(msg.content, str):
            # Plain string content — pass through
            chat_messages.append({"role": msg.role, "content": msg.content})
        elif isinstance(msg.content, list):
            # Multimodal content parts
            text_parts: list[str] = []
            for part in msg.content:
                if part.type == "text":
                    text_parts.append(part.text)
                elif part.type == "image_url" and first_image_b64 is None:
                    url = part.image_url.url
                    if url.startswith("data:"):
                        # data:image/png;base64,<DATA> → extract <DATA>
                        first_image_b64 = url.split(",", 1)[1] if "," in url else None
                    else:
                        logger.warning(
                            f"Remote image URLs not yet supported: {url[:80]}..."
                        )
            combined_text = "\n".join(text_parts) if text_parts else ""
            chat_messages.append({"role": msg.role, "content": combined_text})

    return system_prompt, chat_messages, first_image_b64


@router.post("/chat/completions")
async def openai_chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """
    OpenAI-compatible chat completions endpoint.

    Supports multimodal messages: ``content`` may be a plain string or a
    list of content parts (``text`` / ``image_url``).

    Streaming (default):  returns SSE chunks matching OpenAI's format.
    Non-streaming:        returns a single ChatCompletion JSON object.

    Automatically routes to the correct backend:
    - GGUF models → llama-server via LlamaCppBackend
    - Other models → Unsloth/transformers via InferenceBackend
    """
    llama_backend = get_llama_cpp_backend()
    using_gguf = llama_backend.is_loaded

    # ── Determine which backend is active ─────────────────────
    if using_gguf:
        model_name = llama_backend.model_identifier or payload.model
        if getattr(llama_backend, "_is_audio", False):
            return await generate_audio(payload, request)
    else:
        backend = get_inference_backend()
        if not backend.active_model_name:
            raise HTTPException(
                status_code = 400,
                detail = "No model loaded. Call POST /inference/load first.",
            )
        model_name = backend.active_model_name or payload.model

        # ── Audio TTS path: auto-route to audio generation ────
        # (Whisper is ASR not TTS — handled below in audio input path)
        model_info = backend.models.get(backend.active_model_name, {})
        if model_info.get("is_audio") and model_info.get("audio_type") != "whisper":
            return await generate_audio(payload, request)

        # ── Whisper without audio: return clear error ──
        if model_info.get("audio_type") == "whisper" and not payload.audio_base64:
            raise HTTPException(
                status_code = 400,
                detail = "Whisper models require audio input. Please upload an audio file.",
            )

        # ── Audio INPUT path: decode WAV and route to audio input generation ──
        if payload.audio_base64 and model_info.get("has_audio_input"):
            audio_array = _decode_audio_base64(payload.audio_base64)
            system_prompt, chat_messages, _ = _extract_content_parts(payload.messages)
            cancel_event = threading.Event()
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())

            def audio_input_generate():
                if model_info.get("audio_type") == "whisper":
                    return backend.generate_whisper_response(
                        audio_array = audio_array,
                        cancel_event = cancel_event,
                    )
                return backend.generate_audio_input_response(
                    messages = chat_messages,
                    system_prompt = system_prompt,
                    audio_array = audio_array,
                    temperature = payload.temperature,
                    top_p = payload.top_p,
                    top_k = payload.top_k,
                    min_p = payload.min_p,
                    max_new_tokens = payload.max_tokens or 2048,
                    repetition_penalty = payload.repetition_penalty,
                    cancel_event = cancel_event,
                )

            if payload.stream:

                async def audio_input_stream():
                    try:
                        first_chunk = ChatCompletionChunk(
                            id = completion_id,
                            created = created,
                            model = model_name,
                            choices = [
                                ChunkChoice(
                                    delta = ChoiceDelta(role = "assistant"),
                                    finish_reason = None,
                                )
                            ],
                        )
                        yield f"data: {first_chunk.model_dump_json(exclude_none = True)}\n\n"

                        for chunk_text in audio_input_generate():
                            if await request.is_disconnected():
                                cancel_event.set()
                                return
                            if chunk_text:
                                chunk = ChatCompletionChunk(
                                    id = completion_id,
                                    created = created,
                                    model = model_name,
                                    choices = [
                                        ChunkChoice(
                                            delta = ChoiceDelta(content = chunk_text),
                                            finish_reason = None,
                                        )
                                    ],
                                )
                                yield f"data: {chunk.model_dump_json(exclude_none = True)}\n\n"

                        final_chunk = ChatCompletionChunk(
                            id = completion_id,
                            created = created,
                            model = model_name,
                            choices = [
                                ChunkChoice(delta = ChoiceDelta(), finish_reason = "stop")
                            ],
                        )
                        yield f"data: {final_chunk.model_dump_json(exclude_none = True)}\n\n"
                        yield "data: [DONE]\n\n"
                    except asyncio.CancelledError:
                        cancel_event.set()
                        raise
                    except Exception as e:
                        logger.error(
                            f"Error during audio input streaming: {e}", exc_info = True
                        )
                        yield f"data: {json.dumps({'error': {'message': _friendly_error(e), 'type': 'server_error'}})}\n\n"

                return StreamingResponse(
                    audio_input_stream(),
                    media_type = "text/event-stream",
                    headers = {
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
            else:
                full_text = "".join(audio_input_generate())
                response = ChatCompletion(
                    id = completion_id,
                    created = created,
                    model = model_name,
                    choices = [
                        CompletionChoice(
                            message = CompletionMessage(content = full_text),
                            finish_reason = "stop",
                        )
                    ],
                )
                return JSONResponse(content = response.model_dump())

    # ── Parse messages (handles multimodal content parts) ─────
    system_prompt, chat_messages, extracted_image_b64 = _extract_content_parts(
        payload.messages
    )

    if not chat_messages:
        raise HTTPException(
            status_code = 400,
            detail = "At least one non-system message is required.",
        )

    # ── GGUF path: proxy to llama-server /v1/chat/completions ──
    if using_gguf:

        # Reject images if this GGUF model doesn't support vision
        image_b64 = extracted_image_b64 or payload.image_base64
        if image_b64 and not llama_backend.is_vision:
            raise HTTPException(
                status_code = 400,
                detail = "Image provided but current GGUF model does not support vision.",
            )

        # Convert image to PNG for llama-server (stb_image has limited format support)
        if image_b64:
            try:
                import base64 as _b64
                from io import BytesIO as _BytesIO
                from PIL import Image as _Image

                raw = _b64.b64decode(image_b64)
                img = _Image.open(_BytesIO(raw))
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                buf = _BytesIO()
                img.save(buf, format = "PNG")
                image_b64 = _b64.b64encode(buf.getvalue()).decode("ascii")
            except Exception as e:
                raise HTTPException(
                    status_code = 400, detail = f"Failed to process image: {e}"
                )

        # RAG context + chat history persistence for GGUF path.
        try:
            _ingest_pending_raw_files()
            if chat_messages:
                last_user_msg = next(
                    (m for m in reversed(chat_messages) if m.get("role") == "user"),
                    None,
                )
                last_user_text = (
                    str(last_user_msg.get("content", "")).strip()
                    if last_user_msg
                    else ""
                )
                if last_user_text:
                    rag_context, rag_debug = _get_route_rag_context(
                        last_user_text,
                        return_debug=True,
                        debug_source="last-request",
                    )
                    global _LAST_RAG_DEBUG
                    _LAST_RAG_DEBUG = rag_debug
                    selected_pages = [
                        str(item.get("page", "unknown"))
                        for item in rag_debug.get("selected", [])
                    ]
                    logger.info(
                        "RAG selection for GGUF request: pages=%d chars=%d selected=%s query=%r",
                        len(selected_pages),
                        len(rag_context),
                        selected_pages,
                        last_user_text,
                    )
                    if rag_context:
                        logger.info("Injecting RAG context into GGUF prompt")
                        if _RAG_LOG_INJECTED_CONTEXT:
                            logger.info(
                                "Injected RAG context:\n%s",
                                _loggable_rag_context(rag_context),
                            )
                        rag_block = (
                            "Use the following context to help answer the user's request:\n\n"
                            f"{rag_context}"
                        )
                        if system_prompt:
                            system_prompt = system_prompt.rstrip() + "\n\n" + rag_block
                        else:
                            system_prompt = rag_block
                    else:
                        logger.info("RAG produced empty context for GGUF request")

                _save_chat_history_to_route_wiki(chat_messages)
        except Exception as e:
            logger.warning(f"Failed to apply GGUF RAG/wiki history hooks: {e}")

        # Build message list with system prompt prepended
        gguf_messages = []
        if system_prompt:
            gguf_messages.append({"role": "system", "content": system_prompt})
        gguf_messages.extend(chat_messages)

        cancel_event = threading.Event()

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # ── Tool-calling path (agentic loop) ──────────────────
        use_tools = (
            payload.enable_tools and llama_backend.supports_tools and not image_b64
        )

        if use_tools:
            from core.inference.tools import ALL_TOOLS

            if payload.enabled_tools is not None:
                tools_to_use = [
                    t
                    for t in ALL_TOOLS
                    if t["function"]["name"] in payload.enabled_tools
                ]
            else:
                tools_to_use = ALL_TOOLS

            # ── Tool-use system prompt nudge ──────────────────────
            _tool_names = {t["function"]["name"] for t in tools_to_use}
            _has_web = "web_search" in _tool_names
            _has_code = "python" in _tool_names or "terminal" in _tool_names

            _date_line = f"The current date is {_date.today().isoformat()}."

            # Small models (<9B) struggle with multi-step search plans,
            # so simplify the web tips to avoid plan-then-stall behavior.
            _model_size_b = _extract_model_size_b(model_name)
            _is_small_model = _model_size_b is not None and _model_size_b < 9

            if _is_small_model:
                _web_tips = "Do not repeat the same search query."
            else:
                _web_tips = (
                    "When you search and find a relevant URL in the results, "
                    "fetch its full content by calling web_search with the url parameter. "
                    "Do not repeat the same search query. If a search returns "
                    "no useful results, try rephrasing or fetching a result URL directly."
                )
            _code_tips = (
                "Use code execution for math, calculations, data processing, "
                "or to parse and analyze information from tool results."
            )

            if _has_web and _has_code:
                _nudge = (
                    _date_line + " "
                    "You have access to tools. When appropriate, prefer using "
                    "tools rather than answering from memory. "
                    + _web_tips
                    + " "
                    + _code_tips
                )
            elif _has_code:
                _nudge = (
                    _date_line + " "
                    "You have access to tools. When appropriate, prefer using "
                    "code execution rather than answering from memory. " + _code_tips
                )
            elif _has_web:
                _nudge = (
                    _date_line + " "
                    "You have access to tools. When appropriate, prefer using "
                    "web search for up-to-date or uncertain factual "
                    "information rather than answering from memory. " + _web_tips
                )
            else:
                _nudge = ""

            if _nudge:
                _nudge += _TOOL_ACTION_NUDGE
                # Append nudge to system prompt (preserve user's prompt)
                if system_prompt:
                    system_prompt = system_prompt.rstrip() + "\n\n" + _nudge
                else:
                    system_prompt = _nudge
                # Rebuild gguf_messages with updated system prompt
                gguf_messages = []
                if system_prompt:
                    gguf_messages.append({"role": "system", "content": system_prompt})
                gguf_messages.extend(chat_messages)

            # ── Strip stale tool-call XML from conversation history ─
            for _msg in gguf_messages:
                if _msg.get("role") == "assistant" and isinstance(
                    _msg.get("content"), str
                ):
                    _msg["content"] = _TOOL_XML_RE.sub("", _msg["content"]).strip()

            def gguf_generate_with_tools():
                return llama_backend.generate_chat_completion_with_tools(
                    messages = gguf_messages,
                    tools = tools_to_use,
                    temperature = payload.temperature,
                    top_p = payload.top_p,
                    top_k = payload.top_k,
                    min_p = payload.min_p,
                    max_tokens = payload.max_tokens,
                    repetition_penalty = payload.repetition_penalty,
                    presence_penalty = payload.presence_penalty,
                    cancel_event = cancel_event,
                    enable_thinking = payload.enable_thinking,
                    auto_heal_tool_calls = payload.auto_heal_tool_calls
                    if payload.auto_heal_tool_calls is not None
                    else True,
                    max_tool_iterations = payload.max_tool_calls_per_message
                    if payload.max_tool_calls_per_message is not None
                    else 25,
                    tool_call_timeout = payload.tool_call_timeout
                    if payload.tool_call_timeout is not None
                    else 300,
                    session_id = payload.session_id,
                )

            _tool_sentinel = object()

            async def gguf_tool_stream():
                try:
                    first_chunk = ChatCompletionChunk(
                        id = completion_id,
                        created = created,
                        model = model_name,
                        choices = [
                            ChunkChoice(
                                delta = ChoiceDelta(role = "assistant"),
                                finish_reason = None,
                            )
                        ],
                    )
                    yield f"data: {first_chunk.model_dump_json(exclude_none = True)}\n\n"

                    # Iterate the synchronous generator in a thread so
                    # the event loop stays free for disconnect detection.
                    gen = gguf_generate_with_tools()
                    prev_text = ""
                    _stream_usage = None
                    _stream_timings = None
                    while True:
                        if await request.is_disconnected():
                            cancel_event.set()
                            return

                        event = await asyncio.to_thread(next, gen, _tool_sentinel)
                        if event is _tool_sentinel:
                            break

                        if event["type"] == "status":
                            # Empty status marks an iteration boundary
                            # in the GGUF tool loop (e.g. after a
                            # re-prompt).  Reset the cumulative cursor
                            # so the next assistant turn streams cleanly.
                            if not event["text"]:
                                prev_text = ""
                            # Emit tool status as a custom SSE event
                            # (including empty ones to clear UI badges)
                            status_data = json.dumps(
                                {
                                    "type": "tool_status",
                                    "content": event["text"],
                                }
                            )
                            yield f"data: {status_data}\n\n"
                            continue

                        if event["type"] in ("tool_start", "tool_end"):
                            if event["type"] == "tool_start":
                                prev_text = ""
                            yield f"data: {json.dumps(event)}\n\n"
                            continue

                        if event["type"] == "metadata":
                            _stream_usage = event.get("usage")
                            _stream_timings = event.get("timings")
                            continue

                        # "content" type -- cumulative text
                        # Sanitize the full cumulative then diff against
                        # the last sanitized snapshot so cross-chunk XML
                        # tags are handled correctly.
                        raw_cumulative = event.get("text", "")
                        clean_cumulative = _TOOL_XML_RE.sub("", raw_cumulative)
                        new_text = clean_cumulative[len(prev_text) :]
                        prev_text = clean_cumulative
                        if not new_text:
                            continue
                        chunk = ChatCompletionChunk(
                            id = completion_id,
                            created = created,
                            model = model_name,
                            choices = [
                                ChunkChoice(
                                    delta = ChoiceDelta(content = new_text),
                                    finish_reason = None,
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none = True)}\n\n"

                    final_chunk = ChatCompletionChunk(
                        id = completion_id,
                        created = created,
                        model = model_name,
                        choices = [
                            ChunkChoice(
                                delta = ChoiceDelta(),
                                finish_reason = "stop",
                            )
                        ],
                    )
                    yield f"data: {final_chunk.model_dump_json(exclude_none = True)}\n\n"
                    # Usage chunk (OpenAI-standard: choices=[], usage populated)
                    if _stream_usage or _stream_timings:
                        usage_obj = CompletionUsage(
                            prompt_tokens = (_stream_usage or {}).get("prompt_tokens", 0),
                            completion_tokens = (_stream_usage or {}).get(
                                "completion_tokens", 0
                            ),
                            total_tokens = (_stream_usage or {}).get("total_tokens", 0),
                        )
                        usage_chunk = ChatCompletionChunk(
                            id = completion_id,
                            created = created,
                            model = model_name,
                            choices = [],
                            usage = usage_obj,
                            timings = _stream_timings,
                        )
                        yield f"data: {usage_chunk.model_dump_json(exclude_none = True)}\n\n"
                    yield "data: [DONE]\n\n"

                except asyncio.CancelledError:
                    cancel_event.set()
                    raise
                except Exception as e:
                    import traceback

                    tb = traceback.format_exc()
                    logger.error(f"Error during GGUF tool streaming: {e}\n{tb}")
                    error_chunk = {
                        "error": {
                            "message": _friendly_error(e),
                            "type": "server_error",
                        },
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"

            return StreamingResponse(
                gguf_tool_stream(),
                media_type = "text/event-stream",
                headers = {
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # ── Standard GGUF path (no tools) ─────────────────────

        def gguf_generate():
            return llama_backend.generate_chat_completion(
                messages = gguf_messages,
                image_b64 = image_b64,
                temperature = payload.temperature,
                top_p = payload.top_p,
                top_k = payload.top_k,
                min_p = payload.min_p,
                max_tokens = payload.max_tokens,
                repetition_penalty = payload.repetition_penalty,
                presence_penalty = payload.presence_penalty,
                cancel_event = cancel_event,
                enable_thinking = payload.enable_thinking,
            )

        _gguf_sentinel = object()

        if payload.stream:

            async def gguf_stream_chunks():
                try:
                    # First chunk: role
                    first_chunk = ChatCompletionChunk(
                        id = completion_id,
                        created = created,
                        model = model_name,
                        choices = [
                            ChunkChoice(
                                delta = ChoiceDelta(role = "assistant"),
                                finish_reason = None,
                            )
                        ],
                    )
                    yield f"data: {first_chunk.model_dump_json(exclude_none = True)}\n\n"

                    # Iterate the synchronous generator in a thread so
                    # the event loop stays free for disconnect detection.
                    gen = gguf_generate()
                    prev_text = ""
                    _stream_usage = None
                    _stream_timings = None
                    while True:
                        if await request.is_disconnected():
                            cancel_event.set()
                            return
                        cumulative = await asyncio.to_thread(next, gen, _gguf_sentinel)
                        if cumulative is _gguf_sentinel:
                            break
                        # Capture server metadata for final usage chunk
                        if isinstance(cumulative, dict):
                            if cumulative.get("type") == "metadata":
                                _stream_usage = cumulative.get("usage")
                                _stream_timings = cumulative.get("timings")
                            else:
                                logger.warning(
                                    "gguf_stream_chunks: unexpected dict event: %s",
                                    {
                                        k: v
                                        for k, v in cumulative.items()
                                        if k != "timings"
                                    },
                                )
                            continue
                        new_text = cumulative[len(prev_text) :]
                        prev_text = cumulative
                        if not new_text:
                            continue
                        chunk = ChatCompletionChunk(
                            id = completion_id,
                            created = created,
                            model = model_name,
                            choices = [
                                ChunkChoice(
                                    delta = ChoiceDelta(content = new_text),
                                    finish_reason = None,
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none = True)}\n\n"

                    # Final chunk
                    final_chunk = ChatCompletionChunk(
                        id = completion_id,
                        created = created,
                        model = model_name,
                        choices = [
                            ChunkChoice(
                                delta = ChoiceDelta(),
                                finish_reason = "stop",
                            )
                        ],
                    )
                    yield f"data: {final_chunk.model_dump_json(exclude_none = True)}\n\n"
                    # Usage chunk (OpenAI-standard: choices=[], usage populated)
                    if _stream_usage or _stream_timings:
                        usage_obj = CompletionUsage(
                            prompt_tokens = (_stream_usage or {}).get("prompt_tokens", 0),
                            completion_tokens = (_stream_usage or {}).get(
                                "completion_tokens", 0
                            ),
                            total_tokens = (_stream_usage or {}).get("total_tokens", 0),
                        )
                        usage_chunk = ChatCompletionChunk(
                            id = completion_id,
                            created = created,
                            model = model_name,
                            choices = [],
                            usage = usage_obj,
                            timings = _stream_timings,
                        )
                        yield f"data: {usage_chunk.model_dump_json(exclude_none = True)}\n\n"
                    yield "data: [DONE]\n\n"

                except asyncio.CancelledError:
                    cancel_event.set()
                    raise
                except Exception as e:
                    logger.error(f"Error during GGUF streaming: {e}", exc_info = True)
                    error_chunk = {
                        "error": {
                            "message": _friendly_error(e),
                            "type": "server_error",
                        },
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"

            return StreamingResponse(
                gguf_stream_chunks(),
                media_type = "text/event-stream",
                headers = {
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            try:
                full_text = ""
                for token in gguf_generate():
                    if isinstance(token, dict):
                        continue  # skip metadata dict in non-streaming path
                    full_text = token

                response = ChatCompletion(
                    id = completion_id,
                    created = created,
                    model = model_name,
                    choices = [
                        CompletionChoice(
                            message = CompletionMessage(content = full_text),
                            finish_reason = "stop",
                        )
                    ],
                )
                return JSONResponse(content = response.model_dump())

            except Exception as e:
                logger.error(f"Error during GGUF completion: {e}", exc_info = True)
                raise HTTPException(status_code = 500, detail = str(e))

    # ── Standard Unsloth path ─────────────────────────────────

    # Decode image (from content parts OR legacy field)
    image_b64 = extracted_image_b64 or payload.image_base64
    image = None

    if image_b64:
        try:
            import base64
            from PIL import Image
            from io import BytesIO

            model_info = backend.models.get(backend.active_model_name, {})
            if not model_info.get("is_vision"):
                raise HTTPException(
                    status_code = 400,
                    detail = "Image provided but current model is text-only. Load a vision model.",
                )

            image_data = base64.b64decode(image_b64)
            image = Image.open(BytesIO(image_data))
            image = backend.resize_image(image)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code = 400, detail = f"Failed to decode image: {e}")

    # Shared generation kwargs
    gen_kwargs = dict(
        messages = chat_messages,
        system_prompt = system_prompt,
        image = image,
        temperature = payload.temperature,
        top_p = payload.top_p,
        top_k = payload.top_k,
        min_p = payload.min_p,
        max_new_tokens = payload.max_tokens or 2048,
        repetition_penalty = payload.repetition_penalty,
    )

    # Choose generation path (adapter-controlled or standard)
    cancel_event = threading.Event()

    if payload.use_adapter is not None:

        def generate():
            return backend.generate_with_adapter_control(
                use_adapter = payload.use_adapter,
                cancel_event = cancel_event,
                **gen_kwargs,
            )
    else:

        def generate():
            return backend.generate_chat_response(
                cancel_event = cancel_event, **gen_kwargs
            )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # ── Streaming response ────────────────────────────────────────
    if payload.stream:

        async def stream_chunks():
            try:
                first_chunk = ChatCompletionChunk(
                    id = completion_id,
                    created = created,
                    model = model_name,
                    choices = [
                        ChunkChoice(
                            delta = ChoiceDelta(role = "assistant"),
                            finish_reason = None,
                        )
                    ],
                )
                yield f"data: {first_chunk.model_dump_json(exclude_none = True)}\n\n"

                prev_text = ""
                # Run sync generator in thread pool to avoid blocking
                # the event loop. Critical for compare mode: two SSE
                # requests arrive concurrently but the orchestrator
                # serializes them via _gen_lock. Without run_in_executor
                # the second request's blocking lock acquisition would
                # freeze the entire event loop, stalling both streams.
                _DONE = object()  # sentinel for generator exhaustion
                loop = asyncio.get_event_loop()
                gen = generate()
                while True:
                    # next(gen, _DONE) returns _DONE instead of raising
                    # StopIteration — StopIteration cannot propagate
                    # through asyncio futures (Python limitation).
                    cumulative = await loop.run_in_executor(None, next, gen, _DONE)
                    if cumulative is _DONE:
                        break
                    if await request.is_disconnected():
                        cancel_event.set()
                        backend.reset_generation_state()
                        return
                    new_text = cumulative[len(prev_text) :]
                    prev_text = cumulative
                    if not new_text:
                        continue
                    chunk = ChatCompletionChunk(
                        id = completion_id,
                        created = created,
                        model = model_name,
                        choices = [
                            ChunkChoice(
                                delta = ChoiceDelta(content = new_text),
                                finish_reason = None,
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none = True)}\n\n"

                final_chunk = ChatCompletionChunk(
                    id = completion_id,
                    created = created,
                    model = model_name,
                    choices = [
                        ChunkChoice(
                            delta = ChoiceDelta(),
                            finish_reason = "stop",
                        )
                    ],
                )
                yield f"data: {final_chunk.model_dump_json(exclude_none = True)}\n\n"
                yield "data: [DONE]\n\n"

            except asyncio.CancelledError:
                cancel_event.set()
                backend.reset_generation_state()
                raise
            except Exception as e:
                backend.reset_generation_state()
                logger.error(f"Error during OpenAI streaming: {e}", exc_info = True)
                error_chunk = {
                    "error": {
                        "message": _friendly_error(e),
                        "type": "server_error",
                    },
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"

        return StreamingResponse(
            stream_chunks(),
            media_type = "text/event-stream",
            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Non-streaming response ────────────────────────────────────
    else:
        try:
            full_text = ""
            for token in generate():
                full_text = token

            response = ChatCompletion(
                id = completion_id,
                created = created,
                model = model_name,
                choices = [
                    CompletionChoice(
                        message = CompletionMessage(content = full_text),
                        finish_reason = "stop",
                    )
                ],
            )
            return JSONResponse(content = response.model_dump())

        except Exception as e:
            backend.reset_generation_state()
            logger.error(f"Error during OpenAI completion: {e}", exc_info = True)
            raise HTTPException(status_code = 500, detail = str(e))


# =====================================================================
# Sandbox file serving  (/sandbox/{session_id}/{filename})
# =====================================================================

_SANDBOX_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


@router.get("/sandbox/{session_id}/{filename}")
async def serve_sandbox_file(
    session_id: str,
    filename: str,
    request: Request,
    token: Optional[str] = None,
):
    """
    Serve image files created by Python tool execution.

    Accepts auth via Authorization header OR ?token= query param
    (needed because <img src> cannot send custom headers).
    """
    from fastapi.responses import FileResponse

    # ── Authentication (header or query param) ──────────────────
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        jwt_token = auth_header[7:]
    elif token:
        jwt_token = token
    else:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Missing authentication token",
        )
    from fastapi.security import HTTPAuthorizationCredentials

    creds = HTTPAuthorizationCredentials(scheme = "Bearer", credentials = jwt_token)
    await get_current_subject(creds)

    # ── Filename sanitization ───────────────────────────────────
    safe_filename = os.path.basename(filename)
    if not safe_filename or safe_filename in (".", ".."):
        raise HTTPException(status_code = 404, detail = "Not found")

    # ── Extension allowlist ─────────────────────────────────────
    ext = os.path.splitext(safe_filename)[1].lower()
    media_type = _SANDBOX_MEDIA_TYPES.get(ext)
    if not media_type:
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail = "File type not allowed",
        )

    # ── Path containment check ──────────────────────────────────
    home = os.path.expanduser("~")
    sandbox_root = os.path.realpath(os.path.join(home, "studio_sandbox"))
    safe_session = os.path.basename(session_id.replace("..", ""))
    if not safe_session:
        raise HTTPException(status_code = 404, detail = "Not found")

    file_path = os.path.realpath(
        os.path.join(sandbox_root, safe_session, safe_filename)
    )
    if not file_path.startswith(sandbox_root + os.sep):
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail = "Access denied",
        )

    if not os.path.isfile(file_path):
        raise HTTPException(status_code = 404, detail = "Not found")

    return FileResponse(
        path = file_path,
        media_type = media_type,
        headers = {
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


# =====================================================================
# OpenAI-Compatible Models Listing  (/models → /v1/models)
# =====================================================================


@router.get("/models")
async def openai_list_models(
    current_subject: str = Depends(get_current_subject),
):
    """
    OpenAI-compatible model listing endpoint.

    Returns the currently loaded model in the format expected by
    OpenAI-compatible clients (``GET /v1/models``).
    """
    models = []

    # Check GGUF backend
    llama_backend = get_llama_cpp_backend()
    if llama_backend.is_loaded:
        models.append(
            {
                "id": llama_backend.model_identifier,
                "object": "model",
                "owned_by": "local",
            }
        )

    # Check Unsloth backend
    backend = get_inference_backend()
    if backend.active_model_name:
        models.append(
            {
                "id": backend.active_model_name,
                "object": "model",
                "owned_by": "local",
            }
        )

    return {"object": "list", "data": models}
