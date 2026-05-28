# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Query decomposition for external-provider RAG prefetch.

External providers can't run studio's `search_knowledge_base` tool loop, so
for the prefetch path we retrieve up-front. To match the multi-query
behaviour local models get from the tool-loop system prompt, we spin up the
pre-cached helper GGUF (``unsloth/gemma-4-E2B-it-GGUF``, the same model the
captioner uses) momentarily, ask it to split the user's question into up to
three focused search queries, then unload it.

The helper llama-server is its own subprocess (llama.cpp), spawned with
``kill_orphans=False`` so it can't reap a resident chat model, and always
unloaded in a ``finally``. Any failure (helper can't load, request errors,
empty output) falls back to ``[query]`` — a single raw retrieval — so RAG
prefetch never hard-fails on decomposition.
"""

from __future__ import annotations

from typing import Any, Optional

from loggers import get_logger

# Reuse the exact model the captioner / precache path already downloads.
from core.rag.captioner import _HELPER_REPO, _HELPER_VARIANT, _HELPER_MODEL_NAME

logger = get_logger(__name__)

_MAX_QUERIES = 3
_REQUEST_TIMEOUT_SECONDS = 60.0
_PROMPT = (
    "Split the user's question into up to 3 focused search queries for "
    "retrieving relevant passages from their documents. Prefer fewer when "
    "the question is narrow — one is fine. Output ONLY the queries, one per "
    "line, no numbering, no preamble."
)


def _load_helper() -> Optional[tuple[Any, str, str]]:
    """Spawn a private text-only helper llama-server. Caller unloads it."""
    try:
        from core.inference.llama_cpp import LlamaCppBackend

        # kill_orphans=False: a resident chat-model llama-server (if any)
        # must not be reaped by this transient instance.
        backend = LlamaCppBackend(kill_orphans = False)
        ok = backend.load_model(
            hf_repo = _HELPER_REPO,
            hf_variant = _HELPER_VARIANT,
            model_identifier = f"rag-querygen:{_HELPER_REPO}:{_HELPER_VARIANT}",
            is_vision = False,
            n_ctx = 4096,
            n_gpu_layers = -1,
        )
        if not ok:
            logger.warning("RAG query-decompose: helper failed to start")
            return None
        return backend, backend.base_url, _HELPER_MODEL_NAME
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG query-decompose: helper load raised", error = str(exc))
        return None


def _parse_queries(raw: str, fallback: str) -> list[str]:
    out: list[str] = []
    for line in (raw or "").splitlines():
        # Strip common list markers the model might emit despite the prompt.
        cleaned = line.strip().lstrip("-*0123456789.) ").strip()
        if cleaned:
            out.append(cleaned)
        if len(out) >= _MAX_QUERIES:
            break
    return out or [fallback]


def decompose_query(query: str) -> list[str]:
    """Return up to 3 focused search queries; ``[query]`` on any failure.

    Loads the helper, asks for the decomposition, unloads. Never raises.
    """
    q = (query or "").strip()
    if not q:
        return []

    import httpx

    loaded = _load_helper()
    if loaded is None:
        return [q]
    backend, base_url, model_name = loaded
    try:
        endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": _PROMPT},
                {"role": "user", "content": q},
            ],
            "max_tokens": 160,
            "temperature": 0.0,
            # gemma-4 is a reasoning model; thinking would eat the budget and
            # emit no visible queries (same issue the captioner hit).
            "chat_template_kwargs": {"enable_thinking": False},
        }
        with httpx.Client(timeout = _REQUEST_TIMEOUT_SECONDS) as client:
            response = client.post(endpoint, json = payload)
            response.raise_for_status()
            data = response.json()
        content = (
            data.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        queries = _parse_queries(content if isinstance(content, str) else "", q)
        logger.info("RAG query-decompose: produced queries", n = len(queries))
        return queries
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG query-decompose: request failed", error = str(exc))
        return [q]
    finally:
        try:
            backend.unload_model()
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAG query-decompose: helper unload failed", error = str(exc))
