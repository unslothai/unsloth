# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in CrossEncoder reranker (off by default; shares GPU with chat model)."""

from __future__ import annotations

import gc
import sys
import threading
import time
from typing import Any

from loggers import get_logger
from utils.rag.config import RAG_RERANK_BATCH_SIZE, RAG_RERANKER_MODEL

from .retrieval import Hit

logger = get_logger(__name__)

_lock = threading.Lock()
_model: Any | None = None
_model_name: str | None = None


def _resolve_device() -> str:
    """Prefer CUDA when available; otherwise CPU. Explicit so we don't rely
    on sentence-transformers' auto-detect (which historically picks CPU when
    CUDA_VISIBLE_DEVICES is set funny)."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def _load(model_name: str) -> Any:
    # Stderr print is unconditional so we can see this line even when
    # structlog routing is misbehaving — diagnostics for a previously
    # invisible hang.
    print(
        f"[rag.reranker] _load entered: model={model_name}",
        file = sys.stderr,
        flush = True,
    )
    from sentence_transformers import CrossEncoder

    device = _resolve_device()
    print(
        f"[rag.reranker] device resolved: {device}",
        file = sys.stderr,
        flush = True,
    )
    logger.info(
        "Loading RAG reranker",
        model = model_name,
        device = device,
    )
    started = time.perf_counter()
    print(
        f"[rag.reranker] calling CrossEncoder(...) on {device}",
        file = sys.stderr,
        flush = True,
    )
    model = CrossEncoder(model_name, device = device)
    elapsed = round(time.perf_counter() - started, 2)
    print(
        f"[rag.reranker] CrossEncoder returned in {elapsed}s",
        file = sys.stderr,
        flush = True,
    )
    logger.info(
        "RAG reranker loaded",
        model = model_name,
        device = device,
        elapsed_seconds = elapsed,
    )
    return model


def precache_reranker(model_name: str | None = None) -> None:
    """Download reranker weights into the HF cache (no instantiation).

    Mirrors ``precache_helper_gguf``: runs in a background thread on
    FastAPI startup so the first user-facing rerank doesn't pay the
    ~1.1 GB download. Safe to call when the model is already cached
    (huggingface_hub no-ops on existing files).
    """
    target = model_name or RAG_RERANKER_MODEL
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
        logger.info("Pre-caching RAG reranker", model = target)
        started = time.perf_counter()
        snapshot_download(repo_id = target, repo_type = "model")
        logger.info(
            "RAG reranker cached",
            model = target,
            elapsed_seconds = round(time.perf_counter() - started, 2),
        )
    except Exception as exc:  # noqa: BLE001
        # Non-critical: the lazy loader will retry the download on first
        # use. We log so the user can see what happened.
        logger.warning(
            "RAG reranker precache failed; will download lazily",
            model = target,
            error = str(exc),
        )


def get_reranker(model_name: str | None = None) -> Any:
    global _model, _model_name
    target = model_name or RAG_RERANKER_MODEL
    with _lock:
        if _model is None or _model_name != target:
            unload()
            _model = _load(target)
            _model_name = target
        return _model


def unload() -> None:
    """Drop the reranker; next call lazy-loads again."""
    global _model, _model_name
    with _lock:
        if _model is not None:
            _model = None
            _model_name = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


def rerank(
    query: str,
    pairs: list[tuple[Hit, str]],
    *,
    model_name: str | None = None,
    top_k: int | None = None,
) -> list[Hit]:
    """Re-order (Hit, text) pairs by CrossEncoder score; image hits are appended last."""
    if not pairs:
        return []
    text_pairs = [(h, t) for h, t in pairs if h.kind != "image"]
    image_hits = [h for h, _t in pairs if h.kind == "image"]

    print(
        f"[rag.reranker] rerank entered: n_pairs={len(text_pairs)}",
        file = sys.stderr,
        flush = True,
    )
    model = get_reranker(model_name)
    print(
        "[rag.reranker] reranker model in hand",
        file = sys.stderr,
        flush = True,
    )
    if text_pairs:
        inputs = [(query, text) for _, text in text_pairs]
        print(
            f"[rag.reranker] predict starting: n_inputs={len(inputs)} "
            f"batch_size={RAG_RERANK_BATCH_SIZE}",
            file = sys.stderr,
            flush = True,
        )
        logger.info(
            "RAG reranker predict starting",
            n_inputs = len(inputs),
            batch_size = RAG_RERANK_BATCH_SIZE,
        )
        started = time.perf_counter()
        scores = model.predict(
            inputs,
            batch_size = RAG_RERANK_BATCH_SIZE,
            show_progress_bar = False,
        )
        elapsed = round(time.perf_counter() - started, 2)
        print(
            f"[rag.reranker] predict done in {elapsed}s",
            file = sys.stderr,
            flush = True,
        )
        logger.info(
            "RAG reranker predict done",
            n_inputs = len(inputs),
            elapsed_seconds = elapsed,
        )
        ranked = sorted(
            zip(text_pairs, scores),
            key = lambda item: float(item[1]),
            reverse = True,
        )
        reranked_text = [
            Hit(
                chunk_id = h.chunk_id,
                score = float(s),
                document_id = h.document_id,
                chunk_index = h.chunk_index,
                kind = h.kind,
            )
            for (h, _t), s in ranked
        ]
    else:
        reranked_text = []
    out: list[Hit] = reranked_text + image_hits
    if top_k is not None:
        out = out[:top_k]
    return out
