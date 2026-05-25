# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Optional cross-encoder reranking stage.

Off by default. Callers opt in per request via ``enable_rerank`` on
``SearchRequest``. The reranker model is lazy-loaded on first opt-in
query and competes with the active chat model for GPU memory — keeping
it opt-in protects chat latency on smaller GPUs.

``sentence_transformers.CrossEncoder`` has no unsloth wrapper today;
this module loads it directly. A future ``FastCrossEncoder`` addition
to ``unsloth/models/sentence_transformer.py`` would slot in here by
replacing the import in ``_load``.
"""

from __future__ import annotations

import gc
import threading
from typing import Any

from loggers import get_logger
from utils.rag.config import RAG_RERANK_BATCH_SIZE, RAG_RERANKER_MODEL

from .retrieval import Hit

logger = get_logger(__name__)

_lock = threading.Lock()
_model: Any | None = None
_model_name: str | None = None


def _load(model_name: str) -> Any:
    from sentence_transformers import CrossEncoder

    logger.info("Loading RAG reranker: %s", model_name)
    return CrossEncoder(model_name)


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
    """Drop the reranker reference and trigger a GC pass.

    Useful when memory pressure is high — callers can free the
    reranker without restarting the studio process. Next ``rerank``
    call lazy-loads it again.
    """
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
    """Re-order ``pairs`` by CrossEncoder relevance to ``query``.

    Each pair is ``(Hit, chunk_text)``. Returns Hits with the new
    cross-encoder scores attached. If ``top_k`` is given, truncates.
    """
    if not pairs:
        return []
    # CrossEncoder is text-only; image-kind hits get appended at the end
    # in their original relative order so they're never dropped, just
    # never reranked. Caption-kind hits are eligible (they carry text).
    text_pairs = [(h, t) for h, t in pairs if h.kind != "image"]
    image_hits = [h for h, _t in pairs if h.kind == "image"]

    model = get_reranker(model_name)
    if text_pairs:
        inputs = [(query, text) for _, text in text_pairs]
        scores = model.predict(
            inputs,
            batch_size = RAG_RERANK_BATCH_SIZE,
            show_progress_bar = False,
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
