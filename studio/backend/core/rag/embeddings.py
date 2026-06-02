# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RAG embedder singleton. Independent of the chat InferenceBackend."""

from __future__ import annotations

import logging
import threading
from typing import Any

from utils.rag.config import RAG_EMBED_BATCH_SIZE, RAG_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_model: Any | None = None
_model_name: str | None = None
_embedding_dim: int | None = None


def _load(model_name: str) -> Any:
    logger.info("Loading RAG embedder: %s", model_name)

    from unsloth import FastSentenceTransformer

    # trust_remote_code: nomic-embed-text-v1.5 needs custom modeling for 8K ctx.
    return FastSentenceTransformer.from_pretrained(
        model_name,
        for_inference = True,
        trust_remote_code = True,
    )


def get_embedder(model_name: str | None = None) -> Any:
    global _model, _model_name, _embedding_dim
    target = model_name or RAG_EMBEDDING_MODEL
    with _lock:
        if _model is None or _model_name != target:
            _model = _load(target)
            _model_name = target
            try:
                _embedding_dim = int(_model.get_sentence_embedding_dimension())
            except Exception:
                _embedding_dim = None
        return _model


def get_embedding_dim(model_name: str | None = None) -> int:
    model = get_embedder(model_name)
    global _embedding_dim
    if _embedding_dim is None:
        _embedding_dim = int(model.get_sentence_embedding_dimension())
    return _embedding_dim


def get_active_model_name() -> str | None:
    return _model_name


def encode(
    texts: list[str],
    *,
    model_name: str | None = None,
    batch_size: int | None = None,
    normalize: bool = True,
):
    model = get_embedder(model_name)
    return model.encode(
        texts,
        batch_size = batch_size or RAG_EMBED_BATCH_SIZE,
        normalize_embeddings = normalize,
        convert_to_numpy = True,
        show_progress_bar = False,
    )


def token_counter(model_name: str | None = None):
    """Return a token-count callable backed by the embedder's tokenizer."""
    model = get_embedder(model_name)

    def _count(text: str) -> int:
        try:
            tokens = model.tokenize([text])
            ids = tokens.get("input_ids")
            if ids is None:
                return max(1, len(text) // 4)
            return int(ids.shape[1])
        except Exception:
            return max(1, len(text) // 4)

    return _count
