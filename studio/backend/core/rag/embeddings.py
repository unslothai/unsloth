# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Embedding model singleton for RAG.

Loads the configured embedder via Unsloth's ``FastSentenceTransformer``
wrapper with ``for_inference=True`` (which returns a plain
``sentence_transformers.SentenceTransformer`` instance with proper dtype
and device handling). Lifecycle is fully independent of the chat
``InferenceBackend`` so loading an embedder cannot evict the active
chat model.
"""

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
    from unsloth import FastSentenceTransformer

    logger.info("Loading RAG embedder: %s", model_name)
    return FastSentenceTransformer.from_pretrained(
        model_name,
        for_inference = True,
    )


def get_embedder(model_name: str | None = None) -> Any:
    """Return the cached SentenceTransformer, loading it on first use."""
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
    """Return a ``len(tokenize(text))`` callable using the embedder's tokenizer.

    Avoid loading the model just for chunking by reaching through the
    SentenceTransformer's ``tokenize`` API.
    """
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


# ------------------------------------------------------------------
# Late chunking (Phase 3B-late)
# ------------------------------------------------------------------

_LATE_WINDOW_OVERLAP_TOKENS = 512


def late_chunk_encode(
    doc_text: str,
    char_spans: list[tuple[int, int]],
    *,
    model_name: str | None = None,
    normalize: bool = True,
):
    """Embed each chunk via late-chunking pooling.

    Single forward pass over the full document, then mean-pool the
    token embeddings whose offset ranges fall inside each chunk's
    char span. Chunks therefore carry full-document context via the
    encoder's bidirectional attention — Jina's published technique,
    works with any encoder that exposes per-token outputs.

    When the doc exceeds the embedder's context, falls back to
    windowed late chunking with a 512-token overlap between windows
    so cross-window context is partially preserved.
    """
    import numpy as np

    if not char_spans:
        return []
    model = get_embedder(model_name)
    tokenizer = model.tokenizer
    max_length = int(getattr(model, "max_seq_length", None) or 8192)

    encoded = tokenizer(
        doc_text,
        return_tensors = "pt",
        return_offsets_mapping = True,
        add_special_tokens = True,
        truncation = False,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    n_tokens = int(encoded["input_ids"].shape[1])

    if n_tokens <= max_length:
        token_embeddings = _encode_tokens(model, encoded)
        return _pool_spans(
            token_embeddings,
            offsets,
            char_spans,
            normalize = normalize,
            np_module = np,
            model = model,
            doc_text = doc_text,
        )

    logger.info(
        "Late chunking: doc has %d tokens > model max %d; using windowed pass",
        n_tokens,
        max_length,
    )
    return _windowed_late_chunk_encode(
        doc_text = doc_text,
        char_spans = char_spans,
        model = model,
        max_length = max_length,
        normalize = normalize,
        np_module = np,
    )


def _encode_tokens(model, encoded):
    """Run the embedder's underlying transformer to get per-token last_hidden_state."""
    import torch

    transformer = model[0].auto_model
    device = next(transformer.parameters()).device
    inputs_on_device = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = transformer(**inputs_on_device)
    return outputs.last_hidden_state[0].detach().cpu().numpy()


def _pool_spans(
    token_embeddings,
    offsets,
    char_spans,
    *,
    normalize: bool,
    np_module,
    model,
    doc_text: str,
    token_index_offset: int = 0,
):
    """Mean-pool token embeddings per (char_start, char_end) span.

    `token_index_offset` shifts char_span-derived token indices into
    a sub-window's local frame (used by the windowed code path).
    """
    vectors = []
    n_rows = token_embeddings.shape[0]
    for char_start, char_end in char_spans:
        # Special tokens (CLS / SEP) report offsets (0, 0) — exclude them.
        indices = [
            i - token_index_offset
            for i, (ts, te) in enumerate(offsets)
            if te > ts and te > char_start and ts < char_end
        ]
        indices = [i for i in indices if 0 <= i < n_rows]
        if not indices:
            # Fall back to a standalone encode of the chunk text — rare
            # (would mean tokenizer produced zero non-special tokens for
            # the span), but keeps the pipeline alive.
            vec = model.encode(
                doc_text[char_start:char_end],
                normalize_embeddings = normalize,
                convert_to_numpy = True,
                show_progress_bar = False,
            )
            vectors.append(vec)
            continue
        pooled = token_embeddings[indices].mean(axis = 0)
        if normalize:
            denom = float(np_module.linalg.norm(pooled))
            if denom > 0:
                pooled = pooled / denom
        vectors.append(pooled)
    return vectors


def _windowed_late_chunk_encode(
    *,
    doc_text: str,
    char_spans: list[tuple[int, int]],
    model,
    max_length: int,
    normalize: bool,
    np_module,
):
    """Doc exceeds context window — slice into overlapping windows.

    Each chunk is pooled against the window that contains the most of
    its tokens. The 512-token window overlap means chunks near a
    boundary still see context from both sides.
    """
    import torch

    tokenizer = model.tokenizer
    transformer = model[0].auto_model
    device = next(transformer.parameters()).device

    full = tokenizer(
        doc_text,
        return_tensors = "pt",
        return_offsets_mapping = True,
        add_special_tokens = False,
        truncation = False,
    )
    all_input_ids = full["input_ids"][0]
    all_offsets = full["offset_mapping"][0].tolist()
    n_tokens = int(all_input_ids.shape[0])
    stride = max(1, max_length - _LATE_WINDOW_OVERLAP_TOKENS)

    # Build (start_token, end_token) windows.
    windows: list[tuple[int, int]] = []
    pos = 0
    while pos < n_tokens:
        end = min(pos + max_length, n_tokens)
        windows.append((pos, end))
        if end >= n_tokens:
            break
        pos += stride

    # Cache window → token embeddings (only encode when needed).
    window_embeddings: dict[int, "np_module.ndarray"] = {}

    def _window_embeddings(window_index: int):
        if window_index in window_embeddings:
            return window_embeddings[window_index]
        ws, we = windows[window_index]
        win_ids = all_input_ids[ws:we].unsqueeze(0).to(device)
        win_attn = torch.ones_like(win_ids)
        with torch.no_grad():
            outputs = transformer(input_ids = win_ids, attention_mask = win_attn)
        emb = outputs.last_hidden_state[0].detach().cpu().numpy()
        window_embeddings[window_index] = emb
        return emb

    vectors = []
    for char_start, char_end in char_spans:
        # Collect global token indices in the chunk.
        chunk_token_indices = [
            i
            for i, (ts, te) in enumerate(all_offsets)
            if te > ts and te > char_start and ts < char_end
        ]
        if not chunk_token_indices:
            vec = model.encode(
                doc_text[char_start:char_end],
                normalize_embeddings = normalize,
                convert_to_numpy = True,
                show_progress_bar = False,
            )
            vectors.append(vec)
            continue
        # Pick the window covering the most of this chunk's tokens.
        best_window = 0
        best_overlap = 0
        for wi, (ws, we) in enumerate(windows):
            overlap = sum(1 for ti in chunk_token_indices if ws <= ti < we)
            if overlap > best_overlap:
                best_overlap = overlap
                best_window = wi
        ws, _we = windows[best_window]
        emb = _window_embeddings(best_window)
        local_indices = [
            ti - ws
            for ti in chunk_token_indices
            if ws <= ti < ws + emb.shape[0]
        ]
        if not local_indices:
            vec = model.encode(
                doc_text[char_start:char_end],
                normalize_embeddings = normalize,
                convert_to_numpy = True,
                show_progress_bar = False,
            )
            vectors.append(vec)
            continue
        pooled = emb[local_indices].mean(axis = 0)
        if normalize:
            denom = float(np_module.linalg.norm(pooled))
            if denom > 0:
                pooled = pooled / denom
        vectors.append(pooled)
    return vectors
