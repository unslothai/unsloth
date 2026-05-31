# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


RAG_EMBEDDING_MODEL: str = (
    os.environ.get("UNSLOTH_RAG_EMBEDDING_MODEL", "").strip()
    or "BAAI/bge-small-en-v1.5"
)

# Default embedder per (mode, chunking). (multimodal, late) is rejected at
# KB-create time in routes/rag.py.
#
# Text mode is the default: PDF figures are captioned at ingest (chat VLM or
# helper gemma-3n fallback) and spliced into the page markdown before chunking,
# so a single 384-d text embedder handles retrieval. Multimodal adds image-vector
# rows on top via Qwen3-VL-Embedding-2B (2 B, 2048-d, no CLIP text cap — full
# 512-token chunks embed losslessly).
#
# Alternative multimodal embedders kept for manual override:
#   - "BAAI/BGE-VL-large" — smaller (~400 M / 768-d) but CLIP-family with a
#     77-token text cap; routed via `_BGEVLAdapter` in core/rag/embeddings.py.
RAG_EMBEDDER_MATRIX: dict[tuple[str, str], str] = {
    ("text", "standard"): "BAAI/bge-small-en-v1.5",
    ("text", "late"): "nomic-ai/nomic-embed-text-v1.5",
    ("multimodal", "standard"): "Qwen/Qwen3-VL-Embedding-2B",
}


def resolve_embedder(mode: str, chunking_strategy: str) -> str:
    """Embedder for (mode, chunking); unknown combos fall back to RAG_EMBEDDING_MODEL."""
    return RAG_EMBEDDER_MATRIX.get(
        (mode, chunking_strategy),
        RAG_EMBEDDING_MODEL,
    )


RAG_CHUNK_SIZE: int = _env_int("UNSLOTH_RAG_CHUNK_SIZE", 512)
RAG_CHUNK_OVERLAP: int = _env_int("UNSLOTH_RAG_CHUNK_OVERLAP", 64)

RAG_TOP_K_BM25: int = _env_int("UNSLOTH_RAG_TOP_K_BM25", 30)
RAG_TOP_K_DENSE: int = _env_int("UNSLOTH_RAG_TOP_K_DENSE", 30)
RAG_TOP_K_HYBRID: int = _env_int("UNSLOTH_RAG_TOP_K_HYBRID", 10)

RAG_RRF_K: int = _env_int("UNSLOTH_RAG_RRF_K", 60)

RAG_MAX_UPLOAD_MB: int = _env_int("UNSLOTH_RAG_MAX_UPLOAD_MB", 50)

RAG_EMBED_BATCH_SIZE: int = _env_int("UNSLOTH_RAG_EMBED_BATCH_SIZE", 32)

RAG_RERANKER_MODEL: str = (
    os.environ.get("UNSLOTH_RAG_RERANKER_MODEL", "").strip() or "BAAI/bge-reranker-base"
)
RAG_RERANK_CANDIDATE_K: int = _env_int("UNSLOTH_RAG_RERANK_CANDIDATE_K", 50)
RAG_RERANK_BATCH_SIZE: int = _env_int("UNSLOTH_RAG_RERANK_BATCH_SIZE", 16)

RAG_UPLOAD_EXTS: frozenset[str] = frozenset(
    {".pdf", ".txt", ".md", ".markdown", ".docx", ".html", ".htm"}
)
