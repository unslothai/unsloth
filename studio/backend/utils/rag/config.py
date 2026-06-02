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

# A single text embedder handles retrieval. PDF figures are captioned at ingest
# (chat VLM or helper gemma-3n fallback) and spliced into the page markdown
# before chunking, so the 384-d text embedder covers figure content too.
def resolve_embedder() -> str:
    """The configured RAG embedder."""
    return RAG_EMBEDDING_MODEL


RAG_CHUNK_SIZE: int = _env_int("UNSLOTH_RAG_CHUNK_SIZE", 512)
RAG_CHUNK_OVERLAP: int = _env_int("UNSLOTH_RAG_CHUNK_OVERLAP", 64)

RAG_TOP_K_BM25: int = _env_int("UNSLOTH_RAG_TOP_K_BM25", 30)
RAG_TOP_K_DENSE: int = _env_int("UNSLOTH_RAG_TOP_K_DENSE", 30)
RAG_TOP_K_HYBRID: int = _env_int("UNSLOTH_RAG_TOP_K_HYBRID", 10)

RAG_RRF_K: int = _env_int("UNSLOTH_RAG_RRF_K", 60)

RAG_MAX_UPLOAD_MB: int = _env_int("UNSLOTH_RAG_MAX_UPLOAD_MB", 50)

RAG_EMBED_BATCH_SIZE: int = _env_int("UNSLOTH_RAG_EMBED_BATCH_SIZE", 32)

RAG_UPLOAD_EXTS: frozenset[str] = frozenset(
    {".pdf", ".txt", ".md", ".markdown", ".docx", ".html", ".htm"}
)
