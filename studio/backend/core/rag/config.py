# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RAG config; every value is env-overridable."""

from __future__ import annotations

import os
import re

DEFAULT_EMBEDDING_MODEL = "unsloth/bge-small-en-v1.5"
EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
# Under bge's 512 limit, leaving headroom for the 2 special tokens (else overflow:
# llama-server 500s, ST truncates). Keep <= embedder_max - ~12.
CHUNK_TOKENS = int(os.environ.get("RAG_CHUNK_TOKENS", "500"))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "64"))
TOP_K_LEXICAL = int(os.environ.get("RAG_TOP_K_LEXICAL", "30"))
TOP_K_DENSE = int(os.environ.get("RAG_TOP_K_DENSE", "30"))
TOP_K_HYBRID = int(os.environ.get("RAG_TOP_K_HYBRID", "10"))
RRF_K = int(os.environ.get("RAG_RRF_K", "60"))

# Whole-document context: a thread-attached file under the token budget is injected
# in full (every chunk, in order) instead of top-K retrieval; above it, use retrieval.
THREAD_WHOLE_DOC = os.environ.get("RAG_THREAD_WHOLE_DOC", "1") == "1"
WHOLE_DOC_MAX_TOKENS = int(os.environ.get("RAG_WHOLE_DOC_MAX_TOKENS", "6000"))

UPLOAD_EXTS = {".pdf", ".txt", ".md", ".markdown", ".docx", ".html", ".htm"}
# Reject uploads larger than this, so one pathological file can't drive unbounded parse
# + vision work at ingest. 0 disables the cap. Default 200 MB.
MAX_UPLOAD_BYTES = int(os.environ.get("RAG_MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))

# Extract PDF text as layout-aware Markdown (pymupdf4llm) instead of flat text, so
# tables, headings and lists survive into chunks and retrieval. Falls back to plain
# PyMuPDF text when off, when pymupdf4llm is missing, or when extraction fails.
PDF_MARKDOWN = os.environ.get("RAG_PDF_MARKDOWN", "1") == "1"

# Figure captioning via the loaded vision model: detected figures are transcribed +
# described so they become searchable. On by default, a no-op without a vision model;
# the chat's "Describe figures & charts" toggle overrides it per upload.
CAPTION_IMAGES = os.environ.get("RAG_CAPTION_IMAGES", "1") == "1"
# Total per-document tile budget (figure-bearing pages are tiled, see below).
CAPTION_MAX_IMAGES = int(os.environ.get("RAG_CAPTION_MAX_IMAGES", "24"))
CAPTION_TIMEOUT_S = float(os.environ.get("RAG_CAPTION_TIMEOUT_S", "60"))
# Larger than a one-line caption since captions transcribe every label. FIGURE_DPI is
# high enough to keep small box/axis labels legible when tiles are rendered.
CAPTION_MAX_TOKENS = int(os.environ.get("RAG_CAPTION_MAX_TOKENS", "768"))
FIGURE_DPI = int(os.environ.get("RAG_FIGURE_DPI", "200"))
# Figure pages are tiled into an overlapping ROWS x COLS grid of high-DPI tiles (plus
# an optional full page), so small labels and every sub-figure are covered without
# exact region detection. MAX_PAGES bounds figure pages; MAX_IMAGES bounds total tiles.
FIGURE_TILE_ROWS = int(os.environ.get("RAG_FIGURE_TILE_ROWS", "2"))
FIGURE_TILE_COLS = int(os.environ.get("RAG_FIGURE_TILE_COLS", "2"))
FIGURE_TILE_OVERLAP = float(os.environ.get("RAG_FIGURE_TILE_OVERLAP", "0.12"))
FIGURE_FULLPAGE = os.environ.get("RAG_FIGURE_FULLPAGE", "1") == "1"
CAPTION_MAX_PAGES = int(os.environ.get("RAG_CAPTION_MAX_PAGES", "4"))

# Scanned-PDF OCR: a page with little extractable text is rendered and transcribed by
# the vision model so it becomes searchable. Needs a vision model, else skipped (page
# stays empty). MIN_CHARS is the text length below which a page is treated as scanned.
OCR_SCANNED = os.environ.get("RAG_OCR_SCANNED", "1") == "1"
OCR_MIN_CHARS = int(os.environ.get("RAG_OCR_MIN_CHARS", "16"))
OCR_MAX_PAGES = int(os.environ.get("RAG_OCR_MAX_PAGES", "20"))
OCR_DPI = int(os.environ.get("RAG_OCR_DPI", "150"))
OCR_TIMEOUT_S = float(os.environ.get("RAG_OCR_TIMEOUT_S", "60"))
OCR_MAX_TOKENS = int(os.environ.get("RAG_OCR_MAX_TOKENS", "2048"))

# Embedder backend. "auto": sentence-transformers on a CUDA/ROCm GPU (torch fp16
# wins bulk indexing), else torch-free GGUF llama-server. Switching backends changes
# the vectors, so the index must be rebuilt.
EMBED_BACKEND = os.environ.get("RAG_EMBED_BACKEND", "auto")


def effective_embedding_model() -> str:
    """The embedding model actually in use: the persisted Settings override when
    one is stored, else ``EMBEDDING_MODEL`` (env/default). Read at call time so a
    Settings change applies without a restart."""
    try:
        from utils.embedding_model_settings import get_rag_embedding_model
        return get_rag_embedding_model()
    except Exception:  # noqa: BLE001 - settings store unavailable (tests, early boot)
        return EMBEDDING_MODEL


def _names_gguf(model: str) -> bool:
    """True when "gguf" appears as a whole name segment, so plain substrings
    like "bigguf" don't count."""
    return "gguf" in re.split(r"[^a-z0-9]+", model.lower())


def gguf_repo_for_embedding_model(model: str) -> str:
    """GGUF repo for ``model``, honoring an explicit companion override."""
    if "RAG_EMBED_GGUF_REPO" in os.environ:
        return EMBED_GGUF_REPO
    if model == DEFAULT_EMBEDDING_MODEL:
        return EMBED_GGUF_REPO
    if _names_gguf(model):
        return model
    return f"{model}-GGUF"


def default_gguf_repo() -> str:
    """GGUF companion for the env/default embedding model."""
    return gguf_repo_for_embedding_model(EMBEDDING_MODEL)


def effective_gguf_repo() -> str:
    """GGUF repo for the llama-server backend, tracking the effective model.

    An explicit ``RAG_EMBED_GGUF_REPO`` env always wins. Otherwise any custom
    model (saved in Settings or via ``RAG_EMBEDDING_MODEL``) maps to its
    ``-GGUF`` companion repo (the unsloth convention the default pair follows),
    or is used as-is when it already names a GGUF repo.
    """
    return gguf_repo_for_embedding_model(effective_embedding_model())


# llama-server backend only. F16 over Q8_0: faster (no per-block dequant for this
# tiny model) and exact vs fp32, for ~30MB more on disk.
EMBED_GGUF_REPO = os.environ.get("RAG_EMBED_GGUF_REPO", "unsloth/bge-small-en-v1.5-GGUF")
EMBED_GGUF_VARIANT = os.environ.get("RAG_EMBED_GGUF_VARIANT", "F16")
EMBED_DEVICE = os.environ.get("RAG_EMBED_DEVICE", "auto")  # "auto" | "gpu" | "cpu"
EMBED_HOST = os.environ.get("RAG_EMBED_HOST", "127.0.0.1")
EMBED_PORT = int(os.environ.get("RAG_EMBED_PORT", "0"))  # 0 = auto-pick a free port
EMBED_BATCH = int(os.environ.get("RAG_EMBED_BATCH", "64"))
EMBED_STARTUP_TIMEOUT_S = float(os.environ.get("RAG_EMBED_STARTUP_TIMEOUT_S", "120"))
EMBED_REQUEST_TIMEOUT_S = float(os.environ.get("RAG_EMBED_REQUEST_TIMEOUT_S", "60"))
