# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RAG config; every value is env-overridable."""

from __future__ import annotations

import os

EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "unsloth/bge-small-en-v1.5")
# Under bge's 512 limit, leaving headroom for the 2 special tokens (else overflow:
# llama-server 500s, ST truncates). Keep <= embedder_max - ~12.
CHUNK_TOKENS = int(os.environ.get("RAG_CHUNK_TOKENS", "500"))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "64"))
TOP_K_LEXICAL = int(os.environ.get("RAG_TOP_K_LEXICAL", "30"))
TOP_K_DENSE = int(os.environ.get("RAG_TOP_K_DENSE", "30"))
TOP_K_HYBRID = int(os.environ.get("RAG_TOP_K_HYBRID", "10"))
RRF_K = int(os.environ.get("RAG_RRF_K", "60"))

# Whole-document context: a thread-attached file small enough to fit is injected
# in full (every chunk, in order) instead of top-K retrieval, so the model reads
# the entire file. Above this token budget, fall back to retrieval.
THREAD_WHOLE_DOC = os.environ.get("RAG_THREAD_WHOLE_DOC", "1") == "1"
WHOLE_DOC_MAX_TOKENS = int(os.environ.get("RAG_WHOLE_DOC_MAX_TOKENS", "6000"))

UPLOAD_EXTS = {".pdf", ".txt", ".md", ".markdown", ".docx", ".html", ".htm"}

# Figure captioning via the loaded vision model: detected figures (charts, plots,
# tables, diagrams) are described so their content becomes searchable. On by default
# but a strict no-op without a vision model loaded; MAX_IMAGES bounds per-doc cost
# (worst case MAX_IMAGES model calls). The chat's "Describe figures & charts" toggle
# overrides this per upload.
CAPTION_IMAGES = os.environ.get("RAG_CAPTION_IMAGES", "1") == "1"
CAPTION_MAX_IMAGES = int(os.environ.get("RAG_CAPTION_MAX_IMAGES", "8"))
CAPTION_TIMEOUT_S = float(os.environ.get("RAG_CAPTION_TIMEOUT_S", "30"))
CAPTION_MAX_TOKENS = int(os.environ.get("RAG_CAPTION_MAX_TOKENS", "320"))

# Scanned-PDF OCR: a page with little or no extractable text (an image-only/scanned
# page) is rendered and transcribed by the loaded vision model so it becomes
# searchable and readable like any other page. Needs a vision model loaded; skipped
# (page stays empty) otherwise. MIN_CHARS is the text length below which a page is
# treated as scanned; MAX_PAGES bounds per-doc cost.
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
