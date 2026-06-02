# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RAG config. Every value is env-overridable; paths live in shared Studio
storage roots, not here."""

from __future__ import annotations

import os

EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# Under bge's 512 limit with headroom for the 2 special tokens ([CLS]/[SEP]) the
# embedder adds (else 512 -> 514 overflows: llama-server 500s, ST truncates).
# Keep <= embedder_max - ~12.
CHUNK_TOKENS = int(os.environ.get("RAG_CHUNK_TOKENS", "500"))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "64"))
TOP_K_LEXICAL = int(os.environ.get("RAG_TOP_K_LEXICAL", "30"))
TOP_K_DENSE = int(os.environ.get("RAG_TOP_K_DENSE", "30"))
TOP_K_HYBRID = int(os.environ.get("RAG_TOP_K_HYBRID", "10"))
RRF_K = int(os.environ.get("RAG_RRF_K", "60"))
MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.0"))

# Extensions accepted for ingestion.
UPLOAD_EXTS = {".pdf", ".txt", ".md", ".markdown", ".docx", ".html", ".htm"}

# Figure captioning via the loaded vision model. Off by default (each caption is
# a model call); MAX_IMAGES bounds per-doc cost.
CAPTION_IMAGES = os.environ.get("RAG_CAPTION_IMAGES", "0") == "1"
CAPTION_MAX_IMAGES = int(os.environ.get("RAG_CAPTION_MAX_IMAGES", "8"))
CAPTION_TIMEOUT_S = float(os.environ.get("RAG_CAPTION_TIMEOUT_S", "30"))

# Embedder backend. "auto" (default): sentence-transformers on a CUDA/ROCm GPU
# (torch fp16 is far faster for bulk indexing), else torch-free GGUF llama-server.
# "sentence-transformers"/"llama-server" force one. Switching backends changes the
# vectors, so the index must be rebuilt.
EMBED_BACKEND = os.environ.get("RAG_EMBED_BACKEND", "auto")
# llama-server backend only (ignored for sentence-transformers).
# F16 over Q8_0: faster (no per-block dequant for this tiny model) and exact vs
# fp32, for ~30MB more on disk.
EMBED_GGUF_REPO = os.environ.get(
    "RAG_EMBED_GGUF_REPO", "CompendiumLabs/bge-small-en-v1.5-gguf"
)
EMBED_GGUF_VARIANT = os.environ.get("RAG_EMBED_GGUF_VARIANT", "F16")
EMBED_DEVICE = os.environ.get("RAG_EMBED_DEVICE", "auto")  # "auto" | "gpu" | "cpu"
EMBED_HOST = os.environ.get("RAG_EMBED_HOST", "127.0.0.1")
EMBED_PORT = int(os.environ.get("RAG_EMBED_PORT", "0"))  # 0 = auto-pick a free port
EMBED_BATCH = int(os.environ.get("RAG_EMBED_BATCH", "64"))
EMBED_STARTUP_TIMEOUT_S = float(os.environ.get("RAG_EMBED_STARTUP_TIMEOUT_S", "120"))
EMBED_REQUEST_TIMEOUT_S = float(os.environ.get("RAG_EMBED_REQUEST_TIMEOUT_S", "60"))
