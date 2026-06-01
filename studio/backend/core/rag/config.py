# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RAG engine configuration. Every value is env-overridable with a sensible
default. Paths come from the shared Studio storage roots, not from here."""

from __future__ import annotations

import os

EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_TOKENS = int(os.environ.get("RAG_CHUNK_TOKENS", "512"))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "64"))
TOP_K_LEXICAL = int(os.environ.get("RAG_TOP_K_LEXICAL", "30"))
TOP_K_DENSE = int(os.environ.get("RAG_TOP_K_DENSE", "30"))
TOP_K_HYBRID = int(os.environ.get("RAG_TOP_K_HYBRID", "10"))
RRF_K = int(os.environ.get("RAG_RRF_K", "60"))
MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.0"))

# File extensions accepted for upload / ingestion.
UPLOAD_EXTS = {".pdf", ".txt", ".md", ".markdown", ".docx", ".html", ".htm"}

# Multimodal image captioning. When enabled AND the loaded chat model is
# vision-capable, embedded figures are captioned and the caption text is
# spliced into the page so the normal text path indexes it (no separate image
# vector space). Off by default to keep indexing fast; each caption is a vision
# model call. CAPTION_MAX_IMAGES bounds per-document cost.
CAPTION_IMAGES = os.environ.get("RAG_CAPTION_IMAGES", "0") == "1"
CAPTION_MAX_IMAGES = int(os.environ.get("RAG_CAPTION_MAX_IMAGES", "8"))
CAPTION_TIMEOUT_S = float(os.environ.get("RAG_CAPTION_TIMEOUT_S", "30"))
