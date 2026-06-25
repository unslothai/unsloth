# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chat-surface helpers (not core/inference or core/data_recipe).

Exposes the document-extraction pipeline for files dropped into the chat
composer: PDF via PyMuPDF4LLM, DOCX via mammoth; PPTX unsupported.
"""

from __future__ import annotations

from .document_extractor import (
    DOCUMENT_EXTRACTION_AVAILABLE,
    DEFAULT_DOCUMENT_VISUAL_PAYLOADS,
    DocumentExtractionBusy,
    DocumentExtractionCancelled,
    DocumentExtractionEncrypted,
    DocumentExtractionTimeout,
    DocumentExtractionUnavailable,
    ExtractedFigure,
    ExtractResult,
    _EXTRACT_CONCURRENCY,
    MAX_DOCUMENT_VISUAL_PAYLOADS,
    SUPPORTED_MIME_TYPES,
    SUPPORTED_SUFFIXES,
    _EXTRACT_SEMAPHORE,
    _drain_future_exception,
    document_parser_support,
    document_parser_unavailable_reasons,
    extract_document,
)
from .vlm_capability import (
    VlmCapability,
    detect_loaded_vlm,
    extract_self_base_url,
)

__all__ = [
    "DOCUMENT_EXTRACTION_AVAILABLE",
    "DEFAULT_DOCUMENT_VISUAL_PAYLOADS",
    "DocumentExtractionBusy",
    "DocumentExtractionCancelled",
    "DocumentExtractionEncrypted",
    "DocumentExtractionTimeout",
    "DocumentExtractionUnavailable",
    "ExtractedFigure",
    "ExtractResult",
    "_EXTRACT_CONCURRENCY",
    "MAX_DOCUMENT_VISUAL_PAYLOADS",
    "SUPPORTED_MIME_TYPES",
    "SUPPORTED_SUFFIXES",
    "VlmCapability",
    "_EXTRACT_SEMAPHORE",
    "_drain_future_exception",
    "detect_loaded_vlm",
    "document_parser_support",
    "document_parser_unavailable_reasons",
    "extract_document",
    "extract_self_base_url",
]
