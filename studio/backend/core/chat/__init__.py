# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Chat-surface helpers that do not belong in ``core/inference`` (tightly
coupled to model backends) and explicitly not in ``core/data_recipe``
(owns dataset pipelines).

Exposes the document-extraction pipeline used when a user drops a
PDF / DOCX / HTML / MD / TXT file into the chat composer. PDF parsing
uses PyMuPDF4LLM, DOCX uses mammoth. PPTX is not supported here —
convert to PDF first.
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
    MAX_DOCUMENT_VISUAL_PAYLOADS,
    SUPPORTED_MIME_TYPES,
    SUPPORTED_SUFFIXES,
    _EXTRACT_SEMAPHORE,
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
    "MAX_DOCUMENT_VISUAL_PAYLOADS",
    "SUPPORTED_MIME_TYPES",
    "SUPPORTED_SUFFIXES",
    "VlmCapability",
    "_EXTRACT_SEMAPHORE",
    "detect_loaded_vlm",
    "document_parser_support",
    "document_parser_unavailable_reasons",
    "extract_document",
    "extract_self_base_url",
]
