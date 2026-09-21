# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local scanned-PDF OCR shared by document ingestion and Data Recipes."""

from __future__ import annotations

import logging
import os

from . import config

logger = logging.getLogger(__name__)


class PDFOCRError(ValueError):
    """An incomplete PDF extraction that should be shown to the uploader."""


def unreadable_pages_error(page_numbers) -> PDFOCRError:
    numbers = sorted(page_numbers)
    label = ", ".join(map(str, numbers[:20]))
    if len(numbers) > 20:
        label += f" (and {len(numbers) - 20} more)"
    return PDFOCRError(
        f"Could not read scanned PDF pages: {label}. "
        "Enable OCR and configure Tesseract language data (TESSDATA_PREFIX), "
        "or upload a PDF with a searchable text layer. "
        f"OCR is limited to {config.OCR_MAX_PAGES} scanned pages per upload."
    )


def ocr_pages(path: str, page_numbers) -> dict[int, str]:
    """Transcribe only requested pages using PyMuPDF's integrated Tesseract.

    No downloads or model loading. Tesseract language data must already be installed.
    The caller owns the page budget and decides how to report unresolved pages.
    """
    if not page_numbers:
        return {}
    import pymupdf

    out: dict[int, str] = {}
    with pymupdf.open(path) as doc:
        for number in page_numbers:
            try:
                page = doc[number - 1]
                textpage = page.get_textpage_ocr(
                    language = os.environ.get("RAG_OCR_LANGUAGE", "eng"),
                    dpi = config.OCR_DPI,
                    full = True,
                    tessdata = os.environ.get("TESSDATA_PREFIX") or None,
                )
                text = page.get_text("text", textpage = textpage).strip()
            except Exception:
                # A missing engine/language pack affects every page. Do not repeatedly
                # try an unavailable OCR engine for an entire scanned document.
                logger.warning("Local PDF OCR failed on page %s", number, exc_info = True)
                break
            if text:
                out[number] = text
    return out


def extract_text(path: str, ocr: bool, max_pages: int) -> str:
    """Extract a complete PDF in one process, preserving selectable text."""
    from . import parsers

    pages = parsers.parse(path)
    scanned = [page.page_number for page in pages if page.needs_ocr]
    texts = ocr_pages(path, scanned[:max_pages]) if ocr else {}
    if set(scanned) - texts.keys():
        raise unreadable_pages_error(set(scanned) - texts.keys())
    parts = []
    for page in pages:
        original = page.text.strip()
        text = texts.get(page.page_number, "")
        parts.append(
            text
            if not original or original in text
            else "\n\n".join(filter(None, [original, text]))
        )
    return "\n\n".join(parts)
