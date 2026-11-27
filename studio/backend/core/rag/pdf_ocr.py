# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local scanned-PDF OCR shared by document ingestion and Data Recipes."""

from __future__ import annotations

import logging
import os

from . import config
from .parsers import mupdf_serialized

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


def _ocr_textpage(page):
    return page.get_textpage_ocr(
        language = os.environ.get("RAG_OCR_LANGUAGE", "eng"),
        dpi = config.OCR_DPI,
        full = True,
        tessdata = os.environ.get("TESSDATA_PREFIX") or None,
    )


@mupdf_serialized
def ocr_pages(
    path: str,
    page_numbers,
    *,
    until = None,
) -> dict[int, str]:
    """OCR the requested pages; ``until(number, out)`` returning True stops before that page."""
    if not page_numbers:
        return {}
    import pymupdf

    out: dict[int, str] = {}
    with pymupdf.open(path) as doc:
        for number in page_numbers:
            if until is not None and until(number, out):
                break
            try:
                page = doc[number - 1]
                textpage = _ocr_textpage(page)
                text = page.get_text("text", textpage = textpage).strip()
            except Exception:
                # A missing engine/language pack affects every page, so stop.
                logger.warning("Local PDF OCR failed on page %s", number, exc_info = True)
                break
            if text:
                out[number] = text
    return out


@mupdf_serialized
def ocr_image_blocks(path: str) -> tuple[str, list[dict]]:
    """A one-page image's text as ``ocr_pages`` reads it, and its text blocks with ``box``
    ``[x0, y0, x1, y1]`` as fractions of the page."""
    import pymupdf

    try:
        with pymupdf.open(path) as doc:
            page = doc[0]
            textpage = _ocr_textpage(page)
            text = page.get_text("text", textpage = textpage).strip()
            found = page.get_text("dict", textpage = textpage).get("blocks", ())
            width, height = page.rect.width, page.rect.height
    except Exception:
        logger.warning("Local image OCR failed", exc_info = True)
        return "", []
    blocks = []
    for block in found if width > 0 and height > 0 else ():
        lines = [
            " ".join(" ".join(span.get("text", "") for span in line.get("spans", ())).split())
            for line in block.get("lines", ())
        ]
        lines = [line for line in lines if line]
        if not lines:
            continue
        x0, y0, x1, y1 = block["bbox"]
        box = [x0 / width, y0 / height, x1 / width, y1 / height]
        blocks.append(
            {
                "text": " ".join(lines),
                "lines": len(lines),
                "box": [round(min(1.0, max(0.0, v)), 5) for v in box],
            }
        )
    return text, blocks


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
