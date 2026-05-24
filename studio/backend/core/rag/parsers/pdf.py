# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Layout-aware PDF parsing via pymupdf + pymupdf4llm.

Produces Markdown per page (headings, pipe-tables, lists survive) so the
recursive chunker can split on heading boundaries. Falls back to pypdf
text-extraction only when pymupdf fails to open the file — keeps the
pipeline alive for malformed PDFs.

Image extraction is gated behind `want_images=True` so text-only KBs
pay zero cost for images they don't index.
"""

from __future__ import annotations

import logging
from pathlib import Path

from . import ParsedImage, ParsedPage, ParseResult

logger = logging.getLogger(__name__)


def _extract_with_pymupdf(path: Path, want_images: bool) -> ParseResult:
    import pymupdf
    import pymupdf4llm

    doc = pymupdf.open(str(path))
    try:
        pages: list[ParsedPage] = []
        for page_index in range(len(doc)):
            try:
                md = pymupdf4llm.to_markdown(
                    doc,
                    pages = [page_index],
                    write_images = False,
                    ignore_images = True,
                    show_progress = False,
                )
            except Exception:
                # pymupdf4llm can choke on individual pages (rare). Fall
                # back to plain text extraction for just that page.
                md = doc[page_index].get_text("text") or ""
            md = md.strip()
            if md:
                pages.append(ParsedPage(text = md, page_number = page_index + 1))

        images: list[ParsedImage] = []
        if want_images:
            images = _extract_images_pymupdf(doc, pages)
        return ParseResult(pages = pages, images = images)
    finally:
        doc.close()


def _extract_images_pymupdf(doc, pages: list[ParsedPage]) -> list[ParsedImage]:
    """Pull embedded images and pair each with the nearest text on the same page."""
    captions_by_page: dict[int, str] = {p.page_number: p.text for p in pages if p.page_number}
    out: list[ParsedImage] = []
    for page_index in range(len(doc)):
        page_number = page_index + 1
        try:
            image_list = doc[page_index].get_images(full = True)
        except Exception:
            continue
        for img_info in image_list:
            xref = img_info[0]
            try:
                extracted = doc.extract_image(xref)
            except Exception:
                continue
            image_bytes = extracted.get("image")
            ext = (extracted.get("ext") or "png").lower()
            mime = {
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "gif": "image/gif",
                "webp": "image/webp",
                "bmp": "image/bmp",
                "tiff": "image/tiff",
            }.get(ext, f"image/{ext}")
            if not image_bytes:
                continue
            caption = (captions_by_page.get(page_number, "") or "")[:1500]
            out.append(
                ParsedImage(
                    image_bytes = image_bytes,
                    mime_type = mime,
                    page_number = page_number,
                    nearest_caption = caption,
                )
            )
    return out


def _extract_with_pypdf_fallback(path: Path) -> ParseResult:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[ParsedPage] = []
    for index, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if text:
            pages.append(ParsedPage(text = text, page_number = index + 1))
    return ParseResult(pages = pages, images = [])


def extract(path: Path, *, want_images: bool = False) -> ParseResult:
    try:
        return _extract_with_pymupdf(path, want_images)
    except Exception as exc:
        logger.warning(
            "pymupdf failed for %s (%s: %s); falling back to pypdf",
            path,
            type(exc).__name__,
            exc,
        )
        return _extract_with_pypdf_fallback(path)
