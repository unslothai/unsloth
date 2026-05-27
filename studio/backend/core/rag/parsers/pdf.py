# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""PDF → Markdown via pymupdf4llm; pypdf fallback for malformed files."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from . import ParsedImage, ParsedPage, ParseResult

logger = logging.getLogger(__name__)

# pymupdf4llm wraps OCR'd vector-graphics text with these markers even when
# `ignore_images=True`. Strip the whole block — the VLM captioner produces
# a proper description for the figure, and the marker text just pollutes
# the chunked body / shows up verbatim in citations.
_PICTURE_TEXT_BLOCK_RE = re.compile(
    r"-{3,}\s*Start of picture text\s*-{3,}.*?-{3,}\s*End of picture text\s*-{3,}",
    re.DOTALL | re.IGNORECASE,
)


def _strip_picture_text_markers(md: str) -> str:
    return _PICTURE_TEXT_BLOCK_RE.sub("", md)


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
                # pymupdf4llm can choke on a single page; fall back to plain text.
                md = doc[page_index].get_text("text") or ""
            md = _strip_picture_text_markers(md).strip()
            if md:
                pages.append(ParsedPage(text = md, page_number = page_index + 1))

        images: list[ParsedImage] = []
        if want_images:
            images = _extract_images_pymupdf(doc, pages)
        return ParseResult(pages = pages, images = images)
    finally:
        doc.close()


# Pages smaller than this (in PDF points) are ignored as figure regions —
# bigger than a typical icon/glyph, smaller than a banner.
_MIN_FIGURE_PT = 60
# 2× scale renders at 144 dpi (PDF default is 72 dpi). Enough resolution for
# the captioner to read axis labels, arrow text, and inset photos.
_RENDER_SCALE = 2.0
# Expand the union bbox a few points so caption baselines / borders survive.
_FIGURE_MARGIN_PT = 8.0


def _extract_images_pymupdf(doc, pages: list[ParsedPage]) -> list[ParsedImage]:
    """Render each page's figure region (vector drawings + raster sub-images)
    as a single PNG. Vector schematics like Figure 1 (no embedded raster)
    are visible to the captioner only via rendering — ``page.get_images``
    misses them entirely. We union all non-text geometry on a page into
    one bbox; for academic papers this typically maps 1:1 to "the figure
    on this page".
    """
    import pymupdf

    captions_by_page: dict[int, str] = {
        p.page_number: p.text for p in pages if p.page_number
    }
    out: list[ParsedImage] = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1
        try:
            rects: list[pymupdf.Rect] = []
            for drawing in page.get_drawings() or []:
                rect = drawing.get("rect")
                if rect is not None:
                    rects.append(pymupdf.Rect(rect))
            for info in page.get_image_info(xrefs = True) or []:
                bbox = info.get("bbox")
                if bbox is not None:
                    rects.append(pymupdf.Rect(bbox))
        except Exception:
            continue
        if not rects:
            continue
        union = rects[0]
        for r in rects[1:]:
            union |= r
        if union.width < _MIN_FIGURE_PT or union.height < _MIN_FIGURE_PT:
            continue
        # Expand and clip to page rect so we don't render past page edges.
        union = pymupdf.Rect(
            union.x0 - _FIGURE_MARGIN_PT,
            union.y0 - _FIGURE_MARGIN_PT,
            union.x1 + _FIGURE_MARGIN_PT,
            union.y1 + _FIGURE_MARGIN_PT,
        ) & page.rect
        try:
            matrix = pymupdf.Matrix(_RENDER_SCALE, _RENDER_SCALE)
            pix = page.get_pixmap(clip = union, matrix = matrix, alpha = False)
            png_bytes = pix.tobytes("png")
        except Exception:
            continue
        if not png_bytes:
            continue
        caption = (captions_by_page.get(page_number, "") or "")[:1500]
        out.append(
            ParsedImage(
                image_bytes = png_bytes,
                mime_type = "image/png",
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
