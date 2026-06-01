# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Document parsing -> list[Page], one dispatch with lazy optional deps.

PDFs keep per-page boundaries (``page_number``); other formats (txt, md, docx,
html) return a single page. ``parse(path, want_images=True)`` also returns
embedded images. Heavy imports (PyMuPDF, python-docx) are lazy, so importing
this module never fails on a missing dep.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from html.parser import HTMLParser

logger = logging.getLogger(__name__)


@dataclass(frozen = True)
class Page:
    """One unit of extracted text. ``page_number`` is 1-based (None if N/A)."""

    text: str
    page_number: int | None = None
    char_count: int = 0


@dataclass(frozen = True)
class ParsedImage:
    """A raster image embedded in a source document (PDF only, for now)."""

    image_bytes: bytes
    page_number: int | None
    xref: int


def _page(text: str, page_number: int | None) -> Page:
    return Page(text = text, page_number = page_number, char_count = len(text))


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------
class _Stripper(HTMLParser):
    """Collect visible text, skipping <script>/<style> contents."""

    def __init__(self) -> None:
        super().__init__()
        self._skip = 0
        self.out: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.out.append(data.strip())


def _html(raw: str) -> list[Page]:
    parser = _Stripper()
    parser.feed(raw)
    return [_page("\n".join(parser.out), 1)]


# --------------------------------------------------------------------------
# PDF (PyMuPDF / fitz)
# --------------------------------------------------------------------------
def _pdf(path: str, want_images: bool) -> tuple[list[Page], list[ParsedImage]]:
    import fitz  # PyMuPDF

    pages: list[Page] = []
    images: list[ParsedImage] = []
    doc = fitz.open(path)
    try:
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            pages.append(_page(text, i + 1))
            if want_images:
                for img in page.get_images(full = True):
                    xref = img[0]
                    try:
                        extracted = doc.extract_image(xref)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("skipping image xref %s: %s", xref, exc)
                        continue
                    image_bytes = extracted.get("image")
                    if image_bytes:
                        images.append(
                            ParsedImage(
                                image_bytes = image_bytes,
                                page_number = i + 1,
                                xref = xref,
                            )
                        )
    finally:
        doc.close()
    return pages, images


# --------------------------------------------------------------------------
# PDF figure rendering (for captioning)
# --------------------------------------------------------------------------
def _merge_rects(boxes: list) -> list:
    """Union overlapping rectangles (largest-first) into figure regions."""
    import pymupdf

    rects = [pymupdf.Rect(b) for b in boxes]
    rects = [r for r in rects if r.width > 5 and r.height > 5]
    merged: list = []
    for box in sorted(rects, key = lambda r: -r.get_area()):
        placed = False
        for m in merged:
            if m.intersects(box):
                m |= box
                placed = True
                break
        if not placed:
            merged.append(+box)
    return merged


def render_pdf_figures(
    path: str,
    *,
    dpi: int = 130,
    min_area_frac: float = 0.04,
    min_side: float = 40.0,
    max_figures: int = 8,
) -> list[ParsedImage]:
    """Detect figure regions and render each to a PNG for captioning.

    Academic figures are vector, so raw raster extraction yields fragments.
    Instead we cluster vector drawings + raster placements into boxes, keep the
    ones covering a meaningful slice of the page, and render them. Any failure
    yields an empty list, never an exception.
    """
    try:
        import pymupdf
    except Exception:
        return []

    out: list[ParsedImage] = []
    try:
        doc = pymupdf.open(path)
    except Exception:
        return []
    try:
        for i, page in enumerate(doc):
            boxes: list = []
            try:
                boxes.extend(info["bbox"] for info in page.get_image_info())
            except Exception:
                pass
            try:
                boxes.extend(page.cluster_drawings())
            except Exception:
                pass
            if not boxes:
                continue
            page_area = page.rect.width * page.rect.height
            for box in _merge_rects(boxes):
                if (
                    box.get_area() >= min_area_frac * page_area
                    and box.width >= min_side
                    and box.height >= min_side
                ):
                    try:
                        pix = page.get_pixmap(dpi = dpi, clip = box)
                        out.append(
                            ParsedImage(
                                image_bytes = pix.tobytes("png"),
                                page_number = i + 1,
                                xref = 0,
                            )
                        )
                    except Exception:
                        continue
                    if len(out) >= max_figures:
                        return out
        return out
    finally:
        doc.close()


# --------------------------------------------------------------------------
# DOCX (python-docx)
# --------------------------------------------------------------------------
def _docx(path: str) -> list[Page]:
    import docx

    document = docx.Document(path)
    text = "\n".join(p.text for p in document.paragraphs)
    return [_page(text, None)]


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------
def parse(path: str, *, want_images: bool = False):
    """Parse a file into pages by extension.

    Returns ``list[Page]`` by default. When ``want_images=True`` returns
    ``(list[Page], list[ParsedImage])``; only PDFs yield images, every other
    format returns an empty image list. Raises ValueError on unsupported ext.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        pages, images = _pdf(path, want_images)
        return (pages, images) if want_images else pages

    if ext == ".docx":
        pages = _docx(path)
        return (pages, []) if want_images else pages

    if ext in (".html", ".htm", ".txt", ".md", ".markdown"):
        with open(path, encoding = "utf-8", errors = "replace") as f:
            raw = f.read()
        pages = _html(raw) if ext in (".html", ".htm") else [_page(raw, None)]
        return (pages, []) if want_images else pages

    raise ValueError(f"unsupported file type: {ext}")


def parse_text(text: str) -> list[Page]:
    """Wrap already-extracted text as a single Page (tests / in-memory ingest)."""
    return [_page(text, None)]
