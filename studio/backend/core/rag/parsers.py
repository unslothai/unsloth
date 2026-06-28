# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Document parsing -> list[Page], one dispatch with lazy optional deps.

PDFs keep per-page boundaries (``page_number``); txt/md/docx/html return a single
page. ``parse(path, want_images=True)`` also returns embedded images. Heavy imports
are lazy, so importing this module never fails on a missing dep.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from html.parser import HTMLParser

logger = logging.getLogger(__name__)


@dataclass(frozen = True)
class Page:
    """A unit of extracted text. ``page_number`` is 1-based (None if N/A)."""

    text: str
    page_number: int | None = None
    char_count: int = 0


@dataclass(frozen = True)
class ParsedImage:
    """A raster image embedded in a document (PDF only)."""

    image_bytes: bytes
    page_number: int | None
    xref: int


def _page(text: str, page_number: int | None) -> Page:
    return Page(text = text, page_number = page_number, char_count = len(text))


class _Stripper(HTMLParser):
    """Collect visible text, skipping <script>/<style>."""

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


def _figure_boxes(
    page,
    *,
    min_area_frac: float = 0.04,
    min_side: float = 40.0,
) -> list:
    """Qualifying figure-region rectangles on a page: cluster vector drawings + raster
    placements, merge overlaps, keep the page-spanning ones (area/side filtered)."""
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
        return []
    page_area = page.rect.width * page.rect.height
    keep: list = []
    for box in _merge_rects(boxes):
        if (
            box.get_area() >= min_area_frac * page_area
            and box.width >= min_side
            and box.height >= min_side
        ):
            keep.append(box)
    return keep


def render_pdf_figures(
    path: str,
    *,
    dpi: int = 200,
    min_area_frac: float = 0.04,
    min_side: float = 40.0,
    max_figures: int = 8,
    margin_frac: float = 0.06,
) -> list[ParsedImage]:
    """Detect figure regions and render each to a PNG for captioning.

    Academic figures are vector, so raster extraction yields fragments; instead
    cluster vector drawings + raster placements into boxes, keep the page-spanning
    ones, and render them. ``dpi`` is high enough to keep small box/axis labels
    legible; each region is expanded by ``margin_frac`` (clamped to the page) so
    labels just outside the detected drawing box are not clipped. Any failure
    yields [], never an exception.
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
            for box in _figure_boxes(page, min_area_frac = min_area_frac, min_side = min_side):
                try:
                    mx, my = box.width * margin_frac, box.height * margin_frac
                    clip = (
                        pymupdf.Rect(box.x0 - mx, box.y0 - my, box.x1 + mx, box.y1 + my) & page.rect
                    )
                    pix = page.get_pixmap(dpi = dpi, clip = clip)
                    out.append(
                        ParsedImage(image_bytes = pix.tobytes("png"), page_number = i + 1, xref = 0)
                    )
                except Exception:
                    continue
                if len(out) >= max_figures:
                    return out
        return out
    finally:
        doc.close()


def pages_with_figures(
    path: str,
    *,
    max_pages: int = 4,
    min_area_frac: float = 0.04,
    min_side: float = 40.0,
) -> list[int]:
    """1-based page numbers that contain a qualifying figure region, capped at
    ``max_pages``. Drives figure tiling. Any failure yields []."""
    try:
        import pymupdf
    except Exception:
        return []
    try:
        doc = pymupdf.open(path)
    except Exception:
        return []
    pages: list[int] = []
    try:
        for i, page in enumerate(doc):
            if _figure_boxes(page, min_area_frac = min_area_frac, min_side = min_side):
                pages.append(i + 1)
                if len(pages) >= max_pages:
                    break
        return pages
    finally:
        doc.close()


def render_pdf_figure_tiles(
    path: str,
    page_numbers,
    *,
    dpi: int = 200,
    rows: int = 2,
    cols: int = 2,
    overlap: float = 0.12,
    fullpage: bool = True,
    max_tiles: int = 24,
) -> list[ParsedImage]:
    """Render figure-bearing pages as overlapping high-DPI tiles (plus an optional
    full-page image for global context), each a ``ParsedImage`` keyed by page number.
    Tiling keeps small diagram/axis/box labels legible and covers every sub-figure
    without relying on exact region detection, so figure recall generalizes across
    content density and model strength. Any failure yields [] (or skips a page)."""
    wanted = [int(n) for n in page_numbers]
    if not wanted:
        return []
    try:
        import pymupdf
    except Exception:
        return []
    try:
        doc = pymupdf.open(path)
    except Exception:
        return []
    out: list[ParsedImage] = []
    try:
        for num in wanted:
            if num < 1 or num > doc.page_count:
                continue
            page = doc[num - 1]
            rect = page.rect
            clips: list = [rect] if fullpage else []
            cw, ch = rect.width / cols, rect.height / rows
            ox, oy = cw * overlap, ch * overlap
            for r in range(rows):
                for c in range(cols):
                    clips.append(
                        pymupdf.Rect(
                            rect.x0 + c * cw - ox,
                            rect.y0 + r * ch - oy,
                            rect.x0 + (c + 1) * cw + ox,
                            rect.y0 + (r + 1) * ch + oy,
                        )
                        & rect
                    )
            for clip in clips:
                try:
                    pix = page.get_pixmap(dpi = dpi, clip = clip)
                    out.append(ParsedImage(image_bytes = pix.tobytes("png"), page_number = num, xref = 0))
                except Exception:
                    continue
                if len(out) >= max_tiles:
                    return out
        return out
    finally:
        doc.close()


def render_pdf_pages(
    path: str,
    page_numbers,
    *,
    dpi: int = 150,
) -> dict[int, bytes]:
    """Render whole PDF pages (given as 1-based numbers) to PNG bytes, keyed by
    page number. Backs scanned-page OCR. Any failure yields ``{}`` (or skips that
    page), never an exception.
    """
    wanted = {int(n) for n in page_numbers}
    if not wanted:
        return {}
    try:
        import pymupdf
    except Exception:
        return {}
    try:
        doc = pymupdf.open(path)
    except Exception:
        return {}
    out: dict[int, bytes] = {}
    try:
        for i, page in enumerate(doc):
            num = i + 1
            if num not in wanted:
                continue
            try:
                pix = page.get_pixmap(dpi = dpi)
                out[num] = pix.tobytes("png")
            except Exception:
                continue
        return out
    finally:
        doc.close()


def _docx(path: str) -> list[Page]:
    import docx

    document = docx.Document(path)
    text = "\n".join(p.text for p in document.paragraphs)
    return [_page(text, None)]


def parse(path: str, *, want_images: bool = False):
    """Parse a file into pages by extension. Returns ``list[Page]``, or
    ``(list[Page], list[ParsedImage])`` when ``want_images=True`` (only PDFs yield
    images). Raises ValueError on unsupported ext."""
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
