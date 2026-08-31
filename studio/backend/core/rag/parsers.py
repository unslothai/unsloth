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
import re
from dataclasses import dataclass
from html.parser import HTMLParser

from . import config

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


# pymupdf4llm rebuilds text from positioned glyphs, which mangles complex-shaping
# scripts (RTL Arabic/Hebrew emerge as shaped Presentation Forms, Indic matras drop to
# U+FFFD) and can silently drop most of a heavy-RTL page. When Markdown trips these
# signals we fall back to PyMuPDF's logical-order get_text(). Thresholds mirror the chat
# extractor guard (unslothai/unsloth#5351 review).
_SHAPED_PRESENTATION_FORMS = re.compile("[\ufb1d-\ufdff\ufe70-\ufefc]")
_PDF_FALLBACK_MIN_BAD_GLYPHS = 5
_PDF_FALLBACK_BAD_GLYPH_RATIO = 0.0005
_PDF_INCOMPLETE_RATIO = 0.75
_PDF_INCOMPLETE_MIN_LETTERS = 200


def _markdown_corrupted(text: str) -> bool:
    """True when pymupdf4llm's glyph reconstruction mangled the text: shaped RTL
    Presentation Forms or U+FFFD replacements above a small floor/ratio (so a lone
    legitimate shaped glyph does not force the fallback)."""
    if not text:
        return False
    threshold = max(_PDF_FALLBACK_MIN_BAD_GLYPHS, _PDF_FALLBACK_BAD_GLYPH_RATIO * len(text))
    shaped = len(_SHAPED_PRESENTATION_FORMS.findall(text))
    return shaped > threshold or text.count("\ufffd") > threshold


def _markdown_incomplete(markdown: str, plain: str) -> bool:
    """True when ``markdown`` holds far fewer letters than the raw ``get_text`` layer -- a
    coarse guard for heavy-RTL pages pymupdf4llm silently drops without shaped glyphs."""
    plain_letters = sum(1 for c in plain if c.isalnum())
    if plain_letters < _PDF_INCOMPLETE_MIN_LETTERS:
        return False
    markdown_letters = sum(1 for c in markdown if c.isalnum())
    return markdown_letters < _PDF_INCOMPLETE_RATIO * plain_letters


def _pdf_markdown(doc, pages: range | None = None) -> list[str] | None:
    """Per-page layout-aware Markdown (tables, headings, lists) via pymupdf4llm; index
    i maps to page i+1. Returns None when the lib is missing, extraction fails, or the
    page count does not line up, so the caller falls back to plain PyMuPDF text."""
    try:
        import pymupdf4llm
    except Exception:
        return None
    try:
        kwargs = {"page_chunks": True, "show_progress": False}
        if pages is not None:
            kwargs["pages"] = list(pages)
        chunks = pymupdf4llm.to_markdown(doc, **kwargs)
    except Exception:  # noqa: BLE001 - never let Markdown extraction break ingestion
        logger.warning("pymupdf4llm extraction failed; using plain text", exc_info = True)
        return None
    expected_pages = doc.page_count if pages is None else len(pages)
    if not isinstance(chunks, list) or len(chunks) != expected_pages:
        return None
    return [str(c.get("text") or "") for c in chunks]


def _pdf(
    source: str | bytes,
    want_images: bool,
    max_pages: int | None = None,
) -> tuple[list[Page], list[ParsedImage], int]:
    import fitz  # PyMuPDF

    pages: list[Page] = []
    images: list[ParsedImage] = []
    doc = (
        fitz.open(stream = source, filetype = "pdf") if isinstance(source, bytes) else fitz.open(source)
    )
    try:
        if doc.needs_pass:
            raise ValueError("encrypted PDF requires a password")
        total_pages = doc.page_count
        page_numbers = range(total_pages if max_pages is None else min(total_pages, max_pages))
        if not config.PDF_MARKDOWN:
            md = None
        elif max_pages is None:
            md = _pdf_markdown(doc)
        else:
            md = _pdf_markdown(doc, page_numbers)
        for i, page_number in enumerate(page_numbers):
            page = doc[page_number]
            plain = page.get_text("text") or ""
            candidate = md[i] if md else ""
            # Prefer layout-aware Markdown (keeps tables/headings legible for retrieval),
            # but drop to PyMuPDF's logical-order text when Markdown is off/empty or when
            # pymupdf4llm mangled it (RTL/Indic) or dropped most of the page.
            if (
                candidate
                and not _markdown_corrupted(candidate)
                and not _markdown_incomplete(candidate, plain)
            ):
                text = candidate
            else:
                text = plain
            pages.append(_page(text, page_number + 1))
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
                                page_number = page_number + 1,
                                xref = xref,
                            )
                        )
    finally:
        doc.close()
    return pages, images, total_pages


def parse_pdf_bytes(data: bytes, *, max_pages: int | None = None) -> tuple[list[Page], int]:
    """Extract PDF pages from an in-memory download using the ingestion parser.

    Returns the (capped) pages plus the document's full page count, so a caller
    that set ``max_pages`` can tell a fully-read short PDF from a truncated one."""
    pages, _images, total_pages = _pdf(data, want_images = False, max_pages = max_pages)
    return pages, total_pages


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


def pages_with_figures(
    path: str,
    *,
    max_pages: int = 4,
    min_area_frac: float = 0.04,
    min_side: float = 40.0,
    exclude_pages: set[int] | None = None,
) -> list[int]:
    """1-based page numbers with a qualifying figure region, capped at ``max_pages``;
    drives figure tiling. ``exclude_pages`` (1-based) are skipped: those are the pages
    OCR already transcribed whole, so tiling them would duplicate the vision work. Any
    failure yields []."""
    exclude = exclude_pages or set()
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
            if (i + 1) in exclude:
                continue
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
    """Render figure-bearing pages as overlapping high-DPI tiles (plus an optional full
    page), each a ``ParsedImage`` keyed by page number. Tiling keeps small labels legible
    and covers every sub-figure without exact region detection. Any failure yields []."""
    wanted = [int(n) for n in page_numbers]
    if not wanted:
        return []
    rows, cols = max(1, int(rows)), max(1, int(cols))  # never divide by zero
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


def _docx_table_rows(table) -> list[str]:
    """Each row as pipe-joined cell text (the locator splits anchors on pipes).
    Columns stay aligned to the layout grid (merged cells fill their spanned slots,
    skipped leading/trailing grid columns become empty fields). Cells are walked in
    document order so a nested table, and any text after it, flattens in place."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    rows: list[str] = []
    seen: set = set()  # <w:tc> already emitted; dedups merges spanning columns or rows
    for row in table.rows:
        cells: list[str] = [""] * getattr(row, "grid_cols_before", 0)
        trailing: list[str] = []  # nested rows + any post-nested text, kept in order
        for cell in row.cells:
            # A merged cell shares one <w:tc> across the columns and rows it spans:
            # emit its text once, then placeholders, so columns and rows stay aligned.
            if cell._tc in seen:
                cells.append("")
                continue
            seen.add(cell._tc)
            # Paragraph text before the first nested table is the aligned field; the
            # nested table and anything after it flatten below the row, in order.
            field: list[str] = []
            after_table = False
            for item in cell.iter_inner_content():
                if isinstance(item, Table):
                    after_table = True
                    trailing.extend(_docx_table_rows(item))
                elif isinstance(item, Paragraph):
                    text = " ".join(item.text.split())  # collapse in-cell newlines
                    if text:
                        (trailing if after_table else field).append(text)
            cells.append(" ".join(field))  # empty cells kept so columns line up
        cells.extend([""] * getattr(row, "grid_cols_after", 0))
        if any(c.strip() for c in cells):
            rows.append(" | ".join(cells))
        rows.extend(trailing)
    return rows


def _docx(path: str) -> list[Page]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(path)
    lines: list[str] = []
    # Walk body content in document order: paragraphs alone drop tables entirely.
    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            if block.text.strip():
                lines.append(block.text)
        elif isinstance(block, Table):
            lines.extend(_docx_table_rows(block))
    return [_page("\n".join(lines), None)]


def parse(path: str, *, want_images: bool = False):
    """Parse a file into pages by extension. Returns ``list[Page]``, or
    ``(list[Page], list[ParsedImage])`` when ``want_images=True`` (only PDFs yield
    images). Raises ValueError on unsupported ext."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        pages, images, _total = _pdf(path, want_images)
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
