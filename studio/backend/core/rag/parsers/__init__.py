# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Same shape as the chunker's figure-boundary regex but also captures the label
# (Figure / Fig. / Table / Tab.) AND number, so a VLM caption maps to a specific
# figure on a multi-figure page.
_FIGURE_LINE_RE = re.compile(
    r"^(?P<lead>\**)(?P<label>Figure|Fig\.|Table|Tab\.)\s+"
    r"(?P<num>[A-Z]?\.?\d+(?:\.\d+)?)(?P<trail>\**[\.:])",
    re.MULTILINE | re.IGNORECASE,
)


@dataclass(frozen = True)
class ParsedPage:
    """Markdown text from one page (PDF) or whole doc (others)."""

    text: str
    page_number: int | None = None


@dataclass(frozen = True)
class ParsedImage:
    """Image extracted with want_images=True; nearest_caption may be empty."""

    image_bytes: bytes
    mime_type: str
    page_number: int | None = None
    nearest_caption: str = ""


@dataclass(frozen = True)
class ParseResult:
    pages: list[ParsedPage] = field(default_factory = list)
    images: list[ParsedImage] = field(default_factory = list)

    def __iter__(self):
        return iter(self.pages)

    def __len__(self):
        return len(self.pages)

    def __bool__(self):
        return bool(self.pages) or bool(self.images)


class UnsupportedFormatError(ValueError):
    pass


def _normalize_label(raw: str) -> str:
    head = raw.lower()
    if head.startswith("fig"):
        return "Figure"
    if head.startswith("tab"):
        return "Table"
    return raw.capitalize()


def _splice_inline_at_figure_lines(text: str, captions: list[str]) -> str:
    """Splice each caption right after the matching 'Figure N:' line.

    Captions are consumed in order against the figure caption lines
    appearing in the page text. Each spliced block is prefixed with
    "**Figure N description**:" so a retrieved chunk lets the LLM
    distinguish between multiple figures on the same page.

    Fallback when no figure caption lines exist (or fewer than we have
    captions): leftover captions are appended at the end of the page
    text as generic "**Figure**: ..." entries.
    """
    if not captions:
        return text
    matches = list(_FIGURE_LINE_RE.finditer(text))
    if not matches:
        appendix = "\n\n".join(f"**Figure**: {c}" for c in captions)
        return f"{text}\n\n{appendix}"

    parts: list[str] = []
    cursor = 0
    caps = iter(captions)
    used = 0
    for m in matches:
        # Insertion point = end of the line containing the figure label.
        line_end = text.find("\n", m.end())
        if line_end == -1:
            line_end = len(text)
        parts.append(text[cursor:line_end])
        try:
            cap = next(caps)
        except StopIteration:
            cursor = line_end
            continue
        used += 1
        label = f"{_normalize_label(m.group('label'))} {m.group('num')}"
        parts.append(f"\n\n**{label} description**: {cap}")
        cursor = line_end
    parts.append(text[cursor:])

    leftover = list(caps)
    body = "".join(parts)
    if leftover:
        appendix = "\n\n".join(f"**Figure**: {c}" for c in leftover)
        body = f"{body}\n\n{appendix}"
    return body


def inline_image_captions(
    pages: list[ParsedPage],
    images: list[ParsedImage],
    captions: list[str],
) -> list[ParsedPage]:
    """Splice per-image captions into the markdown of the pages they came from.

    Each caption lands right after the page's matching ``Figure N:`` or
    ``Table N:`` line as ``**Figure N description**: …`` so the chunker
    keeps the VLM description adjacent to the figure's existing in-PDF
    caption text. When a page has more figure caption lines than we have
    captioned images, the extra figure lines are left alone; when there
    are more captions than figure lines (or no figure lines at all),
    leftovers fall back to an end-of-page ``**Figure**: …`` appendix.

    ``captions`` is parallel to ``images`` (same length, same order).
    Empty or whitespace-only captions are skipped. Images without a
    page_number are bucketed onto the single-page documents (DOCX/HTML/
    TXT all collapse to one page).
    """
    if not images or not captions:
        return list(pages)
    if len(captions) != len(images):
        return list(pages)

    per_page: dict[int | None, list[str]] = {}
    for img, cap in zip(images, captions):
        cleaned = (cap or "").strip()
        if not cleaned:
            continue
        per_page.setdefault(img.page_number, []).append(cleaned)

    if not per_page:
        return list(pages)

    out: list[ParsedPage] = []
    null_bucket = per_page.get(None, [])
    for page in pages:
        captions_for_this = per_page.get(page.page_number, [])
        if page.page_number is None and null_bucket:
            captions_for_this = captions_for_this + null_bucket
        if not captions_for_this:
            out.append(page)
            continue
        new_text = _splice_inline_at_figure_lines(page.text, captions_for_this)
        out.append(
            ParsedPage(
                text = new_text,
                page_number = page.page_number,
            )
        )
    return out


def parse(path: Path, *, want_images: bool = False) -> ParseResult:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from .pdf import extract
    elif suffix in (".txt", ".md", ".markdown"):
        from .text import extract
    elif suffix == ".docx":
        from .docx import extract
    elif suffix in (".html", ".htm"):
        from .html import extract
    else:
        raise UnsupportedFormatError(f"Unsupported file type: {suffix}")
    return extract(path, want_images = want_images)
