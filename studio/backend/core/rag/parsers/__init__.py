# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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


def inline_image_captions(
    pages: list[ParsedPage],
    images: list[ParsedImage],
    captions: list[str],
) -> list[ParsedPage]:
    """Splice per-image captions into the markdown of the pages they came from.

    Mirrors PR #5351's chat-composer pattern: figure captions become
    inline text in the page markdown so the chunker indexes them like
    any other content. Captions appear at the end of the page's text
    block as ``**Figure**: …`` lines.

    ``captions`` is parallel to ``images`` (same length, same order).
    Empty or whitespace-only captions are skipped. Images without a
    page_number are bucketed onto the single-page documents (DOCX/HTML/
    TXT all collapse to one page).
    """
    if not images or not captions:
        return list(pages)
    if len(captions) != len(images):
        # Defensive: caller mismatch shouldn't happen but we don't want
        # to lose pages over it.
        return list(pages)

    # Bucket captions per page_number (None bucket → single-page docs).
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
        # If this is the single-page case (no page_number) also flush
        # the null-bucket so DOCX/HTML/TXT pick up captions correctly.
        if page.page_number is None and null_bucket:
            captions_for_this = captions_for_this + null_bucket
        if not captions_for_this:
            out.append(page)
            continue
        appendix = "\n\n".join(f"**Figure**: {cap}" for cap in captions_for_this)
        out.append(
            ParsedPage(
                text = f"{page.text}\n\n{appendix}",
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
