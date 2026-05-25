# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen = True)
class ParsedPage:
    """One page (or page-equivalent) of Markdown-rendered text from a source document.

    For PDFs `page_number` is the 1-indexed physical page. For DOCX / HTML /
    TXT / MD the whole document is one ParsedPage with `page_number = None`.

    Text is expected to be Markdown — heading markers (`#`, `##`, …),
    pipe-tables, and list bullets survive extraction so the chunker can
    split on them. Parsers MUST emit Markdown, not bare plain text.
    """

    text: str
    page_number: int | None = None


@dataclass(frozen = True)
class ParsedImage:
    """One image extracted from a source document.

    Captured only when `parse(..., want_images=True)` is set — the
    multimodal ingestion path in Phase 3B-multimodal consumes these.
    `nearest_caption` is best-effort paragraph-adjacency; can be empty
    when no caption could be paired (the image still ingests, just
    without the paired-caption chunk).
    """

    image_bytes: bytes
    mime_type: str
    page_number: int | None = None
    nearest_caption: str = ""


@dataclass(frozen = True)
class ParseResult:
    """Result of parsing a single source document.

    `pages` is always populated; `images` is empty unless the caller
    passed `want_images=True`. Iteration aliases for `pages` so legacy
    code that did `for page in parse(path)` keeps working.
    """

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
