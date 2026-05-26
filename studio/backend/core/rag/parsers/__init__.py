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
