# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen = True)
class ParsedPage:
    text: str
    page_number: int | None = None


class UnsupportedFormatError(ValueError):
    pass


def parse(path: Path) -> list[ParsedPage]:
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
    return extract(path)
