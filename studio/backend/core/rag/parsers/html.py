# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from pathlib import Path

from . import ParsedPage


_SKIP_TAGS = {"script", "style", "noscript", "template"}


def extract(path: Path) -> list[ParsedPage]:
    from bs4 import BeautifulSoup

    raw = path.read_bytes()
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(_SKIP_TAGS):
        tag.decompose()
    text = soup.get_text(separator = "\n").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = "\n".join(lines)
    if not cleaned:
        return []
    return [ParsedPage(text = cleaned, page_number = None)]
