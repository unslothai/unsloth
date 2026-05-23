# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from pathlib import Path

from . import ParsedPage


def extract(path: Path) -> list[ParsedPage]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            import chardet

            detected = chardet.detect(raw)
            encoding = detected.get("encoding") or "latin-1"
        except ImportError:
            encoding = "latin-1"
        text = raw.decode(encoding, errors = "replace")
    text = text.strip()
    if not text:
        return []
    return [ParsedPage(text = text, page_number = None)]
