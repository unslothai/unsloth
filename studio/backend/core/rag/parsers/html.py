# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HTML → Markdown via markdownify. Images only resolve local file refs."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import ParsedImage, ParsedPage, ParseResult

logger = logging.getLogger(__name__)

_SKIP_TAGS = ("script", "style", "noscript", "template")


def _collect_local_images(soup, html_path: Path) -> list[ParsedImage]:
    images: list[ParsedImage] = []
    base_dir = html_path.parent
    for tag in soup.find_all("img"):
        src = tag.get("src") or ""
        parsed = urlparse(src)
        if parsed.scheme and parsed.scheme not in ("file", ""):
            continue
        local_path = (base_dir / unquote(parsed.path or src)).resolve()
        try:
            local_path.relative_to(base_dir.resolve())
        except ValueError:
            # Path traversal: refuse to read outside the source's directory.
            continue
        if not local_path.is_file():
            continue
        try:
            blob = local_path.read_bytes()
        except OSError:
            continue
        suffix = local_path.suffix.lower().lstrip(".")
        mime = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
            "webp": "image/webp",
            "svg": "image/svg+xml",
        }.get(suffix, f"image/{suffix or 'octet-stream'}")
        caption = tag.get("alt") or tag.get("title") or ""
        images.append(
            ParsedImage(
                image_bytes = blob,
                mime_type = mime,
                page_number = None,
                nearest_caption = caption,
            )
        )
    return images


def extract(path: Path, *, want_images: bool = False) -> ParseResult:
    from bs4 import BeautifulSoup
    from markdownify import markdownify

    raw = path.read_bytes()
    soup = BeautifulSoup(raw, "lxml")
    for tag_name in _SKIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    images: list[ParsedImage] = []
    if want_images:
        images = _collect_local_images(soup, path)

    md = markdownify(
        str(soup),
        heading_style = "ATX",
        strip = list(_SKIP_TAGS),
    )
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    if not md:
        return ParseResult(pages = [], images = images)
    return ParseResult(
        pages = [ParsedPage(text = md, page_number = None)],
        images = images,
    )
