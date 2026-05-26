# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""DOCX → Markdown via mammoth (preserves headings, lists, tables)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from . import ParsedImage, ParsedPage, ParseResult

logger = logging.getLogger(__name__)


# Force common DOCX heading style names to Markdown headings.
_STYLE_MAP = """
p[style-name='Title'] => h1.title:fresh
p[style-name='Subtitle'] => h2.subtitle:fresh
p[style-name='Heading 1'] => h1:fresh
p[style-name='Heading 2'] => h2:fresh
p[style-name='Heading 3'] => h3:fresh
p[style-name='Heading 4'] => h4:fresh
p[style-name='Heading 5'] => h5:fresh
p[style-name='Heading 6'] => h6:fresh
"""


def _html_to_markdown(html: str) -> str:
    from markdownify import markdownify

    md = markdownify(html, heading_style = "ATX", strip = ["script", "style"])
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def extract(path: Path, *, want_images: bool = False) -> ParseResult:
    import mammoth

    images: list[ParsedImage] = []

    if want_images:
        # Capture bytes; suppress src so base64 doesn't land in Markdown.
        def _convert(image):
            with image.open() as image_bytes:
                blob = image_bytes.read()
            mime = (image.content_type or "application/octet-stream").lower()
            images.append(
                ParsedImage(
                    image_bytes = blob,
                    mime_type = mime,
                    page_number = None,
                    nearest_caption = "",
                )
            )
            return {"src": ""}

        convert_image = mammoth.images.img_element(_convert)
    else:
        convert_image = mammoth.images.img_element(lambda _image: {"src": ""})

    with open(path, "rb") as fp:
        result = mammoth.convert_to_html(
            fp,
            convert_image = convert_image,
            style_map = _STYLE_MAP,
        )
    for message in result.messages:
        logger.debug("mammoth %s: %s", getattr(message, "type", "msg"), message.message)

    markdown = _html_to_markdown(result.value)
    if want_images and images:
        # Approximate caption: first 1500 chars of doc.
        caption_pool = markdown[:1500]
        images = [
            ParsedImage(
                image_bytes = img.image_bytes,
                mime_type = img.mime_type,
                page_number = img.page_number,
                nearest_caption = caption_pool,
            )
            for img in images
        ]
    if not markdown:
        return ParseResult(pages = [], images = images)
    return ParseResult(
        pages = [ParsedPage(text = markdown, page_number = None)],
        images = images,
    )
