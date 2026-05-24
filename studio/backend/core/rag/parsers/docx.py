# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""DOCX parsing via mammoth.

Mammoth converts Word documents to Markdown while preserving Heading
styles (`# `, `## `, ...), bullet/numbered lists, tables, and basic
emphasis. This replaces the previous python-docx paragraph-concat
approach that lost all heading metadata.

Images are captured via the `convert_image` handler when
`want_images=True`, falling back to python-docx for inline image bytes
if mammoth's docx adapter can't reach them.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from . import ParsedImage, ParsedPage, ParseResult

logger = logging.getLogger(__name__)


# Mammoth uses some default DOCX-style-name → Markdown mappings, but a
# few common variants ship with non-default names. Map them explicitly
# so we don't lose headings.
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
    """Convert mammoth's HTML output to Markdown via markdownify."""
    from markdownify import markdownify

    md = markdownify(html, heading_style = "ATX", strip = ["script", "style"])
    # markdownify can emit excessive blank lines on tables; tighten up.
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def extract(path: Path, *, want_images: bool = False) -> ParseResult:
    import mammoth

    images: list[ParsedImage] = []

    if want_images:
        # mammoth's image converter is called for every inline image.
        # We capture the bytes here and substitute a stable placeholder
        # in the rendered Markdown so the chunker doesn't trip over
        # base64 blobs. Caption-pairing is approximate — we use the
        # full document text as the caption pool (better than nothing
        # for DOCX where heading→figure adjacency isn't reliable).
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
        # Drop image elements entirely — cheaper and avoids embedding
        # base64 in Markdown when the caller doesn't want images.
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
        # Best-effort: every image inherits the whole doc text as a
        # caption pool. Phase 3B-multimodal will improve this once
        # multimodal embedders consume captions directly.
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
