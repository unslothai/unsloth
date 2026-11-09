# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Renders a chat export (the HTML body the frontend builds) to PDF with PyMuPDF; nothing is stored."""

import asyncio
import html
import io
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject
from loggers import get_logger
from utils.utils import log_and_http_error

router = APIRouter()

logger = get_logger(__name__)

MAX_EXPORT_HTML_CHARS = 64 * 1024 * 1024
MAX_EXPORT_PAGES = 5000
_PAGE_MARGIN_PT = 54

# MuPDF's HTML engine takes a CSS subset; unknown families fall back to its bundled Noto/Droid fonts.
_PDF_CSS = """
body { font-family: sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1f2328; }
h1 { font-size: 17pt; margin: 0 0 4pt 0; }
.meta { color: #6e7781; font-size: 8.5pt; margin: 0 0 14pt 0; }
.conversation-title { font-size: 13.5pt; margin: 16pt 0 8pt 0; }
.message { margin: 0 0 12pt 0; padding: 6pt 10pt; border-left: 2pt solid #d0d7de; }
.message.user { background-color: #f6f8fa; border-left: 2pt solid #8c959f; }
.message.system { border-left: 2pt solid #bf8700; }
.message.tool { border-left: 2pt solid #8250df; }
.role { font-size: 8.5pt; font-weight: bold; color: #57606a; margin: 0 0 4pt 0; }
.content h1, .content h2, .content h3, .content h4 { font-size: 11.5pt; margin: 8pt 0 4pt 0; }
p { margin: 0 0 6pt 0; }
pre { background-color: #f6f8fa; padding: 6pt; font-family: monospace; font-size: 8.5pt; white-space: pre-wrap; }
code { font-family: monospace; font-size: 9pt; }
blockquote { margin: 0 0 6pt 0; padding: 0 0 0 10pt; border-left: 2pt solid #d0d7de; color: #57606a; }
th, td { border: 1px solid #d0d7de; padding: 2pt 5pt; }
details { color: #57606a; margin: 0 0 6pt 0; }
summary { font-style: italic; }
a { color: #0969da; }
hr { margin: 14pt 0; }
"""


class ChatPdfTooLong(Exception):
    pass


class ChatPdfRequest(BaseModel):
    title: str = Field("", max_length = 2000)
    html: str = Field(..., min_length = 1, max_length = MAX_EXPORT_HTML_CHARS)


def render_chat_pdf(title: str, body_html: str) -> bytes:
    """No archive is passed to the Story, so markup can only embed data: images, never local files or URLs."""
    import pymupdf

    heading = html.escape(" ".join(title.split()) or "Conversation")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    document = (
        f"<h1>{heading}</h1><p class='meta'>Exported from Unsloth Studio on {stamp}</p>{body_html}"
    )
    story = pymupdf.Story(html = document, user_css = _PDF_CSS)
    buffer = io.BytesIO()
    writer = pymupdf.DocumentWriter(buffer)
    mediabox = pymupdf.paper_rect("a4")
    where = mediabox + (_PAGE_MARGIN_PT, _PAGE_MARGIN_PT, -_PAGE_MARGIN_PT, -_PAGE_MARGIN_PT)
    pages = 0
    more = 1
    while more:
        if pages >= MAX_EXPORT_PAGES:
            writer.close()
            raise ChatPdfTooLong(f"export exceeds {MAX_EXPORT_PAGES} pages")
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
        pages += 1
    writer.close()

    # Title in the document info, so viewers show it instead of "Untitled".
    pdf = pymupdf.open("pdf", buffer.getvalue())
    try:
        pdf.set_metadata({"title": " ".join(title.split()), "creator": "Unsloth Studio"})
        # The CJK fallback font alone is ~2 MB; keep only the glyphs used.
        pdf.subset_fonts()
        return pdf.tobytes(garbage = 3, deflate = True)
    finally:
        pdf.close()


@router.post("/pdf")
async def export_chat_pdf(
    request: ChatPdfRequest, current_subject: str = Depends(get_current_subject)
) -> Response:
    try:
        content = await asyncio.to_thread(render_chat_pdf, request.title, request.html)
    except ChatPdfTooLong as error:
        raise log_and_http_error(
            error, 413, "This export is too long for one PDF.", log = logger
        ) from error
    except Exception as error:
        raise log_and_http_error(error, 500, "Could not render the PDF.", log = logger) from error
    return Response(content = content, media_type = "application/pdf")
