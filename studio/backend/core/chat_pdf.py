# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chat export to PDF with PyMuPDF. Run as `python -m core.chat_pdf` (JSON {title, html} on stdin, PDF on stdout):
PyMuPDF is not thread safe, so the server renders each export in a child process, never a thread."""

from __future__ import annotations

import html
import io
import itertools
import json
import sys
from pathlib import Path

TOO_LONG_EXIT_CODE = 3
MAX_EXPORT_PAGES = 5000
_SIDE_MARGIN_PT = 60
_VERTICAL_MARGIN_PT = 58

# MuPDF's HTML engine takes a CSS subset; unknown families fall back to its bundled Noto/Droid fonts.
_PDF_CSS = """
@font-face { font-family: Hellix; src: url(Hellix-Regular.ttf); }
@font-face { font-family: Hellix; src: url(Hellix-SemiBold.ttf); font-weight: bold; }
@font-face { font-family: FiraCode; src: url(FiraCode-Regular.ttf); }
body { margin: 0; font-family: sans-serif; font-size: 10.5pt; line-height: 1.48; color: #28292c; overflow-wrap: break-word; }
h1, h2, h3, h4, .masthead { font-family: Hellix, sans-serif; }
h1 { font-size: 24pt; line-height: 1.16; margin: 15pt 0 19pt 0; color: #18191b; }
.masthead { width: 100%; border-collapse: collapse; }
.masthead td { border: 0; border-bottom: 1pt solid #d4d5d6; padding: 0 0 12pt 0; vertical-align: middle; }
.brand { width: 50%; height: 23pt; }
.masthead .masthead-model { padding: 4.5pt 0 7.5pt 0; font-size: 9pt; text-align: right; color: #595c60; }
.conversation-title { font-size: 16pt; margin: 20pt 0 12pt 0; }
.message { margin: 0 0 13pt 0; }
.model { font-size: 9pt; color: #595c60; margin: 0 0 14pt 0; }
.role { page-break-after: avoid; font-size: 10pt; font-weight: bold; margin: 0; color: #424549; }
.message.user .content, .message.system .content { white-space: pre-wrap; padding-left: 0.5pt; }
.content { padding-top: 4pt; }
.assistant .role { color: #212327; }
.content h1 { font-size: 18pt; margin: 14pt 0 8pt 0; }
.content h2 { font-size: 15pt; margin: 13pt 0 7pt 0; }
.content h3, .content h4 { font-size: 12pt; margin: 12pt 0 6pt 0; }
p { margin: 0 0 8pt 0; }
pre { margin: 13pt 0; padding: 0 12pt; font-family: FiraCode, monospace; font-size: 9pt; line-height: 1.6; white-space: pre-wrap; }
code { font-family: FiraCode, monospace; font-size: 9pt; padding: 0 2pt; }
pre code { padding: 0; }
strong { font-weight: bold; }
ul, ol { margin: 8pt 0 10pt 0; padding-left: 20pt; }
li { margin: 0 0 5pt 0; }
blockquote { margin: 10pt 0; padding: 0 0 0 12pt; border-left: 2pt solid #93969a; color: #595c60; }
table { width: 100%; border-collapse: collapse; margin-bottom: 10pt; }
th, td { overflow-wrap: normal; border: 0.6pt solid #dbdcde; padding: 5pt 7pt; }
th { background-color: #f4f4f4; font-weight: bold; }
details { color: #595c60; margin: 0 0 6pt 0; }
summary { font-style: italic; }
a { color: #212327; }
hr { border: 0; border-top: 0.7pt solid #dbdcde; margin: 14pt 0; }
"""


class ChatPdfTooLong(Exception):
    pass


def render_chat_pdf(
    title: str,
    body_html: str,
    model: str | None = None,
) -> bytes:
    """The archive contains only bundled fonts; no filesystem directory is exposed."""
    import pymupdf

    heading = html.escape(" ".join(title.split()) or "Conversation")
    model_label = f"Model: {html.escape(model)}" if model else ""
    document = (
        '<table class="masthead"><tr><td class="brand">&#160;</td>'
        f'<td class="masthead-model">{model_label}</td></tr></table>'
        f"<h1>{heading}</h1>{body_html}"
    )
    archive = pymupdf.Archive()
    font_dir = Path(__file__).resolve().parents[1] / "assets/pdf"
    for name in ("Hellix-Regular.ttf", "Hellix-SemiBold.ttf", "FiraCode-Regular.ttf"):
        archive.add((font_dir.joinpath(name).read_bytes(), name))
    story = pymupdf.Story(html = document, user_css = _PDF_CSS, archive = archive)
    pre = story.body.find("pre", None, None)
    block_number = 0
    while pre is not None:
        pre.set_id(f"pdf-block-{block_number}")
        block_number += 1
        pre = pre.find_next("pre", None, None)
    code_kinds = {}
    for tag in ("code", "span"):
        node = story.body.find(tag, None, None)
        while node is not None:
            ancestor = node.parent
            while ancestor is not None and ancestor.tagname != "pre":
                ancestor = ancestor.parent
            if tag == "code" or ancestor is not None:
                identifier = f"pdf-code-{len(code_kinds)}"
                node.set_id(identifier)
                block = ancestor.get_attribute_value("id") if ancestor is not None else None
                code_kinds[identifier] = block
            node = node.find_next(tag, None, None)
    inline_boxes = []
    block_boxes = {}

    def record_code(position):
        if position.id in code_kinds:
            rect = pymupdf.Rect(position.rect) * matrix
            if rect.width <= 0 or rect.height <= 0:
                return
            block = code_kinds[position.id]
            if block is None:
                inline_boxes.append((pages, tuple(rect)))
            else:
                key = (pages, block)
                if key in block_boxes:
                    block_boxes[key].include_rect(rect)
                else:
                    block_boxes[key] = rect

    buffer = io.BytesIO()
    writer = pymupdf.DocumentWriter(buffer)
    mediabox = pymupdf.paper_rect("letter")
    where = mediabox + (
        _SIDE_MARGIN_PT,
        _VERTICAL_MARGIN_PT,
        -_SIDE_MARGIN_PT,
        -_VERTICAL_MARGIN_PT,
    )
    width, height = where.width, where.height
    pages = 0
    links = []

    def add_link(position):
        if position.open_close & 1 and (position.href or "").startswith(
            ("http:", "https:", "mailto:")
        ):
            rect = pymupdf.Rect(position.rect) * matrix
            last = links[-1] if links else None
            # One box per word: merge only touching boxes, so text between two anchors stays unlinked.
            if (
                last
                and last[:2] == (pages, position.href)
                and last[2].y0 == rect.y0
                and abs(last[2].x1 - rect.x0) < 0.01
            ):
                last[2].include_rect(rect)
            else:
                links.append((pages, position.href, rect))

    more = 1
    while more:
        if pages >= MAX_EXPORT_PAGES:
            writer.close()
            raise ChatPdfTooLong(f"export exceeds {MAX_EXPORT_PAGES} pages")
        device = writer.begin_page(mediabox)
        scale = 1.0
        more, filled = story.place(pymupdf.Rect(0, 0, width, height))
        # MuPDF never shrinks a table below its natural width: re-place this page shrunk to fit it, so no column is cut off.
        for _ in range(4):
            if filled[2] <= width / scale * 1.001:
                break
            scale = width / filled[2]
            more, filled = story.place(pymupdf.Rect(0, 0, width / scale, height / scale))
        matrix = pymupdf.Matrix(scale, 0, 0, scale, where.x0, where.y0)
        if pages == 0:
            first_page_scale = scale
        story.element_positions(record_code)
        story.element_positions(add_link)
        story.draw(device, matrix)
        writer.end_page()
        pages += 1
    writer.close()

    pdf = pymupdf.open("pdf", buffer.getvalue())
    try:
        # Paint backgrounds beneath selectable text. Story's inline backgrounds are
        # unsupported, and padded pre backgrounds can leak onto later pages.
        for (page_number, _), text_rect in block_boxes.items():
            rect = pymupdf.Rect(where.x0, text_rect.y0 - 6, where.x1, text_rect.y1 + 6) & where
            pdf[page_number].draw_rect(
                rect,
                color = (0.85, 0.86, 0.87),
                fill = (0.973, 0.974, 0.978),
                width = 0.6,
                radius = 0.12,
                overlay = False,
            )
        for page_number, coordinates in dict.fromkeys(inline_boxes):
            rect = (pymupdf.Rect(coordinates) + (-2, -1, 2, 1)) & where
            pdf[page_number].draw_rect(
                rect,
                color = None,
                fill = (0.93, 0.934, 0.94),
                radius = 0.12,
                overlay = False,
            )
        wordmark = pymupdf.Rect(
            where.x0, where.y0, where.x0 + 74 * first_page_scale, where.y0 + 23 * first_page_scale
        )
        pdf[0].insert_image(wordmark, filename = str(font_dir / "wordmark.png"))
        if len(pdf) > 1:
            for number, page in enumerate(pdf, start = 1):
                page.insert_textbox(
                    pymupdf.Rect(where.x0, mediabox.y1 - 40, where.x1, mediabox.y1 - 22),
                    f"{number} / {len(pdf)}",
                    align = pymupdf.TEXT_ALIGN_RIGHT,
                    fontname = "helv",
                    fontsize = 8,
                    color = (0.38, 0.39, 0.4),
                )
        pdf.set_metadata({"title": " ".join(title.split()), "creator": "Unsloth"})
        for number, group in itertools.groupby(links, key = lambda link: link[0]):
            page = pdf[number]
            for _, uri, rect in group:
                page.insert_link({"kind": pymupdf.LINK_URI, "from": rect, "uri": uri})
        # The CJK fallback font alone is ~2 MB; keep only the glyphs used.
        pdf.subset_fonts()
        return pdf.tobytes(garbage = 3, deflate = True)
    finally:
        pdf.close()


def main() -> int:
    request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    try:
        pdf = render_chat_pdf(
            str(request.get("title") or ""), str(request["html"]), request.get("model")
        )
    except ChatPdfTooLong as error:
        print(error, file = sys.stderr)
        return TOO_LONG_EXIT_CODE
    sys.stdout.buffer.write(pdf)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
