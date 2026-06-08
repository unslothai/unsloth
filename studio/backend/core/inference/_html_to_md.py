# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Minimal HTML-to-Markdown converter using only the standard library.

Replaces the external ``html2text`` (GPL-3.0) dependency with a ~250-line
``html.parser.HTMLParser`` subclass.  Covers headings, links, bold/italic,
lists, tables, blockquotes, code blocks, and entity decoding.
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser

__all__ = ["html_to_markdown"]

_SKIP_TAGS = frozenset(
    {
        "script",
        "style",
        "head",
        "noscript",
        "svg",
        "math",
        "nav",
        "footer",
    }
)
_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "main",
        "aside",
        "figure",
        "figcaption",
        "details",
        "summary",
        "dl",
        "dt",
        "dd",
    }
)
_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_INLINE_EMPHASIS = {"strong": "**", "b": "**", "em": "*", "i": "*"}


class _MarkdownRenderer(HTMLParser):
    """HTMLParser subclass that emits Markdown tokens into a list."""

    def __init__(self):
        super().__init__(convert_charrefs = False)
        self._out: list[str] = []
        self._skip_depth: int = 0

        # Link state
        self._link_href: str | None = None
        self._link_text_parts: list[str] = []
        self._in_link: bool = False

        # List state
        self._list_stack: list[str] = []  # "ul" or "ol"
        self._ol_counter: list[int] = []

        # Table state
        self._in_table: bool = False
        self._current_row: list[str] = []
        self._cell_parts: list[str] = []
        self._in_cell: bool = False
        self._header_row_done: bool = False
        self._row_has_th: bool = False
        self._is_first_row: bool = False

        # Pre/code state
        self._in_pre: bool = False
        self._pre_parts: list[str] = []
        self._in_inline_code: bool = False

        # Blockquote state -- stack of output buffers so nested
        # blockquotes each collect their own content and get prefixed
        # with the correct number of ">" markers on close.
        self._bq_stack: list[list[str]] = []

    # ------------------------------------------------------------------
    def _emit(self, text: str) -> None:
        if self._in_link:
            self._link_text_parts.append(text)
        elif self._in_cell:
            self._cell_parts.append(text)
        elif self._in_pre:
            self._pre_parts.append(text)
        elif self._bq_stack:
            self._bq_stack[-1].append(text)
        else:
            self._out.append(text)

    # ------------------------------------------------------------------
    def _prefix_blockquote(self, content: str) -> str:
        """Prefix every line of *content* with ``> ``."""
        # Strip trailing whitespace first, then collapse blank lines
        content = re.sub(r"[ \t]+$", "", content, flags = re.MULTILINE)
        content = re.sub(r"\n{3,}", "\n\n", content).strip()
        if not content:
            return ""
        lines = content.split("\n")
        prefixed: list[str] = []
        for line in lines:
            if line.strip():
                prefixed.append("> " + line)
            else:
                prefixed.append(">")
        return "\n".join(prefixed)

    # ------------------------------------------------------------------
    # Table helpers -- flush open cells and rows so that HTML with
    # omitted optional end tags (</td>, </tr>) does not lose data.
    # ------------------------------------------------------------------
    def _finish_cell(self) -> None:
        if not self._in_cell:
            return
        self._in_cell = False
        cell_text = "".join(self._cell_parts).strip().replace("\n", " ")
        cell_text = cell_text.replace("|", "\\|")
        self._current_row.append(cell_text)
        self._cell_parts = []

    def _finish_row(self) -> None:
        if not self._current_row:
            return
        line = "| " + " | ".join(self._current_row) + " |"
        self._emit(line + "\n")
        if not self._header_row_done and (self._row_has_th or self._is_first_row):
            sep = "| " + " | ".join("---" for _ in self._current_row) + " |"
            self._emit(sep + "\n")
            self._header_row_done = True
        self._is_first_row = False
        self._current_row = []
        self._row_has_th = False

    # ------------------------------------------------------------------
    # Link text helper -- normalize whitespace so block-level content
    # inside an <a> does not produce multiline Markdown link labels.
    # ------------------------------------------------------------------
    def _finish_link(self) -> None:
        text = re.sub(r"\s+", " ", "".join(self._link_text_parts)).strip()
        href = self._link_href or ""
        self._in_link = False
        if href and text:
            self._emit(f"[{text}]({href})")
        elif text:
            self._emit(text)

    # ------------------------------------------------------------------
    # Tag handlers
    # ------------------------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()

        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return

        attr_dict = dict(attrs)

        if tag in _HEADING_TAGS:
            level = int(tag[1])
            self._emit("\n\n" + "#" * level + " ")

        elif tag == "a":
            self._link_href = attr_dict.get("href")
            self._link_text_parts = []
            self._in_link = True

        elif tag in _INLINE_EMPHASIS:
            self._emit(_INLINE_EMPHASIS[tag])

        elif tag == "br":
            self._emit("\n")

        elif tag in _BLOCK_TAGS:
            self._emit("\n\n")

        elif tag == "hr":
            self._emit("\n\n---\n\n")

        elif tag == "blockquote":
            self._emit("\n\n")
            self._bq_stack.append([])

        elif tag == "ul":
            self._list_stack.append("ul")
            self._emit("\n")

        elif tag == "ol":
            self._list_stack.append("ol")
            start_attr = attr_dict.get("start")
            try:
                start = int(start_attr) if start_attr is not None else 1
            except (ValueError, TypeError):
                start = 1
            self._ol_counter.append(start - 1)
            self._emit("\n")

        elif tag == "li":
            indent = "  " * max(0, len(self._list_stack) - 1)
            if self._list_stack and self._list_stack[-1] == "ol":
                if self._ol_counter:
                    self._ol_counter[-1] += 1
                    self._emit(f"\n{indent}{self._ol_counter[-1]}. ")
                else:
                    self._emit(f"\n{indent}1. ")
            else:
                self._emit(f"\n{indent}* ")

        elif tag == "pre":
            self._pre_parts = []
            self._in_pre = True

        elif tag == "code" and not self._in_pre:
            self._in_inline_code = True
            self._emit("`")

        elif tag == "table":
            self._in_table = True
            self._header_row_done = False
            self._is_first_row = True
            self._emit("\n\n")

        elif tag == "tr":
            # Flush any open cell/row from a previous row that may
            # have omitted its optional </td> or </tr> end tags.
            self._finish_cell()
            self._finish_row()

        elif tag in ("th", "td"):
            # Flush any open cell (handles omitted </td>/<th>)
            self._finish_cell()
            self._cell_parts = []
            self._in_cell = True
            if tag == "th":
                self._row_has_th = True

        elif tag == "img":
            # Skip images -- keeps fetched page text focused on readable
            # content and avoids data-URI amplification.
            return

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return

        if tag in _HEADING_TAGS:
            self._emit("\n\n")

        elif tag == "a":
            self._finish_link()

        elif tag in _INLINE_EMPHASIS:
            self._emit(_INLINE_EMPHASIS[tag])

        elif tag in _BLOCK_TAGS:
            self._emit("\n\n")

        elif tag == "blockquote":
            if self._bq_stack:
                content = "".join(self._bq_stack.pop())
                prefixed = self._prefix_blockquote(content)
                if prefixed:
                    self._emit("\n\n" + prefixed + "\n\n")

        elif tag == "ul":
            if self._list_stack and self._list_stack[-1] == "ul":
                self._list_stack.pop()
            self._emit("\n")

        elif tag == "ol":
            if self._list_stack and self._list_stack[-1] == "ol":
                self._list_stack.pop()
                if self._ol_counter:
                    self._ol_counter.pop()
            self._emit("\n")

        elif tag == "pre":
            raw = "".join(self._pre_parts)
            self._in_pre = False
            block = "```\n" + raw + "\n```"
            self._emit("\n\n" + block + "\n\n")

        elif tag == "code" and not self._in_pre:
            self._in_inline_code = False
            self._emit("`")

        elif tag in ("th", "td"):
            self._finish_cell()

        elif tag == "tr":
            self._finish_cell()
            self._finish_row()

        elif tag == "table":
            # Flush any remaining row (handles omitted </tr>)
            self._finish_cell()
            self._finish_row()
            self._in_table = False
            self._emit("\n")

    # ------------------------------------------------------------------
    # Text / entity handlers
    # ------------------------------------------------------------------
    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_pre:
            self._pre_parts.append(data)
            return
        # Preserve literal whitespace inside inline <code> spans
        if self._in_inline_code:
            self._emit(data)
            return
        # Collapse all whitespace (including newlines) per HTML rules
        text = re.sub(r"\s+", " ", data)
        # Suppress whitespace-only text nodes between table structural
        # elements (indentation from source HTML) to prevent leading
        # spaces from breaking Markdown table row alignment.
        if self._in_table and not self._in_cell and not text.strip():
            return
        self._emit(text)

    def handle_entityref(self, name: str) -> None:
        if self._skip_depth:
            return
        self._emit(html.unescape(f"&{name};"))

    def handle_charref(self, name: str) -> None:
        if self._skip_depth:
            return
        self._emit(html.unescape(f"&#{name};"))

    # ------------------------------------------------------------------
    # Flush pending buffers (handles truncated HTML from capped fetches)
    # ------------------------------------------------------------------
    def flush_pending(self) -> None:
        """Flush any open side-buffers into ``_out``.

        Called after ``close()`` to recover content from truncated HTML
        where closing tags were never seen (common when ``_fetch_page_text``
        caps the download by byte count).
        """
        # Flush innermost buffers first so their content propagates outward.

        if self._in_link:
            self._finish_link()

        if self._in_inline_code:
            self._in_inline_code = False
            self._emit("`")

        self._finish_cell()
        self._finish_row()

        if self._in_pre:
            raw = "".join(self._pre_parts)
            self._in_pre = False
            block = "```\n" + raw + "\n```"
            self._emit("\n\n" + block + "\n\n")

        # Flatten any open blockquote buffers (innermost first)
        while self._bq_stack:
            content = "".join(self._bq_stack.pop())
            prefixed = self._prefix_blockquote(content)
            if not prefixed:
                continue
            if self._bq_stack:
                self._bq_stack[-1].append("\n\n" + prefixed + "\n\n")
            else:
                self._out.append("\n\n" + prefixed + "\n\n")


# ------------------------------------------------------------------
# Post-processing
# ------------------------------------------------------------------
def _cleanup(text: str) -> str:
    """Normalize whitespace and blank lines in the final output.

    Preserves content inside fenced code blocks verbatim so that
    intentional blank lines in ``<pre>`` content are not collapsed.
    """
    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    blank_run = 0

    for line in lines:
        stripped = line.rstrip(" \t")
        if stripped.startswith("```"):
            in_fence = not in_fence
            blank_run = 0
            out.append(stripped)
            continue

        if in_fence:
            # Preserve code block content exactly as-is
            out.append(line)
            continue

        if not stripped:
            blank_run += 1
            if blank_run <= 1:
                out.append("")
            continue

        blank_run = 0
        out.append(stripped)

    return "\n".join(out).strip()


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def html_to_markdown(source_html: str) -> str:
    """Convert an HTML string to Markdown.

    Handles headings, links, bold/italic, lists (ordered and unordered),
    tables, blockquotes, code blocks, and HTML entities.  ``<script>``,
    ``<style>``, and ``<head>`` sections are stripped entirely.
    """
    # Normalize line endings before parsing
    source_html = source_html.replace("\r\n", "\n").replace("\r", "\n")
    renderer = _MarkdownRenderer()
    renderer.feed(source_html)
    renderer.close()
    renderer.flush_pending()
    raw = "".join(renderer._out)
    return _cleanup(raw)
