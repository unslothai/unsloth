# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Minimal HTML-to-Markdown converter using only the standard library.

Replaces the external ``html2text`` (GPL-3.0) dependency with a ~250-line
``html.parser.HTMLParser`` subclass. Covers headings, links, bold/italic,
lists, tables, blockquotes, code blocks, and entity decoding.

``main_content=True`` additionally applies a readability-style heuristic:
conversion is scoped to the page's ``<article>`` (else ``<main>``) subtree
when one exists and carries substantial text, and known boilerplate
fragments (skip-links, client-side error placeholders, session banners,
cookie prompts) are stripped from the result.
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
        # Never-rendered / non-content elements. Browsers do not display
        # <template> children or <dialog> without .show(), and form widgets
        # carry UI chrome ("Cancel", "Submit feedback"), not page content.
        "template",
        "dialog",
        "button",
        "select",
        "datalist",
        "aside",
    }
)

# Void elements never produce an end tag, so they must not join the
# open-element stack used to bound hidden subtrees.
_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)


def _is_hidden_element(attr_dict: dict) -> bool:
    """True when the element is not rendered by a browser: the ``hidden``
    attribute or ``aria-hidden="true"``. Client-side placeholders (e.g.
    GitHub's ``<div data-show-on-forbidden-error hidden>`` "Uh oh! There was
    an error while loading." blocks) ship in the HTML but are only shown by
    JavaScript on error, so they must not reach the Markdown output."""
    if "hidden" in attr_dict and attr_dict.get("hidden") != "false":
        return True
    return (attr_dict.get("aria-hidden") or "").strip().lower() == "true"
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
    """HTMLParser subclass that emits Markdown tokens into a list.

    ``scope_tags`` restricts emission to the subtree(s) of the given tags
    (e.g. ``{"article"}``): outside them every handler is a no-op, which is
    how the readability-style main-content pass drops page furniture.
    """

    def __init__(self, scope_tags: frozenset[str] | None = None):
        super().__init__(convert_charrefs = False)
        self._out: list[str] = []
        self._skip_depth: int = 0

        # Main-content scoping: emit only while inside a scope tag.
        self._scope_tags = scope_tags
        self._scope_depth: int = 0

        # Hidden-subtree tracking (`hidden` / aria-hidden="true"): a stack of
        # currently-open non-void tags plus the stack indices where a hidden
        # element started. End tags pop to the matching open tag, so an
        # omitted </p>/<li> close cannot leave the renderer stuck hidden.
        self._open_tags: list[str] = []
        self._hidden_marks: list[int] = []

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

        # Blockquote state: stack of buffers so nested blockquotes get the right ">" depth.
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
        # Strip trailing whitespace, then collapse blank lines.
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

    # Table helpers: flush open cells/rows so omitted </td>/</tr> don't lose data.
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

    # Link text helper: normalize whitespace so block content in <a> stays single-line.
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
    # Structural bookkeeping shared by every start tag (skip/hidden/scope).
    def _enter_tag(self, tag: str, attr_dict: dict) -> bool:
        """Track open/hidden/scope state; return True when the tag's content
        should be rendered (False = suppressed)."""
        if tag not in _VOID_TAGS:
            self._open_tags.append(tag)
            if _is_hidden_element(attr_dict):
                self._hidden_marks.append(len(self._open_tags) - 1)
        if self._scope_tags is not None and tag in self._scope_tags:
            self._scope_depth += 1
        if self._hidden_marks:
            return False
        if self._scope_tags is not None and self._scope_depth == 0:
            return False
        return True

    def _exit_tag(self, tag: str) -> bool:
        """Pop to the matching open tag; return True when the end tag should
        be rendered (False = it closed inside a hidden / out-of-scope region)."""
        suppressed = bool(self._hidden_marks) or (
            self._scope_tags is not None and self._scope_depth == 0
        )
        if tag not in _VOID_TAGS:
            # Pop to the innermost matching open tag (recovers omitted closes).
            for i in range(len(self._open_tags) - 1, -1, -1):
                if self._open_tags[i] == tag:
                    del self._open_tags[i:]
                    while self._hidden_marks and self._hidden_marks[-1] >= i:
                        self._hidden_marks.pop()
                    break
        if self._scope_tags is not None and tag in self._scope_tags and self._scope_depth > 0:
            self._scope_depth -= 1
        return not suppressed

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()

        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return

        attr_dict = dict(attrs)
        if not self._enter_tag(tag, attr_dict):
            return

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
            # Flush open cell/row from a prior row that omitted </td>/</tr>.
            self._finish_cell()
            self._finish_row()

        elif tag in ("th", "td"):
            self._finish_cell()  # handles omitted </td>/</th>
            self._cell_parts = []
            self._in_cell = True
            if tag == "th":
                self._row_has_th = True

        elif tag == "img":
            # Skip images: keeps text readable, avoids data-URI amplification.
            return

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return

        if not self._exit_tag(tag):
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
            # Flush remaining row (handles omitted </tr>).
            self._finish_cell()
            self._finish_row()
            self._in_table = False
            self._emit("\n")

    # ------------------------------------------------------------------
    # Text / entity handlers
    # ------------------------------------------------------------------
    def _text_suppressed(self) -> bool:
        if self._skip_depth or self._hidden_marks:
            return True
        return self._scope_tags is not None and self._scope_depth == 0

    def handle_data(self, data: str) -> None:
        if self._text_suppressed():
            return
        if self._in_pre:
            self._pre_parts.append(data)
            return
        # Preserve literal whitespace inside inline <code> spans.
        if self._in_inline_code:
            self._emit(data)
            return
        # Collapse all whitespace (including newlines) per HTML rules.
        text = re.sub(r"\s+", " ", data)
        # Suppress whitespace-only nodes between table elements (source indentation).
        if self._in_table and not self._in_cell and not text.strip():
            return
        self._emit(text)

    def handle_entityref(self, name: str) -> None:
        if self._text_suppressed():
            return
        self._emit(html.unescape(f"&{name};"))

    def handle_charref(self, name: str) -> None:
        if self._text_suppressed():
            return
        self._emit(html.unescape(f"&#{name};"))

    # Flush pending buffers (handles truncated HTML from capped fetches)
    def flush_pending(self) -> None:
        """Flush open side-buffers into ``_out`` after close(), recovering truncated HTML."""
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

        # Flatten any open blockquote buffers (innermost first).
        while self._bq_stack:
            content = "".join(self._bq_stack.pop())
            prefixed = self._prefix_blockquote(content)
            if not prefixed:
                continue
            if self._bq_stack:
                self._bq_stack[-1].append("\n\n" + prefixed + "\n\n")
            else:
                self._out.append("\n\n" + prefixed + "\n\n")


# Post-processing
def _cleanup(text: str) -> str:
    """Normalize whitespace and blank lines, preserving fenced code blocks verbatim."""
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


# Known boilerplate fragments stripped from main-content conversions. Matched
# per line, and only against short lines (a long paragraph that merely
# mentions one of these phrases is kept). Deterministic substring list, no
# scoring. Sources: GitHub page furniture / client-side error placeholders,
# skip-links, cookie banners.
_BOILERPLATE_FRAGMENTS = (
    "skip to content",
    "skip to main content",
    "there was an error while loading",
    "please reload this page",
    "you can't perform that action at this time",
    "you signed in with another tab or window",
    "you signed out in another tab or window",
    "you switched accounts on another tab or window",
    "reload to refresh your session",
    "you must be signed in to change notification settings",
    "uh oh!",
    "{{ message }}",
    "this website uses cookies",
    "we use cookies",
    "accept all cookies",
    "manage cookie preferences",
)
# Only lines shorter than this are eligible for boilerplate dropping; real
# content sentences quoting one of the fragments run longer.
_BOILERPLATE_MAX_LINE_CHARS = 300


def _strip_boilerplate_lines(text: str) -> str:
    """Drop short lines that consist of known page-furniture fragments.

    Fenced code blocks are preserved verbatim: boilerplate never renders
    inside ``<pre>``, while READMEs legitimately quote error strings."""
    out: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if not in_fence and len(line) <= _BOILERPLATE_MAX_LINE_CHARS:
            lowered = line.lower()
            if any(fragment in lowered for fragment in _BOILERPLATE_FRAGMENTS):
                continue
        out.append(line)
    # Collapse blank runs the dropped lines may have left behind.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _render(source_html: str, scope_tags: frozenset[str] | None) -> str:
    renderer = _MarkdownRenderer(scope_tags = scope_tags)
    renderer.feed(source_html)
    renderer.close()
    renderer.flush_pending()
    raw = "".join(renderer._out)
    return _cleanup(raw)


# A scoped conversion below this size is judged not to be the page's main
# content (e.g. an empty <article> stub) and the next candidate is tried.
_MIN_MAIN_CONTENT_CHARS = 200


# Public API
def html_to_markdown(source_html: str, *, main_content: bool = False) -> str:
    """Convert HTML to Markdown (headings, links, emphasis, lists, tables, blockquotes, code, entities).

    ``<script>``, ``<style>``, and ``<head>`` are stripped entirely, as are
    subtrees hidden from rendering (``hidden`` / ``aria-hidden="true"``).

    ``main_content=True`` applies a readability-style heuristic for page
    fetches: prefer the ``<article>`` subtree (GitHub renders READMEs there),
    then ``<main>``, falling back to the whole document, and strip known
    boilerplate fragments from the result.
    """
    # Normalize line endings before parsing.
    source_html = source_html.replace("\r\n", "\n").replace("\r", "\n")
    if main_content:
        for scope in (frozenset({"article"}), frozenset({"main"})):
            text = _strip_boilerplate_lines(_render(source_html, scope))
            if len(text) >= _MIN_MAIN_CONTENT_CHARS:
                return text
        return _strip_boilerplate_lines(_render(source_html, None))
    return _render(source_html, None)
