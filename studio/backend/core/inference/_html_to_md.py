# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Minimal HTML-to-Markdown converter using only the standard library.

Replaces the external ``html2text`` (GPL-3.0) dependency with a ~250-line
``html.parser.HTMLParser`` subclass. Covers headings, links, bold/italic,
lists, tables, blockquotes, code blocks, and entity decoding.

``main_content=True`` also applies a readability-style heuristic: scope
conversion to the page's ``<article>`` (else ``<main>``) subtree when it
carries substantial text, and strip known boilerplate fragments (skip-links,
error placeholders, session banners, cookie prompts) from the result.
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
        # Never-rendered / form-chrome elements, not page content.
        "template",
        "dialog",
        "button",
        "select",
        "datalist",
    }
)
# <aside> is NOT skipped: docs use it for admonition callouts (real content);
# page-furniture asides are excluded by the main-content scoping pass instead.

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


def _style_hides_element(style: str) -> bool:
    """True when an inline ``style`` sets ``display:none`` / ``visibility:hidden``.

    Parsed per property so an unrelated value that merely contains ``none`` is
    not misread as hidden."""
    lowered = style.lower()
    if "none" not in lowered and "hidden" not in lowered:
        return False
    for declaration in style.split(";"):
        prop, sep, value = declaration.partition(":")
        if not sep:
            continue
        prop = prop.strip().lower()
        # Drop any !important flag and keep the first token of the value.
        value = value.split("!", 1)[0].strip().lower()
        if prop == "display" and value == "none":
            return True
        if prop == "visibility" and value == "hidden":
            return True
    return False


def _is_hidden_element(attr_dict: dict) -> bool:
    """True when the element is not rendered: ``hidden`` attribute,
    ``aria-hidden="true"``, or an inline ``style`` hiding it. Such JS-only
    placeholders ship in the HTML but must not reach the output. ``hidden`` is
    enumerated: any present value (even ``hidden="false"``) means not rendered."""
    if "hidden" in attr_dict:
        return True
    if (attr_dict.get("aria-hidden") or "").strip().lower() == "true":
        return True
    return _style_hides_element(attr_dict.get("style") or "")


def _is_aria_heading(attr_dict: dict) -> bool:
    """True for ``role="heading"``, which titles a page just as ``h1``-``h6`` does."""
    return (attr_dict.get("role") or "").strip().lower() == "heading"


# HTML5 optional end tags: a listed start tag implicitly closes an open element
# of the key type (as browsers do), else an unclosed ``<p hidden>``/``<li hidden>``
# swallows every following sibling. Keys: closable elements; values: closers.
_P_CLOSING_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "details",
        "div",
        "dl",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hgroup",
        "hr",
        "main",
        "menu",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "ul",
    }
)
_IMPLICIT_CLOSERS: dict = {
    "p": _P_CLOSING_TAGS,
    "li": frozenset({"li"}),
    "dt": frozenset({"dt", "dd"}),
    "dd": frozenset({"dt", "dd"}),
    "tr": frozenset({"tr"}),
    "td": frozenset({"td", "th", "tr"}),
    "th": frozenset({"td", "th", "tr"}),
    "option": frozenset({"option", "optgroup"}),
    "optgroup": frozenset({"optgroup"}),
}


# Item tag -> container tags that re-scope it: a nested container makes an inner
# item a descendant, not an optional-close sibling, so recovery must stop there
# rather than close (and un-hide) the outer item and leak its nested content.
_CLOSE_BARRIERS: dict = {
    "li": frozenset({"ul", "ol", "menu"}),
    "dt": frozenset({"dl"}),
    "dd": frozenset({"dl"}),
    "tr": frozenset({"table"}),
    "td": frozenset({"table"}),
    "th": frozenset({"table"}),
    "option": frozenset({"select", "datalist"}),
    "optgroup": frozenset({"select", "datalist"}),
}


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

# A <header> is furniture only when almost all links (nav, Wikipedia's language
# dropdown): live link lists measure 0.94-1.00, content headers 0.13-0.90.
_HEADER_LINK_DENSITY = 0.93
# Below this the ratio is too noisy: live link lists start at 182 chars, link-dense
# content headers top out at 93.
_HEADER_MIN_CHARS = 150
# Short labels can carry huge hrefs, so judge rendered size too: content headers
# render to at most 363 chars, link lists to 1609 and up.
_HEADER_MAX_RENDERED_CHARS = 800


class _HeaderFrame:
    """Buffered ``<header>`` output plus the link tally used to judge it.

    Buffering (like ``_bq_stack``) defers the decision to ``</header>``, once the
    whole subtree is known. A header no end tag closes is emitted unchanged."""

    __slots__ = (
        "depth",
        "parts",
        "heading_parts",
        "text_chars",
        "link_chars",
        "stripped",
        "rendered_chars",
        "heading_chars",
        "outer_list_depth",
        "outer_link_seq",
        "outer_cell_seq",
        "outer_in_pre",
        "outer_bq_depth",
    )

    def __init__(
        self, depth: int, link_seq: int, cell_seq: int, in_pre: bool, bq_depth: int, list_depth: int
    ):
        self.depth = depth
        self.parts: list[str] = []
        # Heading output, teed so a heading routed through a nested buffer survives.
        self.heading_parts: list[str] = []
        # Visible chars tallied as they are emitted. Counting here keeps nested
        # headers linear; re-cleaning each parent's cumulative buffer is quadratic.
        # Set by render() so the caller need not re-measure its output.
        self.stripped: bool = False
        self.rendered_chars: int = 0
        self.heading_chars: int = 0
        # Side buffers open at this point enclose the frame; later ones are nested.
        # Links and cells are held by sequence number, not a flag: an inner one
        # replaces the outer in the renderer's single slot, so identity is needed.
        self.outer_list_depth = list_depth
        self.outer_link_seq = link_seq
        self.outer_cell_seq = cell_seq
        self.outer_in_pre = in_pre
        self.outer_bq_depth = bq_depth
        # Both exclude heading text (never dropped, so it must not vote on dropping
        # the rest); only href anchors count as links.
        self.text_chars: int = 0
        self.link_chars: int = 0

    def render(self, closed_by_own_tag: bool) -> str:
        """The buffer, or only its headings when the header is link furniture.

        Without a matching ``</header>`` the header may have adopted the page
        body, so keep it whole."""
        self.stripped = False
        if not closed_by_own_tag:
            return "".join(self.parts)
        headings = "".join(self.heading_parts)
        # Only the droppable part, and only its visible characters: a heading is
        # kept either way, and blank structure that _cleanup collapses (500 empty
        # <div>s) must not make a tiny header look huge.
        droppable = self.rendered_chars - self.heading_chars
        big_enough = self.text_chars >= _HEADER_MIN_CHARS or droppable >= _HEADER_MAX_RENDERED_CHARS
        if big_enough and self.link_chars >= _HEADER_LINK_DENSITY * self.text_chars:
            self.stripped = True
            # The closing tag's blank line is emitted after the heading mark is
            # popped, so terminate the heading or the body runs into it.
            return headings + "\n\n" if headings.strip() else headings
        return "".join(self.parts)


class _MarkdownRenderer(HTMLParser):
    """HTMLParser subclass that emits Markdown tokens into a list.

    ``scope_tags`` restricts emission to the subtree(s) of the given tags
    (e.g. ``{"article"}``): outside them every handler is a no-op, which is
    how the readability-style main-content pass drops page furniture.
    """

    def __init__(
        self,
        scope_tags: frozenset[str] | None = None,
        strip_header: bool = False,
    ):
        super().__init__(convert_charrefs = False)
        self._out: list[str] = []
        self._skip_depth: int = 0

        # Main-content scoping: emit only while inside a scope tag.
        self._scope_tags = scope_tags
        self._scope_depth: int = 0

        # Output boundaries per top-level scope element, so a caller can size each
        # candidate alone and a swarm of tiny sibling cards can't clear the threshold.
        self.scope_segments: list[str] = []
        self._scope_seg_start: int | None = None

        # Hidden-subtree tracking: stack of open non-void tags plus the indices
        # where a hidden element started. End tags pop to the matching tag, so
        # an omitted </p>/<li> close cannot leave the renderer stuck hidden.
        self._open_tags: list[str] = []
        self._hidden_marks: list[int] = []

        # Open <header> buffers, innermost last. Empty unless strip_header.
        self._strip_header = strip_header
        self._header_stack: list[_HeaderFrame] = []
        # Furniture chars removed, so a candidate is sized as the page wrote it.
        self._dropped_chars: int = 0
        self._seg_dropped_start: int = 0
        self.scope_dropped: list[int] = []
        # Open-tag indices of headings, unwound with _hidden_marks.
        self._heading_marks: list[int] = []

        # Link state
        self._link_href: str | None = None
        self._link_text_parts: list[str] = []
        self._in_link: bool = False
        self._link_seq: int = 0
        # A link wrapping a heading emits after the heading mark is gone, so the
        # tee is told to treat that one emit as heading output.
        self._link_had_heading: bool = False
        self._emit_as_heading: bool = False
        # Text under the open <a>, credited only at </a>: an <a> left open adopts
        # body prose, which is not furniture.
        self._link_header_chars: int = 0

        # List state
        self._list_stack: list[str] = []  # "ul" or "ol"
        self._ol_counter: list[int] = []

        # Table state
        self._in_table: bool = False
        self._current_row: list[str] = []
        self._cell_parts: list[str] = []
        self._in_cell: bool = False
        self._cell_seq: int = 0
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
    def _nested_buffer_open(self, frame: _HeaderFrame) -> bool:
        """True when a side buffer opened *inside* *frame* still holds content.

        Such a buffer emits into the frame when it closes; an enclosing one
        (already open at ``<header>``) must not capture it."""
        # Only the buffer _emit would actually pick matters, and in its order: a
        # blockquote inside a cell-enclosed header still routes to the cell, so
        # OR-ing across all of them would wrongly call that nested.
        if self._in_link:
            return self._link_seq != frame.outer_link_seq
        if self._in_cell:
            return self._cell_seq != frame.outer_cell_seq
        if self._in_pre:
            return not frame.outer_in_pre
        return len(self._bq_stack) > frame.outer_bq_depth

    def _emit(self, text: str) -> None:
        frame = self._header_stack[-1] if self._header_stack else None
        # Tee wherever the text is routed, so a heading inside a nested buffer is
        # captured. A link opened inside the frame delivers its text twice (raw,
        # then formatted by _finish_link), so only the formatted form is teed; a
        # link that encloses the frame never re-delivers, so it is teed as it comes.
        in_nested_link = (
            self._in_link and self._link_seq != frame.outer_link_seq if frame else False
        )
        as_heading = (self._heading_marks and not in_nested_link) or self._emit_as_heading
        if frame is not None and as_heading:
            frame.heading_parts.append(text)
            frame.heading_chars += len(text.strip())
        if frame is not None:
            frame.rendered_chars += len(text.strip())
        if frame is not None and not self._nested_buffer_open(frame):
            frame.parts.append(text)
            return
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
        self._link_text_parts = []
        self._emit_as_heading = self._link_had_heading
        self._link_had_heading = False
        # Recovery paths reach here without </a>, so drop the uncredited tally.
        self._link_header_chars = 0
        if href and text:
            self._emit(f"[{text}]({href})")
        elif text:
            self._emit(text)
        self._emit_as_heading = False

    # ------------------------------------------------------------------
    # Tag handlers
    # ------------------------------------------------------------------
    # Structural bookkeeping shared by every start tag (skip/hidden/scope).
    def _close_implicit(self, tag: str) -> None:
        """HTML5 optional-end-tag recovery for a start tag about to open.

        Pops each implicitly-closed ancestor (and its hidden marks), scanning the
        whole stack so an open ``<p>``/``<li>`` still closes under an unclosed inline
        ``<span>``. Stops at a ``_CLOSE_BARRIERS`` container so recovery never crosses
        a nested list/table/dl and leaks the outer item's hidden content. Runs even
        for skipped ``<nav>``/``<footer>``, which also close ``<p>``."""
        barriers = _CLOSE_BARRIERS.get(tag, ())
        while True:
            close_at = None
            for i in range(len(self._open_tags) - 1, -1, -1):
                name = self._open_tags[i]
                if tag in _IMPLICIT_CLOSERS.get(name, ()):
                    close_at = i
                    break
                # A barrier container re-scopes the item; stop before it.
                if name in barriers:
                    break
            if close_at is None:
                break
            del self._open_tags[close_at:]
            while self._hidden_marks and self._hidden_marks[-1] >= close_at:
                self._hidden_marks.pop()
            while self._heading_marks and self._heading_marks[-1] >= close_at:
                self._heading_marks.pop()
            self._close_header_frames(close_at)

    def _close_header_frames(
        self,
        depth: int,
        own_tag: bool = False,
    ) -> None:
        """Judge and emit every buffered header at or below *depth*. Only the
        innermost frame can be the one its own ``</header>`` closed."""
        closed_by_own_tag = own_tag
        while self._header_stack and self._header_stack[-1].depth >= depth:
            self._finalize_nested_buffers(self._header_stack[-1])
            frame = self._header_stack.pop()
            if self._header_stack:
                # Roll the tally outward so an enclosing header is judged whole.
                self._header_stack[-1].text_chars += frame.text_chars
                self._header_stack[-1].link_chars += frame.link_chars
                self._header_stack[-1].heading_parts.extend(frame.heading_parts)
                self._header_stack[-1].heading_chars += frame.heading_chars
            out = frame.render(closed_by_own_tag)
            if frame.stripped:
                self._dropped_chars += max(0, frame.rendered_chars - frame.heading_chars)
            self._emit(out)
            closed_by_own_tag = False

    def _finalize_nested_buffers(self, frame: _HeaderFrame) -> None:
        """Close side buffers opened inside *frame* whose end tags the page omitted.

        Their content is the header's, so it has to land in the frame before the
        strip is judged; otherwise it is emitted afterwards and escapes."""
        if self._in_link and self._link_seq != frame.outer_link_seq:
            # The header boundary proves this anchor did not adopt the body, so
            # its text is link furniture even though no </a> arrived.
            frame.link_chars += self._link_header_chars
            self._finish_link()
        if self._in_inline_code:
            self._in_inline_code = False
            self._emit("`")
        if self._in_cell and self._cell_seq != frame.outer_cell_seq:
            self._finish_cell()
            self._finish_row()
        if self._in_pre and not frame.outer_in_pre:
            raw = "".join(self._pre_parts)
            self._in_pre = False
            self._emit("\n\n```\n" + raw + "\n```\n\n")
        while len(self._bq_stack) > frame.outer_bq_depth:
            prefixed = self._prefix_blockquote("".join(self._bq_stack.pop()))
            if prefixed:
                self._emit("\n\n" + prefixed + "\n\n")
        # A list left open inside the header would otherwise indent the body's
        # own lists under phantom nesting.
        while len(self._list_stack) > frame.outer_list_depth:
            if self._list_stack.pop() == "ol" and self._ol_counter:
                self._ol_counter.pop()

    def _flush_header_frames(self) -> None:
        """Emit every open header unchanged, abandoning the strip."""
        while self._header_stack:
            self._finalize_nested_buffers(self._header_stack[-1])
            self._emit("".join(self._header_stack.pop().parts))

    def _count_header_text(self, text: str) -> None:
        """Tally visible text for the innermost header's link density. Heading text
        is skipped so a long linked heading cannot condemn the byline beside it."""
        if not self._header_stack or self._heading_marks:
            return
        frame = self._header_stack[-1]
        chars = len(text.strip())
        frame.text_chars += chars
        # An anchor with no usable href renders as prose, not as a link.
        if not (self._in_link and self._link_href):
            return
        if self._link_seq == frame.outer_link_seq:
            # An enclosing anchor closes after the header, too late to credit.
            frame.link_chars += chars
        else:
            self._link_header_chars += chars

    def _enter_tag(self, tag: str, attr_dict: dict) -> bool:
        """Track open/hidden/scope state; return True when the tag's content
        should be rendered (False = suppressed). Caller runs ``_close_implicit``
        first so recovery also fires for skipped tags."""
        if tag not in _VOID_TAGS:
            self._open_tags.append(tag)
            if _is_hidden_element(attr_dict):
                self._hidden_marks.append(len(self._open_tags) - 1)
            if tag in _HEADING_TAGS or tag == "hgroup" or _is_aria_heading(attr_dict):
                self._heading_marks.append(len(self._open_tags) - 1)
                if self._in_link:
                    self._link_had_heading = True
            if self._strip_header and tag == "header" and not self._hidden_marks:
                self._header_stack.append(
                    _HeaderFrame(
                        len(self._open_tags) - 1,
                        self._link_seq if self._in_link else -1,
                        self._cell_seq if self._in_cell else -1,
                        self._in_pre,
                        len(self._bq_stack),
                        len(self._list_stack),
                    )
                )
        elif _is_hidden_element(attr_dict):
            # Void elements never join the stack, so suppress a hidden one inline.
            return False
        if self._scope_tags is not None and tag in self._scope_tags:
            # A scope element inside a header would strand its output in the buffer.
            self._flush_header_frames()
            if self._scope_depth == 0:
                self._scope_seg_start = len(self._out)
                self._seg_dropped_start = self._dropped_chars
            self._scope_depth += 1
        if self._hidden_marks:
            return False
        if self._scope_tags is not None and self._scope_depth == 0:
            return False
        return True

    def _exit_tag(self, tag: str) -> bool:
        """Pop to the matching open tag; return True when the end tag should
        be rendered (False = it closed inside a hidden / out-of-scope region)."""
        # Recover an <a> the page left open before the segment is recorded, or its
        # text is stranded in the link buffer.
        if self._in_link and self._scope_tags is not None and tag in self._scope_tags:
            self._finish_link()
        suppressed = bool(self._hidden_marks) or (
            self._scope_tags is not None and self._scope_depth == 0
        )
        # An element that is itself the heading (e.g. <a role="heading">) emits
        # its text when its buffer closes, which happens after the mark below is
        # popped, so flush it here while the tee still recognises it.
        if self._in_link and tag == "a" and self._heading_marks:
            self._finish_link()
        if tag not in _VOID_TAGS:
            # Pop to the innermost matching open tag (recovers omitted closes).
            for i in range(len(self._open_tags) - 1, -1, -1):
                if self._open_tags[i] == tag:
                    del self._open_tags[i:]
                    while self._hidden_marks and self._hidden_marks[-1] >= i:
                        self._hidden_marks.pop()
                    while self._heading_marks and self._heading_marks[-1] >= i:
                        self._heading_marks.pop()
                    self._close_header_frames(i, own_tag = tag == "header")
                    break
        if self._scope_tags is not None and tag in self._scope_tags and self._scope_depth > 0:
            self._scope_depth -= 1
            if self._scope_depth == 0 and self._scope_seg_start is not None:
                self.scope_segments.append("".join(self._out[self._scope_seg_start :]))
                self.scope_dropped.append(self._dropped_chars - self._seg_dropped_start)
                self._scope_seg_start = None
        return not suppressed

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()

        if self._skip_depth:
            # Inside a skipped subtree: only track nested skip depth.
            if tag in _SKIP_TAGS:
                self._skip_depth += 1
            return

        # Recover optional end tags before the skip decision: a skipped
        # <nav>/<footer> still implicitly closes an open <p>, releasing its
        # hidden mark so following siblings render.
        self._close_implicit(tag)

        if tag in _SKIP_TAGS:
            self._skip_depth += 1
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
            self._link_seq += 1
            self._link_header_chars = 0

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
            self._cell_seq += 1
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
            if self._header_stack:
                self._header_stack[-1].link_chars += self._link_header_chars
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
        self._count_header_text(data)
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
        text = html.unescape(f"&{name};")
        self._count_header_text(text)
        self._emit(text)

    def handle_charref(self, name: str) -> None:
        if self._text_suppressed():
            return
        text = html.unescape(f"&#{name};")
        self._count_header_text(text)
        self._emit(text)

    # Flush pending buffers (handles truncated HTML from capped fetches)
    def flush_pending(self) -> None:
        """Flush open side-buffers into ``_out`` after close(), recovering truncated HTML."""
        # Headers first: each frame finalizes the buffers opened inside it, then
        # emits into whatever encloses it, so an enclosing link or cell must still
        # be open here and is finalized below.
        self._flush_header_frames()

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

        # A scope left open by truncated HTML never reached _exit_tag, so its output
        # never joined scope_segments and would score 0. Flush the still-open segment
        # here (after the side-buffers) so a truncated main-content page is scored.
        if self._scope_seg_start is not None:
            self.scope_segments.append("".join(self._out[self._scope_seg_start :]))
            self.scope_dropped.append(self._dropped_chars - self._seg_dropped_start)
            self._scope_seg_start = None
            self._scope_depth = 0


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


# Known boilerplate fragments stripped from main-content conversions, matched
# only against short lines. Sources: GitHub page furniture / client-side error
# placeholders, skip-links, cookie banners.
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
# Only shorter lines are eligible for boilerplate dropping; real content
# sentences quoting a fragment run longer.
_BOILERPLATE_MAX_LINE_CHARS = 300

# Normalized furniture phrases for whole-segment matching. See _line_is_boilerplate.
_BOILERPLATE_NORMALIZED = frozenset(
    re.sub(r"\s+", " ", fragment).strip().casefold().rstrip(".!:")
    for fragment in _BOILERPLATE_FRAGMENTS
)


def _line_is_boilerplate(line: str) -> bool:
    """True only when a whole line is composed of known furniture phrases.

    Splits on sentence terminators and requires every segment to be furniture, so a
    line stacking several phrases is dropped while prose that merely quotes one is
    kept (its other words leave a non-furniture segment)."""
    normalized = re.sub(r"\s+", " ", line).strip().casefold()
    if not normalized:
        return False
    segments = [segment.strip().rstrip(".!:") for segment in re.split(r"[.!]", normalized)]
    segments = [segment for segment in segments if segment]
    return bool(segments) and all(segment in _BOILERPLATE_NORMALIZED for segment in segments)


def _strip_boilerplate_lines(text: str) -> str:
    """Drop short lines that consist entirely of known page-furniture phrases.

    Fenced code blocks are preserved verbatim: boilerplate never renders
    inside ``<pre>``, while READMEs legitimately quote error strings."""
    out: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if not in_fence and len(line) <= _BOILERPLATE_MAX_LINE_CHARS and _line_is_boilerplate(line):
            continue
        out.append(line)
    # Collapse blank runs the dropped lines may have left behind.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _new_renderer(
    source_html: str, scope_tags: frozenset[str] | None, strip_header: bool
) -> _MarkdownRenderer:
    renderer = _MarkdownRenderer(scope_tags = scope_tags, strip_header = strip_header)
    renderer.feed(source_html)
    renderer.close()
    renderer.flush_pending()
    return renderer


def _render(
    source_html: str,
    scope_tags: frozenset[str] | None,
    strip_header: bool = False,
) -> str:
    return _cleanup("".join(_new_renderer(source_html, scope_tags, strip_header)._out))


def _select_main_scope_render(source_html: str, tag: str) -> tuple[int, str]:
    """Length and boilerplate-stripped render of the largest single ``<tag>``
    subtree. Sizing candidates one at a time stops many tiny sibling cards from
    clearing the threshold together, and returning that one subtree keeps
    unrelated siblings (related cards, comment threads) out of the output.

    Candidates are sized with their dropped header furniture added back, so the
    strip never costs an article the size gate or a sibling comparison. A
    candidate that retained nothing is all furniture, and gets no such credit."""
    renderer = _new_renderer(source_html, frozenset({tag}), strip_header = True)
    dropped = renderer.scope_dropped
    best_len = 0
    best_render = ""
    for i, seg in enumerate(renderer.scope_segments):
        rendered = _strip_boilerplate_lines(_cleanup(seg))
        credit = dropped[i] if _non_heading_chars(rendered) >= _MIN_RETAINED_CHARS else 0
        size = len(rendered) + credit
        if size > best_len:
            best_len = size
            best_render = rendered
    return best_len, best_render


# Removed furniture only counts toward a candidate's size once the candidate
# retained this much prose OUTSIDE its headings, so neither an empty scope nor a
# stub carrying one long title can qualify on what was deleted from it.
_MIN_RETAINED_CHARS = 50


_HEADING_LINE = re.compile(r"^(?:\s*(?:[>*+-]|\d+\.)\s*)*#")


def _non_heading_chars(text: str) -> int:
    """Length of *text* ignoring blanks and heading lines, including headings
    behind blockquote or list prefixes (``> # Title``)."""
    return sum(
        len(line) for line in text.split("\n") if line.strip() and not _HEADING_LINE.match(line)
    )


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
    then ``<main>``, falling back to the whole document, reduce a link-only
    ``<header>`` to the heading it carries, and strip known boilerplate
    fragments from the result.
    """
    # Normalize line endings before parsing.
    source_html = source_html.replace("\r\n", "\n").replace("\r", "\n")
    if main_content:
        for scope_tag in ("article", "main"):
            # Render only the chosen subtree so sibling <article>/<main>
            # elements do not leak in once the largest passes the size gate.
            length, rendered = _select_main_scope_render(source_html, scope_tag)
            if length >= _MIN_MAIN_CONTENT_CHARS:
                return rendered
        return _strip_boilerplate_lines(_render(source_html, None, strip_header = True))
    return _render(source_html, None)
