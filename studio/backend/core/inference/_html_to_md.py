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
import secrets
from html.parser import HTMLParser
from urllib.parse import urlsplit

__all__ = ["SiteLinks", "html_to_markdown"]

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
# <aside> is NOT skipped: docs use it for admonition callouts (real content); page-furniture asides are excluded by the
# main-content scoping pass instead.

# void elements never produce an end tag, so they must not join the open-element stack used to bound hidden subtrees
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


def _span_attr(attr_dict: dict, name: str) -> int:
    try:
        value = int(attr_dict.get(name) or 1)
    except ValueError:
        return 1
    return value if value >= 0 else 1


def _is_aria_heading(attr_dict: dict) -> bool:
    """True for ``role="heading"``, which titles a page just as ``h1``-``h6`` does.

    ``role`` is a token list authors use for fallbacks (``role="future-role
    heading"``), so any token counts. WAI-ARIA takes the first token naming a
    real role, which would need the whole role table; matching anywhere is the
    safe direction, since keeping a stray title beats dropping a real one."""
    return "heading" in (attr_dict.get("role") or "").lower().split()


# HTML5 optional end tags: a listed start tag implicitly closes an open element of the key type (as browsers do), else
# an unclosed ``<p hidden>``/``<li hidden>`` swallows every following sibling. Keys: closable elements; values: closers.
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


# Item tag -> container tags that re-scope it: a nested container makes an inner item a descendant, not an
# optional-close sibling, so recovery must stop there rather than close (and un-hide) the outer item and leak its
# nested content.
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
# half the 16K fetch result: repeated span cells must leave room for the source text after the table
_MAX_SPAN_CHARS = 8_000
# a long rowspan cell is usually page layout, so only its first row keeps the text
_MAX_REPEATED_CELL_CHARS = 200
# floor per scope candidate once the page-wide total is spent, so an earlier decoy cannot starve a later article
_MIN_SCOPE_SPAN_CHARS = 256
_INLINE_EMPHASIS = {"strong": "**", "b": "**", "em": "*", "i": "*"}

# measured density: 0.94-1.00 for link lists, 0.13-0.90 for content headers
_HEADER_LINK_DENSITY = 0.93
# below this the ratio is noise: link lists start at 182 chars, link-dense headers stop at 93
_HEADER_MIN_CHARS = 150
# Short labels hide huge hrefs, so size the render too: content peaks at 363, link lists at 1609+.
_HEADER_MAX_RENDERED_CHARS = 800

# pages nest headers one or two deep; past this, closing a frame cannot copy an unbounded chain
_MAX_HEADER_NESTING = 8


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
        "outer_in_code",
        "outer_bq_depth",
        "outer_table_depth",
    )

    def __init__(
        self,
        depth: int,
        link_seq: int,
        cell_seq: int,
        in_pre: bool,
        in_code: int,
        bq_depth: int,
        list_depth: int,
        table_depth: int,
    ):
        self.depth = depth
        self.parts: list[str] = []
        # Teed, so a heading routed through a nested buffer survives.
        self.heading_parts: list[str] = []
        self.stripped: bool = False
        # Tallied on emit; re-cleaning each parent's buffer would be quadratic.
        self.rendered_chars: int = 0
        self.heading_chars: int = 0
        # Buffers open now enclose the frame, later ones nest. Link/cell sequence numbers, not flags: an inner one
        # replaces the outer in the renderer's single slot.
        self.outer_list_depth = list_depth
        self.outer_link_seq = link_seq
        self.outer_cell_seq = cell_seq
        self.outer_in_pre = in_pre
        self.outer_in_code = in_code
        self.outer_bq_depth = bq_depth
        self.outer_table_depth = table_depth
        # both exclude heading text: it is kept anyway, so it must not vote on dropping the rest
        self.text_chars: int = 0
        self.link_chars: int = 0

    def render(
        self,
        closed_by_own_tag: bool,
        strip: bool | None = None,
    ) -> str:
        """The buffer, or only its headings when the header is link furniture.

        Without a matching ``</header>`` the header may have adopted the page
        body, so keep it whole."""
        self.stripped = False
        if not closed_by_own_tag:
            return "".join(self.parts)
        headings = "".join(self.heading_parts)
        # droppable chars only: headings survive, and blank structure _cleanup collapses must not inflate a tiny header
        droppable = self.rendered_chars - self.heading_chars
        big_enough = self.text_chars >= _HEADER_MIN_CHARS or droppable >= _HEADER_MAX_RENDERED_CHARS
        if (
            strip
            if strip is not None
            else (big_enough and self.link_chars >= _HEADER_LINK_DENSITY * self.text_chars)
        ):
            self.stripped = True
            # the closing tag's blank line lands after the heading mark pops, so terminate it here
            return headings + "\n\n" if headings.strip() else headings
        return "".join(self.parts)


class SiteLinks:
    """Same-site links as rendered, so a page too long for its budget can drop their URLs."""

    def __init__(self, page_url: str):
        self._host = urlsplit(page_url).hostname
        self._nonce = secrets.token_hex(8)
        self._marker = re.compile(rf"\x00{self._nonce}:(\d+):([se])\x00")
        self._found: dict[int, tuple[str, str]] = {}
        self._full = ""
        self._stripped = ""

    def note(self, link: str, text: str, href: str) -> str:
        try:
            parts = urlsplit(href)
        except ValueError:
            return link
        if parts.scheme not in ("", "http", "https") or parts.hostname not in (
            None,
            self._host,
        ):
            return link
        index = len(self._found)
        self._found[index] = (link, text)
        return f"\x00{self._nonce}:{index}:s\x00{link}\x00{self._nonce}:{index}:e\x00"

    def clean(self, markdown: str) -> str:
        return self._marker.sub("", markdown)

    def finish(self, markdown: str) -> str:
        self._full = self.clean(markdown)
        self._stripped = self._strip_marked(markdown)
        return self._full

    def strip(self, markdown: str) -> str:
        return self._stripped if markdown == self._full else markdown

    def _strip_marked(self, markdown: str) -> str:
        out: list[str] = []
        last = 0
        while match := self._marker.search(markdown, last):
            index, edge = int(match.group(1)), match.group(2)
            if edge != "s" or index not in self._found:
                out.append(markdown[last : match.end()])
                last = match.end()
                continue
            end_marker = f"\x00{self._nonce}:{index}:e\x00"
            end = markdown.find(end_marker, match.end())
            if end < 0:
                out.append(markdown[last : match.end()])
                last = match.end()
                continue
            link, text = self._found[index]
            rendered_link = markdown[match.end() : end]
            if rendered_link == link.replace("|", "\\|"):
                text = text.replace("|", "\\|")
            out.append(markdown[last : match.start()])
            out.append(text)
            last = end + len(end_marker)
        out.append(markdown[last:])
        return self.clean("".join(out))


class _TableFrame:
    """Outer table state suspended while a nested table is rendered."""

    __slots__ = (
        "current_row",
        "cell_parts",
        "in_cell",
        "cell_seq",
        "header_row_done",
        "row_has_th",
        "is_first_row",
        "cell_colspan",
        "cell_rowspan",
        "row_spans",
        "in_row",
        "outer_bq_depth",
        "parts",
    )

    def __init__(self, renderer: _MarkdownRenderer):
        self.current_row = renderer._current_row
        self.cell_parts = renderer._cell_parts
        self.in_cell = renderer._in_cell
        self.cell_seq = renderer._cell_seq
        self.header_row_done = renderer._header_row_done
        self.row_has_th = renderer._row_has_th
        self.is_first_row = renderer._is_first_row
        self.cell_colspan = renderer._cell_colspan
        self.cell_rowspan = renderer._cell_rowspan
        self.row_spans = renderer._row_spans
        self.in_row = renderer._in_row
        self.outer_bq_depth = len(renderer._bq_stack)
        self.parts: list[str] = []


class _TitleButtonScan:
    """Find buttons that hold their heading's whole title (``<h3><button>Question</button></h3>``, an accordion
    trigger): the heading's first visible button with text, with no visible heading text outside it. Fed the
    renderer's own tokens, so it parses exactly as the renderer does; the kept buttons render on a second pass."""

    def __init__(self) -> None:
        self.keep: set[tuple[int, int]] = set()  # HTMLParser.getpos() of each kept <button>
        self._open: list[str] = []
        self._closable_open = (
            0  # open tags with an optional end tag, as _MarkdownRenderer counts them
        )
        self._muted: list[int] = []  # open-tag indices of hidden / skipped subtrees
        self._button_at: int | None = None
        self._button_frame: list | None = None  # heading whose candidate is the open button
        # visible text outside any button, counted so a heading reads "text since I opened" in O(1)
        self._text_seq = 0
        # per open heading: [open-tag index, candidate position or None, candidate has text, _text_seq at open]
        self._headings: list[list] = []

    def starttag(self, tag: str, attrs: list[tuple[str, str | None]], pos: tuple[int, int]) -> None:
        self._close_implicit(tag)
        if tag in _VOID_TAGS:
            return
        attr_dict = dict(attrs)
        self._open.append(tag)
        if tag in _IMPLICIT_CLOSERS:
            self._closable_open += 1
        index = len(self._open) - 1
        if _is_hidden_element(attr_dict) or (tag in _SKIP_TAGS and tag != "button"):
            self._muted.append(index)
        if tag in _HEADING_TAGS or _is_aria_heading(attr_dict):
            self._headings.append([index, None, False, self._text_seq])
        elif tag == "button" and self._button_at is None:
            self._button_at = index
            frame = self._headings[-1] if self._headings else None
            if frame is not None and frame[1] is None and not self._muted:
                frame[1] = pos
                self._button_frame = frame

    def endtag(self, tag: str) -> None:
        for i in range(len(self._open) - 1, -1, -1):
            if self._open[i] == tag:
                self._pop_to(i)
                return

    def data(self, data: str) -> None:
        if self._muted or not data.strip():
            return
        if self._button_at is None:
            self._text_seq += 1
        elif self._button_frame is not None:
            self._button_frame[2] = True

    def finish(self) -> None:
        # a page cut inside a heading (the fetch cap) still keeps its title
        self._close_headings(0)

    def _close_implicit(self, tag: str) -> None:
        """The renderer's optional-end-tag recovery (``_MarkdownRenderer._close_implicit``), so both see one tree."""
        if not self._closable_open:
            return
        barriers = _CLOSE_BARRIERS.get(tag, ())
        while True:
            close_at = None
            for i in range(len(self._open) - 1, -1, -1):
                name = self._open[i]
                if tag in _IMPLICIT_CLOSERS.get(name, ()):
                    close_at = i
                    break
                if name in barriers:
                    break
            if close_at is None:
                return
            self._pop_to(close_at)

    def _pop_to(self, i: int) -> None:
        if self._button_at is not None and self._button_at >= i:
            self._button_at = None
            # an icon-only control releases the slot for the trigger after it
            if self._button_frame is not None and not self._button_frame[2]:
                self._button_frame[1] = None
            self._button_frame = None
        while self._muted and self._muted[-1] >= i:
            self._muted.pop()
        self._close_headings(i)
        self._closable_open -= sum(1 for name in self._open[i:] if name in _IMPLICIT_CLOSERS)
        del self._open[i:]

    def _close_headings(self, i: int) -> None:
        while self._headings and self._headings[-1][0] >= i:
            _, button, has_text, seq = self._headings.pop()
            if button is not None and has_text and self._text_seq == seq:
                self.keep.add(button)


# (len, hash) of recent pages -> their title buttons, so main-content passes over one page scan it once
_TITLE_BUTTON_CACHE: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
_TITLE_BUTTON_CACHE_SIZE = 4


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
        span_char_limit: int = _MAX_SPAN_CHARS,
        header_decisions: list[bool] | None = None,
        site_links: SiteLinks | None = None,
        page_span_limit: int | None = None,
    ):
        super().__init__(convert_charrefs = False)
        # getpos() of buttons that carry their heading's title, found by _TitleButtonScan
        self.title_buttons: frozenset[tuple[int, int]] = frozenset()
        self._title_scan: _TitleButtonScan | None = None
        self._site_links = site_links
        self._out: list[str] = []
        self._skip_depth: int = 0

        # Main-content scoping: emit only while inside a scope tag.
        self._scope_tags = scope_tags
        self._scope_depth: int = 0

        # one boundary per top-level scope element, so a swarm of tiny cards cannot clear the gate
        self.scope_segments: list[str] = []
        self._scope_seg_start: int | None = None

        # open non-void tags plus hidden-start indices, so an omitted </p>/</li> cannot leave the renderer stuck hidden
        self._open_tags: list[str] = []
        # Open tags that can be closed implicitly; zero lets _close_implicit skip the scan.
        self._closable_open: int = 0
        self._hidden_marks: list[int] = []

        # Open <header> buffers, innermost last. Empty unless strip_header.
        self._strip_header = strip_header
        self._header_stack: list[_HeaderFrame] = []
        self.header_decisions: list[bool] = []
        self._header_decisions = header_decisions
        # Furniture chars removed, so a candidate is sized as the page wrote it.
        self._dropped_chars: int = 0
        self._seg_dropped_start: int = 0
        self.scope_dropped: list[int] = []
        # Heading text per segment: role="heading", hgroup and linked h1 render as prose, so ATX reparsing alone
        # cannot keep them out of the gate.
        self._seg_heading_texts: list[str] = []
        self.scope_heading_prose: list[int] = []
        # Open-tag indices of headings, unwound with _hidden_marks.
        self._heading_marks: list[int] = []

        self._link_href: str | None = None
        self._link_text_parts: list[str] = []
        self._in_link: bool = False
        self._link_seq: int = 0
        # a link wrapping a heading emits after the mark pops, so the tee is told to treat it so
        self._link_had_heading: bool = False
        self._link_heading_parts: list[str] = []
        self._emit_as_heading: bool = False
        self._replaying: bool = False
        # Credited only at </a>: an <a> left open adopts body prose, which is not furniture.
        self._link_header_chars: int = 0

        self._list_stack: list[str] = []  # "ul" or "ol"
        self._ol_counter: list[int] = []
        # Just emitted a list marker: a block opening the item (loose <li><p>) stays on the marker line.
        self._li_marker_pending: bool = False

        self._in_table: bool = False
        self._current_row: list[str] = []
        self._cell_parts: list[str] = []
        self._in_cell: bool = False
        self._cell_seq: int = 0
        self._next_cell_seq: int = 0
        self._header_row_done: bool = False
        self._row_has_th: bool = False
        self._is_first_row: bool = False
        self._cell_colspan: int = 1
        self._cell_rowspan: int = 1
        self._row_spans: dict[int, tuple[str, int]] = {}
        self._table_stack: list[_TableFrame] = []
        self._span_chars: int = 0
        self._has_generated_spans: bool = False
        self._span_char_limit: int = min(_MAX_SPAN_CHARS, max(0, span_char_limit))
        self._scope_span_limit: int = self._span_char_limit
        self._page_span_left: int | None = page_span_limit
        self._in_row: bool = False

        self._in_pre: bool = False
        self._pre_parts: list[str] = []
        # Depth, not a flag: nested <code> opens two spans and each </code> owes a backtick.
        self._inline_code_depth: int = 0

        # Blockquote state: stack of buffers so nested blockquotes get the right ">" depth.
        self._bq_stack: list[list[str]] = []

    def _nested_buffer_open(self, frame: _HeaderFrame) -> bool:
        """True when a side buffer opened *inside* *frame* still holds content.

        Such a buffer emits into the frame when it closes; an enclosing one
        (already open at ``<header>``) must not capture it."""
        # Only the buffer _emit would pick matters, in its order; OR-ing them calls an enclosing one nested.
        if self._in_link:
            return self._link_seq != frame.outer_link_seq
        if self._in_cell:
            return self._cell_seq != frame.outer_cell_seq
        if self._in_pre:
            return not frame.outer_in_pre
        if self._table_stack and self._table_depth() > frame.outer_table_depth:
            return True
        return len(self._bq_stack) > frame.outer_bq_depth

    def _emit(self, text: str) -> None:
        if text:
            self._li_marker_pending = False
        frame = self._header_stack[-1] if self._header_stack else None
        # Tee wherever the text routes, so a heading in a nested buffer is captured. A link opened inside the frame
        # delivers twice, so tee only the formatted form; an enclosing link, once.
        in_nested_link = (
            self._in_link and self._link_seq != frame.outer_link_seq if frame else False
        )
        # replays never tee: the text was teed on the way in, and _finish_link re-arms the tee itself
        as_heading = (
            self._heading_marks and not in_nested_link and not self._replaying
        ) or self._emit_as_heading
        measured = self._site_links.clean(text) if self._site_links is not None else text
        if frame is not None and as_heading:
            frame.heading_parts.append(text)
            frame.heading_chars += len(measured.strip())
        # for the eligibility gate, frame or not; link text waits for _finish_link to count once
        if not self._replaying and (
            (self._heading_marks and not self._in_link) or self._emit_as_heading
        ):
            self._seg_heading_texts.append(text)
        nested_open = self._nested_buffer_open(frame) if frame is not None else False
        # Tally once, on the emit reaching the frame; counting again on flush doubled it.
        if frame is not None and not nested_open:
            frame.rendered_chars += len(measured.strip())
            frame.parts.append(text)
            return
        if self._in_link:
            self._link_text_parts.append(text)
            if self._heading_marks:
                self._link_heading_parts.append(text)
        elif self._in_cell:
            self._cell_parts.append(text)
        elif self._in_pre:
            self._pre_parts.append(text)
        elif self._table_stack and len(self._bq_stack) <= self._table_stack[-1].outer_bq_depth:
            self._table_stack[-1].parts.append(text)
        elif self._bq_stack:
            self._bq_stack[-1].append(text)
        else:
            self._out.append(text)

    def _seg_heading_prose(self) -> int:
        """Heading characters in this segment that the gate would otherwise read as
        body prose. ATX headings carry their own ``#`` here and so score zero."""
        text = "".join(self._seg_heading_texts)
        if self._site_links is not None:
            text = self._site_links.clean(text)
        return _visible_chars(text)

    def _drain_pre(self) -> None:
        """Emit the open ``<pre>`` and empty it, so a late ``</pre>`` cannot replay
        it outside a stripped header and push the article past the fetch cap."""
        raw = "".join(self._pre_parts)
        self._in_pre = False
        self._pre_parts = []
        fence = _fence_for(raw)
        self._emit_replay(f"\n\n{fence}\n{raw}\n{fence}\n\n")

    def _drain_blockquote(self) -> None:
        """Pop the innermost quote and emit it prefixed. Not used by ``flush_pending``,
        which re-nests into the parent stack instead of emitting."""
        prefixed = self._prefix_blockquote("".join(self._bq_stack.pop()))
        if prefixed:
            self._emit_replay("\n\n" + prefixed + "\n\n")

    def _emit_replay(self, text: str) -> None:
        """Emit a flushed buffer's own output, which must not be teed as a heading
        a second time (the raw text was teed when it entered the buffer)."""
        was, self._replaying = self._replaying, True
        try:
            self._emit(text)
        finally:
            self._replaying = was

    def _prefix_blockquote(self, content: str) -> str:
        """Prefix every line of *content* with ``> ``."""
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
    def _table_depth(self) -> int:
        return len(self._table_stack) + int(self._in_table)

    def _start_table(self) -> None:
        if self._in_table:
            self._table_stack.append(_TableFrame(self))
        self._in_table = True
        self._current_row = []
        self._cell_parts = []
        self._in_cell = False
        self._header_row_done = False
        self._row_has_th = False
        self._is_first_row = True
        self._cell_colspan = 1
        self._cell_rowspan = 1
        self._row_spans = {}
        self._in_row = False
        self._emit("\n\n")

    def _finish_table(self) -> None:
        self._finish_cell()
        self._finish_row()
        self._row_spans = {}
        self._emit("\n")
        if not self._table_stack:
            self._in_table = False
            return

        frame = self._table_stack.pop()
        nested = "".join(frame.parts)
        self._current_row = frame.current_row
        self._cell_parts = frame.cell_parts
        self._in_cell = frame.in_cell
        self._cell_seq = frame.cell_seq
        self._header_row_done = frame.header_row_done
        self._row_has_th = frame.row_has_th
        self._is_first_row = frame.is_first_row
        self._cell_colspan = frame.cell_colspan
        self._cell_rowspan = frame.cell_rowspan
        self._row_spans = frame.row_spans
        self._in_row = frame.in_row
        self._in_table = True
        self._emit_replay(nested)

    def _finish_cell(self) -> None:
        if not self._in_cell:
            return
        self._in_cell = False
        cell_text = "".join(self._cell_parts).strip().replace("\n", " ")
        if not self._table_stack:
            cell_text = cell_text.replace("|", "\\|")
        self._cell_parts = []
        self._fill_spanned_cells(0)
        col = len(self._current_row)
        self._current_row.append(cell_text)
        if self._span_chars >= self._span_char_limit:
            return
        outer_cells = sum(frame.in_cell for frame in self._table_stack)
        extra_col_cost = 9 + 2 * outer_cells if not self._header_row_done else 3 + outer_cells
        # a nested table flattens and drops empty cells, so its colspan padding would only spend budget
        extra_cols = (
            0
            if self._table_stack
            else min(
                self._cell_colspan - 1,
                (self._span_char_limit - self._span_chars) // extra_col_cost,
            )
        )
        self._current_row.extend([""] * extra_cols)
        generated = extra_col_cost * extra_cols
        self._span_chars += generated
        self._has_generated_spans |= bool(extra_cols)
        if self._cell_rowspan > 1:
            repeated = cell_text if self._visible_len(cell_text) <= _MAX_REPEATED_CELL_CHARS else ""
            for i in range(extra_cols + 1):
                self._row_spans[col + i] = (repeated if i == 0 else "", self._cell_rowspan)

    def _visible_len(self, text: str) -> int:
        # site-link markers vanish from the output, so they must not spend the span budget
        if self._site_links is not None:
            text = self._site_links.clean(text)
        # budgets assume four ASCII chars per token; a CJK char is about one token
        return len(text) + 3 * sum(not ch.isascii() for ch in text)

    def _fill_spanned_cells(self, width: int) -> None:
        outer_cells = sum(frame.in_cell for frame in self._table_stack)
        while self._span_chars < self._span_char_limit and (
            len(self._current_row) < width or len(self._current_row) in self._row_spans
        ):
            span = self._row_spans.get(len(self._current_row))
            text = span[0] if span else ""
            cost = self._visible_len(text) + 3 + (text.count("|") + 1) * outer_cells
            if self._span_chars + cost > self._span_char_limit:
                self._span_chars = self._span_char_limit
                break
            self._current_row.append(text)
            self._span_chars += cost
            self._has_generated_spans = True

    def _finish_row(self) -> None:
        in_row, self._in_row = self._in_row, False
        if not self._current_row and not (in_row and self._row_spans):
            return
        if not self._current_row:
            # a wholly generated row also adds its opening pipe, newline, and enclosing quote prefixes
            row_cost = (
                3 + 2 * len(self._bq_stack) + sum(frame.in_cell for frame in self._table_stack)
            )
            self._span_chars = min(self._span_char_limit, self._span_chars + row_cost)
        self._fill_spanned_cells(max(self._row_spans, default = -1) + 1)
        if self._span_chars >= self._span_char_limit:
            self._row_spans = {}
        else:
            self._row_spans = {
                col: (text, rows - 1) for col, (text, rows) in self._row_spans.items() if rows > 1
            }
        if not self._current_row:
            return
        if self._table_stack:
            # nested tables flatten into the outer cell: replaying their pipes re-escaped them per level
            line = " ".join(cell for cell in self._current_row if cell)
            if line:
                self._emit_replay(line + "\n")
            self._is_first_row = False
            self._current_row = []
            self._row_has_th = False
            return
        line = "| " + " | ".join(self._current_row) + " |"
        self._emit_replay(line + "\n")
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
        heading_text = re.sub(r"\s+", " ", "".join(self._link_heading_parts)).strip()
        href = self._link_href or ""
        self._in_link = False
        self._link_text_parts = []
        self._link_heading_parts = []
        # an anchor wrapping a heading AND other content tees the title alone, else the nav rides
        partial = bool(heading_text) and heading_text != text
        self._emit_as_heading = self._link_had_heading and not partial
        self._link_had_heading = False
        self._link_header_chars = 0
        if href and text:
            link = f"[{text}]({href})"
            if self._site_links is not None and not self._inline_code_depth:
                link = self._site_links.note(link, text, href)
            self._emit(link)
        elif text:
            self._emit(text)
        self._emit_as_heading = False
        if partial and self._header_stack:
            frame = self._header_stack[-1]
            frame.heading_parts.append(heading_text + "\n\n")
            frame.heading_chars += len(heading_text)
            # Preserved by hand, so tell the gate too or a title-only card reads as body prose.
            self._seg_heading_texts.append(heading_text)

    # Tag handlers. Structural bookkeeping shared by every start tag (skip/hidden/scope).
    def _truncate_open_tags(self, index: int) -> None:
        """Drop the open-tag stack above *index*, keeping the closable count."""
        for name in self._open_tags[index:]:
            if name in _IMPLICIT_CLOSERS:
                self._closable_open -= 1
        del self._open_tags[index:]

    def _close_implicit(self, tag: str) -> None:
        """HTML5 optional-end-tag recovery for a start tag about to open.

        Pops each implicitly-closed ancestor (and its hidden marks), scanning the
        whole stack so an open ``<p>``/``<li>`` still closes under an unclosed inline
        ``<span>``. Stops at a ``_CLOSE_BARRIERS`` container so recovery never crosses
        a nested list/table/dl and leaks the outer item's hidden content. Runs even
        for skipped ``<nav>``/``<footer>``, which also close ``<p>``."""
        if not self._closable_open:
            return
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
            self._truncate_open_tags(close_at)
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
            strip = (
                self._header_decisions[len(self.header_decisions)]
                if self._header_decisions is not None
                else None
            )
            out = frame.render(closed_by_own_tag, strip)
            self.header_decisions.append(frame.stripped)
            if frame.stripped:
                self._dropped_chars += max(0, frame.rendered_chars - frame.heading_chars)
            self._emit(out)
            closed_by_own_tag = False

    def _finalize_nested_buffers(self, frame: _HeaderFrame) -> None:
        """Close side buffers opened inside *frame* whose end tags the page omitted.

        Their content is the header's, so it has to land in the frame before the
        strip is judged; otherwise it is emitted afterwards and escapes."""
        if self._in_link and self._link_seq != frame.outer_link_seq:
            # the header boundary proves this anchor did not adopt the body: its text is furniture
            frame.link_chars += self._link_header_chars
            self._finish_link()
        # code opened OUTSIDE the header is the page's; closing it here leaves </code> unpaired
        while self._inline_code_depth > frame.outer_in_code:
            self._inline_code_depth -= 1
            self._emit("`")
        # before the cell: _finish_row emits, and an open <pre> would swallow the row as CODE|  |
        if self._in_pre and not frame.outer_in_pre:
            self._drain_pre()
        if self._in_cell and self._cell_seq != frame.outer_cell_seq:
            self._finish_cell()
            self._finish_row()
        while len(self._bq_stack) > frame.outer_bq_depth:
            self._drain_blockquote()
        # A list left open in the header would indent the body's lists under phantom nesting.
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
            if tag in _IMPLICIT_CLOSERS:
                self._closable_open += 1
            if _is_hidden_element(attr_dict):
                self._hidden_marks.append(len(self._open_tags) - 1)
            if tag in _HEADING_TAGS or tag == "hgroup" or _is_aria_heading(attr_dict):
                self._heading_marks.append(len(self._open_tags) - 1)
                if self._in_link:
                    self._link_had_heading = True
            if (
                self._strip_header
                and tag == "header"
                and not self._hidden_marks
                and len(self._header_stack) < _MAX_HEADER_NESTING
            ):
                self._header_stack.append(
                    _HeaderFrame(
                        len(self._open_tags) - 1,
                        self._link_seq if self._in_link else -1,
                        self._cell_seq if self._in_cell else -1,
                        self._in_pre,
                        self._inline_code_depth,
                        len(self._bq_stack),
                        len(self._list_stack),
                        self._table_depth(),
                    )
                )
        elif _is_hidden_element(attr_dict):
            return False
        if self._scope_tags is not None and tag in self._scope_tags:
            self._flush_header_frames()
            if self._scope_depth == 0:
                if self._page_span_left is not None:
                    self._page_span_left = max(0, self._page_span_left - self._span_chars)
                    self._span_char_limit = min(
                        self._scope_span_limit, max(self._page_span_left, _MIN_SCOPE_SPAN_CHARS)
                    )
                self._span_chars = 0
                self._scope_seg_start = len(self._out)
                self._seg_dropped_start = self._dropped_chars
                self._seg_heading_texts = []
            self._scope_depth += 1
        if self._hidden_marks:
            return False
        if self._scope_tags is not None and self._scope_depth == 0:
            return False
        return True

    def _exit_tag(self, tag: str) -> bool:
        """Pop to the matching open tag; return True when the end tag should
        be rendered (False = it closed inside a hidden / out-of-scope region)."""
        # Recover an <a> left open before the segment is recorded, or its text is stranded.
        if self._in_link and self._scope_tags is not None and tag in self._scope_tags:
            self._finish_link()
        suppressed = bool(self._hidden_marks) or (
            self._scope_tags is not None and self._scope_depth == 0
        )
        # An element that IS the heading emits when its buffer closes, after the mark pops; flush it here while the tee
        # still recognises it.
        if self._in_link and tag == "a" and self._heading_marks:
            self._finish_link()
        if tag not in _VOID_TAGS:
            for i in range(len(self._open_tags) - 1, -1, -1):
                if self._open_tags[i] == tag:
                    self._truncate_open_tags(i)
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
                self.scope_heading_prose.append(self._seg_heading_prose())
                self._scope_seg_start = None
        return not suppressed

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if self._title_scan is not None:
            self._title_scan.starttag(tag, attrs, self.getpos())
        title_button = (
            tag == "button" and bool(self.title_buttons) and self.getpos() in self.title_buttons
        )

        if self._skip_depth:
            if tag in _SKIP_TAGS:
                self._skip_depth += 1
            return

        # Recover optional end tags before the skip decision: a skipped <nav>/<footer> still implicitly closes an open
        # <p>, releasing its hidden mark so following siblings render.
        self._close_implicit(tag)

        if tag in _SKIP_TAGS and not (title_button and self._heading_marks):
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
            self._link_heading_parts = []
            self._in_link = True
            self._link_seq += 1
            self._link_header_chars = 0

        elif tag in _INLINE_EMPHASIS:
            self._emit(_INLINE_EMPHASIS[tag])

        elif tag == "br":
            self._emit("\n")

        elif tag in _BLOCK_TAGS:
            if not self._li_marker_pending:
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
            self._li_marker_pending = True

        elif tag == "pre":
            self._pre_parts = []
            self._in_pre = True

        elif tag == "code" and not self._in_pre:
            self._inline_code_depth += 1
            self._emit("`")

        elif tag == "table":
            self._start_table()

        elif tag == "tr":
            self._finish_cell()
            self._finish_row()
            self._in_row = True

        elif tag in ("thead", "tbody", "tfoot"):
            self._finish_cell()
            self._finish_row()
            self._row_spans = {}

        elif tag in ("th", "td"):
            self._finish_cell()
            self._cell_parts = []
            self._in_cell = True
            self._next_cell_seq += 1
            self._cell_seq = self._next_cell_seq
            self._cell_colspan = min(max(1, _span_attr(attr_dict, "colspan")), 1000)
            # rowspan="0" runs to the end of the row group
            self._cell_rowspan = _span_attr(attr_dict, "rowspan") or 65534
            if tag == "th":
                self._row_has_th = True

        elif tag == "img":
            return

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._title_scan is not None:
            self._title_scan.endtag(tag)

        # a kept title button closes through _exit_tag; skipped ones always raised _skip_depth
        if tag in _SKIP_TAGS and (self._skip_depth or tag != "button"):
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return

        if not self._exit_tag(tag):
            return

        if tag == "li":
            self._li_marker_pending = False

        elif tag in _HEADING_TAGS:
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
                self._drain_blockquote()

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

        elif tag == "pre" and self._in_pre:
            self._drain_pre()

        # already closed means a frame recovered it; a second backtick codes the rest of the page
        elif tag == "code" and not self._in_pre and self._inline_code_depth:
            self._inline_code_depth -= 1
            self._emit("`")

        elif tag in ("th", "td"):
            self._finish_cell()

        elif tag == "tr":
            self._finish_cell()
            self._finish_row()

        elif tag in ("thead", "tbody", "tfoot"):
            self._finish_cell()
            self._finish_row()
            self._row_spans = {}

        elif tag == "table":
            self._finish_table()

    def _text_suppressed(self) -> bool:
        if self._skip_depth or self._hidden_marks:
            return True
        return self._scope_tags is not None and self._scope_depth == 0

    def handle_data(self, data: str) -> None:
        if self._title_scan is not None:
            self._title_scan.data(data)
        if self._text_suppressed():
            return
        if self._in_pre:
            self._count_header_text(data)
            self._pre_parts.append(data)
            return
        # Preserve literal whitespace inside inline <code> spans.
        if self._inline_code_depth:
            self._count_header_text(data)
            self._emit(data)
            return
        # Collapse all whitespace (including newlines) per HTML rules.
        text = re.sub(r"\s+", " ", data)
        # Sized after collapsing, as the reader sees it: raw spaces in a link cleared the floor at ~100% density and
        # dropped the byline.
        self._count_header_text(text)
        # Suppress whitespace-only nodes between table elements (source indentation).
        if self._in_table and not self._in_cell and not text.strip():
            return
        # Source indentation between <li> and its first block must not end the marker line.
        if self._li_marker_pending and not text.strip():
            return
        self._emit(text)

    def handle_entityref(self, name: str) -> None:
        text = html.unescape(f"&{name};")
        if self._title_scan is not None:
            self._title_scan.data(text)
        if self._text_suppressed():
            return
        self._count_header_text(text)
        self._emit(text)

    def handle_charref(self, name: str) -> None:
        text = html.unescape(f"&#{name};")
        if self._title_scan is not None:
            self._title_scan.data(text)
        if self._text_suppressed():
            return
        self._count_header_text(text)
        self._emit(text)

    def flush_pending(self) -> None:
        """Flush open side-buffers into ``_out`` after close(), recovering truncated HTML."""
        # Headers first: a frame finalizes its inner buffers, then emits into the enclosing link or cell, which must
        # still be open here; it is finalized below.
        self._flush_header_frames()

        if self._in_link:
            self._finish_link()

        while self._inline_code_depth:
            self._inline_code_depth -= 1
            self._emit("`")

        while self._table_stack:
            self._finish_table()
        self._finish_cell()
        self._finish_row()

        if self._in_pre:
            self._drain_pre()

        while self._bq_stack:
            content = "".join(self._bq_stack.pop())
            prefixed = self._prefix_blockquote(content)
            if not prefixed:
                continue
            if self._bq_stack:
                self._bq_stack[-1].append("\n\n" + prefixed + "\n\n")
            else:
                self._out.append("\n\n" + prefixed + "\n\n")

        # A scope left open by truncated HTML never reached _exit_tag, so its output never joined scope_segments and
        # would score 0. Flush the still-open segment here (after the side-buffers) so a truncated main-content page is
        # scored.
        if self._scope_seg_start is not None:
            self.scope_segments.append("".join(self._out[self._scope_seg_start :]))
            self.scope_dropped.append(self._dropped_chars - self._seg_dropped_start)
            self.scope_heading_prose.append(self._seg_heading_prose())
            self._scope_seg_start = None
            self._scope_depth = 0


def _cleanup(text: str) -> str:
    """Normalize whitespace and blank lines, preserving fenced code blocks verbatim."""
    lines = text.split("\n")
    out: list[str] = []
    fence = 0
    blank_run = 0

    for line in lines:
        stripped = line.rstrip(" \t")
        moved = _fence_state(stripped, fence)
        if moved != fence:
            fence = moved
            blank_run = 0
            out.append(stripped)
            continue

        if fence:
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


# Known boilerplate fragments stripped from main-content conversions, matched only against short lines. Sources:
# GitHub page furniture / client-side error placeholders, skip-links, cookie banners.
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
# Only shorter lines are eligible for boilerplate dropping; real content sentences quoting a fragment run longer.
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


def _fence_state(line: str, fence: int) -> int:
    """Fence width after *line*, 0 outside a block. Every pass over rendered text
    must agree with ``_fence_for``: a block opened with a longer fence is closed
    only by a run at least as long, so a literal ``` inside it is content."""
    stripped = line.strip()
    if len(stripped) < 3 or stripped.count("`") != len(stripped):
        return fence
    if not fence:
        return len(stripped)
    return 0 if len(stripped) >= fence else fence


def _strip_boilerplate_lines(text: str, site_links: SiteLinks | None = None) -> str:
    """Drop short lines that consist entirely of known page-furniture phrases.

    Fenced code blocks are preserved verbatim: boilerplate never renders
    inside ``<pre>``, while READMEs legitimately quote error strings."""
    out: list[str] = []
    fence = 0
    for line in text.split("\n"):
        moved = _fence_state(line, fence)
        if moved != fence:
            fence = moved
            out.append(line)
            continue
        measured = site_links.clean(line) if site_links is not None else line
        if (
            not fence
            and len(measured) <= _BOILERPLATE_MAX_LINE_CHARS
            and _line_is_boilerplate(measured)
        ):
            continue
        out.append(line)
    # Collapse blank runs the dropped lines may have left behind.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _new_renderer(
    source_html: str,
    scope_tags: frozenset[str] | None,
    strip_header: bool,
    site_links: SiteLinks | None = None,
    span_char_limit: int | None = None,
    header_decisions: list[bool] | None = None,
) -> _MarkdownRenderer:
    def build(
        title_buttons: frozenset[tuple[int, int]], scan: _TitleButtonScan | None
    ) -> _MarkdownRenderer:
        renderer = _MarkdownRenderer(
            scope_tags = scope_tags,
            strip_header = strip_header,
            site_links = site_links,
            span_char_limit = 2 * len(source_html) if span_char_limit is None else span_char_limit,
            header_decisions = header_decisions,
            page_span_limit = 2 * len(source_html),
        )
        renderer.title_buttons = title_buttons
        renderer._title_scan = scan
        renderer.feed(source_html)
        renderer.close()
        renderer.flush_pending()
        return renderer

    key = (len(source_html), hash(source_html))
    known = _TITLE_BUTTON_CACHE.get(key)
    if known is not None:
        return build(known, None)
    scan = _TitleButtonScan()
    renderer = build(frozenset(), scan)
    scan.finish()
    keep = frozenset(scan.keep)
    if len(_TITLE_BUTTON_CACHE) >= _TITLE_BUTTON_CACHE_SIZE:
        _TITLE_BUTTON_CACHE.clear()  # safe under concurrent fetches, unlike evicting one by one
    _TITLE_BUTTON_CACHE[key] = keep
    # only a page that has accordion titles pays a second pass
    return build(keep, None) if keep else renderer


def _render(
    source_html: str,
    scope_tags: frozenset[str] | None,
    strip_header: bool = False,
    site_links: SiteLinks | None = None,
    span_char_limit: int | None = None,
) -> str:
    renderer = _new_renderer(source_html, scope_tags, strip_header, site_links, span_char_limit)
    return _cleanup("".join(renderer._out))


def _select_main_scope_render(
    source_html: str,
    tag: str,
    site_links: SiteLinks | None,
    span_char_limit: int | None = None,
) -> tuple[int, str]:
    """Length and boilerplate-stripped render of the largest single ``<tag>``
    subtree. Sizing candidates one at a time stops many tiny sibling cards from
    clearing the threshold together, and returning that one subtree keeps
    unrelated siblings (related cards, comment threads) out of the output.

    A candidate earns its place on the prose it RETAINED, then gets its dropped
    header furniture added back to rank against siblings. Furniture must not buy
    eligibility: a card whose header was the only bulk would otherwise clear the
    gate on deleted bytes and suppress the ``<main>`` holding the real page.

    Nor may it dominate: the credit is capped at the retained render, so removed
    furniture can never be the majority of a score. Uncapped, a teaser with a
    1000 link header outranked a sibling holding five times its real text."""
    renderer = _new_renderer(
        source_html,
        frozenset({tag}),
        strip_header = True,
        site_links = site_links,
        span_char_limit = span_char_limit,
    )
    # generated span cells must not lift a scope past the content gate or rank; only span pages pay this second pass
    scoring = (
        _new_renderer(
            source_html,
            frozenset({tag}),
            strip_header = True,
            span_char_limit = 0,
            header_decisions = renderer.header_decisions,
        )
        if renderer._has_generated_spans
        else renderer
    )
    best_len = 0
    best_render = ""
    for i, seg in enumerate(renderer.scope_segments):
        rendered = _strip_boilerplate_lines(_cleanup(seg), site_links)
        scored = _strip_boilerplate_lines(_cleanup(scoring.scope_segments[i]), site_links)
        if site_links is not None:
            scored = site_links.clean(scored)
        prose = _visible_chars(scored) - scoring.scope_heading_prose[i]
        if prose < _MIN_MAIN_CONTENT_CHARS:
            continue
        size = len(scored) + min(scoring.scope_dropped[i], len(scored))
        if size > best_len:
            best_len = size
            best_render = rendered
    return best_len, best_render


def _visible_chars(text: str) -> int:
    """Visible characters in *text*, ignoring blank lines and link destinations.

    Headings are NOT discounted here. The renderer already tallies what it marked
    as a heading (``_seg_heading_prose``), which sees ``role="heading"``, hgroup
    and a linked ``h1``; re-deriving that from ATX syntax could not, and running
    both meant two answers to one question."""
    return sum(_visible_len(line) for line in text.split("\n") if line.strip())


def _fence_for(raw: str) -> str:
    """Fence long enough to survive backticks in *raw*, as CommonMark requires:
    a literal ``` line inside the block would otherwise close it early."""
    longest = run = 0
    for char in raw:
        run = run + 1 if char == "`" else 0
        longest = max(longest, run)
    return "`" * max(3, longest + 1)


def _visible_len(line: str) -> int:
    """Length of *line* without Markdown link destinations. A tracking URL is not
    prose: a card holding one [Read](/x?<300 bytes>) otherwise scored 339 visible
    characters off 4 and displaced the article. Scanned, so no backtracking."""
    total, i, n = 0, 0, len(line)
    open_bracket = False
    while i < n:
        if line[i] == "\\":
            total += 2
            i += 2
            continue
        if line[i] == "[":
            open_bracket = True
        # With no opening bracket, "](" is literal and the parens after it are prose.
        if open_bracket and line[i] == "]" and i + 1 < n and line[i + 1] == "(":
            # Destinations may hold balanced or escaped parens, so the first ) does not end them.
            j, depth = i + 2, 1
            while j < n and depth:
                char = line[j]
                if char == "\\":
                    j += 2
                    continue
                depth += (char == "(") - (char == ")")
                j += 1
            if depth:
                total += 1
                i += 1
                continue
            i = j
            open_bracket = False
            continue
        total += 1
        i += 1
    return total


# A scoped conversion below this size is judged not to be the page's main content (e.g. an empty <article> stub) and the
# next candidate is tried.
_MIN_MAIN_CONTENT_CHARS = 200


def html_to_markdown(
    source_html: str,
    *,
    main_content: bool = False,
    site_links: SiteLinks | None = None,
    max_span_chars: int | None = None,
) -> str:
    """Convert HTML to Markdown (headings, links, emphasis, lists, tables, blockquotes, code, entities).

    ``<script>``, ``<style>``, and ``<head>`` are stripped entirely, as are
    subtrees hidden from rendering (``hidden`` / ``aria-hidden="true"``).

    ``main_content=True`` applies a readability-style heuristic for page
    fetches: prefer the ``<article>`` subtree (GitHub renders READMEs there),
    then ``<main>``, falling back to the whole document, reduce a link-only
    ``<header>`` to the heading it carries, and strip known boilerplate
    fragments from the result.

    ``site_links`` records the links back into the page's own site; the output is unchanged.

    ``max_span_chars`` caps the cells generated for ``rowspan``/``colspan``, so a caller with a
    smaller result budget keeps room for the text after a table.
    """
    source_html = source_html.replace("\r\n", "\n").replace("\r", "\n")
    span_limit = 2 * len(source_html)
    if max_span_chars is not None:
        span_limit = min(span_limit, max_span_chars)
    rendered = ""
    if main_content:
        for scope_tag in ("article", "main"):
            # Render only the chosen subtree so sibling <article>/<main> elements do not leak in.
            length, rendered = _select_main_scope_render(
                source_html, scope_tag, site_links, span_limit
            )
            if length >= _MIN_MAIN_CONTENT_CHARS:
                break
        else:
            rendered = _strip_boilerplate_lines(
                _render(
                    source_html,
                    None,
                    strip_header = True,
                    site_links = site_links,
                    span_char_limit = span_limit,
                ),
                site_links,
            )
    else:
        rendered = _render(source_html, None, site_links = site_links, span_char_limit = span_limit)
    return site_links.finish(rendered) if site_links is not None else rendered
