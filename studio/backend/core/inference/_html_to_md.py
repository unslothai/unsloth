# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stdlib HTML-to-Markdown (replaces GPL html2text); main_content=True scopes to article/main."""

from __future__ import annotations

import html
import itertools
import re
import secrets
import unicodedata
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
# The aside element is not skipped (docs use it for callouts); the main-content scoping pass excludes page-furniture asides instead
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
    """Checks display:none and visibility:hidden per declaration, so a value that merely contains none is not read as hidden"""
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
    """Any present hidden attribute counts, even hidden=false, since presence means not rendered; aria-hidden=true and inline hiding styles count too"""
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
    """Any role token counts: WAI-ARIA's first-token rule needs the full role table, so match loosely."""
    return "heading" in (attr_dict.get("role") or "").lower().split()


# HTML5 optional end tags: a listed start tag implicitly closes an open element of the key type, else an unclosed p hidden swallows later siblings
_P_CLOSING_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "center",
        "dd",
        "details",
        "dialog",
        "dir",
        "div",
        "dl",
        "dt",
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
        "li",
        "listing",
        "main",
        "menu",
        "nav",
        "ol",
        "p",
        "pre",
        "search",
        "section",
        "summary",
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


# A nested container makes an inner item a descendant, so recovery must stop there.
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

# end tags pop through descendants only when their target is in the matching HTML parser scope
_SKIP_RECOVERY_END_TAGS = (
    frozenset(
        {
            "address",
            "applet",
            "article",
            "aside",
            "blockquote",
            "button",
            "caption",
            "center",
            "dd",
            "details",
            "dialog",
            "dir",
            "div",
            "dl",
            "dt",
            "fieldset",
            "figcaption",
            "figure",
            "footer",
            "header",
            "hgroup",
            "li",
            "listing",
            "main",
            "marquee",
            "menu",
            "nav",
            "object",
            "ol",
            "p",
            "pre",
            "search",
            "section",
            "select",
            "summary",
            "table",
            "tbody",
            "td",
            "tfoot",
            "th",
            "thead",
            "tr",
            "ul",
        }
    )
    | _HEADING_TAGS
)
_GENERIC_END_SKIP_TAGS = frozenset({"datalist"})
_SPECIAL_HTML_TAGS = _SKIP_RECOVERY_END_TAGS | frozenset(
    {
        "body",
        "colgroup",
        "form",
        "frameset",
        "head",
        "html",
        "iframe",
        "noembed",
        "noframes",
        "noscript",
        "plaintext",
        "script",
        "style",
        "template",
        "title",
        "xmp",
    }
)
_SCOPE_BARRIERS = frozenset(
    {"applet", "caption", "html", "marquee", "object", "select", "table", "td", "template", "th"}
)
_SVG_SCOPE_BARRIERS = frozenset({"foreignobject", "desc", "title"})
_MATHML_SCOPE_BARRIERS = frozenset({"mi", "mo", "mn", "ms", "mtext", "annotation-xml"})
_MATHML_TEXT_INTEGRATION_POINTS = frozenset({"mi", "mo", "mn", "ms", "mtext"})
_FOREIGN_HTML_BREAKOUT_START_TAGS = frozenset(
    {
        "b",
        "big",
        "blockquote",
        "body",
        "br",
        "center",
        "code",
        "dd",
        "div",
        "dl",
        "dt",
        "em",
        "embed",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "head",
        "hr",
        "i",
        "img",
        "li",
        "listing",
        "menu",
        "meta",
        "nobr",
        "ol",
        "p",
        "pre",
        "ruby",
        "s",
        "small",
        "span",
        "strike",
        "strong",
        "sub",
        "sup",
        "table",
        "tt",
        "u",
        "ul",
        "var",
    }
)
_LIST_ITEM_SCOPE_BARRIERS = _SCOPE_BARRIERS | {"ol", "ul"}
_BUTTON_SCOPE_BARRIERS = _SCOPE_BARRIERS | {"button"}
_IMPLICIT_CLOSE_BARRIERS = {
    tag: _CLOSE_BARRIERS.get(tag, frozenset())
    | (_BUTTON_SCOPE_BARRIERS if tag in _P_CLOSING_TAGS else frozenset())
    for tag in _CLOSE_BARRIERS.keys() | _P_CLOSING_TAGS
}
_TABLE_SCOPE_BARRIERS = frozenset({"html", "table", "template"})
_TABLE_END_TAGS = frozenset({"caption", "table", "tbody", "td", "tfoot", "th", "thead", "tr"})
_TABLE_CONTEXT_END_TAGS = _TABLE_END_TAGS - {"table"}
_TABLE_ONLY_START_TAGS = _TABLE_CONTEXT_END_TAGS | {"col", "colgroup"}
_TABLE_CELL_TAGS = frozenset({"td", "th"})
_TABLE_NESTING_TAGS = _TABLE_CELL_TAGS | {"caption"}
_TABLE_SECTION_TAGS = frozenset({"tbody", "tfoot", "thead"})
_TABLE_SECTION_START_TAGS = _TABLE_SECTION_TAGS | {"caption", "col", "colgroup"}
_INDEXED_OPEN_TAGS = _SKIP_RECOVERY_END_TAGS | _SCOPE_BARRIERS
# repeated span cells use half the 16k fetch limit to leave room for text after the table
_MAX_SPAN_CHARS = 8_000
# long rowspan cells usually encode layout, so only their first row keeps the text
_MAX_REPEATED_CELL_CHARS = 200
# each scope keeps 256 characters after the page-wide budget is spent, preventing decoy starvation
_MIN_SCOPE_SPAN_CHARS = 256
_INLINE_EMPHASIS = {"strong": "**", "b": "**", "em": "*", "i": "*"}

# citation lists after words use commas or en dashes; hyphens, minuses, and thousands stay exponents
_CITATION_LIST = re.compile(r"\d+(?:\s*[–,]\s*\d+)+")
_THOUSANDS = re.compile(r"\d{1,3}(?:,\d{3})+")
_PLAIN_SUFFIXES = frozenset(
    {"st", "nd", "rd", "th", "tm", "sm", "mc", "md", "(tm)", "(sm)", "(r)", "(c)", "mr", "m.r."}
)
# French / Romance ordinals after a digit (1er, 2e, 1º, 2ª); after a letter "e" can be Euler's number
# XVe siècle, François Ier: a Roman numeral takes ordinals like a digit
_ROMAN_NUMERAL_TAIL = re.compile(r"(?<![^\W\d_])[IVXLCDM]+$")
# French superior abbreviations: Mme, Mlle, Mgr, Dr, Pr, no, St, Cie; keyed on the whole base word
_SUPERIOR_ABBREVIATIONS = {
    "M": frozenset({"me", "mes", "lle", "lles", "gr", "e", "r", "rs", "s"}),
    "D": frozenset({"r", "rs", "re", "res"}),
    "P": frozenset({"r", "rs", "re", "res"}),
    "n": frozenset({"o", "os"}),
    "N": frozenset({"o", "os"}),
    "S": frozenset({"t", "te", "ts", "tes", "r"}),
    "J": frozenset({"r"}),
    "C": frozenset({"ie", "ies"}),
}
_LAST_WORD = re.compile(r"(?<![^\W\d_])[^\W\d_]+$")


def _last_word(text: str) -> str:
    match = _LAST_WORD.search(text)
    return match.group() if match else ""


# French second / seconde (2d, 2de, 2ds, 2des) only after a 2: 10<sup>d</sup> stays a power
_SECOND_SUFFIXES = frozenset({"d", "de", "ds", "des"})
_DIGIT_ORDINAL_SUFFIXES = frozenset(
    {
        "e",
        "es",
        "er",
        "ers",
        "re",
        "res",
        "ère",
        "ères",
        "ème",
        "èmes",
        "eme",
        "emes",
        "nd",
        "nde",
        "bis",
        "ter",
        "quater",
        "quinquies",
        "am",
        "pm",
        "a.m.",
        "p.m.",
        "ndes",
        "nds",
        "º",
        "ª",
        "o",
        "a",
    }
)
_MD_DELIMITERS = "*_`"
_STRIP_MD_DELIMITERS = str.maketrans("", "", _MD_DELIMITERS)
# SiteLinks wraps same-site links in invisible \x00 markers; the base is the text before them
_SITE_LINK_MARKER_TAIL = re.compile(r"\x00[0-9a-f]+:\d+:[se]\x00$")
# parts a base lookup reads back: enough for delimiters and link markers, bounded on hostile pages
_SUP_BASE_SCAN_PARTS = 8
_SUP_BASE_SCAN_CHARS = 128
# a caret binds one token: a signed number or one letter goes bare, anything longer in parentheses
_BARE_EXPONENT = re.compile(r"[-+−]?(?:\d+(?:[.,]\d+)?|[^\W\d_])")
# split cents: $19<sup>99</sup> is a price, not an exponent
_PRICE_TAIL = re.compile(r"(\S)\s?\d(?:[\d,.'’]|[ \u00a0\u202f]\d)*$")
# ISO 4217 codes: CHF 19<sup>95</sup> is a price like $19<sup>99</sup>
_CURRENCY_CODES = frozenset(
    (
        "AED AFN ALL AMD ANG AOA ARS AUD AWG AZN BAM BBD BDT BGN BHD BIF BMD BND BOB BRL BSD BTN "
        "BWP BYN BZD CAD CDF CHF CLP CNY COP CRC CUP CVE CZK DJF DKK DOP DZD EGP ERN ETB EUR FJD "
        "FKP GBP GEL GHS GIP GMD GNF GTQ GYD HKD HNL HTG HUF IDR ILS INR IQD IRR ISK JMD JOD JPY "
        "KES KGS KHR KMF KPW KRW KWD KYD KZT LAK LBP LKR LRD LSL LYD MAD MDL MGA MKD MMK MNT MOP "
        "MRU MUR MVR MWK MXN MYR MZN NAD NGN NIO NOK NPR NZD OMR PAB PEN PGK PHP PKR PLN PYG QAR "
        "RON RSD RUB RWF SAR SBD SCR SDG SEK SGD SHP SLE SOS SRD SSP SVC STN SYP SZL THB TJS TMT TND "
        "TOP TRY TTD TWD TZS UAH UGX USD UYU UZS VED VES VND VUV WST XAF XCD XCG XOF XPF YER ZAR ZMW "
        "ZWG ZWL"
    ).split()
)
_THREE_DECIMAL_CURRENCIES = frozenset({"BHD", "IQD", "JOD", "KWD", "LYD", "OMR", "TND"})
# no minor unit: JPY 10<sup>12</sup> is a power, never cents
_ZERO_DECIMAL_CURRENCIES = frozenset(
    "BIF CLP DJF GNF ISK JPY KMF KRW PYG RWF UGX VND VUV XAF XOF XPF".split()
)
_CODE_PRICE_TAIL = re.compile(r"\b([A-Z]{3})[ \u00a0\u202f]?\d(?:[\d,.'’]|[ \u00a0\u202f]\d)*$")
# note markers that keep their plain-text form, like Wikipedia's class="reference"
_FOOTNOTE_CLASSES = frozenset({"reference", "footnote", "footnote-ref", "noteref", "fn", "cite"})
# class token parts (split on - and _) starting with these mark a note too: footnote-reference, citation
_FOOTNOTE_CLASS_PREFIXES = (
    "footnote",
    "noteref",
    "cite",
    "citation",
    "endnote",
    "fnref",
    "reference",
)
_CLASS_PART_SPLIT = re.compile(r"[-_]")
# deeper <sup> nests render as plain text: each tracked level rescans its whole suffix on close
_MAX_SUP_DEPTH = 8


def _visible_tail(text: str) -> str:
    """*text* without the emphasis/code delimiters and link markers the renderer appended."""
    while True:
        trimmed = _SITE_LINK_MARKER_TAIL.sub("", text.rstrip(_MD_DELIMITERS))
        if trimmed == text:
            return text
        text = trimmed


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
        # Store link/cell sequence numbers, not flags, since an inner one replaces the outer in one slot
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
        """Returns the buffer, or only its headings when the header is link furniture; without a closing header tag, keep it whole"""
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
        self._namespaces: list[str] = []
        self._html_integration: list[bool] = []
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
        attr_dict = dict(attrs)
        self._prepare_foreign_html_start(tag, attr_dict)
        self._close_implicit(tag)
        if tag in _VOID_TAGS:
            return
        namespace = self._next_namespace(tag)
        self._open.append(tag)
        self._namespaces.append(namespace)
        self._html_integration.append(self._is_html_integration_point(tag, namespace, attr_dict))
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

    def _next_namespace(self, tag: str) -> str:
        if not self._namespaces or self._namespaces[-1] == "html":
            return "svg" if tag == "svg" else "math" if tag == "math" else "html"
        if self._html_integration[-1] and not (
            self._namespaces[-1] == "math" and tag in {"mglyph", "malignmark"}
        ):
            return "svg" if tag == "svg" else "math" if tag == "math" else "html"
        return self._namespaces[-1]

    @staticmethod
    def _is_html_integration_point(tag: str, namespace: str, attrs: dict) -> bool:
        if namespace == "svg":
            return tag in _SVG_SCOPE_BARRIERS
        if namespace != "math":
            return False
        if tag in _MATHML_TEXT_INTEGRATION_POINTS:
            return True
        encoding = (attrs.get("encoding") or "").strip().lower()
        return tag == "annotation-xml" and encoding in {"text/html", "application/xhtml+xml"}

    def _prepare_foreign_html_start(self, tag: str, attrs: dict) -> None:
        if not self._open or self._namespaces[-1] == "html":
            return
        if tag not in _FOREIGN_HTML_BREAKOUT_START_TAGS and not (
            tag == "font" and any(name in attrs for name in {"color", "face", "size"})
        ):
            return
        for i in range(len(self._open) - 1, -1, -1):
            if self._namespaces[i] == "html" or self._html_integration[i]:
                self._pop_to(i + 1)
                return
        self._pop_to(0)

    def _close_implicit(self, tag: str) -> None:
        """The renderer's optional-end-tag recovery (``_MarkdownRenderer._close_implicit``), so both see one tree."""
        if not self._closable_open:
            return
        barriers = _IMPLICIT_CLOSE_BARRIERS.get(tag, frozenset())
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
        del self._namespaces[i:]
        del self._html_integration[i:]

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
        self._record: list | None = None
        # parser positions of buttons that carry their heading title, found by _TitleButtonScan
        self.title_buttons: frozenset[tuple[int, int]] = frozenset()
        self._title_scan: _TitleButtonScan | None = None
        self._site_links = site_links
        self._out: list[str] = []
        self._skip_tags: list[str] = []
        # tag indices in _skip_tags avoid rescanning the stack for end tags
        self._skip_at: dict[str, list[int]] = {}
        self._skip_ns_at: dict[tuple[str, str], list[int]] = {}
        self._skip_namespaces: list[str] = []
        self._skip_html_at: list[int] = []
        self._skip_html_integration: list[bool] = []
        self._form_pointer: bool = False

        self._scope_tags = scope_tags
        self._scope_depth: int = 0

        # one boundary per top-level scope element, so a swarm of tiny cards cannot clear the gate
        self.scope_segments: list[str] = []
        self._scope_seg_start: int | None = None

        # open non-void tags plus hidden-start indices, so an omitted </p>/</li> cannot leave the renderer stuck hidden
        self._open_tags: list[str] = []
        # tag indices in _open_tags keep scoped end-tag recovery at constant time
        self._open_at: dict[str, list[int]] = {}
        # zero avoids scanning when no tags can close implicitly
        self._closable_open: int = 0
        self._hidden_marks: list[int] = []

        # innermost <header> buffers; empty unless strip_header
        self._strip_header = strip_header
        self._header_stack: list[_HeaderFrame] = []
        self.header_decisions: list[bool] = []
        self._header_decisions = header_decisions
        # furniture chars removed, so a candidate is sized as the page wrote it
        self._dropped_chars: int = 0
        self._seg_dropped_start: int = 0
        self.scope_dropped: list[int] = []
        # Track heading text per segment: role="heading", hgroup and linked h1 render as prose ATX misses
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

        # per open <sup>: its _open_tags index, then (output list, start), the (copy, start) pairs that tee the
        # same text, the base text and whether a currency amount precedes it
        self._sup_starts: list[
            tuple[int, tuple[list[str], int, list[tuple[list[str], int]], str, int] | None]
        ] = []

    def _nested_buffer_open(self, frame: _HeaderFrame) -> bool:
        """An enclosing buffer opened before the header must not capture the frame's content."""
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
        # Tee wherever text routes; a link inside the frame arrives twice, so tee only its formatted form
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
        elif self._in_link and self._heading_marks:
            self._link_heading_parts.append(text)
        self._emit_target().append(text)

    def _emit_target(self) -> list[str]:
        frame = self._header_stack[-1] if self._header_stack else None
        if frame is not None and not self._nested_buffer_open(frame):
            return frame.parts
        if self._in_link:
            return self._link_text_parts
        if self._in_cell:
            return self._cell_parts
        if self._in_pre:
            return self._pre_parts
        if self._table_stack and len(self._bq_stack) <= self._table_stack[-1].outer_bq_depth:
            return self._table_stack[-1].parts
        if self._bq_stack:
            return self._bq_stack[-1]
        return self._out

    def _sup_base(self, target: list[str]) -> str:
        """The visible text a <sup> raises (base = its last character), or "" when none: after whitespace or
        sentence punctuation it is a footnote marker (``fact.<sup>1</sup>``,
        ``<a href="#fn1"><sup>1</sup></a>``) or a fraction numerator (``<sup>1</sup>&frasl;``)."""
        for part in itertools.islice(reversed(target), _SUP_BASE_SCAN_PARTS):
            part = part[-_SUP_BASE_SCAN_CHARS:]
            part = _visible_tail(part)
            if part:
                base = part[-1]
                return part if base.isalnum() or base in ")]}|）］｝" else ""
        return ""

    @staticmethod
    def _after_price(target: list[str]) -> int:
        """Digits of the minor unit when *target* ends in a currency amount ($19 -> 2, KWD 19 -> 3), else 0."""
        context = _visible_tail("".join(p[-40:] for p in target[-8:])[-40:])
        context = context.translate(_STRIP_MD_DELIMITERS)
        price = _PRICE_TAIL.search(context)
        # any Unicode currency sign (Sc): $, €, ₺, ₱, ...; or an ISO code: CHF 19
        if price and unicodedata.category(price.group(1)) == "Sc":
            # ₫ ₲ ₩ have no minor unit (¥ is shared with two-decimal CNY, so it stays 2)
            return 0 if price.group(1) in "₫₲₩" else 2
        code = _CODE_PRICE_TAIL.search(context)
        if (
            not code
            or code.group(1) not in _CURRENCY_CODES
            or code.group(1) in _ZERO_DECIMAL_CURRENCIES
        ):
            return 0
        return 3 if code.group(1) in _THREE_DECIMAL_CURRENCIES else 2

    def _sup_copies(self) -> list[tuple[list[str], int]]:
        copies = [self._link_heading_parts, self._seg_heading_texts]
        if self._header_stack:
            copies.append(self._header_stack[-1].heading_parts)
        return [(copy, len(copy)) for copy in copies]

    def _finish_sup(self, opened) -> None:
        if opened is None or opened[0] is not self._emit_target():
            return
        target, start, copies, base, after_price = opened
        joined = "".join(target[start:])
        raw = joined.strip()
        shown = self._site_links.clean(raw) if self._site_links is not None else raw
        visible = shown.strip(_MD_DELIMITERS)
        if (
            not visible
            or "\n" in visible
            or visible[0] == "["
            # $19<sup>.99</sup> / €19<sup>,99</sup> are split cents; 10<sup>.5</sup> with no currency is a power
            or (visible[0] in ".," and after_price)
            or not any(c.isalnum() for c in visible)
            or visible.lower() in _PLAIN_SUFFIXES
            or (
                len(_last_word(base)) > 1
                and _CITATION_LIST.fullmatch(visible)
                and ("–" in visible or not _THOUSANDS.fullmatch(visible))
            )
            or (base[-1] == "2" and visible.lower() in _SECOND_SUFFIXES)
            or (
                (base[-1].isdigit() or _ROMAN_NUMERAL_TAIL.search(base))
                and visible.lower() in _DIGIT_ORDINAL_SUFFIXES
            )
            or visible in _SUPERIOR_ABBREVIATIONS.get(_last_word(base), ())
            # split cents are two digits; $2<sup>n</sup> or USD 10<sup>6</sup> stays an exponent
            or (len(visible) == after_price and visible.isdigit())
        ):
            return
        # only the emphasis wrapping a whole exponent is renderer syntax; an inner * is an operator
        token = shown.strip(_MD_DELIMITERS)
        bare = _BARE_EXPONENT.fullmatch(token) or (
            token.startswith("(") and token.endswith(")") and token.count("(") == 1
        )
        exponent = f"^{raw}" if bare else f"^({raw})"
        exponent += joined[len(joined.rstrip()) :]
        target[start:] = [exponent]
        # headings are teed into these copies; a stale one renders "E=mc2" or skews the prose gate
        for copy, copy_start in copies:
            if "".join(copy[copy_start:]) == joined:
                copy[copy_start:] = [exponent]

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
            # preserve this in the gate so a title-only card is not counted as body prose
            self._seg_heading_texts.append(heading_text)

    def _truncate_open_tags(self, index: int) -> None:
        """drop open tags above *index* while maintaining the closable count."""
        for name in self._open_tags[index:]:
            if positions := self._open_at.get(name):
                positions.pop()
            if name in _IMPLICIT_CLOSERS:
                self._closable_open -= 1
        del self._open_tags[index:]
        # discard <sup> frames whose ancestor closed to avoid filling the depth cap
        while self._sup_starts and self._sup_starts[-1][0] >= index:
            self._sup_starts.pop()

    def _close_implicit(self, tag: str) -> None:
        """Recovery stops at _CLOSE_BARRIERS containers so an outer item hidden content cannot leak; runs even for skipped nav and footer, which also close p"""
        if not self._closable_open:
            return
        barriers = _IMPLICIT_CLOSE_BARRIERS.get(tag, frozenset())
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
        """Closes side buffers opened in the frame whose end tags were omitted; their content must reach the frame before the strip is judged"""
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
        # anchors without usable hrefs render as prose
        if not (self._in_link and self._link_href):
            return
        if self._link_seq == frame.outer_link_seq:
            # enclosing anchors close after the header, too late to credit
            frame.link_chars += chars
        else:
            self._link_header_chars += chars

    @staticmethod
    def _html_start_namespace(tag: str) -> str:
        if tag == "svg":
            return "svg"
        if tag == "math":
            return "math"
        return "html"

    def _skipped_tag_namespace(self, tag: str) -> str:
        if not self._skip_namespaces:
            return self._html_start_namespace(tag)
        parent_namespace = self._skip_namespaces[-1]
        if parent_namespace == "html":
            return self._html_start_namespace(tag)
        if self._skip_html_integration[-1] and not (
            parent_namespace == "math" and tag in {"mglyph", "malignmark"}
        ):
            return self._html_start_namespace(tag)
        return parent_namespace

    @staticmethod
    def _is_html_integration_point(tag: str, namespace: str, attrs: dict) -> bool:
        if namespace == "svg":
            return tag in _SVG_SCOPE_BARRIERS
        if namespace != "math":
            return False
        if tag in _MATHML_TEXT_INTEGRATION_POINTS:
            return True
        encoding = (attrs.get("encoding") or "").strip().lower()
        return tag == "annotation-xml" and encoding in {"text/html", "application/xhtml+xml"}

    def _truncate_skip_tags(self, index: int) -> None:
        for position, name in enumerate(self._skip_tags[index:], start = index):
            self._skip_at[name].pop()
            self._skip_ns_at[(self._skip_namespaces[position], name)].pop()
        while self._skip_html_at and self._skip_html_at[-1] >= index:
            self._skip_html_at.pop()
        del self._skip_tags[index:]
        del self._skip_namespaces[index:]
        del self._skip_html_integration[index:]

    def _clear_skip_tags(self) -> None:
        self._skip_tags.clear()
        self._skip_at.clear()
        self._skip_ns_at.clear()
        self._skip_namespaces.clear()
        self._skip_html_at.clear()
        self._skip_html_integration.clear()

    def _enter_tag(self, tag: str, attr_dict: dict) -> bool:
        """track tag state; run ``_close_implicit`` first so skipped tags trigger recovery."""
        if tag not in _VOID_TAGS:
            if tag in _INDEXED_OPEN_TAGS:
                self._open_at.setdefault(tag, []).append(len(self._open_tags))
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
        """Recovers an open a element before the segment is recorded, or its text is stranded"""
        if self._in_link and self._scope_tags is not None and tag in self._scope_tags:
            self._finish_link()
        suppressed = bool(self._hidden_marks) or (
            self._scope_tags is not None and self._scope_depth == 0
        )
        # Flush the heading here while the tee still recognises it; it emits only after its mark pops
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

    def _latest_indexed_open_tag(
        self,
        names: frozenset[str],
        after: int = -1,
    ) -> tuple[int, str] | None:
        candidates = (
            (positions[-1], name)
            for name in names
            if (positions := self._open_at.get(name)) and positions[-1] > after
        )
        return max(candidates, default = None)

    def _close_open_target(self, target: int) -> None:
        names = self._open_tags[target:]
        for name in reversed(names):
            self._close_rendered_tag(name)

    def _prepare_heading_start(self, tag: str) -> None:
        if tag not in _HEADING_TAGS or not self._open_tags:
            return
        if self._open_tags[-1] not in _HEADING_TAGS:
            return
        self._close_open_target(len(self._open_tags) - 1)

    def _prepare_sibling_table_start(self) -> bool:
        if not self._in_table:
            return False
        if self._skip_tags and self._skipped_tag_namespace("table") != "html":
            return False
        tables = self._open_at.get("table")
        target = tables[-1] if tables else -1
        if target < 0:
            return True
        if self._latest_indexed_open_tag(_TABLE_NESTING_TAGS, target):
            return False
        if not self._open_target_in_scope("table", target):
            return True
        self._clear_skip_tags()
        self._close_open_target(target)
        return False

    def _prepare_foreign_html_start(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if not self._skip_tags or self._skip_namespaces[-1] == "html":
            return
        if tag not in _FOREIGN_HTML_BREAKOUT_START_TAGS and not (
            tag == "font" and any(name in {"color", "face", "size"} for name, _ in attrs)
        ):
            return
        for i in range(len(self._skip_tags) - 1, -1, -1):
            if self._skip_namespaces[i] == "html" or self._skip_html_integration[i]:
                self._truncate_skip_tags(i + 1)
                return
        self._clear_skip_tags()

    def _prepare_table_start(self, tag: str) -> None:
        if not self._in_table or tag not in _TABLE_ONLY_START_TAGS:
            return
        tables = self._open_at.get("table")
        table = tables[-1] if tables else -1
        if caption := self._latest_indexed_open_tag(frozenset({"caption"}), table):
            self._clear_skip_tags()
            self._close_rendered_tag(caption[1])
        if self._in_cell:
            cell = self._latest_indexed_open_tag(_TABLE_CELL_TAGS, table)
            if cell:
                self._clear_skip_tags()
                self._close_rendered_tag(cell[1])
        if self._in_row and tag not in _TABLE_CELL_TAGS:
            row = self._latest_indexed_open_tag(frozenset({"tr"}), table)
            if row:
                self._clear_skip_tags()
                self._close_rendered_tag(row[1])
        if tag in _TABLE_SECTION_START_TAGS:
            section = self._latest_indexed_open_tag(_TABLE_SECTION_TAGS, table)
            if section:
                self._clear_skip_tags()
                self._close_rendered_tag(section[1])

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._record is not None:
            self._record.append((0, (tag, tuple(attrs), *self.getpos())))
        tag = tag.lower()
        if self._title_scan is not None:
            self._title_scan.starttag(tag, attrs, self.getpos())
        title_button = (
            tag == "button" and bool(self.title_buttons) and self.getpos() in self.title_buttons
        )
        if self._skip_tags and tag == "button" and self._skipped_tag_namespace(tag) == "html":
            button = self._last_skip_position("button", "html")
            if button >= 0 and self._skip_target_in_scope(tag, button):
                self._truncate_skip_tags(button)
        self._prepare_foreign_html_start(tag, attrs)
        if tag in _TABLE_ONLY_START_TAGS and not self._in_table:
            return
        if tag == "table" and self._prepare_sibling_table_start():
            return
        if (
            self._skip_tags
            and tag in _TABLE_ONLY_START_TAGS
            and self._last_skip_position("table", "html") < 0
        ):
            self._prepare_table_start(tag)

        if self._skip_tags:
            if tag not in _VOID_TAGS:
                attr_dict = dict(attrs)
                namespace = self._skipped_tag_namespace(tag)
                template = self._last_skip_position("template", "html")
                if tag == "form" and namespace == "html" and template < 0:
                    if self._form_pointer:
                        return
                    self._form_pointer = True
                position = len(self._skip_tags)
                self._skip_at.setdefault(tag, []).append(position)
                self._skip_ns_at.setdefault((namespace, tag), []).append(position)
                self._skip_tags.append(tag)
                self._skip_namespaces.append(namespace)
                if namespace == "html":
                    self._skip_html_at.append(position)
                self._skip_html_integration.append(
                    self._is_html_integration_point(tag, namespace, attr_dict)
                )
            return

        if tag == "form" and self._form_pointer:
            return

        # close optional tags before skipping so skipped <nav>/<footer> releases an open <p> marker
        self._close_implicit(tag)
        self._prepare_heading_start(tag)
        self._prepare_table_start(tag)

        if tag in _SKIP_TAGS and not (title_button and self._heading_marks):
            self._skip_tags.append(tag)
            self._skip_at = {tag: [0]}
            namespace = self._html_start_namespace(tag)
            self._skip_ns_at = {(namespace, tag): [0]}
            self._skip_namespaces = [namespace]
            self._skip_html_at = [0] if namespace == "html" else []
            self._skip_html_integration = [
                self._is_html_integration_point(tag, namespace, dict(attrs))
            ]
            return

        attr_dict = dict(attrs)
        if tag == "form":
            self._form_pointer = True
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

        elif tag == "sup":
            target = self._emit_target()
            reference = (
                any(
                    token in _FOOTNOTE_CLASSES
                    or any(
                        part == "fn" or part.startswith(_FOOTNOTE_CLASS_PREFIXES)
                        for part in _CLASS_PART_SPLIT.split(token)
                    )
                    for token in (attr_dict.get("class") or "").lower().split()
                )
                or "doc-noteref" in (attr_dict.get("role") or "").lower().split()
            )
            # past the cap nothing is tracked, so the stack stays bounded on hostile pages
            if len(self._sup_starts) < _MAX_SUP_DEPTH:
                self._sup_starts.append(
                    (
                        len(self._open_tags) - 1,
                        None
                        if reference or not (base := self._sup_base(target))
                        else (
                            target,
                            len(target),
                            self._sup_copies(),
                            base,
                            self._after_price(target) if base[-1].isdigit() else 0,
                        ),
                    )
                )

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
            # rowspan="0" spans the remaining row group
            self._cell_rowspan = _span_attr(attr_dict, "rowspan") or 65534
            if tag == "th":
                self._row_has_th = True

        elif tag == "img":
            return

    def handle_endtag(self, tag: str) -> None:
        if self._record is not None:
            self._record.append((1, tag))
        tag = tag.lower()
        if self._title_scan is not None:
            self._title_scan.endtag(tag)

        recovered_target = -1
        if self._skip_tags:
            at = self._skip_at.get(tag)
            template = self._last_skip_position("template", "html")
            if tag == "form" and template < 0:
                if at and self._skip_namespaces[at[-1]] == "html":
                    self._form_pointer = False
                elif not (
                    at
                    and self._skip_namespaces[at[-1]] != "html"
                    and (not self._skip_html_at or self._skip_html_at[-1] <= at[-1])
                ):
                    self._form_pointer = False
                    return
            # template is a scope boundary, so end tags outside it are ignored
            if at and (template < 0 or at[-1] >= template):
                i = at[-1]
                if not self._skip_target_in_scope(tag, i):
                    return
                if tag in _GENERIC_END_SKIP_TAGS and self._skip_barrier_after(
                    _SPECIAL_HTML_TAGS, "html", i
                ):
                    return
                self._truncate_skip_tags(i)
                return
            # only scoped structural end tags pop skipped elements; form and inline end tags do not
            if tag not in _SKIP_RECOVERY_END_TAGS:
                return
            if tag in _HEADING_TAGS:
                heading = self._latest_indexed_open_tag(_HEADING_TAGS)
                target = heading[0] if heading else -1
            else:
                positions = self._open_at.get(tag)
                target = positions[-1] if positions else -1
            if target < 0 or not self._open_target_in_scope(tag, target):
                return
            self._clear_skip_tags()
            recovered_target = target
        # kept title buttons close through _exit_tag
        elif tag in _SKIP_TAGS and tag != "button":
            return

        if tag == "form":
            self._form_pointer = False

        if recovered_target >= 0:
            self._close_open_target(recovered_target)
            return
        if tag in _HEADING_TAGS:
            heading = self._latest_indexed_open_tag(_HEADING_TAGS)
            if heading and self._open_target_in_scope(tag, heading[0]):
                self._close_open_target(heading[0])
            return
        self._close_rendered_tag(tag)

    def _close_rendered_tag(self, tag: str) -> None:
        closing_sup = None
        # only the innermost open <sup> converts in O(1); malformed nesting drops on unwind
        if (
            tag == "sup"
            and self._sup_starts
            and self._open_tags
            and self._open_tags[-1] == "sup"
            and self._sup_starts[-1][0] == len(self._open_tags) - 1
        ):
            closing_sup = self._sup_starts.pop()[1]

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

        elif tag == "sup":
            self._finish_sup(closing_sup)

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

    @staticmethod
    def _scope_barriers(tag: str) -> frozenset[str]:
        if tag == "li":
            return _LIST_ITEM_SCOPE_BARRIERS
        if tag == "p":
            return _BUTTON_SCOPE_BARRIERS
        if tag in _TABLE_END_TAGS:
            return _TABLE_SCOPE_BARRIERS
        return _SCOPE_BARRIERS

    def _last_skip_position(self, tag: str, namespace: str) -> int:
        positions = self._skip_ns_at.get((namespace, tag))
        return positions[-1] if positions else -1

    def _skip_barrier_after(self, names: frozenset[str], namespace: str, target: int) -> bool:
        return any(self._last_skip_position(name, namespace) > target for name in names)

    def _foreign_skip_barrier_after(self, target: int) -> bool:
        return self._skip_barrier_after(
            _SVG_SCOPE_BARRIERS, "svg", target
        ) or self._skip_barrier_after(_MATHML_SCOPE_BARRIERS, "math", target)

    def _skip_target_in_scope(self, tag: str, target: int) -> bool:
        # nothing above the innermost element can be a barrier
        if target == len(self._skip_tags) - 1:
            return True
        barriers = self._scope_barriers(tag)
        if self._skip_barrier_after(barriers, "html", target):
            return False
        target_namespace = self._skip_namespaces[target]
        if target_namespace != "html" and (
            not self._skip_html_at or self._skip_html_at[-1] <= target
        ):
            return True
        return not self._foreign_skip_barrier_after(target)

    def _open_skip_barrier(self, barriers: frozenset[str]) -> bool:
        return self._skip_barrier_after(barriers, "html", -1) or self._foreign_skip_barrier_after(
            -1
        )

    def _open_target_in_scope(self, tag: str, target: int) -> bool:
        barriers = self._scope_barriers(tag)
        if self._open_skip_barrier(barriers):
            return False
        return not any(
            positions and positions[-1] > target
            for name in barriers
            if (positions := self._open_at.get(name))
        )

    def _text_suppressed(self) -> bool:
        if self._skip_tags or self._hidden_marks:
            return True
        return self._scope_tags is not None and self._scope_depth == 0

    def handle_data(self, data: str) -> None:
        if self._record is not None:
            self._record.append((2, data))
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
        # Count after collapsing: raw spaces in a link inflated it past the floor.
        self._count_header_text(text)
        # Suppress whitespace-only nodes between table elements (source indentation).
        if self._in_table and not self._in_cell and not text.strip():
            return
        # Source indentation between <li> and its first block must not end the marker line.
        if self._li_marker_pending and not text.strip():
            return
        self._emit(text)

    def handle_entityref(self, name: str) -> None:
        if self._record is not None:
            self._record.append((3, name))
        text = html.unescape(f"&{name};")
        if self._title_scan is not None:
            self._title_scan.data(text)
        if self._text_suppressed():
            return
        self._count_header_text(text)
        self._emit(text)

    def handle_charref(self, name: str) -> None:
        if self._record is not None:
            self._record.append((4, name))
        text = html.unescape(f"&#{name};")
        if self._title_scan is not None:
            self._title_scan.data(text)
        if self._text_suppressed():
            return
        self._count_header_text(text)
        self._emit(text)

    def replay(self, events: list) -> None:
        """Re-run recorded handler calls in order, restoring ``getpos()`` at each start tag."""
        start, end, data = self.handle_starttag, self.handle_endtag, self.handle_data
        for kind, arg in events:
            if kind == 2:
                data(arg)
            elif kind == 0:
                tag, attrs, self.lineno, self.offset = arg
                start(tag, list(attrs))
            elif kind == 1:
                end(arg)
            elif kind == 3:
                self.handle_entityref(arg)
            else:
                self.handle_charref(arg)

    def flush_pending(self) -> None:
        """Flush open side-buffers into ``_out`` after close(), recovering truncated HTML."""
        # Headers first: a frame emits into the enclosing link or cell, which must still be open.
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

        # Flushes a scope left open by truncated HTML after the side-buffers, else the page scores 0
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


# Boilerplate removed from main content; matched only on short lines (GitHub, skip-links, cookies)
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
    """Every sentence segment must be furniture, so prose that merely quotes one phrase is kept."""
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
    """Fenced code is kept verbatim: boilerplate never renders in pre, and READMEs quote error strings."""
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


class _TokenLog:
    """Handler calls of one document, recorded by the first fed pass and replayed by later ones.
    Sound only while the renderer overrides just these five handlers and reads parser state only
    via ``getpos()`` in ``handle_starttag``."""

    __slots__ = ("events", "armed")

    def __init__(self, armed: bool) -> None:
        self.events: list | None = None
        # record only when html_to_markdown knows another pass follows
        self.armed = armed


def _new_renderer(
    source_html: str,
    scope_tags: frozenset[str] | None,
    strip_header: bool,
    site_links: SiteLinks | None = None,
    span_char_limit: int | None = None,
    header_decisions: list[bool] | None = None,
    tokens: _TokenLog | None = None,
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
        if tokens is None:
            renderer.feed(source_html)
            renderer.close()
        elif tokens.events is None:
            renderer._record = [] if tokens.armed else None
            renderer.feed(source_html)
            renderer.close()
            tokens.events, renderer._record = renderer._record, None
        else:
            renderer.replay(tokens.events)
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
    tokens: _TokenLog | None = None,
) -> tuple[int, str, int, int]:
    """return the best eligible score, subtree, count, and visible length; cap header credit at retained prose."""
    renderer = _new_renderer(
        source_html,
        frozenset({tag}),
        strip_header = True,
        site_links = site_links,
        span_char_limit = span_char_limit,
        tokens = tokens,
    )
    # generated span cells must not lift a scope past the content gate or rank; only span pages pay this second pass
    scoring = (
        _new_renderer(
            source_html,
            frozenset({tag}),
            strip_header = True,
            span_char_limit = 0,
            header_decisions = renderer.header_decisions,
            tokens = tokens,
        )
        if renderer._has_generated_spans
        else renderer
    )
    best_len = 0
    best_render = ""
    best_visible = 0
    for i, seg in enumerate(renderer.scope_segments):
        rendered = _strip_boilerplate_lines(_cleanup(seg), site_links)
        scored = (
            rendered
            if scoring is renderer
            else _strip_boilerplate_lines(_cleanup(scoring.scope_segments[i]), site_links)
        )
        if site_links is not None:
            scored = site_links.clean(scored)
        visible = _visible_chars(scored)
        prose = visible - scoring.scope_heading_prose[i]
        if prose < _MIN_MAIN_CONTENT_CHARS:
            continue
        size = len(scored) + min(scoring.scope_dropped[i], len(scored))
        if size > best_len:
            best_len = size
            best_render = rendered
            best_visible = visible
    return (
        best_len,
        best_render,
        sum(1 for seg in renderer.scope_segments if seg.strip()),
        best_visible,
    )


def _render_main_document(
    source_html: str,
    site_links: SiteLinks | None,
    span_char_limit: int,
    tokens: _TokenLog | None = None,
) -> tuple[int, str]:
    renderer = _new_renderer(source_html, None, True, site_links, span_char_limit, tokens = tokens)
    rendered = _strip_boilerplate_lines(_cleanup("".join(renderer._out)), site_links)
    scoring = (
        _new_renderer(
            source_html,
            None,
            True,
            span_char_limit = 0,
            header_decisions = renderer.header_decisions,
            tokens = tokens,
        )
        if renderer._has_generated_spans
        else renderer
    )
    scored = (
        rendered
        if scoring is renderer
        else _strip_boilerplate_lines(_cleanup("".join(scoring._out)), site_links)
    )
    if site_links is not None:
        scored = site_links.clean(scored)
    return _visible_chars(scored), rendered


def _visible_chars(text: str) -> int:
    """count visible nonblank characters without link targets; callers subtract tracked heading prose."""
    return sum(_visible_len(line) for line in text.split("\n") if line.strip())


_BACKTICK_RUN = re.compile(r"`+")


def _fence_for(raw: str) -> str:
    """Fence long enough to survive backticks in *raw*, as CommonMark requires:
    a literal ``` line inside the block would otherwise close it early."""
    if "`" not in raw:
        return "```"
    return "`" * max(3, max(map(len, _BACKTICK_RUN.findall(raw))) + 1)


def _visible_len(line: str) -> int:
    """Length of *line* without Markdown link destinations. A tracking URL is not
    prose: a card holding one [Read](/x?<300 bytes>) otherwise scored 339 visible
    characters off 4 and displaced the article. Linear: once a destination fails to
    close, the rest are answered from one table instead of a rescan each."""
    if "[" not in line and "\\" not in line:
        return len(line)
    total, i, n = 0, 0, len(line)
    open_bracket = False
    closes = None
    while i < n:
        if line[i] == "\\":
            total += 2
            i += 2
            continue
        if line[i] == "[":
            open_bracket = True
        # With no opening bracket, "](" is literal and the parens after it are prose.
        if open_bracket and line[i] == "]" and i + 1 < n and line[i + 1] == "(":
            if closes is None:
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
                    # Unclosed: later candidates would rescan this tail, so index it once.
                    closes = _destination_ends(line)
            else:
                j = closes.get(i + 2, -1)
                depth = j < 0
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


def _destination_ends(line: str) -> dict:
    """End of the destination scan from each reachable position, for scans that close. All scans
    step over the same positions, so a scan from p closes where the paren-depth prefix sum first
    drops below its value at p."""
    steps: list[int] = []
    sums: list[int] = []
    depth, k, n = 0, 0, len(line)
    while k < n:
        steps.append(k)
        sums.append(depth)
        char = line[k]
        if char == "\\":
            k += 2
            continue
        depth += (char == "(") - (char == ")")
        k += 1
    ends: dict = {}
    first_at: dict = {}  # prefix sum -> earliest later step index
    sums.append(depth)
    for m in range(len(steps), -1, -1):
        if m < len(steps):
            after = first_at.get(sums[m] - 1)
            if after is not None:
                ends[steps[m]] = steps[after - 1] + 1
        first_at[sums[m]] = m
    return ends


# Scoped conversions shorter than this are rejected as main content, e.g. an empty article stub
_MIN_MAIN_CONTENT_CHARS = 200
# absent text = no such tag can tokenize, so that scope finds nothing and its pass is skipped
_ARTICLE_START = re.compile(r"<article", re.IGNORECASE)
_MAIN_START = re.compile(r"<main", re.IGNORECASE)
# absent text = no generated span cells, so a pass needs no span-free scoring re-render
_SPAN_ATTR = re.compile(r"(?:col|row)span", re.IGNORECASE)


def html_to_markdown(
    source_html: str,
    *,
    main_content: bool = False,
    site_links: SiteLinks | None = None,
    max_span_chars: int | None = None,
) -> str:
    """convert HTML to Markdown; main_content prefers a dominant article or main subtree, site_links records same-site links without changing output, and max_span_chars limits generated table cells."""
    source_html = source_html.replace("\r\n", "\n").replace("\r", "\n")
    span_limit = 2 * len(source_html)
    if max_span_chars is not None:
        span_limit = min(span_limit, max_span_chars)
    rendered = ""
    full_rendered = ""
    if main_content:
        has_article = _ARTICLE_START.search(source_html) is not None
        # without <article> the first pass may be followed by a title rescan or the whole-page pass
        tokens = _TokenLog(armed = not has_article)
        if has_article:
            length, rendered, articles, article_visible = _select_main_scope_render(
                source_html, "article", site_links, span_limit, tokens
            )
        else:
            length, rendered, articles, article_visible = 0, "", 0, 0
        if length < _MIN_MAIN_CONTENT_CHARS or articles > 1:
            if _MAIN_START.search(source_html) is not None:
                main_length, main_rendered, _, main_visible = _select_main_scope_render(
                    source_html, "main", site_links, span_limit, tokens
                )
            else:
                main_length, main_rendered, main_visible = 0, "", 0
                # the whole-page pass follows and re-renders span pages to score them
                tokens.armed = tokens.armed or _SPAN_ATTR.search(source_html) is not None
            if articles > 2 and main_length < _MIN_MAIN_CONTENT_CHARS:
                main_length, full_rendered = _render_main_document(
                    source_html, site_links, span_limit, tokens
                )
                main_rendered = full_rendered
                main_visible = main_length
            if main_length >= _MIN_MAIN_CONTENT_CHARS and (
                length < _MIN_MAIN_CONTENT_CHARS or main_visible > 2 * article_visible
            ):
                length, rendered = main_length, main_rendered
        if length < _MIN_MAIN_CONTENT_CHARS:
            if not full_rendered:
                _, full_rendered = _render_main_document(
                    source_html, site_links, span_limit, tokens
                )
            rendered = full_rendered
    else:
        rendered = _render(source_html, None, site_links = site_links, span_char_limit = span_limit)
    return site_links.finish(rendered) if site_links is not None else rendered
