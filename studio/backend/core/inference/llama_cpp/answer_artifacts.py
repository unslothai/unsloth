# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Detecting unclosed code fences, markup and answer artifacts in model output."""

import re

from typing import Optional

# Plan-without-action re-prompt state now lives in tool_call_parser (imported above).

# An artifact is content the model could not have written as a plan; without
# one, "First, let me set up pygame. ```python ... ```" satisfies the intent
# gate and the synthetic STOP turn wipes the code.
#
# A numbered list is deliberately NOT one: over the 300 recorded answers in
# tests/data/plan_vs_answer a list branch decided one turn while costing errors
# in both directions.
#
# `\s*` in the closing tags is spec-legal HTML. Every `[\s\S]{...}?` run stays
# length-bounded or the search backtracks on CRLF and `<html>` spam.
_CLOSED_CODE_FENCE = re.compile(
    r"(?<!`)(?P<bf>`{3,})(?!`)[^\r\n]{0,600}\r?\n[\s\S]{1,4000}?\r?\n[ \t>]*(?P=bf)`*[ \t]*(?:\r?\n|\Z)"
    r"|(?<!~)(?P<tf>~{3,})(?!~)[^\r\n]{0,600}\r?\n[\s\S]{1,4000}?\r?\n[ \t>]*(?P=tf)~*[ \t]*(?:\r?\n|\Z)",
    re.IGNORECASE,
)
_CLOSED_MARKUP_ARTIFACT = re.compile(
    r"(?:<!doctype\b[\s\S]{0,200}?)?<html\b[^>]{0,600}>[^<]{0,400}<[a-zA-Z!/][\s\S]{0,4000}?</html\s*>"
    r"|<svg\b[^>]{0,600}>[^<]{0,400}<[a-zA-Z!/][\s\S]{0,4000}?</svg\s*>",
    re.IGNORECASE,
)
_HAS_ANSWER_ARTIFACT = re.compile(
    # Backtick then tilde fence (models emit ~~~ when the body holds backticks).
    # CommonMark takes 3+ to open and as many to close, on a cleanly ended line, so
    # ``` ```not actually closed ``` does not count.
    r"(?<!`)(?P<bf>`{3,})(?!`)[^\r\n]{0,600}\r?\n[\s\S]{1,4000}?\r?\n[ \t>]*(?P=bf)`*[ \t]*(?:\r?\n|\Z)"
    r"|(?<!~)(?P<tf>~{3,})(?!~)[^\r\n]{0,600}\r?\n[\s\S]{1,4000}?\r?\n[ \t>]*(?P=tf)~*[ \t]*(?:\r?\n|\Z)"
    r"|(?:<!doctype\b[\s\S]{0,200}?)?<html\b[^>]{0,600}>[^<]{0,400}<[a-zA-Z!/][\s\S]{0,4000}?</html\s*>"
    r"|<svg\b[^>]{0,600}>[^<]{0,400}<[a-zA-Z!/][\s\S]{0,4000}?</svg\s*>",
    re.IGNORECASE,
)

_FENCE_RUN_RE = re.compile(r"(?<!`)(?P<backticks>`{3,})(?!`)|(?<!~)(?P<tildes>~{3,})(?!~)")
# Structure, not content: stripped so a quoted fence is judged at its real column.
# One list marker may precede the quote, since "- > ```py" is a quote in a list item.
_BLOCKQUOTE_PREFIX = re.compile(r"^[ \t]*(?:[-*+]|\d{1,9}[.)])?[ \t]*(?:>[ \t]?)+")
# A language token, the only trailing text that makes an inline run an opener
# rather than prose: python3, c++, c#, objective-c, ts-node, bash-session. It may
# contain a dot (asp.net) but never ends in one, which is how "here." stays a
# sentence rather than an info string.
_FENCE_INFO_STRING_RE = re.compile(r"[A-Za-z][\w+#-]*(?:\.[\w+#-]+)*")
# A fence on a list-marker line is block level, so the prose rules below do not apply.
_LIST_MARKER_ONLY = re.compile(r"^[ \t]*(?:[-*+]|\d{1,9}[.)])[ \t]+$")
_ONE_QUOTE_MARKER = re.compile(r"[ \t]*>[ \t]?")


def _has_unclosed_code_fence(text: str) -> bool:
    """True if ``text`` contains a code fence whose closer is missing.

    Models do open a fence mid-line (``First. \\`\\`\\`python``), so an inline
    run counts, but only when it cannot be read as prose instead. A column-0
    fence always counts, blockquoted or not.

    This helper, not `_HAS_ANSWER_ARTIFACT`, is what pairs a quoted closer with
    a quoted opener: the pattern accepts `>` on any closer so a quoted block is
    found at all, and the depth check below rejects the ones that do not match.
    """
    active_char: Optional[str] = None
    active_len = 0
    active_quote = 0
    active_base = 0
    closed_any = False
    for raw_line in text.splitlines():
        # A quote marker four columns in is indented code, not a container. List
        # markers are left alone: at that depth they may continue a list not visible
        # from one line.
        lead = raw_line[: len(raw_line) - len(raw_line.lstrip(" \t"))]
        quote = _BLOCKQUOTE_PREFIX.match(raw_line)
        indented_quote = quote is not None and len(lead.expandtabs(4)) > 3
        if indented_quote:
            quote = None
        quote_depth = quote.group(0).count(">") if quote else 0
        line = raw_line[quote.end() :] if quote else raw_line
        runs = list(_FENCE_RUN_RE.finditer(line))
        for index, m in enumerate(runs):
            fence = m.group("backticks") or m.group("tildes")
            trailing = line[m.end() :].strip()
            ch = fence[0]
            prefix = line[: m.start()]
            indent = len(prefix.expandtabs(4))
            blank_prefix = not prefix.strip()
            if active_char is not None:
                # A closer starts and ends its own line, within 3 columns of the
                # CONTAINER, not of whatever indentation the opener chose. Anything
                # else, "Use three backticks: ```" included, is body.
                if (
                    blank_prefix
                    and active_base <= indent <= active_base + 3
                    and ch == active_char
                    and len(fence) >= active_len
                    and quote_depth == active_quote
                    and not trailing
                ):
                    active_char, active_len, active_quote, active_base = None, 0, 0, 0
                    closed_any = True
                continue
            # A later BACKTICK run closes an inline span ("```python``` is the
            # syntax"), at column zero or not. A different delimiter is info-string
            # text (```markdown title=~~~), and a tilde info string may itself hold
            # tildes (~~~markdown title=~~~~), which CommonMark allows.
            if ch == "`" and any(
                (later.group("backticks") or later.group("tildes"))[0] == ch
                for later in runs[index + 1 :]
            ):
                continue
            if blank_prefix:
                # Its own indentation, so the baseline is 0. Past three columns it
                # is an indented code line and opens nothing.
                if indent <= 3:
                    active_char, active_len = ch, len(fence)
                    active_quote, active_base = quote_depth, 0
                continue
            # A list marker is a container, so its fence is block level and skips the
            # rest. CommonMark has no mid-PROSE fence, but models open one, and only a
            # bare info string tells that opener from prose. Once something HAS
            # closed, an inline run is a mention ("wrap it in ```", "the marker is
            # ```python"); a real second block starts at column 0 and is unaffected.
            in_list = _LIST_MARKER_ONLY.match(prefix) is not None
            if not in_list:
                if indented_quote:
                    continue
                if trailing and not _FENCE_INFO_STRING_RE.fullmatch(trailing):
                    continue
                if closed_any:
                    continue
            active_char, active_len, active_quote = ch, len(fence), quote_depth
            # A list marker IS the container, so its width is the baseline. Mid-sentence
            # there is no container, only how far the sentence got, so 0.
            active_base = indent if in_list else 0
    return active_char is not None


def _has_unclosed_markup_block(text: str) -> bool:
    """True if ``text`` opens an <html>/<svg> block without closing it.

    Counted rather than matched, so a closed block followed by a still-open one
    also qualifies.
    """
    opens_html = len(re.findall(r"<html\b", text, re.IGNORECASE))
    closes_html = len(re.findall(r"</html\s*>", text, re.IGNORECASE))
    if opens_html > closes_html:
        return True
    opens_svg = len(re.findall(r"<svg\b", text, re.IGNORECASE))
    closes_svg = len(re.findall(r"</svg\s*>", text, re.IGNORECASE))
    return opens_svg > closes_svg


# "First, I'll create an <html></html> skeleton" is a plan, not a page.
_EMPTY_MARKUP_SKELETON = re.compile(
    r"<(html|svg)\b[^>]*>\s*</\1\s*>",
    re.IGNORECASE,
)
_DOCTYPE_PREFIX = re.compile(
    r"^<!doctype\b[\s\S]{0,200}?>",
    re.IGNORECASE,
)


def _is_empty_markup_skeleton(matched: str) -> bool:
    """True if ``matched`` is an empty <html></html> / <svg></svg>, doctype
    prefix and surrounding whitespace allowed."""
    candidate = _DOCTYPE_PREFIX.sub("", matched.strip(), count = 1).strip()
    return _EMPTY_MARKUP_SKELETON.fullmatch(candidate) is not None


def _is_blank_fence(matched: str, depth: int = 0) -> bool:
    """True if ``matched`` is a fence whose body is only whitespace.

    The delimiters are on the first and last lines, so what sits between them is
    the answer, and a block holding a single space is no more one than an empty
    `<html></html>` is a page."""
    if matched[:1] not in ("`", "~"):
        return False
    lines = matched.splitlines()
    # ``depth`` is the OPENER's quote depth, which the caller reads from the line the
    # match starts on. A closer sits at that same depth, so a deeper last line means
    # the pattern stopped on a nested delimiter that is really body text, and the
    # block runs past it: content, not a blank fence.
    closing = _BLOCKQUOTE_PREFIX.match(lines[-1]) if lines else None
    if (closing.group(0).count(">") if closing else 0) != depth:
        return False
    for line in lines[1:-1]:
        for _ in range(depth):
            marker = _ONE_QUOTE_MARKER.match(line)
            if marker is None:
                break
            line = line[marker.end() :]
        if line.strip():
            return False
    return True


def _first_real_artifact(text: str):
    """First _HAS_ANSWER_ARTIFACT match with something in it.

    Every match is inspected, so a skeleton followed by a real page counts."""
    for m in _HAS_ANSWER_ARTIFACT.finditer(text):
        if _is_empty_markup_skeleton(m.group(0)):
            continue
        # The container the fence sits in is whatever quotes the line it opens on.
        line_start = text.rfind("\n", 0, m.start()) + 1
        quote = _BLOCKQUOTE_PREFIX.match(text[line_start : m.start()])
        depth = quote.group(0).count(">") if quote else 0
        if _is_blank_fence(m.group(0), depth):
            continue
        return m
    return None


def _strip_markup_outside_fences(text: str) -> str:
    """Drop complete <html>/<svg> blocks, except ones that BEGIN inside a fence.

    An opening tag inside a code example is that example's content, and pairing it
    with a closing tag in the prose after the fence swallowed the fence's own closer:
    ``` ```html\\n<html>\\n``` \\nClose it with </html>.``` then read as unfinished. A
    block that merely encloses a fence is still a real block and goes whole, so a page
    holding a Markdown example is not left in pieces.
    """
    fences = [m.span() for m in _CLOSED_CODE_FENCE.finditer(text)]

    def _keep_examples(m):
        # Either endpoint inside a fence makes the match that example's content:
        # prose may open a tag the fenced fragment goes on to close.
        return m.group(0) if any(s <= m.start() < e or s < m.end() <= e for s, e in fences) else ""

    return _CLOSED_MARKUP_ARTIFACT.sub(_keep_examples, text)


# A reasoning opener ends at a tag boundary: "<think-card>" is an element of the
# answer, not a thought.
_THINK_OPEN_RE = re.compile(r"<think[\s>]")
_THINKING_OPEN_RE = re.compile(r"<thinking[\s>]")
_BRACKET_THINK_OPEN_RE = re.compile(r"\[THINK\]")


def _artifact_spans(text: str) -> "list[tuple[int, int]]":
    """Spans of the complete fences and complete markup in ``text``."""
    return [m.span() for m in _CLOSED_CODE_FENCE.finditer(text)] + [
        m.span() for m in _CLOSED_MARKUP_ARTIFACT.finditer(text)
    ]


def _find_outside_artifacts(
    text: str,
    needle: str,
    spans = None,
) -> int:
    """First index of ``needle`` that is not inside a complete artifact, or -1.

    A reasoning marker shown in a code example is the example, not reasoning.

    The scan comes first: most answers carry no reasoning marker at all, and
    mapping the artifacts is two regex passes over the whole response.
    """
    at = text.find(needle)
    if at < 0:
        return -1
    if spans is None:
        spans = _artifact_spans(text)
    while at >= 0:
        if not any(start <= at < end for start, end in spans):
            return at
        at = text.find(needle, at + 1)
    return -1


def _text_outside_think(text: str) -> str:
    """``text`` with a LEADING reasoning block dropped, leaving what the user was told.

    Only a leading block is reasoning. The loop folds `reasoning_content` in as one
    prefix per turn, so that is the only provenance genuine reasoning has, and a
    `<think>` further in is the model quoting the tag in an example; the Anthropic
    path reads it the same way (`routes/inference.py` `_split_think_segments`). A
    prefilled template sends the opener itself, so the block may arrive carrying
    only its closer.
    """
    stripped = text.lstrip()
    pairs = (
        (_THINKING_OPEN_RE, "<thinking", "</thinking>"),
        (_THINK_OPEN_RE, "<think", "</think>"),
        (_BRACKET_THINK_OPEN_RE, "[THINK]", "[/THINK]"),
    )
    # Whichever opener actually starts the turn owns it, so settle that before
    # looking at bare closers: a leading block may name another marker in its body.
    for lead, _opener, closer in pairs:
        if lead.match(stripped):
            # First closer, plainly. `routes/inference.py` `_split_think_segments`
            # reads a leading block the same way and solves a closer quoted inside
            # the trace with the generator's recorded length, not by guessing which
            # markup is an example: every such guess here let a span run from the
            # thought into the answer and swallow the real boundary.
            close = text.find(closer)
            # No closer: a thought the window cut off runs to the end, and none of
            # it was shown.
            return text[close + len(closer) :] if close >= 0 else ""
    # No leading opener, so a closer with none before it is a prefilled template's:
    # it emits the opening marker itself and only the closer is generated.
    spans = None
    for lead, _opener, closer in pairs:
        if closer in text and spans is None:
            spans = _artifact_spans(text)
        close = _find_outside_artifacts(text, closer, spans)
        # Boundary-aware, so "<think-card>" named in the trace is not read as an
        # opener that would leave the prefilled block unstripped.
        if close >= 0 and not lead.search(text[:close]):
            return text[close + len(closer) :]
    return text


def _has_answer_artifact(text: str) -> bool:
    """True if ``text`` looks like a completed answer artifact.

    A closed code fence, a complete HTML page, or a complete SVG, and none of them
    left unfinished. Empty skeletons do not count.
    """
    # Strip closed artifacts first: a `html = '<html>'` literal in a finished
    # snippet, or backticks inside finished HTML, are content, not open state.
    if _has_unclosed_code_fence(_strip_markup_outside_fences(text)):
        return False
    # A page the artifact is nested in is deliberately NOT inspected: separating an
    # unfinished enclosing page from a tag named in prose needs an HTML parser, and two
    # attempts here each went on to reject a finished answer. Missing the nudge leaves
    # the block on screen, which is the cheaper mistake.
    return _first_real_artifact(text) is not None
