# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Rewriting Deep Research report citations to the sources actually gathered.

A local model invents URLs, renumbers references, and appends its own source list. Every
citation is canonicalized against the run's catalogs, and anything that does not resolve to a
gathered web source or document chunk is stripped rather than shown as supported.
"""

from __future__ import annotations

import re

from markdown_it import MarkdownIt
from markdown_it.rules_inline.backticks import backtick

from core.research.redaction import _escape_link_destination


# Unrolled rather than (?:[^\[\]]+|\[[^\[\]]*\])* : that alternation backtracks catastrophically on
# an unterminated "[Document:", and this runs on the event loop.
_DOCUMENT_CITATION = re.compile(r"\[Document:[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]")
_MARKDOWN_LINK_START = re.compile(r"\[([^\]\n]+)\]\((https?://)")
_SOURCES_HEADING = re.compile(
    r"^(?:#{1,6}\s+|\*\*)?"
    r"(?:Sources?|References?|Bibliography|Works\s+Cited|Source\s+List)"
    r"(?:\*\*)?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_NUMBERED_CITATION = re.compile(r"(?<!\^)\[(\d+)]")
_AUTOLINK = re.compile(r"<(https?://[^>\s]+)>")
# \x00 stops a URL glued to masked code from swallowing its placeholder.
_RAW_URL = re.compile(r"https?://[^\s<>\x00]+")
# Beside the pattern that restores them: a kind missing there leaves a raw sentinel in the
# report. [0-9] not \d, which also matches other scripts' digits.
_PLACEHOLDER_KINDS = ("research-code", "research-citation")
_PLACEHOLDER = re.compile(rf"\x00(?:{'|'.join(_PLACEHOLDER_KINDS)})-[0-9]+\x00")
# For a pass that deletes prose but must carry the code through.
_CODE_PLACEHOLDER = re.compile(r"\x00research-code-[0-9]+\x00")
# remark-gfm renders the indented lines under a footnote definition as prose, where a bare URL
# becomes a live link; the CommonMark parse below calls them code. Validate rather than mask.
_FOOTNOTE_DEFINITION = re.compile(r" {0,3}\[\^[^\]\s]+\]:")
# A mermaid fence is not shown as code, it is executed into a diagram, and mermaid's image shape
# (`A@{ img: "..." }`, mermaid >= 11.3) puts its URL straight on an SVG image the browser then
# fetches, outside the markdown image pipeline that would have vetted it. So the URLs inside one
# stay subject to the catalog, exactly as before code was masked at all. Case-insensitive while
# the renderer's own gate is not: over-matching costs a validated diagram, under-matching leaves
# a report able to name any URL it likes.
_MERMAID_FENCE = re.compile(r"mermaid\b", re.IGNORECASE)


def _citation_title(source: dict, fallback: str) -> str:
    """Title as it may appear in a markdown link label.

    The prompt tells the model to copy titles verbatim from the source catalog, and search
    titles routinely carry a bracket ("[PDF] Annual Report") which makes the citation
    unmatchable, so the catalog and the citation writer strip them the same way.
    """
    title = str(source.get("title") or fallback).replace("[", "").replace("]", "").strip()
    return title or fallback


def _trim_url_tail(raw: str) -> str:
    """Strip trailing prose punctuation that ``_RAW_URL`` swallowed.

    Mirrors GFM extended autolink path validation: walk right to left, dropping
    ``.,;:!?`` and any ``)`` that has no matching ``(`` inside the URL, stopping at the
    first character that is neither. Both rules must run in one interleaved pass, else
    ``https://x/y.)`` keeps a stray dot. Without this, ``(https://x/y)`` never matches
    the catalog and the citation is dropped from the report.
    """
    end = len(raw)
    opening, closing = raw.count("("), raw.count(")")
    while end:
        char = raw[end - 1]
        if char == ")":
            if closing <= opening:
                break
            closing -= 1
        elif char not in ".,;:!?":
            break
        end -= 1
    return raw[:end]


def _placeholder(kind: str, index: int) -> str:
    """Sentinel for text a validator must move but not rewrite.

    NUL delimited: ``_mask_code`` normalizes away any the model wrote, so one cannot reach a
    report and be mistaken for a token.
    """
    return f"\x00{kind}-{index}\x00"


def _record_code_span(state, silent: bool) -> bool:
    start = state.pos
    count = len(state.tokens)
    matched = backtick(state, silent)
    if (
        matched
        and not silent
        # The block's own source only: the image rule re-enters with alt text, where these
        # offsets address the wrong string. get() records nothing rather than raising.
        and state.src is state.env.get("code_source")
        and len(state.tokens) > count
        and state.tokens[-1].type == "code_inline"
    ):
        state.env["code_spans"].append((start, state.pos))
    return matched


# Block maps keep the original lines, indentation and container markers included, so inline
# source is parsed separately to keep code offsets on those same lines.
# Tables enabled to match the renderer: otherwise a row is one paragraph line and backticks in
# two cells pair into a span across the boundary, unvalidating text that still renders as text.
_CODE_MARKDOWN = MarkdownIt("commonmark").enable("table").disable("inline")
_CODE_MARKDOWN.inline.ruler.at("backticks", _record_code_span)


def _footnote_content_lines(lines: list[str]) -> set[int]:
    """Line numbers the renderer reads as footnote content, whatever this parser calls them.

    Loose on purpose: over-collecting only costs a masked block, which errs towards validating.
    """
    covered = set()
    inside = False
    for number, line in enumerate(lines):
        if _FOOTNOTE_DEFINITION.match(line):
            inside = True
        elif not inside:
            continue
        elif not line.strip():
            covered.add(number)
        elif line.startswith(("    ", "\t")):
            covered.add(number)
        else:
            inside = False
    return covered


def _mask_code(text: str, placeholders: dict[str, str]) -> str:
    # CommonMark maps NUL to U+FFFD, so the renderer already shows it that way, and doing it
    # first stops a report spelling a token and being handed another region's text. One code
    # point either way, so the offsets below are unaffected.
    text = text.replace("\x00", "\ufffd")
    offsets = [0, *(match.end() for match in re.finditer(r"\r\n?|\n", text))]
    if offsets[-1] != len(text):
        offsets.append(len(text))
    lines = [text[offsets[number] : offsets[number + 1]] for number in range(len(offsets) - 1)]
    footnote_content = _footnote_content_lines(lines)
    spans = []

    def record_inline(start: int, end: int) -> None:
        source = text[start:end]
        env = {"code_source": source, "code_spans": []}
        _CODE_MARKDOWN.inline.parse(source, _CODE_MARKDOWN, env, [])
        spans.extend((start + first, start + last) for first, last in env["code_spans"])

    def record_row(start: int, end: int) -> None:
        """Cell by cell, as the renderer splits a row before parsing inline.

        Every cell token carries the whole row's map, so cutting at unescaped pipes is what
        keeps a span inside one cell.
        """
        cell = start
        escaped = False
        for index in range(start, end):
            if escaped:
                escaped = False
            elif text[index] == "\\":
                escaped = True
            elif text[index] == "|":
                record_inline(cell, index)
                cell = index + 1
        record_inline(cell, end)

    in_table = False
    for token in _CODE_MARKDOWN.parse(text):
        if token.type == "table_open":
            in_table = True
        elif token.type == "table_close":
            in_table = False
        if token.map is None:
            continue
        # Mask only what the renderer shows as code: an indented block under a footnote
        # definition is prose there, and a mermaid fence is an executed diagram.
        if token.type == "code_block" and token.map[0] in footnote_content:
            continue
        if token.type == "fence" and _MERMAID_FENCE.match((token.info or "").strip()):
            continue
        start, end = (offsets[line] for line in token.map)
        if token.type in {"fence", "code_block"}:
            # Keep the following heading on its own line while the block is masked.
            end = start + len(text[start:end].rstrip("\r\n"))
            spans.append((start, end))
        elif token.type == "tr_open":
            record_row(start, end)
        elif token.type == "inline" and not in_table:
            record_inline(start, end)
    pieces = []
    cursor = 0
    for start, end in sorted(spans):
        pieces.append(text[cursor:start])
        token = _placeholder("research-code", len(placeholders))
        placeholders[token] = text[start:end]
        pieces.append(token)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _restore_placeholders(text: str, placeholders: dict[str, str]) -> str:
    """Substitute every token in one pass.

    A replace() per token rescans the report once per span, costing O(len(report) x spans); one
    sub() is linear, and never rescans restored code for a later token.
    """
    if not placeholders:
        return text

    def restore(match: re.Match) -> str:
        token = match.group(0)
        value = placeholders.get(token)
        if value is not None:
            return value
        # Unreachable while _mask_code normalizes NUL away. If a later path ever skips that,
        # drop the delimiters rather than deliver a NUL.
        return token.replace("\x00", "\ufffd")

    return _PLACEHOLDER.sub(restore, text)


def _validate_masked_sources(report: str, sources: list[dict], placeholders: dict[str, str]) -> str:
    source_by_url = {
        str(source.get("url") or ""): source for source in sources if source.get("url")
    }
    source_urls = list(source_by_url)

    heading = _SOURCES_HEADING.search(report)
    if heading:
        report = report[: heading.start()]

    def citation(url: str) -> str | None:
        source = source_by_url.get(url)
        if source is None:
            return None
        title = _citation_title(source, url)
        token = _placeholder("research-citation", len(placeholders))
        placeholders[token] = f"[{title}]({_escape_link_destination(url)})"
        return token

    def replace_markdown_links(text: str) -> str:
        pieces = []
        cursor = 0
        while match := _MARKDOWN_LINK_START.search(text, cursor):
            destination_start = match.start(2)
            index = match.end(2)
            depth = 0
            escaped = False
            close = None
            destination_end = None
            while index < len(text):
                character = text[index]
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character.isspace():
                    if depth != 0:
                        break
                    destination_end = index
                    title_start = index
                    while title_start < len(text) and text[title_start].isspace():
                        title_start += 1
                    if title_start < len(text) and text[title_start] in {'"', "'"}:
                        quote = text[title_start]
                        title_end = title_start + 1
                        title_escaped = False
                        while title_end < len(text):
                            if title_escaped:
                                title_escaped = False
                            elif text[title_end] == "\\":
                                title_escaped = True
                            elif text[title_end] == quote:
                                break
                            title_end += 1
                        if title_end >= len(text):
                            break
                        title_start = title_end + 1
                        while title_start < len(text) and text[title_start].isspace():
                            title_start += 1
                    if title_start < len(text) and text[title_start] == ")":
                        close = title_start
                    break
                elif character == "(":
                    depth += 1
                elif character == ")":
                    if depth == 0:
                        close = index
                        destination_end = index
                        break
                    depth -= 1
                index += 1
            if close is None:
                pieces.append(text[cursor : match.start()])
                pieces.append(match.group(1).strip())
                cursor = index
                continue
            url = text[destination_start:destination_end].replace(r"\(", "(").replace(r"\)", ")")
            pieces.append(text[cursor : match.start()])
            pieces.append(citation(url) or match.group(1).strip())
            cursor = close + 1
        pieces.append(text[cursor:])
        return "".join(pieces)

    def replace_number(match: re.Match) -> str:
        index = int(match.group(1)) - 1
        if 0 <= index < len(source_urls):
            return citation(source_urls[index]) or match.group(0)
        return match.group(0)

    def replace_autolink(match: re.Match) -> str:
        return citation(match.group(1)) or match.group(1)

    def replace_raw_url(match: re.Match) -> str:
        # Cite whole source URLs; drop other raw URLs. Whole-match avoids prefix collisions.
        raw = match.group(0)
        core = _trim_url_tail(raw)
        if core in source_by_url:
            return (citation(core) or core) + raw[len(core) :]
        # Keep the trimmed tail so dropping the URL cannot unbalance the prose.
        return raw[len(core) :]

    validated = replace_markdown_links(report)
    validated = _AUTOLINK.sub(replace_autolink, validated)
    validated = _NUMBERED_CITATION.sub(replace_number, validated)
    validated = _RAW_URL.sub(replace_raw_url, validated)
    return validated.strip()


def _validate_report_sources(report: str, sources: list[dict]) -> str:
    """Canonicalize citations and remove model-authored source lists."""
    placeholders: dict[str, str] = {}
    validated = _validate_masked_sources(_mask_code(report, placeholders), sources, placeholders)
    return _restore_placeholders(validated, placeholders)


def _document_source_citation(source: dict) -> str:
    filename = str(source.get("filename") or "Document")
    if source.get("page") is not None:
        return f"[Document: {filename}, p. {source['page']}]"
    return f"[Document: {filename}]"


def _allowed_document_citations(sources: list[dict]) -> set[str]:
    allowed = set()
    for source in sources:
        filename = str(source.get("filename") or "Document")
        allowed.add(f"[Document: {filename}]")
        allowed.add(_document_source_citation(source))
    return allowed


def _validate_masked_document_sources(
    report: str, sources: list[dict], placeholders: dict[str, str]
) -> str:
    allowed = _allowed_document_citations(sources)

    def keep_if_allowed(match: re.Match) -> str:
        # Judge it as the model wrote it: the pattern already spans a "]" inside a filename
        # ("budget [final].pdf"), and backticks in one were masked before this pass ran.
        citation = _restore_placeholders(match.group(0), placeholders)
        if citation in allowed:
            return match.group(0)
        # An unsupported citation can reach across code ("[Document: `cmd` ]"); dropping it
        # must not take the code with it.
        return "".join(_CODE_PLACEHOLDER.findall(match.group(0)))

    return _DOCUMENT_CITATION.sub(keep_if_allowed, report)


def _validate_report_document_sources(report: str, sources: list[dict]) -> str:
    placeholders: dict[str, str] = {}
    validated = _validate_masked_document_sources(
        _mask_code(report, placeholders), sources, placeholders
    )
    return _restore_placeholders(validated, placeholders)


def _validate_report(report: str, sources: list[dict], document_sources: list[dict]) -> str:
    # Mask the draft once: removing a URL-only line can split a paragraph, and reparsing
    # would then read an indented citation as code.
    placeholders: dict[str, str] = {}
    validated = _validate_masked_sources(_mask_code(report, placeholders), sources, placeholders)
    validated = _validate_masked_document_sources(validated, document_sources, placeholders)
    return _restore_placeholders(validated, placeholders)
