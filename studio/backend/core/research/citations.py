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


# unrolling avoids catastrophic backtracking on unclosed [Document: that would block the event loop.
_DOCUMENT_CITATION = re.compile(r"\[(?:Document|MCP):[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]")
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
# Must match the restore pattern. [0-9] not \d, which matches other scripts' digits.
_PLACEHOLDER_KINDS = ("research-code", "research-citation")
_PLACEHOLDER = re.compile(rf"\x00(?:{'|'.join(_PLACEHOLDER_KINDS)})-[0-9]+\x00")
_CODE_PLACEHOLDER = re.compile(r"\x00research-code-[0-9]+\x00")
# remark-gfm renders indented footnote lines as prose (live links); CommonMark calls them code.
_FOOTNOTE_DEFINITION = re.compile(r" {0,3}\[\^[^\]\s]+\]:")
# Mermaid fences execute (image URLs fetched by the browser), so their URLs stay validated.
_MERMAID_FENCE = re.compile(r"mermaid\b", re.IGNORECASE)


def _citation_title(source: dict, fallback: str) -> str:
    """Strips brackets from titles, since a [PDF] prefix would make the citation unmatchable."""
    title = str(source.get("title") or fallback).replace("[", "").replace("]", "").strip()
    return title or fallback


def _trim_url_tail(raw: str) -> str:
    """Trims trailing prose punctuation and unmatched ')' like GFM autolinks, in one interleaved pass."""
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
    """NUL-delimited: _mask_code strips any NULs the model wrote, so a copy cannot pass as a token."""
    return f"\x00{kind}-{index}\x00"


def _record_code_span(state, silent: bool) -> bool:
    start = state.pos
    count = len(state.tokens)
    matched = backtick(state, silent)
    if (
        matched
        and not silent
        # Block's own source only: the image rule re-enters with alt text.
        and state.src is state.env.get("code_source")
        and len(state.tokens) > count
        and state.tokens[-1].type == "code_inline"
    ):
        state.env["code_spans"].append((start, state.pos))
    return matched


# Tables enabled to match the renderer, else backticks pair across cells.
_CODE_MARKDOWN = MarkdownIt("commonmark").enable("table").disable("inline")
_CODE_MARKDOWN.inline.ruler.at("backticks", _record_code_span)


def _footnote_content_lines(lines: list[str]) -> set[int]:
    """Footnote content lines per the renderer; loose on purpose, as over-collecting only masks a block."""
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
    # CommonMark maps NUL to U+FFFD; one code point, so offsets are unaffected.
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
        """Cuts at unescaped pipes, as the renderer splits rows, so a code span stays inside one cell."""
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
        if token.type == "code_block" and token.map[0] in footnote_content:
            continue
        if token.type == "fence" and _MERMAID_FENCE.match((token.info or "").strip()):
            continue
        start, end = (offsets[line] for line in token.map)
        if token.type in {"fence", "code_block"}:
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
    """Restores every token in one re.sub pass; a replace per token rescans the report once per span."""
    if not placeholders:
        return text

    def restore(match: re.Match) -> str:
        token = match.group(0)
        value = placeholders.get(token)
        if value is not None:
            return value
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
        title = re.sub(r"(\\*)\|", r"\1\1\\|", _citation_title(source, url))
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
        raw = match.group(0)
        core = _trim_url_tail(raw)
        if core in source_by_url:
            return (citation(core) or core) + raw[len(core) :]
        return raw[len(core) :]

    validated = replace_markdown_links(report)
    validated = _AUTOLINK.sub(replace_autolink, validated)
    validated = _NUMBERED_CITATION.sub(replace_number, validated)
    validated = _RAW_URL.sub(replace_raw_url, validated)
    return validated.strip()


def _validate_report_sources(report: str, sources: list[dict]) -> str:
    """canonicalize citations and remove model-authored source lists."""
    placeholders: dict[str, str] = {}
    validated = _validate_masked_sources(_mask_code(report, placeholders), sources, placeholders)
    return _restore_placeholders(validated, placeholders)


def _document_source_citation(source: dict) -> str:
    filename = str(source.get("filename") or "Document")
    if source.get("kind") == "mcp":
        return f"[MCP: {_mcp_source_label(source)}]"
    if source.get("page") is not None:
        return f"[Document: {filename}, p. {source['page']}]"
    return f"[Document: {filename}]"


def _mcp_source_label(source: dict) -> str:
    filename = str(source.get("filename") or "MCP source")
    label = " ".join(filename.replace("[", "").replace("]", "").split())
    return label or "MCP source"


def _allowed_document_citations(sources: list[dict]) -> set[str]:
    allowed = set()
    for source in sources:
        if source.get("kind") != "mcp":
            allowed.add(f"[Document: {source.get('filename') or 'Document'}]")
        allowed.add(_document_source_citation(source))
    return allowed


def _validate_masked_document_sources(
    report: str, sources: list[dict], placeholders: dict[str, str]
) -> str:
    allowed = _allowed_document_citations(sources)

    def keep_if_allowed(match: re.Match) -> str:
        citation = _restore_placeholders(match.group(0), placeholders)
        if citation in allowed:
            return match.group(0)
        return "".join(_CODE_PLACEHOLDER.findall(match.group(0)))

    return _DOCUMENT_CITATION.sub(keep_if_allowed, report)


def _validate_report_document_sources(report: str, sources: list[dict]) -> str:
    placeholders: dict[str, str] = {}
    validated = _validate_masked_document_sources(
        _mask_code(report, placeholders), sources, placeholders
    )
    return _restore_placeholders(validated, placeholders)


def _validate_report(report: str, sources: list[dict], document_sources: list[dict]) -> str:
    placeholders: dict[str, str] = {}
    validated = _validate_masked_sources(_mask_code(report, placeholders), sources, placeholders)
    validated = _validate_masked_document_sources(validated, document_sources, placeholders)
    return _restore_placeholders(validated, placeholders)
