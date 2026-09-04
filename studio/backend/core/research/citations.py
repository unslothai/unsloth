# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Rewriting Deep Research report citations to the sources actually gathered.

A local model invents URLs, renumbers references, and appends its own source list. Every
citation is canonicalized against the run's catalogs, and anything that does not resolve to a
gathered web source or document chunk is stripped rather than shown as supported.
"""

from __future__ import annotations

import re

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
_RAW_URL = re.compile(r"https?://[^\s<>]+")


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


def _validate_report_sources(report: str, sources: list[dict]) -> str:
    """Canonicalize citations and remove model-authored source lists."""
    source_by_url = {
        str(source.get("url") or ""): source for source in sources if source.get("url")
    }
    source_urls = list(source_by_url)
    placeholders: dict[str, str] = {}

    heading = _SOURCES_HEADING.search(report)
    if heading:
        report = report[: heading.start()]

    def citation(url: str) -> str | None:
        source = source_by_url.get(url)
        if source is None:
            return None
        title = _citation_title(source, url)
        token = f"\x00research-citation-{len(placeholders)}\x00"
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
    for token, link in placeholders.items():
        validated = validated.replace(token, link)
    return validated.strip()


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


def _validate_report_document_sources(report: str, sources: list[dict]) -> str:
    allowed = _allowed_document_citations(sources)
    # Tokenize valid citations first so a "]" inside a filename ("budget [final].pdf") does not
    # truncate them, then strip the invalid ones and restore the valid.
    placeholders: dict[str, str] = {}
    for index, citation in enumerate(sorted(allowed, key = len, reverse = True)):
        if citation in report:
            token = f"\x00document-citation-{index}\x00"
            placeholders[token] = citation
            report = report.replace(citation, token)
    report = _DOCUMENT_CITATION.sub("", report)
    for token, citation in placeholders.items():
        report = report.replace(token, citation)
    return report
