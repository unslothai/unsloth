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
_PLACEHOLDER_KINDS = ("research-code", "research-citation")
# Kept beside the kinds so a new one cannot be restored by a pass that does not know it, which
# would leave the raw sentinel in the delivered report.
_PLACEHOLDER = re.compile(rf"\x00(?:{'|'.join(_PLACEHOLDER_KINDS)})-\d+\x00")


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
    """Sentinel standing in for text a validator must move but not rewrite.

    NUL delimited because the validators are regex passes over prose and a NUL cannot survive
    into a report: ``_mask_code`` normalizes any the model wrote away first.
    """
    return f"\x00{kind}-{index}\x00"


def _record_code_span(state, silent: bool) -> bool:
    start = state.pos
    count = len(state.tokens)
    matched = backtick(state, silent)
    if (
        matched
        and not silent
        and state.src is state.env["code_source"]
        and len(state.tokens) > count
        and state.tokens[-1].type == "code_inline"
    ):
        state.env["code_spans"].append((start, state.pos))
    return matched


# Block maps retain the original lines, including indentation and container markers.
# Parse inline source separately so code offsets refer to those original lines too.
_CODE_MARKDOWN = MarkdownIt("commonmark").disable("inline")
_CODE_MARKDOWN.inline.ruler.at("backticks", _record_code_span)


def _mask_code(text: str, placeholders: dict[str, str]) -> str:
    # CommonMark replaces a literal NUL with U+FFFD, so the renderer already shows the report
    # that way. Doing it before anything is masked also means a report cannot spell a
    # placeholder token itself and have restoration hand it another region's text. Both
    # characters are one code point, so the line offsets below are unaffected.
    text = text.replace("\x00", "\ufffd")
    offsets = [0, *(match.end() for match in re.finditer(r"\r\n?|\n", text))]
    if offsets[-1] != len(text):
        offsets.append(len(text))
    spans = []
    for token in _CODE_MARKDOWN.parse(text):
        if token.map is None:
            continue
        start, end = (offsets[line] for line in token.map)
        if token.type in {"fence", "code_block"}:
            # Keep the following heading on its own line while the block is masked.
            end = start + len(text[start:end].rstrip("\r\n"))
            spans.append((start, end))
        elif token.type == "inline":
            source = text[start:end]
            env = {"code_source": source, "code_spans": []}
            _CODE_MARKDOWN.inline.parse(source, _CODE_MARKDOWN, env, [])
            spans.extend((start + first, start + last) for first, last in env["code_spans"])
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

    A replace() per token rescans the whole report once per masked region, so a code-heavy
    report costs O(len(report) x spans); one sub() is linear. The single pass also means a
    restored code span is never itself searched for a later token, so code that happens to
    contain a token's text survives verbatim.
    """
    if not placeholders:
        return text
    return _PLACEHOLDER.sub(lambda match: placeholders.get(match.group(0), match.group(0)), text)


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
        # Judge the citation as the model wrote it. The pattern already spans a "]" inside a
        # filename ("budget [final].pdf"), and a filename may also contain backticks, which
        # _mask_code replaced with a placeholder before this pass ran.
        citation = _restore_placeholders(match.group(0), placeholders)
        return match.group(0) if citation in allowed else ""

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
