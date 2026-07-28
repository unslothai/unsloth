# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Release notes for the update popup, sourced from CHANGELOG.md.

Notes are keyed to one exact version: the popup asks for the version it is
offering and gets that section or nothing, so an older release's notes can
never appear next to a newer update.

The remote copy on the default branch wins over the bundled one, since the
offered version is newer than the installed checkout. Both reads are lazy,
cached and skipped when update checks are off.
"""

from __future__ import annotations

import os
import re
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

from .update_status import DISABLE_ENV_VAR, RELEASE_NOTES_URL

CHANGELOG_FILENAME = "CHANGELOG.md"
CHANGELOG_RAW_URL = "https://raw.githubusercontent.com/unslothai/unsloth/main/CHANGELOG.md"
CHANGELOG_URL_ENV_VAR = "UNSLOTH_CHANGELOG_URL"
CHANGELOG_PATH_ENV_VAR = "UNSLOTH_CHANGELOG_PATH"
CHANGELOG_TIMEOUT_SECONDS = 3
CHANGELOG_MAX_BYTES = 2 * 1024 * 1024
_CHANGELOG_CHUNK_BYTES = 64 * 1024
_CHANGELOG_MIN_READ_SECONDS = 0.05
CHANGELOG_SUCCESS_TTL_SECONDS = 30 * 60
CHANGELOG_FAILURE_TTL_SECONDS = 5 * 60
RELEASE_NOTES_MAX_CHARS = 20_000

# CommonMark requires a space or tab after the hashes: a non-breaking space
# copied from rich text renders as ordinary text, not a heading.
_HEADING_PATTERN = re.compile(r"^ {0,3}##[ \t]+(?P<title>.*?)[ \t]*$")
_FENCE_PATTERN = re.compile(r"^ {0,3}(?P<marker>`{3,}|~{3,})(?P<rest>.*)$")
# CommonMark type 1 HTML blocks: contents are literal until a closing tag,
# which the spec says need not be the one that opened the block.
_RAW_HTML_OPEN = re.compile(r"^ {0,3}<(pre|script|style|textarea)(?=[\s>]|$)", re.IGNORECASE)
_RAW_HTML_CLOSE = re.compile(r"</(pre|script|style|textarea)\s*>", re.IGNORECASE)
# Types 3 to 5 are literal too: processing instructions, declarations such as
# <!DOCTYPE, and CDATA. Each ends on its own delimiter. Comments (type 2) are
# handled separately because they can also open mid-line.
_RAW_BLOCKS = (
    (_RAW_HTML_OPEN, _RAW_HTML_CLOSE),
    (re.compile(r"^ {0,3}<\?"), re.compile(r"\?>")),
    (re.compile(r"^ {0,3}<!\[CDATA\["), re.compile(r"\]\]>")),
    # A declaration needs an uppercase letter, so `<!note` stays ordinary text
    # rather than hiding every release below it.
    (re.compile(r"^ {0,3}<![A-Z]"), re.compile(r">")),
)
# Type 6 blocks run to the next blank line, so `<details>` only holds Markdown
# once a blank line has closed the block. Open and close tags both start one.
_HTML_BLOCK_OPEN = re.compile(r"^ {0,3}</?([a-zA-Z][a-zA-Z0-9-]*)(?=[\s/>]|$)")
# Lines that are blocks in their own right, so no paragraph is open after them.
_NOT_PARAGRAPH = re.compile(
    r"^ {0,3}(?:#{1,6}([ \t]|$)"
    r"|(?:\*[ \t]*){3,}$|(?:-[ \t]*){3,}$|(?:_[ \t]*){3,}$"
    r"|\[(?:[^\[\]\\]|\\.)+\]:)"
)
# Blocks that are not paragraph text, so a following underline is not setext.
_PARAGRAPH_TEXT = re.compile(r"^ {0,3}(?![-*+>]([ \t]|$)|\d{1,9}[.)]([ \t]|$))\S")
# A line of = or - under a paragraph line makes that line a heading.
_SETEXT_UNDERLINE = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
# A paragraph inside a blockquote can be continued by lines without the marker,
# so those lines belong to the quote rather than to the document.
_BLOCK_QUOTE = re.compile(r"^ {0,3}>")
_QUOTE_MARKER = re.compile(r"^ {0,3}>[ \t]?")
# A list item holds the lines indented to where its own content starts, so a
# heading at that column belongs to the item, not to the document. The marker
# needs whitespace after it, so `2.0` is a version rather than an item.
_LIST_ITEM = re.compile(r"^[ \t]*(?P<marker>[-*+]|\d{1,9}[.)])(?P<space>[ \t]+|$)")
_THEMATIC_BREAK = re.compile(r"^ {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$")
# Content indented more than this after a marker is an indented code block, so
# the item's content starts one column past the marker instead.
_MAX_ITEM_PADDING = 4
_HTML_BLOCK_TAGS = frozenset(
    """
address article aside base basefont blockquote body caption center col colgroup
dd details dialog dir div dl dt fieldset figcaption figure footer form frame
frameset h1 h2 h3 h4 h5 h6 head header hr html iframe legend li link main menu
menuitem nav noframes ol optgroup option p param search section summary table
tbody td tfoot th thead title tr track ul
""".split()
)
# Type 7: any other complete tag alone on a line, which also runs to a blank
# line. It cannot interrupt a paragraph, so it only counts after a break.
_HTML_ATTRIBUTE = (
    r"""(?:\s+[a-zA-Z_:][a-zA-Z0-9_.:-]*(?:\s*=\s*(?:[^\s"'=<>`]+|'[^']*'|"[^"]*"))?)"""
)
_HTML_TAG_ONLY_LINE = re.compile(
    rf"^ {{0,3}}(?:<[a-zA-Z][a-zA-Z0-9-]*{_HTML_ATTRIBUTE}*\s*/?>|</[a-zA-Z][a-zA-Z0-9-]*\s*>)\s*$"
)
# Levels above studio/ are the repo root in a checkout and site-packages in an
# install, so they are searched only when one of these markers is present.
_CHECKOUT_ONLY_LEVELS = (3, 4)
_CHECKOUT_MARKERS = ("pyproject.toml", ".git")
_COMMENT_BLOCK_OPEN = re.compile(r"^ {0,3}<!--")
_COMMENT_OPEN = "<!--"
_COMMENT_CLOSE = "-->"
_VERSION_TOKEN_PATTERN = re.compile(r"^[\[(]?v?(?P<version>[0-9][0-9A-Za-z.!+-]*?)[\])]?$")
_SAFE_VERSION_PATTERN = re.compile(r"^[0-9A-Za-z][0-9A-Za-z.!+-]{0,63}$")


@dataclass(frozen = True)
class _ListState:
    """The open list items, innermost last, by the column their content starts."""

    columns: tuple[int, ...] = ()
    # True while the innermost item has had no content since its marker.
    empty_item: bool = False


@dataclass(frozen = True)
class ChangelogEntry:
    """One `## <version>` section of the changelog."""

    version: str
    heading: str
    body: str


@dataclass(frozen = True)
class ChangelogSource:
    text: str | None
    source: str | None
    error: str | None = None


@dataclass
class _ChangelogCacheEntry:
    source: ChangelogSource
    expires_at: float


_cache_condition = threading.Condition()
_remote_cache: _ChangelogCacheEntry | None = None
_remote_fetching = False


def reset_changelog_cache() -> None:
    """Clear the in-process changelog cache. Intended for tests."""
    global _remote_cache, _remote_fetching
    with _cache_condition:
        _remote_cache = None
        _remote_fetching = False
        _cache_condition.notify_all()


def is_supported_version_query(version: str) -> bool:
    """Whether `version` is shaped like something we can look up at all.

    Sections are indexed only when their version parses, so a query that does
    not parse (`latest`, `main`) can never match and is rejected outright."""
    candidate = version.strip()
    if not _SAFE_VERSION_PATTERN.match(candidate):
        return False
    return _parse_version(candidate) is not None


def parse_changelog(text: str) -> list[ChangelogEntry]:
    """Parse `## <version>` sections, in file order.

    Headings whose first token is not a version (`## Unreleased`, `## Format`)
    end the previous section but are not indexed.
    """
    # A Windows editor can leave a BOM on the first line, hiding a heading.
    text = text.lstrip("﻿")
    entries: list[ChangelogEntry] = []
    heading: str | None = None
    version: str | None = None
    body: list[str] = []
    open_fence: str | None = None
    in_comment = False
    in_raw_html: int | None = None
    in_html_block = False
    after_paragraph = False
    paragraph: list[str] = []
    in_quote = False
    lists = _ListState()

    def flush() -> None:
        if version is not None and heading is not None:
            entries.append(
                ChangelogEntry(
                    version = version,
                    heading = heading,
                    body = "\n".join(body).strip(),
                )
            )

    for line in text.splitlines():
        # The line as list tracking sees it: blank where the renderer shows
        # nothing, so hidden lines neither open nor close an item.
        structural = ""
        # Raw HTML first: its contents are literal, so a fence inside it is not
        # a fence. Only the text after the closing tag can carry a heading.
        if in_raw_html is not None:
            visible, in_raw_html = _strip_raw_html(line, in_raw_html)
        elif in_html_block:
            # A blank line is the only thing that ends a type 6 block.
            in_html_block = line.strip() != ""
            visible = ""
        elif (fence := _FENCE_PATTERN.match(line)) and not in_comment:
            open_fence = _next_fence_state(open_fence, fence.group("marker"), fence.group("rest"))
            # Hidden from heading matching, but its indent still closes a list
            # item it sits to the left of.
            visible = ""
            structural = line
        elif open_fence:
            visible = ""
        else:
            # Commented-out sections are not rendered, so they are not releases.
            visible, in_comment = _strip_comments(line, in_comment)
            # Nor is anything inside a raw HTML block such as <pre>.
            visible, in_raw_html = _strip_raw_html(visible, in_raw_html)
            if visible and in_raw_html is None and _opens_html_block(visible, after_paragraph):
                in_html_block = True
                visible = ""
            structural = visible
        # A `##` inside a fenced block is sample markdown, not a real heading.
        match = _HEADING_PATTERN.match(visible) if visible else None
        # `1.0` over a line of dashes is the same heading written setext style.
        setext = (
            after_paragraph
            and match is None
            and paragraph != []
            and _SETEXT_UNDERLINE.match(visible) is not None
            and (visible.strip()[:1] == "-")
            # Never a release boundary while a list item is open: dedented out
            # of it the dashes are a thematic break, and at its content column
            # the heading belongs to the item.
            and not lists.columns
        )
        if setext:
            if version is not None:
                # The whole paragraph is the heading, and it was read as body
                # as it arrived.
                del body[len(body) - len(paragraph) :]
            flush()
            # A wrapped heading keeps every line, so its first token is still
            # the version.
            heading = "\n".join(paragraph)
            version = _version_from_heading(heading)
            body = []
            paragraph = []
            after_paragraph = False
            continue
        # A dashed underline is not a list marker, so lists are tracked only
        # once setext is ruled out.
        lists = _open_lists(structural, lists, after_paragraph)
        # Indented to an open item's content column, a heading belongs to that
        # item and is not a release boundary.
        if lists.columns and _indent_width(visible) >= lists.columns[0]:
            match = None
        # The line at its own nesting level: past the container's indentation
        # and past a marker on the same line, so `- ## 2.0` reads as a heading.
        column = lists.columns[-1] if lists.columns else 0
        content = _strip_indent(visible, column)
        if (item := _LIST_ITEM.match(content)) is not None:
            content = content[item.end() :]
        # Only ordinary text continues a paragraph. A heading, a block or an
        # indented code line ends one. Four spaces past the container, not past
        # the margin: inside a list item the item's own indent does not count.
        indented_code = not after_paragraph and _indent_width(visible) - column >= 4
        # `===` with no paragraph above it is ordinary text, not an underline.
        # A row of dashes there is a thematic break either way.
        underline = _SETEXT_UNDERLINE.match(visible) is not None and (
            after_paragraph or visible.strip()[:1] == "-"
        )
        after_paragraph = (
            bool(visible.strip())
            and match is None
            and _HEADING_PATTERN.match(content) is None
            and not indented_code
            and _NOT_PARAGRAPH.match(visible) is None
            and not underline
        )
        # A quote's paragraph runs on until something other than plain text, and
        # every line of it belongs to the quote. An empty quote holds no
        # paragraph, so the line below it starts one of the document's own.
        flush_left = visible.lstrip(" \t")
        in_quote = (
            _may_be_lazy(_quote_content(visible))
            if _BLOCK_QUOTE.match(visible)
            else in_quote and _may_be_lazy(flush_left)
        )
        # The lines a later underline turns into one heading. A paragraph starts
        # only on plain text, so `- first` over dashes is a list and a rule.
        # Once open it runs on until something interrupts it, at whatever
        # indentation the wrapped lines carry.
        continues = (
            not _interrupts_paragraph(flush_left)
            if paragraph
            else _PARAGRAPH_TEXT.match(flush_left) is not None
        )
        if after_paragraph and not in_quote and continues:
            paragraph = [*paragraph, visible.strip()]
        else:
            paragraph = []
        if match is None:
            if version is not None:
                body.append(line)
            continue

        flush()
        heading = match.group("title")
        version = _version_from_heading(heading)
        body = []

    flush()
    return entries


def find_release_notes(text: str, version: str) -> ChangelogEntry | None:
    """Return the section for exactly `version`, or None.

    Equality is version-aware (`2026.07.5` matches `2026.7.5`) but never fuzzy:
    a near-miss returns None so the caller shows no notes, not the wrong ones.
    """
    entries = parse_changelog(text)
    for entry in entries:
        # An exact heading wins wherever it sits, so `## 1.0` is never shadowed
        # by an earlier `## 1.0.0`.
        if entry.version == version:
            return entry

    wanted = _parse_version(version)
    for entry in entries:
        if wanted is not None:
            candidate = _parse_version(entry.version)
            if candidate is not None and candidate == wanted:
                return entry
    return None


def get_release_notes(version: str, refresh: bool = False) -> dict[str, Any]:
    """Return release notes for exactly `version` for the update popup.

    `refresh` retries a cached remote failure, so the UI's retry action is not
    stuck behind the failure TTL once connectivity returns.
    """
    version = version.strip()
    if not is_supported_version_query(version):
        return _notes_response(version = version, error = "Unsupported version.")

    local = _read_local_changelog()
    remote = ChangelogSource(text = None, source = None)
    if os.environ.get(DISABLE_ENV_VAR) != "1":
        remote = get_remote_changelog(refresh = refresh)

    # Remote first: the offered version is newer than the local copy.
    for candidate in (remote, local):
        if not candidate.text:
            continue
        entry = find_release_notes(candidate.text, version)
        if entry is not None:
            return _notes_response(
                version = version,
                markdown = entry.body,
                heading = entry.heading,
                source = candidate.source,
            )

    # Nothing matched: a remote failure still matters, since the bundled copy
    # cannot know a version newer than the install. Reporting it lets the UI
    # offer a retry instead of claiming no notes were published.
    return _notes_response(version = version, error = remote.error)


def get_remote_changelog(refresh: bool = False) -> ChangelogSource:
    """Fetch CHANGELOG.md from the repo using a small in-process TTL cache."""
    global _remote_cache, _remote_fetching

    if refresh:
        # Only a cached failure is dropped: a successful fetch stays cached so
        # retries cannot be used to hammer the remote.
        with _cache_condition:
            if _remote_cache and _remote_cache.source.text is None:
                _remote_cache = None

    # A caller waits for an in-flight fetch only as long as that fetch may
    # take. Past that the request is answered from the local copy instead of
    # holding a worker behind a stalled upstream.
    deadline = time.monotonic() + CHANGELOG_TIMEOUT_SECONDS + 1
    while True:
        now = time.monotonic()
        with _cache_condition:
            if _remote_cache and _remote_cache.expires_at > now:
                return _remote_cache.source
            if not _remote_fetching:
                _remote_fetching = True
                break
            if now >= deadline:
                return ChangelogSource(
                    text = None,
                    source = None,
                    error = "Release notes are still loading.",
                )
            _cache_condition.wait(timeout = deadline - now)

    try:
        try:
            source = _fetch_remote_changelog()
        except Exception:
            source = ChangelogSource(
                text = None,
                source = None,
                error = "Could not fetch release notes.",
            )

        ttl = CHANGELOG_SUCCESS_TTL_SECONDS if source.text else CHANGELOG_FAILURE_TTL_SECONDS
        with _cache_condition:
            _remote_cache = _ChangelogCacheEntry(
                source = source, expires_at = time.monotonic() + ttl
            )
        return source
    finally:
        # Releasing the single-flight flag only on the Exception path would
        # strand it on BaseException (KeyboardInterrupt, CancelledError), and
        # every later caller would then wait out the full deadline forever.
        with _cache_condition:
            _remote_fetching = False
            _cache_condition.notify_all()


def _fetch_remote_changelog() -> ChangelogSource:
    url = os.environ.get(CHANGELOG_URL_ENV_VAR, "").strip() or CHANGELOG_RAW_URL
    if not url.startswith(("http://", "https://")):
        return ChangelogSource(text = None, source = None, error = "Invalid changelog URL.")

    request = urllib.request.Request(
        url,
        headers = {
            "User-Agent": "unsloth-studio-update-check",
            # A compressing proxy would otherwise hand back bytes we decode as
            # text and serve as notes.
            "Accept-Encoding": "identity",
        },
    )
    deadline = time.monotonic() + CHANGELOG_TIMEOUT_SECONDS
    try:
        with urllib.request.urlopen(request, timeout = CHANGELOG_TIMEOUT_SECONDS) as response:
            chunks: list[bytes] = []
            received = 0
            while received <= CHANGELOG_MAX_BYTES:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return ChangelogSource(
                        text = None,
                        source = None,
                        error = "Release notes took too long to load.",
                    )
                # One read must not outlast the deadline either: the socket
                # timeout is per operation and would otherwise restart it.
                _limit_read(response, remaining)
                chunk = response.read1(_CHANGELOG_CHUNK_BYTES)
                if not chunk:
                    break
                chunks.append(chunk)
                received += len(chunk)
            body = b"".join(chunks)
        if len(body) > CHANGELOG_MAX_BYTES:
            return ChangelogSource(
                text = None,
                source = None,
                error = "Release notes response was too large.",
            )
        return ChangelogSource(text = body.decode("utf-8", errors = "replace"), source = "remote")
    except TimeoutError:
        return ChangelogSource(
            text = None,
            source = None,
            error = "Release notes took too long to load.",
        )
    except OSError:
        return ChangelogSource(
            text = None,
            source = None,
            error = "Could not reach the changelog for release notes.",
        )
    except UnicodeError:
        return ChangelogSource(text = None, source = None, error = "Malformed changelog.")


def _limit_read(response: Any, remaining: float) -> None:
    """Cap the next socket read at the time left in the fetch budget."""
    sock = getattr(getattr(response, "fp", None), "raw", None)
    sock = getattr(sock, "_sock", None)
    if sock is None:
        return
    try:
        sock.settimeout(max(remaining, _CHANGELOG_MIN_READ_SECONDS))
    except OSError:
        pass


def _read_local_changelog() -> ChangelogSource:
    """Read the CHANGELOG.md bundled with this install, if there is one."""
    for path in _local_changelog_candidates():
        try:
            if not path.is_file():
                continue
            if path.stat().st_size > CHANGELOG_MAX_BYTES:
                continue
            return ChangelogSource(
                text = path.read_text(encoding = "utf-8", errors = "replace"),
                source = "local",
            )
        except OSError:
            continue
    return ChangelogSource(text = None, source = None)


def _is_source_checkout(root: Path) -> bool:
    """Whether `root` is this repository rather than an install directory."""
    try:
        return any((root / marker).exists() for marker in _CHECKOUT_MARKERS)
    except OSError:
        return False


def _local_changelog_candidates() -> list[Path]:
    override = os.environ.get(CHANGELOG_PATH_ENV_VAR, "").strip()
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())

    # changelog.py -> utils -> backend -> studio -> repo root. Repo root first:
    # in a source checkout the editable file must win over the build snapshot
    # that packaging writes into studio/. Installed, those outer levels are
    # site-packages, so they are only used when a checkout marker is there.
    parents = Path(__file__).resolve().parents
    for index in (3, 2, 1, 4):
        if index >= len(parents):
            continue
        root = parents[index]
        if index in _CHECKOUT_ONLY_LEVELS and not _is_source_checkout(root):
            continue
        candidates.append(root / CHANGELOG_FILENAME)

    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _opens_fence(marker: str, rest: str) -> bool:
    """A backtick fence's info string may not contain a backtick."""
    return marker[0] != "`" or "`" not in rest


def _next_fence_state(open_fence: str | None, marker: str, rest: str) -> str | None:
    """Track the open fence marker.

    A closer must be the same character, at least as long, and carry nothing
    after it. So neither a ``` sample nor a ```` line with trailing text ends
    a ```` block early, while an opening fence may still have an info string.
    Only spaces and tabs count as nothing: other Unicode whitespace is content.
    """
    if open_fence is None:
        return marker if _opens_fence(marker, rest) else None
    closes = marker[0] == open_fence[0] and len(marker) >= len(open_fence)
    if closes and not rest.strip(" \t"):
        return None
    return open_fence


def _code_span_ranges(line: str) -> list[tuple[int, int]]:
    """Code span bounds. A run of backticks closes only on a run of its length."""
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(line):
        if line[index] != "`" or _is_escaped(line, index):
            index += 1
            continue
        ticks = _run_length(line, index)
        cursor = index + ticks
        while cursor < len(line):
            if line[cursor] != "`" or _is_escaped(line, cursor):
                cursor += 1
                continue
            candidate = _run_length(line, cursor)
            if candidate == ticks:
                spans.append((index, cursor + ticks))
                break
            cursor += candidate
        else:
            # Nothing closes this run, so it is literal text.
            index += ticks
            continue
        index = spans[-1][1]
    return spans


def _run_length(line: str, index: int) -> int:
    end = index
    while end < len(line) and line[end] == "`":
        end += 1
    return end - index


def _is_escaped(line: str, index: int) -> bool:
    slashes = 0
    while index - 1 - slashes >= 0 and line[index - 1 - slashes] == "\\":
        slashes += 1
    return slashes % 2 == 1


def _strip_comments(line: str, in_comment: bool) -> tuple[str, bool]:
    """Return the line with HTML-comment spans removed, and the trailing state.

    Only a comment that starts a line opens a block and hides the lines below
    it. One written mid-sentence is inline HTML: it hides the rest of its own
    line at most, so a note mentioning `<!--` cannot swallow later releases.
    Delimiters inside inline code are literal and hide nothing.
    """
    if in_comment:
        close = line.find(_COMMENT_CLOSE)
        # The closing line belongs to the block, tail included.
        return ("", False) if close != -1 else ("", True)

    if _COMMENT_BLOCK_OPEN.match(line):
        # `<!-->` and `<!--->` are complete comments, so the closer may overlap
        # the opener. Searching past the opener would miss them and swallow
        # every later release.
        return ("", _COMMENT_CLOSE not in line)

    visible: list[str] = []
    index = 0
    spans = _code_span_ranges(line)
    while index < len(line):
        opening = line.find(_COMMENT_OPEN, index)
        if opening == -1:
            visible.append(line[index:])
            break

        span = next((s for s in spans if s[0] <= opening < s[1]), None)
        if span is not None:
            visible.append(line[index : span[1]])
            index = span[1]
            continue

        visible.append(line[index:opening])
        close = line.find(_COMMENT_CLOSE, opening + len(_COMMENT_OPEN))
        if close == -1:
            # Unterminated inline comment: a heading or a blank line below
            # ends the paragraph, so nothing beyond this line is hidden.
            break
        index = close + len(_COMMENT_CLOSE)
    return "".join(visible), False


def _indent_width(line: str) -> int:
    """Columns of leading whitespace, counting a tab to the next stop of four."""
    width = 0
    for char in line:
        if char == " ":
            width += 1
        elif char == "\t":
            width += 4 - width % 4
        else:
            break
    return width


def _strip_indent(line: str, columns: int) -> str:
    """`line` with up to `columns` columns of leading whitespace removed."""
    width = 0
    index = 0
    while index < len(line) and width < columns and line[index] in " \t":
        width += 1 if line[index] == " " else 4 - width % 4
        index += 1
    return line[index:]


def _interrupts_paragraph(line: str) -> bool:
    """Whether `line` starts a block that can break into an open paragraph.

    A quote marker always can. A list item can only when it has content, and an
    ordered one only when it starts at 1: anything else is text of the
    paragraph it appears to interrupt."""
    if _BLOCK_QUOTE.match(line):
        return True
    item = None if _THEMATIC_BREAK.match(line) else _LIST_ITEM.match(line)
    if item is None:
        return False
    marker = item.group("marker")
    if not line[item.end() :].strip():
        return False
    return marker[-1] not in ".)" or marker[:-1] == "1"


def _quote_content(line: str) -> str:
    """What a blockquote line holds, with its markers stripped."""
    while (marker := _QUOTE_MARKER.match(line)) is not None:
        line = line[marker.end() :]
    return line


def _may_be_lazy(line: str) -> bool:
    """Whether `line` can continue a paragraph it is indented out of.

    Only plain text can: a heading, a fence, a break or an underline starts a
    block of its own, which closes the item instead."""
    return (
        _PARAGRAPH_TEXT.match(line) is not None
        and _NOT_PARAGRAPH.match(line) is None
        and _SETEXT_UNDERLINE.match(line) is None
        and _FENCE_PATTERN.match(line) is None
    )


def _open_lists(line: str, state: _ListState, after_paragraph: bool) -> _ListState:
    """The list items still open after `line`.

    A dedented line closes an item, unless it is a lazy paragraph continuation.
    A new marker nests under a deeper column and replaces a sibling.
    """
    columns = state.columns
    if not line.strip():
        # A blank line leaves the list open, unless the item is still empty:
        # an item may begin with one blank line, and content after that is
        # outside it.
        return _ListState(columns[:-1] if state.empty_item else columns)
    indent = _indent_width(line)
    item = None if _THEMATIC_BREAK.match(line) else _LIST_ITEM.match(line)
    empty = item is not None and not line[item.end() :].strip()
    # Only a marker inside the item holding the paragraph has to interrupt it;
    # one to the left closes that item and opens a sibling instead.
    if (
        item is not None
        and after_paragraph
        and (not columns or indent >= columns[-1])
        and not _interrupts_paragraph(line)
    ):
        # A lazy continuation or an underline, so the open items are untouched.
        return state
    if not (after_paragraph and _may_be_lazy(line)):
        while columns and indent < columns[-1]:
            columns = columns[:-1]
    if item is None:
        return _ListState(columns)
    marker = item.group("marker")
    padding = _indent_width(item.group("space"))
    if padding == 0 or padding > _MAX_ITEM_PADDING:
        # An empty or over-indented item still holds one column of content.
        padding = 1
    while columns and columns[-1] > indent:
        columns = columns[:-1]
    return _ListState((*columns, indent + len(marker) + padding), empty_item = empty)


def _opens_html_block(line: str, after_paragraph: bool) -> bool:
    """True if `line` starts a CommonMark type 6 or type 7 HTML block."""
    match = _HTML_BLOCK_OPEN.match(line)
    if match is not None and match.group(1).lower() in _HTML_BLOCK_TAGS:
        return True
    return not after_paragraph and _HTML_TAG_ONLY_LINE.match(line) is not None


def _strip_raw_html(line: str, open_block: int | None) -> tuple[str, int | None]:
    """Drop the parts of a line inside a raw block, and return the open block.

    The state is the index of the open block in `_RAW_BLOCKS`, or None."""
    if open_block is not None:
        close = _RAW_BLOCKS[open_block][1].search(line)
        return ("", None) if close else ("", open_block)

    # A block only opens at the start of a line; mid-line tags are inline HTML.
    for index, (opener, closer) in enumerate(_RAW_BLOCKS):
        opening = opener.match(line)
        if opening is None:
            continue
        rest = line[opening.end() :]
        close = closer.search(rest)
        return ("", None) if close else ("", index)
    return line, None


def _version_from_heading(heading: str) -> str | None:
    token = heading.split()[0] if heading.split() else ""
    match = _VERSION_TOKEN_PATTERN.match(token)
    if match is None:
        return None
    version = match.group("version")
    return version if _parse_version(version) is not None else None


def _parse_version(version: str) -> Version | None:
    try:
        return Version(version)
    except InvalidVersion:
        return None


def _close_open_fence(markdown: str) -> str:
    """Close a fence the truncation cut in half, so the rest still renders."""
    open_fence: str | None = None
    for line in markdown.splitlines():
        fence = _FENCE_PATTERN.match(line)
        if fence:
            open_fence = _next_fence_state(open_fence, fence.group("marker"), fence.group("rest"))
    return f"{markdown}\n{open_fence}" if open_fence else markdown


def _renders_visibly(markdown: str) -> bool:
    """Whether a section body renders anything at all."""
    in_comment = False
    for line in markdown.splitlines():
        opens_raw = any(opener.match(line) for opener, _ in _RAW_BLOCKS)
        if not in_comment and (_FENCE_PATTERN.match(line) or opens_raw):
            # A code block or raw HTML block renders even when it is empty.
            return True
        visible, in_comment = _strip_comments(line, in_comment)
        if visible.strip():
            return True
    return False


def _notes_response(
    *,
    version: str,
    markdown: str | None = None,
    heading: str | None = None,
    source: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    # A section staged as only an HTML comment renders as nothing, so it counts
    # as unpublished rather than as empty notes.
    if markdown and not _renders_visibly(markdown):
        markdown = None
        source = None

    truncated = False
    if markdown and len(markdown) > RELEASE_NOTES_MAX_CHARS:
        markdown = _close_open_fence(markdown[:RELEASE_NOTES_MAX_CHARS].rstrip())
        truncated = True

    return {
        "version": version,
        "markdown": markdown or None,
        "heading": heading,
        # False means no notes for this exact version; the UI links out.
        "matched": bool(markdown),
        "truncated": truncated,
        "source": source,
        "release_notes_url": RELEASE_NOTES_URL,
        "error": error,
    }
