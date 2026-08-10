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

# CommonMark requires a space, tab or line end after the hashes: a non-breaking
# space copied from rich text renders as text, not a heading, but a bare `##` is
# an empty heading and still ends the release above.
_HEADING_PATTERN = re.compile(r"^ {0,3}##(?:[ \t]+(?P<title>.*?))?[ \t]*$")
_FENCE_PATTERN = re.compile(r"^ {0,3}(?P<marker>`{3,}|~{3,})(?P<rest>.*)$")
# CommonMark type 1 HTML blocks: contents are literal until a closing tag,
# which the spec says need not be the one that opened the block.
_RAW_HTML_OPEN = re.compile(r"^ {0,3}<(pre|script|style|textarea)(?=[\s>]|$)", re.IGNORECASE)
_RAW_HTML_CLOSE = re.compile(r"</(pre|script|style|textarea)\s*>", re.IGNORECASE)
# Types 3 to 5 (processing instructions, declarations, CDATA) are literal too,
# each ending on its own delimiter. Comments open mid-line, so are separate.
_RAW_BLOCKS = (
    (_RAW_HTML_OPEN, _RAW_HTML_CLOSE),
    (re.compile(r"^ {0,3}<\?"), re.compile(r"\?>")),
    (re.compile(r"^ {0,3}<!\[CDATA\["), re.compile(r"\]\]>")),
    # A declaration needs an uppercase letter, so `<!note` stays ordinary text.
    (re.compile(r"^ {0,3}<![A-Z]"), re.compile(r">")),
)
# Type 6 blocks run to the next blank line, so `<details>` only holds Markdown
# once a blank line has closed the block. Open and close tags both start one.
_HTML_BLOCK_OPEN = re.compile(r"^ {0,3}</?([a-zA-Z][a-zA-Z0-9-]*)(?=[\s/>]|$)")
# Blocks that break into an open paragraph, so none is open after them and one
# they are written below is closed rather than continued.
_INTERRUPTS = re.compile(
    r"^ {0,3}(?:#{1,6}([ \t]|$)|(?:\*[ \t]*){3,}$|(?:-[ \t]*){3,}$|(?:_[ \t]*){3,}$)"
)
# A definition is a block of its own but may not interrupt a paragraph, so it
# ends the one above it only when there is none to continue.
_LINK_DEFINITION = re.compile(r"^ {0,3}\[(?:[^\[\]\\]|\\.)+\]:")
# Blocks that are not paragraph text, so a following underline is not setext.
_PARAGRAPH_TEXT = re.compile(r"^ {0,3}(?![-*+>]([ \t]|$)|\d{1,9}[.)]([ \t]|$))\S")
# A line of = or - under a paragraph line makes that line a heading.
_SETEXT_UNDERLINE = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
# A quoted paragraph continues on unmarked lines, which belong to the quote.
_BLOCK_QUOTE = re.compile(r"^ {0,3}>")
_QUOTE_MARKER = re.compile(r"^ {0,3}>[ \t]?")
# A heading at an item's content column belongs to that item, not the document.
# The marker needs whitespace after it, so `2.0` is a version, not an item.
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
# Type 7: any other complete tag alone on a line. It cannot interrupt a
# paragraph, so it only counts after a break.
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
# Stands in for a line the renderer hides. `#` is a block of its own, so list
# tracking reads it like a comment: never a marker, never a lazy continuation.
_HIDDEN_BLOCK = "#"
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


def _markdown_lines(text: str) -> list[str]:
    """``text`` split the way CommonMark ends lines.

    str.splitlines also breaks on U+2028, U+2029, NEL, vertical tab and form
    feed, none of which end a line in Markdown. A separator sitting in prose
    before "## 9.9.9" would otherwise index a release the renderer never shows
    and truncate the notes above it.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


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
    # Content column of the list item the open block belongs to, 0 at document
    # level. A fence and an HTML block are scoped to their container, so the
    # item's end closes them. Only one of the three is ever open.
    block_column = 0
    in_comment = False
    in_raw_html: int | None = None
    in_html_block = False
    after_paragraph = False
    paragraph: list[str] = []
    in_quote = False
    quoted = False
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

    for line in _markdown_lines(text):
        # The line as list tracking sees it: blank wherever nothing renders.
        structural = ""
        opened_block = False
        in_block = open_fence is not None or in_html_block or in_raw_html is not None or in_comment
        # A fence, comment or HTML block inside a list item runs only to the end
        # of that item, so a line dedented out of the item closes both. Lazy
        # continuation reaches into none of them. A raw block or comment inside an
        # item also ends on a blank line: the item takes the break, so what
        # follows is a block of the item's own.
        leaves = (
            _indent_width(line) < block_column
            if line.strip()
            else in_raw_html is not None or in_comment
        )
        if in_block and block_column and leaves:
            open_fence = None
            in_html_block = False
            in_raw_html = None
            in_comment = False
            block_column = 0
            # The paragraph the line could have continued is block content, so
            # it closes the item rather than reading as more of it.
            after_paragraph = False
        # A fence written as a list item's first content opens inside that item, so
        # an opener is read past a marker on the same line. Only an opener: fenced
        # content is literal and a closer carries no marker.
        fence_line = line if open_fence else _item_content(line, after_paragraph)
        # Raw HTML first: its contents are literal, so a fence in it is not one.
        if in_raw_html is not None:
            visible, in_raw_html = _strip_raw_html(line, in_raw_html)
        elif in_html_block:
            # A blank line is the only thing that ends a type 6 block.
            in_html_block = line.strip() != ""
            visible = ""
        elif (fence := _FENCE_PATTERN.match(fence_line)) and not in_comment:
            was_open = open_fence
            open_fence = _next_fence_state(open_fence, fence.group("marker"), fence.group("rest"))
            opened_block = was_open is None and open_fence is not None
            # Hidden from heading matching, but its indent still closes items.
            visible = ""
            structural = line
        elif open_fence:
            visible = ""
        else:
            # A block already open owns this line, so it is content rather than a
            # block written at the column it happens to start in.
            hidden = in_comment or in_raw_html is not None
            # A comment is an HTML block too, so one written as a list item's first
            # content opens inside it exactly as a fence does: the opener is read
            # past a marker on the same line.
            block_open = (
                not in_comment
                and _COMMENT_BLOCK_OPEN.match(_item_content(line, after_paragraph)) is not None
            )
            # Commented-out sections are not rendered, so they are not releases.
            visible, in_comment = _strip_comments(line, in_comment, block_open)
            # An HTML block written as a list item's first content opens inside
            # that item, as a fence does, so an opener is read past a marker on the
            # same line. The marker stays, so its item is still tracked. A comment
            # blanks its own line, so that line is read as written: the block
            # renders as nothing, but the item it is content of still opens.
            source = line if block_open else visible
            content = _item_content(source, after_paragraph)
            marker = source[: len(source) - len(content)]
            # Nor is anything inside a raw HTML block such as <pre>.
            stripped, in_raw_html = _strip_raw_html(content, in_raw_html)
            opened_block = in_raw_html is not None or (block_open and in_comment)
            # Taken before the opener is hidden: it renders as nothing, but its
            # indent still closes a list item it sits left of, and a marker on its
            # line still opens one. A comment or raw block keeps only those, since
            # the text it hides is not Markdown and must open no list.
            if block_open or stripped != content:
                if not hidden:
                    structural = _hidden_structure(line, marker)
                visible = ""
            else:
                visible = marker + stripped
                if visible.strip():
                    structural = visible
                elif not hidden:
                    structural = _hidden_structure(line)
                if stripped and _opens_html_block(stripped, after_paragraph):
                    in_html_block = True
                    opened_block = True
                    visible = ""
        # A `##` inside a fenced block is sample markdown, not a real heading.
        match = _HEADING_PATTERN.match(visible) if visible else None
        # `1.0` over a line of dashes is the same heading written setext style.
        setext = (
            after_paragraph
            and match is None
            and paragraph != []
            and _SETEXT_UNDERLINE.match(visible) is not None
            and (visible.strip()[:1] == "-")
            # Never a boundary inside a list item: dedented the dashes are a
            # thematic break, and at the content column the heading is nested.
            and not lists.columns
        )
        if setext:
            if version is not None:
                # The whole paragraph is the heading, read as body on arrival.
                del body[len(body) - len(paragraph) :]
            flush()
            # A wrapped heading keeps every line, so token one is the version.
            heading = "\n".join(paragraph)
            version = _version_from_heading(heading)
            body = []
            paragraph = []
            after_paragraph = False
            continue
        # A dashed underline is not a list marker, so track lists after setext.
        lazy_marker = _lazy_marker(structural, lists, after_paragraph, quoted)
        lists = _open_lists(structural, lists, after_paragraph, quoted)
        # Taken after the opening line closed the items it is dedented out of,
        # so the block belongs to the item it is really written inside.
        if opened_block:
            block_column = lists.columns[-1] if lists.columns else 0
        elif open_fence is None and not in_html_block and in_raw_html is None and not in_comment:
            block_column = 0
        # At an open item's content column a heading is nested, not a boundary.
        if lists.columns and _indent_width(visible) >= lists.columns[0]:
            match = None
        # The line at its own nesting level: past the container's indentation
        # and past a marker on the same line, so `- ## 2.0` reads as a heading.
        column = lists.columns[-1] if lists.columns else 0
        content = _strip_indent(visible, column)
        if (item := _LIST_ITEM.match(content)) is not None:
            content = content[item.end() :]
        # Only ordinary text continues a paragraph. Indented code counts four
        # spaces past the container, so an item's own indent does not count.
        indented_code = not after_paragraph and _indent_width(visible) - column >= 4
        # An underline ends the paragraph it underlines, so it needs one open in
        # its own container: the quote above owns its own, and a row left of an
        # open item is lazy text of the item's paragraph. Three dashes are a
        # thematic break either way, which `_INTERRUPTS` already ends on.
        underline = (
            _SETEXT_UNDERLINE.match(visible) is not None
            and after_paragraph
            and not quoted
            and _indent_width(visible) >= column
        )
        after_paragraph = (
            # Read inside its container, so an empty item and a fence written as an
            # item's own content leave no paragraph open below them. A marker the
            # paragraph above swallows is its text, not an item.
            (bool(content.strip()) or lazy_marker)
            and match is None
            and _HEADING_PATTERN.match(content) is None
            and _FENCE_PATTERN.match(content) is None
            and not indented_code
            and _INTERRUPTS.match(visible) is None
            and (after_paragraph or _LINK_DEFINITION.match(visible) is None)
            and not underline
        )
        # A quote's paragraph runs on over plain text and owns every line of it.
        # An empty quote holds none, so the line below starts the document's.
        flush_left = visible.lstrip(" \t")
        quote_line = _BLOCK_QUOTE.match(visible) is not None
        in_quote = (
            _may_be_lazy(_quote_content(visible))
            if quote_line
            else in_quote and _continues_paragraph(visible, column)
        )
        if quote_line:
            # The only paragraph a quote line leaves open is the quote's own,
            # and a quote holding a heading or nothing at all leaves none.
            after_paragraph = in_quote
        # Whose paragraph the line below would continue. A quote owns the one its
        # own lines hold, so a marker outside the quote is a block of its own
        # rather than more of the text above it.
        quoted = quote_line or in_quote
        # The lines a later underline turns into one heading. A paragraph opens
        # only on plain text and then runs on until something interrupts it.
        continues = (
            not _interrupts_paragraph(flush_left)
            if paragraph
            else _PARAGRAPH_TEXT.match(flush_left) is not None
        )
        # A paragraph inside an open item is that item's, and only one written
        # at document level can be the heading a later underline makes of it.
        if after_paragraph and not in_quote and not lists.columns and continues:
            paragraph = [*paragraph, visible.strip()]
        else:
            paragraph = []
        if match is None:
            if version is not None:
                body.append(line)
            continue

        flush()
        # An empty heading has no title, so it ends the release above without
        # indexing one: `_version_from_heading` finds no version and `flush` skips.
        heading = match.group("title") or ""
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
        # An exact heading wins, so `## 1.0` is never shadowed by `## 1.0.0`.
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

    # Nothing matched: the bundled copy cannot know a version newer than the
    # install, so report a remote failure and let the UI offer a retry.
    return _notes_response(version = version, error = remote.error)


def get_remote_changelog(refresh: bool = False) -> ChangelogSource:
    """Fetch CHANGELOG.md from the repo using a small in-process TTL cache."""
    global _remote_cache, _remote_fetching

    if refresh:
        # Only a cached failure is dropped, so retries cannot hammer the remote.
        with _cache_condition:
            if _remote_cache and _remote_cache.source.text is None:
                _remote_cache = None

    # A caller waits for an in-flight fetch only as long as it may take, then
    # answers locally rather than holding a worker behind a stalled upstream.
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
            _remote_cache = _ChangelogCacheEntry(source = source, expires_at = time.monotonic() + ttl)
        return source
    finally:
        # Released here, not on the Exception path: stranding the single-flight
        # flag on BaseException makes every later caller wait out the deadline.
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
            # Or a compressing proxy hands back bytes we would decode as notes.
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
                # The socket timeout is per operation, so re-cap it each read.
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

    # changelog.py -> utils -> backend -> studio -> repo root. Repo root first
    # so a checkout's editable file beats the snapshot packaging writes into
    # studio/. Installed, those outer levels are site-packages, hence the marker.
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
    # Collect the runs once: rescanning per opener is quadratic on a line of
    # distinct unmatched runs, and notes are reparsed on every request.
    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(line):
        if line[index] != "`" or _is_escaped(line, index):
            index += 1
            continue
        ticks = _run_length(line, index)
        runs.append((index, ticks))
        index += ticks

    # A run closes only on a later run of its length, so one cursor per length.
    by_length: dict[int, list[int]] = {}
    for position, (_, ticks) in enumerate(runs):
        by_length.setdefault(ticks, []).append(position)

    spans: list[tuple[int, int]] = []
    cursors: dict[int, int] = {}
    current = 0
    while current < len(runs):
        start, ticks = runs[current]
        same = by_length[ticks]
        cursor = cursors.get(ticks, 0)
        while cursor < len(same) and same[cursor] <= current:
            cursor += 1
        cursors[ticks] = cursor
        if cursor >= len(same):
            # Nothing closes this run, so it is literal text.
            current += 1
            continue
        closer = same[cursor]
        cursors[ticks] = cursor + 1
        spans.append((start, runs[closer][0] + ticks))
        current = closer + 1
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


def _strip_comments(line: str, in_comment: bool, block_open: bool) -> tuple[str, bool]:
    """Return the line with HTML-comment spans removed, and the trailing state.

    Only a comment that starts a line opens a block and hides the lines below
    it. One written mid-sentence is inline HTML: it hides the rest of its own
    line at most, so a note mentioning `<!--` cannot swallow later releases.
    Delimiters inside inline code are literal and hide nothing.

    "Starts a line" is read inside the container, so `block_open` is decided by
    the caller from the item's content rather than from the raw line.
    """
    if in_comment:
        close = line.find(_COMMENT_CLOSE)
        # The closing line belongs to the block, tail included.
        return ("", False) if close != -1 else ("", True)

    if block_open:
        # `<!-->` and `<!--->` are complete comments, so the closer may overlap
        # the opener; searching past it would swallow every later release.
        return ("", _COMMENT_CLOSE not in line)

    visible: list[str] = []
    index = 0
    spans = _code_span_ranges(line)
    # Spans are ordered and disjoint and each opener sits at or past the one
    # before, so the search resumes rather than restarts: restarting per opener is
    # quadratic, and a long line of code spans is reparsed on every request.
    cursor = 0
    while index < len(line):
        opening = line.find(_COMMENT_OPEN, index)
        if opening == -1:
            visible.append(line[index:])
            break

        while cursor < len(spans) and spans[cursor][1] <= opening:
            cursor += 1
        if cursor < len(spans) and spans[cursor][0] <= opening:
            visible.append(line[index : spans[cursor][1]])
            index = spans[cursor][1]
            continue

        visible.append(line[index:opening])
        close = line.find(_COMMENT_CLOSE, opening + len(_COMMENT_OPEN))
        if close == -1:
            # Unterminated inline comment: it hides this line and no more.
            break
        index = close + len(_COMMENT_CLOSE)
    return "".join(visible), False


def _hidden_structure(line: str, marker: str = "") -> str:
    """`line` as list tracking sees it once the renderer hides its text.

    A comment or a raw HTML block renders nothing, but it is still a block
    written at its own column, so it closes the items it sits to the left of.
    Only the indentation survives: what is inside the block is not Markdown and
    must not open a list of its own. `marker` is the part of the line that opens
    a list item the block is the content of, which survives with it."""
    if marker:
        return marker + _HIDDEN_BLOCK
    if not line.strip():
        return ""
    return line[: len(line) - len(line.lstrip(" \t"))] + _HIDDEN_BLOCK


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


def _item_content(line: str, after_paragraph: bool) -> str:
    """`line` read from the content column of a list item that opens on it.

    A block written as an item's first content sits inside that item, so
    ``- ```` opens a fence even though its marker is not within three columns of
    the container. The padding is capped the way `_open_lists` caps it, or
    ``-     ```` would read as a fence rather than the indented code it is. A
    marker the paragraph above swallows opens no item, so its line is returned
    whole, as is one four columns past its container. Ported to the frontend as
    `itemContent` in markdown-list-columns.ts."""
    if _indent_width(line) >= 4 or (after_paragraph and not _interrupts_paragraph(line)):
        return line
    item = None if _THEMATIC_BREAK.match(line) else _LIST_ITEM.match(line)
    if item is None:
        return line
    padding = _indent_width(item.group("space"))
    # Over-indented content starts one column past the marker; the rest of the
    # padding is the content's own indentation.
    over = padding - 1 if padding > _MAX_ITEM_PADDING else 0
    return " " * over + line[item.end() :]


def _quote_content(line: str) -> str:
    """What a blockquote line holds, with its markers stripped."""
    while (marker := _QUOTE_MARKER.match(line)) is not None:
        line = line[marker.end() :]
    return line


def _may_be_lazy(line: str) -> bool:
    """Whether `line` can continue a paragraph it is indented out of.

    Only plain text can: a heading, a fence, a break or an HTML block starts a
    block of its own, which closes the item instead. An underline is not one of
    them: it may never be lazy, so `===` written left of an open item is read as
    more of the item's paragraph. Nor is a definition, which is a block of its
    own but may not interrupt a paragraph. A row of dashes still closes the
    item, as `_INTERRUPTS` reads three or more as the thematic break they are."""
    return (
        _PARAGRAPH_TEXT.match(line) is not None
        and _INTERRUPTS.match(line) is None
        and _FENCE_PATTERN.match(line) is None
        # Types 1 to 6 interrupt a paragraph, so a `<div>` left of an open item
        # closes it. Type 7 cannot, and is deliberately excluded.
        and not _opens_html_block(line, True)
    )


def _continues_paragraph(line: str, column: int) -> bool:
    """Whether `line` reads as more of a paragraph open in its container.

    Measured from `column`, where that container's content starts: four columns
    past it the line is an indented code block, which may not interrupt a
    paragraph, so indentation alone never closes the one above it."""
    inner = _strip_indent(line, column)
    return _indent_width(inner) >= 4 or _may_be_lazy(inner)


def _close_dedented(
    columns: tuple[int, ...], line: str, indent: int, after_paragraph: bool
) -> tuple[int, ...]:
    """`columns` with every item `line` is written to the left of closed.

    Read inside the container the item sits in, not from the margin: a line that
    only looks indented there is lazy text of the item's paragraph, which leaves
    the item open rather than closing it."""
    while columns and indent < columns[-1]:
        outer = columns[-2] if len(columns) > 1 else 0
        if after_paragraph and _continues_paragraph(line, outer):
            break
        columns = columns[:-1]
    return columns


def _lazy_marker(line: str, state: _ListState, after_paragraph: bool, quoted: bool) -> bool:
    """Whether a marker-shaped `line` is really text of the paragraph above it.

    Only a marker inside the paragraph's own item interrupts it; one to the left
    closes that item and opens a sibling. A quote owns the paragraph its lines
    hold, so a marker written outside the quote opens a list of its own."""
    item = None if _THEMATIC_BREAK.match(line) else _LIST_ITEM.match(line)
    columns = state.columns
    return (
        item is not None
        and after_paragraph
        and not quoted
        and (not columns or _indent_width(line) >= columns[-1])
        and not _interrupts_paragraph(line)
    )


def _open_lists(
    line: str,
    state: _ListState,
    after_paragraph: bool,
    quoted: bool = False,
) -> _ListState:
    """The list items still open after `line`.

    A dedented line closes an item, unless it is a lazy paragraph continuation.
    A new marker nests under a deeper column and replaces a sibling. `quoted`
    marks a paragraph the blockquote above owns: a marker written outside the
    quote is not text of it, so it opens a list of its own.
    """
    columns = state.columns
    if not line.strip():
        # A blank line leaves the list open, unless the item is still empty: an
        # item may begin with one blank line, and later content is outside it.
        return _ListState(columns[:-1] if state.empty_item else columns)
    indent = _indent_width(line)
    item = None if _THEMATIC_BREAK.match(line) else _LIST_ITEM.match(line)
    empty = item is not None and not line[item.end() :].strip()
    if _lazy_marker(line, state, after_paragraph, quoted):
        # A lazy continuation or an underline, so the open items are untouched.
        return state
    columns = _close_dedented(columns, line, indent, after_paragraph)
    # Four columns past its container the marker is an indented code block, or
    # lazy text of the paragraph above it, so it opens no list of its own.
    if item is None or indent - (columns[-1] if columns else 0) >= 4:
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
    for line in _markdown_lines(markdown):
        fence = _FENCE_PATTERN.match(line)
        if fence:
            open_fence = _next_fence_state(open_fence, fence.group("marker"), fence.group("rest"))
    return f"{markdown}\n{open_fence}" if open_fence else markdown


def _renders_visibly(markdown: str) -> bool:
    """Whether a section body renders anything at all."""
    in_comment = False
    for line in _markdown_lines(markdown):
        opens_raw = any(opener.match(line) for opener, _ in _RAW_BLOCKS)
        if not in_comment and (_FENCE_PATTERN.match(line) or opens_raw):
            # A code block or raw HTML block renders even when it is empty.
            return True
        # No containers are tracked here, so the opener is read at the margin. The
        # answer does not turn on it: an item renders its marker whatever the block
        # inside hides, so a commented-out item renders something either way.
        visible, in_comment = _strip_comments(
            line, in_comment, _COMMENT_BLOCK_OPEN.match(line) is not None
        )
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
    # A section that renders as nothing counts as unpublished, not as empty.
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
