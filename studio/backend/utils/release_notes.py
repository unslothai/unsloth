# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Release notes for the update popup, taken from the newest GitHub release.

Only the announcement is shown: the install instructions, the generated "What's
Changed" and "New Contributors" lists, the "Full Changelog" line and the
appended build provenance are stripped, wherever they were written.

The fetch is lazy, cached, and skipped when update checks are off.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from packaging.version import InvalidVersion, Version

from .update_status import DISABLE_ENV_VAR, RELEASE_NOTES_URL

# A release entry carries its whole body, about 40 KiB lately, so the endpoint's
# maximum of 100 would be near 4 MiB, twice the cap below, and the fetch would
# fail outright. 30 is about 1.2 MiB, and the newest release is at the top.
RELEASES_API_URL = "https://api.github.com/repos/unslothai/unsloth/releases?per_page=30"
RELEASES_URL_ENV_VAR = "UNSLOTH_RELEASES_URL"
RELEASES_TIMEOUT_SECONDS = 3
RELEASES_MAX_BYTES = 2 * 1024 * 1024
_RELEASES_CHUNK_BYTES = 64 * 1024
_RELEASES_MIN_READ_SECONDS = 0.05
RELEASES_SUCCESS_TTL_SECONDS = 30 * 60
RELEASES_FAILURE_TTL_SECONDS = 5 * 60
# Unauthenticated callers get 60 requests an hour per IP, so an address that has
# spent them backs off rather than retrying every 5 minutes. Used when the
# response carries no reset to wait for.
RELEASES_RATE_LIMITED_TTL_SECONDS = 15 * 60
# GitHub says not to request again before X-RateLimit-Reset, so its reset wins
# over the back-off above. The window is an hour, so a skewed or proxied header
# is held to that ceiling rather than trusted outright.
RELEASES_RATE_LIMIT_MAX_SECONDS = 60 * 60
RELEASE_NOTES_MAX_CHARS = 20_000

# The repo also publishes llama.cpp prebuilts (`b8475`), legacy month tags
# (`February-2026`) and `desktop-v...` drafts. Only an Unsloth version tag is an
# announcement the popup should show.
_RELEASE_TAG_PATTERN = re.compile(r"^v\d+(?:\.\d+)+")

# CommonMark needs a space, tab or line end after the hashes: a non-breaking
# space is text, a bare `##` an empty heading that still ends the section above.
_HEADING_PATTERN = re.compile(r"^ {0,3}(?P<hashes>#{1,6})(?:[ \t]+(?P<title>.*?))?[ \t]*$")
# A heading may close with its own run of hashes, which is not part of the title.
_CLOSING_SEQUENCE = re.compile(r"(?:^|[ \t])#+[ \t]*$")
# Inline markup the title carries but the words do not.
_TITLE_MARKUP = re.compile(r"[`*_~]|\[|\]\([^)]*\)|<[^>]*>")
_FULL_CHANGELOG_LINE = re.compile(r"^ {0,3}\*{0,2}full changelog\*{0,2}\s*:", re.IGNORECASE)
_FENCE_PATTERN = re.compile(r"^ {0,3}(?P<marker>`{3,}|~{3,})(?P<rest>.*)$")
# CommonMark type 1 HTML blocks: literal until a closing tag, which the spec
# says need not be the one that opened the block.
_RAW_HTML_OPEN = re.compile(r"^ {0,3}<(pre|script|style|textarea)(?=[\s>]|$)", re.IGNORECASE)
_RAW_HTML_CLOSE = re.compile(r"</(pre|script|style|textarea)\s*>", re.IGNORECASE)
# Types 3 to 5 are literal too, each ending on its own delimiter. Comments open
# mid-line, so they are handled separately.
_RAW_BLOCKS = (
    (_RAW_HTML_OPEN, _RAW_HTML_CLOSE),
    (re.compile(r"^ {0,3}<\?"), re.compile(r"\?>")),
    (re.compile(r"^ {0,3}<!\[CDATA\["), re.compile(r"\]\]>")),
    # A declaration needs an uppercase letter, so `<!note` stays ordinary text.
    (re.compile(r"^ {0,3}<![A-Z]"), re.compile(r">")),
)
# Type 6 runs to the next blank line, so `<details>` holds Markdown only after
# one. Open and close tags both start a block.
_HTML_BLOCK_OPEN = re.compile(r"^ {0,3}</?([a-zA-Z][a-zA-Z0-9-]*)(?=[\s/>]|$)")
# Blocks that break into an open paragraph, closing it rather than continuing it.
_INTERRUPTS = re.compile(
    r"^ {0,3}(?:#{1,6}([ \t]|$)|(?:\*[ \t]*){3,}$|(?:-[ \t]*){3,}$|(?:_[ \t]*){3,}$)"
)
# A definition is a block of its own but may not interrupt a paragraph.
_LINK_DEFINITION = re.compile(r"^ {0,3}\[(?:[^\[\]\\]|\\.)+\]:")
# Blocks that are not paragraph text, so a following underline is not setext.
_PARAGRAPH_TEXT = re.compile(r"^ {0,3}(?![-*+>]([ \t]|$)|\d{1,9}[.)]([ \t]|$))\S")
# A line of = or - under a paragraph line makes that line a heading.
_SETEXT_UNDERLINE = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
# A quoted paragraph continues on unmarked lines, which belong to the quote.
_BLOCK_QUOTE = re.compile(r"^ {0,3}>")
_QUOTE_MARKER = re.compile(r"^ {0,3}>[ \t]?")
# A heading at an item's content column belongs to that item. The marker needs
# whitespace after it, so `2.0` is a version, not an item.
_LIST_ITEM = re.compile(r"^[ \t]*(?P<marker>[-*+]|\d{1,9}[.)])(?P<space>[ \t]+|$)")
_THEMATIC_BREAK = re.compile(r"^ {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$")
# Past this the content after a marker is indented code, so the item's content
# starts one column past the marker instead.
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
# Type 7: any other complete tag alone on a line; it cannot interrupt a paragraph.
_HTML_ATTRIBUTE = (
    r"""(?:\s+[a-zA-Z_:][a-zA-Z0-9_.:-]*(?:\s*=\s*(?:[^\s"'=<>`]+|'[^']*'|"[^"]*"))?)"""
)
_HTML_TAG_ONLY_LINE = re.compile(
    rf"^ {{0,3}}(?:<[a-zA-Z][a-zA-Z0-9-]*{_HTML_ATTRIBUTE}*\s*/?>|</[a-zA-Z][a-zA-Z0-9-]*\s*>)\s*$"
)
_COMMENT_BLOCK_OPEN = re.compile(r"^ {0,3}<!--")
_COMMENT_OPEN = "<!--"
_COMMENT_CLOSE = "-->"
# Stands in for a line the renderer hides: `#` is a block of its own, so list
# tracking never reads it as a marker or a lazy continuation.
_HIDDEN_BLOCK = "#"
_SAFE_VERSION_PATTERN = re.compile(r"^[0-9A-Za-z][0-9A-Za-z.!+-]{0,63}$")

# Sections GitHub or the release workflow generates. Matched on the normalised
# title, so `## What's Changed in Unsloth-Zoo` and a curly apostrophe still
# match. Narrow rather than a substring sweep: "What changed in Gemma 4" is an
# announcement one apostrophe away.
_GENERATED_TITLES = frozenset({"what's changed", "whats changed", "new contributors"})
_GENERATED_PREFIXES = ("what's changed in ", "whats changed in ")
_GENERATED_SUFFIXES = ("zoo changes", "notebooks changes", "changelog")
# The install block, worded differently in almost every release ("Updating /
# installing Unsloth", "To update Unsloth", "Update Unsloth via `pip install`").
# Naming Unsloth or Unsloth separates those from "Updating models is now 2x
# faster", which is a change and not instructions.
_UPGRADE_PREFIXES = ("update", "updating", "to update", "how to update")
_UPGRADE_SUBJECTS = ("unsloth", "studio")
_UPGRADE_TITLES = frozenset({"update instructions", "install instructions"})
_PROVENANCE = "build provenance"
# Platform headings the install block splits its commands across, written as its
# siblings as often as its children, so level alone does not end the block. The
# separator is a slash or a comma, however it is spaced.
_PLATFORM_SEPARATOR = re.compile(r"\s*[/,]\s*")
_PLATFORM_TITLES = frozenset(
    {
        "macos",
        "mac",
        "macos, linux, wsl",
        "macos linux wsl",
        "linux",
        "windows",
        "wsl",
        "docker",
        "pip",
    }
)


@dataclass(frozen = True)
class _ListState:
    """The open list items, innermost last, by the column their content starts."""

    columns: tuple[int, ...] = ()
    # True while the innermost item has had no content since its marker.
    empty_item: bool = False


@dataclass(frozen = True)
class Release:
    """One published GitHub release."""

    tag: str
    name: str
    body: str
    html_url: str
    published_at: str


@dataclass(frozen = True)
class ReleaseSource:
    release: Release | None
    source: str | None
    error: str | None = None


@dataclass
class _ReleaseCacheEntry:
    source: ReleaseSource
    expires_at: float


_cache_condition = threading.Condition()
_remote_cache: _ReleaseCacheEntry | None = None
_remote_fetching = False
# Kept across TTL expiry so a 304 can answer without refetching the body.
_remote_etag: str | None = None
_remote_last_good: ReleaseSource | None = None
# Epoch seconds the rate limit resets at, while it is exhausted.
_rate_limited_until: float = 0.0


def reset_release_notes_cache() -> None:
    """Clear the in-process release cache. Intended for tests."""
    global _remote_cache, _remote_fetching, _remote_etag, _remote_last_good, _rate_limited_until
    with _cache_condition:
        _remote_cache = None
        _remote_fetching = False
        _remote_etag = None
        _remote_last_good = None
        _rate_limited_until = 0.0
        _cache_condition.notify_all()


def is_supported_version_query(version: str) -> bool:
    """Whether `version` is shaped like a version the popup could be offering.

    The version is echoed back rather than used to select a release, so the UI
    can drop a stale response. `latest`, `main` or a path is rejected outright."""
    candidate = version.strip()
    if not _SAFE_VERSION_PATTERN.match(candidate):
        return False
    return _parse_version(candidate) is not None


def _markdown_lines(text: str) -> list[str]:
    """``text`` split the way CommonMark ends lines.

    str.splitlines also breaks on U+2028, U+2029, NEL, vertical tab and form
    feed, none of which end a Markdown line: one in prose would otherwise turn
    the text after it into a heading and cut the announcement there.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


@dataclass(frozen = True)
class Text:
    """A line that is not a heading, as the scanner read it."""

    line: str
    # A document-level `**Full Changelog**: ...` line, which GitHub appends.
    is_full_changelog: bool = False
    # First line of a document-level paragraph: an install block is introduced
    # by one as often as by a heading, so this is where such a block can begin.
    opens_paragraph: bool = False


@dataclass(frozen = True)
class Heading:
    """A heading the renderer would show, at document level."""

    level: int
    title: str
    # The heading's own source lines, in order.
    lines: tuple[str, ...]
    # Lines already emitted as `Text` that turned out to be this heading. Only
    # setext retracts: its title is the paragraph above the underline.
    retract: int = 0


def strip_release_body(text: str) -> str:
    """The announcement in a release body, with the generated sections removed.

    Each section is excised where it stands rather than the body being truncated
    at the first one: maintainers write the install block second of twelve
    sections as often as last, so truncating there loses the rest.

    A section runs to the next heading at its own level or shallower, so its
    subheadings go with it. The install block is the exception twice over: its
    platform headings (`### Windows:`) are written as siblings as often as
    children, so they keep the drop open, and it is introduced by an ordinary
    paragraph as often as by a heading. A paragraph has no level, so the block
    it opens runs to the next heading that is not a platform heading.
    """
    kept: list[str] = []
    # Level of the boilerplate heading being dropped, and whether it was the
    # install block, whose platform siblings go with it.
    drop_level: int | None = None
    drop_upgrade = False
    # Set instead when a paragraph, which has no level to compare against,
    # opened the install block.
    drop_prose = False

    for event in scan_blocks(text):
        if isinstance(event, Text):
            if (
                drop_level is None
                and not drop_prose
                and event.opens_paragraph
                and _is_upgrade(_normalise_title(event.line))
            ):
                drop_prose = True
            if drop_level is None and not drop_prose and not event.is_full_changelog:
                kept.append(event.line)
            continue

        # A setext heading is the paragraph above the underline, already kept
        # line by line, so take it back. Inside a drop nothing was kept.
        if event.retract and drop_level is None and not drop_prose:
            del kept[len(kept) - event.retract :]

        title = _normalise_title(event.title)
        if drop_prose:
            # Platform headings belong to the block however they were nested;
            # any other heading is the announcement resuming.
            if _is_platform(title):
                continue
            drop_prose = False
        elif drop_level is not None:
            # A platform heading belongs to the install block however it was
            # nested, and a deeper heading is a subheading of the dropped one.
            if (drop_upgrade and event.level >= drop_level and _is_platform(title)) or (
                event.level > drop_level
            ):
                continue
            drop_level = None
            drop_upgrade = False

        # A bare "MacOS, Linux, WSL:" heading opens a block with no "Updating".
        upgrade = _is_upgrade(title) or _is_platform(title)
        if upgrade or _is_generated(title):
            drop_level = event.level
            drop_upgrade = upgrade
            continue
        kept.extend(event.lines)

    return "\n".join(kept).strip("\n")


def scan_blocks(text: str):
    """Walk `text` as CommonMark blocks, yielding `Text` and `Heading` events.

    Only headings the renderer would show are `Heading`: one inside a fence, a
    comment, a raw HTML block or a list item arrives as ordinary text.
    """
    # A Windows editor can leave a BOM on the first line, hiding a heading.
    text = text.lstrip("﻿")
    open_fence: str | None = None
    # Content column of the list item the open block belongs to, 0 at document
    # level: a fence or HTML block is scoped to its container, so the item's end
    # closes it. Only one of the three is ever open.
    block_column = 0
    in_comment = False
    in_raw_html: int | None = None
    in_html_block = False
    after_paragraph = False
    # The open paragraph a later underline turns into a heading: as the renderer
    # reads it, and as it was written.
    paragraph: list[str] = []
    paragraph_source: list[str] = []
    in_quote = False
    quoted = False
    lists = _ListState()

    for line in _markdown_lines(text):
        # The line as list tracking sees it: blank wherever nothing renders.
        structural = ""
        opened_block = False
        in_block = open_fence is not None or in_html_block or in_raw_html is not None or in_comment
        # A fence, comment or HTML block inside an item ends with the item, so a
        # dedented line closes both, and none of them takes a lazy continuation.
        # A raw block or comment there also ends on a blank line the item takes.
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
            # The paragraph it could have continued is block content.
            after_paragraph = False
        # A fence written as an item's first content opens inside it, so an opener
        # is read past a marker. Only an opener: fenced content is literal.
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
            # A block already open owns this line, so it is content.
            hidden = in_comment or in_raw_html is not None
            # A comment is an HTML block too, so it opens inside an item as a
            # fence does: the opener is read past a marker on the same line.
            block_open = (
                not in_comment
                and _COMMENT_BLOCK_OPEN.match(_item_content(line, after_paragraph)) is not None
            )
            # Commented-out sections are not rendered, so they are not releases.
            visible, in_comment = _strip_comments(line, in_comment, block_open)
            # The marker stays, so the item is still tracked. A comment blanks its
            # own line, so the raw line is used: the block renders as nothing, but
            # the item it is content of still opens.
            source = line if block_open else visible
            content = _item_content(source, after_paragraph)
            marker = source[: len(source) - len(content)]
            # Nor is anything inside a raw HTML block such as <pre>.
            stripped, in_raw_html = _strip_raw_html(content, in_raw_html)
            opened_block = in_raw_html is not None or (block_open and in_comment)
            # Taken before the opener is hidden: it renders nothing, but its indent
            # still closes items left of it and a marker on its line still opens
            # one. Only those survive; what it hides is not Markdown.
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
            # Never a boundary in a list item: dedented the dashes are a break,
            # at the content column the heading is nested.
            and not lists.columns
        )
        if setext:
            # The whole paragraph is the heading, already emitted line by line,
            # so the caller takes those lines back. Dashes mean level 2.
            yield Heading(
                level = 2,
                title = " ".join(paragraph),
                lines = (*paragraph_source, line),
                retract = len(paragraph_source),
            )
            paragraph = []
            paragraph_source = []
            after_paragraph = False
            continue
        # A dashed underline is not a list marker, so track lists after setext.
        lazy_marker = _lazy_marker(structural, lists, after_paragraph, quoted)
        lists = _open_lists(structural, lists, after_paragraph, quoted)
        # Taken after the opener closed the items it is dedented out of, so the
        # block belongs to the item it is really written inside.
        if opened_block:
            block_column = lists.columns[-1] if lists.columns else 0
        elif open_fence is None and not in_html_block and in_raw_html is None and not in_comment:
            block_column = 0
        # At an open item's content column a heading is nested, not a boundary.
        if lists.columns and _indent_width(visible) >= lists.columns[0]:
            match = None
        # The line at its own nesting level, past the container's indentation and
        # any marker, so `- ## 2.0` reads as a heading.
        column = lists.columns[-1] if lists.columns else 0
        content = _strip_indent(visible, column)
        if (item := _LIST_ITEM.match(content)) is not None:
            content = content[item.end() :]
        # Indented code is four spaces past the container, not past the margin.
        indented_code = not after_paragraph and _indent_width(visible) - column >= 4
        # An underline needs a paragraph open in its own container: a quote owns
        # its own, and a row left of an open item is lazy text of that item.
        underline = (
            _SETEXT_UNDERLINE.match(visible) is not None
            and after_paragraph
            and not quoted
            and _indent_width(visible) >= column
        )
        after_paragraph = (
            # Read inside its container, so an empty item or a fence written as its
            # content leaves no paragraph. A swallowed marker is text, not an item.
            (bool(content.strip()) or lazy_marker)
            and match is None
            and _HEADING_PATTERN.match(content) is None
            and _FENCE_PATTERN.match(content) is None
            and not indented_code
            and _INTERRUPTS.match(visible) is None
            and (after_paragraph or _LINK_DEFINITION.match(visible) is None)
            and not underline
        )
        # A quote owns every line of its paragraph; an empty quote holds none.
        flush_left = visible.lstrip(" \t")
        quote_line = _BLOCK_QUOTE.match(visible) is not None
        in_quote = (
            _may_be_lazy(_quote_content(visible))
            if quote_line
            else in_quote and _continues_paragraph(visible, column)
        )
        if quote_line:
            # A quote line leaves only the quote's own paragraph open, if any.
            after_paragraph = in_quote
        # Whose paragraph the line below continues: a quote owns its own, so a
        # marker outside the quote opens a block rather than continuing it.
        quoted = quote_line or in_quote
        # The lines a later underline turns into one heading: a paragraph opens
        # on plain text and runs until something interrupts it.
        continues = (
            not _interrupts_paragraph(flush_left)
            if paragraph
            else _PARAGRAPH_TEXT.match(flush_left) is not None
        )
        # Only a document-level paragraph can become a setext heading.
        in_paragraph = after_paragraph and not in_quote and not lists.columns and continues
        # Nothing is accumulated yet, so this line is the paragraph's first.
        opens_paragraph = in_paragraph and not paragraph
        if in_paragraph:
            paragraph = [*paragraph, visible.strip()]
            paragraph_source = [*paragraph_source, line]
        else:
            paragraph = []
            paragraph_source = []
        if match is None:
            yield Text(
                line = line,
                is_full_changelog = bool(
                    _FULL_CHANGELOG_LINE.match(visible) and not lists.columns and not quoted
                ),
                opens_paragraph = opens_paragraph,
            )
            continue

        # An empty heading ends the section above and, being neither generated nor
        # an upgrade block, starts a kept one.
        yield Heading(
            level = len(match.group("hashes")),
            title = match.group("title") or "",
            lines = (line,),
        )


def _normalise_title(title: str) -> str:
    """A heading title as its words, for comparison against the known ones."""
    title = _CLOSING_SEQUENCE.sub("", title)
    title = _TITLE_MARKUP.sub("", title)
    # A curly apostrophe is the same word as a straight one.
    title = title.replace("’", "'")
    return " ".join(title.split()).strip(" :.").lower()


def _is_generated(title: str) -> bool:
    """Whether `title` heads a section GitHub or the release workflow wrote."""
    return (
        title in _GENERATED_TITLES
        or title.startswith(_GENERATED_PREFIXES)
        or title.endswith(_GENERATED_SUFFIXES)
    )


def _is_upgrade(title: str) -> bool:
    """Whether `title` heads install instructions or the build provenance."""
    if _PROVENANCE in title or title in _UPGRADE_TITLES:
        return True
    if not title.startswith(_UPGRADE_PREFIXES):
        return False
    return any(subject in title for subject in _UPGRADE_SUBJECTS)


def _is_platform(title: str) -> bool:
    """Whether `title` is one of the install block's per-platform headings.

    The platforms are separated by a slash as readily as by a comma, spaced or
    not, so the separators are read as one thing.
    """
    return _PLATFORM_SEPARATOR.sub(", ", title).strip(" ,") in _PLATFORM_TITLES


def get_release_notes(version: str, refresh: bool = False) -> dict[str, Any]:
    """Return the newest release's notes for the update popup.

    `version` is echoed back, not used to select a release: the pip popup offers
    a PyPI version (`2026.8.7`) and releases are tagged with the Unsloth version
    (`v0.1.60-beta`), so no tag could match it.

    `refresh` retries a cached failure, so the UI's retry is not stuck behind
    the failure TTL once connectivity returns.
    """
    version = version.strip()
    if not is_supported_version_query(version):
        return _notes_response(version = version, error = "Unsupported version.")

    if os.environ.get(DISABLE_ENV_VAR) == "1":
        return _notes_response(version = version)

    remote = get_latest_release(refresh = refresh)
    if remote.release is None:
        return _notes_response(version = version, error = remote.error)

    return _notes_response(
        version = version,
        markdown = strip_release_body(remote.release.body),
        heading = remote.release.name or remote.release.tag,
        tag = remote.release.tag,
        html_url = remote.release.html_url,
        source = remote.source,
        error = remote.error,
    )


def get_latest_release(refresh: bool = False) -> ReleaseSource:
    """The newest published release, using a small in-process TTL cache."""
    global _remote_cache, _remote_fetching

    if refresh:
        # Only a cached failure is dropped, never a rate-limit lockout: retrying
        # into one spends nothing and only delays the reset.
        with _cache_condition:
            rate_limited = _rate_limited_until > time.time()
            if _remote_cache and _remote_cache.source.release is None and not rate_limited:
                _remote_cache = None

    # A caller waits only as long as a fetch may take, then answers without notes
    # rather than holding a worker behind a stalled upstream.
    deadline = time.monotonic() + RELEASES_TIMEOUT_SECONDS + 1
    while True:
        now = time.monotonic()
        with _cache_condition:
            if _remote_cache and _remote_cache.expires_at > now:
                return _remote_cache.source
            if not _remote_fetching:
                _remote_fetching = True
                break
            if now >= deadline:
                return ReleaseSource(
                    release = None,
                    source = None,
                    error = "Release notes are still loading.",
                )
            _cache_condition.wait(timeout = deadline - now)

    try:
        try:
            source, ttl = _fetch_latest_release()
        except Exception:
            source = ReleaseSource(
                release = None,
                source = None,
                error = "Could not fetch release notes.",
            )
            ttl = RELEASES_FAILURE_TTL_SECONDS

        with _cache_condition:
            _remote_cache = _ReleaseCacheEntry(source = source, expires_at = time.monotonic() + ttl)
        return source
    finally:
        # Released here, not on the Exception path: stranding the single-flight
        # flag on BaseException makes every later caller wait out the deadline.
        with _cache_condition:
            _remote_fetching = False
            _cache_condition.notify_all()


def _fetch_latest_release() -> tuple[ReleaseSource, float]:
    """Fetch and select the newest release, with the TTL to cache it for."""
    global _remote_etag, _remote_last_good, _rate_limited_until

    now = time.time()
    if _rate_limited_until > now:
        return (
            ReleaseSource(
                release = None,
                source = None,
                error = "GitHub is rate limiting release note requests.",
            ),
            _rate_limited_until - now,
        )

    url = os.environ.get(RELEASES_URL_ENV_VAR, "").strip() or RELEASES_API_URL
    if not url.startswith(("http://", "https://")):
        return (
            ReleaseSource(release = None, source = None, error = "Invalid releases URL."),
            RELEASES_FAILURE_TTL_SECONDS,
        )

    headers = {
        "User-Agent": "unsloth-studio-update-check",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        # Or a compressing proxy hands back bytes we would decode as notes.
        "Accept-Encoding": "identity",
    }
    if _remote_etag:
        headers["If-None-Match"] = _remote_etag

    request = urllib.request.Request(url, headers = headers)
    deadline = time.monotonic() + RELEASES_TIMEOUT_SECONDS
    try:
        with urllib.request.urlopen(request, timeout = RELEASES_TIMEOUT_SECONDS) as response:
            chunks: list[bytes] = []
            received = 0
            while received <= RELEASES_MAX_BYTES:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return (
                        ReleaseSource(
                            release = None,
                            source = None,
                            error = "Release notes took too long to load.",
                        ),
                        RELEASES_FAILURE_TTL_SECONDS,
                    )
                # The socket timeout is per operation, so re-cap it each read.
                _limit_read(response, remaining)
                chunk = response.read1(_RELEASES_CHUNK_BYTES)
                if not chunk:
                    break
                chunks.append(chunk)
                received += len(chunk)
            body = b"".join(chunks)
            etag = response.headers.get("ETag")
        if len(body) > RELEASES_MAX_BYTES:
            return (
                ReleaseSource(
                    release = None,
                    source = None,
                    error = "Release notes response was too large.",
                ),
                RELEASES_FAILURE_TTL_SECONDS,
            )
        payload = json.loads(body.decode("utf-8", errors = "replace"))
    except urllib.error.HTTPError as error:
        return _http_error_source(error)
    except TimeoutError:
        return (
            ReleaseSource(
                release = None,
                source = None,
                error = "Release notes took too long to load.",
            ),
            RELEASES_FAILURE_TTL_SECONDS,
        )
    except OSError:
        return (
            ReleaseSource(
                release = None,
                source = None,
                error = "Could not reach GitHub for release notes.",
            ),
            RELEASES_FAILURE_TTL_SECONDS,
        )
    except (UnicodeError, json.JSONDecodeError):
        return (
            ReleaseSource(release = None, source = None, error = "Malformed release data."),
            RELEASES_FAILURE_TTL_SECONDS,
        )

    release = select_release(payload)
    if release is None:
        return (
            ReleaseSource(release = None, source = None, error = "No published release found."),
            RELEASES_FAILURE_TTL_SECONDS,
        )
    source = ReleaseSource(release = release, source = "github")
    _remote_etag = etag
    _remote_last_good = source
    return source, RELEASES_SUCCESS_TTL_SECONDS


def _http_error_source(error: urllib.error.HTTPError) -> tuple[ReleaseSource, float]:
    """The answer and TTL for an HTTP status GitHub refused the request with."""
    global _rate_limited_until

    if error.code == 304 and _remote_last_good is not None:
        # Nothing changed, so the release already held still stands.
        return _remote_last_good, RELEASES_SUCCESS_TTL_SECONDS

    if error.code in (403, 429):
        now = time.time()
        # GitHub's order: Retry-After, which is how a secondary limit states its
        # wait, then the primary limit's reset, then a plain back-off. Every one
        # records a deadline, or Retry requests straight back into the limit.
        after = _epoch_header(error.headers.get("Retry-After"))
        reset = (
            _epoch_header(error.headers.get("X-RateLimit-Reset"))
            if error.headers.get("X-RateLimit-Remaining") == "0"
            else None
        )
        if after is not None:
            deadline = now + after
        elif reset is not None:
            deadline = reset
        else:
            deadline = now + RELEASES_RATE_LIMITED_TTL_SECONDS
        # The deadline itself is bounded, not just the first wait on it: the next
        # fetch answers from it, so capping only the TTL left a skewed header
        # parking the popup for as long as it liked.
        _rate_limited_until = min(deadline, now + RELEASES_RATE_LIMIT_MAX_SECONDS)
        ttl = max(_rate_limited_until - now, 0.0)
        return (
            ReleaseSource(
                release = None,
                source = None,
                error = "GitHub is rate limiting release note requests.",
            ),
            ttl,
        )

    return (
        ReleaseSource(release = None, source = None, error = "Could not fetch release notes."),
        RELEASES_FAILURE_TTL_SECONDS,
    )


def _epoch_header(value: str | None) -> float | None:
    try:
        return float((value or "").strip())
    except ValueError:
        return None


def select_release(payload: Any) -> Release | None:
    """The newest published release in a GitHub releases response.

    Ordered by publication, never by tag: `v0.1.60-beta` was published after
    `v0.1.527-beta`, so sorting the tags numerically picks the wrong one.
    """
    if not isinstance(payload, list):
        return None

    newest: Release | None = None
    for entry in payload:
        if not isinstance(entry, dict) or entry.get("draft"):
            continue
        tag = entry.get("tag_name")
        published = entry.get("published_at")
        if not isinstance(tag, str) or not isinstance(published, str) or not published:
            continue
        if not _RELEASE_TAG_PATTERN.match(tag):
            continue
        candidate = Release(
            tag = tag,
            name = entry.get("name") if isinstance(entry.get("name"), str) else "",
            body = entry.get("body") if isinstance(entry.get("body"), str) else "",
            html_url = entry.get("html_url") if isinstance(entry.get("html_url"), str) else "",
            published_at = published,
        )
        if newest is None or candidate.published_at > newest.published_at:
            newest = candidate
    return newest


def _limit_read(response: Any, remaining: float) -> None:
    """Cap the next socket read at the time left in the fetch budget."""
    sock = getattr(getattr(response, "fp", None), "raw", None)
    sock = getattr(sock, "_sock", None)
    if sock is None:
        return
    try:
        sock.settimeout(max(remaining, _RELEASES_MIN_READ_SECONDS))
    except OSError:
        pass


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
    # Collected once: rescanning per opener is quadratic on a line of distinct
    # unmatched runs, and notes are reparsed on every request.
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

    Only a comment that starts a line opens a block. One written mid-sentence is
    inline HTML and hides the rest of its own line at most, so a note mentioning
    `<!--` cannot swallow later releases, and a delimiter in inline code hides
    nothing. "Starts a line" is read inside the container, so the caller decides
    `block_open` from the item's content.
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
    # Spans are ordered and disjoint and openers only move forward, so the search
    # resumes rather than restarts: restarting per opener is quadratic, and a long
    # line of code spans is reparsed on every request.
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

    The block renders nothing but still sits at its own column, so it closes the
    items to its left. Only the indentation survives: its contents are not
    Markdown. `marker` is the part of the line opening the item it is content
    of, which survives with it."""
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

    A quote marker always can. A list item only when it has content, and an
    ordered one only when it starts at 1; anything else is paragraph text."""
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
    ``- ```` opens a fence. The padding is capped as `_open_lists` caps it, or
    ``-     ```` would read as a fence rather than the indented code it is. A
    swallowed marker opens no item, so its line is returned whole, as is one
    four columns past its container. Ported to the frontend as `itemContent`
    in markdown-list-columns.ts."""
    if _indent_width(line) >= 4 or (after_paragraph and not _interrupts_paragraph(line)):
        return line
    item = None if _THEMATIC_BREAK.match(line) else _LIST_ITEM.match(line)
    if item is None:
        return line
    padding = _indent_width(item.group("space"))
    # Over-indented content starts one column past the marker.
    over = padding - 1 if padding > _MAX_ITEM_PADDING else 0
    return " " * over + line[item.end() :]


def _quote_content(line: str) -> str:
    """What a blockquote line holds, with its markers stripped."""
    while (marker := _QUOTE_MARKER.match(line)) is not None:
        line = line[marker.end() :]
    return line


def _may_be_lazy(line: str) -> bool:
    """Whether `line` can continue a paragraph it is indented out of.

    Only plain text can: a heading, fence, break or HTML block closes the item
    instead. An underline may never be lazy, so `===` left of an open item is
    more of the item's paragraph, and nor may a definition interrupt one. Three
    dashes still close the item, as `_INTERRUPTS` reads them as a break."""
    return (
        _PARAGRAPH_TEXT.match(line) is not None
        and _INTERRUPTS.match(line) is None
        and _FENCE_PATTERN.match(line) is None
        # Types 1 to 6 interrupt a paragraph; type 7 cannot, so it is excluded.
        and not _opens_html_block(line, True)
    )


def _continues_paragraph(line: str, column: int) -> bool:
    """Whether `line` reads as more of a paragraph open in its container.

    Measured from `column`, where the container's content starts: four columns
    past it the line is indented code, which may not interrupt a paragraph, so
    indentation alone never closes the one above it."""
    inner = _strip_indent(line, column)
    return _indent_width(inner) >= 4 or _may_be_lazy(inner)


def _close_dedented(
    columns: tuple[int, ...], line: str, indent: int, after_paragraph: bool
) -> tuple[int, ...]:
    """`columns` with every item `line` is written to the left of closed.

    Read inside the container, not from the margin: lazy text of the item's own
    paragraph leaves the item open rather than closing it."""
    while columns and indent < columns[-1]:
        outer = columns[-2] if len(columns) > 1 else 0
        if after_paragraph and _continues_paragraph(line, outer):
            break
        columns = columns[:-1]
    return columns


def _lazy_marker(line: str, state: _ListState, after_paragraph: bool, quoted: bool) -> bool:
    """Whether a marker-shaped `line` is really text of the paragraph above it.

    Only a marker inside the paragraph's own item interrupts it; one to the left
    closes that item. A quote owns its paragraph, so a marker written outside
    the quote opens a list of its own."""
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

    A dedented line closes an item unless it is a lazy paragraph continuation.
    A new marker nests deeper and replaces a sibling. `quoted` marks a paragraph
    the quote above owns, so a marker outside the quote opens a list of its own.
    """
    columns = state.columns
    if not line.strip():
        # An item may begin with one blank line; later content is outside it.
        return _ListState(columns[:-1] if state.empty_item else columns)
    indent = _indent_width(line)
    item = None if _THEMATIC_BREAK.match(line) else _LIST_ITEM.match(line)
    empty = item is not None and not line[item.end() :].strip()
    if _lazy_marker(line, state, after_paragraph, quoted):
        # A lazy continuation or an underline, so the open items are untouched.
        return state
    columns = _close_dedented(columns, line, indent, after_paragraph)
    # Four columns past its container the marker is code, or lazy text, so it
    # opens no list of its own.
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
    """Whether what is left of a release body renders anything at all."""
    in_comment = False
    for line in _markdown_lines(markdown):
        opens_raw = any(opener.match(line) for opener, _ in _RAW_BLOCKS)
        if not in_comment and (_FENCE_PATTERN.match(line) or opens_raw):
            # A code block or raw HTML block renders even when it is empty.
            return True
        # No containers are tracked, so the opener is read at the margin. It does
        # not matter: an item renders its marker whatever the block inside hides.
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
    tag: str | None = None,
    html_url: str | None = None,
    source: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    # A body that renders as nothing is an unannounced release, not empty notes.
    if markdown and not _renders_visibly(markdown):
        markdown = None

    truncated = False
    if markdown and len(markdown) > RELEASE_NOTES_MAX_CHARS:
        markdown = _close_open_fence(markdown[:RELEASE_NOTES_MAX_CHARS].rstrip())
        truncated = True

    return {
        # Echoed, so the UI can drop an answer to a version it has moved on from.
        "version": version,
        "markdown": markdown or None,
        "heading": heading,
        "tag": tag,
        "html_url": html_url or None,
        # False means the release published no notes; the UI links out.
        "matched": bool(markdown),
        "truncated": truncated,
        "source": source if markdown else None,
        "release_notes_url": RELEASE_NOTES_URL,
        "error": error,
    }
