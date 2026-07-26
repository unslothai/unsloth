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
CHANGELOG_SUCCESS_TTL_SECONDS = 30 * 60
CHANGELOG_FAILURE_TTL_SECONDS = 5 * 60
RELEASE_NOTES_MAX_CHARS = 20_000

_HEADING_PATTERN = re.compile(r"^ {0,3}##\s+(?P<title>.*?)\s*$")
_FENCE_PATTERN = re.compile(r"^ {0,3}(?P<marker>`{3,}|~{3,})(?P<rest>.*)$")
# CommonMark type 1 HTML blocks: contents are literal until a closing tag,
# which the spec says need not be the one that opened the block.
_RAW_HTML_OPEN = re.compile(r"^ {0,3}<(pre|script|style|textarea)(?=[\s>]|$)", re.IGNORECASE)
_RAW_HTML_CLOSE = re.compile(r"</(pre|script|style|textarea)\s*>", re.IGNORECASE)
# Type 6 blocks run to the next blank line, so `<details>` only holds Markdown
# once a blank line has closed the block. Open and close tags both start one.
_HTML_BLOCK_OPEN = re.compile(r"^ {0,3}</?([a-zA-Z][a-zA-Z0-9-]*)(?=[\s/>]|$)")
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
_COMMENT_OPEN = "<!--"
_COMMENT_CLOSE = "-->"
_CODE_SPAN_PATTERN = re.compile(r"(`+)(?:.*?)\1")
_VERSION_TOKEN_PATTERN = re.compile(r"^[\[(]?v?(?P<version>[0-9][0-9A-Za-z.!+-]*?)[\])]?$")
_SAFE_VERSION_PATTERN = re.compile(r"^[0-9A-Za-z][0-9A-Za-z.!+-]{0,63}$")


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
    """Whether `version` is shaped like something we can look up at all."""
    return bool(_SAFE_VERSION_PATTERN.match(version.strip()))


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
    in_raw_html = False
    in_html_block = False
    after_paragraph = False

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
        # Raw HTML first: its contents are literal, so a fence inside it is not
        # a fence. Only the text after the closing tag can carry a heading.
        if in_raw_html:
            visible, in_raw_html = _strip_raw_html(line, in_raw_html)
        elif in_html_block:
            # A blank line is the only thing that ends a type 6 block.
            in_html_block = line.strip() != ""
            visible = ""
        elif (fence := _FENCE_PATTERN.match(line)) and not in_comment:
            open_fence = _next_fence_state(open_fence, fence.group("marker"), fence.group("rest"))
            visible = ""
        elif open_fence:
            visible = ""
        else:
            # Commented-out sections are not rendered, so they are not releases.
            visible, in_comment = _strip_comments(line, in_comment)
            # Nor is anything inside a raw HTML block such as <pre>.
            visible, in_raw_html = _strip_raw_html(visible, in_raw_html)
            if visible and not in_raw_html and _opens_html_block(visible, after_paragraph):
                in_html_block = True
                visible = ""
        # A `##` inside a fenced block is sample markdown, not a real heading.
        match = _HEADING_PATTERN.match(visible) if visible else None
        # Only ordinary text continues a paragraph. A heading, a block or an
        # indented code line (four spaces, outside a paragraph) ends one.
        indented_code = not after_paragraph and line[:4] == "    "
        after_paragraph = bool(visible.strip()) and match is None and not indented_code
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
    wanted = _parse_version(version)
    for entry in parse_changelog(text):
        if entry.version == version:
            return entry
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

    while True:
        now = time.monotonic()
        with _cache_condition:
            if _remote_cache and _remote_cache.expires_at > now:
                return _remote_cache.source
            if not _remote_fetching:
                _remote_fetching = True
                break
            _cache_condition.wait(timeout = CHANGELOG_TIMEOUT_SECONDS + 1)

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
        _remote_fetching = False
        _cache_condition.notify_all()
    return source


def _fetch_remote_changelog() -> ChangelogSource:
    url = os.environ.get(CHANGELOG_URL_ENV_VAR, "").strip() or CHANGELOG_RAW_URL
    if not url.startswith(("http://", "https://")):
        return ChangelogSource(text = None, source = None, error = "Invalid changelog URL.")

    request = urllib.request.Request(
        url,
        headers = {"User-Agent": "unsloth-studio-update-check"},
    )
    try:
        with urllib.request.urlopen(request, timeout = CHANGELOG_TIMEOUT_SECONDS) as response:
            body = response.read(CHANGELOG_MAX_BYTES + 1)
        if len(body) > CHANGELOG_MAX_BYTES:
            return ChangelogSource(
                text = None,
                source = None,
                error = "Release notes response was too large.",
            )
        return ChangelogSource(text = body.decode("utf-8", errors = "replace"), source = "remote")
    except OSError:
        return ChangelogSource(
            text = None,
            source = None,
            error = "Could not reach the changelog for release notes.",
        )
    except UnicodeError:
        return ChangelogSource(text = None, source = None, error = "Malformed changelog.")


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


def _next_fence_state(open_fence: str | None, marker: str, rest: str) -> str | None:
    """Track the open fence marker.

    A closer must be the same character, at least as long, and carry nothing
    after it. So neither a ``` sample nor a ```` line with trailing text ends
    a ```` block early, while an opening fence may still have an info string.
    """
    if open_fence is None:
        return marker
    if marker[0] == open_fence[0] and len(marker) >= len(open_fence) and not rest.strip():
        return None
    return open_fence


def _strip_comments(line: str, in_comment: bool) -> tuple[str, bool]:
    """Return the line with HTML-comment spans removed, and the trailing state.

    Delimiters inside inline code are literal, so a note documenting `<!--`
    does not put the parser into comment state.
    """
    visible: list[str] = []
    index = 0
    while index < len(line):
        if in_comment:
            close = line.find(_COMMENT_CLOSE, index)
            if close == -1:
                return "".join(visible), True
            index = close + len(_COMMENT_CLOSE)
            in_comment = False
            continue

        opening = line.find(_COMMENT_OPEN, index)
        if opening == -1:
            visible.append(line[index:])
            break

        span = _CODE_SPAN_PATTERN.search(line, index)
        if span and span.start() <= opening < span.end():
            visible.append(line[index : span.end()])
            index = span.end()
            continue

        visible.append(line[index:opening])
        index = opening + len(_COMMENT_OPEN)
        in_comment = True
    return "".join(visible), in_comment


def _opens_html_block(line: str, after_paragraph: bool) -> bool:
    """True if `line` starts a CommonMark type 6 or type 7 HTML block."""
    match = _HTML_BLOCK_OPEN.match(line)
    if match is not None and match.group(1).lower() in _HTML_BLOCK_TAGS:
        return True
    return not after_paragraph and _HTML_TAG_ONLY_LINE.match(line) is not None


def _strip_raw_html(line: str, in_raw_html: bool) -> tuple[str, bool]:
    """Drop the parts of a line inside a raw HTML block, and return the state."""
    if in_raw_html:
        close = _RAW_HTML_CLOSE.search(line)
        return (line[close.end() :], False) if close else ("", True)

    # A block only opens at the start of a line; mid-line tags are inline HTML.
    opening = _RAW_HTML_OPEN.match(line)
    if opening is None:
        return line, False

    rest = line[opening.end() :]
    close = _RAW_HTML_CLOSE.search(rest)
    return (rest[close.end() :], False) if close else ("", True)


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


def _renders_visibly(markdown: str) -> bool:
    """Whether a section body renders anything at all."""
    in_comment = False
    for line in markdown.splitlines():
        if not in_comment and (_FENCE_PATTERN.match(line) or _RAW_HTML_OPEN.match(line)):
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
        markdown = markdown[:RELEASE_NOTES_MAX_CHARS].rstrip()
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
