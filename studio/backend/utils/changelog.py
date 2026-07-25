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

_HEADING_PATTERN = re.compile(r"^##\s+(?P<title>.*?)\s*$")
_FENCE_PATTERN = re.compile(r"^\s*(?P<marker>`{3,}|~{3,})(?P<rest>.*)$")
_COMMENT_OPEN = "<!--"
_COMMENT_CLOSE = "-->"
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
        fence = _FENCE_PATTERN.match(line)
        if fence and not in_comment:
            open_fence = _next_fence_state(
                open_fence, fence.group("marker"), fence.group("rest")
            )
            visible = ""
        elif open_fence:
            visible = ""
        else:
            # Commented-out sections are not rendered, so they are not releases.
            visible, in_comment = _strip_comments(line, in_comment)
        # A `##` inside a fenced block is sample markdown, not a real heading.
        match = _HEADING_PATTERN.match(visible) if visible else None
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


def get_release_notes(version: str) -> dict[str, Any]:
    """Return release notes for exactly `version` for the update popup."""
    version = version.strip()
    if not is_supported_version_query(version):
        return _notes_response(version = version, error = "Unsupported version.")

    local = _read_local_changelog()
    remote = ChangelogSource(text = None, source = None)
    if os.environ.get(DISABLE_ENV_VAR) != "1":
        remote = get_remote_changelog()

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

    error = remote.error if (remote.error and not local.text) else None
    return _notes_response(version = version, error = error)


def get_remote_changelog() -> ChangelogSource:
    """Fetch CHANGELOG.md from the repo using a small in-process TTL cache."""
    global _remote_cache, _remote_fetching

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


def _local_changelog_candidates() -> list[Path]:
    override = os.environ.get(CHANGELOG_PATH_ENV_VAR, "").strip()
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())

    # changelog.py -> utils -> backend -> studio -> repo root. Repo root first:
    # in a source checkout the editable file must win over the build snapshot
    # that packaging writes into studio/.
    parents = Path(__file__).resolve().parents
    for index in (3, 2, 1, 4):
        if index < len(parents):
            candidates.append(parents[index] / CHANGELOG_FILENAME)

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
    """Return the line with HTML-comment spans removed, and the trailing state."""
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
        visible.append(line[index:opening])
        index = opening + len(_COMMENT_OPEN)
        in_comment = True
    return "".join(visible), in_comment


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


def _notes_response(
    *,
    version: str,
    markdown: str | None = None,
    heading: str | None = None,
    source: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
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
