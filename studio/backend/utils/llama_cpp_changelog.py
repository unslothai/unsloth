# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""New carried changes between two published llama.cpp prebuilts.

The release body is cumulative, so the banner must diff the installed and target
bodies; showing the target body alone relabels old carried PRs as new.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

import structlog

from utils.prebuilt.freshness_flow import (
    RELEASE_CACHE_TTL_SECONDS,
    RELEASE_FAILURE_CACHE_TTL_SECONDS,
)

logger = structlog.get_logger(__name__)

MAX_CHANGES = 50
# The only repository whose notes this module can read. Its bodies are generated
# (cumulative, one bullet per carried PR); --published-repo can point an install
# at upstream or a mirror, whose per-release notes are a different document. On
# ggml-org/llama.cpp, b10721 -> b10734 reported 5 "changes" -- commit-message
# lines and an attestation URL -- and dropped 324 bullets from the 9 releases in
# between, because a per-release body says nothing about what is still carried.
CUMULATIVE_NOTES_REPO = "unslothai/llama.cpp"
# 4 MiB. A release body is a few KB; the cap only stops an unbounded read if the
# far side ever stops behaving like the GitHub API.
MAX_RELEASE_BYTES = 4 * 1024 * 1024
# A forced retry bypasses the memos, so without a floor a user holding down the
# banner's Retry button issues two uncached GitHub calls per click.
FORCE_REFRESH_MIN_INTERVAL_SECONDS = 30.0
_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
# "." and ".." are valid under _REPO because "." is in the class above, and
# "owner/.." would walk out of /repos/ on any normalizing proxy.
_DOT_SEGMENT = re.compile(r"^\.+$")
_BULLET = re.compile(r"^ {0,3}[-*+]\s+(.+?)\s*$")
_LINK = re.compile(r"\[([^\]]+)]\((https://github\.com/[^\s)]+)\)")
_PR_URL = re.compile(r"^https://github\.com/([^/]+/[^/]+)/pull/(\d+)(?:/|$)", re.I)
_ISSUE_URL = re.compile(r"^https://github\.com/([^/]+/[^/]+)/issues/(\d+)(?:/|$)", re.I)
_TEXT_REFERENCE = re.compile(r"(?<![\w./-])([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?)#(\d+)\b")

_release_memo: dict[tuple[str, str], tuple[float, dict]] = {}
_release_failed_at: dict[tuple[str, str], float] = {}
_release_forced_at: dict[tuple[str, str], float] = {}


def _valid_repo(repo: str) -> bool:
    """``owner/name``, with no segment that is only dots."""
    if not isinstance(repo, str) or not _REPO.fullmatch(repo):
        return False
    return not any(_DOT_SEGMENT.fullmatch(part) for part in repo.split("/"))


def _is_cumulative_repo(repo: str) -> bool:
    """Case-folded: GitHub owner/name is case-insensitive and --published-repo
    persists whatever spelling was typed, so ``UnslothAI/llama.cpp`` is the same
    official repository and must not be labelled a custom one."""
    return repo.casefold() == CUMULATIVE_NOTES_REPO.casefold()


def _fetch_release(
    repo: str,
    tag: str,
    timeout: float = 5.0,
) -> Optional[dict]:
    """One exact GitHub release. None on invalid input or any failure."""
    if not _valid_repo(repo) or not tag:
        return None
    from utils.utils import call_with_deadline

    try:
        return call_with_deadline(
            lambda: _fetch_release_blocking(repo, tag, timeout),
            timeout + 1,
            name = "llama-changelog-fetch",
        )
    except TimeoutError as exc:
        logger.debug("llama changelog fetch failed", repo = repo, tag = tag, error = str(exc))
        return None


def _fetch_release_blocking(repo: str, tag: str, timeout: float) -> Optional[dict]:
    encoded_tag = urllib.parse.quote(tag, safe = "")
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "unsloth-studio-llama-changelog",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/releases/tags/{encoded_tag}",
        headers = headers,
    )
    try:
        with urllib.request.urlopen(request, timeout = timeout) as response:
            # Bounded: read one byte past the cap so an oversized body is rejected
            # rather than buffered in full.
            raw = response.read(MAX_RELEASE_BYTES + 1)
        if len(raw) > MAX_RELEASE_BYTES:
            logger.debug("llama changelog release too large", repo = repo, tag = tag)
            return None
        payload = json.loads(raw.decode("utf-8"))
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        OSError,
        # IncompleteRead on a truncated response is an HTTPException, NOT an
        # OSError, so without this a mid-read cut escapes to any direct caller.
        http.client.HTTPException,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        logger.debug("llama changelog fetch failed", repo = repo, tag = tag, error = str(exc))
        return None
    return payload if isinstance(payload, dict) else None


def _release_for_tag(
    repo: str,
    tag: str,
    *,
    force_refresh: bool = False,
) -> Optional[dict]:
    """Exact release with 24h success and 60s failure memoization."""
    key = (repo, tag)
    # Memory-only, so monotonic throughout: a backward clock step must not be able
    # to extend the TTL. freshness_flow uses wall time because it persists to disk.
    now = time.monotonic()
    if force_refresh:
        # Honour the bypass, but not faster than once per key per interval.
        forced_at = _release_forced_at.get(key)
        if forced_at is not None and now - forced_at < FORCE_REFRESH_MIN_INTERVAL_SECONDS:
            force_refresh = False
        else:
            _release_forced_at[key] = now
    if not force_refresh:
        failed_at = _release_failed_at.get(key)
        cached = _release_memo.get(key)
        fresh = cached is not None and now - cached[0] < RELEASE_CACHE_TTL_SECONDS
        if failed_at is not None and now - failed_at < RELEASE_FAILURE_CACHE_TTL_SECONDS:
            # Suppress the retry, but an entry past its TTL is still expired: do
            # not let a recent failure resurrect a stale body as if it were fresh.
            return cached[1] if fresh else None
        if fresh:
            return cached[1]
    release = _fetch_release(repo, tag)
    if release is None:
        _release_failed_at[key] = time.monotonic()
        # Fall back to the last good body only while it is still within its TTL.
        # An exact release that has become unreachable must not keep answering
        # from an expired entry, or the panel presents a stale comparison as matched.
        cached = _release_memo.get(key)
        if cached and time.monotonic() - cached[0] < RELEASE_CACHE_TTL_SECONDS:
            return cached[1]
        return None
    _release_failed_at.pop(key, None)
    _release_memo[key] = (time.monotonic(), release)
    return release


def _plain_text(markdown: str) -> str:
    text = _LINK.sub(lambda match: match.group(1), markdown)
    # Underscores in ROCm_Host / GGML_CUDA_ENABLE_UNIFIED_MEMORY are text, not emphasis.
    text = text.replace("`", "").replace("**", "")
    return re.sub(r"\s+", " ", text).strip()


def _entry(markdown: str) -> dict:
    # Metadata starts at " ([", so a title keeps its parens: GLM-5-Next (GLM-5.3-Flash).
    # rfind, not find: the generator always appends metadata as the final
    # parenthesised group, and a PR title may itself contain " ([" -- with find,
    # "vulkan: handle ([a],[b]) tuples ([#5](...))" renders as "vulkan: handle".
    metadata_at = markdown.rfind(" ([")
    summary_markdown = markdown[:metadata_at] if metadata_at >= 0 else markdown
    links = []
    for label, url in _LINK.findall(markdown[metadata_at:] if metadata_at >= 0 else ""):
        clean_label = _plain_text(label)
        if "/commits/" in url and not clean_label.lower().startswith("commit"):
            clean_label = f"commit {clean_label}"
        links.append({"label": clean_label, "url": url})
    return {"summary": _plain_text(summary_markdown), "links": links}


def _identities(markdown: str) -> set[str]:
    """Stable aliases for one carried change: a patch migrated to an Unsloth carry
    PR links that PR but still says ``ggml-org#24423``, and both must match."""
    identities = set()
    # One namespace for all three notations. GitHub issues and pull requests share
    # a repository's number space, so ``/issues/900``, ``/pull/900`` and the textual
    # ``repo#900`` always denote the same object; giving them separate prefixes can
    # only produce false negatives, never avoid a false positive.
    for _label, url in _LINK.findall(markdown):
        match = _PR_URL.match(url) or _ISSUE_URL.match(url)
        if match:
            identities.add(f"ref:{match.group(1).lower()}#{match.group(2)}")
    for repo, number in _TEXT_REFERENCE.findall(_plain_text(markdown)):
        # Shorthand omits the repo: ``ggml-org#24423`` is ``ggml-org/llama.cpp#24423``.
        if "/" not in repo:
            repo = f"{repo}/llama.cpp"
        identities.add(f"ref:{repo.lower()}#{number}")
    if not identities:
        identities.add(f"text:{_entry(markdown)['summary'].casefold()}")
    return identities


def _bullets(body: object) -> list[str]:
    if not isinstance(body, str):
        return []
    return [match.group(1) for line in body.splitlines() if (match := _BULLET.match(line))]


def release_page_url(repo: str, tag: str) -> Optional[str]:
    """The human release page, so a failed comparison can still offer a way to
    read the notes on GitHub. None when the repo is not a safe ``owner/name``."""
    if not _valid_repo(repo) or not tag:
        return None
    return f"https://github.com/{repo}/releases/tag/{urllib.parse.quote(tag, safe = '')}"


def unavailable_reason(repo: str, installed_tag: str, latest_tag: str) -> str:
    """Why a comparison could not be made, for a caller holding ``None``.

    ``notes_not_itemised`` and ``notes_not_comparable`` are permanent -- the
    release predates the bullet format, or the install tracks a repository whose
    notes are not cumulative -- while ``release_notes_unavailable`` is a fetch
    that may succeed later. The banner uses this to decide whether offering Retry
    is honest.
    """
    if not _valid_repo(repo) or not installed_tag or not latest_tag:
        return "release_notes_unavailable"
    if not _is_cumulative_repo(repo):
        return "notes_not_comparable"
    # Both lookups are memoized, so this re-read costs nothing on the path that
    # has just performed them.
    installed = _release_for_tag(repo, installed_tag)
    if installed is None:
        return "release_notes_unavailable"
    # Only the INSTALLED side can be permanently uncomparable: it is a release
    # that shipped before the bullet format and will never gain one. A target
    # without a usable body is the newest release, so it is a publishing gap that
    # may be filled in -- calling that permanent would withhold a Retry that
    # could actually succeed.
    if not _bullets(installed.get("body")):
        return "notes_not_itemised"
    return "release_notes_unavailable"


def changelog_for_update(
    repo: str,
    installed_tag: str,
    latest_tag: str,
    *,
    force_refresh: bool = False,
) -> Optional[dict]:
    """Return only target bullets absent from the installed release.

    None means no comparison was possible; do not fall back to the cumulative body.
    """
    if not repo or not installed_tag or not latest_tag or installed_tag == latest_tag:
        return None
    if not _is_cumulative_repo(repo):
        return None
    installed = _release_for_tag(repo, installed_tag, force_refresh = force_refresh)
    latest = _release_for_tag(repo, latest_tag, force_refresh = force_refresh)
    if installed is None or latest is None:
        return None

    # Releases before b9625-mix-2d6bd50 (2026-06-14) name carries in prose, so no
    # bullets means unknown, not "carries nothing".
    installed_bullets = _bullets(installed.get("body"))
    if not installed_bullets:
        return None
    # A prose-only target legitimately means "this build carries nothing", since
    # the target is always the newest release. A MISSING or blank body is not the
    # same statement: it carries no information at all, and answering "no new
    # changes" would claim a comparison that never happened.
    latest_body = latest.get("body")
    if not isinstance(latest_body, str) or not latest_body.strip():
        return None
    latest_bullets = _bullets(latest_body)
    old_identities = set().union(*(_identities(item) for item in installed_bullets))
    new_items = []
    seen = set()
    for item in latest_bullets:
        identities = _identities(item)
        if identities & old_identities or identities & seen:
            continue
        parsed = _entry(item)
        if not parsed["summary"]:
            # Claim nothing on behalf of a bullet that is never shown; updating
            # `seen` here would let it suppress a later bullet for the same PR.
            continue
        seen.update(identities)
        new_items.append(parsed)

    total = len(new_items)
    release_url = latest.get("html_url")
    if not isinstance(release_url, str) or not release_url.startswith("https://github.com/"):
        encoded_tag = urllib.parse.quote(latest_tag, safe = "")
        release_url = f"https://github.com/{repo}/releases/tag/{encoded_tag}"
    return {
        "changes": new_items[:MAX_CHANGES],
        "total_changes": total,
        "truncated": total > MAX_CHANGES,
        "release_url": release_url,
    }
