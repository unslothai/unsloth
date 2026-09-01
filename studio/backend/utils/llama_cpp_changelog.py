# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""New carried changes between two published llama.cpp prebuilts.

Unsloth's llama.cpp release body is cumulative: every build repeats the PRs
that are still carried on top of upstream. The update banner must therefore
compare the installed and target release bodies instead of displaying the
target body verbatim, or an old carried PR looks new after every update.
"""

from __future__ import annotations

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
_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_BULLET = re.compile(r"^ {0,3}[-*+]\s+(.+?)\s*$")
_LINK = re.compile(r"\[([^\]]+)]\((https://github\.com/[^\s)]+)\)")
_PR_URL = re.compile(r"^https://github\.com/([^/]+/[^/]+)/pull/(\d+)(?:/|$)", re.I)
_ISSUE_URL = re.compile(r"^https://github\.com/([^/]+/[^/]+)/issues/(\d+)(?:/|$)", re.I)
_TEXT_REFERENCE = re.compile(
    r"(?<![\w./-])([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?)#(\d+)\b"
)

_release_memo: dict[tuple[str, str], tuple[float, dict]] = {}
_release_failed_at: dict[tuple[str, str], float] = {}


def _fetch_release(repo: str, tag: str, timeout: float = 5.0) -> Optional[dict]:
    """Fetch one exact GitHub release. None on invalid input or any failure."""
    if not _REPO.fullmatch(repo) or not tag:
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
            payload = json.loads(response.read().decode("utf-8"))
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        logger.debug("llama changelog fetch failed", repo = repo, tag = tag, error = str(exc))
        return None
    return payload if isinstance(payload, dict) else None


def _release_for_tag(repo: str, tag: str, *, force_refresh: bool = False) -> Optional[dict]:
    """Exact release with 24h success and 60s failure memoization."""
    key = (repo, tag)
    now = time.time()
    if not force_refresh:
        failed_at = _release_failed_at.get(key)
        if (
            failed_at is not None
            and time.monotonic() - failed_at < RELEASE_FAILURE_CACHE_TTL_SECONDS
        ):
            cached = _release_memo.get(key)
            return cached[1] if cached else None
        cached = _release_memo.get(key)
        if cached and now - cached[0] < RELEASE_CACHE_TTL_SECONDS:
            return cached[1]
    release = _fetch_release(repo, tag)
    if release is None:
        _release_failed_at[key] = time.monotonic()
        cached = _release_memo.get(key)
        return cached[1] if cached else None
    _release_failed_at.pop(key, None)
    _release_memo[key] = (now, release)
    return release


def _plain_text(markdown: str) -> str:
    text = _LINK.sub(lambda match: match.group(1), markdown)
    # Release titles regularly contain C/C++ identifiers. Underscores in
    # ROCm_Host or GGML_CUDA_ENABLE_UNIFIED_MEMORY are text, not emphasis.
    text = text.replace("`", "").replace("**", "")
    return re.sub(r"\s+", " ", text).strip()


def _entry(markdown: str) -> dict:
    # Generated release bullets put link metadata after `` ([...``. Preserve
    # parentheses in the title itself, such as ``GLM-5-Next (GLM-5.3-Flash)``.
    metadata_at = markdown.find(" ([")
    summary_markdown = markdown[:metadata_at] if metadata_at >= 0 else markdown
    links = []
    for label, url in _LINK.findall(markdown[metadata_at:] if metadata_at >= 0 else ""):
        clean_label = _plain_text(label)
        if "/commits/" in url and not clean_label.lower().startswith("commit"):
            clean_label = f"commit {clean_label}"
        links.append({"label": clean_label, "url": url})
    return {"summary": _plain_text(summary_markdown), "links": links}


def _identities(markdown: str) -> set[str]:
    """Stable aliases for one carried change.

    A patch can move from its original upstream PR to an Unsloth carry PR. The
    newer bullet then links the carry PR but still says ``ggml-org#24423``.
    Keeping both aliases stops that already-installed patch from becoming new.
    """
    identities = set()
    for _label, url in _LINK.findall(markdown):
        match = _PR_URL.match(url)
        if match:
            identities.add(f"pr:{match.group(1).lower()}#{match.group(2)}")
    for _label, url in _LINK.findall(markdown):
        match = _ISSUE_URL.match(url)
        if match:
            identities.add(f"issue:{match.group(1).lower()}#{match.group(2)}")
    for repo, number in _TEXT_REFERENCE.findall(_plain_text(markdown)):
        # Release shorthand omits the repository because every item is about
        # llama.cpp: ``ggml-org#24423`` means ``ggml-org/llama.cpp#24423``.
        if "/" not in repo:
            repo = f"{repo}/llama.cpp"
        identities.add(f"pr:{repo.lower()}#{number}")
    if not identities:
        identities.add(f"text:{_entry(markdown)['summary'].casefold()}")
    return identities


def _bullets(body: object) -> list[str]:
    if not isinstance(body, str):
        return []
    return [match.group(1) for line in body.splitlines() if (match := _BULLET.match(line))]


def changelog_for_update(
    repo: str,
    installed_tag: str,
    latest_tag: str,
    *,
    force_refresh: bool = False,
) -> Optional[dict]:
    """Return only target bullets absent from the installed release.

    None means the comparison could not be made. Callers must not display the
    target body alone in that case, because it is cumulative and includes old
    entries.
    """
    if not repo or not installed_tag or not latest_tag or installed_tag == latest_tag:
        return None
    installed = _release_for_tag(repo, installed_tag, force_refresh = force_refresh)
    latest = _release_for_tag(repo, latest_tag, force_refresh = force_refresh)
    if installed is None or latest is None:
        return None

    old_identities = set().union(
        *(_identities(item) for item in _bullets(installed.get("body")))
    )
    new_items = []
    seen = set()
    for item in _bullets(latest.get("body")):
        identities = _identities(item)
        if identities & old_identities or identities & seen:
            continue
        seen.update(identities)
        parsed = _entry(item)
        if parsed["summary"]:
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
