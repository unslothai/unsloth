# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared mechanics of the llama.cpp / whisper.cpp prebuilt freshness checks.

The component modules (utils.llama_cpp_freshness / utils.whisper_cpp_freshness)
keep their public names, per-module caches, and version-comparison policy;
everything mechanical (marker walk-up, GitHub release fetch, memo + disk cache,
the freshness report skeleton) lives here, parameterized by call-time callables
so the modules' monkeypatch seams keep working.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import structlog

logger = structlog.get_logger(__name__)

# 24h TTL keeps the GitHub call off the hot path and within rate limits.
RELEASE_CACHE_TTL_SECONDS = 24 * 60 * 60


def read_install_marker(
    binary_path: Optional[str],
    *,
    marker_name: str,
    cache: dict[str, Optional[dict]],
    log_message: str,
) -> Optional[dict]:
    """Walk up from binary_path to find the install marker JSON.
    None = no marker (source build / custom path) or invalid JSON."""
    if not binary_path:
        return None
    cached = cache.get(binary_path)
    if cached is not None or binary_path in cache:
        return cached
    p = Path(binary_path)
    marker: Optional[dict] = None
    # Cover all managed binary layouts (binary is 1-4 dirs deep).
    for parent in p.parents[:5]:
        candidate = parent / marker_name
        if candidate.is_file():
            try:
                marker = json.loads(candidate.read_text(encoding = "utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug(log_message, path = str(candidate), error = str(exc))
                marker = None
            break
    cache[binary_path] = marker
    return marker


def cache_path_for(repo: str, cache_dir: Path) -> Path:
    safe = repo.replace("/", "__")
    return cache_dir / f"{safe}.json"


def load_disk_cache(repo: str, cache_dir: Path) -> Optional[tuple[float, Optional[str]]]:
    path = cache_path_for(repo, cache_dir)
    try:
        payload = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    ts = payload.get("fetched_at")
    tag = payload.get("latest_tag")
    if not isinstance(ts, (int, float)):
        return None
    return float(ts), tag if isinstance(tag, str) else None


def save_disk_cache(
    repo: str, latest_tag: Optional[str], cache_dir: Path, *, log_message: str
) -> None:
    path = cache_path_for(repo, cache_dir)
    try:
        path.parent.mkdir(parents = True, exist_ok = True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"fetched_at": time.time(), "latest_tag": latest_tag}),
            encoding = "utf-8",
        )
        tmp.replace(path)
    except OSError as exc:
        logger.debug(log_message, repo = repo, error = str(exc))


def _fetch_newest_published_release(
    repo: str, timeout: float, *, log_message: str
) -> Optional[dict]:
    """Newest published (non-draft/non-prerelease) release object for `repo`, by
    ``published_at``.

    Resolves "latest" the way the installers do, NOT via GitHub's
    ``/releases/latest`` pointer, which sorts by commit date and can lag the
    build the installer installs (detection and apply then disagree -- the
    downgrade/sticky-banner bug). None on any failure (offline, rate-limited)."""
    import os
    import urllib.error
    import urllib.request

    url = f"https://api.github.com/repos/{repo}/releases?per_page=30"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "unsloth-studio-freshness-check",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers = headers)
    try:
        with urllib.request.urlopen(req, timeout = timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        logger.debug(log_message, repo = repo, error = str(exc))
        return None
    if not isinstance(data, list):
        return None
    published = [
        r
        for r in data
        if isinstance(r, dict)
        and not r.get("draft")
        and not r.get("prerelease")
        and isinstance(r.get("tag_name"), str)
        and r.get("tag_name")
    ]
    if not published:
        return None
    return max(published, key = lambda r: r.get("published_at") or "")


def fetch_latest_release_tag(
    repo: str,
    timeout: float = 5.0,
    *,
    log_message: str,
) -> Optional[str]:
    """Newest published release tag for `repo`, by publish time. None on failure."""
    newest = _fetch_newest_published_release(repo, timeout, log_message = log_message)
    return newest["tag_name"] if newest else None


def fetch_latest_release_assets(
    repo: str,
    timeout: float = 5.0,
    *,
    log_message: str,
) -> Optional[dict[str, int]]:
    """Asset name -> size (bytes) for the newest published release of `repo`,
    selected exactly like fetch_latest_release_tag. None on any failure."""
    newest = _fetch_newest_published_release(repo, timeout, log_message = log_message)
    if newest is None:
        return None
    assets: dict[str, int] = {}
    for a in newest.get("assets") or []:
        name, size = a.get("name"), a.get("size")
        if isinstance(name, str) and isinstance(size, int):
            assets[name] = size
    return assets


def latest_published_release(
    repo: str,
    *,
    force_refresh: bool,
    memo: dict[str, tuple[float, Optional[str]]],
    cache_dir: Callable[[], Path],
    fetch: Callable[[str], Optional[str]],
    save: Callable[[str, Optional[str]], None],
) -> Optional[str]:
    """Latest release tag for `repo`. Memo + disk-cached (24h TTL).
    None when offline and never previously cached."""
    if not repo:
        return None
    now = time.time()
    if not force_refresh:
        cached = memo.get(repo)
        if cached and now - cached[0] < RELEASE_CACHE_TTL_SECONDS:
            return cached[1]
        disk = load_disk_cache(repo, cache_dir())
        if disk and now - disk[0] < RELEASE_CACHE_TTL_SECONDS:
            memo[repo] = disk
            return disk[1]
    latest = fetch(repo)
    if latest is None:
        # Keep the last-good disk value rather than poison it with None.
        disk = load_disk_cache(repo, cache_dir())
        if disk:
            memo[repo] = disk
            return disk[1]
        return None
    memo[repo] = (now, latest)
    save(repo, latest)
    return latest


def latest_release_assets(
    repo: str,
    *,
    force_refresh: bool,
    memo: dict[str, tuple[float, dict[str, int]]],
    fetch: Callable[[str], Optional[dict[str, int]]],
) -> Optional[dict[str, int]]:
    """Newest-release asset sizes for `repo`, memoized (24h TTL). None when
    offline and never fetched. In-memory only -- a restart re-fetches."""
    if not repo:
        return None
    now = time.time()
    if not force_refresh:
        cached = memo.get(repo)
        if cached and now - cached[0] < RELEASE_CACHE_TTL_SECONDS:
            return cached[1]
    assets = fetch(repo)
    if assets is None:
        cached = memo.get(repo)
        return cached[1] if cached else None
    memo[repo] = (now, assets)
    return assets


def parse_installed_at(value: object) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    s = value.replace("Z", "+00:00") if value.endswith("Z") else value
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo = timezone.utc)
    return dt


def check_freshness(
    binary_path: Optional[str],
    *,
    threshold_days: int,
    now: Optional[datetime],
    read_marker: Callable[[Optional[str]], Optional[dict]],
    latest_release: Callable[[str], Optional[str]],
    behind: Callable[[Optional[str], Optional[str]], bool],
    display_tag: Callable[[dict], Any],
    compare_tag: Callable[[dict], Any],
) -> dict:
    """Freshness report skeleton shared by both components; the component's
    marker-tag choice and is_behind policy come in as callables. Fails open on
    missing data (behind/stale stay False)."""
    out: dict = {
        "has_marker": False,
        "stale": False,
        "behind": False,
        "installed_tag": None,
        "latest_tag": None,
        "installed_at_utc": None,
        "age_days": None,
        "published_repo": None,
        "threshold_days": int(threshold_days),
    }
    marker = read_marker(binary_path)
    if not marker:
        return out
    out["has_marker"] = True
    out["installed_tag"] = display_tag(marker)
    out["installed_at_utc"] = marker.get("installed_at_utc")
    out["published_repo"] = marker.get("published_repo")

    installed_full = compare_tag(marker)
    repo = out["published_repo"]
    if not repo or not installed_full:
        return out
    latest = latest_release(repo)
    out["latest_tag"] = latest
    out["behind"] = behind(installed_full, latest)
    if not out["behind"]:
        return out

    installed_at = parse_installed_at(out["installed_at_utc"])
    if installed_at is None:
        return out
    now = now or datetime.now(tz = timezone.utc)
    age_seconds = (now - installed_at).total_seconds()
    out["age_days"] = max(0, int(age_seconds // 86400))
    if age_seconds >= threshold_days * 86400:
        out["stale"] = True
    return out


def format_stale_warning(info: dict, *, component: str) -> str:
    """Human-readable one-liner for stale prebuilt info."""
    age = info.get("age_days")
    installed = info.get("installed_tag") or "unknown"
    latest = info.get("latest_tag") or "unknown"
    age_str = f"{age} day{'s' if age != 1 else ''}" if age is not None else "some time"
    return (
        f"{component} prebuilt is {age_str} behind: installed "
        f"{installed}, latest {latest}. Run `unsloth studio update` "
        f"to refresh."
    )


def reset_caches(
    caches: tuple[dict, ...], *, drop_disk: bool, cache_dir: Callable[[], Path]
) -> None:
    """Drop the in-memory freshness caches; with drop_disk also the on-disk 24h
    release cache (see the component modules for why)."""
    for cache in caches:
        cache.clear()
    if drop_disk:
        import shutil

        # cache_dir() is a dedicated freshness-only subdir, re-created on the next
        # save_disk_cache. ignore_errors so a missing/locked dir is a no-op rather
        # than breaking an otherwise successful install.
        shutil.rmtree(cache_dir(), ignore_errors = True)
