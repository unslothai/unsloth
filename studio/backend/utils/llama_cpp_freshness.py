# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama.cpp prebuilt freshness check.

Reads UNSLOTH_PREBUILT_INFO.json (written by install_llama_prebuilt.py)
and compares the installed release tag against the latest on GitHub.
Surfaced via main.py:lifespan() and /api/inference/status. Fails open
on any missing data so we never show a misleading banner.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

# 3 days matches Unsloth's typical llama.cpp release cadence.
STALENESS_THRESHOLD_DAYS = 3

# 24h TTL keeps the GitHub call off the hot path and within rate limits.
_RELEASE_CACHE_TTL_SECONDS = 24 * 60 * 60

_INSTALL_MARKER_NAME = "UNSLOTH_PREBUILT_INFO.json"

_marker_cache: dict[str, Optional[dict]] = {}
_release_memo: dict[str, tuple[float, Optional[str]]] = {}


def _cache_dir() -> Path:
    """Lazy import so tests can stub storage_roots."""
    try:
        from utils.paths.storage_roots import cache_root

        return cache_root() / "llama_cpp_freshness"
    except Exception:
        return Path.home() / ".unsloth" / "studio" / "cache" / "llama_cpp_freshness"


def read_install_marker(binary_path: Optional[str]) -> Optional[dict]:
    """Walk up from binary_path to find UNSLOTH_PREBUILT_INFO.json.
    None means no marker (source build / custom path) or invalid JSON."""
    if not binary_path:
        return None
    cached = _marker_cache.get(binary_path)
    if cached is not None or binary_path in _marker_cache:
        return cached
    p = Path(binary_path)
    marker: Optional[dict] = None
    # Cover all _find_llama_server_binary layouts:
    #   <install>/llama-server                          (1 up)
    #   <install>/build/bin/llama-server                (3 up, Linux/macOS cmake)
    #   <install>/build/bin/Release/llama-server.exe   (4 up, Windows cmake)
    for parent in p.parents[:5]:
        candidate = parent / _INSTALL_MARKER_NAME
        if candidate.is_file():
            try:
                marker = json.loads(candidate.read_text(encoding = "utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug(
                    "failed to parse install marker",
                    path = str(candidate),
                    error = str(exc),
                )
                marker = None
            break
    _marker_cache[binary_path] = marker
    return marker


def _cache_path_for(repo: str) -> Path:
    safe = repo.replace("/", "__")
    return _cache_dir() / f"{safe}.json"


def _load_disk_cache(repo: str) -> Optional[tuple[float, Optional[str]]]:
    path = _cache_path_for(repo)
    try:
        payload = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    ts = payload.get("fetched_at")
    tag = payload.get("latest_tag")
    if not isinstance(ts, (int, float)):
        return None
    return float(ts), tag if isinstance(tag, str) else None


def _save_disk_cache(repo: str, latest_tag: Optional[str]) -> None:
    path = _cache_path_for(repo)
    try:
        path.parent.mkdir(parents = True, exist_ok = True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"fetched_at": time.time(), "latest_tag": latest_tag}),
            encoding = "utf-8",
        )
        tmp.replace(path)
    except OSError as exc:
        logger.debug("freshness cache write failed", repo = repo, error = str(exc))


def _fetch_latest_release_tag(repo: str, timeout: float = 5.0) -> Optional[str]:
    """GitHub API call. None on any failure (offline, rate-limited, etc)."""
    import urllib.error
    import urllib.request

    url = f"https://api.github.com/repos/{repo}/releases/latest"
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
        logger.debug("freshness fetch failed", repo = repo, error = str(exc))
        return None
    tag = data.get("tag_name")
    return tag if isinstance(tag, str) and tag else None


def latest_published_release(
    repo: str, *, force_refresh: bool = False
) -> Optional[str]:
    """Latest release tag for `repo`. Memo + disk-cached (24h TTL).
    None when offline and never previously cached."""
    if not repo:
        return None
    now = time.time()
    if not force_refresh:
        memo = _release_memo.get(repo)
        if memo and now - memo[0] < _RELEASE_CACHE_TTL_SECONDS:
            return memo[1]
        disk = _load_disk_cache(repo)
        if disk and now - disk[0] < _RELEASE_CACHE_TTL_SECONDS:
            _release_memo[repo] = disk
            return disk[1]
    latest = _fetch_latest_release_tag(repo)
    if latest is None:
        # Keep last-good disk value rather than poisoning with None.
        disk = _load_disk_cache(repo)
        if disk:
            _release_memo[repo] = disk
            return disk[1]
        return None
    _release_memo[repo] = (now, latest)
    _save_disk_cache(repo, latest)
    return latest


def _parse_installed_at(value: object) -> Optional[datetime]:
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


def check_prebuilt_freshness(
    binary_path: Optional[str],
    *,
    threshold_days: int = STALENESS_THRESHOLD_DAYS,
    now: Optional[datetime] = None,
) -> dict:
    """Returns {has_marker, stale, installed_tag, latest_tag,
    installed_at_utc, age_days, published_repo, threshold_days}.
    stale = True iff installed != latest AND age >= threshold.
    Fails open on missing data (stale stays False)."""
    out: dict = {
        "has_marker": False,
        "stale": False,
        "installed_tag": None,
        "latest_tag": None,
        "installed_at_utc": None,
        "age_days": None,
        "published_repo": None,
        "threshold_days": int(threshold_days),
    }
    marker = read_install_marker(binary_path)
    if not marker:
        return out
    out["has_marker"] = True
    out["installed_tag"] = marker.get("tag") or marker.get("release_tag")
    out["installed_at_utc"] = marker.get("installed_at_utc")
    out["published_repo"] = marker.get("published_repo")

    repo = out["published_repo"]
    if not repo or not out["installed_tag"]:
        return out
    latest = latest_published_release(repo)
    out["latest_tag"] = latest
    if not latest or latest == out["installed_tag"]:
        return out

    installed_at = _parse_installed_at(out["installed_at_utc"])
    if installed_at is None:
        return out
    now = now or datetime.now(tz = timezone.utc)
    age_seconds = (now - installed_at).total_seconds()
    out["age_days"] = max(0, int(age_seconds // 86400))
    if age_seconds >= threshold_days * 86400:
        out["stale"] = True
    return out


def format_stale_warning(info: dict) -> str:
    """Human-readable one-liner for stale prebuilt info."""
    age = info.get("age_days")
    installed = info.get("installed_tag") or "unknown"
    latest = info.get("latest_tag") or "unknown"
    age_str = f"{age} day{'s' if age != 1 else ''}" if age is not None else "some time"
    return (
        f"llama.cpp prebuilt is {age_str} behind: installed "
        f"{installed}, latest {latest}. Run `unsloth studio update` "
        f"to refresh."
    )


def reset_caches() -> None:
    """Test-only: drop all in-memory caches."""
    _marker_cache.clear()
    _release_memo.clear()
