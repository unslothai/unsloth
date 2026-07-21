# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""whisper.cpp prebuilt freshness check.

Reads UNSLOTH_WHISPER_PREBUILT_INFO.json (written by install_whisper_prebuilt.py)
and compares the installed release tag against the latest on GitHub.
Surfaced via main.py:lifespan() and /api/inference/status. Fails open
on any missing data so we never show a misleading banner.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

# 3 days matches Unsloth's typical whisper.cpp release cadence.
STALENESS_THRESHOLD_DAYS = 3

# 24h TTL keeps the GitHub call off the hot path and within rate limits.
_RELEASE_CACHE_TTL_SECONDS = 24 * 60 * 60

_INSTALL_MARKER_NAME = "UNSLOTH_WHISPER_PREBUILT_INFO.json"

_marker_cache: dict[str, Optional[dict]] = {}
_release_memo: dict[str, tuple[float, Optional[str]]] = {}
# Newest-release asset sizes (name -> bytes), memoized like the tag (24h TTL).
_assets_memo: dict[str, tuple[float, dict[str, int]]] = {}


def _cache_dir() -> Path:
    """Lazy import so tests can stub storage_roots."""
    try:
        from utils.paths.storage_roots import cache_root
        return cache_root() / "whisper_cpp_freshness"
    except Exception:
        return Path.home() / ".unsloth" / "studio" / "cache" / "whisper_cpp_freshness"


def read_install_marker(binary_path: Optional[str]) -> Optional[dict]:
    """Walk up from binary_path to find UNSLOTH_WHISPER_PREBUILT_INFO.json.
    None = no marker (source build / custom path) or invalid JSON."""
    if not binary_path:
        return None
    cached = _marker_cache.get(binary_path)
    if cached is not None or binary_path in _marker_cache:
        return cached
    p = Path(binary_path)
    marker: Optional[dict] = None
    # Cover all find_whisper_server_binary layouts (binary is 1-4 dirs deep):
    for parent in p.parents[:5]:
        candidate = parent / _INSTALL_MARKER_NAME
        if candidate.is_file():
            try:
                marker = json.loads(candidate.read_text(encoding = "utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug(
                    "failed to parse whisper install marker",
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
        logger.debug("whisper freshness cache write failed", repo = repo, error = str(exc))


def _fetch_latest_release_tag(repo: str, timeout: float = 5.0) -> Optional[str]:
    """Newest published release tag for `repo`, by publish time.

    Resolves "latest" the way install_whisper_prebuilt.py does (newest
    non-draft/non-prerelease by ``published_at``), NOT via GitHub's
    ``/releases/latest`` pointer. That pointer sorts by commit date and can lag
    behind the build the installer actually installs, so detection and apply
    disagreed -- the cause of the downgrade/sticky banner. None on any failure
    (offline, rate-limited, etc)."""
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
        logger.debug("whisper freshness fetch failed", repo = repo, error = str(exc))
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
    newest = max(published, key = lambda r: r.get("published_at") or "")
    return newest["tag_name"]


def latest_published_release(repo: str, *, force_refresh: bool = False) -> Optional[str]:
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


def _fetch_latest_release_assets(repo: str, timeout: float = 5.0) -> Optional[dict[str, int]]:
    """Asset name -> size (bytes) for the newest published release of `repo`,
    selected exactly like _fetch_latest_release_tag. None on any failure."""
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
        logger.debug("whisper freshness asset fetch failed", repo = repo, error = str(exc))
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
    newest = max(published, key = lambda r: r.get("published_at") or "")
    assets: dict[str, int] = {}
    for a in newest.get("assets") or []:
        name, size = a.get("name"), a.get("size")
        if isinstance(name, str) and isinstance(size, int):
            assets[name] = size
    return assets


def latest_release_assets(repo: str, *, force_refresh: bool = False) -> Optional[dict[str, int]]:
    """Newest-release asset sizes for `repo`, memoized (24h TTL). None when
    offline and never fetched. In-memory only -- a restart simply re-fetches."""
    if not repo:
        return None
    now = time.time()
    if not force_refresh:
        memo = _assets_memo.get(repo)
        if memo and now - memo[0] < _RELEASE_CACHE_TTL_SECONDS:
            return memo[1]
    assets = _fetch_latest_release_assets(repo)
    if assets is None:
        memo = _assets_memo.get(repo)
        return memo[1] if memo else None
    _assets_memo[repo] = (now, assets)
    return assets


def _asset_platform_suffix(asset: str, installed_tag: Optional[str]) -> Optional[str]:
    """The tag-independent ``<os>-<arch>-<accel>.<ext>`` suffix that identifies
    this host's bundle, obtained by stripping the ``whisper-<tag>-`` prefix from
    the installed asset name. Falls back to anchoring on the platform token when
    the marker tag does not line up with the asset's embedded tag."""
    if isinstance(installed_tag, str) and installed_tag:
        tag = installed_tag if installed_tag.startswith("v") else f"v{installed_tag}"
        prefix = f"whisper-{tag}-"
        if asset.startswith(prefix):
            return asset[len(prefix) :]
    m = re.search(r"-((?:linux|macos|windows)-.*)$", asset)
    return m.group(1) if m else None


def update_download_size_bytes(
    marker: Optional[dict],
    latest_tag: Optional[str],
    repo: Optional[str],
    *,
    force_refresh: bool = False,
) -> Optional[int]:
    """Download size of the latest-release asset matching this host's installed
    bundle (same platform/arch/accel suffix as the installed asset). None when
    there is no marker asset, the latest assets can't be read, or no match.

    whisper.cpp assets are named ``whisper-<tag>-<os>-<arch>-<accel>.<ext>`` and
    the fork builds every slice itself, so there is only the publish repo to
    consult (no upstream/binary_repo passthrough)."""
    if not marker or not latest_tag or not repo:
        return None
    installed_asset = marker.get("asset")
    if not isinstance(installed_asset, str) or not installed_asset:
        return None
    suffix = _asset_platform_suffix(installed_asset, marker.get("release_tag"))
    if not suffix:
        return None
    assets = latest_release_assets(repo, force_refresh = force_refresh)
    if not assets:
        return None
    latest_v = latest_tag if latest_tag.startswith("v") else f"v{latest_tag}"
    want = f"whisper-{latest_v}-{suffix}"
    if want in assets:
        return assets[want]
    # Tag formatting can vary; fall back to the platform+accel suffix.
    for name, size in assets.items():
        if name.endswith(suffix):
            return size
    return None


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


def parse_release_version(tag: object) -> Optional[tuple]:
    """Comparable key ``(upstream_major, upstream_minor, upstream_patch,
    unsloth_serial)`` for a whisper.cpp release tag like ``v1.9.1-unsloth.2``.

    Tolerant of a leading ``v`` and of a missing ``-unsloth.N`` suffix (a missing
    serial is treated as 0). The upstream version is padded to major/minor/patch.
    None for anything whose version component is not purely numeric."""
    if not isinstance(tag, str):
        return None
    s = tag.strip()
    if not s:
        return None
    if s[0] in ("v", "V"):
        s = s[1:]
    serial = 0
    m = re.search(r"-unsloth\.(\d+)$", s)
    if m:
        serial = int(m.group(1))
        s = s[: m.start()]
    parts = s.split(".")
    nums: list[int] = []
    for part in parts:
        if not part.isdigit():
            return None
        nums.append(int(part))
    if not nums:
        return None
    while len(nums) < 3:
        nums.append(0)
    return (nums[0], nums[1], nums[2], serial)


def is_behind(installed: Optional[str], latest: Optional[str]) -> bool:
    """Whether `installed` is genuinely behind `latest`.

    - identical tags -> not behind (clears the sticky banner post-update)
    - both parse -> behind only when the latest version key is strictly greater
      (a lower/equal version is never "behind" -> downgrade guard)
    - either fails to parse and they differ -> behind (plain inequality)
    """
    if not installed or not latest:
        return False
    installed, latest = installed.strip(), latest.strip()
    if installed == latest:
        return False
    installed_key, latest_key = parse_release_version(installed), parse_release_version(latest)
    if installed_key is not None and latest_key is not None:
        return latest_key > installed_key
    # One side is unparseable and the tags already differ: treat as behind.
    return True


def check_prebuilt_freshness(
    binary_path: Optional[str],
    *,
    threshold_days: int = STALENESS_THRESHOLD_DAYS,
    now: Optional[datetime] = None,
) -> dict:
    """Returns {has_marker, stale, behind, installed_tag, latest_tag,
    installed_at_utc, age_days, published_repo, threshold_days}.
    behind = installed genuinely older than latest (see is_behind).
    stale = behind AND age >= threshold.
    Fails open on missing data (behind/stale stay False)."""
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
    marker = read_install_marker(binary_path)
    if not marker:
        return out
    out["has_marker"] = True
    # The whisper marker records a single ``release_tag`` (e.g. v1.9.1-unsloth.2);
    # both display and comparison use it directly.
    out["installed_tag"] = marker.get("release_tag")
    out["installed_at_utc"] = marker.get("installed_at_utc")
    out["published_repo"] = marker.get("published_repo")

    installed_full = marker.get("release_tag")
    repo = out["published_repo"]
    if not repo or not installed_full:
        return out
    latest = latest_published_release(repo)
    out["latest_tag"] = latest
    out["behind"] = is_behind(installed_full, latest)
    if not out["behind"]:
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
        f"whisper.cpp prebuilt is {age_str} behind: installed "
        f"{installed}, latest {latest}. Run `unsloth studio update` "
        f"to refresh."
    )


def reset_caches(*, drop_disk: bool = False) -> None:
    """Drop the in-memory freshness caches. The no-arg form is test-only.

    With ``drop_disk = True`` also delete the on-disk 24h release cache. Used by
    the post-install/update path: in-memory clearing alone leaves the stale
    same-version value on disk, so if the post-install GitHub refresh can't reach
    the network, ``latest_published_release`` would replay that stale disk value
    (see its last-good fallback) and the banner could linger. Dropping the disk
    cache makes latest read as None in that offline case, so the banner fails
    open (off) instead of pointing at the just-replaced build."""
    _marker_cache.clear()
    _release_memo.clear()
    _assets_memo.clear()
    if drop_disk:
        import shutil

        # _cache_dir() is a dedicated freshness-only subdir; it is re-created on
        # the next _save_disk_cache. ignore_errors so a missing/locked dir is a
        # no-op rather than breaking an otherwise successful install.
        shutil.rmtree(_cache_dir(), ignore_errors = True)
