# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""whisper.cpp prebuilt freshness check.

Reads UNSLOTH_WHISPER_PREBUILT_INFO.json (written by install_whisper_prebuilt.py)
and compares the installed release tag against the latest on GitHub. Surfaced via
utils.whisper_cpp_update (GET /api/whisper/update-status and the combined
llama+whisper update status). Fails open on any missing data so we never show a
misleading banner.

The mechanics (marker walk-up, GitHub fetch, memo + disk cache, report skeleton)
live in utils.prebuilt.freshness_flow; this module keeps the whisper version
policy and the per-module caches its tests patch.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

from utils.prebuilt import freshness_flow as _flow
from utils.prebuilt.whisper_layout import lookup_marker

logger = structlog.get_logger(__name__)

# 3 days matches Unsloth's typical whisper.cpp release cadence.
STALENESS_THRESHOLD_DAYS = 3

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
    marker = lookup_marker(binary_path).marker
    _marker_cache[binary_path] = marker
    return marker


def _load_disk_cache(repo: str) -> Optional[tuple[float, Optional[str]]]:
    return _flow.load_disk_cache(repo, _cache_dir())


def _save_disk_cache(repo: str, latest_tag: Optional[str]) -> None:
    _flow.save_disk_cache(
        repo, latest_tag, _cache_dir(), log_message = "whisper freshness cache write failed"
    )


def _fetch_latest_release_tag(repo: str, timeout: float = 5.0) -> Optional[str]:
    """Newest published release tag for `repo`, by publish time (see
    freshness_flow for why this is not GitHub's /releases/latest pointer)."""
    return _flow.fetch_latest_release_tag(
        repo, timeout, log_message = "whisper freshness fetch failed"
    )


def latest_published_release(repo: str, *, force_refresh: bool = False) -> Optional[str]:
    """Latest release tag for `repo`. Memo + disk-cached (24h TTL).
    None when offline and never previously cached."""
    return _flow.latest_published_release(
        repo,
        force_refresh = force_refresh,
        memo = _release_memo,
        cache_dir = lambda: _cache_dir(),
        fetch = lambda r: _fetch_latest_release_tag(r),
        save = lambda r, tag: _save_disk_cache(r, tag),
    )


def _fetch_latest_release_assets(repo: str, timeout: float = 5.0) -> Optional[dict[str, int]]:
    """Asset name -> size (bytes) for the newest published release of `repo`,
    selected exactly like _fetch_latest_release_tag. None on any failure."""
    return _flow.fetch_latest_release_assets(
        repo, timeout, log_message = "whisper freshness asset fetch failed"
    )


def latest_release_assets(repo: str, *, force_refresh: bool = False) -> Optional[dict[str, int]]:
    """Newest-release asset sizes for `repo`, memoized (24h TTL). None when
    offline and never fetched. In-memory only -- a restart simply re-fetches."""
    return _flow.latest_release_assets(
        repo,
        force_refresh = force_refresh,
        memo = _assets_memo,
        fetch = lambda r: _fetch_latest_release_assets(r),
    )


def _asset_platform_suffix(asset: str, installed_tag: Optional[str]) -> Optional[str]:
    """The tag-independent ``<os>-<arch>-<accel>.<ext>`` suffix identifying this
    host's bundle, from stripping the ``whisper-<tag>-`` prefix off the installed
    asset name. Falls back to anchoring on the platform token when the marker tag
    does not line up with the asset's embedded tag."""
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
    bundle (same platform/arch/accel suffix). None when there is no marker asset,
    the latest assets can't be read, or no match.

    Assets are named ``whisper-<tag>-<os>-<arch>-<accel>.<ext>`` and the fork
    builds every slice itself, so only the publish repo is consulted (no
    upstream/binary_repo passthrough)."""
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


def parse_release_version(tag: object) -> Optional[tuple]:
    """Comparable key ``(upstream_major, upstream_minor, upstream_patch,
    unsloth_serial)`` for a tag like ``v1.9.1-unsloth.2``.

    Tolerant of a leading ``v`` and a missing ``-unsloth.N`` suffix (serial then
    0); the upstream version is padded to major/minor/patch. None when the version
    component is not purely numeric."""
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
    # The whisper marker records a single ``release_tag`` (e.g. v1.9.1-unsloth.2);
    # both display and comparison use it directly.
    return _flow.check_freshness(
        binary_path,
        threshold_days = threshold_days,
        now = now,
        read_marker = lambda p: read_install_marker(p),
        latest_release = lambda repo: latest_published_release(repo),
        behind = lambda installed, latest: is_behind(installed, latest),
        display_tag = lambda marker: marker.get("release_tag"),
        compare_tag = lambda marker: marker.get("release_tag"),
    )


def reset_caches(*, drop_disk: bool = False) -> None:
    """Drop the in-memory freshness caches. The no-arg form is test-only.

    With ``drop_disk = True`` also delete the on-disk 24h release cache. Used by
    the post-install/update path: clearing memory alone leaves the stale
    same-version value on disk, so an offline post-install GitHub refresh would
    replay it (latest_published_release's last-good fallback) and the banner could
    linger. Dropping the disk cache makes latest read None in that offline case,
    so the banner fails open (off) rather than point at the just-replaced build."""
    _flow.reset_caches(
        (_marker_cache, _release_memo, _assets_memo),
        drop_disk = drop_disk,
        cache_dir = lambda: _cache_dir(),
    )
