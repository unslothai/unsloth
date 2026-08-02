# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama.cpp prebuilt freshness check.

Reads UNSLOTH_PREBUILT_INFO.json (written by install_llama_prebuilt.py)
and compares the installed release tag against the latest on GitHub.
Surfaced via main.py:lifespan() and /api/inference/status. Fails open
on any missing data so we never show a misleading banner.

The mechanics (marker walk-up, GitHub fetch, memo + disk cache, report
skeleton) live in utils.prebuilt.freshness_flow; this module keeps the
llama version policy and the per-module caches its tests patch.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

from utils.prebuilt import freshness_flow as _flow

logger = structlog.get_logger(__name__)

# 3 days matches Unsloth's typical llama.cpp release cadence.
STALENESS_THRESHOLD_DAYS = 3

_INSTALL_MARKER_NAME = "UNSLOTH_PREBUILT_INFO.json"

_marker_cache: dict[str, Optional[dict]] = {}
_release_memo: dict[str, tuple[float, Optional[str]]] = {}
# Newest-release asset sizes (name -> bytes), memoized like the tag (24h TTL).
_assets_memo: dict[str, tuple[float, dict[str, int]]] = {}


def _cache_dir() -> Path:
    """Lazy import so tests can stub storage_roots."""
    try:
        from utils.paths.storage_roots import cache_root
        return cache_root() / "llama_cpp_freshness"
    except Exception:
        return Path.home() / ".unsloth" / "studio" / "cache" / "llama_cpp_freshness"


def read_install_marker(binary_path: Optional[str]) -> Optional[dict]:
    """Walk up from binary_path to find UNSLOTH_PREBUILT_INFO.json.
    None = no marker (source build / custom path) or invalid JSON."""
    return _flow.read_install_marker(
        binary_path,
        marker_name = _INSTALL_MARKER_NAME,
        cache = _marker_cache,
        log_message = "failed to parse install marker",
    )


def _load_disk_cache(repo: str) -> Optional[tuple[float, Optional[str]]]:
    return _flow.load_disk_cache(repo, _cache_dir())


def _save_disk_cache(repo: str, latest_tag: Optional[str]) -> None:
    _flow.save_disk_cache(
        repo, latest_tag, _cache_dir(), log_message = "freshness cache write failed"
    )


def _fetch_latest_release_tag(repo: str, timeout: float = 5.0) -> Optional[str]:
    """Newest published release tag for `repo`, by publish time (see
    freshness_flow for why this is not GitHub's /releases/latest pointer)."""
    return _flow.fetch_latest_release_tag(repo, timeout, log_message = "freshness fetch failed")


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
        repo, timeout, log_message = "freshness asset fetch failed"
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


def update_download_size_bytes(
    marker: Optional[dict],
    latest_tag: Optional[str],
    repo: Optional[str],
    *,
    force_refresh: bool = False,
) -> Optional[int]:
    """Download size of the latest-release asset matching this host's installed
    bundle (same platform/arch/runtime suffix as the installed asset). None when
    there is no marker asset, the latest assets can't be read, or no match."""
    if not marker or not latest_tag or not repo:
        return None
    installed_asset = marker.get("asset")
    if not isinstance(installed_asset, str):
        return None
    # Tag-independent platform suffix: accept the fork's "app-*" bundles and the
    # upstream ggml-org "ubuntu-*"/"win-*" prebuilts ("windows" before "win").
    m = re.search(r"-((?:linux|ubuntu|windows|win|macos|darwin)-.*)$", installed_asset)
    if not m:
        return None
    suffix = m.group(1)
    # Upstream ubuntu/win assets live in the marker's binary_repo, not the fork
    # publish repo; try the publish repo first, then it.
    repos = [repo]
    binary_repo = marker.get("binary_repo")
    if isinstance(binary_repo, str) and binary_repo and binary_repo != repo:
        repos.append(binary_repo)
    want = f"app-{latest_tag}-{suffix}"
    for r in repos:
        assets = latest_release_assets(r, force_refresh = force_refresh)
        if not assets:
            continue
        if want in assets:
            return assets[want]
        # Tag formatting can vary (mix suffixes); fall back to the platform suffix.
        for name, size in assets.items():
            if name.endswith(suffix):
                return size
    return None


def _parse_installed_at(value: object) -> Optional[datetime]:
    return _flow.parse_installed_at(value)


def parse_base_build(tag: object) -> Optional[int]:
    """Numeric base build from a release tag. Handles both a plain ``bNNNN`` and
    a mix-build tag like ``b9596-mix-<sha>`` (anchored at the start, so the mix
    suffix doesn't defeat it). None for anything not starting with ``bNNNN``."""
    if not isinstance(tag, str):
        return None
    m = re.match(r"b(\d+)", tag.strip())
    return int(m.group(1)) if m else None


def is_behind(installed: Optional[str], latest: Optional[str]) -> bool:
    """Whether `installed` is genuinely behind `latest`, comparing the FULL
    release identity (so a mix build can legitimately be the latest) with a
    base-build guard so a lagging GitHub /releases/latest can never read as an
    update or a downgrade.

    - identical tags -> not behind (clears the sticky banner post-update)
    - higher base build on `latest` -> behind; lower -> NOT behind (downgrade guard)
    - same base build: a different/new mix -> behind, but a bare ``bNNNN`` never
      supersedes a mix build (extra PRs) at that base -> not behind
    - non-bNNNN tags -> behind (plain inequality, since they already differ)
    """
    if not installed or not latest:
        return False
    installed, latest = installed.strip(), latest.strip()
    if installed == latest:
        return False
    ib, lb = parse_base_build(installed), parse_base_build(latest)
    if ib is None or lb is None:
        return True
    if lb != ib:
        return lb > ib
    # Same base build, different tags: offer a mix (latest carries a suffix), but
    # never offer a bare base over a mix install at the same base.
    return latest != f"b{lb}"


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
    # The marker records both a normalized base tag ("tag", e.g. b9596) and the
    # full release tag ("release_tag", e.g. b9596-mix-<sha>). Display prefers the
    # normalized base; comparison uses the FULL identity, since GitHub
    # /releases/latest returns the full tag_name -- comparing the normalized base
    # against the full latest is what produced the permanent "downgrade" banner
    # on every mix release. Deliberately opposite fallbacks.
    return _flow.check_freshness(
        binary_path,
        threshold_days = threshold_days,
        now = now,
        read_marker = lambda p: read_install_marker(p),
        latest_release = lambda repo: latest_published_release(repo),
        behind = lambda installed, latest: is_behind(installed, latest),
        display_tag = lambda marker: marker.get("tag") or marker.get("release_tag"),
        compare_tag = lambda marker: marker.get("release_tag") or marker.get("tag"),
    )


def format_stale_warning(info: dict) -> str:
    """Human-readable one-liner for stale prebuilt info."""
    return _flow.format_stale_warning(info, component = "llama.cpp")


def reset_caches(*, drop_disk: bool = False) -> None:
    """Drop the in-memory freshness caches. The no-arg form is test-only.

    With ``drop_disk = True`` also delete the on-disk 24h release cache. Used by
    the post-install/update path: in-memory clearing alone leaves the stale
    same-base value on disk, so if the post-install GitHub refresh can't reach
    the network, ``latest_published_release`` would replay that stale disk value
    (see its last-good fallback) and the banner could linger. Dropping the disk
    cache makes latest read as None in that offline case, so the banner fails
    open (off) instead of pointing at the just-replaced build."""
    _flow.reset_caches(
        (_marker_cache, _release_memo, _assets_memo),
        drop_disk = drop_disk,
        cache_dir = lambda: _cache_dir(),
    )
