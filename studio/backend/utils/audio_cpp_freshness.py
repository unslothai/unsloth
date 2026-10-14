# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""audio.cpp prebuilt freshness: the newest published release of the configured audio.cpp repo (unslothai/audio.cpp unless UNSLOTH_AUDIO_CPP_REPO says otherwise) and the version policy utils.audio_cpp_update compares the install record against. Fails open on any missing data so we never offer a misleading update. The mechanics (GitHub fetch, memo + disk cache, the shared rate-limit lockout) live in utils.prebuilt.freshness_flow; this module keeps the audio version policy and the per-module caches its tests patch. Unlike whisper it does not use freshness_flow.check_freshness, which reads the repo from the install marker: an install from another repo is checked against the configured one."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

import structlog

from utils.prebuilt import freshness_flow as _flow

logger = structlog.get_logger(__name__)

_release_memo: dict[str, tuple[float, Optional[str]]] = {}
# The combined update status is polled, so an offline host pays the fetch deadline once a minute, not on every poll.
_release_failed_at: dict[str, float] = {}
# Newest-release asset sizes (name -> bytes), memoized like the tag (24h TTL).
_assets_memo: dict[str, tuple[float, dict[str, int]]] = {}
# An offline host with a disk-cached newer tag still asks for its assets: without this every poll pays the fetch deadline.
_assets_failed_at: dict[str, float] = {}

# A leading v?X.Y[.Z], any middle label (``-audio8-perf-hotfix``), then an optional ``-unsloth.N``.
_VERSION = re.compile(r"v?(\d+)\.(\d+)(?:\.(\d+))?(?:-\S*)?")
_UNSLOTH_SERIAL = re.compile(r"-unsloth\.(\d+)$")


def _cache_dir() -> Path:
    """Lazy import so tests can stub storage_roots."""
    try:
        from utils.paths.storage_roots import cache_root
        return cache_root() / "audio_cpp_freshness"
    except Exception:
        return Path.home() / ".unsloth" / "studio" / "cache" / "audio_cpp_freshness"


def _save_disk_cache(repo: str, latest_tag: Optional[str]) -> None:
    _flow.save_disk_cache(
        repo, latest_tag, _cache_dir(), log_message = "audio.cpp freshness cache write failed"
    )


def _fetch_latest_release_tag(repo: str, timeout: float = 5.0) -> Optional[str]:
    """Newest published release tag for `repo`, by publish time (see freshness_flow for why this is not GitHub's /releases/latest pointer)."""
    return _flow.fetch_latest_release_tag(
        repo, timeout, log_message = "audio.cpp freshness fetch failed"
    )


def latest_published_release(repo: str, *, force_refresh: bool = False) -> Optional[str]:
    """Newest release tag for `repo`, with 24h success and 60s failure caches. None when offline and never previously cached."""
    return _flow.latest_published_release(
        repo,
        force_refresh = force_refresh,
        memo = _release_memo,
        failed_at = _release_failed_at,
        cache_dir = lambda: _cache_dir(),
        fetch = lambda r: _fetch_latest_release_tag(r),
        save = lambda r, tag: _save_disk_cache(r, tag),
    )


def cached_latest_release(repo: str) -> Optional[str]:
    """The newest release tag already known for `repo`, from memory or the disk cache whatever its age. Never touches the network: the Audio page status reads this inside an async handler. A stale value can only under-report, since an older "latest" is never behind (see is_behind)."""
    if not repo:
        return None
    cached = _release_memo.get(repo)
    if cached:
        return cached[1]
    disk = _flow.load_disk_cache(repo, _cache_dir())
    return disk[1] if disk else None


def _fetch_latest_release_assets(repo: str, timeout: float = 5.0) -> Optional[dict[str, int]]:
    """Asset name -> size (bytes) for the newest published release of `repo`, selected exactly like _fetch_latest_release_tag. None on any failure."""
    return _flow.fetch_latest_release_assets(
        repo, timeout, log_message = "audio.cpp freshness asset fetch failed"
    )


def latest_release_assets(repo: str, *, force_refresh: bool = False) -> Optional[dict[str, int]]:
    """Newest-release asset sizes for `repo`, memoized (24h TTL) with the same 60s failure cache as the tag. None when offline and never fetched. In-memory only, so a restart simply re-fetches."""
    failed = _assets_failed_at.get(repo)
    if (
        not force_refresh
        and failed is not None
        and time.monotonic() - failed < _flow.RELEASE_FAILURE_CACHE_TTL_SECONDS
    ):
        cached = _assets_memo.get(repo)
        return cached[1] if cached else None
    fetched = []

    def _fetch(r: str) -> Optional[dict[str, int]]:
        assets = _fetch_latest_release_assets(r)
        fetched.append(assets)
        return assets

    assets = _flow.latest_release_assets(
        repo, force_refresh = force_refresh, memo = _assets_memo, fetch = _fetch
    )
    if fetched and fetched[-1] is None:
        _assets_failed_at[repo] = time.monotonic()
    elif fetched:
        _assets_failed_at.pop(repo, None)
    return assets


def parse_release_version(tag: object) -> Optional[tuple]:
    """Comparable key ``(major, minor, patch, unsloth_serial)`` for a tag like ``v0.9.1-unsloth.1``. Tolerant of a missing ``v``, patch or ``-unsloth.N`` (each then 0) and of a middle label such as ``-audio8-perf-hotfix``, which is ignored. None for anything else. Keep in step with release_version_key in install_audio_cpp_prebuilt.py, which runs before the backend can be imported; a parity test holds the two to one table."""
    if not isinstance(tag, str):
        return None
    s = tag.strip()
    m = _VERSION.fullmatch(s)
    if not m:
        return None
    serial = _UNSLOTH_SERIAL.search(s)
    return (
        int(m.group(1)),
        int(m.group(2)),
        int(m.group(3) or 0),
        int(serial.group(1)) if serial else 0,
    )


def is_behind(installed: Optional[str], latest: Optional[str]) -> bool:
    """Whether `installed` is genuinely behind `latest`: identical tags are not behind; when both parse, behind only when the latest key is strictly greater (the downgrade guard: v0.8.2-audio8-perf-hotfix-unsloth.1 published after v0.9.1-unsloth.1 is never offered over it). Unlike whisper, a tag that does not parse is never behind: with no order there is no proof the install is older."""
    if not installed or not latest:
        return False
    installed, latest = installed.strip(), latest.strip()
    if installed == latest:
        return False
    installed_key, latest_key = parse_release_version(installed), parse_release_version(latest)
    if installed_key is None or latest_key is None:
        return False
    return latest_key > installed_key


def reset_caches(*, drop_disk: bool = False) -> None:
    """Drop the in-memory freshness caches. The no-arg form is test-only. With ``drop_disk = True`` also delete the on-disk 24h release cache, used after an install: clearing memory alone leaves the old value on disk, so an offline refresh would replay it and the offer could point at the build just replaced."""
    _flow.reset_caches(
        (_release_memo, _release_failed_at, _assets_memo, _assets_failed_at),
        drop_disk = drop_disk,
        cache_dir = lambda: _cache_dir(),
    )
