# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Web update status helpers for browser-served Unsloth Studio.

This module is intentionally side-effect light: no network work happens at
import time or from /api/health. The PyPI check is lazy, cached, and only used
for normal PyPI-managed installs.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

PACKAGE_NAME = "unsloth"
PYPI_JSON_URL = "https://pypi.org/pypi/unsloth/json"
PYPI_TIMEOUT_SECONDS = 3
PYPI_RESPONSE_MAX_BYTES = 5 * 1024 * 1024
PYPI_SUCCESS_TTL_SECONDS = 12 * 60 * 60
PYPI_FAILURE_TTL_SECONDS = 60 * 60
RELEASE_NOTES_URL = "https://unsloth.ai/docs/new/changelog"
DISABLE_ENV_VAR = "UNSLOTH_DISABLE_UPDATE_CHECK"

LOCAL_INSTALL_SOURCES = {"editable", "local_path", "vcs", "local_repo"}


@dataclass(frozen = True)
class LatestVersionResult:
    latest_version: str | None
    checked_at: str
    reason: str | None = None
    error: str | None = None


@dataclass
class _LatestVersionCacheEntry:
    result: LatestVersionResult
    expires_at: float


_cache_condition = threading.Condition()
_latest_version_cache: _LatestVersionCacheEntry | None = None
_latest_version_fetching = False


def reset_update_status_cache() -> None:
    """Clear the in-process PyPI cache. Intended for tests."""
    global _latest_version_cache, _latest_version_fetching
    with _cache_condition:
        _latest_version_cache = None
        _latest_version_fetching = False
        _cache_condition.notify_all()


def detect_install_source() -> str:
    """Return a coarse install source without exposing local paths.

    Sources are intentionally conservative. PEP 610 local/vcs metadata wins.
    Legacy source installs are treated as local only when package files resolve
    outside site-packages/dist-packages and under a Git checkout.
    """
    try:
        dist = distribution(PACKAGE_NAME)
    except PackageNotFoundError:
        return (
            "local_repo"
            if _path_has_git_parent(_repo_root_from_this_file())
            else "unknown"
        )

    try:
        direct_url = dist.read_text("direct_url.json")
    except Exception:
        return "unknown"
    if direct_url:
        return _source_from_direct_url(direct_url)

    for package_path in _distribution_package_paths(dist):
        if not _path_is_under_python_package_dir(package_path) and _path_has_git_parent(
            package_path
        ):
            return "local_repo"

    return "pypi"


def get_studio_install_source_status(current_version: str) -> dict[str, Any]:
    """Return install-source metadata without remote update checks."""
    install_source = detect_install_source()
    reason = None
    if install_source in LOCAL_INSTALL_SOURCES:
        reason = "local_source"
    elif install_source == "unknown":
        reason = "unknown_source"

    return _status_response(
        current_version = current_version,
        latest_version = None,
        install_source = install_source,
        reason = reason,
    )


def get_studio_update_status(current_version: str) -> dict[str, Any]:
    """Return public, read-only update status for the web UI."""
    install_source = detect_install_source()

    if os.environ.get(DISABLE_ENV_VAR) == "1":
        return _status_response(
            current_version = current_version,
            latest_version = None,
            install_source = install_source,
            reason = "disabled",
        )

    if install_source in LOCAL_INSTALL_SOURCES:
        return _status_response(
            current_version = current_version,
            latest_version = None,
            install_source = install_source,
            reason = "local_source",
        )

    if install_source != "pypi":
        return _status_response(
            current_version = current_version,
            latest_version = None,
            install_source = install_source,
            reason = "unknown_source",
        )

    current = _parse_current_version(current_version)
    if current is None:
        return _status_response(
            current_version = current_version,
            latest_version = None,
            install_source = install_source,
            reason = "invalid_current_version"
            if current_version != "dev"
            else "dev_build",
        )
    latest_result = get_latest_pypi_version()
    if latest_result.latest_version is None:
        return _status_response(
            current_version = current_version,
            latest_version = None,
            install_source = install_source,
            reason = latest_result.reason or "offline",
            error = latest_result.error,
            checked_at = latest_result.checked_at,
        )

    try:
        latest = Version(latest_result.latest_version)
    except InvalidVersion:
        return _status_response(
            current_version = current_version,
            latest_version = latest_result.latest_version,
            install_source = install_source,
            reason = "invalid_latest_version",
            error = "PyPI returned an invalid version.",
            checked_at = latest_result.checked_at,
        )

    if latest > current:
        return _status_response(
            current_version = current_version,
            latest_version = latest_result.latest_version,
            install_source = install_source,
            update_available = True,
            can_show_web_notification = True,
            checked_at = latest_result.checked_at,
        )

    return _status_response(
        current_version = current_version,
        latest_version = latest_result.latest_version,
        install_source = install_source,
        reason = "current_not_older",
        checked_at = latest_result.checked_at,
    )


def get_latest_pypi_version() -> LatestVersionResult:
    """Return the latest PyPI version using a small in-process TTL cache."""
    global _latest_version_cache, _latest_version_fetching

    while True:
        now = time.monotonic()
        with _cache_condition:
            if _latest_version_cache and _latest_version_cache.expires_at > now:
                return _latest_version_cache.result
            if not _latest_version_fetching:
                _latest_version_fetching = True
                break
            _cache_condition.wait(timeout = PYPI_TIMEOUT_SECONDS + 1)

    try:
        result = _fetch_latest_pypi_version()
    except Exception:
        result = LatestVersionResult(
            latest_version = None,
            checked_at = _utc_now_iso(),
            reason = "offline",
            error = "Could not check PyPI update metadata.",
        )

    ttl = (
        PYPI_SUCCESS_TTL_SECONDS if result.latest_version else PYPI_FAILURE_TTL_SECONDS
    )
    with _cache_condition:
        _latest_version_cache = _LatestVersionCacheEntry(
            result = result,
            expires_at = time.monotonic() + ttl,
        )
        _latest_version_fetching = False
        _cache_condition.notify_all()
    return result


def _fetch_latest_pypi_version() -> LatestVersionResult:
    checked_at = _utc_now_iso()
    request = urllib.request.Request(
        PYPI_JSON_URL,
        headers = {"User-Agent": "unsloth-studio-update-check"},
    )

    try:
        with urllib.request.urlopen(request, timeout = PYPI_TIMEOUT_SECONDS) as response:
            body = response.read(PYPI_RESPONSE_MAX_BYTES + 1)
        if len(body) > PYPI_RESPONSE_MAX_BYTES:
            return LatestVersionResult(
                latest_version = None,
                checked_at = checked_at,
                reason = "malformed_response",
                error = "PyPI returned oversized update metadata.",
            )
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return LatestVersionResult(
            latest_version = None,
            checked_at = checked_at,
            reason = "malformed_response",
            error = "PyPI returned malformed update metadata.",
        )
    except OSError:
        return LatestVersionResult(
            latest_version = None,
            checked_at = checked_at,
            reason = "offline",
            error = "Could not reach PyPI for update metadata.",
        )

    latest = (
        payload.get("info", {}).get("version") if isinstance(payload, dict) else None
    )
    if not isinstance(latest, str) or not latest.strip():
        return LatestVersionResult(
            latest_version = None,
            checked_at = checked_at,
            reason = "malformed_response",
            error = "PyPI update metadata did not include a version.",
        )

    return LatestVersionResult(latest_version = latest.strip(), checked_at = checked_at)


def _status_response(
    *,
    current_version: str,
    latest_version: str | None,
    install_source: str,
    reason: str | None = None,
    error: str | None = None,
    update_available: bool = False,
    can_show_web_notification: bool = False,
    checked_at: str | None = None,
) -> dict[str, Any]:
    return {
        "current_version": current_version,
        "latest_version": latest_version,
        "update_available": update_available,
        "install_source": install_source,
        "can_show_web_notification": can_show_web_notification,
        "release_notes_url": RELEASE_NOTES_URL,
        "checked_at": checked_at or _utc_now_iso(),
        "reason": reason,
        "error": error,
    }


def _source_from_direct_url(direct_url: str) -> str:
    try:
        payload = json.loads(direct_url)
    except json.JSONDecodeError:
        return "unknown"

    if not isinstance(payload, dict):
        return "unknown"

    dir_info = payload.get("dir_info")
    if isinstance(dir_info, dict) and dir_info.get("editable") is True:
        return "editable"

    if isinstance(payload.get("vcs_info"), dict):
        return "vcs"

    url = payload.get("url")
    if isinstance(url, str) and url.startswith("file:"):
        return "local_path"

    return "unknown"


def _distribution_package_paths(dist: Any) -> list[Path]:
    paths: list[Path] = []
    files = getattr(dist, "files", None) or []
    for file in files:
        text = str(file)
        if not text.startswith(("unsloth/", "unsloth_cli/", "studio/")):
            continue
        try:
            paths.append(Path(dist.locate_file(file)).resolve())
        except OSError:
            continue
    return paths


def _path_is_under_python_package_dir(path: Path) -> bool:
    return any(part in {"site-packages", "dist-packages"} for part in path.parts)


def _path_has_git_parent(path: Path) -> bool:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return True
    return False


def _repo_root_from_this_file() -> Path:
    # update_status.py -> utils -> backend -> studio -> repo root
    try:
        return Path(__file__).resolve().parents[3]
    except IndexError:
        return Path(__file__).resolve().parent


def _parse_current_version(current_version: str) -> Version | None:
    if current_version == "dev":
        return None
    try:
        return Version(current_version)
    except InvalidVersion:
        return None


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond = 0)
        .isoformat()
        .replace("+00:00", "Z")
    )
