# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Remote manifest fetch and update-status cache for Unsloth Studio.

Uses only stdlib (urllib.request, json, time, calendar) so it can run
without any third-party dependencies.  The module exposes two public
functions:

  fetch_and_cache_update_status()  -- fetches the manifest, reads the
      local UNSLOTH_STUDIO_INFO.json, compares CRITICAL_TIME, and caches
      the result at module level.

  get_update_status()  -- returns the cached UpdateStatus (or defaults
      when the fetch has not completed yet).
"""

from __future__ import annotations

import calendar
import json
import time
import urllib.request
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

_MANIFEST_URL = (
    "https://raw.githubusercontent.com/unslothai/unsloth/main/"
    "UNSLOTH_UPDATE_DETAILS.json"
)
_STUDIO_INFO_PATH = Path.home() / ".unsloth" / "studio" / "UNSLOTH_STUDIO_INFO.json"
_FETCH_TIMEOUT = 8  # seconds


@dataclass
class UpdateStatus:
    critical: bool = False
    announcement_badge: Optional[str] = None
    announcement_message: Optional[str] = None
    announcement_url: Optional[str] = None
    manifest_fetched: bool = False


_cached_status: UpdateStatus = UpdateStatus()


def _parse_iso_utc(s: str) -> float:
    """Parse an ISO-8601 UTC string (ending in Z) to a Unix timestamp."""
    s = s.strip().rstrip("Z")
    try:
        t = time.strptime(s, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        t = time.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
    return float(calendar.timegm(t))


def fetch_and_cache_update_status() -> UpdateStatus:
    """Fetch the remote manifest, compare with local info, and cache."""
    global _cached_status

    try:
        req = urllib.request.Request(_MANIFEST_URL, method = "GET")
        with urllib.request.urlopen(req, timeout = _FETCH_TIMEOUT) as resp:
            manifest = json.loads(resp.read().decode("utf-8"))
    except Exception:
        _log.debug("manifest fetch failed", exc_info = True)
        return _cached_status

    status = UpdateStatus(manifest_fetched = True)

    # -- Critical time check --
    critical_time_str = manifest.get("CRITICAL_TIME")
    if critical_time_str:
        try:
            critical_ts = _parse_iso_utc(critical_time_str)
            installed_ts = 0.0
            if _STUDIO_INFO_PATH.is_file():
                try:
                    info = json.loads(_STUDIO_INFO_PATH.read_text(encoding = "utf-8"))
                    installed_ts = _parse_iso_utc(info.get("installed_at_utc", ""))
                except Exception:
                    _log.debug("failed to read studio info", exc_info = True)
            if installed_ts < critical_ts:
                status.critical = True
        except Exception:
            _log.debug("critical time comparison failed", exc_info = True)

    # -- Announcement --
    announcement = manifest.get("announcement")
    if isinstance(announcement, dict):
        status.announcement_badge = announcement.get("badge") or None
        status.announcement_message = announcement.get("message") or None
        status.announcement_url = announcement.get("url") or None

    _cached_status = status
    return status


def get_update_status() -> UpdateStatus:
    """Return the cached update status (safe to call before fetch completes)."""
    return _cached_status
