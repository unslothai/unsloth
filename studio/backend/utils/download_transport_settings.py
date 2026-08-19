# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The install's default download transport: HTTPS, Xet, or Auto.

New installs get HTTPS; installs from before this default keep Auto, what they already ran on.
Seeded once and persisted, since the evidence below grows as an install is used.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from hub.utils.hf_cache_state import TRANSPORT_AUTO, TRANSPORT_HTTP, VALID_TRANSPORT_MODES
from loggers import get_logger

logger = get_logger(__name__)

DOWNLOAD_TRANSPORT_SETTING_KEY = "download_transport_mode"
# Resumes a cancelled transfer, and needs nothing but ordinary TLS.
DEFAULT_DOWNLOAD_TRANSPORT = TRANSPORT_HTTP
# What an install from before this change was already on.
LEGACY_DOWNLOAD_TRANSPORT = TRANSPORT_AUTO

# One row in any of these means the install has been used.
_PRIOR_USE_TABLES = ("app_settings", "chat_threads", "training_runs")

_seed_lock = threading.Lock()
_seeded_default: Optional[str] = None


def normalize_transport_mode(value: Any) -> Optional[str]:
    """``value`` as one of http/xet/auto, or None when it is not a transport mode."""
    if not isinstance(value, str):
        return None
    mode = value.strip().lower()
    return mode if mode in VALID_TRANSPORT_MODES else None


def _has_prior_studio_use() -> bool:
    """Whether this install was used before the default changed.

    Positive evidence only: a row the user wrote, or a manifest from an earlier download. Anything
    unreadable answers False, so grandfathering never rests on a guess.
    """
    try:
        from storage.studio_db import get_connection

        conn = get_connection()
        try:
            for table in _PRIOR_USE_TABLES:
                exists = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
                    (table,),
                ).fetchone()
                if exists is None:
                    continue
                if conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None:
                    return True
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - an unreadable db must not block a download
        logger.debug("download transport seed: studio.db unreadable (%s)", exc)

    try:
        from hub.utils import state_dir

        manifests = state_dir.manifests_dir()
        if manifests is not None and manifests.is_dir():
            return any(manifests.iterdir())
    except Exception as exc:  # noqa: BLE001 - same
        logger.debug("download transport seed: manifests unreadable (%s)", exc)
    return False


def seeded_default_transport_mode() -> str:
    """The mode an install with nothing stored gets, decided once per process."""
    global _seeded_default
    if _seeded_default is not None:
        return _seeded_default
    with _seed_lock:
        if _seeded_default is None:
            _seeded_default = (
                LEGACY_DOWNLOAD_TRANSPORT
                if _has_prior_studio_use()
                else DEFAULT_DOWNLOAD_TRANSPORT
            )
    return _seeded_default


def get_download_transport_mode() -> str:
    """The install's download transport, seeding and persisting one on first read."""
    try:
        from storage.studio_db import get_app_setting

        stored = normalize_transport_mode(get_app_setting(DOWNLOAD_TRANSPORT_SETTING_KEY, None))
    except Exception as exc:  # noqa: BLE001 - fall back rather than fail a download
        logger.debug("download transport read failed (%s)", exc)
        stored = None
    if stored is not None:
        return stored

    seeded = seeded_default_transport_mode()
    try:
        from storage.studio_db import upsert_app_settings

        upsert_app_settings({DOWNLOAD_TRANSPORT_SETTING_KEY: seeded})
        logger.info("Seeded the download transport for this install: %s", seeded)
    except Exception as exc:  # noqa: BLE001 - an unwritable db just re-seeds next time
        logger.debug("download transport seed not persisted (%s)", exc)
    return seeded


def set_download_transport_mode(value: Any) -> str:
    """Persist the install's download transport. Raises ValueError on anything else."""
    mode = normalize_transport_mode(value)
    if mode is None:
        raise ValueError("Download transport must be one of: http, xet, auto.")

    from storage.studio_db import upsert_app_settings

    upsert_app_settings({DOWNLOAD_TRANSPORT_SETTING_KEY: mode})
    return mode


def reset_seed_cache_for_tests() -> None:
    """Drop the per-process seed so a test can set up a different install."""
    global _seeded_default
    with _seed_lock:
        _seeded_default = None
