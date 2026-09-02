# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Durable record of what an account fetched into an installation-wide cache.

The Hub cache, the model cache and the dictation snapshots are shared by design,
so a private repository one account downloaded with its own token is on disk for
the whole box. Telling its downloader apart from an account that merely read the
name off an inventory is what these grants are for.

Written into the granting account's OWN studio.db rather than held in a process
dictionary. In memory alone the grant did not survive a restart, so the account
that paid for the download was refused its own cached copy on the next boot, and
offline there was no Hub call left to recover it. That failure has now been
reported twice, for datasets and for dictation models, which is why it lives in
one place instead of being solved again per cache.
"""

from __future__ import annotations

_MAX_TRACKED = 512


def _setting(kind: str) -> str:
    return f"workspace_grants:{kind}"


def record_grant(kind: str, key: str, subject: str) -> None:
    """Remember that ``subject`` obtained ``key`` itself."""
    from storage.studio_db import get_app_setting, upsert_app_settings
    from utils.workspace_context import run_in_workspace

    if not isinstance(key, str) or not key.strip():
        return
    entry = key.strip()

    def _write() -> None:
        held = get_app_setting(_setting(kind), []) or []
        if not isinstance(held, list) or entry in held:
            if isinstance(held, list) and entry in held:
                return
            held = []
        upsert_app_settings({_setting(kind): ([*held, entry])[-_MAX_TRACKED:]})

    try:
        run_in_workspace(subject, _write)
    except Exception:  # noqa: BLE001 - a grant that cannot be written is re-earned
        pass                                # by the next download, never fatal to one.


def has_grant(kind: str, key: str) -> bool:
    """Whether the CALLING workspace's own database records this grant."""
    from storage.studio_db import get_app_setting

    if not isinstance(key, str) or not key.strip():
        return False
    try:
        held = get_app_setting(_setting(kind), []) or []
    except Exception:  # noqa: BLE001 - an unreadable database grants nothing
        return False
    return isinstance(held, list) and key.strip() in held


def clear_grants(kind: str, subject: str) -> None:
    """Drop a retired account's grants.

    Retirement renames the workspace, which normally takes these with it, but
    this runs before the rename and the rename is allowed to fail: a grant left
    behind is one a namesake reads back.
    """
    from storage.studio_db import upsert_app_settings
    from utils.workspace_context import run_in_workspace

    try:
        run_in_workspace(subject, upsert_app_settings, {_setting(kind): []})
    except Exception:  # noqa: BLE001 - a database that cannot be written holds no
        pass                                # grant this process will go on to read.
