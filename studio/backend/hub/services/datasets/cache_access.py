# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Who may read an installation-wide cached dataset."""

from __future__ import annotations

import threading
from typing import Optional

# Dataset repo -> the accounts that downloaded it on this installation. The Hub
# cache is shared by design, so a private dataset one account fetched with its
# own token is on disk for the whole box; this is what tells its downloader
# apart from an account that merely read the repository name off the inventory.
_dataset_downloaders: dict[str, set[str]] = {}
_lock = threading.Lock()
_MAX_TRACKED = 512


# Job key -> (repo, workspace) for a download that has started but not finished.
# Starting one proves nothing: the worker has not authenticated yet, so recording
# the grant here let a caller with any nonempty token defeat the gate simply by
# asking for a doomed download of a repository somebody else had already cached.
_pending_downloads: dict[str, tuple[str, str]] = {}


def note_dataset_download_attempt(key: str, repo_id: str) -> None:
    """Remember who asked for this job, pending the worker actually succeeding."""
    from utils.workspace_context import current_workspace_subject

    if not isinstance(repo_id, str) or not repo_id.strip():
        return
    with _lock:
        _pending_downloads[key] = (repo_id.strip(), current_workspace_subject())
        while len(_pending_downloads) > _MAX_TRACKED:
            _pending_downloads.pop(next(iter(_pending_downloads)))


def confirm_dataset_download(key: str) -> None:
    """Grant the access a finished download earned. Only the account whose own job
    completed: an adopter never recorded an attempt, so joining somebody's
    transfer does not inherit their credential's reach."""
    with _lock:
        pending = _pending_downloads.pop(key, None)
        if pending is None:
            return
        repo_id, subject = pending
        _dataset_downloaders.setdefault(repo_id, set()).add(subject)
        while len(_dataset_downloaders) > _MAX_TRACKED:
            _dataset_downloaders.pop(next(iter(_dataset_downloaders)))
    _persist_grant(repo_id, subject)


# The grant, written into the downloading account's OWN studio.db. In memory
# alone it did not survive a restart, so the account that fetched a private
# dataset with its own token lost its cached copy on the next boot, and offline
# there was no Hub call left to recover it. Retirement renames the workspace,
# which takes the persisted grants with it.
_GRANTS_SETTING = "dataset_cache_grants"


def _persist_grant(repo_id: str, subject: str) -> None:
    from storage.studio_db import get_app_setting, upsert_app_settings
    from utils.workspace_context import run_in_workspace

    def _write() -> None:
        held = get_app_setting(_GRANTS_SETTING, []) or []
        if not isinstance(held, list):
            held = []
        if repo_id in held:
            return
        upsert_app_settings({_GRANTS_SETTING: ([*held, repo_id])[-_MAX_TRACKED:]})

    try:
        run_in_workspace(subject, _write)
    except Exception:  # noqa: BLE001 - a grant that cannot be written is re-earned
        pass  # by the next download; never fail one over it.


def _persisted_grant(repo_id: str) -> bool:
    """Whether this workspace's own database records the download."""
    from storage.studio_db import get_app_setting

    try:
        held = get_app_setting(_GRANTS_SETTING, []) or []
    except Exception:  # noqa: BLE001 - an unreadable database grants nothing
        return False
    return isinstance(held, list) and repo_id in held


def forget_workspace(subject: str) -> None:
    """Drop an account's grants and pending attempts, for retirement: the subject is
    a reusable username, so a surviving grant lets a namesake preview the previous
    holder's cached private datasets."""
    with _lock:
        for repo_id in list(_dataset_downloaders):
            holders = _dataset_downloaders.get(repo_id)
            if not holders:
                continue
            holders.discard(subject)
            if not holders:
                _dataset_downloaders.pop(repo_id, None)
        for key in [key for key, value in _pending_downloads.items() if value[1] == subject]:
            _pending_downloads.pop(key, None)
    # The written grants too. Retirement renames the workspace, which normally
    # takes them with it, but this runs before the rename and the rename is
    # allowed to fail: a grant left behind is one a namesake reads back.
    _clear_persisted_grants(subject)


def _clear_persisted_grants(subject: str) -> None:
    from storage.studio_db import upsert_app_settings
    from utils.workspace_context import run_in_workspace
    try:
        run_in_workspace(subject, upsert_app_settings, {_GRANTS_SETTING: []})
    except Exception:  # noqa: BLE001 - a database that cannot be written holds no
        pass  # grant this process will go on to read.


def caller_may_read_cached_dataset(repo_id: Optional[str]) -> bool:
    """Whether this account could have obtained this dataset itself.

    True for the owner, for whoever downloaded it here (in this process or in an
    earlier one, which is what the persisted grant answers), and for a repo the
    Hub confirms it serves anonymously. False for one somebody else pulled into
    the shared cache, and for one whose visibility cannot be established: an
    unreachable Hub withholds rather than guesses.
    """
    from auth.storage import is_installation_owner
    from utils.workspace_context import LEGACY_WORKSPACE_SUBJECT, current_workspace_subject

    if not isinstance(repo_id, str) or not repo_id.strip():
        return True
    subject = current_workspace_subject()
    if subject == LEGACY_WORKSPACE_SUBJECT or is_installation_owner():
        return True
    key = repo_id.strip()
    with _lock:
        downloaders = _dataset_downloaders.get(key)
    if downloaders and subject in downloaders:
        return True
    if _persisted_grant(key):
        return True
    from routes.inference import _hub_repo_is_anonymously_readable

    # Confirmed public, not merely unrefuted: offline, rate limited and any other
    # transient failure all answer None, and treating that as permission handed
    # over the inventory and the cached rows of a private dataset for exactly as
    # long as the Hub was unreachable.
    return _hub_repo_is_anonymously_readable(key, "dataset") is True
