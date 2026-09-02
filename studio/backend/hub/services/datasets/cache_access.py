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

# One kind of grant among several; see utils/workspace_grants for the durable half.
_GRANT_KIND = "dataset"


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
    from utils.workspace_grants import record_grant

    record_grant(_GRANT_KIND, repo_id, subject)


def forget_workspace(subject: str) -> None:
    """Drop an account's grants and pending attempts, for retirement: the subject is
    a reusable username, so a surviving grant lets a namesake preview the previous
    holder's cached private datasets."""
    from utils.workspace_grants import clear_grants

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
    clear_grants(_GRANT_KIND, subject)


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
    from utils.workspace_grants import has_grant

    if has_grant(_GRANT_KIND, key):
        return True
    from routes.inference import _hub_repo_is_anonymously_readable

    # Confirmed public, not merely unrefuted: offline, rate limited and any other
    # transient failure all answer None, and treating that as permission handed
    # over the inventory and the cached rows of a private dataset for exactly as
    # long as the Hub was unreachable.
    return _hub_repo_is_anonymously_readable(key, "dataset") is True
