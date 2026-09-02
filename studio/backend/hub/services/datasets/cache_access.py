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


def note_dataset_downloader(repo_id: str) -> None:
    from utils.workspace_context import current_workspace_subject

    if not isinstance(repo_id, str) or not repo_id.strip():
        return
    key = repo_id.strip()
    with _lock:
        _dataset_downloaders.setdefault(key, set()).add(current_workspace_subject())
        while len(_dataset_downloaders) > _MAX_TRACKED:
            _dataset_downloaders.pop(next(iter(_dataset_downloaders)))


def caller_may_read_cached_dataset(repo_id: Optional[str]) -> bool:
    """Whether this account could have obtained this dataset itself.

    True for the owner, for an account that downloaded it here, and for a
    repository the Hub serves anonymously, which is the ordinary case. False
    only for a private or gated repository somebody else pulled into the shared
    cache: reading its rows, or even listing it, is that account's disclosure.
    An unanswerable Hub is treated as public, so an offline installation keeps
    working exactly as it did.
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
    from routes.inference import _hub_repo_is_anonymously_readable

    return _hub_repo_is_anonymously_readable(key, "dataset") is not False
