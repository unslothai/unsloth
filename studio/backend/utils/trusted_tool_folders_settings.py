# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Owner-chosen folders whose files Python and Terminal tool calls may read and run without asking."""

from __future__ import annotations

import os
from typing import Any

TRUSTED_TOOL_FOLDERS_SETTING_KEY = "trusted_tool_folders"
MAX_TRUSTED_TOOL_FOLDERS = 32


def get_trusted_tool_folders() -> list[str]:
    """Raises when the store cannot be read: the settings page must not mistake that for no folders."""
    from storage.studio_db import get_app_setting
    from utils.account_context import OWNER, run_as

    value = run_as(OWNER, get_app_setting, TRUSTED_TOOL_FOLDERS_SETTING_KEY, None)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _stored() -> list[str]:
    try:
        return get_trusted_tool_folders()
    except Exception:  # noqa: BLE001 - an unreadable store trusts nothing
        return []


def trusted_tool_folders_for_caller() -> tuple[str, ...]:
    """The owner's own chats only: a managed account or an API key keeps today's prompts."""
    from utils.account_context import is_owner_context, via_api_key_context

    if not is_owner_context() or via_api_key_context():
        return ()
    return tuple(_stored())


def _refusal(real: str) -> "str | None":
    from core.inference import os_sandbox
    from storage.studio_db import is_denied_system_path

    if os.path.dirname(real) == real:
        return "a drive or filesystem root"
    if os.path.normcase(real) in os_sandbox._never_a_model_library() or is_denied_system_path(real):
        return "a system folder, your home folder, or the folder holding every home"
    if os_sandbox.overlaps_studio_state(real):
        return "Unsloth's own data folder, or a folder holding it"
    # The sandbox would drop it at launch, so saving it would report trust that never applies.
    if not os_sandbox.grantable_trusted_roots((real,)):
        return (
            "a folder holding a credential file (.env, token, .ssh and the like) or a Hugging Face "
            f"token, or more than {os_sandbox._TRUSTED_FOLDER_SCAN_LIMIT} entries to check"
        )
    return None


def set_trusted_tool_folders(value: Any) -> list[str]:
    """Each entry is stored as its real path, so a symlink or `..` spelling cannot widen it later."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("Trusted folders must be a list of folder paths.")
    folders: list[str] = []
    for number, item in enumerate(value, 1):
        text = item.strip()
        if not text:
            continue
        expanded = os.path.expanduser(text)
        if not os.path.isabs(expanded):
            raise ValueError(f"Folder {number} is not an absolute path.")
        real = os.path.realpath(expanded)
        if not os.path.isdir(real):
            raise ValueError(f"Folder {number} does not exist or is not a folder.")
        reason = _refusal(real)
        if reason:
            raise ValueError(f"Folder {number} cannot be trusted: it is {reason}.")
        if real not in folders:
            folders.append(real)
    if len(folders) > MAX_TRUSTED_TOOL_FOLDERS:
        raise ValueError(f"At most {MAX_TRUSTED_TOOL_FOLDERS} trusted folders.")

    from storage.studio_db import upsert_app_settings
    from utils.account_context import OWNER, run_as

    run_as(OWNER, upsert_app_settings, {TRUSTED_TOOL_FOLDERS_SETTING_KEY: folders})
    return folders
