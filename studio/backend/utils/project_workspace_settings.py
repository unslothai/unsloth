# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The owner's saved creation location; existing project root paths are never moved."""

from __future__ import annotations

import os
import stat
import tempfile
import threading
from pathlib import Path

from utils.account_context import OWNER, is_owner_context, run_as

PROJECTS_HOME_SETTING_KEY = "projects_home"
PROJECTS_HOME_HISTORY_SETTING_KEY = "projects_home_history"
PROJECTS_HOME_ENV = "UNSLOTH_STUDIO_PROJECTS_HOME"
_settings_lock = threading.RLock()


def _default_projects_home() -> Path:
    from utils.paths.storage_roots import documents_root
    return documents_root() / "Unsloth Studio" / "Projects"


def _stored_settings() -> dict:
    from utils.paths.storage_roots import studio_db_path

    # Resolving paths must not create a database during installation or a first launch.
    # Only an absent database means defaults: permission/I/O errors must not silently
    # reroute a new project or remove a private location from confinement.
    try:
        os.stat(run_as(OWNER, studio_db_path))
    except FileNotFoundError:
        return {}
    from storage.studio_db import get_app_settings

    return run_as(
        OWNER,
        get_app_settings,
        [PROJECTS_HOME_SETTING_KEY, PROJECTS_HOME_HISTORY_SETTING_KEY],
    )


def _saved_path(settings: dict) -> Path | None:
    value = settings.get(PROJECTS_HOME_SETTING_KEY)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or not Path(value).is_absolute():
        raise ValueError("The saved projects folder is invalid. Choose a folder in Settings.")
    return Path(value)


def get_projects_home_status() -> dict:
    override = (os.environ.get(PROJECTS_HOME_ENV) or "").strip()
    if override:
        home, source = Path(override).expanduser(), "environment"
    else:
        saved = _saved_path(_stored_settings())
        home, source = (
            (saved, "studio") if saved is not None else (_default_projects_home(), "default")
        )
    return {
        "projects_home": str(home),
        "source": source,
        "editable": source != "environment",
        "is_custom": source == "studio",
        "environment_variable": PROJECTS_HOME_ENV if source == "environment" else None,
    }


def owner_project_workspaces_roots() -> tuple[Path, ...]:
    """Locations retained for Library and protection, not grants to read their contents."""
    settings = _stored_settings()
    roots = [_default_projects_home()]
    override = (os.environ.get(PROJECTS_HOME_ENV) or "").strip()
    if override:
        roots.append(Path(override).expanduser())
    saved = _saved_path(settings)
    if saved is not None:
        roots.append(saved)
    history = settings.get(PROJECTS_HOME_HISTORY_SETTING_KEY) or []
    if not isinstance(history, list) or any(
        not isinstance(value, str) or not Path(value).is_absolute() for value in history
    ):
        raise ValueError("The saved projects folder history is invalid.")
    roots.extend(Path(value) for value in history)
    return tuple(dict.fromkeys(roots))


def _validate_projects_home(value: str) -> Path:
    from core.inference.tool_confinement import project_home_overlaps_tool_read_roots
    from core.inference.tools import shared_sandbox_root
    from storage.studio_db import (
        contains_sensitive_path_component,
        is_denied_system_path,
        list_chat_project_root_paths,
    )
    from utils.hf_cache_settings import known_hf_cache_homes, known_hf_hub_caches
    from utils.paths.path_utils import is_network_or_device_path, is_path_within
    from utils.paths.storage_roots import (
        shared_project_workspaces_root,
        shared_tmp_root,
        studio_root,
    )

    if is_network_or_device_path(value.strip()):
        raise ValueError("Choose a local projects folder.")
    candidate = Path(value.strip()).expanduser()
    if not value.strip() or not candidate.is_absolute():
        raise ValueError("Choose an absolute path to an existing projects folder.")
    try:
        candidate = candidate.resolve(strict = True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("Choose an existing, accessible projects folder.") from exc
    if is_network_or_device_path(candidate):
        raise ValueError("Choose a local projects folder.")
    try:
        mode = candidate.stat().st_mode
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("Choose an existing, accessible projects folder.") from exc
    if not stat.S_ISDIR(mode):
        raise ValueError("The selected projects location is not a folder.")
    if candidate.parent == candidate or candidate == Path.home().resolve():
        raise ValueError("Choose a projects folder inside your home folder or a drive.")
    if is_denied_system_path(str(candidate)) or contains_sensitive_path_component(str(candidate)):
        raise ValueError("System, credential and config folders cannot be used for projects.")
    if project_home_overlaps_tool_read_roots(candidate):
        raise ValueError(
            "Choose a projects folder outside software and interpreter folders shared with tool processes."
        )
    for reserved in (
        studio_root(),
        shared_tmp_root(),
        shared_sandbox_root(),
        shared_project_workspaces_root() / "Accounts",
        *known_hf_cache_homes(),
        *known_hf_hub_caches(),
    ):
        reserved = Path(reserved).resolve()
        if is_path_within(str(candidate), str(reserved), allow_root = True) or is_path_within(
            str(reserved), str(candidate), allow_root = True
        ):
            raise ValueError(
                "Choose a projects folder outside Unsloth's internal, account and model cache folders."
            )
    # Deleting a project recursively removes its managed workspace. Nesting a new
    # project's home there would also remove the new project's files.
    for root in list_chat_project_root_paths():
        if root and is_path_within(str(candidate), os.path.realpath(root), allow_root = True):
            raise ValueError("Choose a projects folder outside existing project workspaces.")
    try:
        with tempfile.NamedTemporaryFile(prefix = ".unsloth-write-test-", dir = candidate):
            pass
    except OSError as exc:
        raise ValueError("Unsloth does not have permission to write to this folder.") from exc
    return candidate


def set_projects_home(projects_home: str | None) -> dict:
    if not is_owner_context():
        raise PermissionError("Only the installation owner can change the projects folder.")
    if (os.environ.get(PROJECTS_HOME_ENV) or "").strip():
        raise RuntimeError("The projects folder is managed by UNSLOTH_STUDIO_PROJECTS_HOME.")
    with _settings_lock:
        next_home = _validate_projects_home(projects_home) if projects_home is not None else None
        # Never evict old locations: a project can still live there after many changes.
        history = owner_project_workspaces_roots()
        from storage.studio_db import upsert_app_settings

        upsert_app_settings(
            {
                PROJECTS_HOME_SETTING_KEY: str(next_home) if next_home is not None else None,
                PROJECTS_HOME_HISTORY_SETTING_KEY: [str(path) for path in history],
            }
        )
    return get_projects_home_status()
