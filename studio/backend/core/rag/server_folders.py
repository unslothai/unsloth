# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Source folders explicitly exposed by the server operator, never by an HTTP caller."""

import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
ENV_NAME = "UNSLOTH_RAG_SOURCE_FOLDERS"


def configured_folders() -> dict[str, str]:
    raw = os.environ.get(ENV_NAME, "[]")
    try:
        paths = json.loads(raw)
        if not isinstance(paths, list) or any(
            not isinstance(path, str) or not Path(path).is_absolute() for path in paths
        ):
            raise ValueError("expected a JSON array of absolute folder paths")
    except (ValueError, TypeError):
        logger.warning("%s must be a JSON array of absolute folder paths", ENV_NAME)
        return {}
    # Keep the configured path out of browser requests and responses.
    return {hashlib.sha256(path.encode("utf-8")).hexdigest(): path for path in paths}


def resolve_folder(folder_id: str) -> tuple[str, tuple[int, int]]:
    from core.rag import folder_sync

    path = configured_folders().get(folder_id)
    if path is None:
        raise ValueError("Server source folder is not configured")
    normalized = folder_sync.validate_folder_path(path)
    return normalized, folder_sync._root_identity(normalized)


def available_folders() -> list[dict[str, str]]:
    folders = []
    for folder_id in configured_folders():
        try:
            path, _ = resolve_folder(folder_id)
        except (ValueError, RuntimeError):
            continue
        folders.append({"id": folder_id, "displayName": Path(path).name})
    return folders
