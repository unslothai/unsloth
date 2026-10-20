# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Source folders explicitly exposed by the server operator, never by an HTTP caller."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
ENV_NAME = "UNSLOTH_RAG_SOURCE_FOLDERS"
# The picker lists these on every Sources panel mount, so a bad setting is reported once.
_warned: set[str] = set()


def _warn_once(message: str) -> None:
    if message not in _warned:
        _warned.add(message)
        logger.warning(message)


def configured_folders() -> dict[str, str]:
    raw = os.environ.get(ENV_NAME, "").strip()
    if not raw:
        return {}
    try:
        paths = json.loads(raw)
        if not isinstance(paths, list) or any(
            not isinstance(path, str) or not Path(path).is_absolute() for path in paths
        ):
            raise ValueError
    except ValueError:
        _warn_once(f"{ENV_NAME} must be a JSON array of absolute folder paths; ignoring it")
        return {}
    # Keep the configured path out of browser requests and responses.
    return {
        hashlib.sha256(path.encode("utf-8", "surrogatepass")).hexdigest(): path for path in paths
    }


def _resolve(path: str) -> tuple[str, tuple[int, int]]:
    from core.rag import folder_sync
    normalized = folder_sync.validate_folder_path(path)
    return normalized, folder_sync._root_identity(normalized)


def resolve_folder(folder_id: str) -> tuple[str, tuple[int, int]]:
    path = configured_folders().get(folder_id)
    if path is None:
        raise ValueError("Server source folder is not configured")
    return _resolve(path)


def _labels(paths: list[str]) -> list[str]:
    """Shortest trailing path that tells the folders apart: /a/docs and /b/docs read a/docs, b/docs.
    Never the anchor, which would put a whole absolute path in the browser."""
    parts = [Path(path).parts for path in paths]
    limit = [max(1, len(p) - 1) if Path(path).anchor else len(p) for p, path in zip(parts, paths)]
    depth = [1] * len(paths)
    while True:
        labels = [str(Path(*p[-d:])) for p, d in zip(parts, depth)]
        clashing = [
            i for i, label in enumerate(labels) if labels.count(label) > 1 and depth[i] < limit[i]
        ]
        if not clashing:
            # Still equal only below different drives or shares (C:\docs, D:\docs): name the drive.
            return [
                label if labels.count(label) == 1 else f"{label} ({Path(path).drive})"
                for label, path in zip(labels, paths)
            ]
        for i in clashing:
            depth[i] += 1


def available_folders() -> list[dict[str, str]]:
    resolved: dict[str, str] = {}
    for folder_id, path in configured_folders().items():
        try:
            normalized, _ = _resolve(path)
        except (ValueError, RuntimeError) as exc:
            _warn_once(f"{ENV_NAME}: skipping {path}: {exc}")
            continue
        if normalized not in resolved.values():
            resolved[folder_id] = normalized
    labels = _labels(list(resolved.values()))
    return [{"id": folder_id, "displayName": label} for folder_id, label in zip(resolved, labels)]


def display_name(folder_id: str) -> str | None:
    return next((f["displayName"] for f in available_folders() if f["id"] == folder_id), None)
