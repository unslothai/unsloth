# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Helpers for naming and describing Unsloth training runs."""

from __future__ import annotations

import re
import time
from typing import Any, Optional

_INVALID_SEGMENT_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_RUN_DIR_NAME_CHARS = 255
_PROJECT_MARKER = "__project-"
_PROJECT_MARKER_ESCAPE = f"{_PROJECT_MARKER}-"
_UNSLOTH_ORG_PREFIX = "unsloth_"

# We emit a bare epoch; hand-made folders often use a date-time. Anything else
# (``_final``, ``_v2``, ``_8b``) is part of the model name, not a stamp.
_RUN_DIR_TIMESTAMP = re.compile(r"\A\d{6,}(?:[-_]\d{2,})?\Z")

# ``validate_repo_id`` transcribed to keep this module stdlib-only. A folder name is user
# input, so trust the parse only when the Hub would accept what falls out of it.
_REPO_NAME = re.compile(r"\A(?!.*(?:--|\.\.))(?![-.])[\w.-]{1,96}(?<![-.])\Z")


def _is_valid_repo_name(name: str) -> bool:
    return bool(_REPO_NAME.match(name)) and not name.endswith(".git")


def _trim_segment(segment: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    return segment[:max_chars].strip("._-")


def _escape_project_marker(segment: str) -> str:
    return segment.replace(_PROJECT_MARKER, _PROJECT_MARKER_ESCAPE)


def _unescape_project_marker(segment: str) -> str:
    return segment.replace(_PROJECT_MARKER_ESCAPE, _PROJECT_MARKER)


def _appended_project_marker_index(segment: str) -> int:
    marker_index = segment.rfind(_PROJECT_MARKER)
    while marker_index >= 0 and segment.startswith(_PROJECT_MARKER_ESCAPE, marker_index):
        marker_index = segment.rfind(_PROJECT_MARKER, 0, marker_index)
    return marker_index


def normalize_project_name(project_name: Any) -> Optional[str]:
    """Return a trimmed project name, or None when empty/invalid."""
    if not isinstance(project_name, str):
        return None
    normalized = " ".join(project_name.strip().split())
    return normalized or None


def slugify_project_name(project_name: Any) -> Optional[str]:
    """Convert a project name into a filesystem-safe suffix."""
    normalized = normalize_project_name(project_name)
    if normalized is None:
        return None

    slug = _INVALID_SEGMENT_CHARS.sub("-", normalized).strip("-._")
    if not slug:
        return None
    return slug.lower()


def build_default_output_dir_name(
    model_name: str,
    project_name: Any = None,
    *,
    timestamp: Optional[int] = None,
) -> str:
    """Build the default training output folder name."""
    from utils.paths import default_run_dir_name

    timestamp_part = str(int(time.time() if timestamp is None else timestamp))
    timestamp_suffix = f"_{timestamp_part}"
    model_segment = _escape_project_marker(default_run_dir_name(model_name))
    project_slug = slugify_project_name(project_name)
    if not project_slug:
        max_model_chars = _MAX_RUN_DIR_NAME_CHARS - len(timestamp_suffix)
        model_segment = _trim_segment(model_segment, max_model_chars) or "model"
        return f"{model_segment}{timestamp_suffix}"

    max_project_chars = (
        _MAX_RUN_DIR_NAME_CHARS - len("model") - len(_PROJECT_MARKER) - len(timestamp_suffix)
    )
    project_slug = _trim_segment(project_slug, max_project_chars) or "project"
    project_suffix = f"{_PROJECT_MARKER}{project_slug}{timestamp_suffix}"
    max_model_chars = _MAX_RUN_DIR_NAME_CHARS - len(project_suffix)
    model_segment = _trim_segment(model_segment, max_model_chars) or "model"
    return f"{model_segment}{project_suffix}"


def model_segment_from_default_output_dir_name(output_dir_name: str) -> Optional[str]:
    """Return the encoded model segment from a default run folder name."""
    parts = str(output_dir_name or "").rsplit("_", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    model_segment = parts[0]
    marker_index = _appended_project_marker_index(model_segment)
    if marker_index >= 0:
        model_segment = model_segment[:marker_index]
    model_segment = _unescape_project_marker(model_segment)
    return model_segment or None


def _model_segment_from_run_dir_name(output_dir_name: str) -> Optional[str]:
    """``model_segment_from_default_output_dir_name`` widened to date-time stamps.

    The strict inverse gates on ``isdigit()`` because it only reads folders we wrote; this
    one also reads folders we did not. Same shape otherwise, project suffix and escape included.
    """
    head, separator, last_segment = str(output_dir_name or "").rpartition("_")
    if not separator or not _RUN_DIR_TIMESTAMP.match(last_segment):
        return None
    marker_index = _appended_project_marker_index(head)
    if marker_index >= 0:
        head = head[:marker_index]
    return _unescape_project_marker(head) or None


def base_model_from_run_dir_name(dir_name: str) -> Optional[str]:
    """``unsloth_<model>_<timestamp>`` -> ``unsloth/<model>``, else None.

    The last resort when no config names a base model. It sits beside
    ``build_default_output_dir_name`` because it is that function read backwards; keeping the
    pair together is what stops the parse drifting from the names we write.

    None matters as much as a name here. Without a timestamp the folder is not one we wrote,
    and every caller already asks the user instead. Guessing reaches the Hub: ``unsloth/`` is
    rejected outright, and a truncated ``unsloth/llama_3`` for ``unsloth_llama_3_8b`` is worse,
    being a valid id that does not exist.
    """
    model_segment = _model_segment_from_run_dir_name(dir_name)
    if model_segment is None or not model_segment.startswith(_UNSLOTH_ORG_PREFIX):
        return None
    model_name = model_segment[len(_UNSLOTH_ORG_PREFIX) :]
    if not _is_valid_repo_name(model_name):
        return None
    return f"unsloth/{model_name}"


def extract_project_name(config: Any) -> Optional[str]:
    """Read and normalize a project name from a stored config dict."""
    if not isinstance(config, dict):
        return None
    return normalize_project_name(config.get("project_name"))
