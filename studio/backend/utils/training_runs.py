# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Helpers for naming and describing Studio training runs."""

from __future__ import annotations

import re
import time
from typing import Any, Optional

_INVALID_SEGMENT_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


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
    run_parts = [model_name.replace("/", "_")]
    project_slug = slugify_project_name(project_name)
    if project_slug:
        run_parts.append(project_slug)
    run_parts.append(str(int(time.time() if timestamp is None else timestamp)))
    return "_".join(run_parts)


def extract_project_name(config: Any) -> Optional[str]:
    """Read and normalize a project name from a stored config dict."""
    if not isinstance(config, dict):
        return None
    return normalize_project_name(config.get("project_name"))
