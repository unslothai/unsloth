# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import re
from typing import Optional

_VALID_REPO_ID_SEGMENT = re.compile(r"^[A-Za-z0-9_](?:[A-Za-z0-9._-]*[A-Za-z0-9_])?$")
_MAX_REPO_ID_LENGTH = 96
_MODEL_CACHE_PREFIX = "models--"


def is_valid_repo_id(repo_id: str) -> bool:
    """Validate Hugging Face ``repo_name`` or ``namespace/repo_name`` IDs."""
    if not repo_id or repo_id != repo_id.strip():
        return False
    if repo_id.endswith(".git"):
        return False
    if "--" in repo_id or ".." in repo_id:
        return False
    segments = repo_id.split("/")
    if len(segments) not in (1, 2):
        return False
    # Match huggingface_hub.validate_repo_id: the 96-char limit applies per segment (repo name /
    # namespace), not to the whole "namespace/repo_name" string.
    return all(
        segment not in ("", ".", "..")
        and len(segment) <= _MAX_REPO_ID_LENGTH
        and _VALID_REPO_ID_SEGMENT.fullmatch(segment) is not None
        for segment in segments
    )


def hf_cache_repo_id(path: Optional[str]) -> Optional[str]:
    """``.../models--org--name/snapshots/<sha>`` -> ``org/name``, else None.

    A model loaded from the HF cache is identified by its snapshot dir, whose
    basename is a commit hash; recover the repo id so callers don't show that.
    """
    if not path:
        return None
    parts = str(path).replace("\\", "/").split("/")
    for index, part in enumerate(parts):
        # Only inside the real cache layout: a "models--" name alone is not a repo id.
        if (
            part[: len(_MODEL_CACHE_PREFIX)].lower() == _MODEL_CACHE_PREFIX
            and index + 1 < len(parts)
            and parts[index + 1].lower() == "snapshots"
        ):
            return part[len(_MODEL_CACHE_PREFIX) :].replace("--", "/")
    return None
