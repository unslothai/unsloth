# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hub service layer."""

from __future__ import annotations

from typing import Iterable

from fastapi import HTTPException

from hub.utils.hf_cache_state import resolve_destructive_case_matches


def resolve_destructive_repo_ids(
    repo_id: str,
    candidates: Iterable[str],
    *,
    noun: str,
) -> set[str]:
    """Cache-dir repo ids a destructive op on *repo_id* may target.

    Refuses with 409 when only case-insensitive matches exist and they are
    ambiguous (multiple cached repos differing only by case), so a delete can
    never remove the wrong casing. *noun* is the plural shown to the user
    (e.g. "models", "datasets")."""
    resolved = resolve_destructive_case_matches(repo_id, candidates)
    if resolved is None:
        raise HTTPException(
            status_code = 409,
            detail = (
                f"Multiple cached {noun} differ only by case. "
                "Delete the exact repo casing from On Device."
            ),
        )
    return resolved
