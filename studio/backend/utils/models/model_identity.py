# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep pinned Hugging Face cache paths out of saved model metadata."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Optional

from core.inference.model_ids import hf_cache_repo_id


def _identifier(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        identifier = os.fspath(value)
    except TypeError:
        return None
    identifier = str(identifier).strip()
    return identifier or None


def _cache_path_matches_repo(value: Any, repo_id: str) -> bool:
    cached_repo_id = _snapshot_repo_id(value)
    return bool(cached_repo_id and cached_repo_id.casefold() == repo_id.casefold())


def _snapshot_repo_id(value: Any) -> Optional[str]:
    identifier = _identifier(value)
    if not identifier:
        return None
    parts = identifier.replace("\\", "/").split("/")
    has_revision = any(
        part.startswith("models--")
        and parts[index + 1 : index + 2] == ["snapshots"]
        and bool(parts[index + 2 : index + 3])
        and bool(parts[index + 2])
        for index, part in enumerate(parts)
    )
    return hf_cache_repo_id(identifier) if has_revision else None


def _set_standard_identity(config: Any, attribute: str, repo_id: str) -> bool:
    if config is None or not _cache_path_matches_repo(getattr(config, attribute, None), repo_id):
        return False
    try:
        setattr(config, attribute, repo_id)
    except (AttributeError, TypeError):
        return False
    return True


def restore_hf_cache_repo_identity(
    model: Any,
    load_target: Any,
    *,
    expected_repo_id: Optional[str] = None,
) -> Optional[str]:
    """Restore standard Hub metadata after loading an exact cached snapshot.

    Only complete Hugging Face snapshot paths are handled. The loaded weights
    stay pinned, while ordinary local models, files, and custom fields remain
    untouched. Returns the repository id when a standard field changed.
    """
    target_repo_id = _snapshot_repo_id(load_target)
    if not target_repo_id:
        return None

    repo_id = target_repo_id
    if expected_repo_id is not None:
        expected = _identifier(expected_repo_id)
        if not expected or expected.casefold() != target_repo_id.casefold():
            return None
        repo_id = expected

    changed = _set_standard_identity(
        getattr(model, "config", None),
        "_name_or_path",
        repo_id,
    )
    # PreTrainedModel.__init__ copies config.name_or_path onto the instance, so
    # updating the config alone leaves this stale. PEFT reads exactly this slot
    # (mapping_func.py: model.__dict__.get("name_or_path")) and overwrites
    # base_model_name_or_path with it, which is how a pinned snapshot path ends up
    # in adapter_config.json, the checkpoints, the run card and every export.
    changed = _set_standard_identity(model, "name_or_path", repo_id) or changed
    changed = _set_standard_identity(model, "_hf_repo", repo_id) or changed

    peft_config = getattr(model, "peft_config", None)
    if isinstance(peft_config, Mapping):
        for adapter_config in peft_config.values():
            changed = (
                _set_standard_identity(
                    adapter_config,
                    "base_model_name_or_path",
                    repo_id,
                )
                or changed
            )

    return repo_id if changed else None
