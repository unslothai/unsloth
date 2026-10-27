# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-model sampling defaults a user saved; a request that leaves a field unset gets them before the
model's recommendation. Not stored in the launch overrides: a row there counts as a load override
and shadows a broader one."""

from __future__ import annotations

import threading
from typing import Any, Mapping, Optional

from utils.account_context import current_account
from utils.openai_auto_switch_settings import (
    _cached_setting,
    _fold_case_insensitive_path,
    _invalidate,
    _looks_like_filesystem_path,
    split_quant_suffix,
)

MODEL_SAMPLING_DEFAULTS_SETTING_KEY = "model_sampling_defaults"


def normalize_sampling_defaults(payload: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """The usable fields of ``payload``, each validated the way a request value is; anything else is dropped."""
    from utils.inference.inference_config import SAMPLING_FIELD_NAMES, _clean_sampling_value

    if not isinstance(payload, Mapping):
        return {}
    cleaned: dict[str, Any] = {}
    for field in SAMPLING_FIELD_NAMES:
        value = _clean_sampling_value(field, payload.get(field))
        if value is not None:
            cleaned[field] = value
    return cleaned


def _stored(account) -> Any:
    return _cached_setting(MODEL_SAMPLING_DEFAULTS_SETTING_KEY, None, account)


def get_model_sampling_defaults(account = None) -> dict[str, dict]:
    raw = _stored(account or current_account())
    if not isinstance(raw, dict):
        return {}
    return {
        key: value for key, value in raw.items() if isinstance(key, str) and isinstance(value, dict)
    }


# Rebuilt only when the cached setting is re-read, so a request does dict lookups, not a scan.
_index_lock = threading.Lock()
_index_cache: dict[str, tuple[Any, dict[str, dict], dict[str, Optional[dict]]]] = {}


def _fold(key: str) -> Optional[str]:
    """The launch overrides' folding rule: repo ids and Windows / UNC / WSL paths fold, a POSIX path does not."""
    if _looks_like_filesystem_path(key):
        return _fold_case_insensitive_path(key)
    return key.casefold()


def _index(account) -> tuple[dict[str, dict], dict[str, Optional[dict]]]:
    raw = _stored(account)
    with _index_lock:
        hit = _index_cache.get(account.account_id)
        if hit is not None and hit[0] is raw:
            return hit[1], hit[2]
    exact: dict[str, dict] = {}
    folded: dict[str, Optional[dict]] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            entry = normalize_sampling_defaults(value)
            exact[key] = entry
            # An ambiguous fold matches nothing.
            fold = _fold(key)
            if fold is not None:
                folded[fold] = None if fold in folded else entry
    with _index_lock:
        _index_cache[account.account_id] = (raw, exact, folded)
    return exact, folded


def _lookup(
    exact: dict[str, dict], folded: dict[str, Optional[dict]], candidates: list[str]
) -> Optional[dict]:
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
    for candidate in candidates:
        fold = _fold(candidate)
        found = folded.get(fold) if fold is not None else None
        if found is not None:
            return found
    return None


def sampling_defaults_key(model_id: str) -> str:
    """The key one model's defaults are stored under: per model, not per quant, and a cached repo by
    its repo id rather than a snapshot path that a new revision would orphan."""
    from core.inference.model_ids import hf_cache_repo_id

    key = model_id.strip()
    split = split_quant_suffix(key)
    if split is not None:
        key = split[0]
    return hf_cache_repo_id(key) or key


def saved_sampling_defaults(model_id: Optional[str]) -> dict[str, Any]:
    """The saved defaults for ``model_id``, or ``{}``."""
    if not model_id or not isinstance(model_id, str):
        return {}
    exact, folded = _index(current_account())
    if not exact:
        return {}
    candidates = [model_id]
    key = sampling_defaults_key(model_id)
    if key not in candidates:
        candidates.append(key)
    found = _lookup(exact, folded, candidates)
    return dict(found) if found is not None else {}


def set_model_sampling_defaults(
    model_id: str, sampling: Optional[Mapping[str, Any]]
) -> dict[str, Any]:
    """Replace one model's saved defaults; nothing usable removes the row. Returns what was stored."""
    if not model_id or not model_id.strip():
        raise ValueError("model_id is required.")
    entry = normalize_sampling_defaults(sampling)
    from storage.studio_db import upsert_app_setting_map_entry

    key = sampling_defaults_key(model_id)
    upsert_app_setting_map_entry(MODEL_SAMPLING_DEFAULTS_SETTING_KEY, key, entry or None)
    # A row an earlier save left under the literal spelling would otherwise outrank this one.
    if model_id.strip() != key:
        upsert_app_setting_map_entry(MODEL_SAMPLING_DEFAULTS_SETTING_KEY, model_id.strip(), None)
    _invalidate(MODEL_SAMPLING_DEFAULTS_SETTING_KEY, current_account())
    return entry
