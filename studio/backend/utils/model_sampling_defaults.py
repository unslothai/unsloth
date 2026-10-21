# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-model sampling defaults a user saved; a request that leaves a field unset gets them before the
model's recommendation. Not stored in the launch overrides: a row there counts as a load override
and shadows a broader one."""

from __future__ import annotations

import threading
from typing import Any, Mapping, Optional

from utils.account_context import OWNER, current_account, is_owner_context
from utils.openai_auto_switch_settings import _cached_setting, _invalidate

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
            # Repo ids are case-insensitive and the browser lowercases some; an ambiguous fold matches nothing.
            fold = key.casefold()
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
        found = folded.get(candidate.casefold())
        if found is not None:
            return found
    return None


def saved_sampling_defaults(model_id: Optional[str]) -> dict[str, Any]:
    """The saved defaults for ``model_id``, or ``{}``. A cached repo loads from its snapshot path, so
    the repo id it came from is tried too. A managed account without its own row reads the owner's."""
    if not model_id or not isinstance(model_id, str):
        return {}
    accounts = [current_account()]
    if not is_owner_context():
        accounts.append(OWNER)
    indexes = [_index(account) for account in accounts]
    if not any(exact for exact, _folded in indexes):
        return {}
    from core.inference.model_ids import hf_cache_repo_id

    candidates = [model_id]
    repo_id = hf_cache_repo_id(model_id)
    if repo_id and repo_id not in candidates:
        candidates.append(repo_id)
    for exact, folded in indexes:
        found = _lookup(exact, folded, candidates)
        if found is not None:
            return dict(found)
    return {}


def set_model_sampling_defaults(
    model_id: str, sampling: Optional[Mapping[str, Any]]
) -> dict[str, Any]:
    """Replace one model's saved defaults; nothing usable removes the row. Returns what was stored."""
    if not model_id or not model_id.strip():
        raise ValueError("model_id is required.")
    entry = normalize_sampling_defaults(sampling)
    from storage.studio_db import upsert_app_setting_map_entry

    upsert_app_setting_map_entry(
        MODEL_SAMPLING_DEFAULTS_SETTING_KEY, model_id.strip(), entry or None
    )
    _invalidate(MODEL_SAMPLING_DEFAULTS_SETTING_KEY, current_account())
    return entry
