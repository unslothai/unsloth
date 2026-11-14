# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Account-specific GGUF defaults for API requests that name a model without a quant."""

from typing import Optional

from utils.account_context import current_account
from utils.model_sampling_defaults import sampling_defaults_key
from utils.openai_auto_switch_settings import (
    _cached_setting,
    _fold_case_insensitive_path,
    _invalidate,
    _looks_like_filesystem_path,
)

MODEL_GGUF_PREFERENCES_KEY = "model_gguf_preferences"


def _preference_key(model_id: str) -> str:
    """Reuse per-model identity rules, folding repo/Windows keys while preserving POSIX case."""
    key = sampling_defaults_key(model_id)
    if _looks_like_filesystem_path(key):
        return _fold_case_insensitive_path(key) or key
    return key.casefold()


def get_model_gguf_preference(model_id: str) -> Optional[str]:
    """Return this account's saved quant for a model or its HF snapshot path, else None."""
    stored = _cached_setting(MODEL_GGUF_PREFERENCES_KEY, None, current_account())
    entry = stored.get(_preference_key(model_id)) if isinstance(stored, dict) else None
    variant = entry.get("gguf_variant") if isinstance(entry, dict) else None
    return variant if isinstance(variant, str) else None


def set_model_gguf_preference(model_id: str, variant: Optional[str]) -> Optional[str]:
    """Persist one validated quant, or remove it with None, without replacing other models.

    Storage and cache invalidation use the acting account. This changes no model
    residency and performs no download; ordinary API admission still owns both.
    """
    from hub.utils.paths import is_valid_gguf_variant
    from storage.studio_db import upsert_app_setting_map_entry

    if not model_id.strip():
        raise ValueError("model_id is required.")
    if variant is not None and not is_valid_gguf_variant(variant):
        raise ValueError("Invalid GGUF variant.")
    upsert_app_setting_map_entry(
        MODEL_GGUF_PREFERENCES_KEY,
        _preference_key(model_id),
        {"gguf_variant": variant} if variant is not None else None,
    )
    _invalidate(MODEL_GGUF_PREFERENCES_KEY, current_account())
    return variant


def with_model_gguf_preference(requested: str) -> str:
    """Qualify a bare API model with its saved quant; explicit suffixes and disabled switching win.

    Reuse the existing :variant path so load reuse, permission checks, missing
    variants and opt-in downloads obey the same rules as an explicit API choice.
    """
    from core.inference.openai_auto_download import split_model_ref
    from utils.openai_auto_switch_settings import get_openai_auto_switch_enabled

    _, variant = split_model_ref(requested)
    if variant is not None or not get_openai_auto_switch_enabled():
        return requested
    preferred = get_model_gguf_preference(requested)
    if preferred is None and _cached_setting(MODEL_GGUF_PREFERENCES_KEY, None, current_account()):
        from core.inference.local_model_resolver import resolve_local_gguf

        # API clients also use public aliases for local folders. Resolve those
        # through the existing account-scoped inventory, never by guessing a basename.
        # No inventory scan is added for accounts that have not saved a preference.
        resolved = resolve_local_gguf(requested)
        if resolved is not None:
            preferred = get_model_gguf_preference(resolved[2]) or get_model_gguf_preference(
                resolved[0]
            )
    return f"{requested.strip()}:{preferred}" if preferred else requested
