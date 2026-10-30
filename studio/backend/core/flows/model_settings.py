# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LLM nodes with "Use the model's saved settings" on read Chat's per-model memory at run time: the
`inferenceParamsByModel` chat setting for sampling, and the per-model launch override (what Chat's run settings
remember, and what an API load applies) for load settings. No store of their own: Save and Load in the editor
write and read those same rows. Keys a connected input sets are left alone.

A card's own load settings are kept in that override's shape (`load`), so one mapping turns either into the
load request: the one auto-switch loads use."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

USE_SAVED = "use_model_settings"

# Chat's remembered key -> the node's param.
SAVED_TO_PARAM = {
    "systemPrompt": "system",
    "temperature": "temperature",
    "topP": "top_p",
    "topK": "top_k",
    "minP": "min_p",
    "repetitionPenalty": "repetition_penalty",
    "presencePenalty": "presence_penalty",
    "maxTokens": "max_tokens",
}

# Sent as the request field of the same name when set; unset leaves the server's choice.
REQUEST_PARAMS = (
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "reasoning_effort",
)

# Chat records a "Max" output cap as this, meaning the load's whole window.
_MAX_TOKENS_AS_MAX = 2**31 - 1


def find_saved(by_model: Any, model_id: Optional[str]) -> Optional[dict]:
    if not model_id or not isinstance(by_model, dict):
        return None
    entry = by_model.get(model_id)
    if entry is None:
        folded = model_id.casefold()
        entry = next(
            (v for k, v in by_model.items() if isinstance(k, str) and k.casefold() == folded), None
        )
    return entry if isinstance(entry, dict) else None


def overlay(
    params: dict,
    saved: Optional[dict],
    spec: dict,
    skip: set[str] = frozenset(),
) -> dict:
    """``params`` with the saved values the node has a param for. Unsaved keys keep the card's own."""
    if not saved:
        return params
    by_id = {p["id"]: p for p in spec["params"]}
    out = dict(params)
    for key, pid in SAVED_TO_PARAM.items():
        value = saved.get(key)
        if pid not in by_id or pid in skip:
            continue
        if key == "minP" and saved.get("minPMode") == "server-default":
            out[pid] = None
            continue
        if value is None:
            continue
        if key == "maxTokens" and value >= _MAX_TOKENS_AS_MAX:
            continue
        if key == "systemPrompt":
            if isinstance(value, str):
                out[pid] = value
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        p = by_id[pid]
        low, high = p.get("min"), p.get("max")
        if low is not None:
            value = max(low, value)
        if high is not None:
            value = min(high, value)
        out[pid] = int(value) if p["kind"] == "int" else value
    return out


def read_saved(model_id: Optional[str]) -> Optional[dict]:
    try:
        from storage.studio_db import list_chat_settings
        return find_saved(list_chat_settings().get("inferenceParamsByModel"), model_id)
    except Exception:
        logger.debug("flows: could not read saved model settings", exc_info = True)
        return None


def request_params(params: dict) -> dict:
    return {k: params[k] for k in REQUEST_PARAMS if params.get(k) not in (None, "")}


LOAD = "load"


def saved_load(model_id: Optional[str], variant: Optional[str] = None) -> dict:
    if not model_id:
        return {}
    try:
        from utils.openai_auto_switch_settings import resolve_override_for_load

        return dict(resolve_override_for_load(model_id, None, variant)[1] or {})
    except Exception:
        logger.debug("flows: could not read saved load settings", exc_info = True)
        return {}


def looks_gguf(repo: Optional[str], variant: Optional[str]) -> bool:
    return bool(variant) or "gguf" in (repo or "").lower()


def load_kwargs(load: Any, *, is_gguf: bool) -> Optional[dict]:
    """The load options a card's settings ask for, or None for "however it loads by default"."""
    if not isinstance(load, dict) or not load:
        return None
    from utils.openai_auto_switch_settings import model_override_load_kwargs, normalize_model_override

    return model_override_load_kwargs(normalize_model_override(load), is_gguf = is_gguf) or None


def card_load_kwargs(
    params: dict, resident: Optional[str] = None, resident_variant: Optional[str] = None
) -> Optional[dict]:
    """What the scheduler groups by, before the run resolves saved settings: theirs when the card follows them.
    A card with Model unset reads as the resident model and its quant, so GGUF-only and quant-specific settings
    match what the run resolves."""
    choice = params.get("model")
    picked = choice.get("repo") if isinstance(choice, dict) else None
    repo = picked or resident
    variant = (choice.get("variant") if isinstance(choice, dict) else None) if picked else resident_variant
    load = params.get(LOAD) or {}
    if params.get(USE_SAVED) and repo:
        load = {**load, **saved_load(repo, variant)}
    try:
        return load_kwargs(load, is_gguf = looks_gguf(repo, variant))
    except Exception:
        return None
