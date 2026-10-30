# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LLM nodes with "Use the model's saved settings" on read Chat's per-model memory (the
`inferenceParamsByModel` chat setting) at run time. No store of their own: Save and Load in the editor
write and read that same row. Keys a connected input sets are left alone."""

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

SAMPLING_PARAMS = ("top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty")

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


def sampling(params: dict) -> dict:
    """The optional sampling knobs a node sets; unset ones are left to the server."""
    return {k: params[k] for k in SAMPLING_PARAMS if params.get(k) is not None}
