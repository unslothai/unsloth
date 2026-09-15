# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from typing import Any

REASONING_EFFORT_SCALE = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

MODEL_CAPABILITY_PROVIDERS = frozenset({"openrouter"})


def _sorted_efforts(values: Any) -> list[str] | None:
    if not isinstance(values, list):
        return None
    known = {v for v in values if isinstance(v, str) and v in REASONING_EFFORT_SCALE}
    return [level for level in REASONING_EFFORT_SCALE if level in known]


def _string_list(values: Any) -> list[str] | None:
    if not isinstance(values, list):
        return None
    items = [v.strip().lower() for v in values if isinstance(v, str) and v.strip()]
    return items or None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if value > 0 else None


def openrouter_model_capabilities(raw: dict[str, Any]) -> dict[str, Any] | None:
    model_id = raw.get("id")
    if not isinstance(model_id, str) or not model_id.strip():
        return None
    architecture = raw.get("architecture")
    architecture = architecture if isinstance(architecture, dict) else {}
    top_provider = raw.get("top_provider")
    top_provider = top_provider if isinstance(top_provider, dict) else {}

    reasoning_raw = raw.get("reasoning")
    reasoning: dict[str, Any] | None = None
    if isinstance(reasoning_raw, dict):
        default_effort = reasoning_raw.get("default_effort")
        reasoning = {
            "supported_efforts": (
                list(REASONING_EFFORT_SCALE)
                if "supported_efforts" in reasoning_raw
                and reasoning_raw["supported_efforts"] is None
                else _sorted_efforts(reasoning_raw.get("supported_efforts"))
            ),
            "mandatory": reasoning_raw.get("mandatory") is True,
            "default_effort": default_effort if default_effort in REASONING_EFFORT_SCALE else None,
            "default_enabled": (
                reasoning_raw.get("default_enabled")
                if isinstance(reasoning_raw.get("default_enabled"), bool)
                else None
            ),
        }

    return {
        "id": model_id.strip(),
        "input_modalities": _string_list(architecture.get("input_modalities")),
        "reasoning": reasoning,
        "max_output_tokens": _positive_int(top_provider.get("max_completion_tokens")),
        "supported_parameters": _string_list(raw.get("supported_parameters")),
    }


def provider_model_capabilities(
    provider_type: str, models: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if provider_type != "openrouter":
        return []
    mapped = (openrouter_model_capabilities(m) for m in models if isinstance(m, dict))
    return [entry for entry in mapped if entry is not None]
