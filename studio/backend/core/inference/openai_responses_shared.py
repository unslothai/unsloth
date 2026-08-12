# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared OpenAI Responses request and SSE normalization helpers."""

from __future__ import annotations

from typing import Any


def _normalize_schema_node(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema

    normalized = dict(schema)
    for key in (
        "additionalProperties",

        "additionalItems",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    ):
        if key in normalized:
            value = normalized[key]
            if key == "items" and isinstance(value, list):
                normalized[key] = [_normalize_schema_node(item) for item in value]
            else:
                normalized[key] = _normalize_schema_node(value)

    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        value = normalized.get(key)
        if isinstance(value, list):
            normalized[key] = [_normalize_schema_node(item) for item in value]

    for key in (
        "$defs",
        "definitions",
        "dependencies",
        "dependentSchemas",
        "patternProperties",
        "properties",
    ):
        value = normalized.get(key)
        if isinstance(value, dict):
            normalized[key] = {
                name: _normalize_schema_node(subschema) for name, subschema in value.items()
            }

    schema_type = normalized.get("type")
    if schema_type == "object" or (
        isinstance(schema_type, list) and "object" in schema_type
    ):
        properties = normalized.get("properties")
        normalized["properties"] = properties if isinstance(properties, dict) else {}
    return normalized


def normalize_function_schema(schema: Any) -> dict[str, Any]:
    """Ensure every object schema has properties, including nested combinators."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    return _normalize_schema_node(schema)


def responses_function_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def responses_function_output(call_id: str, output: str) -> dict[str, Any]:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    }


def response_event_type(event: Any, event_name: str = "") -> str:
    """Return a validated Responses event type from JSON or the SSE event field."""
    if not isinstance(event, dict):
        raise ValueError("Responses event must be an object")
    value = event.get("type") or event_name
    if not isinstance(value, str) or not value:
        raise ValueError("Responses event type is missing")
    return value


def responses_usage_to_chat(usage: Any) -> dict[str, Any] | None:
    """Translate Responses token usage into Chat Completions usage names."""
    if not isinstance(usage, dict):
        return None
    prompt_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    result: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(usage.get("total_tokens") or prompt_tokens + completion_tokens),
    }
    input_details = usage.get("input_tokens_details")
    if isinstance(input_details, dict):
        result["prompt_tokens_details"] = dict(input_details)
    output_details = usage.get("output_tokens_details")
    if isinstance(output_details, dict):
        result["completion_tokens_details"] = dict(output_details)
    return result
