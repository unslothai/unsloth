# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tool schemas as llama-server should receive them: llama.cpp's grammar admits a nested
object's optional keys only in declared order, so a key written out of order is silently
dropped (#10839). A permissive union branch lifts that without hiding the original."""

from __future__ import annotations

from typing import Any

_PERMISSIVE_OBJECT = {"type": "object", "additionalProperties": True}
_MAP_KEYWORDS = ("$defs", "definitions", "dependentSchemas", "patternProperties", "properties")
_SINGLE_KEYWORDS = (
    "additionalProperties",
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
)
_LIST_KEYWORDS = ("allOf", "anyOf", "items", "oneOf", "prefixItems")


def _has_reorderable_keys(schema: dict) -> bool:
    kind = schema.get("type")
    if isinstance(kind, list):
        is_object = "object" in kind
    else:
        is_object = kind is None or kind == "object"
    properties = schema.get("properties")
    if not is_object or not isinstance(properties, dict):
        return False
    required = schema.get("required")
    required = set(required) if isinstance(required, list) else set()
    return sum(name not in required for name in properties) >= 2


def _is_relaxed(schema: dict) -> bool:
    branches = schema.get("anyOf")
    return (
        len(schema) == 1
        and isinstance(branches, list)
        and len(branches) == 2
        and branches[0] == _PERMISSIVE_OBJECT
    )


def _relax(schema: Any, *, nested: bool) -> Any:
    if not isinstance(schema, dict) or _is_relaxed(schema):
        return schema
    out = schema
    for keyword in _MAP_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, dict):
            relaxed = {key: _relax(value, nested = True) for key, value in children.items()}
            if any(relaxed[key] is not children[key] for key in children):
                out = {**out, keyword: relaxed}
    for keyword in _SINGLE_KEYWORDS:
        child = schema.get(keyword)
        if isinstance(child, dict):
            relaxed = _relax(child, nested = True)
            if relaxed is not child:
                out = {**out, keyword: relaxed}
    for keyword in _LIST_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, list):
            relaxed = [_relax(value, nested = True) for value in children]
            if any(new is not old for new, old in zip(relaxed, children)):
                out = {**out, keyword: relaxed}
    if nested and _has_reorderable_keys(out):
        return {"anyOf": [dict(_PERMISSIVE_OBJECT), out]}
    return out


def relax_nested_object_key_order(parameters: Any) -> Any:
    return _relax(parameters, nested = False)


def llama_grammar_tools(tools: Any) -> Any:
    if not isinstance(tools, list):
        return tools
    out = []
    changed = False
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        parameters = function.get("parameters") if isinstance(function, dict) else None
        relaxed = relax_nested_object_key_order(parameters)
        if relaxed is parameters:
            out.append(tool)
            continue
        out.append({**tool, "function": {**function, "parameters": relaxed}})
        changed = True
    return out if changed else tools
