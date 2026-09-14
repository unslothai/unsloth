# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tool schemas as llama-server should receive them: llama.cpp's grammar admits a nested
object's optional keys only in declared order, so a key written out of order is silently
dropped (#10839). A permissive union branch lifts that without hiding the original."""

from __future__ import annotations

from typing import Any

_PERMISSIVE_OBJECT = {"type": "object", "additionalProperties": True}
# Child mode per keyword: True wraps, False never wraps, None inherits the parent's instance.
# Definitions and allOf parts stay bare because llama.cpp merges a $ref or allOf part by its
# own properties, and a wrapped part contributes none.
_MAP_KEYWORDS = {
    "$defs": False,
    "definitions": False,
    "dependentSchemas": True,
    "patternProperties": True,
    "properties": True,
}
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
_LIST_KEYWORDS = {"allOf": False, "anyOf": None, "items": True, "oneOf": None, "prefixItems": True}


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


def _resolve_ref(ref: Any, root: dict) -> Any:
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None
    node: Any = root
    for part in ref[2:].split("/"):
        if not isinstance(node, dict):
            return None
        node = node.get(part.replace("~1", "/").replace("~0", "~"))
    return node


def _relax(schema: Any, *, nested: bool, root: dict) -> Any:
    if not isinstance(schema, dict) or _is_relaxed(schema):
        return schema
    out = schema
    for keyword, child_nested in _MAP_KEYWORDS.items():
        children = schema.get(keyword)
        if isinstance(children, dict):
            relaxed = {
                key: _relax(value, nested = child_nested, root = root)
                for key, value in children.items()
            }
            if any(relaxed[key] is not children[key] for key in children):
                out = {**out, keyword: relaxed}
    for keyword in _SINGLE_KEYWORDS:
        child = schema.get(keyword)
        if isinstance(child, dict):
            relaxed = _relax(child, nested = True, root = root)
            if relaxed is not child:
                out = {**out, keyword: relaxed}
    for keyword, child_nested in _LIST_KEYWORDS.items():
        children = schema.get(keyword)
        if isinstance(children, list):
            mode = nested if child_nested is None else child_nested
            relaxed = [_relax(value, nested = mode, root = root) for value in children]
            if any(new is not old for new, old in zip(relaxed, children)):
                out = {**out, keyword: relaxed}
    if not nested:
        return out
    target = _resolve_ref(out.get("$ref"), root)
    if _has_reorderable_keys(out) or (isinstance(target, dict) and _has_reorderable_keys(target)):
        return {"anyOf": [dict(_PERMISSIVE_OBJECT), out]}
    return out


def relax_nested_object_key_order(parameters: Any) -> Any:
    root = parameters if isinstance(parameters, dict) else {}
    return _relax(parameters, nested = False, root = root)


def unrelaxed(schema: Any) -> Any:
    return schema["anyOf"][1] if isinstance(schema, dict) and _is_relaxed(schema) else schema


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
