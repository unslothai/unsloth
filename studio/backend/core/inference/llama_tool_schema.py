# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama.cpp's grammar keeps a nested object's optional keys only in declared order (#10839)."""

from __future__ import annotations

from typing import Any

_PERMISSIVE_OBJECT = {"type": "object", "additionalProperties": True}
_ANNOTATIONS = ("description", "nullable", "title")
# True wraps, False never (llama.cpp merges $ref/allOf parts by their own keys), None inherits.
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


# A generated wrapper, so a caller's own union of the same shape is never mistaken for one.
class _RelaxedUnion(dict):
    original: Any = None


_OBJECT_ONLY = (
    "additionalProperties",
    "dependentRequired",
    "dependentSchemas",
    "maxProperties",
    "minProperties",
    "patternProperties",
    "properties",
    "propertyNames",
    "required",
    "unevaluatedProperties",
)


def _wrap(schema: dict) -> _RelaxedUnion:
    if "$ref" in schema or "anyOf" in schema or "oneOf" in schema:
        # templates read these outer fields; the grammar takes anyOf before them.
        fields = {
            key: schema[key]
            for key in (*_ANNOTATIONS, "type", "properties", "required")
            if key in schema
        }
        wrapped = _RelaxedUnion({**fields, "anyOf": [dict(_PERMISSIVE_OBJECT), schema]})
    else:
        # Chat templates read the node's own type/properties/required; llama.cpp's grammar takes
        # anyOf before them, so both stay. A type list keeps each other type with its own constraints,
        # as llama.cpp expands it.
        kind = schema.get("type")
        scalar = {
            key: value
            for key, value in schema.items()
            if key not in _OBJECT_ONLY and key not in _ANNOTATIONS
        }
        others = (
            [{**scalar, "type": t} for t in kind if t != "object"] if isinstance(kind, list) else []
        )
        wrapped = _RelaxedUnion({**schema, "anyOf": [dict(_PERMISSIVE_OBJECT), *others]})
    wrapped.original = schema
    return wrapped


def _optional_key_count(schema: Any) -> int:
    if not isinstance(schema, dict):
        return 0
    kind = schema.get("type")
    if isinstance(kind, list):
        is_object = "object" in kind
    else:
        is_object = kind is None or kind == "object"
    properties = schema.get("properties")
    if not is_object or not isinstance(properties, dict):
        return 0
    required = schema.get("required")
    required = set(required) if isinstance(required, list) else set()
    return sum(name not in required for name in properties)


def _resolve_ref(ref: Any, root: dict) -> Any:
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None
    node: Any = root
    for part in ref[2:].split("/"):
        if not isinstance(node, dict):
            return None
        node = node.get(part.replace("~1", "/").replace("~0", "~"))
    return node


def _follow_refs(schema: Any, root: dict) -> Any:
    seen = set()
    while isinstance(schema, dict) and isinstance(schema.get("$ref"), str):
        ref = schema["$ref"]
        target = _resolve_ref(ref, root)
        if ref in seen or not isinstance(target, dict):
            break
        seen.add(ref)
        schema = target
    return schema


def _reorderable(
    schema: dict,
    root: dict,
    seen: frozenset = frozenset(),
) -> bool:
    target = _follow_refs(schema, root)
    if _optional_key_count(target) >= 2:
        return True
    # A union reached only through a $ref is never walked as nested, so its branches decide.
    if target is not schema and id(target) not in seen:
        for keyword in ("anyOf", "oneOf"):
            branches = target.get(keyword)
            if isinstance(branches, list) and any(
                isinstance(branch, dict) and _reorderable(branch, root, seen | {id(target)})
                for branch in branches
            ):
                return True
    parts = target.get("allOf")
    if not isinstance(parts, list):
        return False
    return sum(_optional_key_count(_follow_refs(part, root)) for part in parts) >= 2


def _relax(schema: Any, *, nested: bool, root: dict) -> Any:
    if not isinstance(schema, dict) or isinstance(schema, _RelaxedUnion):
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
    if nested and _reorderable(out, root):
        return _wrap(out)
    return out


def relax_nested_object_key_order(parameters: Any) -> Any:
    root = parameters if isinstance(parameters, dict) else {}
    return _relax(parameters, nested = False, root = root)


def unrelaxed(schema: Any) -> Any:
    return schema.original if isinstance(schema, _RelaxedUnion) else schema


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
