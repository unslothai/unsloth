# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

from .comparators import get_comparator, is_comparator


@dataclass
class LeafNode:
    comparator: str
    params: dict = field(default_factory=dict)


@dataclass
class ObjectNode:
    fields: dict  # str -> Node


@dataclass
class ArrayNode:
    item: "Node"


Node = Union[LeafNode, ObjectNode, ArrayNode]


# ── Standard JSON Schema → field-schema translation ─────────────────────────
# Users often paste a real JSON Schema (draft 2020-12 etc.) into the "schema"
# field. That isn't our field→comparator mapping, so we detect it and derive a
# comparator for each field from its JSON Schema type/format/enum.

_JSON_ONLY_TYPES = {"object", "array", "number", "integer", "boolean", "null"}
_DATE_FORMATS = {"date", "date-time", "datetime", "time"}


def _looks_like_json_schema(d: dict) -> bool:
    """Heuristic (checked only at the root): is this a standard JSON Schema
    rather than a json_score field→comparator mapping? Once true, the whole
    tree is translated as JSON Schema."""
    if "$schema" in d or "$id" in d:
        return True
    t = d.get("type")
    if isinstance(t, list):
        return True
    if isinstance(t, str) and t in _JSON_ONLY_TYPES:
        return True
    return isinstance(d.get("properties"), dict)


def _pick_type(t: Any) -> Any:
    """A JSON Schema `type` may be a list (e.g. ["string","null"]); take the
    first non-null entry."""
    if isinstance(t, list):
        return next((x for x in t if x != "null"), None)
    return t


def json_schema_to_node(node: Any) -> Node:
    """Translate a standard JSON Schema node directly into a scoring Node,
    choosing a comparator from each field's type/format/enum.

    Mapping: number/integer → "numeric"; boolean/enum → "categorical"; string
    with a date/time `format` → "date"; other string → "string"; object →
    fields from `properties`; array → item from `items`; anything unrecognised
    (``$ref``, combinators, untyped) → "string" (best-effort text match).

    Building Nodes directly (rather than an intermediate field-schema dict)
    avoids the leaf-vs-object ambiguity when a property is literally named
    "type"."""
    if not isinstance(node, dict):
        return LeafNode("string", {})
    if isinstance(node.get("enum"), list):
        return LeafNode("categorical", {})
    t = _pick_type(node.get("type"))
    if t == "object" or isinstance(node.get("properties"), dict):
        props = node.get("properties")
        fields = (
            {k: json_schema_to_node(v) for k, v in props.items()}
            if isinstance(props, dict)
            else {}
        )
        return ObjectNode(fields)
    if t == "array" or "items" in node:
        items = node.get("items")
        if isinstance(items, list):  # tuple validation → use the first item schema
            items = items[0] if items else None
        item = (
            json_schema_to_node(items)
            if isinstance(items, dict)
            else LeafNode("string", {})
        )
        return ArrayNode(item)
    if t in ("number", "integer"):
        return LeafNode("numeric", {})
    if t == "boolean":
        return LeafNode("categorical", {})
    if t == "string":
        fmt = node.get("format")
        if isinstance(fmt, str) and fmt.lower() in _DATE_FORMATS:
            return LeafNode("date", {})
        return LeafNode("string", {})
    return LeafNode("string", {})


def normalize_schema(raw: Any) -> Node:
    """Convert a user-facing schema (str / dict / single-element list) into nodes.

    Rules:
      - str                         -> LeafNode(name)
      - dict with "type" -> known comparator -> LeafNode(type, params)
      - any other dict              -> ObjectNode (values recursively normalized)
      - single-element list         -> ArrayNode (the element is the item-schema)
    A dict whose only structural cue is a field literally named "type" mapping to
    a comparator name will be read as a leaf — a documented limitation.
    """
    # Accept a standard JSON Schema by translating it straight to nodes.
    if isinstance(raw, dict) and _looks_like_json_schema(raw):
        return json_schema_to_node(raw)

    if isinstance(raw, str):
        if not is_comparator(raw):
            raise ValueError(
                f"Unknown comparator {raw!r}. Use a registered comparator name."
            )
        return LeafNode(raw, {})

    if isinstance(raw, list):
        if len(raw) != 1:
            raise ValueError(
                "array schema must be a single-element list [item_schema]"
            )
        return ArrayNode(normalize_schema(raw[0]))

    if isinstance(raw, dict):
        t = raw.get("type")
        if isinstance(t, str) and is_comparator(t):
            params = {k: v for k, v in raw.items() if k != "type"}
            get_comparator(t, **params)  # validate name + params early
            return LeafNode(t, params)
        return ObjectNode({k: normalize_schema(v) for k, v in raw.items()})

    raise TypeError(f"Invalid schema node: {raw!r}")
