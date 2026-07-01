# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

from .comparators import get_comparator, is_comparator


@dataclass
class LeafNode:
    comparator: str
    params: dict = field(default_factory = dict)


@dataclass
class ObjectNode:
    fields: dict  # str -> Node


@dataclass
class ArrayNode:
    item: "Node"


Node = Union[LeafNode, ObjectNode, ArrayNode]


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
    if isinstance(raw, str):
        if not is_comparator(raw):
            raise ValueError(f"Unknown comparator {raw!r}. Use a registered comparator name.")
        return LeafNode(raw, {})

    if isinstance(raw, list):
        if len(raw) != 1:
            raise ValueError("array schema must be a single-element list [item_schema]")
        return ArrayNode(normalize_schema(raw[0]))

    if isinstance(raw, dict):
        t = raw.get("type")
        if isinstance(t, str) and is_comparator(t):
            params = {k: v for k, v in raw.items() if k != "type"}
            get_comparator(t, **params)  # validate name + params early
            return LeafNode(t, params)
        return ObjectNode({k: normalize_schema(v) for k, v in raw.items()})

    raise TypeError(f"Invalid schema node: {raw!r}")
