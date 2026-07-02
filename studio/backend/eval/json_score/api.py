# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import re
from ast import literal_eval
from typing import Any

from json_repair import repair_json

from .core import ScoreNode, _leaf_count, _score
from .schema import ArrayNode, LeafNode, Node, ObjectNode, normalize_schema

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _resolve_node(schema: Any) -> Node | None:
    """Normalize `schema` to a Node once. Passes an already-Node through
    unchanged so callers can hoist normalization out of a hot loop."""
    if schema is None:
        return None
    if isinstance(schema, (LeafNode, ObjectNode, ArrayNode)):
        return schema
    return normalize_schema(schema)


def json_anls_score(
    ground_truth: Any,
    prediction: Any,
    schema: Any = None,
    *,
    default_comparator: str = "string",
    return_key_scores: bool = False,
):
    """Score a predicted JSON document against ground truth.

    Returns a float in [0, 1], or (float, ScoreNode) if return_key_scores=True.
    ``schema`` may be a raw DSL value or an already-normalized Node.
    """
    node = _resolve_node(schema)
    result = _score(ground_truth, prediction, node, default_comparator)
    if return_key_scores:
        return result.score, result
    return result.score


def _extract_json(text: Any) -> Any | None:
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str):
        return None
    s = text.strip()
    fence = _FENCE_RE.search(s)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = s.find(open_c), s.rfind(close_c)
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(s[i : j + 1])
            except (ValueError, TypeError):
                continue
    if s.startswith(("{", "[")):
        try:
            value = literal_eval(s)
            if isinstance(value, (dict, list)):
                return value
        except (ValueError, SyntaxError):
            pass

    try:
        repaired = repair_json(s, return_objects = True)
        if isinstance(repaired, (dict, list)):
            return repaired
    except Exception:
        pass
    return None


def score_from_text(
    ground_truth: Any,
    raw_text: Any,
    schema: Any = None,
    *,
    default_comparator: str = "string",
    return_key_scores: bool = False,
):
    """Extract JSON from raw model output, then score.

    Returns the same shape as ``json_anls_score`` (a float, or ``(float,
    ScoreNode)`` when ``return_key_scores=True``). Unparseable output scores 0.0.
    ``schema`` may be a raw DSL value or an already-normalized Node.
    """
    node = _resolve_node(schema)
    pred = _extract_json(raw_text)
    if pred is None:
        n = max(_leaf_count(ground_truth, node), 1)
        zero = ScoreNode(0.0, n, note = "unparseable prediction")
        if return_key_scores:
            return 0.0, zero
        return 0.0
    result = _score(ground_truth, pred, node, default_comparator)
    if return_key_scores:
        return result.score, result
    return result.score
