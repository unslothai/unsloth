# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import re
from typing import Any

from .core import ScoreNode, _leaf_count, _score
from .schema import Node, normalize_schema

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


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
    """
    node = normalize_schema(schema) if schema is not None else None
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
    """
    pred = _extract_json(raw_text)
    if pred is None:
        node: Node | None = normalize_schema(schema) if schema is not None else None
        n = max(_leaf_count(ground_truth, node), 1)
        zero = ScoreNode(0.0, n, note="unparseable prediction")
        if return_key_scores:
            return 0.0, zero
        return 0.0
    return json_anls_score(
        ground_truth,
        pred,
        schema,
        default_comparator=default_comparator,
        return_key_scores=return_key_scores,
    )
