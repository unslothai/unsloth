# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
from typing import Any

from eval.json_score import json_anls_score, score_from_text
from eval.json_score.core import ScoreNode
from .base import ConfigField, MetricResult, MetricSpec, Scorer


def _serialize(node: ScoreNode) -> dict:
    out: dict[str, Any] = {"score": node.score, "n_leaves": node.n_leaves}
    if node.note is not None:
        out["note"] = node.note
    if node.matched_option is not None:
        out["matched_option"] = node.matched_option
    children = node.children
    if isinstance(children, dict):
        out["children"] = {k: _serialize(v) for k, v in children.items()}
    elif isinstance(children, list):
        out["children"] = [_serialize(v) for v in children]
    return out


def _build(config: dict) -> Scorer:
    schema = config.get("schema")
    default_comparator = config.get("default_comparator", "string")

    def score(prediction: str, reference: Any) -> MetricResult:
        ref = reference
        if isinstance(reference, str):
            try:
                ref = json.loads(reference)
            except (ValueError, TypeError):
                return MetricResult(score=0.0, error="reference is not valid JSON")
        try:
            value, node = score_from_text(
                ref, prediction, schema,
                default_comparator=default_comparator, return_key_scores=True,
            )
        except (ValueError, TypeError) as exc:
            return MetricResult(score=0.0, error=f"scoring failed: {exc}")
        err = None if node.note != "unparseable prediction" else "unparseable prediction"
        return MetricResult(score=value, breakdown=_serialize(node), error=err)

    return score


SPEC = MetricSpec(
    name="json_document",
    label="JSON document score",
    reference_kind="json",
    config_fields=[
        ConfigField("schema", "json", None, "Field schema or JSON Schema (optional)"),
        ConfigField("default_comparator", "string", "string", "Default comparator"),
    ],
    build=_build,
)
