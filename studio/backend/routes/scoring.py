# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Document scoring API — type-aware JSON document score (see eval.json_score)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from loggers import get_logger

from auth.authentication import get_current_subject
from eval.json_score import json_anls_score
from eval.json_score.core import ScoreNode
from models import ScoreRequest, ScoreResponse

logger = get_logger(__name__)

router = APIRouter()


def _serialize(node: ScoreNode) -> dict:
    """Flatten a ScoreNode tree into JSON-friendly nested dicts."""
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


@router.post("/score", response_model=ScoreResponse)
async def score_document(
    payload: ScoreRequest,
    current_subject: str = Depends(get_current_subject),
):
    """Score a predicted JSON document against ground truth, field by field."""
    try:
        score, node = json_anls_score(
            payload.ground_truth,
            payload.prediction,
            payload.score_schema,
            default_comparator=payload.default_comparator,
            return_key_scores=True,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid schema or input: {exc}")

    breakdown = _serialize(node) if payload.return_key_scores else None
    return ScoreResponse(score=score, breakdown=breakdown)
