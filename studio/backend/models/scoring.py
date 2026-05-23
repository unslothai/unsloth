# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Request/response models for the document scoring API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ScoreRequest(BaseModel):
    # `schema` is the natural JSON key but shadows BaseModel.schema, so expose it
    # via an alias and accept either name.
    model_config = ConfigDict(populate_by_name=True)

    ground_truth: Any = Field(..., description="Reference (ground-truth) JSON document.")
    prediction: Any = Field(..., description="Model-predicted JSON document.")
    score_schema: Optional[Any] = Field(
        default=None,
        alias="schema",
        description=(
            "Per-field comparator schema (field path -> comparator). "
            "Omit to score every leaf as a string (ANLS)."
        ),
    )
    default_comparator: str = Field(
        default="string",
        description="Comparator for fields not named in the schema.",
    )
    return_key_scores: bool = Field(
        default=True,
        description="Include the per-field breakdown tree in the response.",
    )


class ScoreResponse(BaseModel):
    score: float = Field(..., description="Overall document score in [0, 1].")
    breakdown: Optional[dict] = Field(
        default=None,
        description="Per-field score tree (present when return_key_scores is true).",
    )
