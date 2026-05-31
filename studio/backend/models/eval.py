# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class EvalDatasetRef(BaseModel):
    is_local: bool
    path: Optional[str] = None
    name: Optional[str] = None
    split: str = "train"
    subset: Optional[str] = None


class EvalStartRequest(BaseModel):
    model_identifier: str
    dataset: EvalDatasetRef
    input_column: str
    reference_column: str
    metric_name: str
    metric_config: dict = Field(default_factory=dict)
    system_prompt: str = ""
    template: Optional[str] = None
    instruction: str = ""               # user message text when input is an image
    limit: Optional[int] = 100          # None = all rows
    max_new_tokens: int = 256
    temperature: float = 0.0


class EvalLastResult(BaseModel):
    idx: int
    score: float
    error: Optional[str] = None
    input: Optional[str] = None
    prediction: Optional[str] = None


class EvalProgress(BaseModel):
    run_id: str
    status: str
    done: int
    total: int
    avg_score: float
    eta_sec: Optional[float] = None
    last_result: Optional[EvalLastResult] = None


class EvalRunSummary(BaseModel):
    id: str
    status: str
    model_identifier: str
    dataset_ref: str
    metric_name: str
    started_at: str
    ended_at: Optional[str] = None
    num_examples: Optional[int] = None
    avg_score: Optional[float] = None
    display_name: Optional[str] = None


class EvalResultRow(BaseModel):
    idx: int
    input_text: Optional[str] = None
    prediction_text: Optional[str] = None
    reference_text: Optional[str] = None
    score: Optional[float] = None
    breakdown: Optional[dict] = None
    error: Optional[str] = None


class EvalRunDetail(BaseModel):
    run: EvalRunSummary
    results: list[EvalResultRow]
    total_results: int


class MetricConfigField(BaseModel):
    name: str
    type: str
    default: Any = None
    label: str
    options: Optional[list[str]] = None


class MetricInfo(BaseModel):
    name: str
    label: str
    reference_kind: str
    config_fields: list[MetricConfigField]
