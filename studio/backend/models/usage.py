# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pydantic schemas for Token Usage Analytics."""

from typing import Literal, Optional, List
from pydantic import BaseModel


class UsageEvent(BaseModel):
    id: str
    ts: int
    model: str
    source: Literal["local", "api"]
    provider: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    endpoint: Optional[str] = None
    status: str
    session_id: Optional[str] = None


class UsageSummaryRow(BaseModel):
    period: str
    model: str
    source: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class UsageSummaryResponse(BaseModel):
    rows: List[UsageSummaryRow]
    granularity: str


class UsageRetentionSetting(BaseModel):
    mode: Literal["months", "forever"] = "forever"
    value: Optional[int] = None


class UsageExportParams(BaseModel):
    format: Literal["csv", "json"] = "csv"
    start: Optional[int] = None
    end: Optional[int] = None
