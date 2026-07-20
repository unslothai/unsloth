# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Schemas for account-scoped Studio assets."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

URL_SEGMENT_ID_PATTERN = r"^[^/\\]+$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra = "forbid", strict = True)


class RecipeCreateRequest(StrictModel):
    id: str = Field(min_length = 1, max_length = 128, pattern = URL_SEGMENT_ID_PATTERN)
    name: str = Field(min_length = 1, max_length = 200)
    payload: dict[str, Any]
    learningRecipeId: str | None = Field(default = None, max_length = 128)
    learningRecipeTitle: str | None = Field(default = None, max_length = 200)
    createdAt: int | None = Field(default = None, ge = 0)


class RecipeUpdateRequest(StrictModel):
    id: str | None = Field(
        default = None, min_length = 1, max_length = 128, pattern = URL_SEGMENT_ID_PATTERN
    )
    name: str = Field(min_length = 1, max_length = 200)
    payload: dict[str, Any]
    learningRecipeId: str | None = Field(default = None, max_length = 128)
    learningRecipeTitle: str | None = Field(default = None, max_length = 200)
    revision: int = Field(ge = 1)
    createdAt: int | None = Field(default = None, ge = 0)
    updatedAt: int | None = Field(default = None, ge = 0)


class RecipeRecord(StrictModel):
    id: str
    name: str
    payload: dict[str, Any]
    learningRecipeId: str | None = None
    learningRecipeTitle: str | None = None
    revision: int
    createdAt: int
    updatedAt: int


class RecipeListResponse(StrictModel):
    recipes: list[RecipeRecord]


class ExecutionUpsertRequest(StrictModel):
    id: str | None = Field(
        default = None, min_length = 1, max_length = 128, pattern = URL_SEGMENT_ID_PATTERN
    )
    recipeId: str | None = Field(
        default = None, min_length = 1, max_length = 128, pattern = URL_SEGMENT_ID_PATTERN
    )
    revision: int | None = Field(default = None, ge = 0)
    jobId: str | None = Field(default = None, max_length = 128)
    kind: str | None = None
    run_name: str | None = Field(default = None, max_length = 200)
    status: str | None = None
    rows: int | None = Field(default = None, ge = 0)
    recipeSignature: str | None = None
    stage: str | None = Field(default = None, max_length = 200)
    current_column: str | None = Field(default = None, max_length = 200)
    completed_columns: list[str] | None = None
    progress: dict[str, Any] | None = None
    column_progress: dict[str, Any] | None = None
    batch: dict[str, Any] | None = None
    source_progress: dict[str, Any] | None = None
    model_usage: dict[str, Any] | None = None
    lastEventId: int | None = Field(default = None, ge = 0)
    datasetTotal: int | None = Field(default = None, ge = 0)
    analysis: dict[str, Any] | None = None
    error: str | None = None
    createdAt: int = Field(ge = 0)
    updatedAt: int | None = Field(default = None, ge = 0)
    finishedAt: int | None = Field(default = None, ge = 0)


class ExecutionRecord(ExecutionUpsertRequest):
    id: str
    recipeId: str
    revision: int
    updatedAt: int


class ExecutionListResponse(StrictModel):
    executions: list[ExecutionRecord]
    nextCursor: str | None = None
    resumable: ExecutionRecord | None = None


class ImportLedger(StrictModel):
    source: str
    recipes: list[str]
    executions: list[str]


class BootstrapResponse(StrictModel):
    subject: str
    importLedger: ImportLedger


class LegacyImportRequest(StrictModel):
    source: str = Field(min_length = 1, max_length = 128)
    confirmSubject: str = Field(min_length = 1)
    recipes: list[Any] = Field(default_factory = list, max_length = 100)
    executions: list[Any] = Field(default_factory = list, max_length = 500)


class LegacyImportItemResult(StrictModel):
    id: str
    outcome: Literal[
        "imported",
        "already_imported",
        "redacted",
        "id_retired",
        "rejected",
        "missing_parent",
    ]
    reason: str | None = None
    redactedPaths: list[str] | None = None


class LegacyImportResponse(StrictModel):
    recipes: list[LegacyImportItemResult]
    executions: list[LegacyImportItemResult]
    summary: dict[str, int]
