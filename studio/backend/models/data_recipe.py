"""
Pydantic schemas for Data Recipe (DataDesigner) API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RecipePayload(BaseModel):
    recipe: dict[str, Any] = Field(default_factory=dict)
    run: dict[str, Any] | None = None
    ui: dict[str, Any] | None = None


class PreviewResponse(BaseModel):
    dataset: list[dict[str, Any]] = Field(default_factory=list)
    processor_artifacts: dict[str, Any] | None = None


class ValidateError(BaseModel):
    message: str
    path: str | None = None
    code: str | None = None


class ValidateResponse(BaseModel):
    valid: bool
    errors: list[ValidateError] = Field(default_factory=list)
    raw_detail: str | None = None


class JobCreateResponse(BaseModel):
    job_id: str

