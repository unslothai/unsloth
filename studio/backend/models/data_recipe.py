# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Pydantic schemas for Data Recipe (DataDesigner) API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RecipePayload(BaseModel):
    recipe: dict[str, Any] = Field(default_factory = dict)
    run: dict[str, Any] | None = None
    ui: dict[str, Any] | None = None


class PreviewResponse(BaseModel):
    dataset: list[dict[str, Any]] = Field(default_factory = list)
    processor_artifacts: dict[str, Any] | None = None
    analysis: dict[str, Any] | None = None


class ValidateError(BaseModel):
    message: str
    path: str | None = None
    code: str | None = None


class ValidateResponse(BaseModel):
    valid: bool
    errors: list[ValidateError] = Field(default_factory = list)
    raw_detail: str | None = None


class JobCreateResponse(BaseModel):
    job_id: str


class SeedInspectRequest(BaseModel):
    dataset_name: str = Field(min_length = 1)
    hf_token: str | None = None
    subset: str | None = None
    split: str | None = "train"
    preview_size: int = Field(default = 10, ge = 1, le = 50)


class SeedInspectUploadRequest(BaseModel):
    filename: str = Field(min_length = 1)
    content_base64: str = Field(min_length = 1)
    preview_size: int = Field(default = 10, ge = 1, le = 50)
    seed_source_type: str | None = None
    unstructured_chunk_size: int | None = Field(default = None, ge = 1, le = 20000)
    unstructured_chunk_overlap: int | None = Field(default = None, ge = 0, le = 20000)


class SeedInspectResponse(BaseModel):
    dataset_name: str
    resolved_path: str
    columns: list[str] = Field(default_factory = list)
    preview_rows: list[dict[str, Any]] = Field(default_factory = list)
    split: str | None = None
    subset: str | None = None


class McpToolsListRequest(BaseModel):
    mcp_providers: list[dict[str, Any]] = Field(default_factory = list)
    timeout_sec: float | None = Field(default = None, gt = 0)


class McpToolsProviderResult(BaseModel):
    name: str
    tools: list[str] = Field(default_factory = list)
    error: str | None = None


class McpToolsListResponse(BaseModel):
    providers: list[McpToolsProviderResult] = Field(default_factory = list)
    duplicate_tools: dict[str, list[str]] = Field(default_factory = dict)
