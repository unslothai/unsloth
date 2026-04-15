# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Pydantic schemas for Data Recipe (DataDesigner) API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


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


class PublishDatasetRequest(BaseModel):
    repo_id: str = Field(min_length = 3, description = "Hugging Face dataset repo ID")
    description: str = Field(
        min_length = 1,
        max_length = 4000,
        description = "Short dataset description for the dataset card",
    )
    hf_token: str | None = Field(
        default = None,
        description = "Optional Hugging Face token for private or write-protected repos",
    )
    private: bool = Field(
        default = False,
        description = "Create or update the dataset repo as private",
    )
    artifact_path: str | None = Field(
        default = None,
        description = "Execution artifact path captured by the UI for completed runs",
    )


class PublishDatasetResponse(BaseModel):
    success: bool = True
    url: str
    message: str


class SeedInspectRequest(BaseModel):
    dataset_name: str = Field(min_length = 1)
    hf_token: str | None = None
    subset: str | None = None
    split: str | None = "train"
    preview_size: int = Field(default = 10, ge = 1, le = 50)


class SeedInspectUploadRequest(BaseModel):
    # Legacy single-file flow (mutually exclusive with file_ids)
    filename: str | None = None
    content_base64: str | None = None
    # Multi-file flow (mutually exclusive with content_base64)
    block_id: str | None = None
    file_ids: list[str] | None = None
    file_names: list[str] | None = None
    # Shared fields
    preview_size: int = Field(default = 10, ge = 1, le = 50)
    seed_source_type: str | None = None
    unstructured_chunk_size: int | None = Field(default = None, ge = 1, le = 20000)
    unstructured_chunk_overlap: int | None = Field(default = None, ge = 0, le = 20000)

    @model_validator(mode = "after")
    def _check_mutual_exclusivity(self) -> "SeedInspectUploadRequest":
        has_legacy = self.content_base64 is not None
        has_multi = self.file_ids is not None
        if has_legacy and has_multi:
            raise ValueError("Provide either content_base64 or file_ids, not both")
        if not has_legacy and not has_multi:
            raise ValueError("Provide either content_base64 or file_ids")
        if has_multi:
            if len(self.file_ids) == 0:
                raise ValueError("file_ids must not be empty")
            if not self.block_id:
                raise ValueError("block_id is required when using file_ids")
            if self.file_names is None or len(self.file_ids) != len(self.file_names):
                raise ValueError(
                    "file_names must be provided and same length as file_ids"
                )
        if has_legacy:
            if not self.filename:
                raise ValueError("filename is required when using content_base64")
        return self


class SeedInspectResponse(BaseModel):
    dataset_name: str
    resolved_path: str
    columns: list[str] = Field(default_factory = list)
    preview_rows: list[dict[str, Any]] = Field(default_factory = list)
    split: str | None = None
    subset: str | None = None
    resolved_paths: list[str] | None = None


class UnstructuredFileUploadResponse(BaseModel):
    file_id: str
    filename: str
    size_bytes: int
    status: str  # "ok" or "error"
    error: str | None = None


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
