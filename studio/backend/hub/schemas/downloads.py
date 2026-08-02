# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pydantic schemas for the Hub download manager (/api/hub/downloads/*)."""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional


DownloadJobState = Literal["idle", "running", "cancelling", "cancelled", "complete", "error"]


class DownloadModelRequest(BaseModel):
    """Body for POST /api/hub/download.

    The HuggingFace token travels in the internal Hub token header.
    """

    repo_id: str = Field(
        ...,
        description = "HuggingFace repo ID, e.g. 'unsloth/Qwen3-4B-GGUF'",
    )
    gguf_variant: Optional[str] = Field(
        None,
        description = "Quantization label (e.g. 'Q4_K_M'). Required for GGUF repos.",
    )
    use_xet: bool = Field(
        True,
        description = "Use Xet parallel chunked transport. Default True; set False for HTTP Range-resume.",
    )


class CancelDownloadRequest(BaseModel):
    repo_id: str = Field(..., description = "HuggingFace repo ID")
    gguf_variant: Optional[str] = Field(
        None,
        description = "GGUF variant label; omit for safetensors snapshots",
    )
    generation: Optional[int] = Field(
        None,
        description = "Download generation tag from a prior start; passing it scopes the cancel to that exact run.",
    )


class DownloadJobStatus(BaseModel):
    """Live state of a background download job."""

    state: DownloadJobState = Field(
        ...,
        description = "Current download job state.",
    )
    error: Optional[str] = Field(None, description = "Error message if state == 'error'")
    generation: int = Field(
        0,
        description = "Current run generation; an adopting client stores it so a later cancel is scoped to this exact run.",
    )


class DownloadStartResponse(BaseModel):
    job_key: str
    state: str
    accepted: bool
    generation: int


class CancelDownloadResponse(BaseModel):
    job_key: str
    state: str


class ActiveDownload(BaseModel):
    """One in-flight download for a repo. ``variant`` is null for safetensors."""

    repo_id: Optional[str] = None
    variant: Optional[str] = None
    transport: Optional[str] = None
    state: str
    generation: int = Field(
        0,
        description = "Current run generation; an adopting client stores it so a later cancel is scoped to this exact run.",
    )


class ActiveDownloadsResponse(BaseModel):
    downloads: List[ActiveDownload]


class TransportCapability(BaseModel):
    available: bool
    reason: Optional[str] = None


class TransportCapabilities(BaseModel):
    http: TransportCapability
    xet: TransportCapability


class TransportStatusResponse(BaseModel):
    has_partial: bool
    last_transport: Optional[str] = None
    resumable: bool


class DownloadProgressResponse(BaseModel):
    downloaded_bytes: int
    # Finalized-blob bytes only (no ``.incomplete``). Registry-loss completion
    # fallbacks key off this so a partial isn't mistaken for a finished download.
    completed_bytes: int = 0
    complete_on_disk: bool = Field(
        False,
        description = (
            "True only when the backend verified a usable completed snapshot/variant on disk."
        ),
    )
    expected_bytes: int
    progress: float
    cache_path: Optional[str] = None


class DownloadDatasetRequest(BaseModel):
    """Body for POST /api/hub/datasets/download.

    The HuggingFace token travels in the internal Hub token header.
    """

    repo_id: str = Field(..., description = "HuggingFace dataset repo ID")
    use_xet: bool = Field(
        True,
        description = "Use Xet parallel chunked transport. Default True; set False for HTTP Range-resume.",
    )


class CancelDatasetDownloadRequest(BaseModel):
    repo_id: str = Field(..., description = "HuggingFace dataset repo ID")
    generation: Optional[int] = Field(None, description = "Download generation")


class DatasetDownloadJobStatus(BaseModel):
    """Live state of a background dataset download job."""

    state: DownloadJobState = Field(
        ...,
        description = "Current dataset download job state.",
    )
    error: Optional[str] = Field(None, description = "Error message if state == 'error'")
    generation: int = Field(
        0,
        description = "Current run generation; an adopting client stores it so a later cancel is scoped to this exact run.",
    )


class DatasetDownloadStartResponse(BaseModel):
    repo_id: str
    state: str
    accepted: bool
    generation: int


class CancelDatasetDownloadResponse(BaseModel):
    repo_id: str
    state: str
