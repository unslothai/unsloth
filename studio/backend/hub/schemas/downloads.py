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
        description = "Legacy transport flag, superseded by transport_mode. Kept so an older "
        "frontend or a scripted caller keeps working.",
    )
    transport_mode: Optional[Literal["auto", "xet", "http"]] = Field(
        None,
        description = "Transport preference. 'auto' (the default in the UI) lets the backend pick "
        "per machine: it knows this host's RAM, its hf_xet build, and whether Xet has "
        "been failing here. 'xet'/'http' force one. Omitted -> use_xet decides.",
    )
    scope_id: Optional[str] = Field(
        None,
        description = "Marks a partial-by-design download of `files` only (e.g. 'diffusion', "
        "whose loader reads a scoped subset of a repo). Keyed separately from the full "
        "snapshot of the same repo, so neither one's manifest describes the other.",
    )
    files: List[str] = Field(
        default_factory = list,
        description = "Exact files to fetch. Required with scope_id, ignored without it.",
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
    # True when this start attached to a job already running rather than starting one.
    attached: bool = False
    generation: int
    # The transport the job is really on, which can differ from the one the client asked for.
    transport: Optional[str] = None
    # Set only when an adopted job had fallen back from Xet to HTTP: stopping it still writes the
    # original marker, so it is a restart, not a resume.
    # Set only on a job that fell back from Xet to HTTP mid-flight: cancelling it still writes the original
    # transport's marker, so the partial is restart-only even though the worker is on resumable HTTP.
    cancel_transport: Optional[str] = None


class CancelDownloadResponse(BaseModel):
    job_key: str
    state: str


class ActiveDownload(BaseModel):
    """One in-flight download for a repo. ``variant`` is null for safetensors."""

    repo_id: Optional[str] = None
    variant: Optional[str] = None
    transport: Optional[str] = None
    cancel_transport: Optional[str] = None
    state: str
    files: Optional[List[str]] = Field(
        None,
        description = (
            "For a SCOPED job (variant '@name'), the exact file list it is fetching; null for a "
            "full-snapshot or variant download. Every file set of one repo shares the scope slot, "
            "so an adopting client needs this to tell whether a live job is its own transfer or a "
            "sibling checkpoint's."
        ),
    )
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
    auto_resolves_to: Literal["xet", "http"] = "xet"
    auto_reason: Optional[str] = None
    # False when the installed huggingface_hub refetches an interrupted file from zero, so no transport
    # can offer a byte-resume and the UI must not label one.
    partials_resumable: bool = True


class TransportStatusResponse(BaseModel):
    has_partial: bool
    last_transport: Optional[str] = None
    resumable: bool


class DownloadProgressResponse(BaseModel):
    downloaded_bytes: int
    # Finalized-blob bytes only: registry-loss completion fallbacks key off this, so a partial is not
    # mistaken for a finished download.
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
    cache_measured: bool = Field(
        True,
        description = (
            "False when the cache could not be scanned at all (an unreadable root). The "
            "reading is then unknown, not empty: cache_path is null either way, and "
            "hydration must not retire a persisted job on a scan that never happened."
        ),
    )
    target_present: Optional[bool] = Field(
        None,
        description = (
            "Whether this TARGET (variant) has anything in the cache, as opposed to the "
            "shared repo directory existing. Null where it cannot be established."
        ),
    )


class DownloadDatasetRequest(BaseModel):
    """Body for POST /api/hub/datasets/download.

    The HuggingFace token travels in the internal Hub token header.
    """

    repo_id: str = Field(..., description = "HuggingFace dataset repo ID")
    use_xet: bool = Field(
        True,
        description = "Legacy transport flag, superseded by transport_mode. Kept so an older "
        "frontend or a scripted caller keeps working.",
    )
    transport_mode: Optional[Literal["auto", "xet", "http"]] = Field(
        None,
        description = "Transport preference. 'auto' (the default in the UI) lets the backend pick "
        "per machine: it knows this host's RAM, its hf_xet build, and whether Xet has "
        "been failing here. 'xet'/'http' force one. Omitted -> use_xet decides.",
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
    # Attached to a job already running, rather than starting one (see above).
    attached: bool = False
    generation: int
    # The transport the job is really on, and its cancel marker (see above).
    transport: Optional[str] = None
    cancel_transport: Optional[str] = None


class CancelDatasetDownloadResponse(BaseModel):
    repo_id: str
    state: str
