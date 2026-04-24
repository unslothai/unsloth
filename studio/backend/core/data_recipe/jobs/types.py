# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


JobStatus = Literal[
    "created",
    "pending",
    "active",
    "cancelling",
    "cancelled",
    "error",
    "completed",
]


@dataclass
class Progress:
    done: int | None = None
    total: int | None = None
    percent: float | None = None
    eta_sec: float | None = None
    rate: float | None = None
    ok: int | None = None
    failed: int | None = None


@dataclass
class BatchProgress:
    idx: int | None = None
    total: int | None = None


@dataclass
class SourceProgress:
    source: str = "github"
    status: str | None = None
    repo: str | None = None
    resource: str | None = None
    page: int | None = None
    page_items: int | None = None
    fetched_items: int | None = None
    estimated_total: int | None = None
    percent: float | None = None
    rate_remaining: int | None = None
    retry_after_sec: int | None = None
    message: str | None = None
    updated_at: float | None = None


@dataclass
class ModelUsage:
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    tps: float | None = None
    requests_success: int | None = None
    requests_failed: int | None = None
    requests_total: int | None = None
    rpm: float | None = None


@dataclass
class Job:
    job_id: str
    status: JobStatus = "created"
    stage: str | None = None
    current_column: str | None = None
    progress: Progress = field(default_factory = Progress)
    column_progress: Progress = field(default_factory = Progress)
    batch: BatchProgress = field(default_factory = BatchProgress)
    source_progress: SourceProgress | None = None
    rows: int | None = None
    cols: int | None = None
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None

    analysis: dict[str, Any] | None = None
    artifact_path: str | None = None
    execution_type: str | None = None
    dataset: list[dict[str, Any]] | None = None
    processor_artifacts: dict[str, Any] | None = None
    model_usage: dict[str, ModelUsage] = field(default_factory = dict)
    progress_columns_total: int | None = None
    source_progress_estimated_total: int | None = None
    completed_columns: list[str] = field(default_factory = list)
    # Id of the internal sk-unsloth-* API key minted for a local-model
    # workflow. Revoked when the job terminates so the key's live window
    # matches the run rather than its 24h TTL.
    internal_api_key_id: int | None = None
    _current_usage_model: str | None = None
    _in_usage_summary: bool = False
    _seen_generation_columns: list[str] = field(default_factory = list)
    _column_done: dict[str, int] = field(default_factory = dict)
    _source_counts: dict[str, int] = field(default_factory = dict)
    _source_seen_pages: set[str] = field(default_factory = set)
