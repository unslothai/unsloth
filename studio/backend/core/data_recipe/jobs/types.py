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
    progress: Progress = field(default_factory=Progress)
    batch: BatchProgress = field(default_factory=BatchProgress)
    rows: int | None = None
    cols: int | None = None
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None

    analysis: dict[str, Any] | None = None
    artifact_path: str | None = None
    model_usage: dict[str, ModelUsage] = field(default_factory=dict)
    _current_usage_model: str | None = None
    _in_usage_summary: bool = False

