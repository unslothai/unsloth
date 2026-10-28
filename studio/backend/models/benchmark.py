# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pydantic schemas for Benchmark API."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Any


class BenchmarkTaskInfo(BaseModel):
    """Metadata for an available benchmark task."""

    model_config = ConfigDict(populate_by_name = True)

    id: str = Field(..., description = "Task identifier (e.g. 'mmlu')")
    name: str = Field(..., description = "Display name (e.g. 'MMLU')")
    description: Optional[str] = Field(None, description = "Short description")
    task_type: Optional[str] = Field(
        None, description = "Task type: 'log_likelihood' or 'generation'"
    )


class BenchmarkTasksResponse(BaseModel):
    """Available benchmark tasks list."""

    model_config = ConfigDict(populate_by_name = True)

    tasks: list[BenchmarkTaskInfo] = Field(..., description = "Available benchmark tasks")


class BenchmarkRunRequest(BaseModel):
    """Request for running a benchmark on a model."""

    model_config = ConfigDict(populate_by_name = True)

    checkpoint_path: str = Field(..., description = "Path to the model checkpoint")
    model_source: str = Field(
        ...,
        description = "Source of the model: 'checkpoint', 'hf', or 'local'",
    )
    hf_token: Optional[str] = Field(
        None,
        description = "Hugging Face token for accessing gated models",
    )
    task: str = Field("mmlu", description = "Benchmark task to run (e.g. 'mmlu', 'hellaswag')")
    batch_size: str = Field(
        "auto", description = "Concurrency for lm_eval (integer or 'auto' = server slot count)"
    )
    num_fewshot: Optional[int] = Field(
        None, description = "Number of few-shot examples (None = task default)"
    )
    max_tokens: Optional[int] = Field(
        None, description = "Maximum tokens to generate per sample (None = 32768)"
    )


class BenchmarkStatusResponse(BaseModel):
    """Current benchmark backend status."""

    model_config = ConfigDict(populate_by_name = True)

    is_benchmark_active: bool = Field(
        False,
        description = "True while a benchmark operation is running",
    )
    last_op_status: Optional[str] = Field(
        None,
        description = "Outcome of the most recently finished op: success / error / cancelled",
    )
    last_op_error: Optional[str] = Field(
        None,
        description = "Error message of the most recently finished op, if it failed",
    )
    last_op_seq: Optional[int] = Field(
        None,
        description = "Counter of runs started; tells a client whether the finished op is the one it started",
    )


class BenchmarkOperationResponse(BaseModel):
    """Generic response for benchmark operations."""

    model_config = ConfigDict(populate_by_name = True)

    success: bool = Field(..., description = "True if the operation succeeded")
    message: str = Field(..., description = "Human-readable status or error message")
    details: Optional[dict[str, Any]] = Field(
        default = None,
        description = "Optional extra details about the operation",
    )


class BenchmarkTaskConfigResponse(BaseModel):
    """Task config metadata loaded from the task's YAML."""

    model_config = ConfigDict(populate_by_name = True)

    task_id: str = Field(..., description = "Task identifier")
    num_fewshot: Optional[int] = Field(
        None, description = "Default few-shot count (null = unknown, 0 = not supported)"
    )


class BenchmarkRunMetric(BaseModel):
    """A single metric value for a benchmark run."""

    model_config = ConfigDict(populate_by_name = True)

    name: str = Field(..., description = "Metric name (e.g. 'exact_match,strict-match')")
    score: float = Field(..., description = "Metric score")
    stderr: Optional[str] = Field(None, description = "Standard error, if available")


class BenchmarkRunSummary(BaseModel):
    """Summary of a completed benchmark run, as shown in a list."""

    model_config = ConfigDict(populate_by_name = True)

    id: str = Field(..., description = "Run identifier (directory name)")
    task: str = Field(..., description = "Benchmark task name")
    model: str = Field(..., description = "Model identifier")
    metrics: list[BenchmarkRunMetric] = Field(..., description = "All metric values")
    default_metric: str = Field(
        "", description = "Key of the primary metric (e.g. 'exact_match,strict-match')"
    )
    n_samples: int = Field(0, description = "Number of samples evaluated")
    num_fewshot: Optional[int] = Field(None, description = "Few-shot count")
    created_at: str = Field(..., description = "ISO timestamp of the run")
    output_path: str = Field(..., description = "Absolute path to results directory")


class BenchmarkRunListResponse(BaseModel):
    """Paginated list of past benchmark runs."""

    model_config = ConfigDict(populate_by_name = True)

    runs: list[BenchmarkRunSummary] = Field(..., description = "Past benchmark runs, newest first")


class BenchmarkSampleResult(BaseModel):
    """Result for a single evaluation sample."""

    model_config = ConfigDict(populate_by_name = True)

    doc_id: int = Field(..., description = "Sample index")
    question: str = Field(..., description = "Input question")
    target: str = Field(..., description = "Expected answer")
    response: Optional[str] = Field(None, description = "Model response (filtered)")
    raw_response: Optional[str] = Field(None, description = "Raw model response")
    correct: bool = Field(False, description = "Whether the model answered correctly")


class BenchmarkRunDetail(BaseModel):
    """Full detail for a single benchmark run."""

    model_config = ConfigDict(populate_by_name = True)

    id: str = Field(..., description = "Run identifier (directory name)")
    task: str = Field(..., description = "Benchmark task name")
    model: str = Field(..., description = "Model identifier")
    metrics: list[BenchmarkRunMetric] = Field(..., description = "All metrics")
    n_samples: int = Field(0, description = "Number of samples evaluated")
    num_fewshot: Optional[int] = Field(None, description = "Few-shot count")
    created_at: str = Field(..., description = "ISO timestamp of the run")
    output_path: str = Field(..., description = "Absolute path to results directory")
    samples: list[BenchmarkSampleResult] = Field(
        default_factory = list, description = "Per-sample results"
    )
    correct_count: int = Field(0, description = "Number of correct samples")
    total_count: int = Field(0, description = "Total number of samples")


class BenchmarkGraphRequest(BaseModel):
    """Request to generate a benchmark comparison chart."""

    model_config = ConfigDict(populate_by_name = True)

    run_ids: list[str] = Field(..., description = "Ordered list of run IDs to chart")
    chart_type: str = Field(
        "bar",
        description = "Chart type: 'bar', 'line', 'grouped_bar', or 'radar'",
    )
    metric: str = Field(..., description = "Metric name to plot (e.g. 'exact_match,strict-match')")
    width: float = Field(10.0, description = "Chart width in inches", ge = 4, le = 20)
    height: float = Field(6.0, description = "Chart height in inches", ge = 3, le = 14)
    theme: str = Field(
        "unsloth-dark",
        description = "Chart theme: 'unsloth-dark', 'dark', 'light', a built-in matplotlib style, or a custom style name",
    )


class BenchmarkExportRequest(BaseModel):
    """Request to export benchmark runs as JSON."""

    model_config = ConfigDict(populate_by_name = True)

    run_ids: list[str] = Field(..., description = "List of run IDs to export")
