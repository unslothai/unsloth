# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pydantic schemas for Benchmark API."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Optional, Any


class RecipeHeldoutBaseline(BaseModel):
    mode: Literal["none", "adapter_off", "alias"] = "none"
    alias: Optional[str] = None


class RecipeHeldoutSpec(BaseModel):
    """What the Recipe held-out task scores: held-out rows and the model blocks that answer and judge them."""

    rows: list[dict[str, Any]] = Field(default_factory = list, max_length = 200)
    max_rows: int = Field(20, ge = 1, le = 200)
    prompt_column: str = Field(..., min_length = 1)
    reference_column: Optional[str] = None
    system_prompt: str = ""
    system_override: bool = Field(
        False, description = "system_prompt replaces each row's own system message"
    )
    system_column: Optional[str] = Field(
        None, description = "A column holding each row's own system message, as training used it"
    )
    scorer: Literal["reference", "json", "markdown", "python", "judge"] = "reference"
    rubric: str = ""
    model_alias: str = Field(..., min_length = 1)
    baseline: RecipeHeldoutBaseline = RecipeHeldoutBaseline()
    judge_alias: Optional[str] = None
    local_adapter: bool = Field(
        False, description = "The local model has a LoRA loaded; a local judge turns it off"
    )
    model_providers: list[dict[str, Any]] = Field(default_factory = list)
    model_configs: list[dict[str, Any]] = Field(default_factory = list)


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
    limit: Optional[int] = Field(
        None, ge = 1, description = "Score only the first N samples of each task (None = all)"
    )
    recipe: Optional[RecipeHeldoutSpec] = Field(
        None, description = "Inputs for the recipe_heldout task (Flows Benchmark node)"
    )
    expected_model: Optional[str] = Field(
        None,
        max_length = 1024,
        description = "The model the caller loaded to score; the run fails if another is loaded "
        "by the time scoring starts",
    )
    run_token: Optional[str] = Field(
        None,
        max_length = 64,
        description = "A caller-chosen token; /cancel with the same token stops only this run",
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


class BenchmarkRunMetric(BaseModel):
    """A single metric value for a benchmark run."""

    model_config = ConfigDict(populate_by_name = True)

    name: str = Field(..., description = "Metric name (e.g. 'exact_match,strict-match')")
    score: float = Field(..., description = "Metric score")
    stderr: Optional[str] = Field(None, description = "Standard error, if available")


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
