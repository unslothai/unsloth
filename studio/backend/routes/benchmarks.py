# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Benchmark run storage, backed by studio.db. The frontend runs the sweep and posts
the run here as it goes, so a run survives a reload and shows up on every device."""

from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from auth.authentication import authenticated_via_api_key, get_current_subject
from hub.utils.host_paths import host_paths_visible, redact_host_paths
from storage.benchmark_runs_db import delete_run, get_run, list_runs, upsert_run

router = APIRouter()

MAX_RESULTS = 20_000
MAX_RUNS_LISTED = 200


def _redact_run(run: dict[str, Any], *, via_api_key: bool) -> dict[str, Any]:
    """`model` and `config.tuneModel` hold the resolved absolute path of a local GGUF, the same
    host identity the inference status route hides from API-key callers. The path redactor keys
    on field name and neither is one it knows, so route them through identity keys it does and
    map the references back."""
    if host_paths_visible(via_api_key):
        return run
    config = run.get("config")
    tune = config.get("tuneModel") if isinstance(config, dict) else None
    ids = redact_host_paths(
        {"active_model": run.get("model"), "model_name": tune},
        via_api_key = via_api_key,
    )
    out = dict(run)
    out["model"] = ids["active_model"]
    if isinstance(config, dict) and "tuneModel" in config:
        out["config"] = {**config, "tuneModel": ids["model_name"]}
    return out


class BenchmarkResult(BaseModel):
    model_config = ConfigDict(extra = "ignore")

    variant: str = Field(max_length = 500)
    rep: int = Field(ge = 0)
    warmup: bool = False
    promptIndex: int = Field(default = 0, ge = 0)
    tps: Optional[float] = None
    promptTps: Optional[float] = None
    promptTokens: Optional[int] = None
    genTokens: Optional[int] = None
    ttftMs: Optional[float] = None
    wallMs: float
    clientTps: Optional[float] = None
    draftN: Optional[int] = None
    draftAccepted: Optional[int] = None
    loadMs: Optional[float] = None
    at: int


class BenchmarkRun(BaseModel):
    model_config = ConfigDict(extra = "ignore")

    id: str = Field(max_length = 128)
    kind: Literal["sweep"] = "sweep"
    sweep: str = Field(max_length = 64)
    model: str = Field(max_length = 1000)
    ggufVariant: Optional[str] = Field(default = None, max_length = 200)
    kv: Optional[str] = Field(default = None, max_length = 64)
    context: Optional[int] = None
    config: dict[str, Any]
    meta: dict[str, Any] = Field(default_factory = dict)
    base: list[dict[str, Any]] = Field(default_factory = list, max_length = 200)
    outcomes: list[dict[str, Any]] = Field(default_factory = list, max_length = 2_000)
    results: list[BenchmarkResult] = Field(default_factory = list, max_length = MAX_RESULTS)
    createdAt: int
    finishedAt: Optional[int] = None


@router.get("/runs")
def get_runs(
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    return {
        "runs": [
            _redact_run(run, via_api_key = via_api_key)
            for run in list_runs(limit = MAX_RUNS_LISTED)
        ]
    }


@router.get("/runs/{run_id}")
def get_one(
    run_id: str,
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code = 404, detail = "Benchmark run not found")
    return _redact_run(run, via_api_key = via_api_key)


@router.put("/runs/{run_id}")
def put_run(
    run_id: str,
    run: BenchmarkRun,
    current_subject: str = Depends(get_current_subject),
):
    if run.id != run_id:
        raise HTTPException(status_code = 400, detail = "ID mismatch")
    return upsert_run(run.model_dump())


@router.delete("/runs/{run_id}", status_code = 204)
def remove_run(run_id: str, current_subject: str = Depends(get_current_subject)):
    if not delete_run(run_id):
        raise HTTPException(status_code = 404, detail = "Benchmark run not found")
