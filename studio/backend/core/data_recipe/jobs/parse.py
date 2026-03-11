# SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
# Copyright © 2025 Unsloth AI

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .constants import (
    STAGE_BATCH,
    STAGE_COLUMN_CONFIG,
    STAGE_CREATE,
    STAGE_DAG,
    STAGE_GENERATING,
    STAGE_HEALTHCHECK,
    STAGE_PREVIEW,
    STAGE_PROFILING,
    STAGE_SAMPLING,
    USAGE_RESET_STAGES,
)
from .types import Job, ModelUsage, Progress


@dataclass(frozen = True)
class ParsedUpdate:
    stage: str | None = None
    current_column: str | None = None
    progress: Progress | None = None
    rows: int | None = None
    cols: int | None = None
    batch_idx: int | None = None
    batch_total: int | None = None
    usage_model: str | None = None
    usage_input_tokens: int | None = None
    usage_output_tokens: int | None = None
    usage_total_tokens: int | None = None
    usage_tps: float | None = None
    usage_requests_success: int | None = None
    usage_requests_failed: int | None = None
    usage_requests_total: int | None = None
    usage_rpm: float | None = None
    usage_section_start: bool | None = None


# kinda of a bummber but currently only option, Best effort parser from data-designer logs -> structured status for UI.
_RE_SAMPLERS = re.compile(
    r"Preparing samplers to generate (?P<rows>\d+) records across (?P<cols>\d+) columns"
)
_RE_COLCFG = re.compile(r"model config for column '(?P<col>[^']+)'")
_RE_PROCESSING_COL = re.compile(r"Processing .* column '(?P<col>[^']+)'")
_RE_PROGRESS = re.compile(
    r"progress: (?P<done>\d+)/(?P<total>\d+) \((?P<pct>\d+)%\) complete, "
    r"(?P<ok>\d+) ok, (?P<failed>\d+) failed, (?P<rate>[0-9.]+) rec/s, eta (?P<eta>[0-9.]+)s"
)
_RE_BATCH = re.compile(r"Processing batch (?P<idx>\d+) of (?P<total>\d+)")
_RE_USAGE_MODEL = re.compile(r"model:\s*(?P<model>.+)$")
_RE_USAGE_TOKENS = re.compile(
    r"tokens:\s*input=(?P<input>\d+),\s*output=(?P<output>\d+),\s*total=(?P<total>\d+),\s*tps=(?P<tps>[0-9.]+)"
)
_RE_USAGE_REQUESTS = re.compile(
    r"requests:\s*success=(?P<success>\d+),\s*failed=(?P<failed>\d+),\s*total=(?P<total>\d+),\s*rpm=(?P<rpm>[0-9.]+)"
)


def parse_log_message(msg: str) -> ParsedUpdate | None:
    m = _RE_SAMPLERS.search(msg)
    if m:
        return ParsedUpdate(
            stage = STAGE_SAMPLING,
            rows = int(m.group("rows")),
            cols = int(m.group("cols")),
        )

    if "Sorting column configs into a Directed Acyclic Graph" in msg:
        return ParsedUpdate(stage = STAGE_DAG)
    if "Running health checks for models" in msg:
        return ParsedUpdate(stage = STAGE_HEALTHCHECK)
    if "Preview generation in progress" in msg:
        return ParsedUpdate(stage = STAGE_PREVIEW)
    if "Creating Data Designer dataset" in msg:
        return ParsedUpdate(stage = STAGE_CREATE)
    if "Measuring dataset column statistics" in msg:
        return ParsedUpdate(stage = STAGE_PROFILING)

    m = _RE_COLCFG.search(msg)
    if m:
        col = m.group("col")
        return ParsedUpdate(stage = STAGE_COLUMN_CONFIG, current_column = col)

    m = _RE_PROCESSING_COL.search(msg)
    if m:
        col = m.group("col")
        return ParsedUpdate(stage = STAGE_GENERATING, current_column = col)

    m = _RE_PROGRESS.search(msg)
    if m:
        p = Progress(
            done = int(m.group("done")),
            total = int(m.group("total")),
            percent = float(m.group("pct")),
            ok = int(m.group("ok")),
            failed = int(m.group("failed")),
            rate = float(m.group("rate")),
            eta_sec = float(m.group("eta")),
        )
        return ParsedUpdate(stage = STAGE_GENERATING, progress = p)

    m = _RE_BATCH.search(msg)
    if m:
        return ParsedUpdate(
            stage = STAGE_BATCH,
            batch_idx = int(m.group("idx")),
            batch_total = int(m.group("total")),
        )

    if "Model usage summary" in msg:
        return ParsedUpdate(usage_section_start = True)

    m = _RE_USAGE_MODEL.search(msg)
    if m and "|-- model:" in msg:
        return ParsedUpdate(usage_model = str(m.group("model")).strip())

    m = _RE_USAGE_TOKENS.search(msg)
    if m:
        return ParsedUpdate(
            usage_input_tokens = int(m.group("input")),
            usage_output_tokens = int(m.group("output")),
            usage_total_tokens = int(m.group("total")),
            usage_tps = float(m.group("tps")),
        )

    m = _RE_USAGE_REQUESTS.search(msg)
    if m:
        return ParsedUpdate(
            usage_requests_success = int(m.group("success")),
            usage_requests_failed = int(m.group("failed")),
            usage_requests_total = int(m.group("total")),
            usage_rpm = float(m.group("rpm")),
        )

    return None


def apply_update(job: Job, update: ParsedUpdate) -> None:
    if update.stage is not None:
        job.stage = update.stage
    if update.current_column is not None:
        job.current_column = update.current_column
        if (
            update.stage == STAGE_GENERATING
            and update.current_column not in job._seen_generation_columns
        ):
            job._seen_generation_columns.append(update.current_column)
    if update.rows is not None:
        job.rows = update.rows
    if update.cols is not None:
        job.cols = update.cols
    if update.progress is not None:
        job.column_progress = update.progress
        if (
            job.current_column
            and update.progress.done is not None
            and update.progress.total is not None
            and update.progress.total > 0
            and update.progress.done >= update.progress.total
            and job.current_column not in job.completed_columns
        ):
            job.completed_columns.append(job.current_column)
        job.progress = _compute_overall_progress(job, update.progress)
    if update.batch_idx is not None:
        job.batch.idx = update.batch_idx
    if update.batch_total is not None:
        job.batch.total = update.batch_total

    if update.stage in USAGE_RESET_STAGES:
        # usage summary is a short block so we reset once we move into the next stage.
        job._in_usage_summary = False

    if update.usage_section_start is not None:
        job._in_usage_summary = update.usage_section_start
        if update.usage_section_start:
            job._current_usage_model = None

    if not job._in_usage_summary:
        return

    if update.usage_model is not None:
        name = update.usage_model.strip().strip("'").strip('"')
        job._current_usage_model = name
        if name not in job.model_usage:
            job.model_usage[name] = ModelUsage(model = name)

    if job._current_usage_model is None:
        return

    usage = job.model_usage.get(job._current_usage_model)
    if usage is None:
        return

    if update.usage_input_tokens is not None:
        usage.input_tokens = update.usage_input_tokens
    if update.usage_output_tokens is not None:
        usage.output_tokens = update.usage_output_tokens
    if update.usage_total_tokens is not None:
        usage.total_tokens = update.usage_total_tokens
    if update.usage_tps is not None:
        usage.tps = update.usage_tps
    if update.usage_requests_success is not None:
        usage.requests_success = update.usage_requests_success
    if update.usage_requests_failed is not None:
        usage.requests_failed = update.usage_requests_failed
    if update.usage_requests_total is not None:
        usage.requests_total = update.usage_requests_total
    if update.usage_rpm is not None:
        usage.rpm = update.usage_rpm


def _compute_overall_progress(job: Job, column_progress: Progress) -> Progress:
    if not job.rows:
        return column_progress

    total_rows = max(1, int(job.rows))
    current_done = 0 if column_progress.done is None else int(column_progress.done)
    current_done = max(0, min(current_done, total_rows))
    total_columns = max(1, int(job.progress_columns_total or 1))

    if job.current_column:
        job._column_done[job.current_column] = current_done

    if len(job._column_done) == 0:
        done = current_done
    else:
        sum_done = sum(
            max(0, min(value, total_rows)) for value in job._column_done.values()
        )
        done = int(sum_done / total_columns)

    prev_done = int(job.progress.done or 0)
    if done < prev_done:
        done = prev_done
    if done > total_rows:
        done = total_rows
    percent = (done / total_rows) * 100 if total_rows > 0 else 100.0
    prev_percent = float(job.progress.percent or 0.0)
    if percent < prev_percent:
        percent = prev_percent

    return Progress(
        done = done,
        total = total_rows,
        percent = percent,
        eta_sec = column_progress.eta_sec,
        rate = column_progress.rate,
        ok = column_progress.ok,
        failed = column_progress.failed,
    )


def coerce_event(obj: Any) -> dict:
    """Normalize worker payload into event dict."""
    return obj if isinstance(obj, dict) else {"type": "log", "message": str(obj)}
