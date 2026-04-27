# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import re
import time
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
    STAGE_SOURCE,
    USAGE_RESET_STAGES,
)
from .types import Job, ModelUsage, Progress, SourceProgress


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
    source_progress: SourceProgress | None = None


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
_RE_GITHUB_PAGE = re.compile(
    r"^\[(?P<repo>[^\]\s]+/[^\]\s]+)\]\s+"
    r"(?P<resource>issues|PRs|commits)\s+page\s+(?P<page>\d+)\s+"
    r"\(\+(?P<items>\d+)\).*?\bremaining=(?P<remaining>\d+)",
    re.IGNORECASE,
)
_RE_GITHUB_RATE_LIMIT = re.compile(
    r"Rate limit hit\. Sleeping (?P<seconds>\d+)s until reset\.",
    re.IGNORECASE,
)
_RE_GITHUB_SECONDARY_RATE_LIMIT = re.compile(
    r"Secondary rate limit(?: on REST)?\. Sleep (?P<seconds>\d+)s\.",
    re.IGNORECASE,
)
_RE_GITHUB_REST_RATE_LIMIT = re.compile(
    r"REST 403/429, sleep (?P<seconds>\d+)",
    re.IGNORECASE,
)
_RE_GITHUB_TRANSIENT = re.compile(
    r"^(?P<api>GraphQL|REST) (?P<code>\d{3}) transient, retrying",
    re.IGNORECASE,
)
_RE_GITHUB_NETWORK_RETRY = re.compile(
    r"^(?P<api>GraphQL|REST) network error: .* Retry\.",
    re.IGNORECASE,
)
_RE_GITHUB_TRIAL_LIMIT = re.compile(
    r"Trial limit reached for (?P<resource>issues|PRs|commits) \((?P<items>\d+)\)",
    re.IGNORECASE,
)
_RE_GITHUB_COMPLETE = re.compile(
    r"Scraper complete\. GraphQL calls=\d+ REST calls=\d+",
    re.IGNORECASE,
)


def parse_log_message(msg: str) -> ParsedUpdate | None:
    m = _RE_GITHUB_PAGE.search(msg)
    if m:
        resource_raw = m.group("resource")
        resource = "pulls" if resource_raw.lower() == "prs" else resource_raw.lower()
        repo = m.group("repo")
        page = int(m.group("page"))
        page_items = int(m.group("items"))
        return ParsedUpdate(
            stage = STAGE_SOURCE,
            source_progress = SourceProgress(
                source = "github",
                status = "fetching",
                repo = repo,
                resource = resource,
                page = page,
                page_items = page_items,
                rate_remaining = int(m.group("remaining")),
                message = (
                    f"Scraping GitHub source: {repo} "
                    f"{resource} page {page} (+{page_items})"
                ),
            ),
        )

    m = _RE_GITHUB_RATE_LIMIT.search(msg)
    if m:
        seconds = int(m.group("seconds"))
        return ParsedUpdate(
            stage = STAGE_SOURCE,
            source_progress = SourceProgress(
                source = "github",
                status = "rate_limited",
                retry_after_sec = seconds,
                message = (
                    "Waiting for GitHub rate limit. "
                    "Studio will resume automatically."
                ),
            ),
        )

    m = _RE_GITHUB_SECONDARY_RATE_LIMIT.search(msg)
    if m:
        seconds = int(m.group("seconds"))
        return ParsedUpdate(
            stage = STAGE_SOURCE,
            source_progress = SourceProgress(
                source = "github",
                status = "rate_limited",
                retry_after_sec = seconds,
                message = (
                    "Waiting for GitHub secondary rate limit. "
                    "Studio will resume automatically."
                ),
            ),
        )

    m = _RE_GITHUB_REST_RATE_LIMIT.search(msg)
    if m:
        seconds = int(m.group("seconds"))
        return ParsedUpdate(
            stage = STAGE_SOURCE,
            source_progress = SourceProgress(
                source = "github",
                status = "rate_limited",
                retry_after_sec = seconds,
                message = (
                    "Waiting for GitHub rate limit. "
                    "Studio will resume automatically."
                ),
            ),
        )

    m = _RE_GITHUB_TRIAL_LIMIT.search(msg)
    if m:
        resource_raw = m.group("resource")
        resource = "pulls" if resource_raw.lower() == "prs" else resource_raw.lower()
        items = int(m.group("items"))
        return ParsedUpdate(
            stage = STAGE_SOURCE,
            source_progress = SourceProgress(
                source = "github",
                status = "fetching",
                resource = resource,
                message = f"GitHub {resource} trial limit reached ({items}).",
            ),
        )

    m = _RE_GITHUB_TRANSIENT.search(msg)
    if m:
        api = m.group("api")
        code = m.group("code")
        return ParsedUpdate(
            stage = STAGE_SOURCE,
            source_progress = SourceProgress(
                source = "github",
                status = "retrying",
                message = f"GitHub {api} returned {code}; retrying automatically.",
            ),
        )

    m = _RE_GITHUB_NETWORK_RETRY.search(msg)
    if m:
        api = m.group("api")
        return ParsedUpdate(
            stage = STAGE_SOURCE,
            source_progress = SourceProgress(
                source = "github",
                status = "retrying",
                message = f"GitHub {api} request failed; retrying automatically.",
            ),
        )

    if _RE_GITHUB_COMPLETE.search(msg):
        return ParsedUpdate(
            stage = STAGE_SOURCE,
            source_progress = SourceProgress(
                source = "github",
                status = "completed",
                message = "GitHub source scrape complete.",
            ),
        )

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
    if update.source_progress is not None:
        _apply_source_progress(job, update.source_progress)

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


def _apply_source_progress(job: Job, progress: SourceProgress) -> None:
    previous = job.source_progress
    now = time.time()

    page_items = progress.page_items
    if progress.repo and progress.resource and progress.page is not None:
        page_key = f"{progress.repo}:{progress.resource}:{progress.page}"
        count_key = f"{progress.repo}:{progress.resource}"
        if page_key not in job._source_seen_pages:
            job._source_seen_pages.add(page_key)
            job._source_counts[count_key] = int(
                job._source_counts.get(count_key, 0)
            ) + int(page_items or 0)

    fetched_items = sum(job._source_counts.values())
    if fetched_items <= 0:
        fetched_items = progress.fetched_items or (
            previous.fetched_items if previous else None
        )

    estimated_total = (
        progress.estimated_total
        or job.source_progress_estimated_total
        or (previous.estimated_total if previous else None)
    )
    percent: float | None = progress.percent
    if percent is None and estimated_total and fetched_items is not None:
        raw_percent = (float(fetched_items) / float(max(1, estimated_total))) * 100.0
        percent = 100.0 if progress.status == "completed" else min(99.0, raw_percent)
    if percent is None and previous is not None:
        percent = previous.percent

    job.source_progress = SourceProgress(
        source = "github",
        status = progress.status or (previous.status if previous else None),
        repo = progress.repo or (previous.repo if previous else None),
        resource = progress.resource or (previous.resource if previous else None),
        page = (
            progress.page
            if progress.page is not None
            else (previous.page if previous else None)
        ),
        page_items = (
            page_items
            if page_items is not None
            else (previous.page_items if previous else None)
        ),
        fetched_items = fetched_items,
        estimated_total = estimated_total,
        percent = percent,
        rate_remaining = (
            progress.rate_remaining
            if progress.rate_remaining is not None
            else (previous.rate_remaining if previous else None)
        ),
        retry_after_sec = progress.retry_after_sec,
        message = progress.message or (previous.message if previous else None),
        updated_at = now,
    )


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
