# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import io
import csv
import json
from typing import Optional, Literal
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse

from auth.authentication import get_current_subject
from storage.usage_log import aggregate_usage, query_events
from storage.studio_db import get_app_setting, upsert_app_settings
from models.usage import (
    UsageSummaryRow,
    UsageSummaryResponse,
    UsageRetentionSetting,
)

router = APIRouter()


@router.get("/summary", response_model = UsageSummaryResponse)
async def get_usage_summary(
    granularity: Literal["day", "week", "month", "year"] = Query("day"),
    start_ts: Optional[int] = Query(None),
    end_ts: Optional[int] = Query(None),
    model: Optional[str] = Query(None),
    current_subject: str = Depends(get_current_subject),
):
    rows = aggregate_usage(
        granularity = granularity,
        start_ts = start_ts,
        end_ts = end_ts,
        model = model,
    )
    return UsageSummaryResponse(
        granularity = granularity,
        rows = [UsageSummaryRow(**row) for row in rows],
    )


@router.get("/export")
async def export_usage(
    format: Literal["csv", "json"] = Query("csv"),
    start_ts: Optional[int] = Query(None),
    end_ts: Optional[int] = Query(None),
    current_subject: str = Depends(get_current_subject),
):
    def iter_csv():
        yield "id,ts,model,source,provider,prompt_tokens,completion_tokens,total_tokens,endpoint,status,session_id\n"
        offset = 0
        limit = 1000
        while True:
            events = query_events(
                start_ts = start_ts,
                end_ts = end_ts,
                limit = limit,
                offset = offset,
            )
            if not events:
                break

            output = io.StringIO()
            writer = csv.writer(output)
            for event in events:
                writer.writerow(
                    [
                        event["id"],
                        event["ts"],
                        event["model"],
                        event["source"],
                        event.get("provider", ""),
                        event["prompt_tokens"],
                        event["completion_tokens"],
                        event["total_tokens"],
                        event.get("endpoint", ""),
                        event["status"],
                        event.get("session_id", ""),
                    ]
                )
            yield output.getvalue()
            offset += limit

    def iter_json():
        offset = 0
        limit = 1000
        first = True
        yield "[\n"
        while True:
            events = query_events(
                start_ts = start_ts,
                end_ts = end_ts,
                limit = limit,
                offset = offset,
            )
            if not events:
                break

            for event in events:
                if not first:
                    yield ",\n"
                first = False
                yield json.dumps(event)
            offset += limit
        yield "\n]"

    if format == "csv":
        return StreamingResponse(
            iter_csv(),
            media_type = "text/csv",
            headers = {"Content-Disposition": 'attachment; filename="usage_export.csv"'},
        )
    else:
        return StreamingResponse(
            iter_json(),
            media_type = "application/json",
            headers = {"Content-Disposition": 'attachment; filename="usage_export.json"'},
        )


@router.get("/settings", response_model = UsageRetentionSetting)
async def get_usage_settings(current_subject: str = Depends(get_current_subject)):
    setting = get_app_setting("usage_retention_policy", {"mode": "forever", "value": None})
    return UsageRetentionSetting(**setting)


@router.put("/settings", response_model = UsageRetentionSetting)
async def update_usage_settings(
    settings: UsageRetentionSetting, current_subject: str = Depends(get_current_subject)
):
    upsert_app_settings({"usage_retention_policy": settings.model_dump()})
    # Apply immediately
    try:
        from storage.usage_log import enforce_retention
        enforce_retention()
    except Exception as exc:
        import structlog
        structlog.get_logger(__name__).warning("usage_retention.enforced_failed", error = str(exc))
    return settings
