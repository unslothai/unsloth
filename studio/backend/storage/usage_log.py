# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Storage module for token usage events."""

import time
import structlog
from typing import Optional, List, Dict, Any

from storage.studio_db import get_connection, get_app_setting
from core.inference.api_monitor import ApiMonitorEntry

logger = structlog.get_logger(__name__)


def record_event(entry: ApiMonitorEntry) -> None:
    """Record a terminal ApiMonitorEntry as a usage event."""
    source = "api" if entry.provider_type else "local"
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO usage_events (
                id, ts, model, source, provider,
                prompt_tokens, completion_tokens, total_tokens,
                endpoint, status, session_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                int(entry.start_time * 1000) if entry.start_time else int(time.time() * 1000),
                entry.model or "unknown",
                source,
                entry.provider_type,
                entry.prompt_tokens or 0,
                entry.completion_tokens or 0,
                entry.total_tokens or 0,
                entry.endpoint,
                entry.status or "unknown",
                entry.session_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _build_where_clause(
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    model: Optional[str] = None,
    source: Optional[str] = None,
) -> tuple[str, list]:
    conditions = []
    params = []
    
    if start_ts is not None:
        conditions.append("ts >= ?")
        params.append(start_ts)
    if end_ts is not None:
        conditions.append("ts <= ?")
        params.append(end_ts)
    if model is not None:
        conditions.append("model = ?")
        params.append(model)
    if source is not None:
        conditions.append("source = ?")
        params.append(source)
        
    where = " AND ".join(conditions) if conditions else "1=1"
    return f"WHERE {where}", params


def count_events(
    *,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    model: Optional[str] = None,
    source: Optional[str] = None,
) -> int:
    """Count total usage events matching the filters."""
    where, params = _build_where_clause(start_ts, end_ts, model, source)
    conn = get_connection()
    try:
        row = conn.execute(f"SELECT COUNT(*) as c FROM usage_events {where}", params).fetchone()
        return row["c"]
    finally:
        conn.close()


def query_events(
    *,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    model: Optional[str] = None,
    source: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Query raw usage events."""
    where, params = _build_where_clause(start_ts, end_ts, model, source)
    
    query = f"SELECT * FROM usage_events {where} ORDER BY ts DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
        if offset is not None:
            query += " OFFSET ?"
            params.append(offset)
            
    conn = get_connection()
    try:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def aggregate_usage(
    *,
    granularity: str,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Aggregate usage by period, model, and source."""
    if granularity not in ("day", "week", "month", "year"):
        raise ValueError(f"Invalid granularity: {granularity}")
        
    where, params = _build_where_clause(start_ts, end_ts, model, None)
    
    time_expr = {
        "day": "strftime('%Y-%m-%d', ts / 1000, 'unixepoch')",
        "week": "strftime('%Y-%W', ts / 1000, 'unixepoch')",
        "month": "strftime('%Y-%m', ts / 1000, 'unixepoch')",
        "year": "strftime('%Y', ts / 1000, 'unixepoch')",
    }[granularity]
    
    query = f"""
        SELECT 
            {time_expr} as period,
            model,
            source,
            SUM(prompt_tokens) as prompt_tokens,
            SUM(completion_tokens) as completion_tokens,
            SUM(total_tokens) as total_tokens
        FROM usage_events
        {where}
        GROUP BY period, model, source
        ORDER BY period DESC, model ASC, source ASC
    """
    
    conn = get_connection()
    try:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def sum_tokens_in_window(
    *,
    since_ts: int,
    model: Optional[str] = None,
    source: Optional[str] = None,
) -> int:
    """Get the sum of total_tokens since a specific timestamp."""
    where, params = _build_where_clause(since_ts, None, model, source)
    conn = get_connection()
    try:
        row = conn.execute(f"SELECT SUM(total_tokens) as total FROM usage_events {where}", params).fetchone()
        return row["total"] or 0
    finally:
        conn.close()


def enforce_retention() -> None:
    """Enforce the retention policy by deleting old rows."""
    policy = get_app_setting("usage_retention_policy", {"mode": "forever", "value": None})
    if policy.get("mode") != "months" or not policy.get("value"):
        return
        
    months = policy["value"]
    cutoff_ts = int((time.time() - (months * 30 * 24 * 60 * 60)) * 1000)
    
    conn = get_connection()
    try:
        cursor = conn.execute("DELETE FROM usage_events WHERE ts < ?", (cutoff_ts,))
        deleted = cursor.rowcount
        conn.commit()
        if deleted > 0:
            logger.info("usage_retention.enforced", deleted_rows = deleted, cutoff_months = months)
    finally:
        conn.close()
