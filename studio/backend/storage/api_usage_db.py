# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Durable, content-free receipts for authenticated external API usage."""

from __future__ import annotations

from dataclasses import dataclass

from storage.studio_db import get_connection


# Kept aligned with the API monitor's defensive upper bound. The storage layer
# validates independently because callers can invoke it directly in tests or
# future integrations.
MAX_TOKEN_COUNT = 1 << 40


@dataclass(frozen = True, slots = True)
class ApiUsageReceipt:
    """Terminal scalar usage only. Prompts, replies and credentials never enter it."""

    id: str
    subject: str
    endpoint: str
    model: str
    status: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    created_at: int
    kind: str = "request"
    via_api_key: bool = True


def _valid_token_count(value: object) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= MAX_TOKEN_COUNT
    )


def record_api_usage(receipt: ApiUsageReceipt) -> bool:
    """Insert one external request receipt, returning whether a row was added.

    The monitor id is the idempotency key, so repeated completion notification
    cannot inflate profile totals. Invalid or zero-usage receipts are ignored.
    """
    if receipt.kind != "request" or receipt.via_api_key is not True:
        return False
    if not receipt.id or not receipt.subject or not receipt.endpoint or not receipt.status:
        return False
    counts = (receipt.prompt_tokens, receipt.completion_tokens, receipt.total_tokens)
    if not all(_valid_token_count(value) for value in counts) or not any(counts):
        return False
    if (
        not isinstance(receipt.created_at, int)
        or isinstance(receipt.created_at, bool)
        or receipt.created_at <= 0
        or receipt.created_at > (1 << 63) - 1
    ):
        return False

    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO api_usage_events
                (id, subject, endpoint, model, status,
                 prompt_tokens, completion_tokens, total_tokens, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.id,
                receipt.subject,
                receipt.endpoint,
                receipt.model or "default",
                receipt.status,
                receipt.prompt_tokens,
                receipt.completion_tokens,
                receipt.total_tokens,
                receipt.created_at,
            ),
        )
        conn.commit()
        inserted = cursor.rowcount == 1
    finally:
        conn.close()

    if inserted:
        # Lazy import avoids making profile aggregation part of schema startup.
        from storage.profile_stats_db import invalidate_profile_stats_cache

        invalidate_profile_stats_cache()
    return inserted
