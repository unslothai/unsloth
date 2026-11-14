# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import HTTPException

from storage.studio_db import get_app_setting, get_connection, upsert_app_settings
from storage.token_usage_ledger import window_usage

SETTINGS_KEY = "token_limit"
OVERRIDE_KEY = "token_limit_override"

PERIODS: tuple[str, ...] = ("day", "week", "month")
MAX_LIMIT = 1 << 40
WARNING_FRACTION = 0.8

DEFAULT_SETTINGS: dict[str, Any] = {
    "enabled": False,
    "period": "day",
    "limit": 150_000,
    "timezone": "",
}

_override_lock = threading.Lock()


def _clean_settings(raw: Any) -> dict[str, Any]:
    settings = dict(DEFAULT_SETTINGS)
    if not isinstance(raw, dict):
        return settings
    if isinstance(raw.get("enabled"), bool):
        settings["enabled"] = raw["enabled"]
    if raw.get("period") in PERIODS:
        settings["period"] = raw["period"]
    limit = raw.get("limit")
    if isinstance(limit, int) and not isinstance(limit, bool) and 0 < limit <= MAX_LIMIT:
        settings["limit"] = limit
    if isinstance(raw.get("timezone"), str):
        settings["timezone"] = raw["timezone"][:64]
    return settings


def load_settings() -> dict[str, Any]:
    return _clean_settings(get_app_setting(SETTINGS_KEY, reuse_keeper = True))


def _zone(name: str):
    if name:
        try:
            return ZoneInfo(name)
        except (ValueError, ZoneInfoNotFoundError, OSError):
            pass
    return timezone.utc


def period_window(
    period: str,
    tz_name: str,
    now_ms: Optional[int] = None,
) -> tuple[int, int]:
    zone = _zone(tz_name)
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    now = datetime.fromtimestamp(now_ms / 1000, zone)
    day = now.replace(hour = 0, minute = 0, second = 0, microsecond = 0, fold = 0)
    if period == "week":
        start = day - timedelta(days = day.weekday())
        end = start + timedelta(days = 7)
    elif period == "month":
        start = day.replace(day = 1)
        end = start.replace(year = start.year + start.month // 12, month = start.month % 12 + 1)
    else:
        start = day
        end = start + timedelta(days = 1)
    # Re-attach the zone so a DST day is 23 or 25 hours, not 24 from the wrong offset.
    start = start.replace(tzinfo = None).replace(tzinfo = zone)
    end = end.replace(tzinfo = None).replace(tzinfo = zone)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def _used(start_ms: int) -> int:
    conn = get_connection()
    try:
        return window_usage(conn, start_ms)
    finally:
        conn.close()


def _override_state(start_ms: int) -> Optional[str]:
    entry = get_app_setting(OVERRIDE_KEY, {})
    if isinstance(entry, dict) and entry.get("periodStart") == start_ms:
        state = entry.get("state")
        return state if state in ("granted", "used") else None
    return None


def _set_override(start_ms: int, state: str) -> None:
    upsert_app_settings({OVERRIDE_KEY: {"periodStart": start_ms, "state": state}}, read_back = False)


def status() -> dict[str, Any]:
    settings = load_settings()
    start_ms, end_ms = period_window(settings["period"], settings["timezone"])
    used = _used(start_ms) if settings["enabled"] else 0
    return {
        "settings": settings,
        "periodStart": start_ms,
        "resetsAt": end_ms,
        "used": used,
        "reached": settings["enabled"] and used >= settings["limit"],
        "warning": settings["enabled"] and used >= settings["limit"] * WARNING_FRACTION,
        "override": _override_state(start_ms) if settings["enabled"] else None,
    }


def _require_usage_accounting() -> None:
    from core.inference.api_monitor import api_monitor
    if not api_monitor.enabled:
        raise HTTPException(
            status_code = 409,
            detail = "Token allowances require API usage monitoring. Unset UNSLOTH_STUDIO_DISABLE_API_MONITOR and restart, or disable the allowance.",
        )


def update_settings(patch: dict[str, Any]) -> dict[str, Any]:
    with _override_lock:
        merged = _clean_settings({**load_settings(), **patch})
        if merged["enabled"]:
            _require_usage_accounting()
        upsert_app_settings({SETTINGS_KEY: merged}, read_back = False)
    return status()


def grant_override() -> dict[str, Any]:
    with _override_lock:
        settings = load_settings()
        if not settings["enabled"]:
            raise HTTPException(status_code = 409, detail = "The token limit is off.")
        _require_usage_accounting()
        start_ms, _end_ms = period_window(settings["period"], settings["timezone"])
        if _used(start_ms) < settings["limit"]:
            raise HTTPException(status_code = 409, detail = "This limit has not been reached.")
        if _override_state(start_ms) is not None:
            raise HTTPException(
                status_code = 409,
                detail = "The one-time override for this period was already used.",
            )
        _set_override(start_ms, "granted")
    return status()


def restore_override(start_ms: int) -> None:
    """Gives back an override whose request was refused before it generated anything."""
    with _override_lock:
        if _override_state(start_ms) == "used":
            _set_override(start_ms, "granted")


def enforce() -> Optional[int]:
    """Returns the period start when this request spent the one-time override."""
    settings = load_settings()
    if not settings["enabled"]:
        return None
    _require_usage_accounting()
    start_ms, end_ms = period_window(settings["period"], settings["timezone"])
    used = _used(start_ms)
    if used < settings["limit"]:
        return None
    with _override_lock:
        if _override_state(start_ms) == "granted":
            _set_override(start_ms, "used")
            return start_ms
    period = {"day": "daily", "week": "weekly", "month": "monthly"}[settings["period"]]
    resets = datetime.fromtimestamp(end_ms / 1000, _zone(settings["timezone"]))
    raise HTTPException(
        status_code = 429,
        detail = (
            f"You have reached your {period} token limit ({used:,} of {settings['limit']:,} "
            f"tokens). It resets {resets:%Y-%m-%d %H:%M}."
        ),
        headers = {
            "X-Unsloth-Token-Limit": "reached",
            "Retry-After": str(max(1, (end_ms - int(time.time() * 1000)) // 1000)),
        },
    )


async def refund_if_unused(claim: dict, *, failed: bool) -> None:
    # A 503 no-model refusal generated nothing, but a 502 can follow output, and a success through
    # a path the monitor does not see must never refund. A monitored cancel / fail counts.
    if (failed or claim.get("failed")) and claim.get("spent") and not claim.get("generated"):
        await asyncio.to_thread(restore_override, claim["spent"])


class OverrideRefundMiddleware:
    """Restores a spent override when its request generated nothing (e.g. no model loaded)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        from core.inference.api_monitor import override_claim_scope
        from core.inference.llama_keepwarm import _RESPONSE_FAILED_SCOPE_KEY

        status = None

        async def send_status(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        with override_claim_scope({"spent": None, "generated": False}) as claim:
            try:
                await self.app(scope, receive, send_status)
            finally:
                # A 200 stream can still fail in band (flagged for the keep-warm middleware).
                await refund_if_unused(
                    claim,
                    failed = status is None
                    or status >= 400
                    or bool(scope.get(_RESPONSE_FAILED_SCOPE_KEY)),
                )
