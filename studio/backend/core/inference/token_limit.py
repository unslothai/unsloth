# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import HTTPException

from storage.studio_db import get_app_setting, get_connection, upsert_app_settings
from storage.token_usage_ledger import window_usage

SETTINGS_KEY = "token_limit"
OVERRIDES_KEY = "token_limit_overrides"

PERIODS: tuple[str, ...] = ("day", "week", "month")
MODES: tuple[str, ...] = ("combined", "split")
METHODS: tuple[str, ...] = ("total", "incremental")
MAX_LIMIT = 1 << 40
WARNING_FRACTION = 0.8

_PASSWORD_ITERATIONS = 200_000
_PASSWORD_SCHEME = "pbkdf2_sha256"

DEFAULT_SETTINGS: dict[str, Any] = {
    "enabled": False,
    "period": "day",
    "mode": "combined",
    "method": "total",
    "limit": 150_000,
    "localLimit": 150_000,
    "apiLimit": 150_000,
    "timezone": "",
    "passwordHash": None,
}

_override_lock = threading.Lock()


class TokenLimitPasswordError(Exception):
    """The limiter is password protected and the password given was missing or wrong."""


def hash_password(password: str, *, salt: Optional[bytes] = None) -> str:
    salt = salt if salt is not None else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PASSWORD_ITERATIONS)
    return f"{_PASSWORD_SCHEME}${_PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt_hex, digest_hex = stored.split("$")
        if scheme != _PASSWORD_SCHEME:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), digest_hex)


def _clean_settings(raw: Any) -> dict[str, Any]:
    settings = dict(DEFAULT_SETTINGS)
    if not isinstance(raw, dict):
        return settings
    if isinstance(raw.get("enabled"), bool):
        settings["enabled"] = raw["enabled"]
    for key, allowed in (("period", PERIODS), ("mode", MODES), ("method", METHODS)):
        if raw.get(key) in allowed:
            settings[key] = raw[key]
    for key in ("limit", "localLimit", "apiLimit"):
        value = raw.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and 0 < value <= MAX_LIMIT:
            settings[key] = value
    if isinstance(raw.get("timezone"), str):
        settings["timezone"] = raw["timezone"][:64]
    if isinstance(raw.get("passwordHash"), str) and raw["passwordHash"]:
        settings["passwordHash"] = raw["passwordHash"]
    return settings


def load_settings() -> dict[str, Any]:
    return _clean_settings(get_app_setting(SETTINGS_KEY))


def public_settings(settings: dict[str, Any]) -> dict[str, Any]:
    visible = {key: value for key, value in settings.items() if key != "passwordHash"}
    visible["passwordProtected"] = bool(settings.get("passwordHash"))
    return visible


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
        if start.month == 12:
            end = start.replace(year = start.year + 1, month = 1)
        else:
            end = start.replace(month = start.month + 1)
    else:
        start = day
        end = start + timedelta(days = 1)
    start = start.replace(tzinfo = None).replace(tzinfo = zone)
    end = end.replace(tzinfo = None).replace(tzinfo = zone)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def _pools(settings: dict[str, Any]) -> dict[str, int]:
    if settings["mode"] == "split":
        return {"local": settings["localLimit"], "api": settings["apiLimit"]}
    return {"combined": settings["limit"]}


def request_pool(settings: dict[str, Any], *, api: bool) -> str:
    if settings["mode"] != "split":
        return "combined"
    return "api" if api else "local"


@dataclass(frozen = True)
class PoolUsage:
    pool: str
    used: int
    limit: int

    @property
    def reached(self) -> bool:
        return self.used >= self.limit

    @property
    def warning(self) -> bool:
        return self.used >= self.limit * WARNING_FRACTION


def _pool_usage(settings: dict[str, Any], start_ms: int) -> dict[str, PoolUsage]:
    conn = get_connection()
    try:
        counts = window_usage(conn, start_ms, settings["method"])
    finally:
        conn.close()
    local, external, api = (counts.get(source, 0) for source in ("local", "external", "api"))
    used = {
        "combined": local + external + api,
        "local": local,
        "api": external + api,
    }
    return {
        pool: PoolUsage(pool = pool, used = used[pool], limit = limit)
        for pool, limit in _pools(settings).items()
    }


def _overrides() -> dict[str, Any]:
    value = get_app_setting(OVERRIDES_KEY, {})
    return value if isinstance(value, dict) else {}


def _override_state(overrides: dict[str, Any], pool: str, start_ms: int) -> Optional[str]:
    entry = overrides.get(pool)
    if isinstance(entry, dict) and entry.get("periodStart") == start_ms:
        state = entry.get("state")
        return state if state in ("granted", "used") else None
    return None


def status() -> dict[str, Any]:
    settings = load_settings()
    start_ms, end_ms = period_window(settings["period"], settings["timezone"])
    pools = []
    if settings["enabled"]:
        overrides = _overrides()
        for usage in _pool_usage(settings, start_ms).values():
            override = _override_state(overrides, usage.pool, start_ms)
            pools.append(
                {
                    "pool": usage.pool,
                    "used": usage.used,
                    "limit": usage.limit,
                    "reached": usage.reached,
                    "warning": usage.warning,
                    "override": override,
                }
            )
    return {
        "settings": public_settings(settings),
        "periodStart": start_ms,
        "resetsAt": end_ms,
        "pools": pools,
    }


def update_settings(
    patch: dict[str, Any], *, password: Optional[str], new_password: Optional[str]
) -> dict[str, Any]:
    with _override_lock:
        current = load_settings()
        stored_hash = current.get("passwordHash")
        if stored_hash and not (password and verify_password(password, stored_hash)):
            raise TokenLimitPasswordError()
        merged = _clean_settings({**current, **patch, "passwordHash": stored_hash})
        if new_password is not None:
            merged["passwordHash"] = hash_password(new_password) if new_password else None
        upsert_app_settings({SETTINGS_KEY: merged}, read_back = False)
    return status()


def grant_override(pool: str) -> dict[str, Any]:
    settings = load_settings()
    if not settings["enabled"]:
        raise HTTPException(status_code = 409, detail = "The token limit is off.")
    start_ms, _end_ms = period_window(settings["period"], settings["timezone"])
    usage = _pool_usage(settings, start_ms).get(pool)
    if usage is None:
        raise HTTPException(status_code = 400, detail = f"Unknown token limit pool: {pool}")
    if not usage.reached:
        raise HTTPException(status_code = 409, detail = "This limit has not been reached.")
    with _override_lock:
        overrides = _overrides()
        if _override_state(overrides, pool, start_ms) is not None:
            raise HTTPException(
                status_code = 409,
                detail = "The one-time override for this period was already used.",
            )
        overrides[pool] = {"periodStart": start_ms, "state": "granted"}
        upsert_app_settings({OVERRIDES_KEY: overrides}, read_back = False)
    return status()


def _period_label(period: str) -> str:
    return {"day": "daily", "week": "weekly", "month": "monthly"}.get(period, period)


def enforce(*, api: bool) -> None:
    settings = load_settings()
    if not settings["enabled"]:
        return
    start_ms, end_ms = period_window(settings["period"], settings["timezone"])
    pool = request_pool(settings, api = api)
    usage = _pool_usage(settings, start_ms)[pool]
    if not usage.reached:
        return
    with _override_lock:
        overrides = _overrides()
        state = _override_state(overrides, pool, start_ms)
        if state == "granted":
            overrides[pool] = {"periodStart": start_ms, "state": "used"}
            upsert_app_settings({OVERRIDES_KEY: overrides}, read_back = False)
            return
    scope = {"local": "local model ", "api": "API "}.get(pool, "")
    retry_after = max(1, (end_ms - int(time.time() * 1000)) // 1000)
    raise HTTPException(
        status_code = 429,
        detail = (
            f"You have reached your {_period_label(settings['period'])} {scope}token limit "
            f"({usage.used:,} of {usage.limit:,} tokens). It resets "
            f"{datetime.fromtimestamp(end_ms / 1000, _zone(settings['timezone'])):%Y-%m-%d %H:%M}."
        ),
        headers = {"X-Unsloth-Token-Limit": pool, "Retry-After": str(retry_after)},
    )
