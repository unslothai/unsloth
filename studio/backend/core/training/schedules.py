# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""When a scheduled training start runs, and the request saved for it."""

from __future__ import annotations

import re
from typing import Any

from core.scheduled_tasks.schedule import OnceSchedule, next_run_after
from models.training import TrainingStartRequest

# A start picked for the current minute is due now; only an earlier minute counts as the past.
_CURRENT_MINUTE_GRACE_MS = 60_000
_LOCAL_MINUTE = re.compile(r"^\d{4}-\d{2}-\d{2}T([01]\d|2[0-3]):[0-5]\d$")


class InvalidStartTime(ValueError):
    """A start time or time zone that cannot produce a run time."""


def resolve_start_time(start_at: str, timezone: str, now_ms: int) -> int:
    """The epoch ms a local wall-clock start runs at; raises InvalidStartTime."""
    if not _LOCAL_MINUTE.match(start_at):
        raise InvalidStartTime("Pick a valid start time.")
    try:
        run_at = next_run_after(
            OnceSchedule(kind = "once", at = start_at),
            timezone,
            now_ms - _CURRENT_MINUTE_GRACE_MS,
        )
    except ValueError as exc:
        raise InvalidStartTime(str(exc)) from None
    if run_at is None:
        raise InvalidStartTime("Pick a start time in the future.")
    return run_at


def request_payload(request: TrainingStartRequest) -> dict[str, Any]:
    """The JSON stored encrypted for the due-time start."""
    return request.model_dump(mode = "json")


def request_from_payload(payload: dict[str, Any]) -> TrainingStartRequest:
    """Rebuild a stored start request; raises pydantic's ValidationError when it no longer validates."""
    return TrainingStartRequest.model_validate(payload)
