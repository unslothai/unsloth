# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""When a task runs next: wall clock in its IANA zone; a skipped time runs past the gap, a repeated one once."""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime, time as wall_time, timedelta, timezone
from typing import Annotated, Any, Literal, Optional, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, TypeAdapter, field_validator

_CLOCK = re.compile(r"^([01][0-9]|2[0-3]):([0-5][0-9])$")
_LOCAL_MINUTE_FORMAT = "%Y-%m-%dT%H:%M"
_MAX_TIMEZONE_LENGTH = 64


class InvalidSchedule(ValueError):
    """A schedule or time zone that cannot produce a run time."""


def _check_clock(value: str) -> str:
    if not _CLOCK.match(value):
        raise ValueError("time must be HH:MM on a 24-hour clock")
    return value


def _check_local_minute(value: str) -> str:
    try:
        datetime.strptime(value, _LOCAL_MINUTE_FORMAT)
    except ValueError:
        raise ValueError("at must be YYYY-MM-DDTHH:MM") from None
    return value


Clock = Annotated[str, AfterValidator(_check_clock)]
LocalMinute = Annotated[str, AfterValidator(_check_local_minute)]


class OnceSchedule(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    kind: Literal["once"]
    at: LocalMinute


class DailySchedule(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    kind: Literal["daily"]
    time: Clock


class WeeklySchedule(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    kind: Literal["weekly"]
    time: Clock
    # ISO weekdays: 1 is Monday, 7 is Sunday.
    days: list[int] = Field(min_length = 1, max_length = 7)

    @field_validator("days")
    @classmethod
    def _iso_weekdays(cls, value: list[int]) -> list[int]:
        if any(day < 1 or day > 7 for day in value):
            raise ValueError("days must be ISO weekdays, 1 (Monday) to 7 (Sunday)")
        return sorted(set(value))


class MonthlySchedule(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    kind: Literal["monthly"]
    time: Clock
    # A month without this day runs on its last day.
    day: int = Field(ge = 1, le = 31)


Schedule = Annotated[
    Union[OnceSchedule, DailySchedule, WeeklySchedule, MonthlySchedule],
    Field(discriminator = "kind"),
]
_SCHEDULE = TypeAdapter(Schedule)


def parse_schedule(value: Any) -> Schedule:
    """A schedule from its JSON form; raises pydantic's ValidationError."""
    return _SCHEDULE.validate_python(value)


def resolve_timezone(name: str) -> ZoneInfo:
    if not name or len(name) > _MAX_TIMEZONE_LENGTH:
        raise InvalidSchedule("Unknown time zone.")
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        raise InvalidSchedule(f"Unknown time zone: {name}") from None


def _utc(day: date, clock: wall_time, zone: ZoneInfo) -> datetime:
    # fold=0 (PEP 495) is what moves a skipped time past the gap and picks the first of a repeated one.
    return datetime.combine(day, clock, tzinfo = zone).astimezone(timezone.utc)


def _next_after(schedule: Schedule, zone: ZoneInfo, after: datetime) -> Optional[datetime]:
    if isinstance(schedule, OnceSchedule):
        moment = datetime.strptime(schedule.at, _LOCAL_MINUTE_FORMAT)
        candidate = _utc(moment.date(), moment.time(), zone)
        return candidate if candidate > after else None
    hour, minute = (int(part) for part in schedule.time.split(":"))
    clock = wall_time(hour, minute)
    today = after.astimezone(zone).date()
    if isinstance(schedule, MonthlySchedule):
        year, month = today.year, today.month
        for _ in range(2):
            last_day = calendar.monthrange(year, month)[1]
            candidate = _utc(date(year, month, min(schedule.day, last_day)), clock, zone)
            if candidate > after:
                return candidate
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        return None
    days = set(schedule.days) if isinstance(schedule, WeeklySchedule) else None
    # Eight days reach the same weekday a week on, whichever single day is set.
    for offset in range(8):
        day = today + timedelta(days = offset)
        if days is not None and day.isoweekday() not in days:
            continue
        candidate = _utc(day, clock, zone)
        if candidate > after:
            return candidate
    return None


def next_run_after(schedule: Schedule, timezone_name: str, after_ms: int) -> Optional[int]:
    """The first run strictly after ``after_ms`` (epoch ms), or None once a schedule has none left."""
    zone = resolve_timezone(timezone_name)
    after = datetime.fromtimestamp(after_ms / 1000, tz = timezone.utc)
    candidate = _next_after(schedule, zone, after)
    return None if candidate is None else int(candidate.timestamp() * 1000)
