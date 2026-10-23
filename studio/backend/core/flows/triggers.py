# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Schedule and webhook triggers. Only the owner account's flows fire, only while Studio runs, and only once the owner
approved triggers for the flow. A time missed while Studio was closed is not made up; a fire while the flow's last
triggered run is still going is skipped."""

from __future__ import annotations

import asyncio
import re
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Optional

from core.flows import capabilities
from loggers import get_logger
from storage import flows_db
from utils.account_context import OWNER, arun_as, current_account

logger = get_logger(__name__)

KINDS = {"trigger.schedule": "schedule", "trigger.webhook": "webhook"}
TICK_S = 15.0
MIN_EVERY_MIN = 1
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{20,80}$")


def new_token() -> str:
    return secrets.token_urlsafe(24)


# Cron (minute hour day-of-month month day-of-week; *, lists, ranges, steps)

_CRON_RANGES = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))


def _cron_field(text: str, low: int, high: int) -> set[int]:
    found: set[int] = set()
    for part in text.split(","):
        step = 1
        if "/" in part:
            part, step_text = part.split("/", 1)
            step = int(step_text)
            if step < 1:
                raise ValueError("a step must be at least 1")
        if part == "*":
            start, end = low, high
        elif not part:
            # "0, 9 * * *" (a stray comma) must not quietly mean every minute.
            raise ValueError("a list in a cron field has an empty item")
        elif "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
        else:
            start = end = int(part)
        if start < low or end > high or start > end:
            raise ValueError(f"{part} is outside {low}-{high}")
        found.update(range(start, end + 1, step))
    return found


def parse_cron(expr: str) -> list[set[int]]:
    fields = expr.split()
    if len(fields) != 5:
        raise ValueError("a cron schedule has five fields: minute hour day month weekday")
    sets = [_cron_field(f, lo, hi) for f, (lo, hi) in zip(fields, _CRON_RANGES)]
    # Sunday is 0, and 7 as most cron tools also accept.
    sets[4] = {0 if d == 7 else d for d in sets[4]}
    return sets


def cron_matches(sets: list[set[int]], when: datetime) -> bool:
    weekday = (when.weekday() + 1) % 7  # Monday=1 ... Sunday=0
    day_ok, weekday_ok = when.day in sets[2], weekday in sets[4]
    # As in cron: with both day fields restricted, either one matching is enough.
    if len(sets[2]) < 31 and len(sets[4]) < 7:
        days = day_ok or weekday_ok
    else:
        days = day_ok and weekday_ok
    return when.minute in sets[0] and when.hour in sets[1] and when.month in sets[3] and days


# Index of a flow's triggers, kept in step with the saved document


def wanted(doc: dict) -> list[dict]:
    found = []
    for node in doc.get("nodes") or []:
        kind = KINDS.get(node.get("type"))
        if kind is None:
            continue
        params = node.get("params") or {}
        config: dict[str, Any] = {
            "enabled": params.get("enabled") is True and not node.get("disabled")
        }
        if kind == "schedule":
            try:
                every = int(params.get("every_minutes") or 60)
            except (TypeError, ValueError):
                every = 60
            config["every_minutes"] = max(MIN_EVERY_MIN, min(every, 10080))
            cron = str(params.get("cron") or "").strip()
            if cron:
                try:
                    parse_cron(cron)
                    config["cron"] = cron
                except ValueError as exc:
                    config["cron_error"] = str(exc)
        found.append({"node_id": node["id"], "kind": kind, "config": config})
    return found


def sync(flow_id: str, doc: dict) -> None:
    """Only the owner account's triggers ever fire, so only its index is kept, and only the owner's own hand updates
    it: an API key or MCP save never switches a trigger on (``fire`` also refuses one the saved flow has off)."""
    account = current_account()
    if not account.is_owner or account.via_api_key:
        return
    flows_db.replace_triggers(flow_id, wanted(doc), new_token)


def due(row: dict, now: datetime, since_ms: int) -> bool:
    """Whether a schedule fires at ``now``; ``since_ms`` is the later of its last fire and Studio's start, so time
    spent closed never counts toward a catch-up fire."""
    config = row["config"]
    if not config.get("enabled") or config.get("cron_error"):
        return False
    now_ms = int(now.timestamp() * 1000)
    if config.get("cron"):
        if not cron_matches(parse_cron(config["cron"]), now):
            return False
        # Once per matching minute.
        return now_ms // 60000 > since_ms // 60000
    return now_ms - since_ms >= int(config.get("every_minutes") or 60) * 60000


# Firing


def _busy(flow_id: str) -> bool:
    from core.flows.engine import get_engine
    for rs in get_engine().runs.values():
        if (
            rs.flow_id == flow_id
            and rs.origin.get("kind") == "trigger"
            and rs.status in flows_db.ACTIVE_RUN_STATUSES
        ):
            return True
    return False


async def fire(
    app: Any,
    flow_id: str,
    node_id: str,
    payload: dict,
    at_ms: Optional[int] = None,
    kind: Optional[str] = None,
) -> Optional[str]:
    """Start a triggered run as the owner; returns its id, or None when it was skipped (logged)."""
    from core.flows.engine import get_engine

    record = flows_db.get_flow(flow_id)
    if record is None:
        return None
    # The saved document is the truth: a trigger switched off or removed there never fires, even if the index lags.
    if not any(
        w["node_id"] == node_id
        and w["config"].get("enabled")
        and (kind is None or w["kind"] == kind)
        for w in wanted(record["doc"])
    ):
        logger.info("flows: trigger %s/%s skipped: it is off in the saved flow", flow_id, node_id)
        return None
    if not capabilities.is_approved(flows_db.get_capabilities(flow_id), "trigger", "trigger"):
        logger.info(
            "flows: trigger %s/%s skipped: triggers are not approved for this flow",
            flow_id,
            node_id,
        )
        return None
    if _busy(flow_id):
        logger.info(
            "flows: trigger %s/%s skipped: its last triggered run is still going", flow_id, node_id
        )
        # A skipped scheduled time counts as taken, so it is not started late once the last run ends.
        if at_ms is not None:
            flows_db.mark_trigger_fired(flow_id, node_id, at_ms)
        return None
    rs = await get_engine().start_run(
        app = app,
        flow_id = flow_id,
        doc = record["doc"],
        options = {"params": {"__trigger": {node_id: payload}}},
        origin = "trigger",
    )
    flows_db.mark_trigger_fired(flow_id, node_id, at_ms or flows_db.now_ms())
    return rs.id


class Scheduler:
    def __init__(self) -> None:
        self.task: Optional[asyncio.Task] = None
        self.started_ms = flows_db.now_ms()

    async def tick(
        self,
        app: Any,
        now: Optional[datetime] = None,
    ) -> list[str]:
        now = now or datetime.now(timezone.utc).astimezone()
        started = []
        for row in flows_db.list_triggers():
            if row["kind"] != "schedule":
                continue
            since = max(row["last_fired_at"] or 0, self.started_ms)
            try:
                if not due(row, now, since):
                    continue
                payload = {"fired_at": now.isoformat(timespec = "seconds")}
                run_id = await fire(
                    app,
                    row["flow_id"],
                    row["node_id"],
                    payload,
                    int(now.timestamp() * 1000),
                    kind = "schedule",
                )
            except Exception:
                logger.warning(
                    "flows: schedule %s/%s failed to start",
                    row["flow_id"],
                    row["node_id"],
                    exc_info = True,
                )
                continue
            if run_id:
                started.append(run_id)
        return started

    async def _loop(self, app: Any) -> None:
        while True:
            try:
                await arun_as(OWNER, self.tick(app))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("flows: trigger scheduler tick failed", exc_info = True)
            await asyncio.sleep(TICK_S)

    def start(self, app: Any) -> None:
        self.started_ms = flows_db.now_ms()
        self.task = asyncio.create_task(self._loop(app), name = "flow-trigger-scheduler")

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None


SCHEDULER = Scheduler()


# Webhook rate limit: per token, in memory.
_HITS: dict[str, list[float]] = {}
WEBHOOK_PER_MINUTE = 30


def rate_limited(token: str) -> bool:
    now = time.monotonic()
    hits = [t for t in _HITS.get(token, []) if now - t < 60]
    if len(hits) >= WEBHOOK_PER_MINUTE:
        _HITS[token] = hits
        return True
    hits.append(now)
    _HITS[token] = hits
    if len(_HITS) > 10000:
        # Only buckets with no call in the last minute go.
        for key in [k for k, v in _HITS.items() if not v or now - v[-1] >= 60]:
            del _HITS[key]
    return False
