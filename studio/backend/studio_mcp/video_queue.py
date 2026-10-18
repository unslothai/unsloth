# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Videos queued over MCP. Unsloth Studio makes one video at a time and refuses a second while one runs, so an agent submitting a shot list would have to retry each clip itself. generate_video instead queues a refused clip here, in order, and a background task per account submits each one once the video before it ends; get_job and cancel take the queue id.

The queue lives only as long as the Unsloth Studio MCP server process: a restart drops every queued video that has not started. It is per account, shared by that account's keys, and an id from another account reads as unknown."""

from __future__ import annotations

import asyncio
import contextlib
import secrets
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Optional, Union
from urllib.parse import quote

import httpx
from fastmcp.exceptions import ToolError

from studio_mcp.caller import Caller
from studio_mcp.errors import raise_for_route
from studio_mcp.forward import forward, parse_json
from studio_mcp.outputs import JobStatus, VideoJobRef
from studio_mcp.tools import as_dict, number, opt_text, try_json

# core/inference/video_families.py VIDEO_GENERATION_BUSY_MSG; a test pins the two together.
BUSY_MESSAGE = "A video generation is already in progress."
PREFIX = "vq_"
POLL_S = 2.0
RETRY_S = 5.0
MAX_MISSES = 15
MAX_RETRY_AFTER_S = 120.0
CAPACITY = 64
MAX_WAITING = 32
RUNNING = ("queued", "in_progress")
PROGRESS = "/api/inference/video/generate-progress"


@dataclass
class QueuedVideo:
    id: str
    account_id: str
    caller: Caller = field(repr = False)
    body: dict = field(repr = False)
    hints: Optional[dict] = field(default = None, repr = False)
    # queued, submitted, failed or cancelled.
    status: str = "queued"
    video_id: Optional[str] = None
    error: Optional[str] = None
    # True while its POST is in flight, when it can no longer be withdrawn.
    posting: bool = False
    enqueued_at: float = field(default_factory = time.monotonic)


_items: "OrderedDict[str, QueuedVideo]" = OrderedDict()
_waiting: dict[str, deque] = {}
# account -> (video id, monotonic submission time) of the video this account's queue waits on.
_running: dict[str, tuple[str, float]] = {}
_admission: dict[str, asyncio.Lock] = {}
# account -> seconds the last video in this queue took.
_clip_s: dict[str, float] = {}
# Held here so each pump outlives the tool call that started it.
_pumps: dict[str, asyncio.Task] = {}


def is_queue_id(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(PREFIX)


def _video_path(video_id: str) -> str:
    return f"/v1/videos/{quote(video_id, safe = '')}"


def _json(response: httpx.Response) -> dict:
    try:
        return as_dict(parse_json(response.content))
    except (ValueError, UnicodeDecodeError):
        return {}


def _is_busy(response: httpx.Response) -> bool:
    if response.status_code != 409:
        return False
    error = _json(response).get("error")
    return (
        isinstance(error, dict)
        and error.get("code") is None
        and error.get("message") == BUSY_MESSAGE
    )


def _ended(item: QueuedVideo) -> bool:
    # A submitted video runs until the next one starts (one at a time), so only the one this
    # account's queue still waits on may be running; its queue id must keep working until then.
    if item.status == "queued" or item.posting:
        return False
    running = _running.get(item.account_id)
    return not (item.status == "submitted" and running and running[0] == item.video_id)


def _evict() -> None:
    while len(_items) > CAPACITY:
        oldest = next((k for k, item in _items.items() if _ended(item)), None)
        if oldest is None:
            return
        _items.pop(oldest)


def _enqueue(caller: Caller, body: dict, hints: Optional[dict]) -> QueuedVideo:
    waiting = _waiting.setdefault(caller.account_id, deque())
    if len(waiting) >= MAX_WAITING:
        raise ToolError(
            f"{MAX_WAITING} videos are already queued for this account in Unsloth Studio; "
            "wait for one to start or cancel one."
        )
    if sum(len(queue) for queue in _waiting.values()) >= CAPACITY:
        raise ToolError(
            f"{CAPACITY} videos are already queued in Unsloth Studio; try again once some have started."
        )
    item = QueuedVideo(PREFIX + secrets.token_urlsafe(12), caller.account_id, caller, body, hints)
    waiting.append(item)
    _items[f"{item.account_id}:{item.id}"] = item
    _evict()
    return item


def _ensure_pump(account_id: str) -> None:
    task = _pumps.get(account_id)
    if task is None or task.done():
        _pumps[account_id] = asyncio.get_running_loop().create_task(_pump(account_id))


def _settle(
    item: QueuedVideo,
    status: str,
    *,
    error: Optional[str] = None,
) -> None:
    waiting = _waiting.get(item.account_id)
    if waiting and item in waiting:
        waiting.remove(item)
    item.status, item.error, item.posting = status, error, False


async def _wait_for_running(caller: Caller) -> None:
    """Until the video this queue waits on is no longer queued or in progress (or gone)."""
    seen, misses = False, 0
    while (running := _running.get(caller.account_id)) is not None:
        video_id, submitted_at = running
        read = await try_json(caller, _video_path(video_id))
        job = as_dict(read)
        # A failed read is no proof the video ended: only a run of them (it is gone) ends the wait.
        misses = 0 if isinstance(read, dict) else misses + 1
        if job.get("status") in RUNNING or 0 < misses < MAX_MISSES:
            seen = seen or job.get("status") in RUNNING
            await asyncio.sleep(POLL_S)
            continue
        # Only a video this wait saw running gives a clip time; an old one would inflate every ETA.
        if job.get("status") == "completed" and seen:
            _clip_s[caller.account_id] = time.monotonic() - submitted_at
        if _running.get(caller.account_id) == running:
            _running.pop(caller.account_id)


def _retry_after(response: httpx.Response) -> Optional[float]:
    # gpu_busy comes flat ({error, retry_after}) or inside detail / an OpenAI error.
    body = _json(response) if response.status_code in (409, 503) else {}
    busy = next(
        (
            c
            for c in (body, body.get("detail"), body.get("error"))
            if isinstance(c, dict) and "gpu_busy" in (c.get("error"), c.get("code"))
        ),
        None,
    )
    if busy is None:
        return None
    try:
        seconds = float(response.headers.get("retry-after") or busy.get("retry_after"))
    except (TypeError, ValueError):
        seconds = RETRY_S
    return min(max(seconds, 0.0), MAX_RETRY_AFTER_S)


async def _submit(item: QueuedVideo) -> Optional[float]:
    """POST the head once. Returns how long to wait before trying again, or None once it is settled."""
    item.posting = True
    try:
        response = await forward(
            item.caller, "POST", "/v1/videos", json_body = item.body, hub_header = True
        )
    except httpx.HTTPError:
        # Unsloth Studio restarting or slow to answer: the clip stays queued.
        return RETRY_S
    finally:
        item.posting = False
    if _is_busy(response):
        return RETRY_S
    delay = _retry_after(response)
    if delay is not None:
        return delay
    try:
        job = raise_for_route(response, hints = item.hints)
    except ToolError as exc:
        _settle(item, "failed", error = str(exc))
        return None
    video_id = opt_text(as_dict(job).get("id"))
    if video_id is None:
        _settle(item, "failed", error = "Unsloth Studio returned no video job")
        return None
    item.video_id = video_id
    _settle(item, "submitted")
    _running[item.account_id] = (video_id, time.monotonic())
    return None


async def _pump(account_id: str) -> None:
    while True:
        caller = None
        while waiting := _waiting.get(account_id):
            head = waiting[0]
            caller = head.caller
            try:
                await _wait_for_running(head.caller)
                if head.status != "queued" or not waiting or waiting[0] is not head:
                    continue
                delay = await _submit(head)
                if delay is not None:
                    await asyncio.sleep(delay)
            except asyncio.CancelledError:
                raise
            except Exception:
                _settle(head, "failed", error = "Unsloth Studio could not start the queued video.")
        if caller is None:
            return
        # Follow the last clip it started to the end, so its record can be dropped once it is done.
        with contextlib.suppress(Exception):
            await _wait_for_running(caller)
        if not _waiting.get(account_id):
            return


async def post_or_queue(
    caller: Caller,
    body: dict,
    *,
    hints: Optional[dict] = None,
) -> Union[dict, VideoJobRef]:
    """POST the video, or queue it when Unsloth Studio is busy with another video (or this account already has videos waiting, to keep their order). Returns the route's job, or the queued video's reference."""
    # One admission at a time per account: a second call must not start while the first, refused as
    # busy, is still on its way into the queue, or it would overtake it.
    async with _admission.setdefault(caller.account_id, asyncio.Lock()):
        return await _admit(caller, body, hints)


async def _admit(caller: Caller, body: dict, hints: Optional[dict]) -> Union[dict, VideoJobRef]:
    if not _waiting.get(caller.account_id):
        response = await forward(caller, "POST", "/v1/videos", json_body = body, hub_header = True)
        if not _is_busy(response):
            job = raise_for_route(response, hints = hints)
            video_id = opt_text(as_dict(job).get("id"))
            if video_id:
                _running[caller.account_id] = (video_id, time.monotonic())
            return job
    item = _enqueue(caller, body, hints)
    _ensure_pump(caller.account_id)
    position, eta = await _position_eta(item)
    return VideoJobRef.from_route(
        body, id = item.id, status = "queued", queue_position = position, eta_s = eta
    )


async def _position_eta(item: QueuedVideo) -> tuple[Optional[int], Optional[float]]:
    waiting = _waiting.get(item.account_id) or deque()
    if item not in waiting:
        return None, None
    position = waiting.index(item) + 1
    try:
        return position, await _eta(item.caller, position)
    except Exception:
        return position, None


async def _eta(caller: Caller, position: int) -> Optional[float]:
    """remaining(running) + (position - 1) * clip; None when Unsloth Studio gives nothing to go on."""
    running = _running.get(caller.account_id)
    if running is None:
        return None
    video_id, submitted_at = running
    elapsed = time.monotonic() - submitted_at
    remaining = None
    progress = as_dict(await try_json(caller, PROGRESS))
    if progress.get("active") is True:
        remaining = number(progress.get("eta_seconds"))
    if remaining is None:
        percent = number(as_dict(await try_json(caller, _video_path(video_id))).get("progress"))
        if percent is None or not 0 < percent < 100:
            return None
        remaining = elapsed * (100 - percent) / percent
    clip = _clip_s.get(caller.account_id) or (elapsed + remaining)
    return round(max(remaining, 0.0) + (position - 1) * clip, 1)


def lookup(account_id: str, queue_id: str) -> QueuedVideo:
    item = _items.get(f"{account_id}:{queue_id}")
    if item is None:
        raise ToolError("No such video job")
    return item


async def job_status(item: QueuedVideo) -> JobStatus:
    """get_job's answer for a queue id that has not become a real video."""
    # Read before the await: the queue can start the item while its position is being priced.
    status = item.status
    position, eta = await _position_eta(item) if status == "queued" else (None, None)
    return JobStatus(
        kind = "video",
        id = item.id,
        status = status,
        queue_position = position,
        eta_s = eta,
        error = item.error,
    )


def cancel(account_id: str, queue_id: str) -> tuple[bool, str]:
    item = lookup(account_id, queue_id)
    if item.video_id:
        return (
            False,
            f"It already started as {item.video_id}; cancel(kind='video') stops the running video.",
        )
    if item.posting:
        return False, "It is being started now; cancel(kind='video') stops the running video."
    if item.status != "queued":
        return False, f"The queued video already {item.status}."
    _settle(item, "cancelled")
    return True, "Removed the queued video before it started."


def _reset() -> None:
    """Test hook."""
    for task in _pumps.values():
        with contextlib.suppress(RuntimeError):  # its event loop may be closed already
            task.cancel()
    for state in (_items, _waiting, _running, _clip_s, _pumps, _admission):
        state.clear()
