# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every kind of long-running Unsloth Studio work behind one interface: ``follow`` reads a job and waits for it to move, ``cancel`` stops one. Each kind plugs in as a ``Kind`` registered here (the built-in ones in job_kinds); get_job and cancel only pick the kind."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, NamedTuple, Optional

from fastmcp import Context
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult

from studio_mcp import loading
from studio_mcp.caller import Caller
from studio_mcp.media import ReturnMedia, media_result
from studio_mcp.outputs import CancelResult, JobStatus
from studio_mcp.tools import as_dict


class Stop(NamedTuple):
    """What a cancel did: whether it stopped something, what to tell the agent, and the job it acted on."""

    cancelled: bool
    message: Optional[str]
    id: Optional[str]


# read(caller, job_id, rows = ..., offset = ..., return_media = ..., preview_px = ...)
Read = Callable[..., Awaitable[ToolResult]]
# cancel(caller, job_id, save)
Cancel = Callable[[Caller, Optional[str], bool], Awaitable[Stop]]


@dataclass(frozen = True)
class Kind:
    name: str
    read: Optional[Read] = None
    cancel: Optional[Cancel] = None
    # Seconds a caller not using wait_s should leave before asking again.
    poll_after_s: float = 15
    # Without an id a read follows the current job rather than listing recent ones, so it can wait.
    follows_current: bool = False
    # What a cancel's id is, for a kind whose cancel needs one.
    needs_id: Optional[str] = None
    # False for a kind whose cancel stops whatever is running and takes no id.
    takes_id: bool = True


KINDS: dict[str, Kind] = {}


def register(kind: Kind, *aliases: str) -> Kind:
    """Plug in a kind of work; ``aliases`` name it too (a model download is "download" to read, "model_download" to cancel)."""
    for name in (kind.name, *aliases):
        KINDS[name] = kind
    return kind


def kind_of(name: str) -> Kind:
    kind = KINDS.get(name)
    if kind is None:
        raise ToolError(f"Unknown job kind {name!r}")
    return kind


# The least time follow gives its first read and, at the end of a wait, its rows read.
FIRST_READ_S = 30
LAST_READ_S = 5
# Job statuses that mean the job is still going; a wait waits only on these.
RUNNING = {"running", "queued", "in_progress", "created", "pending", "active", "cancelling"}


def _moment(structured: Any) -> tuple[Any, Any, Any]:
    # (status, phase, percent): a wait ends when the first two change. Export has no percent, only a phase.
    job = as_dict(structured)
    phase = as_dict(job.get("export")).get("phase") or as_dict(job.get("training")).get("phase")
    return job.get("status"), phase, job.get("progress_percent")


async def follow(
    caller: Caller,
    name: str,
    job_id: Optional[str],
    *,
    wait_s: int = 0,
    ctx: Optional[Context] = None,
    rows: Optional[int] = None,
    offset: int = 0,
    return_media: ReturnMedia = "inline",
    preview_px: Optional[int] = None,
) -> ToolResult:
    """Read job ``job_id`` of kind ``name`` and, given ``wait_s``, wait for it to finish or move on."""
    kind = kind_of(name)
    if kind.read is None:
        raise ToolError(f"Unknown job kind {name!r}")

    def read(rows: Optional[int]) -> Awaitable[ToolResult]:
        return kind.read(
            caller,
            job_id,
            rows = rows,
            offset = offset,
            return_media = return_media,
            preview_px = preview_px,
        )

    deadline = time.monotonic() + min(wait_s, loading.MAX_WAIT_S)
    # Forwarded reads have no HTTP timeout, so each one here is bounded: the first by the wait (at
    # least FIRST_READ_S), the polls and the final rows read by what is left of it.
    first = max(min(wait_s, loading.MAX_WAIT_S), FIRST_READ_S)
    try:
        result = await asyncio.wait_for(read(rows), timeout = first)
    except asyncio.TimeoutError:
        raise ToolError(
            f"Unsloth Studio did not report on the {name} job within {first:g} s; call get_job again"
        ) from None
    start = _moment(result.structured_content)
    # A listing has nothing to wait on.
    waits = job_id is not None or kind.follows_current
    moved = polled = False
    while waits and start[0] in RUNNING:
        left = deadline - time.monotonic()
        if left <= 0:
            break
        await asyncio.sleep(min(loading.POLL_INTERVAL_S, left))
        try:
            # Recipe rows are read once, for the answer, not on every poll; a read that stalls past
            # the deadline ends the wait with the last answer.
            latest = await asyncio.wait_for(
                read(None), timeout = max(deadline - time.monotonic(), 0.01)
            )
        except asyncio.TimeoutError:
            break
        except Exception:
            # Like a load's progress poll: a failed read during the wait is skipped.
            continue
        result, polled = latest, True
        now = _moment(result.structured_content)
        if now[:2] != start[:2]:
            moved = True
            break
        if ctx is not None and now[2] is not None:
            await ctx.report_progress(min(now[2] / 100.0, 1.0), 1.0, str(now[1] or now[0]))
    if polled and rows:
        try:
            result = await asyncio.wait_for(
                read(rows), timeout = max(deadline - time.monotonic(), LAST_READ_S)
            )
        except asyncio.TimeoutError:
            pass  # The answer goes out without rows; the next get_job reads them.
    # A wait that ended because the job moved on can be followed again at once.
    return _with_poll_after(result, 0 if moved else kind.poll_after_s)


def _with_poll_after(result: ToolResult, after: float) -> ToolResult:
    structured = as_dict(result.structured_content)
    if structured.get("status") not in RUNNING or structured.get("jobs") is not None:
        return result
    status = JobStatus.model_validate({**structured, "poll_after_s": after})
    return media_result(list(result.content[1:]), status)


async def cancel(
    caller: Caller,
    name: str,
    job_id: Optional[str],
    save: bool = True,
) -> CancelResult:
    """Stop job ``job_id`` of kind ``name``; ``cancelled`` says whether this call stopped something."""
    kind = kind_of(name)
    if kind.cancel is None:
        raise ToolError(f"Unknown job kind {name!r}")
    if kind.needs_id and not job_id:
        raise ToolError(f"cancel(kind={name!r}) needs id: {kind.needs_id}.")
    if not kind.takes_id and job_id:
        raise ToolError(
            f"cancel(kind={name!r}) takes no id: this cancel stops the running one; leave id out."
        )
    stop = await kind.cancel(caller, job_id, save)
    return CancelResult(kind = name, id = stop.id, cancelled = stop.cancelled, message = stop.message)


from studio_mcp import job_kinds as _built_in  # noqa: E402,F401  registers the built-in kinds
