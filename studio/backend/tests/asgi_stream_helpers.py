# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Wait for SSE frames; if the driving task failed first, raise its exception, not a bare timeout."""

from __future__ import annotations

import asyncio


async def wait_for_frame(
    event: asyncio.Event,
    task: asyncio.Task,
    *,
    timeout: float = 20.0,
    what: str = "the expected SSE frame",
) -> None:
    """The timeout is a deadlock backstop only: a timeout means the event never happened, not slow."""
    waiter = asyncio.ensure_future(event.wait())
    try:
        done, _ = await asyncio.wait(
            {waiter, task}, timeout = timeout, return_when = asyncio.FIRST_COMPLETED
        )
        # Check the task first: a send() that sets the event then raises would otherwise be swallowed.
        if task in done:
            exc = task.exception()
            if exc is not None:
                when = "after" if event.is_set() else "before"
                raise AssertionError(f"the request failed {when} {what} was sent: {exc!r}") from exc
        if waiter in done:
            return
        if task in done:
            raise AssertionError(f"the request completed without sending {what}")
        raise AssertionError(
            f"timed out after {timeout}s waiting for {what}, and the request task is still "
            f"running: the path is parked rather than failing"
        )
    finally:
        waiter.cancel()
