# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Wait for an SSE frame without swallowing the reason it never arrived.

The slot-release tests drive the ASGI app in a task and wait on an event a fake ``send()`` sets. If
the request path raises first, the wait times out and the ``finally``'s ``gather(...,
return_exceptions = True)`` discards the real exception, leaving a bare ``TimeoutError``. That is
how #8700's unguarded ``llama_backend.context_length`` read surfaced: four 20-second "flakes"
hiding an ``AttributeError``. So if the driving task has failed, raise ITS exception.
"""

from __future__ import annotations

import asyncio


async def wait_for_frame(
    event: asyncio.Event,
    task: asyncio.Task,
    *,
    timeout: float = 20.0,
    what: str = "the expected SSE frame",
) -> None:
    """Wait for *event*, surfacing *task*'s exception if it died first.

    The timeout is a deadlock backstop, not a performance assertion: these tests take milliseconds
    when they pass, so a timeout means "never happened", not "too slow".
    """
    waiter = asyncio.ensure_future(event.wait())
    try:
        done, _ = await asyncio.wait(
            {waiter, task}, timeout = timeout, return_when = asyncio.FIRST_COMPLETED
        )
        # The task is checked FIRST, even when the frame did go out. A send() that sets the event
        # and then raises leaves both futures done, and returning on the frame there would drop the
        # exception into the caller's gather(return_exceptions = True) -- the exact silence this
        # helper exists to break.
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
