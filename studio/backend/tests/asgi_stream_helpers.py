# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Wait for an SSE frame without swallowing the reason it never arrived.

These slot-release tests drive the real ASGI app in a task and wait on an ``asyncio.Event`` that a
fake ``send()`` sets when the frame of interest goes out. If the request path raises before that
frame, the event never fires, the wait hits its timeout, and the ``finally`` clause's
``gather(task, return_exceptions = True)`` throws the real exception away. What CI then prints is a
bare ``TimeoutError`` with no traceback and no attribute name.

That is not hypothetical. #8700 added an unguarded ``llama_backend.context_length`` read to the
chat-completions path; the doubles here did not have the attribute, and four tests spent 20 seconds
each waiting for a frame that an ``AttributeError`` had already prevented. Read as a timing flake,
the obvious "fixes" are to raise the timeout or mark them flaky, and both would have buried a real
one-line bug.

So: if the driving task has already failed, raise ITS exception, not the timeout.
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

    The timeout stays generous on purpose: it is a deadlock backstop, not a performance assertion.
    These tests take tens of milliseconds when they pass, so a timeout never means "too slow", it
    means "never happened" -- and this makes the request path say why.
    """
    waiter = asyncio.ensure_future(event.wait())
    try:
        done, _ = await asyncio.wait(
            {waiter, task}, timeout = timeout, return_when = asyncio.FIRST_COMPLETED
        )
        if waiter in done:
            return
        if task in done:
            # Re-raises inside the test, with the original traceback.
            exc = task.exception()
            if exc is not None:
                raise AssertionError(
                    f"the request failed before {what} was sent: {exc!r}"
                ) from exc
            raise AssertionError(
                f"the request completed without sending {what}"
            )
        raise AssertionError(
            f"timed out after {timeout}s waiting for {what}, and the request task is still "
            f"running: the path is parked rather than failing"
        )
    finally:
        waiter.cancel()
