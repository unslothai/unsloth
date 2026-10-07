# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""_stop_on_cancel must let a cancelled upstream read finish closing its httpcore connection."""

import asyncio
import threading

import anyio

from routes.inference import _stop_on_cancel


def test_cancelled_read_finishes_closing_the_upstream():
    closed = asyncio.Event()

    async def upstream():
        yield "data: first"
        try:
            # the runtime keeps generating while the read is pending
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)  # model connection cleanup that requires another await
            closed.set()
            raise

    async def main():
        cancel_event = threading.Event()
        with anyio.CancelScope() as scope:
            async for _line in _stop_on_cancel(upstream(), cancel_event):
                scope.cancel()  # the client left after the first frame
        await asyncio.wait_for(closed.wait(), timeout = 2)

    asyncio.run(main())
