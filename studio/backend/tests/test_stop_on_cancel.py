# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""_stop_on_cancel must let a cancelled upstream read finish closing its connection.

When the client drops a managed-runtime stream (NPU, OpenVINO), Starlette cancels the relay through an
anyio cancel scope, which cancels every await that follows. The pending read then closes the upstream
response in its cancellation handler (httpcore does this). If _stop_on_cancel waits for that read with
asyncio.gather, the gather is cancelled too and cancels the read a second time mid-close: httpcore marks
the stream closed without closing the socket, and the runtime keeps generating an unread reply.
"""

import asyncio
import threading

import anyio

from routes.inference import _stop_on_cancel


def test_cancelled_read_finishes_closing_the_upstream():
    closed = asyncio.Event()

    async def upstream():
        yield "data: first"
        try:
            # A read that never completes: the runtime is still generating.
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)  # closing the connection takes an await or two
            closed.set()
            raise

    async def main():
        cancel_event = threading.Event()
        with anyio.CancelScope() as scope:
            async for _line in _stop_on_cancel(upstream(), cancel_event):
                scope.cancel()  # the client left after the first frame
        await asyncio.wait_for(closed.wait(), timeout = 2)

    asyncio.run(main())
