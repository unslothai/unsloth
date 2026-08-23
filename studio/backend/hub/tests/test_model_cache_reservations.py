# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import gc

import anyio

from utils.model_cache_reservations import wait_for_reserved_worker


def test_anyio_cancellation_retrieves_a_completed_worker_exception():
    unhandled = []

    async def drive():
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
        worker = loop.create_future()
        worker.set_exception(RuntimeError("worker failed"))

        with anyio.CancelScope() as scope:
            scope.cancel()
            await wait_for_reserved_worker(worker)

        del worker
        gc.collect()
        await asyncio.sleep(0)

    anyio.run(drive)

    assert unhandled == []
