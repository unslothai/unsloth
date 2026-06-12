# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import asyncio
import os
import sys
from types import SimpleNamespace

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import routes.inference as inf_mod  # noqa: E402


def test_preheader_send_cleanup_on_disconnect_and_cancel():
    async def _run(cancel_parent):
        state = SimpleNamespace(disconnected = False, closed = False, cancelled = False)
        started = asyncio.Event()

        class _Client:
            async def send(
                self,
                req,
                stream = False,
            ):
                started.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    state.cancelled = True
                    raise

            async def aclose(self):
                state.closed = True

        class _Request:
            async def is_disconnected(self):
                return state.disconnected

        task = asyncio.create_task(
            inf_mod._send_stream_with_preheader_cancel(_Client(), object(), request = _Request())
        )
        await started.wait()
        if cancel_parent:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("helper cancellation did not propagate")
        else:
            state.disconnected = True
            assert await task is None
        assert state.closed
        assert state.cancelled

    asyncio.run(_run(False))
    asyncio.run(_run(True))
