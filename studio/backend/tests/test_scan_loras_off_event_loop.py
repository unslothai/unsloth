# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GET /api/models/loras must not block the event loop.

The scan walks two directories and reads a tokenizer per checkpoint. It ran directly in
the coroutine, so for its whole duration the server could not serve anything else,
streamed chat tokens included. The page polls this route, so the stall repeated.
"""

from __future__ import annotations

import asyncio
import time

import routes.models as models_routes


def test_the_scan_does_not_stall_other_requests(monkeypatch, tmp_path):
    """A heartbeat coroutine keeps ticking while a deliberately slow scan runs.

    On the unfixed route the sleep happens inside the coroutine, so the heartbeat gets no
    turn at all between the call and its return.
    """
    scan_seconds = 0.3
    heartbeat_seconds = 0.01

    # Three positionals, matching the real _scan_loras_sync: a two-argument stand-in fails on the call.
    def _slow_scan(outputs_dir: str, exports_dir: str, hf_token):
        time.sleep(scan_seconds)
        return []

    monkeypatch.setattr(models_routes, "_scan_loras_sync", _slow_scan)
    monkeypatch.setattr(models_routes, "resolve_output_dir", lambda value: tmp_path)
    monkeypatch.setattr(models_routes, "resolve_export_dir", lambda value: tmp_path)

    ticks: list[float] = []
    during: list[int] = []

    async def _drive():
        stop = False

        async def heartbeat():
            while not stop:
                ticks.append(time.perf_counter())
                await asyncio.sleep(heartbeat_seconds)

        beat = asyncio.create_task(heartbeat())
        await asyncio.sleep(heartbeat_seconds * 5)
        # Ticks strictly between the call and its return: over the whole tick list the stall leaves no
        # gap at all, because the last tick lands before the blocking call and stop/cancel runs before
        # the heartbeat is rescheduled.
        before = len(ticks)
        await models_routes.scan_loras(
            outputs_dir = str(tmp_path),
            exports_dir = str(tmp_path),
            current_subject = "test-user",
        )
        during.append(len(ticks) - before)
        stop = True
        beat.cancel()
        try:
            await beat
        except asyncio.CancelledError:
            pass

    # asyncio.run, not pytest.mark.asyncio: pytest-asyncio is not a dependency of this backend.
    asyncio.run(_drive())

    # Blocking records exactly 0 while a loaded runner lands far below an idle one's ~30: the floor separates.
    assert during[0] >= 3, f"heartbeat ran {during[0]} times during a {scan_seconds}s scan"
