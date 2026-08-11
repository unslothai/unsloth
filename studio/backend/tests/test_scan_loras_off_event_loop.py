# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GET /api/models/loras must not block the event loop.

The scan walks two directories and reads a config per checkpoint. It ran directly in the
coroutine, so for its whole duration the server could not serve anything else, streamed
chat tokens included. The page polls this route, so the stall repeated.
"""

from __future__ import annotations

import asyncio
import time

import pytest

import routes.models as models_routes


@pytest.mark.asyncio
async def test_the_scan_does_not_stall_other_requests(monkeypatch, tmp_path):
    """A heartbeat coroutine keeps ticking while a deliberately slow scan runs.

    On the unfixed route the sleep happens inside the coroutine, so the heartbeat gets no
    turn until the scan finishes and records a single long gap.
    """
    scan_seconds = 0.3

    def _slow_scan(outputs_dir: str, exports_dir: str):
        time.sleep(scan_seconds)
        return []

    monkeypatch.setattr(models_routes, "_scan_loras_sync", _slow_scan)
    monkeypatch.setattr(models_routes, "resolve_output_dir", lambda value: tmp_path)
    monkeypatch.setattr(models_routes, "resolve_export_dir", lambda value: tmp_path)

    ticks: list[float] = []
    stop = False

    async def heartbeat():
        while not stop:
            ticks.append(time.perf_counter())
            await asyncio.sleep(0.01)

    beat = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.05)
    await models_routes.scan_loras(
        outputs_dir = str(tmp_path),
        exports_dir = str(tmp_path),
        current_subject = "test-user",
    )
    stop = True
    beat.cancel()
    try:
        await beat
    except asyncio.CancelledError:
        pass

    longest_gap = max(b - a for a, b in zip(ticks, ticks[1:]))
    assert (
        longest_gap < scan_seconds / 2
    ), f"event loop blocked for {longest_gap:.3f}s during the scan"
