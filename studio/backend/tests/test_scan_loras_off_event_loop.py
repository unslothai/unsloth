# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GET /api/models/loras must run its scan off the event loop, so other requests keep being served."""

from __future__ import annotations

import asyncio
import time

import routes.models as models_routes


def test_the_scan_does_not_stall_other_requests(monkeypatch, tmp_path):
    """A heartbeat coroutine must keep ticking during a slow scan; blocking would stop it entirely."""
    scan_seconds = 0.3
    heartbeat_seconds = 0.01

    # Same arity as the real _scan_loras_sync, so the test fails on the assertion, not on the call.
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
        # Count only ticks between the call and its return; a whole-list gap cannot catch the stall.
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

    # asyncio.run: pytest-asyncio is not a dependency of this backend.
    asyncio.run(_drive())

    # Idle records ~30, blocking records 0, so the floor is deliberately loose.
    assert during[0] >= 3, f"heartbeat ran {during[0]} times during a {scan_seconds}s scan"
