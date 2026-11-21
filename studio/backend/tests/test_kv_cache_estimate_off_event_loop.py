# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Blocking disk reads run once per row, so they must leave the event loop or streamed chat stalls."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

# Installs process-wide stubs so this module runs standalone.
import test_kv_cache_estimation  # noqa: F401,E402

import routes.models as models_routes  # noqa: E402


def test_the_estimate_does_not_stall_other_requests(monkeypatch, tmp_path):
    """A heartbeat must keep ticking during a slow resolve; the unfixed route gave it no turn at all."""
    resolve_seconds = 0.3
    heartbeat_seconds = 0.01

    def _slow_resolve(repo_id: str, quant: str, is_local: bool):
        time.sleep(resolve_seconds)
        return None, 0

    monkeypatch.setattr(models_routes, "_resolve_quant_gguf", _slow_resolve)

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
        # Only ticks strictly between call and return; the stall leaves no gap in the whole list.
        before = len(ticks)
        result = await models_routes.get_kv_cache_estimate(
            repo_id = "org/repo",
            quant = "Q4_K_M",
            n_ctx = 4096,
            cache_type_kv = None,
            n_parallel = 1,
            speculative_type = None,
            request = None,
            current_subject = "test-user",
        )
        during.append(len(ticks) - before)
        assert result["kv_bytes"] is None
        stop = True
        beat.cancel()
        try:
            await beat
        except asyncio.CancelledError:
            pass

    asyncio.run(_drive())

    # Loaded runner records well below ~30 ticks; blocking records exactly 0.
    assert during[0] >= 3, f"heartbeat ran {during[0]} times during a {resolve_seconds}s estimate"
