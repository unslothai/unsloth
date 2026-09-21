# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GET /kv-cache-estimate must not block the event loop.

The route reads a GGUF header, walks every HF cache root in _resolve_quant_gguf,
looks for a drafter and probes the llama-server binary. All of that is blocking
disk work, and the memory bar calls it once per visible row, so a long model list
ran it many times over. Doing that inside the coroutine stalls everything else
the process is serving, streamed chat tokens included.

Mirrors test_scan_loras_off_event_loop.py.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

# Installs the process-wide loggers/structlog/httpx stubs, so this module can be
# run on its own rather than only after something else has imported them.
import test_kv_cache_estimation  # noqa: F401,E402

import routes.models as models_routes  # noqa: E402


def test_the_estimate_does_not_stall_other_requests(monkeypatch, tmp_path):
    """A heartbeat coroutine keeps ticking while a deliberately slow resolve runs.

    On the unfixed route the sleep happens inside the coroutine, so the heartbeat
    gets no turn at all between the call and its return.
    """
    resolve_seconds = 0.3
    heartbeat_seconds = 0.01

    def _slow_resolve(repo_id: str, quant: str, is_local: bool):
        time.sleep(resolve_seconds)
        return None, 0  # no path -> the handler returns its null answer

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
        # Ticks recorded strictly between the call and its return: a gap measured
        # over the whole list cannot separate the two versions, because the stall
        # leaves no gap in the list at all.
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
        # The answer still has to be the route's, not a coroutine object.
        assert result["kv_bytes"] is None
        stop = True
        beat.cancel()
        try:
            await beat
        except asyncio.CancelledError:
            pass

    asyncio.run(_drive())

    # A loaded runner lands well below the ~30 an idle one records; blocking
    # records exactly 0, so the floor is loose and still separates the two.
    assert during[0] >= 3, f"heartbeat ran {during[0]} times during a {resolve_seconds}s estimate"
