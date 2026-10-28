# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Live prefill / decode frames for the Unsloth chat UI (``generation_progress``)."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.sse_control_frames import (  # noqa: E402
    is_ui_control_sse_line,
    sanitize_provider_sse_line,
)
from routes import inference as route  # noqa: E402


def _frame(line: str) -> dict:
    assert line.startswith("data: ") and line.endswith("\n\n")
    return json.loads(line[len("data: ") :])


def test_prefill_samples_become_frames_once_each():
    signal = route._PrefillProgressSignal(None, live = True)
    assert signal.wants_timings, "a live signal must ask llama-server for per-token timings"
    assert signal.live_frame() is None

    signal(
        {
            "prompt_progress": {"total": 5316, "processed": 2048, "cache": 0, "time_ms": 1500},
            "prompt_per_second": 1365.33,
        }
    )
    frame = _frame(signal.live_frame())
    assert frame == {
        "type": "generation_progress",
        "phase": "prefill",
        "processed": 2048,
        "total": 5316,
        "cached": 0,
        "tok_per_s": 1365.3,
    }
    # Sent once; the stall loop polls far more often than llama-server reports a batch.
    assert signal.live_frame() is None


def test_prefill_counts_are_clamped_and_bad_rates_dropped():
    signal = route._PrefillProgressSignal(None, live = True)
    signal(
        {
            "prompt_progress": {"total": 100, "processed": 250, "cache": 400},
            "prompt_per_second": float("inf"),
        }
    )
    frame = _frame(signal.live_frame())
    assert frame["processed"] == 100
    assert frame["cached"] == 100
    assert frame["tok_per_s"] is None


def test_decode_frames_are_throttled(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(route.time, "monotonic", lambda: clock["now"])
    signal = route._PrefillProgressSignal(None, live = True)

    signal({"predicted_n": 10, "predicted_per_second": 48.7})
    assert _frame(signal.live_frame()) == {
        "type": "generation_progress",
        "phase": "decode",
        "tokens": 10,
        "tok_per_s": 48.7,
    }
    signal({"predicted_n": 11, "predicted_per_second": 49.0})
    assert signal.live_frame() is None, "a second decode frame inside the interval"
    clock["now"] += route._LIVE_DECODE_FRAME_INTERVAL_S
    assert _frame(signal.live_frame())["tokens"] == 11
    assert signal.live_frame() is None, "the same sample twice"


def test_a_new_prompt_pass_drops_the_last_rounds_decode_figures():
    signal = route._PrefillProgressSignal(None, live = True)
    signal({"predicted_n": 50, "predicted_per_second": 40.0})
    signal({"prompt_progress": {"total": 3000, "processed": 2048}})
    assert _frame(signal.live_frame())["phase"] == "prefill"
    assert signal.live_frame() is None, "round 1's decode frame must not follow round 2's prefill"


def test_decode_frame_carries_draft_acceptance():
    signal = route._PrefillProgressSignal(None, live = True)
    signal({"predicted_n": 40, "predicted_per_second": 90.0, "draft_n": 23, "draft_n_accepted": 14})
    frame = _frame(signal.live_frame())
    assert (frame["draft_n"], frame["draft_accepted"]) == (23, 14)


def test_not_live_keeps_the_old_behaviour():
    calls = []

    def inner(sample):
        calls.append(sample)

    signal = route._PrefillProgressSignal(inner)
    signal({"prompt_progress": {"total": 10, "processed": 5}})
    signal({"predicted_n": 3, "predicted_per_second": 20.0})
    assert signal.live_frame() is None
    assert signal.advanced() is True
    assert len(calls) == 2, "the monitor callback still sees every sample"
    assert route._PrefillProgressSignal(None).wants_timings is False


def test_stall_wait_sends_progress_while_the_generator_is_silent(monkeypatch):
    monkeypatch.setattr(route, "_LIVE_PROGRESS_POLL_S", 0.01)
    signal = route._PrefillProgressSignal(None, live = True)

    async def scenario():
        release = asyncio.Event()

        async def step():
            await release.wait()
            return "token"

        task = asyncio.create_task(step())
        lines = []

        async def collect():
            async for line in route._stall_wait_frames(task, signal, first_timeout = 15.0):
                lines.append(line)

        collector = asyncio.create_task(collect())
        await asyncio.sleep(0.03)
        signal({"prompt_progress": {"total": 4000, "processed": 2048}})
        await asyncio.sleep(0.05)
        release.set()
        await collector
        return lines, task.result()

    lines, result = asyncio.run(scenario())
    assert result == "token"
    assert [_frame(line)["processed"] for line in lines] == [2048]


def test_a_step_faster_than_the_poll_still_sends_what_it_recorded():
    # Fast tokens (or tool-call markup stripped to no text) never reach the poll.
    signal = route._PrefillProgressSignal(None, live = True)

    async def scenario():
        async def step():
            signal({"prompt_progress": {"total": 4000, "processed": 4000}})
            signal({"predicted_n": 3, "predicted_per_second": 20.0})
            return "token"

        task = asyncio.create_task(step())
        await task
        return [line async for line in route._stall_wait_frames(task, signal, first_timeout = 15.0)]

    lines = asyncio.run(scenario())
    assert [_frame(line)["phase"] for line in lines] == ["prefill", "decode"]
    assert _frame(lines[0])["processed"] == 4000 and _frame(lines[1])["tokens"] == 3


def test_stall_wait_still_sends_the_comment_after_the_stall_interval(monkeypatch):
    monkeypatch.setattr(route, "_LIVE_PROGRESS_POLL_S", 0.01)

    async def scenario(live: bool):
        signal = route._PrefillProgressSignal(None, live = live)
        signal({"prompt_progress": {"total": 4000, "processed": 1}})
        signal.live_frame()
        release = asyncio.Event()

        async def step():
            await release.wait()

        task = asyncio.create_task(step())
        lines = []

        async def collect():
            async for line in route._stall_wait_frames(task, signal, first_timeout = 0.05):
                lines.append(line)

        collector = asyncio.create_task(collect())
        await asyncio.sleep(0.12)
        release.set()
        await collector
        return lines

    for live in (False, True):
        lines = asyncio.run(scenario(live))
        assert lines and lines[0] == route._OPENAI_PREFILL_PROGRESS_SSE, (live, lines)
        assert all(not line.startswith("data:") for line in lines)


@pytest.mark.parametrize(
    "line",
    [
        'data: {"type":"generation_progress","phase":"prefill","processed":1,"total":2}',
    ],
)
def test_providers_cannot_forge_the_frame(line):
    assert sanitize_provider_sse_line(line) is None
    assert is_ui_control_sse_line(line)
