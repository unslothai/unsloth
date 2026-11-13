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
        "time_ms": 1500,
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
    assert frame["time_ms"] is None, "an older llama-server without time_ms"


def test_decode_frames_are_throttled(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(route.time, "monotonic", lambda: clock["now"])
    signal = route._PrefillProgressSignal(None, live = True)

    signal({"predicted_n": 10, "predicted_per_second": 48.7})
    assert signal.live_frame() is None, "the pass's first sample opens the interval"
    clock["now"] += route._LIVE_DECODE_FRAME_INTERVAL_S
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


def test_the_first_frame_waits_an_interval_past_the_first_token(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(route.time, "monotonic", lambda: clock["now"])
    signal = route._PrefillProgressSignal(None, live = True)
    signal({"predicted_n": 1, "predicted_per_second": 0.0})
    assert signal.live_frame() is None, "a lone '1 token' held the line for a whole interval"
    clock["now"] += 0.01
    signal({"predicted_n": 2, "predicted_per_second": 12.5})
    assert signal.live_frame() is None, "one step's rate is noise"
    clock["now"] += route._LIVE_DECODE_FRAME_INTERVAL_S
    signal({"predicted_n": 31, "predicted_per_second": 118.0})
    assert _frame(signal.live_frame())["tokens"] == 31
    signal({"prompt_progress": {"total": 3000, "processed": 3000}})
    signal.live_frame()
    clock["now"] += route._LIVE_DECODE_FRAME_INTERVAL_S
    signal({"predicted_n": 1, "predicted_per_second": 0.0})
    signal({"predicted_n": 2, "predicted_per_second": 40.0})
    assert signal.live_frame() is None


def test_a_new_prompt_pass_drops_the_last_rounds_decode_figures():
    signal = route._PrefillProgressSignal(None, live = True)
    signal({"predicted_n": 50, "predicted_per_second": 40.0})
    signal({"prompt_progress": {"total": 3000, "processed": 2048}})
    assert _frame(signal.live_frame())["phase"] == "prefill"
    assert signal.live_frame() is None, "round 1's decode frame must not follow round 2's prefill"


def test_decode_frame_carries_draft_acceptance(monkeypatch):
    monkeypatch.setattr(route, "_LIVE_DECODE_FRAME_INTERVAL_S", 0.0)
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


def test_a_step_faster_than_the_poll_still_sends_what_it_recorded(monkeypatch):
    # Fast tokens (or tool-call markup stripped to no text) never reach the poll.
    monkeypatch.setattr(route, "_LIVE_DECODE_FRAME_INTERVAL_S", 0.0)
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


def test_a_compaction_report_becomes_one_frame_ahead_of_the_prefill():
    signal = route._PrefillProgressSignal(None, live = True)
    signal.compacting()
    signal({"prompt_progress": {"total": 4000, "processed": 512}})
    assert _frame(signal.live_frame()) == {"type": "generation_progress", "phase": "compacting"}
    assert _frame(signal.live_frame())["phase"] == "prefill"
    assert signal.live_frame() is None
    signal.compacting()
    assert _frame(signal.live_frame())["phase"] == "compacting"


def test_a_compaction_report_is_dropped_when_not_live():
    signal = route._PrefillProgressSignal(None)
    signal.compacting()
    assert signal.live_frame() is None


def test_the_last_passs_unsent_figures_never_follow_a_compaction_report(monkeypatch):
    # The tool call decoded within the throttle, so its decode figures were still pending when the fit began.
    monkeypatch.setattr(route, "_LIVE_DECODE_FRAME_INTERVAL_S", 0.0)
    signal = route._PrefillProgressSignal(None, live = True)
    signal({"prompt_progress": {"total": 4000, "processed": 4000}})
    signal({"predicted_n": 30, "predicted_per_second": 20.0})
    signal.compacting()
    assert _frame(signal.live_frame())["phase"] == "compacting"
    assert signal.live_frame() is None, "a decode frame here would clear the label on the client"


def test_a_tool_result_drops_the_last_passs_unsent_figures(monkeypatch):
    monkeypatch.setattr(route, "_LIVE_DECODE_FRAME_INTERVAL_S", 0.0)
    signal = route._PrefillProgressSignal(None, live = True)
    signal({"prompt_progress": {"total": 4000, "processed": 4000}})
    signal({"predicted_n": 30, "predicted_per_second": 20.0})
    signal.round_ended()
    assert signal.live_frame() is None
    signal({"prompt_progress": {"total": 4100, "processed": 4050}})
    assert _frame(signal.live_frame())["processed"] == 4050


def test_stall_wait_sends_the_compaction_frame_while_the_fit_blocks(monkeypatch):
    monkeypatch.setattr(route, "_LIVE_PROGRESS_POLL_S", 0.01)
    signal = route._PrefillProgressSignal(None, live = True)

    async def scenario():
        release = asyncio.Event()

        async def step():
            await release.wait()
            return "context_truncated"

        task = asyncio.create_task(step())
        lines = []

        async def collect():
            async for line in route._stall_wait_frames(task, signal, first_timeout = 15.0):
                lines.append(line)

        collector = asyncio.create_task(collect())
        signal.compacting()
        await asyncio.sleep(0.05)
        # Sent while the step is still blocked, not when the fit returns.
        sent_before_release = list(lines)
        release.set()
        await collector
        return sent_before_release, lines

    before, lines = asyncio.run(scenario())
    assert [_frame(line)["phase"] for line in before] == ["compacting"]
    assert len(lines) == 1


def _chars(candidate):
    return sum(len(str(message.get("content", ""))) for message in candidate)


def _fit(
    messages,
    on_compacting,
    counter = _chars,
    **kwargs,
):
    from core.inference import chat_compaction
    return chat_compaction._fit_with_instruction_pins(
        messages,
        context_length = 1000,
        max_tokens = 128,
        count_tokens = counter,
        on_compacting = on_compacting,
        **kwargs,
    )


def _turns(size):
    return [
        {"role": "user", "content": "A" * size},
        {"role": "assistant", "content": "B" * size},
        {"role": "user", "content": "C" * 50},
    ]


def test_the_fit_reports_once_when_it_starts_evicting():
    reports = []
    fitted, info = _fit(_turns(600), lambda: reports.append(True))
    assert info is not None and info["dropped_messages"] > 0
    assert reports == [True], "one report per fit, however many counts the trim takes"


def test_a_fit_that_keeps_everything_reports_nothing():
    reports = []
    fitted, info = _fit(_turns(100), lambda: reports.append(True))
    assert info is None
    assert reports == []


def test_a_forced_reset_reports(monkeypatch):
    from core.inference import chat_compaction

    monkeypatch.setattr(chat_compaction, "_archive_is_degraded", lambda: False)
    reports = []
    fitted, info = _fit(
        _turns(100),
        lambda: reports.append(True),
        force_reset = True,
        context_policy = "checkpoint",
        can_reset = True,
    )
    assert info is not None and info["checkpoint_started"]
    assert reports == [True]


def test_a_replay_only_fit_never_reports():
    reports = []
    _fit(_turns(600), lambda: reports.append(True), replay_only = True, sticky_dropped = 1)
    assert reports == []


def _compacted_chat(recent_size):
    """Four turns a previous request already cut (the saved boundary), then the turns still in the window."""
    old = [
        {"role": role, "content": letter * 300}
        for role, letter in (("user", "A"), ("assistant", "B"), ("user", "C"), ("assistant", "D"))
    ]
    recent = [
        {"role": "user", "content": "E" * recent_size},
        {"role": "assistant", "content": "F" * recent_size},
        {"role": "user", "content": "G" * 50},
    ]
    return old + recent


_REPLAYS = {
    "rolling": {"context_policy": "rolling", "sticky_is_checkpoint": False, "keeps_boundary": True},
    "checkpoint": {
        "context_policy": "checkpoint",
        "sticky_is_checkpoint": True,
        "can_reset": False,
    },
}


@pytest.mark.parametrize("policy", sorted(_REPLAYS))
def test_replaying_a_saved_cut_is_not_reported(policy):
    # The full history is over budget on every request after the first compaction; replaying the cut it
    # already made evicts nothing new, so "Compacting context..." must not flash on each later turn.
    reports = []
    fitted, info = _fit(
        _compacted_chat(100), lambda: reports.append(True), sticky_dropped = 4, **_REPLAYS[policy]
    )
    assert info is not None and info["fits"] and info["dropped_messages"] == 4
    assert reports == []


@pytest.mark.parametrize("policy", sorted(_REPLAYS))
def test_a_replay_that_no_longer_fits_is_reported(policy):
    reports = []
    fitted, info = _fit(
        _compacted_chat(500), lambda: reports.append(True), sticky_dropped = 4, **_REPLAYS[policy]
    )
    assert info is not None and info["dropped_messages"] > 4
    assert reports == [True]


def test_a_new_checkpoint_past_a_saved_cut_is_reported(monkeypatch):
    # No system message, so the checkpoint block adds one, and a block the template prices at nothing leaves the
    # reset's own count under budget: only the eviction decision itself shows that a new epoch started.
    from core.inference import chat_compaction

    monkeypatch.setattr(chat_compaction, "_archive_is_degraded", lambda: False)
    reports = []
    fitted, info = _fit(
        _compacted_chat(450),
        lambda: reports.append(True),
        counter = lambda candidate: _chars([m for m in candidate if m.get("role") != "system"]),
        sticky_dropped = 4,
        context_policy = "checkpoint",
        sticky_is_checkpoint = True,
        can_reset = True,
    )
    assert info is not None and info["checkpoint_started"] and info["dropped_messages"] > 4
    assert reports == [True]


def test_a_failing_report_never_breaks_the_fit():
    def boom():
        raise RuntimeError("closed stream")

    fitted, info = _fit(_turns(600), boom)
    assert info is not None and info["fits"] is True


def test_only_a_live_signal_is_a_compaction_reporter():
    from core.inference import chat_compaction

    signal = route._PrefillProgressSignal(None, live = True)
    assert chat_compaction._compaction_reporter(signal) == signal.compacting
    assert chat_compaction._compaction_reporter(lambda sample: None) is None
    assert chat_compaction._compaction_reporter(None) is None
