# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio-measured timings on provider usage chunks, so API connections show tok/s like llama.cpp."""

import asyncio
import json
import types

import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import with_studio_timings


def _data(payload) -> str:
    return "data: " + json.dumps(payload)


def _delta(**delta) -> str:
    return _data({"choices": [{"index": 0, "delta": delta, "finish_reason": None}]})


def _usage(**usage) -> str:
    return _data({"choices": [], "usage": usage})


@pytest.fixture
def clock(monkeypatch):
    """Each time.monotonic() call reads the next tick; asyncio keeps the real clock."""
    ticks: list[float] = []
    monkeypatch.setattr(ep_mod, "time", types.SimpleNamespace(monotonic = lambda: ticks.pop(0)))
    return ticks


def _run(lines: list[str]) -> list[dict]:
    async def upstream():
        for line in lines:
            yield line

    async def collect():
        return [line async for line in with_studio_timings(upstream())]

    out = asyncio.run(collect())
    return [json.loads(line[5:]) for line in out if line != "data: [DONE]"]


def test_usage_chunk_gets_studio_timings(clock):
    # start 10.0, first delta 10.5, last delta 12.5; the usage chunk reads no clock, so its delay is ignored.
    clock.extend([10.0, 10.5, 12.5])
    chunks = _run(
        [
            _delta(role = "assistant"),
            _delta(content = "Hi"),
            _delta(content = " there"),
            _data({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            _usage(
                prompt_tokens = 109,
                completion_tokens = 40,
                total_tokens = 149,
                prompt_tokens_details = {"cached_tokens": 9},
            ),
            "data: [DONE]",
        ]
    )
    assert all("timings" not in c for c in chunks[:-1])
    t = chunks[-1]["timings"]
    assert t["source"] == "studio"
    assert t["predicted_n"] == 40
    assert t["predicted_ms"] == pytest.approx(2000.0)
    assert t["predicted_per_second"] == pytest.approx(20.0)
    assert t["cache_n"] == 9
    assert t["prompt_n"] == 100
    assert t["prompt_ms"] == pytest.approx(500.0)
    assert t["prompt_per_second"] == pytest.approx(200.0)


def test_upstream_timings_are_never_overridden(clock):
    clock.extend([0.0, 1.0, 2.0])
    real = {"predicted_n": 5, "predicted_ms": 50.0, "predicted_per_second": 100.0}
    line = _data({"choices": [], "usage": {"completion_tokens": 5}, "timings": real})
    chunks = _run([_delta(content = "x"), line])
    assert chunks[-1]["timings"] == real


def test_no_usage_means_no_timings(clock):
    clock.extend([0.0, 1.0, 2.0])
    chunks = _run([_delta(content = "x"), _delta(content = "y"), "data: [DONE]"])
    assert all("timings" not in c for c in chunks)


def test_zero_duration_has_no_rates(clock):
    # Content and usage in one chunk at the same instant as the request start.
    clock.extend([3.0, 3.0, 3.0])
    line = _data(
        {
            "choices": [{"index": 0, "delta": {"content": "x"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1},
        }
    )
    t = _run([line])[0]["timings"]
    assert t["predicted_ms"] == 0.0 and t["prompt_ms"] == 0.0
    assert "predicted_per_second" not in t and "prompt_per_second" not in t


def test_burst_reply_has_no_generation_rate(clock):
    # The whole reply arrives within 1ms (one network read): 186 tokens over 1ms is not a speed.
    clock.extend([1.0, 2.07, 2.071])
    chunks = _run(
        [
            _delta(content = "Sloths move slowly"),
            _delta(content = " because of their diet."),
            _usage(prompt_tokens = 354, completion_tokens = 186),
        ]
    )
    t = chunks[-1]["timings"]
    assert t["predicted_n"] == 186
    assert t["predicted_ms"] == pytest.approx(1.0)
    assert "predicted_per_second" not in t
    assert t["prompt_ms"] == pytest.approx(1070.0)


def test_anthropic_cache_read_and_hidden_reasoning(clock):
    clock.extend([0.0, 1.0, 3.0])
    chunks = _run(
        [
            _delta(content = "ans"),
            _delta(content = "wer"),
            _usage(
                prompt_tokens = 30,
                completion_tokens = 28,
                cache_read_input_tokens = 10,
                completion_tokens_details = {"reasoning_tokens": 8},
            ),
        ]
    )
    t = chunks[-1]["timings"]
    assert t["cache_n"] == 10 and t["prompt_n"] == 20
    # Reasoning never streamed, so it ran before the first delta and is left out of the rate.
    assert t["predicted_n"] == 20
    assert t["predicted_per_second"] == pytest.approx(10.0)


def test_streamed_reasoning_counts_as_generated(clock):
    clock.extend([0.0, 1.0, 3.0])
    chunks = _run(
        [
            _delta(reasoning_content = "think"),
            _delta(content = "answer"),
            _usage(completion_tokens = 28, completion_tokens_details = {"reasoning_tokens": 13}),
        ]
    )
    assert chunks[-1]["timings"]["predicted_n"] == 28
