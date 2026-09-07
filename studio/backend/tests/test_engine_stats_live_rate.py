# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""engine_stats must not attribute work to the tick its counter moved on.

Neither token counter moves while the work happens: tokens_predicted_total is flushed
once per generation, and prompt_tokens_total only on a decode that produced output or
when every slot goes idle. So a whole generation, or a prefill spanning several polls,
lands in one scrape, and dividing it by the poll interval reports the rate of a window
the work did not run in.
Measured in the field: 48,216 engine_stats records, gen_tok_s == 0.0 in 47,629 of
them (98.77%), and two records at 150.6 and 183.7 tok/s on a 103.7 GB Q4 MoE whose
measured ceiling is 24.6 tok/s.

llama-server already measures the time itself and flushes it with the count, in the
same add_prompt() and metrics_on_prediction() calls, so the counters are read as a
pair and the poll cadence drops out of the arithmetic entirely.

The clock is faked here because the rate is the thing under test: the shared
_drive helper in test_llama_stats.py runs at a 1 ms interval on the real clock,
which cannot express a 10-second poll.
"""

from __future__ import annotations

import core.inference.llama_stats as ls
from core.inference.llama_stats import LlamaServerStatsLogger

_TICK_S = 10.0


class _Capture:
    def __init__(self):
        self.events = []

    def info(self, event, **kw):
        self.events.append((event, dict(kw)))

    def debug(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def _drive(
    snaps,
    monkeypatch,
    tick_s = _TICK_S,
):
    """Run _run() over `snaps` on a clock that advances tick_s per scrape."""
    clock = {"t": 1000.0}
    monkeypatch.setattr(ls.time, "monotonic", lambda: clock["t"])

    cap = _Capture()
    lg = LlamaServerStatsLogger("http://127.0.0.1:0", cap)
    lg._interval = 0.001  # the real sleep between ticks, not the faked elapsed time
    state = {"i": 0}

    def fake_scrape():
        i = state["i"]
        state["i"] += 1
        if i >= len(snaps):
            lg.stop()
            return None
        clock["t"] += tick_s
        return snaps[i]

    lg._scrape = fake_scrape
    lg._run()
    return [kw for ev, kw in cap.events if ev == "engine_stats"]


def _busy(
    predicted = 0.0,
    predicted_s = 0.0,
    prompt = 0.0,
    prompt_s = 0.0,
    decode = 0.0,
    running = 1.0,
    waiting = 0.0,
):
    """One scrape with no throughput gauges, so the counter path is exercised."""
    return {
        "tokens_predicted_total": predicted,
        "tokens_predicted_seconds_total": predicted_s,
        "prompt_tokens_total": prompt,
        "prompt_seconds_total": prompt_s,
        "n_decode_total": decode,
        "requests_processing": running,
        "requests_deferred": waiting,
    }


def test_a_generation_is_priced_by_the_seconds_it_reports_with_it(monkeypatch):
    """The 183.7 tok/s record. 1837 tokens appear in one scrape after 80 seconds of
    generation, and the engine says so: the honest rate is 1837/80."""
    snaps = [_busy(decode = float(i)) for i in range(8)] + [
        _busy(predicted = 1837.0, predicted_s = 80.0, decode = 8.0)
    ]
    stats = _drive(snaps, monkeypatch)

    assert max(s["gen_tok_s"] for s in stats) == 23.0
    # What the old arithmetic (count / poll interval) would have said.
    assert 1837.0 / _TICK_S == 183.7


def test_an_idle_gap_is_not_charged_to_the_generation_after_it(monkeypatch):
    """A server that sat idle for a minute and then generated 100 tokens in 20
    seconds did 5 tok/s, not 1.4: idle seconds are in no counter."""
    idle = [_busy(running = 0.0) for _ in range(6)]
    working = [_busy(decode = 1.0), _busy(predicted = 100.0, predicted_s = 20.0, decode = 2.0)]
    stats = _drive(idle + working, monkeypatch)

    assert max(s["gen_tok_s"] for s in stats) == 5.0


def test_deferred_requests_do_not_stretch_the_denominator(monkeypatch):
    """A deferred request is queued, not generating. Treating the seconds it waits as
    engine time is the mirror of the bug above and understates the rate instead.

    Trace: one tick of generation, six ticks holding a queued request with no slot
    processing, then the release. The 100 tokens took the 20 seconds the engine
    reports, whatever the queue did in between."""
    snaps = [
        _busy(decode = 1.0),
        *[_busy(running = 0.0, waiting = 1.0, decode = 1.0) for _ in range(6)],
        _busy(predicted = 100.0, predicted_s = 20.0, decode = 2.0),
    ]
    stats = _drive(snaps, monkeypatch)

    assert max(s["gen_tok_s"] for s in stats) == 5.0
    # What charging the queued ticks would have said.
    assert round(100.0 / 80.0, 1) == 1.2


def test_a_generation_already_running_at_the_first_poll_is_priced_whole(monkeypatch):
    """The logger can start mid-generation, and it has no reading from before its
    first scrape. Measuring against elapsed poll time would then omit up to a whole
    interval from the denominator and put the rate back above the ceiling; the
    engine's own seconds cover the part that happened before the poller existed."""
    snaps = [
        _busy(predicted = 200.0, predicted_s = 40.0, decode = 4.0),
        _busy(predicted = 1837.0, predicted_s = 80.0, decode = 8.0),
    ]
    stats = _drive(snaps, monkeypatch)

    # 1637 tokens in the 40 seconds between the two readings.
    assert max(s["gen_tok_s"] for s in stats) == 40.9
    assert 1637.0 / _TICK_S == 163.7


def test_two_concurrent_generations_are_not_divided_by_the_gap_between_them(monkeypatch):
    """Two 1837-token generations run together and release one poll apart. /metrics
    carries no per-slot counter, so the second release says only that the engine has
    now produced 3674 tokens across 160 generation-seconds."""
    snaps = (
        [_busy(decode = float(i), running = 2.0) for i in range(8)]
        + [_busy(predicted = 1837.0, predicted_s = 80.0, decode = 8.0, running = 1.0)]
        + [_busy(predicted = 3674.0, predicted_s = 160.0, decode = 9.0, running = 0.0)]
    )
    stats = _drive(snaps, monkeypatch)

    assert [r for r in (s["gen_tok_s"] for s in stats) if r] == [23.0, 23.0]
    # What dividing the second release by the gap would have said.
    assert 1837.0 / _TICK_S == 183.7


def test_a_long_prefill_is_not_attributed_to_the_tick_it_flushed_on(monkeypatch):
    """The prompt counter needs the same treatment as the generation counter.

    llama-server flushes it only on a decode that produced output, so a prefill
    spanning many polls stays flat and then arrives whole: 130k tokens on one tick
    reads as 13,000 tok/s against a real 200. The seconds counter is flushed by the
    same call, so the pair is the prefill's own rate."""
    snaps = [_busy(prompt = 0.0)] + [_busy() for _ in range(64)] + [
        _busy(prompt = 130000.0, prompt_s = 650.0)
    ]
    stats = _drive(snaps, monkeypatch)

    assert max(s["prompt_tok_s"] for s in stats) == 200.0
    assert 130000.0 / _TICK_S == 13000.0


def test_the_decode_counter_reports_while_the_token_counters_are_still(monkeypatch):
    """The reason the line read 0 for 98.8% of the time: both token counters sit
    still through a healthy generation. n_decode_total moves on every
    llama_decode(), so it is the live signal, and it is reported as calls rather
    than as tokens, over the tick it moved in."""
    snaps = [_busy(decode = float(i * 20)) for i in range(4)]
    stats = _drive(snaps, monkeypatch)

    assert all(s["gen_tok_s"] == 0.0 for s in stats)
    assert [s["decode_calls_s"] for s in stats] == [0.0, 2.0, 2.0, 2.0]


def test_a_build_without_the_seconds_counters_reports_no_rate_rather_than_one(monkeypatch):
    """Nothing in /metrics then says how long the tokens took, and the poll interval
    is not an answer. The line still goes out, carrying what is measurable."""
    snaps = [
        {"tokens_predicted_total": 0.0, "prompt_tokens_total": 0.0, "requests_processing": 1.0},
        {"tokens_predicted_total": 1837.0, "prompt_tokens_total": 0.0, "requests_processing": 1.0},
    ]
    stats = _drive(snaps, monkeypatch)

    assert stats and all(s["gen_tok_s"] == 0.0 for s in stats)


def test_the_llama_cpp_gauge_still_wins_when_it_reports(monkeypatch):
    """predicted_tokens_seconds is llama.cpp's own per-generation average. Where it
    is present and non-zero it is authoritative, and this change does not touch it."""
    snaps = [
        {
            "tokens_predicted_total": 0.0,
            "prompt_tokens_total": 0.0,
            "predicted_tokens_seconds": 24.6,
            "requests_processing": 1.0,
        },
        {
            "tokens_predicted_total": 1837.0,
            "tokens_predicted_seconds_total": 80.0,
            "prompt_tokens_total": 0.0,
            "predicted_tokens_seconds": 24.6,
            "requests_processing": 1.0,
        },
    ]
    stats = _drive(snaps, monkeypatch)

    assert all(s["gen_tok_s"] == 24.6 for s in stats)
