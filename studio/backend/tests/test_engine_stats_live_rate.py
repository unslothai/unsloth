# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""engine_stats must not attribute a whole generation to the tick it was counted on.

llama-server updates tokens_predicted_total and prompt_tokens_total once per
generation, from callback_on_reset when the slot is released. Dividing that count
by the poll interval reports the rate of a window the generation did not run in.
Measured in the field: 48,216 engine_stats records, gen_tok_s == 0.0 in 47,629 of
them (98.77%), and two records at 150.6 and 183.7 tok/s on a 103.7 GB Q4 MoE whose
measured ceiling is 24.6 tok/s.

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
    prompt = 0.0,
    decode = 0.0,
    running = 1.0,
):
    """One scrape with no throughput gauges, so the counter path is exercised."""
    return {
        "tokens_predicted_total": predicted,
        "prompt_tokens_total": prompt,
        "n_decode_total": decode,
        "requests_processing": running,
    }


def test_a_generation_is_not_attributed_to_the_tick_it_was_counted_on(monkeypatch):
    """The 183.7 tok/s record. 1837 tokens appear in one scrape after 70 seconds of
    a held slot, so the honest rate is 1837/80, not 1837/10."""
    snaps = [_busy(decode = float(i)) for i in range(8)] + [_busy(predicted = 1837.0, decode = 8.0)]
    stats = _drive(snaps, monkeypatch)

    reported = max(s["gen_tok_s"] for s in stats)
    assert reported == 23.0, reported
    # What the old arithmetic (count / poll interval) would have said.
    assert 1837.0 / _TICK_S == 183.7


def test_an_idle_gap_is_not_charged_to_the_generation_after_it(monkeypatch):
    """Busy seconds, not wall seconds: a server that sat idle for a minute and then
    generated 100 tokens in 20 seconds did 5 tok/s, not 1.4."""
    idle = [_busy(running = 0.0) for _ in range(6)]
    working = [_busy(decode = 1.0), _busy(predicted = 100.0, decode = 2.0)]
    stats = _drive(idle + working, monkeypatch)

    assert max(s["gen_tok_s"] for s in stats) == 5.0


def test_two_generations_back_to_back_each_get_their_own_window(monkeypatch):
    """The window resets when the counter moves, so the second generation is not
    credited with the first one's seconds."""
    snaps = [
        _busy(decode = 0.0),
        _busy(predicted = 100.0, decode = 1.0),
        _busy(predicted = 100.0, decode = 2.0),
        _busy(predicted = 200.0, decode = 3.0),
    ]
    stats = _drive(snaps, monkeypatch)

    rates = [s["gen_tok_s"] for s in stats]
    # 100 tokens over one busy tick, then 100 over two.
    assert rates == [0.0, 10.0, 0.0, 5.0], rates


def test_the_decode_counter_reports_while_the_token_counters_are_still(monkeypatch):
    """The reason the line read 0 for 98.8% of the time: both token counters sit
    still through a healthy generation. n_decode_total moves on every
    llama_decode(), so it is the live signal, and it is reported as calls rather
    than as tokens."""
    snaps = [_busy(decode = float(i * 20)) for i in range(4)]
    stats = _drive(snaps, monkeypatch)

    assert all(s["gen_tok_s"] == 0.0 for s in stats)
    assert [s["decode_calls_s"] for s in stats] == [0.0, 2.0, 2.0, 2.0]


def test_a_build_without_the_decode_counter_still_reports(monkeypatch):
    """n_decode_total is not on every llama-server. Its absence must not stop the
    line or fabricate a rate."""
    snaps = [
        {"tokens_predicted_total": 0.0, "prompt_tokens_total": 0.0, "requests_processing": 1.0},
        {"tokens_predicted_total": 50.0, "prompt_tokens_total": 0.0, "requests_processing": 1.0},
    ]
    stats = _drive(snaps, monkeypatch)

    assert stats
    assert all(s["decode_calls_s"] == 0.0 for s in stats)
    assert max(s["gen_tok_s"] for s in stats) == 5.0


def test_a_prompt_counter_gets_the_same_treatment(monkeypatch):
    """prompt_tokens_total flushes only on a decode that produced output, so a long
    prefill lands in one tick the same way."""
    snaps = [_busy(decode = float(i)) for i in range(5)] + [_busy(prompt = 9000.0, decode = 5.0)]
    stats = _drive(snaps, monkeypatch)

    # 9000 tokens over the 50 seconds of held slot this poller actually observed.
    # The first scrape sets the baseline and contributes no elapsed time, which is
    # the honest floor: nothing before it was measured.
    assert max(s["prompt_tok_s"] for s in stats) == 180.0


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
            "prompt_tokens_total": 0.0,
            "predicted_tokens_seconds": 24.6,
            "requests_processing": 1.0,
        },
    ]
    stats = _drive(snaps, monkeypatch)

    assert all(s["gen_tok_s"] == 24.6 for s in stats)
