# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""engine_stats must not attribute a whole generation to the tick it was counted on.

llama-server updates tokens_predicted_total once per generation, from the
callback_on_reset installed beside slot.reset(). Dividing that count by the poll
interval reports the rate of a window the generation did not run in. prompt_tokens_total
is different and is deliberately left alone: metrics_post_decode() flushes it on every
decode step, so its delta already covers the tick it is read in.
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


def test_a_window_the_engine_never_left_reports_what_it_produced(monkeypatch):
    """The window closes when the engine goes idle, not when a counter moves.

    /metrics carries no per-slot token counter, so a release says how many tokens
    the ENGINE has produced and not which generation produced them. Under
    continuous load the honest statement is therefore the engine's throughput over
    the busy window it never left: 200 tokens in 30 busy seconds. Closing the
    window on each release instead is what lets a second, still-running generation
    be divided by the gap between two releases -- see the overlapping test below.
    """
    snaps = [
        _busy(decode = 0.0),
        _busy(predicted = 100.0, decode = 1.0),
        _busy(predicted = 100.0, decode = 2.0),
        _busy(predicted = 200.0, decode = 3.0),
    ]
    stats = _drive(snaps, monkeypatch)

    rates = [s["gen_tok_s"] for s in stats]
    # 100 tokens over one busy tick, then 200 over the three the engine stayed up.
    assert rates == [0.0, 10.0, 0.0, 6.7], rates


def test_an_idle_tick_between_two_generations_closes_the_window(monkeypatch):
    """The control for the test above: the window does still close, and the second
    generation is then measured on its own seconds alone."""
    snaps = [
        _busy(decode = 0.0),
        _busy(predicted = 100.0, decode = 1.0),
        _busy(predicted = 100.0, decode = 1.0, running = 0.0),
        _busy(predicted = 100.0, decode = 2.0),
        _busy(predicted = 200.0, decode = 3.0),
    ]
    stats = _drive(snaps, monkeypatch)

    assert [s["gen_tok_s"] for s in stats][-1] == 5.0


def test_the_second_of_two_concurrent_generations_is_not_divided_by_the_gap(
    monkeypatch,
):
    """The 183.7 tok/s record again, reached the other way.

    Two 1837-token generations run together for 80 seconds and release one poll
    apart. Discarding the busy window on the first release leaves the second with
    the 10 seconds between them, which reports the second generation at exactly the
    impossible rate this whole change exists to remove. The window is the engine's,
    so the second release is priced against the 90 seconds the engine was up and the
    3674 tokens it produced in them.
    """
    snaps = (
        [_busy(predicted = 0.0, decode = float(i), running = 2.0) for i in range(8)]
        + [_busy(predicted = 1837.0, decode = 8.0, running = 1.0)]
        + [_busy(predicted = 3674.0, decode = 9.0, running = 0.0)]
    )
    stats = _drive(snaps, monkeypatch)

    rates = [s["gen_tok_s"] for s in stats]
    assert [r for r in rates if r] == [23.0, 40.8], rates
    # What discarding the window on the first release would have said.
    assert 1837.0 / _TICK_S == 183.7


def test_the_interval_a_single_slot_finished_in_is_still_counted(monkeypatch):
    """A generation that completes between scrapes is reported by a scrape that
    shows no slot at all: llama-server has already released it. That interval is
    still the engine working -- the tokens arrived in it -- so leaving it out
    shortens the denominator and puts the rate back above the hardware ceiling.
    1837 tokens over 80 seconds is 23.0 tok/s; over 70 it is 26.2."""
    snaps = [_busy(decode = float(i)) for i in range(8)] + [
        _busy(predicted = 1837.0, decode = 8.0, running = 0.0)
    ]
    stats = _drive(snaps, monkeypatch)

    assert max(s["gen_tok_s"] for s in stats) == 23.0
    assert round(1837.0 / 70.0, 1) == 26.2


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


def test_the_prompt_counter_is_not_charged_the_decode_that_preceded_it(monkeypatch):
    """The prompt counter must NOT get the generation counter's treatment.

    The two are updated differently, which is the whole reason only one of them needs
    correcting. tokens_predicted_total is flushed once per generation, from the
    callback installed beside slot.reset(); prompt_tokens_total is flushed from
    metrics_post_decode() on every decode step. So a prompt delta already covers the
    tick it is read in, and charging it the accumulated busy time would divide a
    prefill by the decode that ran before it: measured at 33.3 tok/s for a prefill
    whose real rate is 200.

    Trace: one 2000-token prefill, five ticks of decode with the slot held and the
    prompt counter flat, then a second identical prefill."""
    snaps = [_busy(prompt = 0.0)] + [_busy(prompt = 2000.0)] * 6 + [_busy(prompt = 4000.0)]
    stats = _drive(snaps, monkeypatch)

    rates = [s["prompt_tok_s"] for s in stats]
    # 2000 tokens in one 10 s tick, both times, whatever happened in between.
    assert [r for r in rates if r] == [200.0, 200.0], rates


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
