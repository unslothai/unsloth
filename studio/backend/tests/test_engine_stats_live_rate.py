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

The two sides are not symmetric, and the tests here follow the source rather than the
symmetry. add_prompt(n, n, t_us) counts every prompt token as a decode step, so the
prompt counters are a matched pair and their ratio is the prefill's own rate.
metrics_on_prediction() passes n_gen and n_gen - 1, because the first generated token
comes from the prompt batch for free, so the generation counters are NOT a pair -- a
one-token completion is 1 token against ~0 seconds. Generation throughput therefore
comes from llama.cpp's own gauge or from nowhere.

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
    gen_gauge = None,
):
    """One scrape. No throughput gauges unless asked, so the counter path is exercised."""
    snap = {
        "tokens_predicted_total": predicted,
        "tokens_predicted_seconds_total": predicted_s,
        "prompt_tokens_total": prompt,
        "prompt_seconds_total": prompt_s,
        "n_decode_total": decode,
        "requests_processing": running,
        "requests_deferred": waiting,
    }
    if gen_gauge is not None:
        snap["predicted_tokens_seconds"] = gen_gauge
    return snap


def test_a_prefill_is_priced_by_the_seconds_it_reports_with_it(monkeypatch):
    """1837 prompt tokens appear in one scrape after 80 seconds of prefill, and the
    engine says so: the honest rate is 1837/80, not 1837 over the poll interval."""
    snaps = [_busy(decode = float(i)) for i in range(8)] + [
        _busy(prompt = 1837.0, prompt_s = 80.0, decode = 8.0)
    ]
    stats = _drive(snaps, monkeypatch)

    assert max(s["prompt_tok_s"] for s in stats) == 23.0
    # What the old arithmetic (count / poll interval) would have said.
    assert 1837.0 / _TICK_S == 183.7


def test_an_idle_gap_is_not_charged_to_the_prefill_after_it(monkeypatch):
    """A server that sat idle for a minute and then processed 100 prompt tokens in 20
    seconds did 5 tok/s, not 1.4: idle seconds are in no counter."""
    idle = [_busy(running = 0.0) for _ in range(6)]
    working = [_busy(decode = 1.0), _busy(prompt = 100.0, prompt_s = 20.0, decode = 2.0)]
    stats = _drive(idle + working, monkeypatch)

    assert max(s["prompt_tok_s"] for s in stats) == 5.0


def test_deferred_requests_do_not_stretch_the_denominator(monkeypatch):
    """A deferred request is queued, not running. Treating the seconds it waits as engine
    time is the mirror of the bug above and understates the rate instead.

    Trace: one tick of work, six ticks holding a queued request with no slot processing,
    then the flush. The 100 tokens took the 20 seconds the engine reports, whatever the
    queue did in between."""
    snaps = [
        _busy(decode = 1.0),
        *[_busy(running = 0.0, waiting = 1.0, decode = 1.0) for _ in range(6)],
        _busy(prompt = 100.0, prompt_s = 20.0, decode = 2.0),
    ]
    stats = _drive(snaps, monkeypatch)

    assert max(s["prompt_tok_s"] for s in stats) == 5.0
    # What charging the queued ticks would have said.
    assert round(100.0 / 80.0, 1) == 1.2


def test_work_already_running_at_the_first_poll_is_priced_whole(monkeypatch):
    """The logger can start mid-prefill, and it has no reading from before its first
    scrape. Measuring against elapsed poll time would then omit up to a whole interval
    from the denominator; the engine's own seconds cover the part that happened before
    the poller existed."""
    snaps = [
        _busy(prompt = 200.0, prompt_s = 40.0, decode = 4.0),
        _busy(prompt = 1837.0, prompt_s = 80.0, decode = 8.0),
    ]
    stats = _drive(snaps, monkeypatch)

    # 1637 tokens in the 40 seconds between the two readings.
    assert max(s["prompt_tok_s"] for s in stats) == 40.9
    assert 1637.0 / _TICK_S == 163.7


def test_a_long_prefill_is_not_attributed_to_the_tick_it_flushed_on(monkeypatch):
    """llama-server flushes the prompt counter only on a decode that produced output, so
    a prefill spanning many polls stays flat and then arrives whole: 130k tokens on one
    tick reads as 13,000 tok/s against a real 200. The seconds counter is flushed by the
    same call, so the pair is the prefill's own rate."""
    snaps = (
        [_busy(prompt = 0.0)]
        + [_busy() for _ in range(64)]
        + [_busy(prompt = 130000.0, prompt_s = 650.0)]
    )
    stats = _drive(snaps, monkeypatch)

    assert max(s["prompt_tok_s"] for s in stats) == 200.0
    assert 130000.0 / _TICK_S == 13000.0


def test_the_decode_counter_reports_while_the_token_counters_are_still(monkeypatch):
    """The reason the line read 0 for 98.8% of the time: both token counters sit still
    through a healthy generation. n_decode_total moves on every llama_decode(), so it is
    the live signal, and it is reported as calls rather than as tokens, over the tick it
    moved in."""
    snaps = [_busy(decode = float(i * 20), gen_gauge = 0.0) for i in range(4)]
    stats = _drive(snaps, monkeypatch)

    assert all(s["gen_tok_s"] == 0.0 for s in stats)
    # Nothing on the first line: one sample is not a rate.
    assert [s.get("decode_calls_s") for s in stats] == [None, 2.0, 2.0, 2.0]


def test_a_build_without_the_decode_counter_omits_the_field(monkeypatch):
    """n_decode_total is not on every llama-server. Printing 0.0 there would state the
    engine was measured making no calls, which is the opposite of what
    engine_progress_unmeasurable says about the same build."""
    snaps = [
        {"tokens_predicted_total": 0.0, "prompt_tokens_total": 0.0, "requests_processing": 1.0},
        {"tokens_predicted_total": 50.0, "prompt_tokens_total": 0.0, "requests_processing": 1.0},
    ]
    stats = _drive(snaps, monkeypatch)

    assert stats and all("decode_calls_s" not in s for s in stats)


def test_a_zero_gauge_is_a_reading_and_not_a_missing_one(monkeypatch):
    """A one-token completion. llama-server's gauge numerator is the decode steps, which
    excludes the token produced from the prompt batch, so it reports 0 for that request.
    The counters include the free token and its generation time is near zero, so falling
    through to them would divide one token by a millisecond."""
    snaps = [
        _busy(gen_gauge = 0.0),
        _busy(predicted = 1.0, predicted_s = 0.0001, gen_gauge = 0.0),
    ]
    stats = _drive(snaps, monkeypatch)

    assert stats and all(s["gen_tok_s"] == 0.0 for s in stats)
    # What reading the zero as absent would have said.
    assert 1.0 / 0.0001 == 10000.0


def test_a_zero_gauge_beside_moved_counters_is_still_a_reading(monkeypatch):
    """Another client scraping /metrics between polls empties the throughput bucket too,
    so Studio can see a zero gauge for a window that did carry work. That reading is
    indistinguishable from the test above -- a completion whose only token was free moves
    the counters and reports zero for the same reason -- and the counters cannot break the
    tie, since they credit each generation with that free token. Understating a window
    another scraper took is the cost of never fabricating one."""
    snaps = [
        _busy(predicted = 100.0, predicted_s = 5.0, gen_gauge = 20.0),
        _busy(predicted = 300.0, predicted_s = 15.0, gen_gauge = 0.0),
        _busy(predicted = 400.0, predicted_s = 20.0, gen_gauge = 20.0),
    ]
    stats = _drive(snaps, monkeypatch)

    assert [s["gen_tok_s"] for s in stats] == [20.0, 0.0, 20.0]
    # What filling the gap from the counters would have said.
    assert 200.0 / 10.0 == 20.0


def test_tokens_with_no_seconds_yet_are_kept_for_the_tick_that_brings_them(monkeypatch):
    """/metrics renders doubles at six significant digits, so a long-lived server can
    resolve a short request in the token total while the seconds total still rounds to
    the same value. Dropping the numerator there would charge those tokens to whatever
    seconds arrive next; the baseline is held until a usable pair exists.

    Trace: 100 tokens arrive against an unchanged seconds total, then 100 more with the
    20 seconds that produced all 200."""
    snaps = [
        _busy(prompt = 100.0, prompt_s = 1000.0),
        _busy(prompt = 200.0, prompt_s = 1000.0),
        _busy(prompt = 300.0, prompt_s = 1020.0),
    ]
    stats = _drive(snaps, monkeypatch)

    assert [s["prompt_tok_s"] for s in stats] == [0.0, 0.0, 10.0]


def test_seconds_that_resolve_before_their_tokens_are_kept_too(monkeypatch):
    """The same rounding boundary, crossed the other way round. Advancing the baseline on
    seconds alone discards them, and the tokens that follow are then divided by only the
    newer time -- the inflated rate again, from the opposite direction.

    Trace: 20 seconds arrive against an unchanged token total, then the 100 tokens they
    produced, then 100 more in 20 seconds of their own."""
    snaps = [
        _busy(prompt = 100.0, prompt_s = 1000.0),
        _busy(prompt = 100.0, prompt_s = 1020.0),
        _busy(prompt = 200.0, prompt_s = 1020.0),
        _busy(prompt = 300.0, prompt_s = 1040.0),
    ]
    stats = _drive(snaps, monkeypatch)

    assert [s["prompt_tok_s"] for s in stats] == [0.0, 0.0, 5.0, 5.0]
    # What discarding the held seconds would have said for the third tick.
    assert 100.0 / 20.0 == 5.0


def test_a_build_without_the_generation_gauge_omits_the_field(monkeypatch):
    """Nothing else in /metrics says how long the generated tokens took: the seconds
    counter times n_gen - 1 steps while the token counter counts n_gen, so their ratio is
    not a rate. The line still goes out, carrying what is measurable."""
    snaps = [
        _busy(predicted = 0.0),
        _busy(predicted = 1.0, predicted_s = 0.0001),
    ]
    stats = _drive(snaps, monkeypatch)

    assert stats and all("gen_tok_s" not in s for s in stats)
    # What the counter ratio would have said for that one-token completion.
    assert 1.0 / 0.0001 == 10000.0


def test_the_llama_cpp_gauge_still_wins_when_it_reports(monkeypatch):
    """predicted_tokens_seconds is llama.cpp's own per-generation average. Where it is
    present and non-zero it is authoritative, and this change does not touch it."""
    snaps = [
        _busy(gen_gauge = 24.6),
        _busy(predicted = 1837.0, predicted_s = 80.0, gen_gauge = 24.6),
    ]
    stats = _drive(snaps, monkeypatch)

    assert stats and all(s["gen_tok_s"] == 24.6 for s in stats)
