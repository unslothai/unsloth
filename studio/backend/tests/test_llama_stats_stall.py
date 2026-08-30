# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the engine stall report.

Reproduces a real user report: llama-server held one slot (requests_processing=1)
for 22.5 minutes while nothing advanced. The stats poller emitted 135 identical
`engine_stats gen_tok_s=0.0 running=1` lines at level=info and nothing else -- no
warning, no timeout, no recovery.

The signal is n_decode_total, NOT the token counters. llama-server updates
tokens_predicted_total once per generation (metrics_on_prediction, called from
callback_on_reset when the slot is released) and flushes prompt_tokens_total only
on a decode that produced output. Both therefore sit still through a healthy long
prefill and a healthy long decode, so a token-counter check would flag every slow
generation as wedged. n_decode_total increments on every llama_decode() call.

This only reports. Cancelling on this evidence is unsafe, and Studio reaps a run
that stops progressing from its own per-token event stream instead.
"""

import core.inference.llama_stats as ls
from core.inference.llama_stats import LlamaServerStatsLogger


class _Capture:
    def __init__(self):
        self.events = []

    def info(self, event, **kw):
        self.events.append((event, dict(kw)))

    def warning(self, event, **kw):
        self.events.append((event, dict(kw)))

    def debug(self, *a, **k):
        pass


def _drive(
    snaps,
    monkeypatch,
    *,
    tick_s = 10.0,
    stall_timeout_s = 600.0,
):
    """Run _run() synchronously over `snaps` on a fake clock, then stop.

    Each scrape advances the clock by `tick_s`, mirroring the 10s poll interval
    the user's log was recorded at, so a multi-minute stall costs no wall time.
    """
    cap = _Capture()
    clock = {"t": 1000.0}
    monkeypatch.setattr(ls.time, "monotonic", lambda: clock["t"])

    lg = LlamaServerStatsLogger("http://127.0.0.1:0", cap, stall_timeout_s = stall_timeout_s)
    lg._interval = 0.001  # bypass the 1s floor for a fast, synchronous run
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
    return cap


def _stalls(cap):
    return [kw for ev, kw in cap.events if ev == "engine_no_decode_progress"]


def _wedged(n, *, decode = 8192.0):
    """The user's signature: a held slot that never calls llama_decode()."""
    return [
        {
            "tokens_predicted_total": 4096.0,
            "prompt_tokens_total": 512.0,
            "n_decode_total": decode,
            "requests_processing": 1.0,
            "requests_deferred": 0.0,
        }
        for _ in range(n)
    ]


def test_reproduces_the_reported_signature(monkeypatch):
    # 135 samples, every one identical: running=1, waiting=0, both rates 0.0.
    cap = _drive(_wedged(135), monkeypatch, stall_timeout_s = 0.0)
    stats = [kw for ev, kw in cap.events if ev == "engine_stats"]
    assert len(stats) == 135
    assert all(s["running"] == 1 and s["waiting"] == 0 for s in stats)
    assert all(s["gen_tok_s"] == 0.0 and s["prompt_tok_s"] == 0.0 for s in stats)


def test_held_slot_with_no_decode_calls_is_reported(monkeypatch):
    cap = _drive(_wedged(120), monkeypatch)  # 120 x 10s = 1200s, past 600s
    stalls = _stalls(cap)
    assert len(stalls) == 1, "a stall episode must be reported exactly once"
    assert stalls[0]["running"] == 1
    assert stalls[0]["stalled_s"] >= 600.0


def test_nothing_is_ever_cancelled(monkeypatch):
    # The poller must expose no cancellation hook at all: acting on this signal
    # would kill healthy generations, which is the bug this test guards.
    lg = LlamaServerStatsLogger("http://127.0.0.1:0", _Capture())
    assert not hasattr(lg, "_on_stall")
    import inspect

    assert "on_stall" not in inspect.signature(ls.maybe_start_stats_logger).parameters


def test_a_healthy_long_decode_is_never_flagged(monkeypatch):
    # THE REGRESSION THIS FILE EXISTS FOR. llama-server does not move
    # tokens_predicted_total until the slot is released, so a generation running
    # at 5 tok/s for 20 minutes shows completely static token counters while
    # decoding normally. Only n_decode_total distinguishes it from a wedge.
    snaps = [
        {
            "tokens_predicted_total": 4096.0,  # frozen for the whole generation
            "prompt_tokens_total": 512.0,  # flushed once at first token
            "n_decode_total": 10000.0 + i,  # the engine is working
            "requests_processing": 1.0,
        }
        for i in range(120)
    ]
    cap = _drive(snaps, monkeypatch)
    assert not _stalls(cap), "a healthy long generation must never be flagged"


def test_a_healthy_long_prefill_is_never_flagged(monkeypatch):
    # Multi-batch prefill: prompt_tokens_total does not flush until a decode
    # produces output, so it too is static while the engine is busy.
    snaps = [
        {
            "tokens_predicted_total": 0.0,
            "prompt_tokens_total": 0.0,
            "n_decode_total": 500.0 + i,
            "requests_processing": 1.0,
        }
        for i in range(120)
    ]
    cap = _drive(snaps, monkeypatch)
    assert not _stalls(cap)


def test_idle_engine_is_never_flagged(monkeypatch):
    snaps = [{"n_decode_total": 9.0, "requests_processing": 0.0} for _ in range(120)]
    cap = _drive(snaps, monkeypatch)
    assert not _stalls(cap)


def test_recovery_rearms_the_report(monkeypatch):
    snaps = (
        _wedged(80)
        + [{"n_decode_total": 9000.0 + i, "requests_processing": 1.0} for i in range(5)]
        + _wedged(80, decode = 20000.0)
    )
    cap = _drive(snaps, monkeypatch)
    assert len(_stalls(cap)) == 2


def test_a_stalled_gauge_does_not_mask_a_wedge(monkeypatch):
    # predicted_tokens_seconds stuck at a stale nonzero value while nothing
    # decodes. A rate-based check would miss this.
    snaps = [
        {
            "n_decode_total": 8192.0,
            "predicted_tokens_seconds": 42.0,
            "requests_processing": 1.0,
        }
        for _ in range(120)
    ]
    cap = _drive(snaps, monkeypatch)
    assert len(_stalls(cap)) == 1


def test_absent_decode_counter_disables_reporting(monkeypatch):
    # A build whose /metrics lacks n_decode_total reads 0.0 through .get()
    # forever, indistinguishable from a wedge. Report once, then stay quiet.
    snaps = [{"requests_processing": 1.0} for _ in range(120)]
    cap = _drive(snaps, monkeypatch)
    assert not _stalls(cap)
    unmeasurable = [kw for ev, kw in cap.events if ev == "engine_progress_unmeasurable"]
    assert len(unmeasurable) == 1
    assert unmeasurable[0]["missing"] == "n_decode_total"


def test_disabled_by_zero_timeout(monkeypatch):
    cap = _drive(_wedged(120), monkeypatch, stall_timeout_s = 0.0)
    assert not _stalls(cap)


def test_counter_reset_after_reload_is_not_a_stall(monkeypatch):
    # A model reload restarts llama-server, so n_decode_total goes back to 0.
    # That is a change, so it must re-arm rather than read as progress-free.
    snaps = _wedged(20) + [
        {"n_decode_total": float(i), "requests_processing": 1.0} for i in range(5)
    ]
    cap = _drive(snaps, monkeypatch, stall_timeout_s = 600.0)
    assert not _stalls(cap), "200s of wedge then a reset must not trip the report"
