# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the engine stall watchdog.

Reproduces a real user report: llama-server held one slot (requests_processing=1)
for 22.5 minutes while tokens_predicted_total never advanced once. The stats
poller emitted 135 identical `engine_stats gen_tok_s=0.0 running=1` lines at
level=info and nothing else -- no warning, no timeout, no recovery. The UI showed
an eternal spinner and an external API client gave up.

Progress here is measured on the cumulative counters, never the rate gauges: a
gauge can read stale-but-nonzero while decode is wedged, which is the exact case
being caught.
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
    stall_timeout_s = 180.0,
    on_stall = None,
):
    """Run _run() synchronously over `snaps` on a fake clock, then stop.

    Each scrape advances the clock by `tick_s`, mirroring the 10s poll interval
    the user's log was recorded at, so a multi-minute stall costs no wall time.
    """
    cap = _Capture()
    clock = {"t": 1000.0}
    monkeypatch.setattr(ls.time, "monotonic", lambda: clock["t"])

    lg = LlamaServerStatsLogger(
        "http://127.0.0.1:0",
        cap,
        stall_timeout_s = stall_timeout_s,
        on_stall = on_stall,
    )
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


def _wedged(
    n,
    *,
    predicted = 4096.0,
    prompt = 512.0,
):
    """The user's signature: a held slot whose counters never move."""
    return [
        {
            "tokens_predicted_total": predicted,
            "prompt_tokens_total": prompt,
            "requests_processing": 1.0,
            "requests_deferred": 0.0,
        }
        for _ in range(n)
    ]


def test_reproduces_the_reported_signature():
    # 135 samples, every one identical: running=1, waiting=0, both rates 0.0.
    # This is what the user's 22.5 minutes of log looked like.
    import pytest

    cap = _drive(_wedged(135), pytest.MonkeyPatch(), stall_timeout_s = 0.0)
    stats = [kw for ev, kw in cap.events if ev == "engine_stats"]
    assert len(stats) == 135
    assert all(s["running"] == 1 and s["waiting"] == 0 for s in stats)
    assert all(s["gen_tok_s"] == 0.0 and s["prompt_tok_s"] == 0.0 for s in stats)


def test_held_slot_with_static_counters_is_reported_and_reaped(monkeypatch):
    reaped = []
    cap = _drive(
        _wedged(60),  # 60 ticks x 10s = 600s, well past the 180s threshold
        monkeypatch,
        on_stall = lambda **kw: reaped.append(kw),
    )
    stalls = [kw for ev, kw in cap.events if ev == "engine_stall_detected"]
    assert len(stalls) == 1, "a stall episode must be reported exactly once"
    assert stalls[0]["running"] == 1
    assert stalls[0]["stalled_s"] >= 180.0
    assert len(reaped) == 1, "the reap callback must fire once per stall episode"
    assert reaped[0]["running"] == 1


def test_slow_but_progressing_generation_is_never_reaped(monkeypatch):
    # One token every 10s is legitimate on CPU. It must never be reaped, however
    # long it runs -- progress, not speed, is the test.
    snaps = [
        {
            "tokens_predicted_total": float(i),
            "prompt_tokens_total": 512.0,
            "requests_processing": 1.0,
        }
        for i in range(120)
    ]
    reaped = []
    cap = _drive(snaps, monkeypatch, on_stall = lambda **kw: reaped.append(kw))
    assert not [ev for ev, _ in cap.events if ev == "engine_stall_detected"]
    assert not reaped


def test_long_prefill_is_not_mistaken_for_a_stall(monkeypatch):
    # Prompt tokens advancing with zero generated tokens is prefill, not a wedge.
    snaps = [
        {
            "tokens_predicted_total": 0.0,
            "prompt_tokens_total": float(i * 128),
            "requests_processing": 1.0,
        }
        for i in range(120)
    ]
    reaped = []
    cap = _drive(snaps, monkeypatch, on_stall = lambda **kw: reaped.append(kw))
    assert not [ev for ev, _ in cap.events if ev == "engine_stall_detected"]
    assert not reaped


def test_idle_engine_is_never_reaped(monkeypatch):
    # No slot held: static counters are simply idleness.
    snaps = [
        {"tokens_predicted_total": 9.0, "prompt_tokens_total": 9.0, "requests_processing": 0.0}
        for _ in range(120)
    ]
    reaped = []
    cap = _drive(snaps, monkeypatch, on_stall = lambda **kw: reaped.append(kw))
    assert not [ev for ev, _ in cap.events if ev == "engine_stall_detected"]
    assert not reaped


def test_recovery_rearms_the_watchdog(monkeypatch):
    # Stall, then progress, then stall again: two distinct episodes.
    snaps = (
        _wedged(40)
        + [
            {
                "tokens_predicted_total": 5000.0 + i,
                "prompt_tokens_total": 512.0,
                "requests_processing": 1.0,
            }
            for i in range(5)
        ]
        + _wedged(40, predicted = 9000.0)
    )
    reaped = []
    cap = _drive(snaps, monkeypatch, on_stall = lambda **kw: reaped.append(kw))
    assert len([ev for ev, _ in cap.events if ev == "engine_stall_detected"]) == 2
    assert len(reaped) == 2


def test_stalled_gauge_does_not_mask_a_wedge(monkeypatch):
    # predicted_tokens_seconds stuck at a stale nonzero value while the counter
    # is frozen. Rate-based detection would miss this; counter-based catches it.
    snaps = [
        {
            "tokens_predicted_total": 4096.0,
            "prompt_tokens_total": 512.0,
            "predicted_tokens_seconds": 42.0,
            "requests_processing": 1.0,
        }
        for _ in range(60)
    ]
    reaped = []
    _drive(snaps, monkeypatch, on_stall = lambda **kw: reaped.append(kw))
    assert len(reaped) == 1


def test_disabled_by_zero_timeout(monkeypatch):
    reaped = []
    cap = _drive(
        _wedged(60), monkeypatch, stall_timeout_s = 0.0, on_stall = lambda **kw: reaped.append(kw)
    )
    assert not [ev for ev, _ in cap.events if ev == "engine_stall_detected"]
    assert not reaped


def test_absent_counters_disable_the_watchdog_instead_of_reaping(monkeypatch):
    # A build whose /metrics lacks the token counters reads 0.0 through .get()
    # forever, which looks exactly like a wedge. Reaping there would kill healthy
    # generations, so it must report once and never reap.
    snaps = [{"requests_processing": 1.0, "requests_deferred": 0.0} for _ in range(60)]
    reaped = []
    cap = _drive(snaps, monkeypatch, on_stall = lambda **kw: reaped.append(kw))
    assert not reaped, "must never reap when progress is unmeasurable"
    assert not [ev for ev, _ in cap.events if ev == "engine_stall_detected"]
    unmeasurable = [kw for ev, kw in cap.events if ev == "engine_progress_unmeasurable"]
    assert len(unmeasurable) == 1, "reported exactly once, not every tick"
    assert unmeasurable[0]["missing"] == ["prompt_tokens_total", "tokens_predicted_total"]


def test_a_raising_callback_does_not_kill_the_poller(monkeypatch):
    # The daemon thread must survive a bad reap hook; stats must keep flowing.
    def boom(**kw):
        raise RuntimeError("reap failed")

    cap = _drive(_wedged(60), monkeypatch, on_stall = boom)
    assert [ev for ev, _ in cap.events if ev == "engine_stall_detected"]
    stats = [kw for ev, kw in cap.events if ev == "engine_stats"]
    assert len(stats) == 60, "poller must keep emitting after a failed reap"
