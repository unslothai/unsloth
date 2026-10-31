# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stall report uses n_decode_total, since token counters freeze during healthy prefill and decode."""

import pytest

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
    """Runs the poller over snaps on a fake clock, advancing tick_s per scrape instead of sleeping."""
    cap = _Capture()
    clock = {"t": 1000.0}
    monkeypatch.setattr(ls.time, "monotonic", lambda: clock["t"])

    lg = LlamaServerStatsLogger("http://127.0.0.1:0", cap, stall_timeout_s = stall_timeout_s)
    lg._interval = 0.001  # bypass the 1s floor
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
    """The user's signature: a held slot that never calls llama_decode().

    The gauges are in the snapshot because /metrics renders them from the same table."""
    return [
        {
            "tokens_predicted_total": 4096.0,
            "prompt_tokens_total": 512.0,
            "predicted_tokens_seconds": 0.0,
            "prompt_tokens_seconds": 0.0,
            "n_decode_total": decode,
            "requests_processing": 1.0,
            "requests_deferred": 0.0,
        }
        for _ in range(n)
    ]


def test_reproduces_the_reported_signature(monkeypatch):
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
    # No cancellation hook: acting on this signal would kill healthy generations.
    lg = LlamaServerStatsLogger("http://127.0.0.1:0", _Capture())
    assert not hasattr(lg, "_on_stall")
    import inspect

    assert "on_stall" not in inspect.signature(ls.maybe_start_stats_logger).parameters


def test_a_healthy_long_decode_is_never_flagged(monkeypatch):
    # tokens_predicted_total is frozen until slot release; only n_decode_total moves.
    snaps = [
        {
            "tokens_predicted_total": 4096.0,
            "prompt_tokens_total": 512.0,
            "n_decode_total": 10000.0 + i,
            "requests_processing": 1.0,
        }
        for i in range(120)
    ]
    cap = _drive(snaps, monkeypatch)
    assert not _stalls(cap), "a healthy long generation must never be flagged"


def test_a_healthy_long_prefill_is_never_flagged(monkeypatch):
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
    # Missing n_decode_total reads 0.0 forever; report once, then stay quiet.
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
    snaps = _wedged(20) + [
        {"n_decode_total": float(i), "requests_processing": 1.0} for i in range(5)
    ]
    cap = _drive(snaps, monkeypatch, stall_timeout_s = 600.0)
    assert not _stalls(cap), "200s of wedge then a reset must not trip the report"


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "NaN", "Infinity"])
def test_a_non_finite_stall_timeout_falls_back_to_the_default(monkeypatch, raw):
    """Non-finite stall timeouts (nan, inf) must fall back to the default, or the report never arms."""
    monkeypatch.setenv("UNSLOTH_STUDIO_ENGINE_STALL_TIMEOUT_S", raw)
    cap = _Capture()
    assert ls._env_float("UNSLOTH_STUDIO_ENGINE_STALL_TIMEOUT_S", 600.0, cap) == 600.0
    assert [ev for ev, _ in cap.events] == ["engine_stats_env_ignored"]


@pytest.mark.parametrize("raw", ["nan", "inf"])
def test_a_non_finite_interval_falls_back_to_the_default(monkeypatch, raw):
    """An infinite interval parks the poll loop in stop.wait() until shutdown."""
    monkeypatch.setenv("UNSLOTH_STUDIO_ENGINE_STATS_INTERVAL_S", raw)
    assert ls._env_float("UNSLOTH_STUDIO_ENGINE_STATS_INTERVAL_S", 10.0, _Capture()) == 10.0


def test_a_non_finite_timeout_still_leaves_the_stall_report_armed(monkeypatch):
    """The end state: a malformed setting must not silence the wedge report."""
    monkeypatch.setenv("UNSLOTH_STUDIO_ENGINE_STALL_TIMEOUT_S", "nan")
    applied = ls._env_float("UNSLOTH_STUDIO_ENGINE_STALL_TIMEOUT_S", 600.0, _Capture())
    cap = _drive(_wedged(120), monkeypatch, stall_timeout_s = applied)
    assert _stalls(cap), "a nan timeout must not disable the stall report"


def test_a_garbage_timeout_still_falls_back_without_warning(monkeypatch):
    """Unparseable text was already handled; it must stay quiet, not warn."""
    monkeypatch.setenv("UNSLOTH_STUDIO_ENGINE_STALL_TIMEOUT_S", "not-a-number")
    cap = _Capture()
    assert ls._env_float("UNSLOTH_STUDIO_ENGINE_STALL_TIMEOUT_S", 600.0, cap) == 600.0
    assert cap.events == []


@pytest.mark.parametrize("raw", ["1e10", "1e18", "1e308"])
def test_an_oversized_finite_interval_is_clamped(monkeypatch, raw):
    """Event.wait() turns a timeout into an absolute deadline, and one far enough out
    raises "timestamp out of range for platform time_t" as soon as the wait is entered,
    killing the poll thread and taking the stats and the stall report with it."""
    monkeypatch.setenv("UNSLOTH_STUDIO_ENGINE_STATS_INTERVAL_S", raw)
    cap = _Capture()
    applied = ls._env_float("UNSLOTH_STUDIO_ENGINE_STATS_INTERVAL_S", 10.0, cap)
    assert applied == ls._MAX_ENV_SECONDS
    assert [ev for ev, _ in cap.events] == ["engine_stats_env_clamped"]


def test_the_clamped_interval_is_a_wait_every_platform_accepts():
    """Clamped stall interval must stay under threading.TIMEOUT_MAX; the Windows ceiling is 49.7 days."""
    import threading

    assert ls._MAX_ENV_SECONDS <= threading.TIMEOUT_MAX
    # Windows ceiling asserted everywhere so Linux cannot pass a cap Windows rejects.
    assert ls._MAX_ENV_SECONDS <= (2**32 - 1) / 1000.0
    event = threading.Event()
    threading.Timer(0.05, event.set).start()
    assert event.wait(ls._MAX_ENV_SECONDS) is True
