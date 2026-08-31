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
    inflight = None,
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
        inflight = inflight,
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


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "NaN", "Infinity"])
def test_a_non_finite_stall_timeout_falls_back_to_the_default(monkeypatch, raw):
    """Both spellings disable the very report the variable was set to configure.

    nan loses every comparison, so max(0.0, nan) keeps 0.0 and the stall line never
    arms; inf can never be reached by an elapsed time. Neither may parse.
    """
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


def test_a_held_slot_with_nothing_of_ours_in_flight_is_visible_in_the_log(monkeypatch):
    """The second user report, and why this field exists.

    A capture showed running=1 with both rates at 0.0 for 700 seconds after the last
    /v1/chat/completions returned 200. From those four fields alone that is
    indistinguishable from a healthy long prefill, so the report could not be settled
    either way. studio_inflight=0 alongside running=1 says llama-server is holding a
    slot for work Studio has already finished; studio_inflight=1 says our own request
    is still out there and the engine is doing what it was asked.
    """
    cap = _drive(_wedged(3), monkeypatch, inflight = lambda: 0)
    stats = [kw for ev, kw in cap.events if ev == "engine_stats"]
    assert stats and all(s["studio_inflight"] == 0 for s in stats)
    assert all(s["running"] == 1 for s in stats)

    busy = _drive(_wedged(3), monkeypatch, inflight = lambda: 2)
    assert all(kw["studio_inflight"] == 2 for ev, kw in busy.events if ev == "engine_stats")


def test_the_stall_report_carries_the_in_flight_count_too(monkeypatch):
    cap = _drive(_wedged(120), monkeypatch, inflight = lambda: 0)
    assert _stalls(cap)[0]["studio_inflight"] == 0


def test_the_field_is_omitted_rather_than_zeroed_when_it_cannot_be_read(monkeypatch):
    """Omitted, never 0: a reader must not mistake "unknown" for "nothing in flight",
    which is the exact inference the field exists to support."""

    def _boom():
        raise RuntimeError("monitor unavailable")

    for hook in (None, _boom, lambda: None):
        cap = _drive(_wedged(3), monkeypatch, inflight = hook)
        stats = [kw for ev, kw in cap.events if ev == "engine_stats"]
        assert stats and all("studio_inflight" not in s for s in stats), hook


def test_a_broken_in_flight_hook_never_stops_the_poller(monkeypatch):
    def _boom():
        raise RuntimeError("monitor unavailable")

    cap = _drive(_wedged(135), monkeypatch, stall_timeout_s = 0.0, inflight = _boom)
    assert len([kw for ev, kw in cap.events if ev == "engine_stats"]) == 135


def test_the_real_hook_reads_the_api_monitor(monkeypatch):
    """Wired end to end, not just shaped like it: maybe_start_stats_logger passes the
    hook and it resolves the live monitor rather than a stub of one."""
    from core.inference.api_monitor import api_monitor

    assert ls._api_monitor_running_count() == api_monitor.active_count()
    assert isinstance(ls._api_monitor_running_count(), int)

    captured = {}

    class _Stub:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def start(self):
            pass

    monkeypatch.setattr(ls, "LlamaServerStatsLogger", _Stub)
    ls.maybe_start_stats_logger("http://127.0.0.1:0", _Capture())
    assert captured["inflight"] is ls._api_monitor_running_count


def test_the_clamped_interval_is_a_wait_every_platform_accepts():
    """The end state, exercised rather than asserted about.

    A hand-picked constant got this wrong once: a one year cap waits fine on Linux, whose
    threading.TIMEOUT_MAX is about 9.2e9 seconds, and raises OverflowError on Windows,
    where the timeout becomes a DWORD of milliseconds and the ceiling is 49.7 days. Only
    the Windows CI leg caught it, so the bound is checked here directly as well rather
    than left to whichever runner happens to be strictest.
    """
    import threading

    assert ls._MAX_ENV_SECONDS <= threading.TIMEOUT_MAX
    # The Windows ceiling, asserted on every platform so Linux cannot pass a cap Windows
    # would reject.
    assert ls._MAX_ENV_SECONDS <= (2**32 - 1) / 1000.0
    event = threading.Event()
    threading.Timer(0.05, event.set).start()
    assert event.wait(ls._MAX_ENV_SECONDS) is True


def test_a_model_load_is_not_counted_as_a_request_in_flight():
    """THE DISTINCTION THE FIELD DEPENDS ON.

    api_monitor records a model load as a lifecycle row whose status is "running". A
    plain count of running rows would therefore report studio_inflight=1 while the
    engine holds a slot for a load, which is the opposite of the reading the field
    exists to support. active_count() excludes lifecycle rows; a first cut of this
    change did not, and nothing but this test distinguished them.
    """
    from core.inference.api_monitor import ApiMonitor

    monitor = ApiMonitor(max_entries = 16)
    monitor.record_lifecycle(event = "load", model = "unsloth/Qwen3-27B-GGUF", running = True)
    assert any(e.status == "running" for e in monitor._entries), (
        "the load must be a running row, or this test proves nothing"
    )
    assert monitor.active_count() == 0

    # And through the hook itself, against the live singleton, so swapping
    # active_count() for a raw count of running rows fails here rather than only in
    # the helper above.
    from core.inference.api_monitor import api_monitor

    entry_id = api_monitor.record_lifecycle(
        event = "load", model = "unsloth/Qwen3-27B-GGUF", running = True
    )
    try:
        assert any(e.status == "running" for e in api_monitor._entries)
        assert ls._api_monitor_running_count() == 0
    finally:
        api_monitor.discard(entry_id)


def test_the_count_is_read_under_the_monitors_lock_while_it_mutates():
    """The poller reads this from a daemon thread every 10s while request threads add
    and finish rows. It must neither race the deque nor deadlock against the terminal
    callback condition, which shares the same lock."""
    import threading

    from core.inference.api_monitor import ApiMonitor

    monitor = ApiMonitor(max_entries = 64)
    stop = threading.Event()
    seen = []
    errors = []

    def churn(n):
        try:
            for i in range(300):
                entry_id = monitor.record_lifecycle(event = "load", model = f"m-{n}-{i}", running = True)
                monitor.finish(entry_id)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def poll():
        try:
            while not stop.is_set():
                seen.append(monitor.active_count())
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    reader = threading.Thread(target = poll, daemon = True)
    reader.start()
    writers = [threading.Thread(target = churn, args = (n,)) for n in range(4)]
    for w in writers:
        w.start()
    for w in writers:
        w.join(timeout = 30)
        assert not w.is_alive(), "a writer blocked on the monitor lock"
    stop.set()
    reader.join(timeout = 10)
    assert not reader.is_alive(), "active_count() deadlocked against the writers"
    assert not errors, errors
    assert seen and all(isinstance(v, int) and v >= 0 for v in seen)


def test_a_disabled_api_monitor_omits_the_count_rather_than_reporting_zero():
    """UNSLOTH_STUDIO_DISABLE_API_MONITOR makes the monitor record nothing, so
    active_count() is a constant 0 -- and 0 is the one value that asserts the slot is
    held by work Studio already finished. Every live generation would be logged as a
    leaked slot, which is the precise misreading this field exists to prevent."""
    from core.inference.api_monitor import ApiMonitor

    disabled = ApiMonitor(max_entries = 8, enabled = False)
    assert disabled.enabled is False
    assert disabled.active_count() == 0, "a disabled monitor still counts zero"

    import core.inference.api_monitor as am

    real = am.api_monitor
    am.api_monitor = disabled
    try:
        assert ls._api_monitor_running_count() is None
    finally:
        am.api_monitor = real
    assert ls._api_monitor_running_count() is not None


def test_the_count_is_process_wide_and_only_zero_is_conclusive():
    """The monitor does not record which backend a request went to, so a concurrent
    external-provider call raises this count without occupying a llama-server slot.
    Only the zero direction is load bearing, and the comment at the emit site says so
    rather than claiming the count is per engine."""
    import inspect

    src = inspect.getsource(ls.LlamaServerStatsLogger._run)
    assert "process-wide" in src
    assert "ONE direction" in src
    hook = inspect.getdoc(ls._api_monitor_running_count) or ""
    assert "Process-wide" in hook and "not per engine" in hook
