# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""engine_stats should describe a burst, not tick every 10s through one.

A generation running for a minute produced six near-identical lines ("about 250
tok/s") plus a trailing one at running=0. Captured from a real session:

    05:50:18  gen=240.8  running=1
    05:50:28  gen=249.8  running=1
    05:50:38  gen=254.7  running=1
    05:50:48  gen=255.1  running=0

Each burst now logs its first tick, a heartbeat while it is still going, and one
closing line carrying the peak and mean of what was collapsed. Suppressed ticks
stay at debug, and --verbose restores a line per tick.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

from core.inference.llama_stats import LlamaServerStatsLogger  # noqa: E402

_VERBOSE_ENV = (
    "UNSLOTH_STUDIO_ACCESS_LOG_DEDUP_MS",
    "UNSLOTH_STUDIO_ACCESS_LOG_POLL_DEDUP_MS",
)


class _Log:
    def __init__(self):
        self.info_calls = []
        self.debug_calls = []

    def info(self, event, **kw):
        self.info_calls.append((event, kw))

    def debug(self, event, **kw):
        self.debug_calls.append((event, kw))


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in _VERBOSE_ENV:
        monkeypatch.delenv(name, raising = False)


def _logger(heartbeat = 30.0):
    log = _Log()
    return LlamaServerStatsLogger("http://127.0.0.1:1", log, 10.0, heartbeat), log


def _tick(sl, now, gen, prompt, running, waiting = 0):
    """Drive one iteration's worth of decision-making."""
    active = bool(running or waiting or gen or prompt)
    if not active:
        sl._close_burst()
        return
    starting = sl._burst is None
    if starting:
        sl._begin_burst(now)
    sl._accumulate(gen, prompt, running, waiting)
    if starting or (now - sl._burst["last_log"]) >= sl._heartbeat:
        sl._burst["last_log"] = now
        sl._log.info("engine_stats", gen_tok_s = gen, prompt_tok_s = prompt,
                     running = running, waiting = waiting)
    else:
        sl._burst["collapsed"] += 1
        sl._log.debug("engine_stats (collapsed)", gen_tok_s = gen)


def test_a_one_minute_burst_logs_two_lines_not_seven():
    sl, log = _logger()
    for i, gen in enumerate([240.8, 249.8, 254.7, 251.0, 252.0, 253.0]):
        _tick(sl, i * 10.0, gen, 2400.0, 1)
    _tick(sl, 60.0, 0.0, 0.0, 0)  # generation ends
    # First tick, one heartbeat at 30s, and the closing summary.
    assert len(log.info_calls) == 3, log.info_calls
    assert log.info_calls[-1][1]["burst_collapsed"] == 4


def test_the_closing_line_carries_peak_and_mean():
    sl, log = _logger()
    for i, gen in enumerate([100.0, 300.0, 200.0]):
        _tick(sl, i * 10.0, gen, 1000.0, 1)
    _tick(sl, 30.0, 0.0, 0.0, 0)
    summary = log.info_calls[-1][1]
    assert summary["peak_gen_tok_s"] == 300.0
    assert summary["gen_tok_s"] == pytest.approx(200.0)
    assert summary["burst_ticks"] == 3
    assert summary["peak_running"] == 1


def test_collapsed_ticks_are_still_emitted_at_debug():
    sl, log = _logger()
    # t=0 logs (burst start), t=10 and t=20 are collapsed, t=30 hits the heartbeat.
    for i in range(4):
        _tick(sl, i * 10.0, 250.0, 2400.0, 1)
    assert len(log.debug_calls) == 2, log.debug_calls
    assert len(log.info_calls) == 2, log.info_calls


def test_a_single_tick_burst_adds_no_summary():
    sl, log = _logger()
    _tick(sl, 0.0, 250.0, 2400.0, 1)
    _tick(sl, 10.0, 0.0, 0.0, 0)
    # One line for the burst, and nothing to summarise on top of it.
    assert len(log.info_calls) == 1, log.info_calls


def test_a_new_burst_starts_fresh():
    sl, log = _logger()
    for i in range(4):
        _tick(sl, i * 10.0, 250.0, 2400.0, 1)
    _tick(sl, 40.0, 0.0, 0.0, 0)
    first = len(log.info_calls)
    for i in range(4):
        _tick(sl, 100.0 + i * 10.0, 250.0, 2400.0, 1)
    _tick(sl, 140.0, 0.0, 0.0, 0)
    assert len(log.info_calls) == first * 2
    assert sl._burst is None


def test_idle_ticks_alone_log_nothing():
    sl, log = _logger()
    for i in range(5):
        _tick(sl, i * 10.0, 0.0, 0.0, 0)
    assert log.info_calls == []


def test_heartbeat_zero_restores_a_line_per_tick():
    sl, log = _logger(heartbeat = 0.0)
    for i in range(4):
        _tick(sl, i * 10.0, 250.0, 2400.0, 1)
    assert len(log.info_calls) == 4


def test_verbose_env_is_detected(monkeypatch):
    for name in _VERBOSE_ENV:
        monkeypatch.setenv(name, "0")
    sl, _ = _logger()
    assert sl._verbose is True


def test_partial_verbose_env_is_not_verbose(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_ACCESS_LOG_DEDUP_MS", "0")
    monkeypatch.setenv("UNSLOTH_STUDIO_ACCESS_LOG_POLL_DEDUP_MS", "10000")
    sl, _ = _logger()
    assert sl._verbose is False
