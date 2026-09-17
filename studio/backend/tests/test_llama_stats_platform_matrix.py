# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Simulation suite: the engine stats poller across platforms, backends and bad input.

The poller reads llama-server /metrics over loopback HTTP and nothing else, so the
platform and GPU axes reduce to "what does this build's /metrics look like". These
walk the [Windows, Linux, WSL, macOS] x [NVIDIA, AMD/ROCm, Vulkan, CPU] product as
metric payloads, plus the malformed and hostile bodies a daemon thread must survive
without dying.
"""

from __future__ import annotations

import itertools
import sys
import threading
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import core.inference.llama_stats as ls  # noqa: E402
from core.inference.llama_stats import LlamaServerStatsLogger  # noqa: E402


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
    cap = _Capture()
    clock = {"t": 1000.0}
    monkeypatch.setattr(ls.time, "monotonic", lambda: clock["t"])
    lg = LlamaServerStatsLogger("http://127.0.0.1:0", cap, stall_timeout_s = stall_timeout_s)
    lg._interval = 0.001
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


# ------------------------------------------------------- platform x backend matrix

PLATFORMS = ["win32", "linux", "linux-wsl", "darwin"]
BACKENDS = ["cuda", "rocm", "vulkan", "cpu"]


@pytest.mark.parametrize("platform,backend", list(itertools.product(PLATFORMS, BACKENDS)))
def test_healthy_generation_is_never_flagged_on_any_platform(platform, backend, monkeypatch):
    """The signature that matters is identical everywhere: llama-server freezes the
    token counters for the whole of a healthy generation and only n_decode_total moves.

    A build that reported a stall here would cancel real work on that platform.
    """
    monkeypatch.setattr(sys, "platform", platform.replace("-wsl", ""))
    snaps = [
        {
            "tokens_predicted_total": 4096.0,  # frozen until the slot is released
            "prompt_tokens_total": 512.0,  # flushed once, at the first token
            "n_decode_total": 30000.0 + i,
            "requests_processing": 1.0,
            "requests_deferred": 0.0,
        }
        for i in range(180)  # 30 simulated minutes
    ]
    cap = _drive(snaps, monkeypatch)
    assert not _stalls(cap), f"{platform}/{backend} flagged a healthy generation"


@pytest.mark.parametrize("platform,backend", list(itertools.product(PLATFORMS, BACKENDS)))
def test_a_real_wedge_is_reported_on_any_platform(platform, backend, monkeypatch):
    monkeypatch.setattr(sys, "platform", platform.replace("-wsl", ""))
    snaps = [
        {
            "tokens_predicted_total": 4096.0,
            "prompt_tokens_total": 512.0,
            "n_decode_total": 30000.0,  # frozen: the engine is not decoding
            "requests_processing": 1.0,
        }
        for _ in range(120)
    ]
    cap = _drive(snaps, monkeypatch)
    assert len(_stalls(cap)) == 1, f"{platform}/{backend} missed a wedge"


def test_cpu_only_slow_decode_is_never_flagged(monkeypatch):
    # A CPU build can sit many seconds between decode calls. As long as the counter
    # moves at all between scrapes, it is alive.
    snaps = []
    decode = 100.0
    for i in range(240):
        if i % 20 == 0:  # a decode call only every 200 simulated seconds
            decode += 1
        snaps.append({"n_decode_total": decode, "requests_processing": 1.0})
    cap = _drive(snaps, monkeypatch)
    assert not _stalls(cap)


# ------------------------------------------------------------------ scrape robustness


def _scrape_body(body, monkeypatch):
    class _Resp:
        status = 200

        def read(self):
            return body.encode() if isinstance(body, str) else body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(ls.urllib.request, "urlopen", lambda *a, **k: _Resp())
    return LlamaServerStatsLogger("http://127.0.0.1:0", _Capture())._scrape()


def test_scrape_handles_windows_line_endings(monkeypatch):
    body = "llamacpp:n_decode_total 12\r\nllamacpp:requests_processing 1\r\n"
    m = _scrape_body(body, monkeypatch)
    assert m["n_decode_total"] == 12.0
    assert m["requests_processing"] == 1.0


def test_scrape_handles_scientific_notation_and_labels(monkeypatch):
    body = 'llamacpp:n_decode_total{model="a-b_c.d"} 1.5e3\nllamacpp:requests_processing 2\n'
    m = _scrape_body(body, monkeypatch)
    assert m["n_decode_total"] == 1500.0


def test_scrape_survives_malformed_values(monkeypatch):
    body = "llamacpp:n_decode_total NaNsense\nllamacpp:requests_processing 1\n"
    m = _scrape_body(body, monkeypatch)
    # The bad line is skipped, the good one still parses.
    assert m.get("requests_processing") == 1.0


def test_scrape_survives_non_utf8_bytes(monkeypatch):
    body = b"llamacpp:n_decode_total 5\n\xff\xfe garbage\nllamacpp:requests_processing 1\n"
    m = _scrape_body(body, monkeypatch)
    assert m["n_decode_total"] == 5.0


def test_scrape_on_empty_body_is_falsy(monkeypatch):
    assert not _scrape_body("", monkeypatch)


def test_scrape_failure_returns_none(monkeypatch):
    def _boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(ls.urllib.request, "urlopen", _boom)
    assert LlamaServerStatsLogger("http://127.0.0.1:0", _Capture())._scrape() is None


# ------------------------------------------------------------------- lifecycle edges


def test_scrape_failures_do_not_advance_the_stall_clock(monkeypatch):
    # /metrics briefly unavailable during a model swap must not read as a wedge.
    cap = _Capture()
    clock = {"t": 0.0}
    monkeypatch.setattr(ls.time, "monotonic", lambda: clock["t"])
    lg = LlamaServerStatsLogger("http://127.0.0.1:0", cap, stall_timeout_s = 600.0)
    lg._interval = 0.001
    state = {"i": 0}

    def fake_scrape():
        state["i"] += 1
        clock["t"] += 10.0
        if state["i"] > 200:
            lg.stop()
            return None
        return None  # every scrape fails

    lg._scrape = fake_scrape
    lg._run()
    assert not _stalls(cap)


def test_stop_is_honoured_promptly(monkeypatch):
    lg = LlamaServerStatsLogger("http://127.0.0.1:0", _Capture())
    lg._interval = 1.0
    lg._scrape = lambda: {"n_decode_total": 1.0, "requests_processing": 0.0}
    t = threading.Thread(target = lg._run, daemon = True)
    t.start()
    lg.stop()
    t.join(timeout = 10)
    assert not t.is_alive(), "stop() must end the poll loop"


def test_zero_and_negative_interval_are_clamped():
    assert LlamaServerStatsLogger("http://x", _Capture(), 0.0)._interval >= 1.0
    assert LlamaServerStatsLogger("http://x", _Capture(), -5.0)._interval >= 1.0


def test_negative_stall_timeout_is_clamped_to_disabled():
    lg = LlamaServerStatsLogger("http://x", _Capture(), stall_timeout_s = -1.0)
    assert lg._stall_timeout == 0.0


def test_base_url_trailing_slash_is_normalised():
    assert LlamaServerStatsLogger("http://h:1/", _Capture())._url == "http://h:1/metrics"
    assert LlamaServerStatsLogger("http://h:1", _Capture())._url == "http://h:1/metrics"


def test_env_defaults_are_read(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_ENGINE_STALL_TIMEOUT_S", "42")
    monkeypatch.setenv("UNSLOTH_STUDIO_ENGINE_STATS_INTERVAL_S", "7")
    started = {}
    monkeypatch.setattr(
        LlamaServerStatsLogger,
        "start",
        lambda self: started.update(timeout = self._stall_timeout, interval = self._interval),
    )
    ls.maybe_start_stats_logger("http://127.0.0.1:1", _Capture())
    assert started == {"timeout": 42.0, "interval": 7.0}


def test_garbage_env_falls_back_to_defaults(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_ENGINE_STALL_TIMEOUT_S", "not-a-number")
    monkeypatch.setenv("UNSLOTH_STUDIO_ENGINE_STATS_INTERVAL_S", "")
    started = {}
    monkeypatch.setattr(
        LlamaServerStatsLogger,
        "start",
        lambda self: started.update(timeout = self._stall_timeout, interval = self._interval),
    )
    ls.maybe_start_stats_logger("http://127.0.0.1:1", _Capture())
    assert started["timeout"] == 600.0
    assert started["interval"] >= 1.0


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF", " Off "])
def test_stats_logger_can_be_disabled_entirely(value, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_ENGINE_STATS", value)
    assert ls.maybe_start_stats_logger("http://127.0.0.1:1", _Capture()) is None
