# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Losing the optional metrics probe costs the cross-check, and only the cross-check.

`start_cell` says so in as many words -- "Metrics are a cross-check, not the measurement. Losing
them costs the cross-check and nothing else" -- and the handler in `open()` did something else
entirely. `Tracing.start` had already succeeded by the time `MetricsWindow.open()` ran, so
clearing `self.capture` there abandoned a LIVE tracing session:

  `close()` returns early while `self.capture is None`, so `Tracing.end` is never sent.
  `detach()` is guarded on the same attribute, so it cannot stop it either.
  Tracing is per-browser (`TraceCapture`: "a second `Tracing.start` fails with 'Tracing has
  already been started (possibly in another tab)'"), so EVERY later window fails to start and
  reports `tracing did not start for this window`.

One failed `Performance.getMetrics` therefore cost the tracing instrument for the rest of the run
-- and left a `recordAsMuchAsPossible` capture recording underneath every number taken after it.
It does not take a broken browser to get there: `start_cell` enables the Performance domain under
a bare `except`, and `read_metrics` raises rather than returning empty when that domain is off.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.instruments.tracing import TracingInstrument  # noqa: E402

TRACE_TEXT = json.dumps(
    {
        "traceEvents": [
            {
                "name": "RunTask",
                "ph": "X",
                "ts": 0,
                "dur": 1000,
                "pid": 1,
                "tid": 1,
                "cat": "toplevel",
            }
        ]
    }
)


class _Cdp:
    """The CDP calls this instrument makes, answered the way a browser answers them.

    `tracing_active` is the browser-global state the bug leaks: a session left running here is a
    session no later window can start over.
    """

    def __init__(self, *, metrics_fail: bool = False) -> None:
        self.metrics_fail = metrics_fail
        self.tracing_active = False
        self.sent: list[str] = []
        self.starts_refused = 0
        self._handlers: dict = {}

    def on(self, event: str, handler) -> None:
        self._handlers[event] = handler

    def send(
        self,
        method: str,
        params: dict | None = None,
    ):
        self.sent.append(method)
        if method == "Tracing.start":
            if self.tracing_active:
                self.starts_refused += 1
                raise RuntimeError("Tracing has already been started (possibly in another tab)")
            self.tracing_active = True
            return {}
        if method == "Tracing.end":
            if not self.tracing_active:
                raise RuntimeError("Tracing is not started")
            self.tracing_active = False
            complete = self._handlers.get("Tracing.tracingComplete")
            if complete is not None:
                complete({"stream": "h1", "dataLossOccurred": False, "streamCompression": "none"})
            return {}
        if method == "IO.read":
            return {"data": TRACE_TEXT, "eof": True}
        if method == "IO.close":
            return {}
        if method == "Performance.enable":
            return {}
        if method == "Performance.getMetrics":
            if self.metrics_fail:
                raise RuntimeError("Performance domain is not enabled")
            return {"metrics": [{"name": "TaskDuration", "value": 0.001}]}
        return {}


def _instrument(cdp) -> TracingInstrument:
    inst = TracingInstrument()
    inst.attach(types.SimpleNamespace(cdp = cdp, page = None, paths = None))
    inst.start_cell(types.SimpleNamespace(cell_id = "r10K.A0.rep0", instrument_level = 1))
    return inst


def _window(name: str):
    return types.SimpleNamespace(name = name)


# ── the trace survives the probe ─────────────────────────────────────────────────────────────


def test_a_second_window_is_still_traced_after_the_metrics_probe_fails():
    cdp = _Cdp(metrics_fail = True)
    inst = _instrument(cdp)

    inst.open(_window("w1"))
    inst.close(_window("w1"))
    inst.open(_window("w2"))
    payload = inst.close(_window("w2"))

    assert cdp.starts_refused == 0
    assert payload["active"] is True


def test_the_first_window_is_still_traced_and_only_loses_the_cross_check():
    cdp = _Cdp(metrics_fail = True)
    inst = _instrument(cdp)

    inst.open(_window("w1"))
    payload = inst.close(_window("w1"))

    assert payload["active"] is True
    assert "Tracing.end" in cdp.sent
    assert payload.get("task_duration_crosscheck_drift") is None


def test_no_tracing_session_is_left_running_in_the_browser():
    """The leak itself. A capture nothing holds is a capture nothing can stop, and it keeps
    recording underneath every measurement taken after it."""

    cdp = _Cdp(metrics_fail = True)
    inst = _instrument(cdp)

    inst.open(_window("w1"))
    inst.close(_window("w1"))
    inst.detach()

    assert cdp.tracing_active is False


# ── the controls ─────────────────────────────────────────────────────────────────────────────


def test_a_healthy_window_still_cross_checks_against_the_metrics():
    """The control that matters: the probe is still taken when it can be."""

    cdp = _Cdp()
    inst = _instrument(cdp)

    inst.open(_window("w1"))
    payload = inst.close(_window("w1"))

    assert payload["active"] is True
    assert cdp.sent.count("Performance.getMetrics") == 2
    assert cdp.tracing_active is False


def test_a_capture_that_never_started_is_dropped_and_says_so():
    """The other control: when `Tracing.start` itself fails there is nothing to end, and the
    window must report that it was not traced rather than pretend it holds a capture."""

    cdp = _Cdp()
    cdp.tracing_active = True  # something else owns the browser's one tracing session
    inst = _instrument(cdp)

    inst.open(_window("w1"))
    payload = inst.close(_window("w1"))

    assert cdp.starts_refused == 1
    assert payload["active"] is False
    assert "Tracing.end" not in cdp.sent


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
