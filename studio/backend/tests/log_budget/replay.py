# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Replay a Studio session through the real middleware in virtual time.

No sleeping and no wall clock. The middleware takes its timestamps from
``time.perf_counter`` in its own module namespace, so swapping that namespace for a clock
the test advances by hand makes a thirty-minute session run instantly and identically on a
loaded CI runner. Real sleeps would put every assertion within scheduler noise of a window
boundary, which is how a guard like this becomes flaky and then gets deleted.

The middleware itself is real. So is the dedup state, the quiet-success suppressor and the
shared liveness bucket. Only the clock and the terminal application are substituted.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional


class FakeClock:
    """Stands in for the ``time`` module inside ``loggers.handlers``.

    Only ``perf_counter`` is used by the middleware; anything else raises rather than
    silently falling through to the real module, so a future call site that starts reading
    the wall clock shows up here instead of quietly reintroducing nondeterminism.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def perf_counter(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def __getattr__(self, name: str):
        raise AttributeError(
            f"loggers.handlers called time.{name}(), which the log-budget replay does not "
            f"model. Add it to FakeClock deliberately rather than letting the guard fall "
            f"back to the wall clock."
        )


class LogCapture:
    """Collects what the middleware logged, in order."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    def info(self, event, **kw):
        self.events.append(("info", event, kw))

    def error(self, event, **kw):
        self.events.append(("error", event, kw))

    def warning(self, event, **kw):
        self.events.append(("warning", event, kw))

    def paths(self) -> list[str]:
        return [kw["path"] for _lvl, ev, kw in self.events if "path" in kw]

    def records_for(self, path: str) -> list[dict]:
        return [kw for _lvl, _ev, kw in self.events if kw.get("path") == path]


@dataclass
class Request:
    method: str
    path: str
    status: int = 200
    query: bytes = b""
    # How long the handler takes. Zero by default, which is what every scenario written
    # before this field wanted: a request that costs no virtual time. It exists because
    # the suppressors key on the STATUS CODE and never on the duration, so "a 200 that
    # took a minute" is a case the harness could not express at all, and therefore could
    # not budget or defend.
    duration_ms: float = 0.0


@dataclass
class ReplayResult:
    capture: LogCapture
    sent: list[Request] = field(default_factory = list)

    @property
    def emitted(self) -> int:
        return len(self.capture.events)


def _app_returning(
    status: int,
    duration_ms: float = 0.0,
    clock: "FakeClock | None" = None,
):
    async def app(scope, receive, send):
        # Advance BEFORE responding: the middleware stamps its window on the end time, so
        # a duration added afterwards would be invisible to the very rule under test.
        if duration_ms and clock is not None:
            clock.advance(duration_ms / 1000.0)
        await send({"type": "http.response.start", "status": status, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    return app


async def _noop_receive():
    return {"type": "http.disconnect"}


async def _noop_send(message):
    return None


def install(
    handlers,
    monkeypatch,
    clock: Optional[FakeClock] = None,
) -> FakeClock:
    """Point the middleware at a virtual clock. Returns the clock."""
    clock = clock or FakeClock()
    monkeypatch.setattr(handlers, "time", clock)
    return clock


def replay(
    handlers,
    monkeypatch,
    polled: dict,
    duration_s: float,
    boot: tuple = (),
    clock: Optional[FakeClock] = None,
    durations: Optional[dict] = None,
) -> ReplayResult:
    """Drive one middleware instance through ``boot`` then ``duration_s`` of polling.

    One instance for the whole run, because the de-duplication state lives on the instance
    and a fresh one per request would suppress nothing and quietly pass every budget.
    """
    from loggers.handlers import LoggingMiddleware

    clock = install(handlers, monkeypatch, clock)
    capture = LogCapture()
    monkeypatch.setattr(handlers, "logger", capture)

    middleware_by_status: dict[tuple[int, float], object] = {}
    result = ReplayResult(capture = capture)

    # A single middleware object shared by every status, so its dedup map is the real one.
    shared_state = LoggingMiddleware(_app_returning(200))

    def send_request(request: Request) -> None:
        key = (request.status, request.duration_ms)
        app = middleware_by_status.get(key)
        if app is None:
            app = _app_returning(request.status, request.duration_ms, clock)
            middleware_by_status[key] = app
        shared_state.app = app
        scope = {
            "type": "http",
            "path": request.path,
            "method": request.method,
            "query_string": request.query,
        }
        asyncio.run(shared_state(scope, _noop_receive, _noop_send))
        result.sent.append(request)

    for method, path, status in boot:
        send_request(Request(method = method, path = path, status = status))
        clock.advance(0.05)

    # Whole-second ticks, so every period in the registry lands on an exact tick and the
    # expectation formula and the replay agree by construction rather than by rounding.
    tick = 0.5
    started_at = clock.now
    next_due = {path: 0.0 for path in polled}
    while clock.now - started_at < duration_s:
        elapsed = clock.now - started_at
        for path, (period, _provenance) in polled.items():
            if elapsed + 1e-9 >= next_due[path]:
                send_request(
                    Request(
                        method = "GET",
                        path = path,
                        status = 200,
                        duration_ms = durations.get(path, 0.0) if durations else 0.0,
                    )
                )
                next_due[path] = elapsed + period
        clock.advance(tick)

    return result
