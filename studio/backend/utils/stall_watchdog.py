# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dump every thread's stack from inside the backend while its event loop is stalled (#9712).

Two capture paths. Loop stuck with the GIL free: the watchdog thread counts consecutive slow
no-op probes and dumps from Python. GIL held: no Python thread runs, so each beat re-arms
``faulthandler.dump_traceback_later`` as a dead man's switch; its C timer thread dumps without
the GIL. Off unless UNSLOTH_STUDIO_STALL_WATCHDOG=1. Dumps go to the stderr fd (below run.py's
tee). Assumes it is the only user of faulthandler's process-global delayed-dump timer.
"""

from __future__ import annotations

import asyncio
import faulthandler
import os
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Callable, Optional, TextIO

import structlog

logger = structlog.get_logger(__name__)

ENABLE_ENV_VAR = "UNSLOTH_STUDIO_STALL_WATCHDOG"

BEAT_INTERVAL_S = 2.5
# Healthy mac smoke runs: ~50ms worst latency, single-probe outliers to ~3.4s.
PROBE_SLOW_S = 1.0
# 3 slow beats = 6-8.5s unresponsive: dumps before the shortest recorded stall (10.03s) ends.
SLOW_PROBES_BEFORE_DUMP = 3
DEAD_MAN_TIMEOUT_S = 8.0
# Shared by both capture paths.
DUMP_COOLDOWN_S = 600.0


def stand_down_for_the_warm() -> bool:
    """True from process start until the warm is over: a switch armed before the warm's
    GIL-holding ``import torch`` cannot be disarmed and would dump a known stall."""
    from utils.torch_warmup import DISABLE_ENV_VAR as _WARM_DISABLED
    from utils.torch_warmup import warm_status

    status = warm_status()
    if status["started"]:
        return bool(status["alive"] and not status["finished"])
    return os.environ.get(_WARM_DISABLED) != "1"


async def _noop() -> None:
    return None


class StallWatchdog:
    """One daemon thread beating against the event loop. start() / stop()."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        suppress: Optional[Callable[[], bool]] = None,
        dump_file: Optional[TextIO] = None,
        beat_interval_s: float = BEAT_INTERVAL_S,
        probe_slow_s: float = PROBE_SLOW_S,
        slow_probes_before_dump: int = SLOW_PROBES_BEFORE_DUMP,
        dead_man_timeout_s: float = DEAD_MAN_TIMEOUT_S,
        dump_cooldown_s: float = DUMP_COOLDOWN_S,
    ) -> None:
        import sys

        self._loop = loop
        self._suppress = suppress or (lambda: False)
        self._dump_file = dump_file if dump_file is not None else sys.stderr
        self._beat_interval_s = beat_interval_s
        self._probe_slow_s = probe_slow_s
        self._slow_probes_before_dump = slow_probes_before_dump
        self._dead_man_timeout_s = dead_man_timeout_s
        self._dump_cooldown_s = dump_cooldown_s

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._slow_streak = 0
        self._stall_started: Optional[float] = None
        self._last_dump: Optional[float] = None
        self._dead_man_armed = False
        self._arm_failure_logged = False

    def start(self) -> None:
        self._thread = threading.Thread(
            target = self._run,
            daemon = True,
            name = "stall-watchdog",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._cancel_dead_man()
        thread = self._thread
        if thread is not None:
            thread.join(timeout = 2.0)

    def _run(self) -> None:
        last_beat = time.monotonic()
        while not self._stop_event.is_set():
            beat_started = time.monotonic()
            # Beat gap past the switch timeout: the switch fired; start the cooldown.
            gap = beat_started - last_beat
            if self._dead_man_armed and gap > self._dead_man_timeout_s:
                self._last_dump = beat_started
                self._dead_man_armed = False
                logger.warning(
                    "stall watchdog was itself blocked for %.1fs; if the GIL was held, "
                    "faulthandler wrote a thread dump to stderr (system sleep also lands here)",
                    gap,
                )
            last_beat = beat_started

            if self._suppress_now():
                self._stop_event.wait(min(self._beat_interval_s, 0.5))
                continue

            self._arm_dead_man()
            self._probe(beat_started)

            elapsed = time.monotonic() - beat_started
            self._stop_event.wait(max(0.0, self._beat_interval_s - elapsed))
        self._cancel_dead_man()

    def _suppress_now(self) -> bool:
        try:
            suppressed = bool(self._suppress())
        except Exception:
            suppressed = False
        if suppressed:
            self._cancel_dead_man()
            self._slow_streak = 0
            self._stall_started = None
        return suppressed

    def _probe(self, beat_started: float) -> None:
        try:
            future = asyncio.run_coroutine_threadsafe(_noop(), self._loop)
        except RuntimeError:
            self._stop_event.wait(self._beat_interval_s)
            return
        try:
            future.result(timeout = self._probe_slow_s)
        except FutureTimeoutError:
            # Not cancelled: that leaves a never-awaited coroutine warning.
            self._slow_streak += 1
            if self._stall_started is None:
                self._stall_started = beat_started
            if self._slow_streak >= self._slow_probes_before_dump:
                self._dump_from_python()
            return
        except Exception:
            pass
        if self._slow_streak:
            stalled_for = time.monotonic() - (self._stall_started or beat_started)
            logger.warning(
                "event loop answering again after %.1fs (%d consecutive slow probes)",
                stalled_for,
                self._slow_streak,
            )
        self._slow_streak = 0
        self._stall_started = None

    def _in_cooldown(self) -> bool:
        return (
            self._last_dump is not None
            and time.monotonic() - self._last_dump < self._dump_cooldown_s
        )

    def _arm_dead_man(self) -> None:
        if self._in_cooldown():
            self._cancel_dead_man()
            return
        try:
            faulthandler.dump_traceback_later(
                self._dead_man_timeout_s,
                repeat = False,
                file = self._dump_file,
                exit = False,
            )
            self._dead_man_armed = True
        except Exception as exc:
            self._dead_man_armed = False
            if not self._arm_failure_logged:
                self._arm_failure_logged = True
                logger.warning(
                    "stall watchdog cannot arm faulthandler (%s); GIL-held stalls "
                    "will not be dumped, only slow-probe ones",
                    exc,
                )

    def _cancel_dead_man(self) -> None:
        if self._dead_man_armed:
            faulthandler.cancel_dump_traceback_later()
            self._dead_man_armed = False

    def _dump_from_python(self) -> None:
        if self._in_cooldown():
            return
        now = time.monotonic()
        self._last_dump = now
        # Else this beat's switch could fire a second dump inside the cooldown.
        self._cancel_dead_man()
        stalled_for = now - (self._stall_started or now)
        try:
            self._dump_file.write(
                f"\nstall watchdog: event loop unresponsive for {stalled_for:.1f}s "
                f"({self._slow_streak} consecutive slow probes), dumping all threads\n"
            )
            self._dump_file.flush()
            faulthandler.dump_traceback(file = self._dump_file, all_threads = True)
        except Exception as exc:
            logger.warning("stall watchdog could not write a thread dump: %s", exc)


_watchdog: Optional[StallWatchdog] = None
_watchdog_lock = threading.Lock()


def start_stall_watchdog(
    loop: asyncio.AbstractEventLoop, *, suppress: Optional[Callable[[], bool]] = None
) -> Optional[StallWatchdog]:
    """Start the process-wide watchdog. Returns None unless the env opts in."""
    if os.environ.get(ENABLE_ENV_VAR) != "1":
        return None
    global _watchdog
    with _watchdog_lock:
        if _watchdog is not None:
            _watchdog.stop()
        _watchdog = StallWatchdog(loop, suppress = suppress)
        _watchdog.start()
        return _watchdog


def stop_stall_watchdog() -> None:
    global _watchdog
    with _watchdog_lock:
        if _watchdog is not None:
            _watchdog.stop()
            _watchdog = None
