# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dump every thread's stack while the event loop is stalled, from inside the process.

#9712: the backend stops answering every route for 10-33s on macOS CI, then recovers.
The stalls are too rare to sit next to with py-spy (once in 1711 runs) and over before
anyone could attach, so the dump has to come from the stalled process itself, taken
while the stall is still in progress.

A stall has two shapes, and they need different capture mechanisms:

- The event loop is stuck but the GIL is free (a blocking call that slipped onto the
  loop, a syscall that will not return). Python threads still run, so a watchdog
  thread can time a no-op scheduled onto the loop and dump from Python after enough
  consecutive slow probes.
- Something is holding the GIL. No Python thread runs at all, the watchdog included,
  so counting slow probes observes nothing until the stall is already over. For this
  shape the watchdog re-arms ``faulthandler.dump_traceback_later`` on every beat: a
  dead man's switch. faulthandler's timeout thread is C code that dumps without
  acquiring the GIL, so when the beats stop, it fires mid-stall and writes the frame
  every thread is actually in -- including the one sitting on the GIL.

Diagnostic tooling, so it is off unless UNSLOTH_STUDIO_STALL_WATCHDOG=1: a normal
desktop run carries neither the extra thread nor the faulthandler timer. The mac
smoke workflow sets the env for every phase, which is where #9712's stalls were
observed.

Dumps go to the stderr file descriptor: the CI workflow redirects it to the
``logs/*.log`` it uploads, and the desktop launcher captures the child's pipe. A
direct terminal launch shows dumps on the console but its on-disk session log
misses them -- faulthandler writes at C level, underneath run.py's tee -- which is
the price of staying visible to the two consumers that diagnose #9712. The
structlog markers the watchdog emits around a stall do go through the tee.

The dead man's switch cannot be disarmed once something has the GIL, so it must
never be armed while the coordinated warm could still start: the warm's
``import torch`` holds the GIL for tens of seconds on a healthy process, and a beat
that armed just before it grabbed the GIL would dump over the one stall with a
known frame. The watchdog therefore stands down from its first beat until the warm
is over (see ``stand_down_for_the_warm``), the same window the launcher's health
watchdog holds its startup grace open for.

faulthandler's delayed-dump timer is process-global and this module assumes it is
its only user; nothing else in the backend arms it.

Not a replacement for the launcher-side health watchdog in commands.rs: that one
decides whether to kill the process from outside. This one only ever writes
diagnostics, from inside.
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
# Passing runs of the mac smoke report worst-case route latency around 50ms, with
# outliers to ~3.4s on a saturated instance. 1s flags a probe as slow without
# counting those single-probe outliers as a stall on their own.
PROBE_SLOW_S = 1.0
# Three slow beats in a row is 6-8.5s of continuously unresponsive loop depending
# on where in a beat the stall lands: past any healthy run, and early enough to
# dump before the shortest stall on record (10.03s) recovers.
SLOW_PROBES_BEFORE_DUMP = 3
# The dead man's switch fires this long after the last re-arm, so 5.5-8s into a
# GIL-held stall. Sized for the same 10s floor as the slow-probe path.
DEAD_MAN_TIMEOUT_S = 8.0
# One dump per stall is the useful number; a host that stalls chronically should
# not fill its log with them. Applies to both capture paths.
DUMP_COOLDOWN_S = 600.0


def stand_down_for_the_warm() -> bool:
    """True from process start until the coordinated warm is over.

    Suppressing only while the warm is *running* leaves a window between the
    watchdog's first beat and start_background_warm(), and a switch armed in that
    window cannot be disarmed once the warm has the GIL. When the warm is switched
    off entirely, no warm is coming and the watchdog engages immediately.
    """
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

    # -- the beat ---------------------------------------------------------------

    def _run(self) -> None:
        last_beat = time.monotonic()
        while not self._stop_event.is_set():
            beat_started = time.monotonic()
            # The watchdog going quiet is itself the signal in the GIL-held shape:
            # the switch fired while this thread could not run. Say so on recovery,
            # with the one number the raw dump cannot carry, and start the cooldown
            # so back-to-back stalls do not each leave a dump.
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
                # Shorter waits while standing down: a full beat's sleep after the
                # warm finishes is a window where a stall goes entirely uncaptured.
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
            # Loop closed; shutdown is racing us. The stop() call will land shortly.
            self._stop_event.wait(self._beat_interval_s)
            return
        try:
            future.result(timeout = self._probe_slow_s)
        except FutureTimeoutError:
            # Left to finish on its own once the loop recovers: cancelling a task
            # that never started leaves a never-awaited coroutine warning behind.
            self._slow_streak += 1
            if self._stall_started is None:
                self._stall_started = beat_started
            if self._slow_streak >= self._slow_probes_before_dump:
                self._dump_from_python()
            return
        except Exception:
            # A failed no-op means the loop answered; that is all the probe asks.
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

    # -- dumps ------------------------------------------------------------------

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
            # Once: a dump target with no usable file descriptor never grows one.
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
        # The switch armed earlier this beat is still live and would fire a second
        # dump if a GIL freeze followed inside the cooldown; the cooldown owns both
        # paths, so take it down now rather than at the next beat.
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
