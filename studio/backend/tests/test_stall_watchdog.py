# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Invariant: a stalled event loop gets its thread stacks dumped while still stalled.

#9712: the backend stops serving every route for 10-33s and recovers before anyone
can attach py-spy. The watchdog exists so the next occurrence writes its own
diagnosis. Two stall shapes, two capture paths, both exercised here for real:

- Loop stuck, GIL free: the watchdog thread still runs, counts consecutive slow
  no-op probes, and dumps from Python.
- GIL held: no Python thread runs, the watchdog included. The dead man's switch it
  re-arms every beat has to fire from faulthandler's C thread mid-stall. The test
  holds the GIL genuinely (a busy loop under a long sys.setswitchinterval) rather
  than mocking the freeze, because the C thread firing without the GIL is the one
  property the whole design rests on.

CPU-only, no network, no GPU, no weights. Slowest test holds the GIL ~1.2s.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest

from utils.stall_watchdog import (
    StallWatchdog,
    stand_down_for_the_warm,
    start_stall_watchdog,
    stop_stall_watchdog,
)

_BACKEND_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture()
def loop_in_thread():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target = loop.run_forever, daemon = True, name = "test-loop")
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout = 5)
    # Drain whatever probes the watchdog left queued, or closing the loop leaves
    # never-awaited no-op coroutines behind as warnings.
    loop.run_until_complete(asyncio.sleep(0.1))
    loop.close()


def _wait_for(predicate, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_a_blocked_loop_dumps_after_consecutive_slow_probes(loop_in_thread, tmp_path):
    """The GIL-free shape: something blocking sits on the loop, Python threads still
    run, so the slow-probe counter reaches its threshold and dumps. The dump must
    name the blocking frame, otherwise it answers nothing."""
    dump_path = tmp_path / "dump.txt"

    def blocked_in_a_syscall():
        time.sleep(1.2)

    with dump_path.open("w") as dump_file:
        watchdog = StallWatchdog(
            loop_in_thread,
            dump_file = dump_file,
            beat_interval_s = 0.05,
            probe_slow_s = 0.05,
            slow_probes_before_dump = 3,
            # Out of the way: this test is about the Python-side path.
            dead_man_timeout_s = 30.0,
            dump_cooldown_s = 0.0,
        )
        watchdog.start()
        try:
            loop_in_thread.call_soon_threadsafe(blocked_in_a_syscall)
            assert _wait_for(
                lambda: "stall watchdog:" in dump_path.read_text(errors = "replace")
            ), "no dump was written while the loop was blocked"
        finally:
            watchdog.stop()

    text = dump_path.read_text(errors = "replace")
    assert "consecutive slow probes" in text
    assert (
        "blocked_in_a_syscall" in text
    ), "the dump does not show the frame the loop is stuck in; it cannot answer what #9712 asks"


def test_a_gil_holding_stall_is_dumped_mid_stall(loop_in_thread, tmp_path):
    """The #9712 shape: the GIL is held, so the watchdog thread itself is frozen and
    slow-probe counting observes nothing until the stall is over. Only the dead
    man's switch can capture this, firing from C while no Python thread runs."""
    dump_path = tmp_path / "dump.txt"
    released = threading.Event()

    def holds_the_gil():
        old = sys.getswitchinterval()
        # A pure-Python busy loop only releases the GIL at switch-interval
        # boundaries; pushing the interval past the hold keeps it for the duration.
        sys.setswitchinterval(10.0)
        try:
            deadline = time.monotonic() + 1.2
            while time.monotonic() < deadline:
                pass
        finally:
            sys.setswitchinterval(old)
            released.set()

    with dump_path.open("w") as dump_file:
        watchdog = StallWatchdog(
            loop_in_thread,
            dump_file = dump_file,
            beat_interval_s = 0.05,
            probe_slow_s = 0.05,
            slow_probes_before_dump = 1000,  # the Python path must not be the one firing
            dead_man_timeout_s = 0.4,
            dump_cooldown_s = 0.0,
        )
        watchdog.start()
        try:
            # Let it arm at least once before the freeze.
            assert _wait_for(lambda: watchdog._dead_man_armed, timeout_s = 5.0)
            loop_in_thread.call_soon_threadsafe(holds_the_gil)
            assert released.wait(timeout = 30.0)
            assert _wait_for(lambda: "Timeout" in dump_path.read_text(errors = "replace"))
        finally:
            watchdog.stop()

    text = dump_path.read_text(errors = "replace")
    assert "holds_the_gil" in text, (
        "the mid-stall dump does not show the GIL holder's frame; identifying it is "
        "the point of the dead man's switch"
    )


def test_repeated_gil_stalls_dump_once_per_cooldown(loop_in_thread, tmp_path):
    """The switch fires on its own once armed, so the cooldown has to gate the
    arming: a host that stalls chronically must leave one dump, and silence, until
    the window passes."""
    dump_path = tmp_path / "dump.txt"

    def hold_gil_for(seconds: float):
        released = threading.Event()

        def holder():
            old = sys.getswitchinterval()
            sys.setswitchinterval(10.0)
            try:
                deadline = time.monotonic() + seconds
                while time.monotonic() < deadline:
                    pass
            finally:
                sys.setswitchinterval(old)
                released.set()

        loop_in_thread.call_soon_threadsafe(holder)
        assert released.wait(timeout = 30.0)

    with dump_path.open("w") as dump_file:
        watchdog = StallWatchdog(
            loop_in_thread,
            dump_file = dump_file,
            beat_interval_s = 0.05,
            probe_slow_s = 0.05,
            slow_probes_before_dump = 1000,
            dead_man_timeout_s = 0.3,
            dump_cooldown_s = 60.0,
        )
        watchdog.start()
        try:
            assert _wait_for(lambda: watchdog._dead_man_armed, timeout_s = 5.0)
            hold_gil_for(0.8)
            assert _wait_for(lambda: "Timeout" in dump_path.read_text(errors = "replace"))
            # Let the recovery beat land and start the cooldown before stalling again.
            assert _wait_for(lambda: watchdog._last_dump is not None, timeout_s = 5.0)
            hold_gil_for(0.8)
            time.sleep(0.3)
        finally:
            watchdog.stop()

    text = dump_path.read_text(errors = "replace")
    assert text.count("Timeout") == 1, (
        "a second stall inside the cooldown window dumped again; a chronically "
        "stalling host would fill its log with stacks"
    )


def test_a_python_dump_disarms_the_switch_for_the_cooldown(loop_in_thread, tmp_path):
    """The two paths share the cooldown. A slow-probe dump leaves the switch armed
    from earlier in the same beat unless it is taken down with it, and a GIL freeze
    right after would land a second dump inside the window."""
    dump_path = tmp_path / "dump.txt"

    def blocks_the_loop():
        time.sleep(0.5)

    def holds_the_gil():
        old = sys.getswitchinterval()
        sys.setswitchinterval(10.0)
        try:
            deadline = time.monotonic() + 0.8
            while time.monotonic() < deadline:
                pass
        finally:
            sys.setswitchinterval(old)

    with dump_path.open("w") as dump_file:
        watchdog = StallWatchdog(
            loop_in_thread,
            dump_file = dump_file,
            beat_interval_s = 0.05,
            probe_slow_s = 0.05,
            slow_probes_before_dump = 2,
            dead_man_timeout_s = 0.3,
            dump_cooldown_s = 60.0,
        )
        watchdog.start()
        try:
            loop_in_thread.call_soon_threadsafe(blocks_the_loop)
            assert _wait_for(lambda: "stall watchdog:" in dump_path.read_text(errors = "replace"))
            loop_in_thread.call_soon_threadsafe(holds_the_gil)
            time.sleep(1.4)
        finally:
            watchdog.stop()

    text = dump_path.read_text(errors = "replace")
    assert "Timeout" not in text, (
        "the dead man's switch fired inside the cooldown a slow-probe dump started; "
        "one stall episode left two dumps"
    )


def test_stands_down_until_the_warm_is_over(monkeypatch):
    """The dead man's switch cannot be disarmed once the warm holds the GIL, so the
    gate has to be closed before the warm starts, and stay closed until it is over.
    Gating on 'warm running' left the window between the watchdog's first beat and
    start_background_warm(), and every cold start dumped over the torch import."""
    import utils.stall_watchdog as sw
    import utils.torch_warmup as tw

    def status(started: bool, finished: bool, alive: bool):
        monkeypatch.setattr(
            tw,
            "warm_status",
            lambda: {"started": started, "finished": finished, "alive": alive, "stages": {}},
        )

    monkeypatch.delenv(tw.DISABLE_ENV_VAR, raising = False)
    status(started = False, finished = False, alive = False)
    assert sw.stand_down_for_the_warm() is True, (
        "not standing down before the warm starts is the arming race: the switch "
        "armed in that window dumps over `import torch` on every cold start"
    )
    status(started = True, finished = False, alive = True)
    assert sw.stand_down_for_the_warm() is True
    status(started = True, finished = True, alive = False)
    assert sw.stand_down_for_the_warm() is False
    # A warm that died mid-stage is not coming back; staying down would blind the
    # watchdog for the rest of the session.
    status(started = True, finished = False, alive = False)
    assert sw.stand_down_for_the_warm() is False

    monkeypatch.setenv(tw.DISABLE_ENV_VAR, "1")
    status(started = False, finished = False, alive = False)
    assert sw.stand_down_for_the_warm() is False, (
        "with the warm switched off no warm is coming; standing down anyway keeps "
        "the watchdog disarmed for the whole session"
    )


def test_no_dump_while_suppressed(loop_in_thread, tmp_path):
    """The warm's `import torch` is a legitimate long GIL hold with a known frame.
    While the suppress callable says so, neither capture path may fire."""
    dump_path = tmp_path / "dump.txt"

    with dump_path.open("w") as dump_file:
        watchdog = StallWatchdog(
            loop_in_thread,
            suppress = lambda: True,
            dump_file = dump_file,
            beat_interval_s = 0.05,
            probe_slow_s = 0.05,
            slow_probes_before_dump = 1,
            dead_man_timeout_s = 0.2,
            dump_cooldown_s = 0.0,
        )
        watchdog.start()
        try:
            loop_in_thread.call_soon_threadsafe(time.sleep, 0.8)
            time.sleep(1.2)
        finally:
            watchdog.stop()

    assert dump_path.read_text(errors = "replace") == "", (
        "the watchdog dumped during a suppressed window; every warm import would "
        "land a spurious dump in the log"
    )


def test_off_unless_the_env_opts_in(loop_in_thread, monkeypatch):
    """Diagnostic tooling: a desktop run must carry neither the thread nor the
    faulthandler timer, so nothing starts without the env the CI workflow sets."""
    monkeypatch.delenv("UNSLOTH_STUDIO_STALL_WATCHDOG", raising = False)
    assert start_stall_watchdog(loop_in_thread) is None
    assert not any(t.name == "stall-watchdog" for t in threading.enumerate())


def test_start_and_stop_manage_one_process_wide_instance(loop_in_thread, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_STALL_WATCHDOG", "1")
    first = start_stall_watchdog(loop_in_thread)
    try:
        assert first is not None
        second = start_stall_watchdog(loop_in_thread)
        assert second is not None
        assert _wait_for(
            lambda: not (first._thread and first._thread.is_alive()), timeout_s = 5.0
        ), "starting a second watchdog left the first one's thread running"
    finally:
        stop_stall_watchdog()
    assert _wait_for(
        lambda: not any(t.name == "stall-watchdog" for t in threading.enumerate()),
        timeout_s = 5.0,
    )


def test_the_lifespan_wires_it_in():
    """Cross-file guard: the module only does anything because main.py starts it at
    startup and retires it at shutdown entry, and either side can be edited alone."""
    source = (_BACKEND_DIR / "main.py").read_text(encoding = "utf-8")
    assert (
        "start_stall_watchdog(asyncio.get_running_loop()" in source
    ), "main.py no longer starts the stall watchdog; the next #9712 stall leaves no dump behind"
    assert "suppress = stand_down_for_the_warm" in source, (
        "the watchdog no longer stands down for the warm; every cold start would "
        "dump over the torch import"
    )
    assert "stop_stall_watchdog()" in source, (
        "main.py never retires the watchdog, so shutdown's deliberate blocking "
        "reads as a stall and dumps into the exit log"
    )

    workflow = _BACKEND_DIR.parent.parent / ".github" / "workflows" / "studio-mac-ui-smoke.yml"
    assert workflow.is_file(), f"{workflow} moved; update this guard"
    assert "UNSLOTH_STUDIO_STALL_WATCHDOG" in workflow.read_text(encoding = "utf-8"), (
        "the watchdog is opt-in and the mac smoke workflow no longer opts in; the "
        "one place #9712 stalls have been seen runs without it"
    )
