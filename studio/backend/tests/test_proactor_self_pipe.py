# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
from pathlib import Path
import socket
import sys
import threading
import time
import types as _types

import pytest


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

from utils import proactor_self_pipe as psp

windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason = "ProactorEventLoop is Windows-only"
)


@pytest.fixture
def reads(monkeypatch):
    """Count self-pipe read callbacks, restoring the class patch afterwards."""
    from asyncio import proactor_events

    base = proactor_events.BaseProactorEventLoop
    # Count against CPython's own method even if an earlier test in this process installed the guard.
    original = psp._cpython_loop_self_reading or base._loop_self_reading
    counter = {"n": 0}

    def counting(self, f = None):
        counter["n"] += 1
        return original(self, f)

    monkeypatch.setattr(base, "_loop_self_reading", counting)
    monkeypatch.setattr(psp, "_installed", False)
    monkeypatch.setattr(psp, "_cpython_loop_self_reading", None)
    monkeypatch.setattr(psp, "_socketpair", socket.socketpair)
    yield counter


def _run_after_peer_eof(close_every_new_pair = False):
    """Half-close the self-pipe peer, idle for a while, then prove a cross-thread wakeup still lands."""

    async def main():
        loop = asyncio.get_running_loop()
        await asyncio.sleep(0.1)
        loop._csock.shutdown(socket.SHUT_WR)
        if close_every_new_pair:

            def pair_then_close():
                ssock, csock = socket.socketpair()
                csock.shutdown(socket.SHUT_WR)
                return ssock, csock

            psp._socketpair = pair_then_close
        await asyncio.sleep(1.5)
        woke = loop.create_future()
        threading.Timer(0.05, lambda: loop.call_soon_threadsafe(woke.set_result, True)).start()
        return await asyncio.wait_for(woke, 3)

    loop = asyncio.ProactorEventLoop()
    try:
        return loop.run_until_complete(main())
    finally:
        loop.close()


@windows_only
def test_unguarded_loop_spins_on_self_pipe_eof(reads):
    # Pins the CPython behaviour the guard exists for: without it, every read completes instantly with b"".
    assert _run_after_peer_eof() is True
    assert reads["n"] > 1000


@windows_only
def test_guard_rebuilds_self_pipe_after_eof(reads):
    assert psp.install_proactor_self_pipe_guard() is True
    assert _run_after_peer_eof() is True
    assert reads["n"] < 50


@windows_only
def test_guard_backs_off_when_every_new_pair_is_closed(reads):
    assert psp.install_proactor_self_pipe_guard() is True
    assert _run_after_peer_eof(close_every_new_pair = True) is True
    assert reads["n"] < 50


@windows_only
def test_guard_leaves_normal_wakeups_alone(reads):
    assert psp.install_proactor_self_pipe_guard() is True

    async def main():
        loop = asyncio.get_running_loop()
        pipe = loop._ssock
        for _ in range(20):
            woke = loop.create_future()
            threading.Thread(
                target = lambda: loop.call_soon_threadsafe(woke.set_result, None)
            ).start()
            await asyncio.wait_for(woke, 2)
        return pipe is loop._ssock

    loop = asyncio.ProactorEventLoop()
    try:
        assert loop.run_until_complete(main()) is True
    finally:
        loop.close()


@windows_only
def test_guard_keeps_old_pair_and_retries_when_socketpair_fails(reads, monkeypatch):
    assert psp.install_proactor_self_pipe_guard() is True
    monkeypatch.setattr(psp, "_REBUILD_BACKOFF_SECONDS", 0.2)
    failures = {"left": 1}

    def flaky_socketpair():
        if failures["left"]:
            failures["left"] -= 1
            raise OSError("no buffer space")
        return socket.socketpair()

    async def main():
        loop = asyncio.get_running_loop()
        await asyncio.sleep(0.1)
        old_ssock = loop._ssock
        psp._socketpair = flaky_socketpair
        loop._csock.shutdown(socket.SHUT_WR)
        await asyncio.sleep(0.1)
        assert loop._ssock is old_ssock
        await asyncio.sleep(0.5)
        assert loop._ssock is not old_ssock
        woke = loop.create_future()
        threading.Timer(0.05, lambda: loop.call_soon_threadsafe(woke.set_result, True)).start()
        return await asyncio.wait_for(woke, 2)

    loop = asyncio.ProactorEventLoop()
    try:
        assert loop.run_until_complete(main()) is True
    finally:
        loop.close()
    assert failures["left"] == 0
    assert reads["n"] < 50


@windows_only
def test_guard_moves_signal_wakeup_fd_on_main_thread(reads):
    import signal

    assert threading.current_thread() is threading.main_thread()
    assert psp.install_proactor_self_pipe_guard() is True

    def current_wakeup_fd():
        fd = signal.set_wakeup_fd(-1)
        signal.set_wakeup_fd(fd)
        return fd

    async def main():
        loop = asyncio.get_running_loop()
        await asyncio.sleep(0.1)
        old_fd = loop._csock.fileno()
        assert current_wakeup_fd() == old_fd
        loop._csock.shutdown(socket.SHUT_WR)
        await asyncio.sleep(0.2)
        return old_fd, loop._csock.fileno(), current_wakeup_fd()

    loop = asyncio.ProactorEventLoop()
    try:
        old_fd, new_fd, wakeup_fd = loop.run_until_complete(main())
    finally:
        loop.close()
    assert new_fd != old_fd
    assert wakeup_fd == new_fd


class _RecordingLogger:
    def __init__(self):
        self.lines = []

    def warning(self, msg, *args):
        self.lines.append(("warning", msg % args))

    def debug(self, msg, *args):
        self.lines.append(("debug", msg % args))


@windows_only
def test_persistent_socketpair_failure_warns_once_per_streak_and_keeps_wakeups(reads, monkeypatch):
    assert psp.install_proactor_self_pipe_guard() is True
    monkeypatch.setattr(psp, "_REBUILD_BACKOFF_SECONDS", 0.1)
    log = _RecordingLogger()
    monkeypatch.setattr(psp, "logger", log)
    attempts = {"n": 0}

    def failing_socketpair():
        attempts["n"] += 1
        raise OSError("no buffer space")

    async def wakeup_delay(loop):
        woke = loop.create_future()
        sent = {}

        def send():
            sent["at"] = time.monotonic()
            loop.call_soon_threadsafe(woke.set_result, None)

        threading.Timer(0.05, send).start()
        await asyncio.wait_for(woke, 5)
        return time.monotonic() - sent["at"]

    async def main():
        loop = asyncio.get_running_loop()
        await asyncio.sleep(0.1)
        psp._socketpair = failing_socketpair
        loop._csock.shutdown(socket.SHUT_WR)
        await asyncio.sleep(1.0)
        during_failure = await wakeup_delay(loop)
        # Recovery resets the streak, so a later failure streak warns again.
        psp._socketpair = socket.socketpair
        await asyncio.sleep(0.3)
        psp._socketpair = failing_socketpair
        loop._csock.shutdown(socket.SHUT_WR)
        await asyncio.sleep(0.5)
        return during_failure

    loop = asyncio.ProactorEventLoop()
    try:
        during_failure = loop.run_until_complete(main())
    finally:
        loop.close()
    assert during_failure < 0.5
    assert 8 <= attempts["n"] <= 30
    warnings = [line for level, line in log.lines if level == "warning"]
    # "rebuilding it" once, then the first failure of each of the two streaks; every other retry logs at debug.
    assert len(warnings) == 3, warnings
    assert reads["n"] < 50


@windows_only
def test_swap_closes_new_sockets_when_setblocking_fails(monkeypatch):
    made = []

    class _Sock:
        def __init__(self):
            self.closed = False

        def setblocking(self, flag):
            raise OSError("setblocking failed")

        def close(self):
            self.closed = True

    def broken_socketpair():
        made.extend([_Sock(), _Sock()])
        return made[-2], made[-1]

    monkeypatch.setattr(psp, "_socketpair", broken_socketpair)
    monkeypatch.setattr(psp, "logger", _RecordingLogger())
    loop = _types.SimpleNamespace(_ssock = object(), _csock = object())
    old = (loop._ssock, loop._csock)
    assert psp._swap_self_pipe(loop) is False
    assert all(sock.closed for sock in made) and len(made) == 2
    assert (loop._ssock, loop._csock) == old


@windows_only
def test_guard_leaves_a_foreign_signal_wakeup_fd_alone(reads):
    import signal

    assert threading.current_thread() is threading.main_thread()
    assert psp.install_proactor_self_pipe_guard() is True
    foreign_r, foreign_w = socket.socketpair()
    foreign_w.setblocking(False)

    async def main():
        loop = asyncio.get_running_loop()
        await asyncio.sleep(0.1)
        signal.set_wakeup_fd(foreign_w.fileno())
        try:
            loop._csock.shutdown(socket.SHUT_WR)
            await asyncio.sleep(0.2)
        finally:
            # The loop's old socket is closed by now, so hand the fd back to its current one.
            current = signal.set_wakeup_fd(loop._csock.fileno())
        return current

    loop = asyncio.ProactorEventLoop()
    try:
        assert loop.run_until_complete(main()) == foreign_w.fileno()
    finally:
        loop.close()
        foreign_r.close()
        foreign_w.close()


def test_install_is_idempotent_and_skips_other_platforms(monkeypatch):
    monkeypatch.setattr(psp, "_installed", False)
    monkeypatch.setattr(psp.sys, "platform", "linux")
    assert psp.install_proactor_self_pipe_guard() is False
    monkeypatch.setattr(psp, "_installed", True)
    assert psp.install_proactor_self_pipe_guard() is True
