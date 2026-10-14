# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import os
import signal
import subprocess
import sys
import time

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402
from utils.process_lifetime import mark_process_shutting_down, mark_quit_signal_received  # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason = "POSIX process groups")

_LLAMA_SERVER_LIKE = r"""
import os, signal, sys, time
seen = []
def handler(signum, frame):
    if seen:
        os._exit(1)
    seen.append(signum)
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)
print("ready", flush = True)
while not seen:
    time.sleep(0.01)
time.sleep(float(sys.argv[1]))
sys.exit(0)
"""


def _backend_holding(proc) -> LlamaCppBackend:
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = proc
    backend._healthy = True
    backend._owns_pidfile = False
    return backend


def _spawn(shutdown_s = 0.5):
    proc = subprocess.Popen(
        [sys.executable, "-c", _LLAMA_SERVER_LIKE, str(shutdown_s)],
        stdout = subprocess.PIPE,
        text = True,
    )
    assert proc.stdout.readline().strip() == "ready"
    assert os.getpgid(proc.pid) == os.getpgrp()
    return proc


def test_ctrl_c_shutdown_lets_llama_server_exit_on_its_own():
    proc = _spawn()
    try:
        os.kill(proc.pid, signal.SIGINT)
        mark_quit_signal_received()
        _backend_holding(proc)._kill_process(teardown = True)
        assert proc.returncode == 0, "llama-server got a second signal mid-shutdown"
    finally:
        proc.kill()
        proc.wait()
        proc.stdout.close()


def test_llama_server_stuck_after_ctrl_c_is_killed_not_signalled_twice():
    proc = _spawn(shutdown_s = 30)
    try:
        os.kill(proc.pid, signal.SIGINT)
        mark_quit_signal_received()
        _backend_holding(proc)._kill_process(teardown = True)
        assert proc.returncode == -signal.SIGKILL
    finally:
        proc.kill()
        proc.wait()
        proc.stdout.close()


def test_unloading_a_model_still_stops_llama_server_at_once():
    proc = _spawn()
    try:
        started = time.monotonic()
        _backend_holding(proc)._kill_process()
        assert proc.returncode == 0
        assert time.monotonic() - started < 1.5
    finally:
        proc.kill()
        proc.wait()
        proc.stdout.close()


def test_quit_without_a_signal_still_stops_llama_server_at_once():
    proc = _spawn()
    try:
        mark_process_shutting_down()
        started = time.monotonic()
        _backend_holding(proc)._kill_process(teardown = True)
        assert proc.returncode == 0
        assert time.monotonic() - started < 1.5
    finally:
        proc.kill()
        proc.wait()
        proc.stdout.close()


def test_servers_share_one_grace_period():
    procs = [_spawn(), _spawn()]
    try:
        mark_quit_signal_received()
        started = time.monotonic()
        for proc in procs:
            _backend_holding(proc)._kill_process(teardown = True)
        assert time.monotonic() - started < 4
    finally:
        for proc in procs:
            proc.kill()
            proc.wait()
            proc.stdout.close()
