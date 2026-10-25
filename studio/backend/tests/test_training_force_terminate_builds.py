# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Force-terminating a training run also stops the processes its worker started: a kernel source
build runs in its own session (so a timeout can stop its compilers together) and would otherwise
outlive the worker, keep compiling, and leave its install lock free for a peer to race."""

import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from core.training import training


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().split()[2] != "Z"
    except OSError:
        return True


@pytest.mark.skipif(os.name == "nt", reason = "POSIX sessions")
def test_a_build_in_its_own_session_dies_with_the_run(tmp_path):
    pid_file = tmp_path / "build.pid"
    worker_script = (
        "import subprocess, sys, time; "
        "build = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], "
        "start_new_session = True); "
        f"open({str(pid_file)!r}, 'w').write(str(build.pid)); "
        "time.sleep(120)"
    )
    worker = subprocess.Popen([sys.executable, "-c", worker_script])
    try:
        for _ in range(100):
            if pid_file.exists() and pid_file.read_text():
                break
            time.sleep(0.05)
        build = int(pid_file.read_text())
        assert _alive(build)
        descendants = training._terminate_descendants(worker.pid)
        assert build in {child.pid for child in descendants}
        worker.terminate()
        worker.wait(timeout = 10)
        training._reap_descendants(descendants)
        for _ in range(50):
            if not _alive(build):
                break
            time.sleep(0.1)
        assert not _alive(build), "the build outlived the force-terminated worker"
    finally:
        if worker.poll() is None:
            worker.kill()
        if pid_file.exists() and pid_file.read_text() and _alive(int(pid_file.read_text())):
            os.kill(int(pid_file.read_text()), 9)


def test_force_terminate_stops_descendants_before_the_worker(monkeypatch):
    order = []
    proc = SimpleNamespace(
        pid = 4242,
        is_alive = lambda: True,
        terminate = lambda: order.append("worker"),
        join = lambda timeout = None: None,
        kill = lambda: order.append("kill"),
    )
    backend = training.TrainingBackend.__new__(training.TrainingBackend)
    backend.__dict__.update(
        _lock = threading.Lock(),
        _proc = proc,
        _cancel_requested = False,
        _cancel_cleanup_output_dir = None,
        _output_dir = None,
        _pump_thread = None,
    )
    monkeypatch.setattr(
        training, "_terminate_descendants", lambda pid: order.append(("descendants", pid)) or ["c"]
    )
    monkeypatch.setattr(training, "_reap_descendants", lambda d: order.append(("reap", tuple(d))))
    backend.force_terminate()
    assert order[0] == ("descendants", 4242) and order[1] == "worker"
    assert ("reap", ("c",)) in order
