# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""No child outlives the Studio that spawned it.

The chain this closes: a tool call runs under a shell wrapper, the kill path
reaped only the wrapper on Windows, and the orphaned venv python then made
`unsloth studio update` refuse to run until it was killed by hand.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

IS_WINDOWS = sys.platform == "win32"


def _alive(pid: int) -> bool:
    if IS_WINDOWS:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True,
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _kill(pid: int) -> None:
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            os.kill(pid, 9)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 1. Killing a tool call takes the payload with it
# ---------------------------------------------------------------------------
def test_tool_kill_takes_the_shell_payload_with_it(tmp_path):
    from core.inference.tools import _get_shell_cmd, _kill_process_tree

    pidfile = tmp_path / "child.pid"
    payload = (
        f"import os,time,pathlib;"
        f"pathlib.Path(r'{pidfile}').write_text(str(os.getpid()));"
        f"time.sleep(120)"
    )
    command = f'"{sys.executable}" -c "{payload}"'
    argv = _get_shell_cmd(command)
    print(f"\n[{sys.platform}] shell wrapper argv[0] = {argv[0]}")

    kwargs = {}
    if not IS_WINDOWS:
        kwargs["start_new_session"] = True  # what the real spawn does on POSIX
    proc = subprocess.Popen(argv, **kwargs)

    for _ in range(100):
        if pidfile.is_file() and pidfile.read_text().strip():
            break
        time.sleep(0.1)
    grandchild = int(pidfile.read_text().strip())
    assert _alive(grandchild)

    try:
        _kill_process_tree(proc)
        proc.wait(timeout=10)
        time.sleep(2.0)
        survived = _alive(grandchild)
        print(f"payload pid {grandchild} alive after _kill_process_tree: {survived}")
        # Windows reaches the payload via taskkill /T /F, POSIX via killpg. It
        # used to survive on Windows, orphaning the venv python that then blocked
        # `unsloth studio update`.
        assert not survived, "the payload under the shell wrapper was orphaned"
    finally:
        _kill(grandchild)


# ---------------------------------------------------------------------------
# 2. One surviving venv process blocks `unsloth studio update` (Windows)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not IS_WINDOWS, reason="the update gate is Windows-only")
def test_update_gate_blocks_on_a_single_orphan(tmp_path):
    from unsloth_cli import _studio_runtime_gate

    studio_home = tmp_path / "studio_home"
    venv = studio_home / "unsloth_studio"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    venv_python = venv / "Scripts" / "python.exe"
    assert venv_python.is_file()

    # No orphan: the gate lets the update through.
    _studio_runtime_gate.ensure_managed_environment_is_idle(studio_home)

    orphan = subprocess.Popen([str(venv_python), "-c", "import time; time.sleep(120)"])
    try:
        time.sleep(2.0)
        with pytest.raises(RuntimeError) as excinfo:
            _studio_runtime_gate.ensure_managed_environment_is_idle(studio_home)
        print(f"\nupdate gate said: {excinfo.value}")
        assert "in use by" in str(excinfo.value)
        assert str(orphan.pid) in str(excinfo.value)
    finally:
        _kill(orphan.pid)


@pytest.mark.skipif(IS_WINDOWS, reason="contrast case for POSIX")
def test_update_gate_is_a_noop_on_posix(tmp_path):
    """On Linux/macOS nothing checks for a running Studio before an update."""
    from unsloth_cli import _studio_runtime_gate

    _studio_runtime_gate.ensure_managed_environment_is_idle(tmp_path / "anything")


# ---------------------------------------------------------------------------
# 3. Shutdown paths that are not signals
# ---------------------------------------------------------------------------
def test_console_close_runs_the_graceful_shutdown():
    """Closing the console window is not a signal, so it needs its own handler."""
    import run

    assert hasattr(run, "_install_windows_console_handler")
    if not IS_WINDOWS:
        assert run._install_windows_console_handler(lambda *a: None) is False
        return

    calls = []
    assert run._install_windows_console_handler(lambda *a: calls.append(a)) is True


def test_windows_job_status_is_reported():
    """A silent failure meant nobody could tell the guarantee was off."""
    from utils import process_lifetime

    process_lifetime.initialize_parent_lifetime()
    in_force, detail = process_lifetime.windows_job_status()
    print(f"\n[{sys.platform}] job status: in_force={in_force} detail={detail!r}")
    assert detail != "not attempted"
    if IS_WINDOWS or sys.platform.startswith("linux"):
        assert in_force, detail
    else:
        # macOS: no kernel mechanism, so it must say so rather than imply cover.
        assert not in_force


def test_macos_style_orphans_are_recorded_and_reaped(tmp_path, monkeypatch):
    """The on-disk child record is the only reaper macOS has after a crash."""
    from utils import process_lifetime as pl

    record = tmp_path / "child_processes.json"
    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(record))
    pl._tracked_pids.clear()

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        pl.adopt_pid(child.pid)
        assert record.is_file(), "the child was not recorded"

        # Pretend a previous Studio wrote this and died. The identity is what
        # decides, not the pid: a dead pid is recycled fast on a busy machine
        # (macOS especially), and an owner whose start time no longer matches is
        # a different process, so its recorded children are orphans.
        import json

        payload = json.loads(record.read_text())
        payload["owner_identity"] = "a-previous-studio-that-is-gone"
        record.write_text(json.dumps(payload))

        reaped = pl.reap_recorded_children()
        print(f"\nreaped: {reaped}")
        assert child.pid in reaped
        # Reap our own zombie so the liveness check means something.
        child.wait(timeout=10)
        assert not _alive(child.pid)
        assert not record.exists(), "the record should be consumed"
    finally:
        _kill(child.pid)


def test_liveness_probe_does_not_kill_what_it_probes():
    """os.kill(pid, 0) is TerminateProcess on Windows, so the probe cannot use it."""
    from utils import process_lifetime as pl

    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        for _ in range(3):
            assert pl._pid_alive(victim.pid) is True
            time.sleep(0.3)
        assert victim.poll() is None, "the liveness probe killed the process"
    finally:
        _kill(victim.pid)

    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait(timeout=30)
    assert pl._pid_alive(dead.pid) is False


def test_a_live_owner_is_never_reaped(tmp_path, monkeypatch):
    """Two Studios at once must not kill each other's children."""
    from utils import process_lifetime as pl

    record = tmp_path / "child_processes.json"
    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(record))
    pl._tracked_pids.clear()

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        pl.adopt_pid(child.pid)
        # owner_pid is this process, which is very much alive.
        assert pl.reap_recorded_children() == []
        assert _alive(child.pid)
    finally:
        _kill(child.pid)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
