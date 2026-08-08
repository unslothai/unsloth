# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where the Windows orphan guarantee holds, and where it stops.

Children inherit a Job Object with KILL_ON_JOB_CLOSE. The desktop updater clears
that flag before launching the installer, so both sides are pinned here with
real processes.
"""

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

IS_WINDOWS = sys.platform == "win32"
BACKEND = str(Path(__file__).resolve().parents[1])


def _alive(pid: int) -> bool:
    if IS_WINDOWS:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output = True, text = True
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
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output = True)
        else:
            os.kill(pid, 9)
    except Exception:
        pass


PARENT = textwrap.dedent(
    """
    import os, subprocess, sys, time
    sys.path.insert(0, sys.argv[1])
    if sys.argv[3] == "job":
        from utils.process_lifetime import initialize_parent_lifetime
        initialize_parent_lifetime()
    from core.inference.tools import _get_shell_cmd
    payload = (
        "import os,time,pathlib;"
        "pathlib.Path(r'%s').write_text(str(os.getpid()));"
        "time.sleep(180)" % sys.argv[2]
    )
    argv = _get_shell_cmd('"%s" -c "%s"' % (sys.executable, payload))
    kw = {}
    if sys.argv[3] == "job":
        # The helper Studio's own spawns go through: PR_SET_PDEATHSIG on Linux,
        # nothing on macOS, job inheritance on Windows.
        from utils.process_lifetime import child_popen_kwargs
        kw = child_popen_kwargs()
    subprocess.Popen(argv, **kw)
    time.sleep(180)
    """
)


def _run_case(tmp_path: Path, mode: str) -> bool:
    """Start a parent, let it spawn a shell-wrapped payload, hard-kill the
    parent, and report whether the payload survived."""
    pidfile = tmp_path / f"{mode}.pid"
    script = tmp_path / f"parent_{mode}.py"
    script.write_text(PARENT)
    parent = subprocess.Popen([sys.executable, str(script), BACKEND, str(pidfile), mode])
    try:
        for _ in range(200):
            if pidfile.is_file() and pidfile.read_text().strip():
                break
            time.sleep(0.1)
        payload_pid = int(pidfile.read_text().strip())
        assert _alive(payload_pid)

        # Hard kill, no tree flag: exactly what "End Task" / a crash does.
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/PID", str(parent.pid), "/F"], capture_output = True)
        else:
            os.kill(parent.pid, 9)
        parent.wait(timeout = 30)
        time.sleep(5)
        survived = _alive(payload_pid)
        print(f"\n[{sys.platform}] mode={mode}: payload {payload_pid} survived = {survived}")
        return survived
    finally:
        try:
            _kill(int(pidfile.read_text().strip()))
        except Exception:
            pass
        _kill(parent.pid)


@pytest.mark.skipif(not IS_WINDOWS, reason = "job objects are Windows-only")
def test_job_object_reaps_the_whole_tree(tmp_path):
    """With the guarantee in force, even a shell grandchild is reaped."""
    assert _run_case(tmp_path, "job") is False


@pytest.mark.skipif(not IS_WINDOWS, reason = "job objects are Windows-only")
def test_without_the_job_the_grandchild_is_orphaned(tmp_path):
    """This is the state the desktop updater leaves the app in: the job is
    still there but KILL_ON_JOB_CLOSE has been cleared and nothing restores
    it, so a hard exit leaks every child."""
    assert _run_case(tmp_path, "nojob") is True


@pytest.mark.skipif(IS_WINDOWS, reason = "POSIX contrast")
def test_posix_orphan_behaviour(tmp_path):
    """Linux reaps via PR_SET_PDEATHSIG on the direct child; macOS has no
    equivalent and relies entirely on the cooperative shutdown path."""
    survived = _run_case(tmp_path, "job")
    if sys.platform == "darwin":
        assert survived, "macOS now reaps orphans -- update this repro"
    else:
        assert not survived


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
