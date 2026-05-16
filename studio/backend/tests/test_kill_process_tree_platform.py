# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Cross-platform contract tests for _kill_process_tree.

The post-PR-5375 hardening pass added ``os.setsid`` to the sandbox
pre-exec; the cancel/timeout supervisor calls ``_kill_process_tree`` to
SIGKILL the resulting process group. ``os.getpgid``/``os.killpg`` are
Unix-only -- on Windows the helper must fall back to ``proc.kill()`` +
``taskkill /T``. We simulate Windows by stripping the platform-specific
attributes off ``os`` and verifying the helper still reaches the kill
path without raising.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference import tools as tools_mod


def _spawn_sleep(seconds: int = 60):
    """Spawn a lightweight sleeper. Prefers /bin/sleep (no Python startup
    cost) and falls back to ``python -c sleep`` on platforms that don't
    have /bin/sleep on PATH (mostly Windows)."""
    if sys.platform != "win32":
        from shutil import which

        sleep_bin = which("sleep") or "/bin/sleep"
        if Path(sleep_bin).exists():
            return subprocess.Popen(
                [sleep_bin, str(seconds)],
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
                start_new_session = True,
            )
    # Fallback: minimal Python sleeper. Adds ~25-40 MB per test which
    # is fine for single-test runs but is the reason we prefer
    # /bin/sleep when available (the suite spawns one per test).
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(%d)" % seconds],
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
    )


@pytest.fixture()
def short_proc():
    """A subprocess that sleeps long enough to be killable."""
    proc = _spawn_sleep()
    try:
        yield proc
    finally:
        try:
            proc.kill()
        except Exception:
            pass


class TestUnixPath:
    """On Linux/macOS the pgid path runs and reaps the child."""

    @pytest.mark.skipif(
        not (hasattr(os, "getpgid") and hasattr(os, "killpg")),
        reason = "No process-group APIs on this platform",
    )
    def test_kill_terminates_subprocess(self, short_proc):
        assert short_proc.poll() is None
        tools_mod._kill_process_tree(short_proc)
        # Give the OS a moment to reap. _kill_process_tree does not block.
        deadline = time.time() + 5.0
        while short_proc.poll() is None and time.time() < deadline:
            time.sleep(0.05)
        assert short_proc.poll() is not None, "subprocess should have died"

    @pytest.mark.skipif(
        not (hasattr(os, "getpgid") and hasattr(os, "killpg")),
        reason = "No process-group APIs on this platform",
    )
    def test_no_raise_on_already_exited(self):
        # poll() already returned non-None: helper must early-return.
        class Dead:
            pid = 0

            def poll(self):
                return 0

        # Should not raise even though pid 0 has no pgid.
        tools_mod._kill_process_tree(Dead())


class TestWindowsFallback:
    """Simulate a Windows runtime where os lacks getpgid/killpg."""

    def test_kill_falls_back_to_proc_kill_when_pgid_missing(
        self, short_proc, monkeypatch
    ):
        # Strip the Unix-only attributes so the helper takes the
        # Windows branch.
        if hasattr(os, "getpgid"):
            monkeypatch.delattr(os, "getpgid", raising = False)
        if hasattr(os, "killpg"):
            monkeypatch.delattr(os, "killpg", raising = False)
        # Also lie about sys.platform so the taskkill fallback runs.
        monkeypatch.setattr(tools_mod.sys, "platform", "win32")
        # taskkill won't exist on Linux; capture its absence as a no-op
        # via subprocess.run rather than failing the test.
        import subprocess as _sp

        with mock.patch.object(_sp, "run", return_value = None):
            tools_mod._kill_process_tree(short_proc)
        deadline = time.time() + 5.0
        while short_proc.poll() is None and time.time() < deadline:
            time.sleep(0.05)
        assert short_proc.poll() is not None, "subprocess should have died"

    def test_no_attribute_error_on_simulated_windows(self, monkeypatch):
        """Regression: AttributeError used to skip the kill entirely.

        Before the fix, ``os.getpgid(...)`` raised ``AttributeError`` on
        Windows; the helper's exception list only covered
        ProcessLookupError + PermissionError, so ``AttributeError``
        bubbled and the supervisor skipped the kill. The fix gates on
        ``hasattr(os, ...)`` first, so this test pins that contract.
        """
        if hasattr(os, "getpgid"):
            monkeypatch.delattr(os, "getpgid", raising = False)
        if hasattr(os, "killpg"):
            monkeypatch.delattr(os, "killpg", raising = False)
        monkeypatch.setattr(tools_mod.sys, "platform", "win32")

        class FakeProc:
            pid = 9999

            def __init__(self):
                self.killed = False

            def poll(self):
                return None if not self.killed else 0

            def kill(self):
                self.killed = True

        fp = FakeProc()
        import subprocess as _sp

        with mock.patch.object(_sp, "run", return_value = None):
            tools_mod._kill_process_tree(fp)  # must not raise
        assert fp.killed
