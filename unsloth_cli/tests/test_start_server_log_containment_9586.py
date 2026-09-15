# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for issue #9586 channel 4: the start-server log left in the tempdir.

``_start_studio_server()`` opens
``Path(tempfile.gettempdir()) / f"unsloth-start-server-{os.getpid()}.log"``. The name
carries the pid, so a test that drives it for real accumulates a file per run rather than
colliding with the last one.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def test_the_tempdir_is_private_to_this_test():
    """The containment itself.

    ``tempfile.tempdir`` is the documented override; TMPDIR/TEMP/TMP are consulted by
    ``gettempdir()`` only once and then cached, so setting those in a fixture would be
    too late in a session that had already resolved one.
    """
    resolved = Path(tempfile.gettempdir()).resolve()
    assert resolved.name == "tempdir"
    assert resolved.is_dir()

    system_tmp = Path(os.environ.get("TEMP") or os.environ.get("TMPDIR") or "/tmp").resolve()
    assert resolved != system_tmp


def test_the_start_server_log_lands_in_the_private_tempdir(monkeypatch):
    """The pin that matters, by the real path expression.

    Built the way ``_start_studio_server`` builds it rather than by calling it, which
    would spawn a server. If the containment regressed, this would resolve into the
    shared tempdir the whole machine can traverse.
    """
    log_path = Path(tempfile.gettempdir()) / f"unsloth-start-server-{os.getpid()}.log"

    private = Path(tempfile.gettempdir()).resolve()
    assert private in log_path.resolve().parents

    system_tmp = Path(os.environ.get("TEMP") or os.environ.get("TMPDIR") or "/tmp")
    assert log_path.resolve().parent != system_tmp.resolve()


def test_the_real_driver_leaves_the_shared_tempdir_clean(monkeypatch):
    """End to end against the actual function, with only the spawn stubbed.

    This is the case the issue observed: a test that drives ``_start_studio_server`` for
    real. The stubs mirror
    ``test_start_studio_server_forwards_tool_flags_via_command_and_env``, so the log is
    opened exactly as it is in production.
    """
    from unsloth_cli.commands import start as start_mod

    system_tmp = Path(os.environ.get("TEMP") or os.environ.get("TMPDIR") or "/tmp").resolve()
    before = set(system_tmp.glob("unsloth-start-server-*.log"))

    class FakePopen:
        def __init__(self, command, **kwargs):
            self.pid = 1

        def poll(self):
            return None

    # _start_studio_server sets this module global and registers an atexit shutdown for
    # it. Restoring it through monkeypatch keeps this test from leaving process state
    # behind, which would be an odd thing for a test in this series to do.
    monkeypatch.setattr(start_mod, "_auto_served_server", None)
    monkeypatch.setattr(start_mod.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(start_mod, "_studio_healthy", lambda base, timeout = 3.0: True)
    # Must contain the key marker: the readiness loop spins until _SERVER_START_TIMEOUT_S
    # without it, which is what the sibling test's stub is doing too.
    monkeypatch.setattr(start_mod, "_log_tail", lambda path, lines = 20: "API Key: sk-unsloth-x")
    monkeypatch.setattr(start_mod.time, "sleep", lambda _s: None)

    start_mod._start_studio_server(
        "http://127.0.0.1:8888",
        "unsloth/M-GGUF",
        start_mod.LoadOptions(),
    )

    assert set(system_tmp.glob("unsloth-start-server-*.log")) == before

    # And it did land somewhere -- a test that wrote nothing would pass the check above
    # for the wrong reason.
    private = Path(tempfile.gettempdir())
    assert list(private.glob("unsloth-start-server-*.log"))
