# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#9586: driving _start_studio_server must not leave its log in the shared tempdir."""

import os
import tempfile
from pathlib import Path


def test_start_server_log_stays_out_of_shared_tempdir(monkeypatch):
    from unsloth_cli.commands import start as start_mod

    # gettempdir()'s own lookup order, so this is the dir it would use without the fixture.
    env_tmp = next((os.environ[v] for v in ("TMPDIR", "TEMP", "TMP") if os.environ.get(v)), "/tmp")
    system_tmp = Path(env_tmp).resolve()
    before = set(system_tmp.glob("unsloth-start-server-*.log"))

    class FakePopen:
        def __init__(self, command, **kwargs):
            # Nonexistent pid: the atexit shutdown must never signal a real process group.
            self.pid = 2**31 - 1

        def poll(self):
            return None

    monkeypatch.setattr(start_mod, "_auto_served_server", None)
    monkeypatch.setattr(start_mod.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(start_mod, "_studio_healthy", lambda base, timeout = 3.0: True)
    # Without the key marker the readiness loop spins until _SERVER_START_TIMEOUT_S.
    monkeypatch.setattr(start_mod, "_log_tail", lambda path, lines = 20: "API Key: sk-unsloth-x")
    monkeypatch.setattr(start_mod.time, "sleep", lambda _s: None)

    start_mod._start_studio_server(
        "http://127.0.0.1:8888",
        "unsloth/M-GGUF",
        start_mod.LoadOptions(),
    )

    assert set(system_tmp.glob("unsloth-start-server-*.log")) == before
    assert list(Path(tempfile.gettempdir()).glob("unsloth-start-server-*.log"))
