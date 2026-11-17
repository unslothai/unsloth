# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

import unsloth_cli.commands.start as start_cli


class Spawned(Exception):
    pass


def test_auto_started_server_runs_on_this_interpreter_without_unsloth_on_path(
    monkeypatch, tmp_path
):
    commands = []

    def popen(command, **kwargs):
        commands.append(command)
        raise Spawned

    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(start_cli.subprocess, "Popen", popen)
    try:
        with pytest.raises(Spawned):
            start_cli._start_studio_server(
                "http://127.0.0.1:8888", "unsloth/Qwen3-0.6B-GGUF", start_cli.LoadOptions()
            )
    finally:
        (Path(tempfile.gettempdir()) / f"unsloth-start-server-{os.getpid()}.log").unlink(
            missing_ok = True
        )

    command = commands[0]
    assert command[0] == sys.executable
    assert command[command.index("-c") + 2] == "run"
