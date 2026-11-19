# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Trusted tool folders holding a socket or FIFO, apart from the credential-name tests (lint AV008)."""

import os
import socket
import tempfile
from pathlib import Path

import pytest

from core.inference import os_sandbox, tool_path_approval as gate
from utils import trusted_tool_folders_settings as trusted


@pytest.fixture
def folder(linkable_temp_base, monkeypatch):
    tools_dir = Path(tempfile.mkdtemp(dir = linkable_temp_base)) / "tools"
    tools_dir.mkdir()
    (tools_dir / "run.sh").write_text("echo hi\n")
    monkeypatch.setattr(trusted, "_stored", lambda: [os.path.realpath(tools_dir)])
    monkeypatch.setattr(gate, "_studio_db_exists", lambda: True)
    return tools_dir


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason = "POSIX IPC nodes")
@pytest.mark.parametrize("kind", ["socket", "fifo"])
def test_a_folder_holding_a_host_channel_is_not_granted(folder, kind):
    path = folder / "sub" / "agent.sock"
    path.parent.mkdir()
    if kind == "fifo":
        os.mkfifo(path)
        assert os_sandbox.trusted_tool_folder_roots() == ()
        return
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(path))
        assert os_sandbox.trusted_tool_folder_roots() == ()
    finally:
        server.close()


def test_the_same_folder_without_a_channel_is_granted(folder):
    assert os_sandbox.trusted_tool_folder_roots() == (os.path.realpath(folder),)
