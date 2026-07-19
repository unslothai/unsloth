# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Accuracy guards for the sandbox-paths note appended to the python/terminal
tool descriptions (#7242).

The note must not misdirect the model with claims that are false for the real
sandbox: (1) the sandbox is not network-isolated -- the python tool can reach an
allowlist of public hosts (github.com / huggingface.co / pypi.org), so it must
not claim it "cannot reach ... remote hosts" at all; (2) a project sandbox is a
shared per-project directory, so a new thread can inherit files from prior
threads -- the note must not claim the workdir "starts empty" or "persists only
for this conversation".
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.tools import (
    PYTHON_TOOL,
    TERMINAL_TOOL,
    _SANDBOX_PATHS_NOTE,
    _bash_exec,
)


def test_note_does_not_claim_full_network_isolation():
    lowered = _SANDBOX_PATHS_NOTE.lower()
    # The old absolute claim is false for the python tool (allowlisted egress).
    assert "cannot reach other machines or remote hosts" not in lowered
    # It should instead scope the block to private/arbitrary hosts and name the
    # allowlist so the model still fetches valid public URLs.
    assert "allowlist" in lowered
    assert "github.com" in lowered and "huggingface.co" in lowered
    assert "arbitrary" in lowered or "private" in lowered


def test_note_does_not_claim_project_sandbox_starts_empty():
    lowered = _SANDBOX_PATHS_NOTE.lower()
    # Project sandboxes are shared across threads, so these are inaccurate.
    assert "starts empty" not in lowered
    assert "persists only for this conversation" not in lowered
    # It should acknowledge that project files may already be present.
    assert "project" in lowered


def test_note_does_not_claim_local_files_are_inaccessible():
    lowered = _SANDBOX_PATHS_NOTE.lower()
    # On a locally hosted Studio the child runs on the host with no filesystem
    # isolation on this branch (Landlock is a separate change), and cat is an
    # auto-safe terminal command, so an exact local path is readable; the note
    # must not claim otherwise. It should frame the workdir as the default work
    # location instead.
    assert "cannot see the user's own computer" not in lowered
    assert "default location for your work" in lowered


def test_note_is_appended_to_both_tool_descriptions():
    assert PYTHON_TOOL["function"]["description"].endswith(_SANDBOX_PATHS_NOTE)
    assert TERMINAL_TOOL["function"]["description"].endswith(_SANDBOX_PATHS_NOTE)


def test_note_scopes_network_block_to_the_terminal_not_all_shell_commands():
    # The terminal leaves the network namespace intact and does not block git/pip,
    # so the note must not claim shell network is fully blocked; it names the
    # commands that are blocked (curl / wget) and attributes the host allowlist to
    # the python tool.
    lowered = _SANDBOX_PATHS_NOTE.lower()
    assert "shell network commands are blocked" not in lowered
    assert "curl" in lowered and "wget" in lowered
    assert "python tool can fetch" in lowered


def test_note_distinguishes_attachments_from_sandbox_uploads():
    # Chat/project document uploads land in the RAG store, not the sandbox workdir,
    # so the note must not tell the model that uploaded files appear via `ls`.
    lowered = _SANDBOX_PATHS_NOTE.lower()
    assert "was uploaded" not in lowered
    assert "or that is uploaded" not in lowered
    assert "retrieved separately" in lowered
    assert "attach" in lowered


def test_blocked_network_command_message_points_to_python_egress():
    # A blocked curl/wget must not tell the model the whole sandbox is offline: the
    # python tool can still fetch allowlisted public hosts.
    msg = _bash_exec("curl https://raw.githubusercontent.com/foo/bar/main/x.py").lower()
    assert "blocked command" in msg
    assert "cannot reach other machines or remote hosts" not in msg
    assert "python" in msg
    assert "github.com" in msg


def test_blocked_network_command_message_scopes_claim_to_the_command():
    # The block is by command name; the terminal keeps its network namespace and
    # does not block git/pip, so `git clone http://<private-host>/repo` can still
    # reach a private host. The message must not assert the destination itself is
    # unreachable, and must attribute the block to the command by name.
    msg = _bash_exec("wget http://10.0.0.5/internal/repo.tar.gz").lower()
    assert "by name" in msg
    assert "not reachable" not in msg
    assert "are not reachable" not in msg
    assert "hosts are not reachable" not in msg


def test_blocked_network_command_message_does_not_recommend_chat_upload():
    # Chat attachments land in the RAG store, not the sandbox workdir, so telling
    # the user to upload the file to chat is a dead end. The message must instead
    # point at an accessible path or placing the file in the working directory.
    msg = _bash_exec("curl https://raw.githubusercontent.com/foo/bar/main/x.py").lower()
    assert "upload" not in msg
    assert "working directory" in msg or "path the sandbox can read" in msg


def test_code_execution_nudge_does_not_deny_local_file_access():
    # On this no-Landlock branch the child runs on the host with only cwd set, so an
    # exact local path the user supplies is readable. The code-execution nudge must
    # frame the workdir as the default work location, not assert the user's own
    # computer is inaccessible, and it must still allow an exact path.
    from routes.inference import _TOOL_CODE_TIP

    lowered = _TOOL_CODE_TIP.lower()
    assert "cannot access the user's own computer" not in lowered
    assert "default" in lowered and "location for your work" in lowered
    assert "give an exact path" in lowered
