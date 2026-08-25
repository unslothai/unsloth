# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every conversation runs its tools in its own sandbox directory.

Parallel chats lean on this: two conversations can be mid tool call at the same
time, so a shared working directory would let one overwrite the other's files.
The session id is the chat's thread id (or project-<id> for project chats), and
the dir is derived from it here.

UNSLOTH_STUDIO_HOME is redirected per test, so nothing touches the real install.
"""

import os
import sys

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)


@pytest.fixture
def sandbox_root(tmp_path):
    """Where the sandboxes live for these tests: under the studio home."""
    return tmp_path / "sandbox"


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """_get_workdir against a throwaway studio home, with its cache cleared."""
    from core.inference import tools

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_HOME", raising = False)
    monkeypatch.setattr(os.path, "expanduser", lambda path: str(tmp_path))
    monkeypatch.setattr(tools, "_workdirs", {})
    monkeypatch.setattr(tools, "_legacy_sandbox_migrated", True)
    return tools._get_workdir


def test_two_conversations_get_two_directories(workdir, tmp_path):
    a = workdir("thread-alpha")
    b = workdir("thread-beta")
    assert a != b
    assert os.path.basename(a) == "thread-alpha"
    assert os.path.basename(b) == "thread-beta"
    assert os.path.isdir(a) and os.path.isdir(b)
    assert os.path.dirname(a) == os.path.dirname(b) == str(tmp_path / "sandbox")


def test_the_same_conversation_keeps_its_directory(workdir):
    # A later turn, or a tool continuation, must land back in the same place.
    assert workdir("thread-alpha") == workdir("thread-alpha")


def test_a_directory_is_private_to_its_conversation(workdir):
    from core.inference import tools

    a = workdir("thread-alpha")
    b = workdir("thread-beta")
    with open(os.path.join(a, "secret.txt"), "w", encoding = "utf-8") as f:
        f.write("alpha")
    # Our own ownership marker aside, nothing of alpha's is visible here.
    assert os.listdir(b) == [tools._SANDBOX_MARKER]


def test_project_chats_deliberately_share_one_workspace(workdir, monkeypatch):
    # Chats in a project are meant to see each other's files.
    from core.inference import tools
    monkeypatch.setattr(tools, "_project_workdir_info_for", lambda sid: ("/tmp/project-ws", False))
    assert tools._get_workdir("project-abc") == "/tmp/project-ws"


@pytest.mark.parametrize(
    "session_id",
    ["../escape", "a/b", "", "  ", "x" * 65],
)
def test_a_session_id_cannot_escape_the_sandbox_root(workdir, tmp_path, session_id):
    resolved = workdir(session_id) if session_id else workdir(None)
    root = os.path.realpath(str(tmp_path / "sandbox"))
    assert os.path.realpath(resolved).startswith(root + os.sep)
    # A name the filesystem can hold, derived from the id rather than the id
    # itself, so nothing in it can traverse and no two ids share it.
    name = os.path.basename(resolved)
    assert name == "_default" if not session_id else name.startswith("_id-")


def test_no_session_id_falls_back_to_default(workdir):
    assert os.path.basename(workdir(None)) == "_default"


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX permission bits")
def test_directories_are_private_to_the_user(workdir):
    assert os.stat(workdir("thread-alpha")).st_mode & 0o777 == 0o700
