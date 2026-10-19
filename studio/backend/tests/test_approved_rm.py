# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An `rm` the user approved in the prompt runs; a recursive one never does (#9972)."""

from __future__ import annotations

import os
import sys

import pytest

from core.inference import tools

_SESSION = "__LOCALID_approved_rm"

posix_only = pytest.mark.skipif(sys.platform == "win32", reason = "terminal runs bash here")


@pytest.mark.parametrize(
    "command",
    [
        "rm notes.txt",
        "rm -f notes.txt",
        "rm -v a.txt b.txt",
        "rm my-report.txt",
        "ls | xargs rm",
    ],
)
def test_plain_rm_is_lifted_when_approved(command):
    blocked = tools._find_blocked_commands(command)
    assert "rm" in blocked
    assert tools._approved_blocked_commands(command, blocked) == set()


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf build",
        "rm -fr build",
        "rm -Rf build",
        "rm -r build",
        "rm -R build",
        "rm -v -r build",
        "rm --recursive build",
        "rm --rec build",
        "rm '-rf' build",
        'rm "-""r"f build',
        "rm -\\rf build",
        "rm -{r,f} build",
        "rm -?f build",
        "bash -c 'rm -rf build'",
        "find . -name x -exec rm -rf {} +",
        "X=-rf; rm $X build",
        "rm $(echo -rf) build",
        "rm `echo -rf` build",
        "rm notes.txt && grep -r foo .",
    ],
)
def test_recursive_or_expanded_rm_is_never_lifted(command):
    blocked = tools._find_blocked_commands(command)
    assert "rm" in blocked
    assert "rm" in tools._approved_blocked_commands(command, blocked)


def test_approval_lifts_only_rm():
    command = "rm notes.txt && sudo ls"
    assert tools._approved_blocked_commands(command, tools._find_blocked_commands(command)) == {
        "sudo"
    }


@posix_only
def test_approved_rm_deletes_and_unapproved_is_blocked():
    workdir = tools._get_workdir(_SESSION)
    target = os.path.join(workdir, "notes.txt")
    open(target, "w").close()

    out = tools._bash_exec("rm notes.txt", None, 60, _SESSION)
    assert out == "Blocked command(s) for safety: rm"
    assert os.path.exists(target)

    tools._bash_exec("rm notes.txt", None, 60, _SESSION, host_access_approved = True)
    assert not os.path.exists(target)


@posix_only
@pytest.mark.parametrize("command", ["rm -rf build", "rm -r build", "rm --recursive build"])
def test_approved_recursive_rm_is_refused(command):
    workdir = tools._get_workdir(_SESSION)
    tree = os.path.join(workdir, "build")
    os.makedirs(os.path.join(tree, "sub"), exist_ok = True)
    open(os.path.join(tree, "sub", "f.txt"), "w").close()

    out = tools._bash_exec(command, None, 60, _SESSION, host_access_approved = True)
    assert out == tools._RECURSIVE_DELETE_REFUSED
    assert os.path.exists(os.path.join(tree, "sub", "f.txt"))


@posix_only
def test_approval_does_not_lift_other_blocked_commands():
    out = tools._bash_exec("sudo true", None, 60, _SESSION, host_access_approved = True)
    assert out == "Blocked command(s) for safety: sudo"
