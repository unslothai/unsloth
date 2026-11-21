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
        "rm -fv --interactive=never my-report.txt",
        "rm 'my file.txt'",
        "rm -- -weird.txt",
        "rm ~/notes.txt",
        "rm notes.txt\n",
        "  rm notes.txt  ",
    ],
)
def test_plain_rm_is_approvable(command):
    assert tools._find_blocked_commands(command) == {"rm"}
    assert tools._is_approvable_rm(command)


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
        "rm -\\\nrf build",
        "rm -{r,f} build",
        "rm {-rf,build}",
        "rm -?f build",
        "rm *",
        "rm *.log",
        "ls | xargs rm",
        "./rm build -delete",
        "PATH=. rm build",
        "rm() { find build -delete; }; rm",
        "bash -c 'rm -rf build'",
        "find . -name x -exec rm -rf {} +",
        "X=-rf; rm $X build",
        "rm $(echo -rf) build",
        "rm `echo -rf` build",
        "rm a.txt && rm -rf build",
        "rm -f",
        "rm --",
        "rm notes.txt\nrm -rf build",
        "rm -i notes.txt",
        "rm -fI a b c d",
        "rm --interactive notes.txt",
        "rm --interactive=once a b c d",
    ],
)
def test_anything_but_a_plain_rm_is_not_approvable(command):
    assert "rm" in tools._find_blocked_commands(command)
    assert not tools._is_approvable_rm(command)


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

    open(target, "w").close()
    tools._bash_exec("rm notes.txt\n", None, 60, _SESSION, host_access_approved = True)
    assert not os.path.exists(target)


@posix_only
@pytest.mark.parametrize(
    "command",
    ["rm -rf build", "rm -r build", "rm --recursive build", "rm *", "ls | xargs rm"],
)
def test_approved_recursive_rm_is_refused(command):
    workdir = tools._get_workdir(_SESSION)
    tree = os.path.join(workdir, "build")
    os.makedirs(os.path.join(tree, "sub"), exist_ok = True)
    open(os.path.join(tree, "sub", "f.txt"), "w").close()
    # A file named like the flag, which a glob or xargs would hand rm as `-rf`.
    open(os.path.join(workdir, "-rf"), "w").close()

    out = tools._bash_exec(command, None, 60, _SESSION, host_access_approved = True)
    assert out == tools._APPROVED_RM_REFUSED
    assert os.path.exists(os.path.join(tree, "sub", "f.txt"))


@posix_only
@pytest.mark.parametrize(
    "command,expected",
    [("sudo true", "sudo"), ("rm notes.txt && sudo true", "rm, sudo")],
)
def test_approval_does_not_lift_other_blocked_commands(command, expected):
    out = tools._bash_exec(command, None, 60, _SESSION, host_access_approved = True)
    assert out == f"Blocked command(s) for safety: {expected}"
