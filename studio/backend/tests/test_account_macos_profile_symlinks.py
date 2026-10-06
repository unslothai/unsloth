# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Regression test for the macOS half of the managed-account link invariant.

Linux already withholds LANDLOCK_ACCESS_FS_MAKE_SYM from every writable root
(test_managed_child_cannot_plant_a_link_in_its_own_tree). The macOS profile
granted a plain file-write* on the same roots, and file-write* covers
symlink creation, so a managed tool could plant a link that the unconfined job
child later follows out of the account's roots.

"""

import sys
from pathlib import Path

import pytest

from core.inference import tool_confinement
from core.inference import tools
from utils.account_context import AccountContext, run_as

ALICE = AccountContext("alice-id", "alice")


@pytest.fixture(autouse = True)
def _roots(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    return tmp_path


def test_macos_profile_refuses_symlink_creation_in_the_writable_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(tool_confinement.shutil, "which", lambda name: "/usr/bin/sandbox-exec")
    run_as(ALICE, tools._get_workdir, "chat")
    profile = run_as(ALICE, tools._account_confinement).wrap(["true"])[2]

    deny = "(deny file-write-create (vnode-type SYMLINK))"
    assert deny in profile, "file-write* grants symlink creation unless it is subtracted back out"
    # Later rules win, so the deny has to sit after every writable grant.
    assert profile.rstrip().endswith(deny)
    assert profile.index(deny) > profile.rindex("(allow file-read* file-write* (subpath")
