# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Regression test for the macOS half of the managed-account temp invariant.

Linux grants a confined tool no read on the system temp tree at all
(``_SYSTEM_READ_ROOTS`` names no /tmp), only the account's own ``tmp_root()``
among the writable roots. The macOS profile granted ``file-read*`` on the whole
shared ``/private/tmp``, so a managed tool could read every same-UID temp file
the owner or any other application on the host had left there; the hidden-root
deny only covers ``<tempdir>/unsloth-studio``. The child's TMPDIR already points
inside its own sandbox dir, so the shared tree is not needed for startup.
"""

import re
import sys
from pathlib import Path

import pytest

from core.inference import tool_confinement
from core.inference import tools
from utils.account_context import AccountContext, run_as
from utils.paths.storage_roots import tmp_root

ALICE = AccountContext("alice-id", "alice")

_RULE = re.compile(
    r'^\((allow|deny) (file-[^ (]*(?: file-[^ (]*)*)((?: \(subpath "(?:[^"\\]|\\.)*"\))+)\)$'
)
_SUBPATH = re.compile(r'\(subpath "((?:[^"\\]|\\.)*)"\)')


def _grants_read(profile: str, path: str) -> bool:
    """Later rules win in an SBPL profile, so the last matching rule decides."""
    granted = False
    for line in profile.splitlines():
        match = _RULE.match(line.strip())
        if not match or "file-read*" not in match.group(2).split():
            continue
        for raw in _SUBPATH.findall(match.group(3)):
            root = raw.replace('\\"', '"').replace("\\\\", "\\")
            if path == root or path.startswith(root.rstrip("/") + "/"):
                granted = match.group(1) == "allow"
    return granted


@pytest.fixture(autouse = True)
def _roots(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    return tmp_path


def _profile(monkeypatch) -> str:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(tool_confinement.shutil, "which", lambda name: "/usr/bin/sandbox-exec")
    run_as(ALICE, tools._get_workdir, "chat")
    return run_as(ALICE, tools._account_confinement).wrap(["true"])[2]


def test_macos_profile_grants_no_read_on_the_shared_temp_tree(monkeypatch):
    profile = _profile(monkeypatch)
    account_tmp = str(Path(run_as(ALICE, tmp_root)).resolve())
    # Every temp tree a macOS child can reach: the world-writable one, and the per-user
    # darwin tree launchd points TMPDIR at.
    for root in ("/private/tmp", "/tmp", "/private/var/tmp", "/private/var/folders/ab/cd/T"):
        assert not _grants_read(
            profile, root
        ), f"the profile grants a managed tool read on the shared temp tree {root}"
        outsider = str(Path(root) / "some-other-app.sock")
        assert not _grants_read(
            profile, outsider
        ), f"the profile grants a managed tool read on {outsider}"
    # The account keeps its own temp root, wherever tempfile puts it.
    assert _grants_read(profile, account_tmp)
    assert _grants_read(profile, account_tmp + "/scratch.txt")
