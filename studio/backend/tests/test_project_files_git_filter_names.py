# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Project "changes" runs git in a folder a managed account's tools write: a filter driver from the
repository's own config must not run on the server, unconfined."""

import os
import platform
import shutil
from pathlib import Path

import pytest

from core import project_files
from core.inference import tools
from utils.account_context import run_as

from .test_account_tool_confinement import BOB, LANDLOCK, auth_env, isolated, matrix  # noqa: F401
from .test_project_files import _git, _init_repo

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason = "git is not installed")
posix = pytest.mark.skipif(platform.system() == "Windows", reason = "uses a shell script")


def _plant_filter(repo: Path, name: str, marker: Path) -> None:
    script = repo / "evil.sh"
    script.write_text(f"#!/bin/sh\ntouch '{marker}'\ncat\n")
    script.chmod(0o755)
    _git(repo, "config", f"filter.{name}.clean", str(script))
    (repo / ".gitattributes").write_text(f"*.py filter={name}\n")
    # Same size, new mtime: git must re-hash, which runs the clean filter.
    (repo / "edit.py").write_text("EDIT.py\n")
    os.utime(repo / "edit.py", (2_000_000_000, 2_000_000_000))


@needs_git
@posix
def test_a_filter_name_with_an_equals_sign_is_not_run(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    marker = tmp_path / "ran"
    _plant_filter(repo, "x=y", marker)
    project_files.git_changes(str(repo))
    assert not marker.exists()


@needs_git
@posix
@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
def test_a_managed_accounts_git_runs_under_its_tool_confinement(tmp_path, monkeypatch):
    workdir = Path(run_as(BOB, tools._get_workdir, "chat"))
    repo = workdir / "repo"
    _init_repo(repo)
    marker = tmp_path / "ran"  # outside every root the account may write
    _plant_filter(repo, "evil", marker)
    # As if the account rewrote .git/config between the filter scan and `git status`.
    monkeypatch.setattr(project_files, "_disabled_filters", lambda *a, **k: [])
    out = run_as(BOB, project_files.git_changes, str(repo))
    assert not marker.exists()
    # The account still sees its changes.
    assert out["repo"] is True
    assert "edit.py" in [c["path"] for c in out["changes"]]
