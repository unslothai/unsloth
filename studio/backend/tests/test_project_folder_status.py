# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os

import pytest

from storage import studio_db
from utils.account_context import OWNER, as_api_caller, run_as
from utils.paths.folder_info import git_head_branch, tilde_path

_OPT_IN = "UNSLOTH_STUDIO_ALLOW_API_FULL_ACCESS"


def _project(folder):
    st = os.stat(folder)
    return {
        "id": "p1",
        "workingFolder": os.path.realpath(folder),
        "workingFolderIdentity": f"{st.st_dev}:{st.st_ino}",
    }


def test_status_is_none_without_a_chosen_folder():
    assert studio_db.project_working_folder_status({"id": "p1", "workingFolder": None}) is None
    assert studio_db.project_working_folder_status(None) is None


def test_status_follows_the_folder_on_disk(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    project = _project(repo)
    assert studio_db.project_working_folder_status(project) == "ok"
    assert studio_db.available_project_working_folder(project) == os.path.realpath(repo)

    repo.rename(tmp_path / "moved")
    assert studio_db.project_working_folder_status(project) == "missing"
    assert studio_db.available_project_working_folder(project) is None

    repo.mkdir()
    (repo / "x").write_text("x")
    if _project(repo)["workingFolderIdentity"] != project["workingFolderIdentity"]:
        assert studio_db.project_working_folder_status(project) == "replaced"
        assert studio_db.available_project_working_folder(project) is None


def test_a_file_or_link_at_the_path_counts_as_replaced(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    project = _project(repo)
    repo.rmdir()
    repo.write_text("not a folder")
    assert studio_db.project_working_folder_status(project) == "replaced"


def test_an_api_key_never_reaches_the_chosen_folder(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    project = _project(repo)
    monkeypatch.delenv(_OPT_IN, raising = False)
    api = as_api_caller(OWNER)
    assert run_as(api, studio_db.available_project_working_folder, project) is None
    assert run_as(api, studio_db.project_working_folder_status, project) == "ok"

    monkeypatch.setenv(_OPT_IN, "1")
    assert run_as(api, studio_db.available_project_working_folder, project) == os.path.realpath(
        repo
    )


def test_project_response_carries_status_display_and_branch(tmp_path, monkeypatch):
    from routes import chat_history

    home = tmp_path / "home"
    repo = home / "code" / "app"
    (repo / ".git").mkdir(parents = True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/feature/login\n")
    monkeypatch.setenv("HOME", str(home.resolve()))
    monkeypatch.setenv("USERPROFILE", str(home.resolve()))
    out = chat_history._project_out(
        {**_project(repo), "name": "App", "createdAt": 1, "updatedAt": 1}
    )
    assert out.workingFolderStatus == "ok"
    assert out.workingFolderDisplay == os.path.join("~", "code", "app")
    assert out.workingFolderBranch == "feature/login"

    managed = chat_history._project_out(
        {"id": "p2", "name": "Managed", "createdAt": 1, "updatedAt": 1}
    )
    assert managed.workingFolderStatus is None
    assert managed.workingFolderDisplay is None
    assert managed.workingFolderBranch is None


def test_a_missing_folder_reports_no_branch(tmp_path):
    from routes import chat_history

    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents = True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    project = {**_project(repo), "name": "App", "createdAt": 1, "updatedAt": 1}
    repo.rename(tmp_path / "elsewhere")
    out = chat_history._project_out(project)
    assert out.workingFolderStatus == "missing"
    assert out.workingFolderBranch is None


@pytest.mark.parametrize(
    ("path", "home", "expected"),
    [
        ("/Users/me/code/app", "/Users/me", "~/code/app"),
        ("/Users/me", "/Users/me", "~"),
        ("/Users/meow/code", "/Users/me", "/Users/meow/code"),
        ("/srv/data", "/Users/me", "/srv/data"),
        ("/srv/data", "/", "/srv/data"),
        ("C:\\Users\\me\\code", "C:\\Users\\me", "~\\code"),
    ],
)
def test_tilde_path(path, home, expected):
    assert tilde_path(path, home) == expected


def test_git_head_branch(tmp_path):
    assert git_head_branch(str(tmp_path)) is None

    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents = True)
    head = repo / ".git" / "HEAD"
    head.write_text("ref: refs/heads/main\n")
    assert git_head_branch(str(repo)) == "main"
    head.write_text("0123456789abcdef0123456789abcdef01234567\n")
    assert git_head_branch(str(repo)) == "0123456"
    head.write_text("garbage")
    assert git_head_branch(str(repo)) is None


def test_git_head_branch_follows_a_worktree_pointer(tmp_path):
    gitdir = tmp_path / "main" / ".git" / "worktrees" / "wt"
    gitdir.mkdir(parents = True)
    (gitdir / "HEAD").write_text("ref: refs/heads/topic\n")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {gitdir}\n")
    assert git_head_branch(str(worktree)) == "topic"
