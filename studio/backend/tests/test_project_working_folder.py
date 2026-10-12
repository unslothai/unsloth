# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A project can run its tools in an existing folder of the user's (#10873, #9321).

The folder is only ever pointed at: the managed workspace stays where it was, a project delete
removes that and nothing else, and a folder that moved or was replaced is never followed.
"""

import os
import platform
import shutil
import stat
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from storage import studio_db


@pytest.fixture
def projects_home(tmp_path):
    """macOS tmp_path is under /private, which the project-delete guard refuses."""
    candidate = tmp_path / "Projects"
    resolved = str(candidate.resolve())
    check = os.path.normcase(resolved) if platform.system() == "Windows" else resolved
    if any(check == p or check.startswith(p + os.sep) for p in studio_db._denied_path_prefixes()):
        candidate = Path.home() / ".unsloth-studio-tests" / uuid.uuid4().hex
    candidate.mkdir(parents = True, exist_ok = True)
    try:
        yield candidate
    finally:
        if ".unsloth-studio-tests" in candidate.parts:
            shutil.rmtree(candidate, ignore_errors = True)


@pytest.fixture
def repo_parent(tmp_path):
    """macOS tmp_path is under /private, which Link folder refuses as a system directory."""
    from hub.storage.scan_folders import is_denied_system_path

    if not is_denied_system_path(os.path.realpath(tmp_path)):
        yield tmp_path
        return
    candidate = Path.home() / ".unsloth-studio-tests" / uuid.uuid4().hex
    candidate.mkdir(parents = True)
    try:
        yield candidate
    finally:
        shutil.rmtree(candidate, ignore_errors = True)


@pytest.fixture
def env(tmp_path, monkeypatch, projects_home, repo_parent):
    from core.inference import tools

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(projects_home))
    monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_HOME", raising = False)
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    monkeypatch.setattr(tools, "_workdirs", {})
    monkeypatch.setattr(tools, "_legacy_sandbox_migrated", True)
    repo = repo_parent / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("print('hi')\n")
    os.chmod(repo, 0o755)
    project = studio_db.upsert_chat_project(
        {"id": "p1", "name": "Repo", "createdAt": 1, "updatedAt": 1}
    )
    return SimpleNamespace(tools = tools, repo = repo, project = project)


def _identity(path):
    st = os.stat(path)
    return (st.st_dev, st.st_ino)


def _use_repo(env):
    return studio_db.set_chat_project_working_folder("p1", str(env.repo), _identity(env.repo))


def test_tools_run_in_the_users_folder_and_back(env):
    managed = env.tools._get_project_workdir("project-p1")
    assert managed == os.path.realpath(env.project["sandboxPath"])
    assert env.tools._get_workdir("project-p1") == managed

    saved = _use_repo(env)
    assert saved["workingFolder"] == str(env.repo)
    assert env.tools._get_project_workdir("project-p1") == os.path.realpath(env.repo)
    assert env.tools._get_workdir("project-p1") == os.path.realpath(env.repo)

    cleared = studio_db.set_chat_project_working_folder("p1", None)
    assert cleared["workingFolder"] is None
    assert env.tools._get_workdir("project-p1") == managed


def test_setting_and_clearing_are_idempotent(env):
    first = _use_repo(env)
    second = _use_repo(env)
    assert first["workingFolder"] == second["workingFolder"] == str(env.repo)
    assert studio_db.set_chat_project_working_folder("p1", None)["workingFolder"] is None
    assert studio_db.set_chat_project_working_folder("p1", None)["workingFolder"] is None
    assert studio_db.set_chat_project_working_folder("missing", None) is None


def test_a_saved_record_cannot_set_or_drop_the_folder(env):
    studio_db.upsert_chat_project({**env.project, "workingFolder": "/etc", "updatedAt": 2})
    assert studio_db.get_chat_project("p1")["workingFolder"] is None
    _use_repo(env)
    studio_db.upsert_chat_project({**env.project, "name": "Renamed", "updatedAt": 3})
    project = studio_db.get_chat_project("p1")
    assert project["name"] == "Renamed"
    assert project["workingFolder"] == str(env.repo)


def test_a_moved_or_replaced_folder_falls_back_to_the_managed_one(env):
    _use_repo(env)
    managed = os.path.realpath(env.project["sandboxPath"])
    moved = env.repo.with_name("repo-moved")
    env.repo.rename(moved)
    assert env.tools._get_project_workdir("project-p1") == managed
    env.repo.mkdir()
    assert env.tools._get_project_workdir("project-p1") == managed
    assert not os.path.exists(env.repo / "main.py")


def test_the_users_folder_keeps_its_mode(env):
    _use_repo(env)
    # Windows reports 0o777 and ignores chmod.
    before = stat.S_IMODE(os.stat(env.repo).st_mode)
    env.tools._get_workdir("project-p1")
    assert stat.S_IMODE(os.stat(env.repo).st_mode) == before


def test_a_folder_cleared_mid_resolve_still_keeps_its_mode(env, monkeypatch):
    # Cleared between the resolve and the chmod.
    before = stat.S_IMODE(os.stat(env.repo).st_mode)
    monkeypatch.setattr(env.tools, "_project_workdir_for", lambda _sid: os.path.realpath(env.repo))
    env.tools._get_workdir("project-p1")
    assert stat.S_IMODE(os.stat(env.repo).st_mode) == before


def test_tools_in_the_users_folder_skip_the_isolation_scan(env, monkeypatch):
    from core.inference import os_sandbox

    # An isolation host whose workdir scan refuses the folder (an external hard link, say).
    def refuse(plan):
        raise os_sandbox.WorkdirUnsafeError(f"the session workdir {plan.workdir} is unsafe")

    monkeypatch.setattr(os_sandbox, "prepare_tool_launch", refuse)
    managed = env.tools.execute_tool("terminal", {"command": "pwd"}, session_id = "project-p1")
    assert "unsafe" in managed
    _use_repo(env)
    out = env.tools.execute_tool("terminal", {"command": "pwd"}, session_id = "project-p1")
    assert os.path.realpath(env.repo) in out and "unsafe" not in out
    ran = env.tools.execute_tool(
        "python", {"code": "import os; print(os.getcwd())"}, session_id = "project-p1"
    )
    assert os.path.realpath(env.repo) in ran


def test_project_delete_never_touches_the_users_folder(env):
    _use_repo(env)
    project = studio_db.get_chat_project("p1")
    studio_db.delete_project_workspace(project)
    assert not os.path.exists(project["rootPath"])
    assert (env.repo / "main.py").read_text() == "print('hi')\n"


def test_other_accounts_never_get_a_host_folder(env, monkeypatch):
    _use_repo(env)
    monkeypatch.setattr(env.tools, "is_owner_context", lambda: False)
    assert env.tools._get_project_workdir("project-p1") is None


def test_agents_md_is_read_from_the_users_folder(env, monkeypatch):
    from core import agents_md

    monkeypatch.setattr(agents_md, "agents_md_enabled", lambda: True)
    (env.repo / "AGENTS.md").write_text("Use tabs.\n")
    assert "Use tabs." not in agents_md.agents_md_text(studio_db.get_chat_project("p1"))
    _use_repo(env)
    assert "Use tabs." in agents_md.agents_md_text(studio_db.get_chat_project("p1"))


def test_add_to_project_lands_where_the_tools_run(env, tmp_path):
    from core.inference import gallery_projects

    image = tmp_path / "gen-1.png"
    image.write_bytes(b"png")
    _use_repo(env)
    added = gallery_projects.copy_into_project(image, "p1", "images")
    assert added["path"] == os.path.join(os.path.realpath(env.repo), "images", "gen-1.png")
    (env.repo / "images" / "gen-2.png").write_bytes(b"mine")
    other = tmp_path / "gen-2.png"
    other.write_bytes(b"generated")
    assert gallery_projects.copy_into_project(other, "p1", "images")["already"] is True
    assert (env.repo / "images" / "gen-2.png").read_bytes() == b"mine"


def test_the_library_does_not_list_the_users_folder(env):
    from core import library

    assert any(s.project_id == "p1" for s in library._sandbox_sessions())
    _use_repo(env)
    assert not any(s.project_id == "p1" for s in library._sandbox_sessions())
    assert not library._sandbox_session_eligible("project-p1")


def _grant(path):
    return SimpleNamespace(canonical_path = path)


def test_route_sets_and_clears_the_folder(env, monkeypatch):
    from routes import chat_history
    from utils import native_path_leases

    monkeypatch.setattr(
        native_path_leases, "verify_native_path_lease", lambda lease, **_: _grant(env.repo)
    )
    body = chat_history.ChatProjectWorkingFolder(nativePathLease = "lease")
    saved = chat_history.set_project_working_folder("p1", body, current_subject = "u")
    assert saved.workingFolder == os.path.realpath(env.repo)
    cleared = chat_history.clear_project_working_folder("p1", current_subject = "u")
    assert cleared.workingFolder is None
    assert (env.repo / "main.py").exists()


def test_route_refuses_studios_own_folders_and_other_accounts(env, monkeypatch, tmp_path):
    from routes import chat_history
    from utils import native_path_leases
    from utils.paths.storage_roots import project_workspaces_root, shared_project_workspaces_root

    body = chat_history.ChatProjectWorkingFolder(nativePathLease = "lease")
    monkeypatch.delenv("UNSLOTH_STUDIO_PROJECTS_HOME")
    monkeypatch.setenv(
        "UNSLOTH_STUDIO_DOCUMENTS_HOME", str(tmp_path.parent / f"{tmp_path.name}-docs")
    )
    other_account = shared_project_workspaces_root() / "Accounts" / ("b" * 32) / "Projects" / "x"
    other_account.mkdir(parents = True)
    project_workspaces_root().mkdir(parents = True, exist_ok = True)
    for reserved in (project_workspaces_root(), other_account, tmp_path):
        monkeypatch.setattr(
            native_path_leases,
            "verify_native_path_lease",
            lambda lease, _p = reserved, **_: _grant(_p),
        )
        with pytest.raises(HTTPException) as exc:
            chat_history.set_project_working_folder("p1", body, current_subject = "u")
        assert exc.value.status_code == 400
    assert studio_db.get_chat_project("p1")["workingFolder"] is None

    monkeypatch.setattr(chat_history, "is_owner_context", lambda: False)
    with pytest.raises(HTTPException) as exc:
        chat_history.set_project_working_folder("p1", body, current_subject = "u")
    assert exc.value.status_code == 403


def test_route_refuses_the_shared_temp_root(env, monkeypatch):
    from routes import chat_history
    from utils import native_path_leases
    from utils.paths import storage_roots

    monkeypatch.setattr(
        native_path_leases,
        "verify_native_path_lease",
        lambda lease, **_: _grant(env.repo),
    )
    monkeypatch.setattr(storage_roots, "shared_tmp_root", lambda: env.repo / "unsloth-studio")
    body = chat_history.ChatProjectWorkingFolder(nativePathLease = "lease")
    with pytest.raises(HTTPException) as exc:
        chat_history.set_project_working_folder("p1", body, current_subject = "u")
    assert exc.value.status_code == 400


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason = "needs a POSIX read-only directory, which root can still write",
)
def test_route_refuses_a_read_only_folder(env, monkeypatch):
    from routes import chat_history
    from utils import native_path_leases

    monkeypatch.setattr(
        native_path_leases,
        "verify_native_path_lease",
        lambda lease, **_: _grant(env.repo),
    )
    os.chmod(env.repo, 0o555)
    try:
        body = chat_history.ChatProjectWorkingFolder(nativePathLease = "lease")
        with pytest.raises(HTTPException) as exc:
            chat_history.set_project_working_folder("p1", body, current_subject = "u")
        assert exc.value.status_code == 400
        assert "not writable" in exc.value.detail
    finally:
        os.chmod(env.repo, 0o755)


def test_route_refuses_a_bad_grant(env, monkeypatch):
    from routes import chat_history
    from utils import native_path_leases

    def refuse(lease, **_):
        raise native_path_leases.NativePathLeaseError("Native path grant was already used.")

    monkeypatch.setattr(native_path_leases, "verify_native_path_lease", refuse)
    body = chat_history.ChatProjectWorkingFolder(nativePathLease = "lease")
    with pytest.raises(HTTPException) as exc:
        chat_history.set_project_working_folder("p1", body, current_subject = "u")
    assert exc.value.status_code == 400
