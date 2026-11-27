# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Choosing a parent for new projects must not silently approve its unrelated files."""

import os
from pathlib import Path
import select
import subprocess
import sys

import pytest

from core.inference import tool_confinement, tools, tool_path_approval as gate
from storage import studio_db
from utils import project_workspace_settings as settings
from utils.account_context import AccountContext, run_as
from utils.paths import storage_roots


@pytest.fixture
def tmp_path(linkable_tmp_path):
    return linkable_tmp_path


@pytest.fixture
def project_homes(tmp_path, monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_PROJECTS_HOME", raising = False)
    monkeypatch.setenv("UNSLOTH_STUDIO_DOCUMENTS_HOME", str(tmp_path / "Documents"))
    monkeypatch.setattr(gate, "_silent_roots_cache", None)
    homes = (tmp_path / "current", tmp_path / "previous")
    for home in homes:
        home.mkdir()
    studio_db.upsert_app_settings(
        {
            settings.PROJECTS_HOME_SETTING_KEY: str(homes[0]),
            settings.PROJECTS_HOME_HISTORY_SETTING_KEY: [str(homes[1])],
        }
    )
    return homes


@pytest.mark.parametrize("location", [0, 1])
def test_saved_project_homes_allow_only_created_project_directories(project_homes, location):
    home = project_homes[location]
    studio_db.upsert_app_settings({settings.PROJECTS_HOME_SETTING_KEY: str(home)})
    project = studio_db.upsert_chat_project(
        {"id": "p-safe", "name": "Safe", "createdAt": 1, "updatedAt": 1}
    )
    studio_db.upsert_app_settings({settings.PROJECTS_HOME_SETTING_KEY: str(project_homes[0])})

    own_file = str(Path(project["rootPath"]) / "sandbox" / "output.txt")
    sibling = str(home / "personal-notes.txt")
    assert gate._path_needs_approval(own_file, writing = True) is False
    assert gate._path_needs_approval(sibling, writing = True) is True
    assert gate._path_needs_approval(sibling) is True
    assert (
        tools.is_high_risk_tool_call("python", {"code": f"open({sibling!r}, 'w').write('x')"})
        is True
    )


def test_default_and_environment_project_grants_stay_unchanged(
    project_homes, tmp_path, monkeypatch
):
    default = storage_roots.documents_root() / "Unsloth Studio" / "Projects"
    assert gate._path_needs_approval(str(default / "output.txt"), writing = True) is False
    override = tmp_path / "operator-projects"
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(override))
    assert gate._path_needs_approval(str(override / "output.txt"), writing = True) is False
    assert gate._path_needs_approval(str(project_homes[0] / "personal.txt"), writing = True) is True


def test_members_keep_only_their_own_project_grant(project_homes):
    member = AccountContext("member-id", "member")
    root = run_as(member, storage_roots.project_workspaces_root)
    assert run_as(member, gate._project_silent_roots) == (str(root),)
    assert (
        run_as(
            member, gate._path_needs_approval, str(project_homes[0] / "personal.txt"), writing = True
        )
        is True
    )


def test_an_empty_project_root_does_not_grant_the_current_directory(project_homes, monkeypatch):
    studio_db.upsert_chat_project(
        {"id": "legacy-empty", "name": "Legacy", "createdAt": 1, "updatedAt": 1}
    )
    conn = studio_db.get_connection()
    try:
        conn.execute("UPDATE chat_projects SET root_path = '' WHERE id = 'legacy-empty'")
        conn.commit()
    finally:
        conn.close()
    unrelated = project_homes[0] / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    path = str(unrelated / "personal-notes.txt")
    assert gate._path_needs_approval(path) is True
    assert gate._path_needs_approval(path, writing = True) is True


@pytest.mark.parametrize(
    "platform, source",
    [
        ("darwin", "interpreter"),
        ("linux", "interpreter"),
        ("darwin", "sandbox-site"),
        ("darwin", "system"),
        ("linux", "system"),
    ],
)
def test_project_homes_cannot_overlap_existing_tool_read_grants(
    tmp_path, monkeypatch, platform, source
):
    from utils.paths import path_utils

    # The platform branches are simulated; do not probe the host's macOS volume API.
    monkeypatch.setattr(path_utils, "macos_volume_ignores_case", lambda path: False)
    parent = tmp_path / "software"
    root = parent / "runtime"
    child = root / "projects"
    child.mkdir(parents = True)
    sibling = tmp_path / "private-projects"
    sibling.mkdir()
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(tool_confinement, "_interpreter_roots", lambda: [])
    monkeypatch.setattr(tool_confinement, "_MACOS_READ_ROOTS", ())
    monkeypatch.setattr(tool_confinement, "_SYSTEM_READ_ROOTS", ())
    monkeypatch.setattr(tools, "_SANDBOX_SITE_DIR", str(tmp_path / "other-site"))
    if source == "interpreter":
        monkeypatch.setattr(tool_confinement, "_interpreter_roots", lambda: [str(root)])
    elif source == "sandbox-site":
        monkeypatch.setattr(tools, "_SANDBOX_SITE_DIR", str(root))
    else:
        attribute = "_MACOS_READ_ROOTS" if platform == "darwin" else "_SYSTEM_READ_ROOTS"
        monkeypatch.setattr(tool_confinement, attribute, (str(root),))

    for location in (parent, root, child):
        assert tool_confinement.project_home_overlaps_tool_read_roots(location), location
    assert not tool_confinement.project_home_overlaps_tool_read_roots(sibling)


def test_linux_private_mounts_are_not_treated_as_shared_software(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(tool_confinement, "_interpreter_roots", lambda: [])
    monkeypatch.setattr(tool_confinement, "_SYSTEM_READ_ROOTS", ("/run",))
    assert not tool_confinement.project_home_overlaps_tool_read_roots(
        Path("/run/media/owner/drive/projects")
    )
    assert tool_confinement.project_home_overlaps_tool_read_roots(Path("/run/software/projects"))


def test_a_symlink_cannot_select_a_shared_interpreter_folder(project_homes, tmp_path, monkeypatch):
    prefix = tmp_path / "interpreter"
    selected = prefix / "projects"
    selected.mkdir(parents = True)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(selected, target_is_directory = True)
    except (NotImplementedError, OSError):
        pytest.skip("the host cannot create directory symlinks")
    monkeypatch.setattr(tool_confinement, "_interpreter_roots", lambda: [str(prefix)])
    previous = settings.get_projects_home_status()
    with pytest.raises(ValueError, match = "software and interpreter"):
        settings.set_projects_home(str(alias))
    assert settings.get_projects_home_status() == previous


@pytest.mark.skipif(sys.platform != "darwin", reason = "macOS volume case behavior")
def test_case_variants_cannot_select_a_shared_interpreter_folder(
    project_homes, tmp_path, monkeypatch
):
    prefix = tmp_path / "SharedInterpreter"
    selected = prefix / "projects"
    selected.mkdir(parents = True)
    alternate = prefix.with_name(prefix.name.swapcase()) / "projects"
    if not alternate.is_dir():
        pytest.skip("the test volume is case-sensitive")
    monkeypatch.setattr(tool_confinement, "_interpreter_roots", lambda: [str(prefix)])
    previous = settings.get_projects_home_status()
    with pytest.raises(ValueError, match = "software and interpreter"):
        settings.set_projects_home(str(alternate))
    assert settings.get_projects_home_status() == previous


@pytest.mark.skipif(
    sys.platform != "darwin" or not tool_confinement.shutil.which("sandbox-exec"),
    reason = "needs macOS sandbox-exec",
)
def test_an_existing_macos_child_cannot_read_projects_created_after_a_location_change(
    project_homes, tmp_path, monkeypatch
):
    prefix = tmp_path / "readable-software"
    unsafe_home = prefix / "projects"
    unsafe_home.mkdir(parents = True)
    public = prefix / "public.txt"
    public.write_text("SHARED_SOFTWARE_VISIBLE\n")
    safe_home = tmp_path / "new-private-projects"
    safe_home.mkdir()
    interpreter_roots = tool_confinement._interpreter_roots()
    monkeypatch.setattr(
        tool_confinement, "_interpreter_roots", lambda: [*interpreter_roots, str(prefix)]
    )
    member = AccountContext("member-id", "member")
    confinement = run_as(member, tool_confinement._macos_confinement, tools._SANDBOX_SITE_DIR)
    assert confinement is not None
    child = subprocess.Popen(
        confinement.wrap(
            [
                "/bin/sh",
                "-c",
                'printf "READY\\n"; IFS= read -r target; /bin/cat "$1"; /bin/cat "$target"',
                "sh",
                str(public),
            ]
        ),
        stdin = subprocess.PIPE,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
        cwd = "/",
    )
    try:
        ready, _, _ = select.select([child.stdout], [], [], 15)
        assert ready, "Sandboxed child did not become ready within 15s"
        # Read only the handshake bytes; readline could block on a partial line.
        assert os.read(child.stdout.fileno(), len(b"READY\n")) == b"READY\n"
        with pytest.raises(ValueError, match = "software and interpreter"):
            settings.set_projects_home(str(unsafe_home))
        settings.set_projects_home(str(safe_home))
        project = studio_db.upsert_chat_project(
            {"id": "p-private", "name": "Private", "createdAt": 1, "updatedAt": 1}
        )
        secret = Path(project["rootPath"]) / "sandbox" / "private.txt"
        secret.write_text("NEW_OWNER_PROJECT_PRIVATE")
        output, _ = child.communicate(str(secret) + "\n", timeout = 15)
        assert "SHARED_SOFTWARE_VISIBLE" in output, output
        assert "NEW_OWNER_PROJECT_PRIVATE" not in output, output
        assert child.returncode != 0
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout = 5)
