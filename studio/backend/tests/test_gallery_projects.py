# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Copying a gallery item into a chat project's folder."""

from __future__ import annotations

import pytest

import core.inference.gallery_projects as gp
import storage.studio_db as studio_db


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "Projects" / "demo-1234"
    (root / "sandbox").mkdir(parents = True)
    record = {"id": "p1", "rootPath": str(root), "sandboxPath": str(root / "sandbox")}
    monkeypatch.setattr(
        studio_db, "ensure_chat_project_workspace", lambda pid: record if pid == "p1" else None
    )
    return root


@pytest.fixture
def media(tmp_path):
    path = tmp_path / "gallery" / "abc123.png"
    path.parent.mkdir()
    path.write_bytes(b"\x89PNG fake bytes")
    return path


def test_the_copy_lands_in_the_projects_sandbox_folder(project, media):
    result = gp.copy_into_project(media, "p1", "images")
    dest = project / "sandbox" / "images" / "abc123.png"
    assert result == {"path": str(dest), "already": False}
    assert dest.read_bytes() == media.read_bytes()
    # A copy, not a move: the gallery keeps its item.
    assert media.exists()


def test_adding_the_same_item_twice_reports_it_is_already_there(project, media):
    gp.copy_into_project(media, "p1", "images")
    assert gp.copy_into_project(media, "p1", "images")["already"] is True
    assert sorted(p.name for p in (project / "sandbox" / "images").iterdir()) == ["abc123.png"]


def test_an_unknown_project_is_refused(project, media):
    with pytest.raises(gp.ProjectNotFound):
        gp.copy_into_project(media, "gone", "images")


def test_a_sandbox_outside_the_project_root_is_refused(tmp_path, media, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(
        studio_db,
        "ensure_chat_project_workspace",
        lambda pid: {"id": pid, "rootPath": str(root), "sandboxPath": str(elsewhere)},
    )
    with pytest.raises(gp.ProjectNotFound):
        gp.copy_into_project(media, "p1", "images")
    assert not (elsewhere / "images").exists()


def test_a_failed_copy_leaves_no_partial_file(project, media, monkeypatch):
    def boom(src, dst):
        open(dst, "wb").write(b"half")
        raise OSError("disk full")

    monkeypatch.setattr(gp.shutil, "copyfile", boom)
    with pytest.raises(OSError):
        gp.copy_into_project(media, "p1", "images")
    assert list((project / "sandbox" / "images").iterdir()) == []
