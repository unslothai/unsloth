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


def test_an_edited_project_copy_is_kept(project, media, copy_mode):
    gp.copy_into_project(media, "p1", "images")
    dest = project / "sandbox" / "images" / "abc123.png"
    dest.write_bytes(b"edited in the project, a different size")
    assert gp.copy_into_project(media, "p1", "images")["already"] is True
    assert dest.read_bytes() == b"edited in the project, a different size"


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


@pytest.fixture(params = [True, False], ids = ["dir_fd", "by_path"])
def copy_mode(request, monkeypatch):
    if request.param and not gp._USE_DIR_FD:
        pytest.skip("no dir_fd support on this platform")
    monkeypatch.setattr(gp, "_USE_DIR_FD", request.param)
    return request.param


def test_a_failed_copy_leaves_no_partial_file(project, media, copy_mode, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(gp.shutil, "copyfileobj", boom)
    monkeypatch.setattr(gp.shutil, "copyfile", boom)
    with pytest.raises(OSError):
        gp.copy_into_project(media, "p1", "images")
    assert list((project / "sandbox" / "images").iterdir()) == []


def test_a_symlinked_media_folder_is_refused(project, media, tmp_path, copy_mode):
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / "sandbox" / "images").symlink_to(outside, target_is_directory = True)
    with pytest.raises(OSError):
        gp.copy_into_project(media, "p1", "images")
    assert list(outside.iterdir()) == []


def test_concurrent_adds_of_one_item_all_succeed(project, media, copy_mode):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: gp.copy_into_project(media, "p1", "images"), range(8)))
    assert all(r["path"].endswith("abc123.png") for r in results)
    folder = project / "sandbox" / "images"
    assert [p.name for p in folder.iterdir()] == ["abc123.png"]
    assert (folder / "abc123.png").read_bytes() == media.read_bytes()


def test_temp_names_are_unique_per_call():
    assert gp._tmp_name("a.png") != gp._tmp_name("a.png")


def test_a_folder_swapped_mid_copy_is_refused_without_dir_fd(project, media, tmp_path, monkeypatch):
    monkeypatch.setattr(gp, "_USE_DIR_FD", False)
    folder = project / "sandbox" / "images"
    folder.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    real_copyfile = gp.shutil.copyfile

    def swap_then_copy(src, dst):
        folder.rename(project / "sandbox" / "images-old")
        folder.symlink_to(outside, target_is_directory = True)
        return real_copyfile(src, dst)

    monkeypatch.setattr(gp.shutil, "copyfile", swap_then_copy)
    with pytest.raises(PermissionError):
        gp.copy_into_project(media, "p1", "images")
    assert list(outside.iterdir()) == []
