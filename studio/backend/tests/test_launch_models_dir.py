# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth studio --models-dir` registers Custom Folders at startup (#8250)."""

import os

import pytest

from hub.services.models import local_inventory
from storage import studio_db


@pytest.fixture(autouse = True)
def _registrable_tmp(monkeypatch):
    # macOS denies /private/var, where pytest keeps tmp_path.
    from hub.storage import scan_folders
    monkeypatch.setattr(studio_db, "_denied_path_prefixes", lambda: [])
    monkeypatch.setattr(scan_folders, "_denied_path_prefixes", lambda: [])


def _registered() -> set[str]:
    return {row["path"] for row in studio_db.list_scan_folders()}


def test_folders_from_env_are_registered_once(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    monkeypatch.setenv(local_inventory.LAUNCH_MODELS_DIRS_ENV, os.pathsep.join([str(a), str(b)]))

    added = local_inventory.register_launch_scan_folders()
    assert set(added) == {str(a.resolve()), str(b.resolve())}
    assert {str(a.resolve()), str(b.resolve())} <= _registered()

    before = studio_db.list_scan_folders()
    assert local_inventory.register_launch_scan_folders() == []
    assert studio_db.list_scan_folders() == before


def test_rejected_folder_is_skipped_not_fatal(tmp_path):
    good = tmp_path / "good"
    good.mkdir()
    raw = os.pathsep.join([str(tmp_path / "missing"), str(good)])
    assert local_inventory.register_launch_scan_folders(raw) == [str(good.resolve())]
    assert str((tmp_path / "missing").resolve()) not in _registered()


def test_unset_env_is_a_no_op(monkeypatch):
    monkeypatch.delenv(local_inventory.LAUNCH_MODELS_DIRS_ENV, raising = False)
    before = studio_db.list_scan_folders()
    assert local_inventory.register_launch_scan_folders() == []
    assert studio_db.list_scan_folders() == before
