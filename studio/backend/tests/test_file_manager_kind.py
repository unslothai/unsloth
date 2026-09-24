# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/api/health reports which file manager Reveal would open, so the UI offers it only where a
window can appear: never in a container or on a Linux server with no display."""

from __future__ import annotations

import pytest

from utils.paths import file_manager, path_utils


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(file_manager.sys, "platform", "linux")
    monkeypatch.setattr(file_manager.os, "name", "posix")
    monkeypatch.setattr(path_utils, "_IS_WSL", False)
    monkeypatch.setattr(file_manager, "_in_container", lambda: False)
    for name in ("DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(name, raising = False)
    return monkeypatch


def test_mac_and_windows_name_their_file_managers(monkeypatch):
    monkeypatch.setattr(file_manager.sys, "platform", "darwin")
    assert file_manager.file_manager_kind() == "finder"
    monkeypatch.setattr(file_manager.sys, "platform", "win32")
    monkeypatch.setattr(file_manager.os, "name", "nt")
    assert file_manager.file_manager_kind() == "explorer"


def test_headless_linux_has_none(linux):
    assert file_manager.file_manager_kind() is None


def test_linux_desktop_session(linux):
    linux.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert file_manager.file_manager_kind() == "files"


def test_wsl_reveals_in_windows_explorer(linux):
    linux.setattr(path_utils, "_IS_WSL", True)
    assert file_manager.file_manager_kind() == "explorer"


def test_container_has_none_even_with_a_display(linux):
    linux.setenv("DISPLAY", ":0")
    linux.setattr(file_manager, "_in_container", lambda: True)
    assert file_manager.file_manager_kind() is None
