# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# tao forces Wayland CSD only for windows created undecorated; a later set_decorations(false) leaves KWin's frame.

import json
from pathlib import Path


TAURI = Path(__file__).resolve().parents[2] / "studio/src-tauri"


def _config(name: str) -> dict:
    return json.loads((TAURI / name).read_text(encoding = "utf-8"))


def _base_main_window() -> dict:
    windows = _config("tauri.conf.json")["app"]["windows"]
    return next(window for window in windows if window["label"] == "main")


def test_shared_main_window_stays_decorated_for_macos() -> None:
    assert _base_main_window()["decorations"] is True


def test_only_linux_overrides_the_window_list() -> None:
    for platform in ("windows", "macos"):
        assert "windows" not in _config(f"tauri.{platform}.conf.json").get("app", {}), platform


def test_linux_main_window_matches_shared_one_except_decorations() -> None:
    # RFC 7396 merge replaces arrays wholesale, so Linux restates the whole entry.
    linux_windows = _config("tauri.linux.conf.json")["app"]["windows"]
    assert linux_windows == [{**_base_main_window(), "decorations": False}]
