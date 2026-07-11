# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from pathlib import Path

from utils.paths import external_media


def _stub_windows(monkeypatch, existing_drives):
    """Simulate Windows exposing only *existing_drives* (e.g. {"C", "D"}) as
    readable drive roots, so the test does not depend on the host's real FS."""
    monkeypatch.setattr(external_media.platform, "system", lambda: "Windows")
    present = {f"{d.upper()}:\\" for d in existing_drives}
    monkeypatch.setattr(external_media.os.path, "isdir", lambda p: str(p) in present)
    monkeypatch.setattr(external_media.os, "access", lambda p, _mode: str(p) in present)


def test_windows_drive_roots_empty_off_windows(monkeypatch):
    # Regression guard: the helper is a no-op on Linux/macOS so it can never
    # change the folder-browser allowlist on the platforms CI actually runs on.
    monkeypatch.setattr(external_media.platform, "system", lambda: "Linux")
    assert external_media.windows_drive_roots() == []
    monkeypatch.setattr(external_media.platform, "system", lambda: "Darwin")
    assert external_media.windows_drive_roots() == []


def test_windows_drive_roots_lists_readable_drives(monkeypatch):
    _stub_windows(monkeypatch, {"C", "D", "E"})

    roots = external_media.windows_drive_roots(drive_letters = "CDEF")

    # F is absent, so it is skipped; the rest are exposed in order.
    assert roots == [Path("C:\\"), Path("D:\\"), Path("E:\\")]


def test_windows_drive_roots_skips_absent_and_unreadable(monkeypatch):
    _stub_windows(monkeypatch, {"C"})

    roots = external_media.windows_drive_roots(drive_letters = "CDE")

    assert roots == [Path("C:\\")]


def test_windows_drive_roots_ignores_bad_letters_and_dedupes(monkeypatch):
    _stub_windows(monkeypatch, {"C", "D"})

    roots = external_media.windows_drive_roots(
        drive_letters = ["c:", "C", "D", "1", "AB", "", "  d  "],
    )

    assert roots == [Path("C:\\"), Path("D:\\")]
