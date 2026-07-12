# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""System-directory denylist enforcement for the folder browser.

Guards the fix for the two path-traversal holes exposed once the browser
allowlist can contain a whole Windows drive root (C:\\) or a legacy-registered
filesystem root (/): the browse endpoints must re-apply the same
``_denied_path_prefixes()`` policy that ``add_scan_folder`` enforces, so
``/etc``, ``/proc``, ``C:\\Windows`` and ``C:\\Program Files`` stay unbrowseable
even when they descend from an allowlisted root. Windows/macOS branches are
exercised on this POSIX host by AST-extracting the pure helper and backing it
with ``ntpath`` / a mocked ``platform``.
"""

from __future__ import annotations

import ast
import ntpath
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from hub.storage import scan_folders
from storage import studio_db


_BACKEND_ROOT = Path(__file__).resolve().parent.parent


class _HTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _extract_is_denied_windows():
    """is_denied_system_path (+ _denied_path_prefixes) from studio_db.py run
    under faithful Windows semantics (ntpath) on a POSIX host."""
    src = (_BACKEND_ROOT / "storage" / "studio_db.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    funcs = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_denied_path_prefixes", "is_denied_system_path"}
    ]
    module = ast.Module(body=funcs, type_ignores=[])
    ast.fix_missing_locations(module)

    win_os = SimpleNamespace(
        sep="\\",
        environ={
            "SystemRoot": r"C:\Windows",
            "ProgramFiles": r"C:\Program Files",
            "ProgramFiles(x86)": r"C:\Program Files (x86)",
        },
        path=SimpleNamespace(normcase=ntpath.normcase),
    )
    ns = {
        "os": win_os,
        "platform": SimpleNamespace(system=lambda: "Windows"),
        # /run has no Windows analog, so the carve-out is never reached.
        "is_linux_run_media_path": lambda _p: False,
    }
    exec(compile(module, "<extracted studio_db.py>", "exec"), ns)
    return ns["is_denied_system_path"]


# --------------------------------------------------------------------------- #
# is_denied_system_path -- Linux (real helper, this host)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "path",
    ["/etc", "/etc/ssl/private", "/proc", "/proc/1", "/sys", "/dev", "/boot",
     "/run", "/run/systemd/private", "/run/media", "/run/media/dspofu"],
)
def test_is_denied_system_path_linux_denies_system_dirs(monkeypatch, path):
    monkeypatch.setattr(studio_db.platform, "system", lambda: "Linux")
    assert studio_db.is_denied_system_path(path) is True


@pytest.mark.parametrize(
    "path",
    ["/run/media/dspofu/nvmeB", "/run/media/dspofu/nvmeB/models"],
)
def test_is_denied_system_path_linux_allows_run_media_mounts(monkeypatch, path):
    # The removable-media carve-out keeps /run/media/<user>/<volume> browseable.
    # is_linux_run_media_path already keys off the (Linux) host platform.
    monkeypatch.setattr(studio_db.platform, "system", lambda: "Linux")
    assert studio_db.is_denied_system_path(path) is False


@pytest.mark.parametrize(
    "path",
    ["/etc-backup", "/etcetera", "/home/u/models", "/mnt/data", "/devices", "/", "/opt/models"],
)
def test_is_denied_system_path_linux_allows_non_system(monkeypatch, path):
    monkeypatch.setattr(studio_db.platform, "system", lambda: "Linux")
    assert studio_db.is_denied_system_path(path) is False


def test_legacy_and_hub_denylist_agree(monkeypatch):
    monkeypatch.setattr(studio_db.platform, "system", lambda: "Linux")
    monkeypatch.setattr(scan_folders.platform, "system", lambda: "Linux")
    for p in ["/etc", "/proc/1", "/home/u", "/boot", "/opt/x"]:
        assert studio_db.is_denied_system_path(p) == scan_folders.is_denied_system_path(p)


# --------------------------------------------------------------------------- #
# is_denied_system_path -- Windows (ntpath-backed), case-insensitive + collisions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "path",
    [r"C:\Windows", r"C:\Windows\System32", r"c:\windows", r"C:\WINDOWS\Temp",
     r"C:\Program Files", r"C:\Program Files\x", r"C:\Program Files (x86)\y", r"c:\program files"],
)
def test_is_denied_system_path_windows_denies_system_dirs(path):
    is_denied = _extract_is_denied_windows()
    assert is_denied(path) is True


@pytest.mark.parametrize(
    "path",
    [r"C:\Models", r"D:\models", r"C:\WindowsApps", r"C:\ProgramData",
     r"C:\Program Files Extra", r"E:\gguf", r"C:\Users\me\models"],
)
def test_is_denied_system_path_windows_allows_non_system(path):
    is_denied = _extract_is_denied_windows()
    assert is_denied(path) is False


# --------------------------------------------------------------------------- #
# _resolve_browse_target -- real-FS integration (legacy browser)
# --------------------------------------------------------------------------- #
def _extract_resolver():
    """Extract the legacy browse resolver; its inline imports resolve to the
    real storage.studio_db (so is_denied_system_path is the real policy)."""
    src = (_BACKEND_ROOT / "routes" / "models.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {
        "_is_path_inside_allowlist",
        "_normalize_browse_request_path",
        "_browse_relative_parts",
        "_match_browse_child",
        "_resolve_browse_target",
    }
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.Module(body=funcs, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {
        "os": os,
        "Path": Path,
        "Optional": Optional,
        "HTTPException": _HTTPException,
        "logger": SimpleNamespace(warning=lambda *a, **k: None, debug=lambda *a, **k: None),
    }
    exec(compile(module, "<extracted routes/models.py>", "exec"), ns)
    return ns["_resolve_browse_target"]


def test_resolve_browse_target_blocks_etc_via_root():
    # Registering "/" must not make /etc browsable (Codex #3 regression guard).
    resolve = _extract_resolver()
    with pytest.raises(_HTTPException) as exc:
        resolve("/etc", [Path("/")])
    assert exc.value.status_code == 403


def test_resolve_browse_target_blocks_stale_denied_root():
    # A stale scan-folder row pointing straight at a denied dir is refused by
    # the browse-time denylist even though it is its own allowlist root.
    resolve = _extract_resolver()
    with pytest.raises(_HTTPException) as exc:
        resolve("/etc", [Path("/etc")])
    assert exc.value.status_code == 403
    assert "System directories" in exc.value.detail


def test_resolve_browse_target_allows_root_itself():
    resolve = _extract_resolver()
    assert resolve("/", [Path("/")]) == Path("/")


def test_resolve_browse_target_allows_legit_nested_dir(tmp_path):
    resolve = _extract_resolver()
    base = tmp_path / "allowed"
    sub = base / "models" / "gguf"
    sub.mkdir(parents=True)
    assert resolve(str(sub), [base]) == sub.resolve()


def test_resolve_browse_target_symlink_escape_blocked(tmp_path):
    resolve = _extract_resolver()
    base = tmp_path / "allowed"
    base.mkdir()
    link = base / "escape"
    try:
        link.symlink_to("/etc", target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unsupported on this host")
    with pytest.raises(_HTTPException) as exc:
        resolve(str(link), [base])
    assert exc.value.status_code == 403


# --------------------------------------------------------------------------- #
# add_scan_folder -- filesystem-root rejection parity (legacy == hub)
# --------------------------------------------------------------------------- #
def test_legacy_add_scan_folder_rejects_filesystem_root(monkeypatch):
    monkeypatch.setattr(studio_db.platform, "system", lambda: "Linux")
    with pytest.raises(ValueError, match="filesystem root"):
        studio_db.add_scan_folder("/")


def test_hub_add_scan_folder_rejects_filesystem_root(monkeypatch):
    monkeypatch.setattr(scan_folders.platform, "system", lambda: "Linux")
    with pytest.raises(ValueError, match="filesystem root"):
        scan_folders.add_scan_folder("/")
