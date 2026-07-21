# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""System-directory denylist enforcement for the folder browser.

Once the allowlist can hold a whole Windows drive root (C:\\) or a legacy /
root, the browse endpoints must re-apply the ``_denied_path_prefixes()`` policy
``add_scan_folder`` enforces, so /etc, /proc, C:\\Windows, C:\\Program Files stay
unbrowseable even under an allowlisted root. Windows/macOS branches run on this
POSIX host by AST-extracting the pure helper with ``ntpath`` / a mocked ``platform``.
"""

from __future__ import annotations

import ast
import ntpath
import os
import posixpath
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from hub.storage import scan_folders
from storage import studio_db
from utils.paths.external_media import is_local_filesystem_root


_BACKEND_ROOT = Path(__file__).resolve().parent.parent


class _HTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _extract_is_denied_windows():
    """is_denied_system_path (+ _denied_path_prefixes) from studio_db.py under Windows semantics (ntpath) on a POSIX host."""
    src = (_BACKEND_ROOT / "storage" / "studio_db.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)
    funcs = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_denied_path_prefixes", "is_denied_system_path"}
    ]
    module = ast.Module(body = funcs, type_ignores = [])
    ast.fix_missing_locations(module)

    win_os = SimpleNamespace(
        sep = "\\",
        environ = {
            "SystemRoot": r"C:\Windows",
            "ProgramFiles": r"C:\Program Files",
            "ProgramFiles(x86)": r"C:\Program Files (x86)",
        },
        path = SimpleNamespace(normcase = ntpath.normcase),
    )
    ns = {
        "os": win_os,
        "platform": SimpleNamespace(system = lambda: "Windows"),
        # /run has no Windows analog, so the carve-out is never reached.
        "is_linux_run_media_path": lambda _p: False,
    }
    exec(compile(module, "<extracted studio_db.py>", "exec"), ns)
    return ns["is_denied_system_path"]


# is_denied_system_path -- Linux (real helper, this host)
@pytest.mark.parametrize(
    "path",
    [
        "/etc",
        "/etc/ssl/private",
        "/proc",
        "/proc/1",
        "/sys",
        "/dev",
        "/boot",
        "/run",
        "/run/systemd/private",
        "/run/media",
        "/run/media/dspofu",
    ],
)
def test_is_denied_system_path_linux_denies_system_dirs(monkeypatch, path):
    monkeypatch.setattr(studio_db.platform, "system", lambda: "Linux")
    assert studio_db.is_denied_system_path(path) is True


@pytest.mark.parametrize(
    "path",
    ["/run/media/dspofu/nvmeB", "/run/media/dspofu/nvmeB/models"],
)
def test_is_denied_system_path_linux_allows_run_media_mounts(monkeypatch, path):
    # The /run/media/<user>/<volume> carve-out keeps removable media browseable.
    monkeypatch.setattr(studio_db.platform, "system", lambda: "Linux")
    assert studio_db.is_denied_system_path(path) is False


@pytest.mark.parametrize(
    "path",
    [
        "/etc-backup",
        "/etcetera",
        "/home/u/models",
        "/mnt/data",
        "/devices",
        "/",
        "/opt/models",
    ],
)
def test_is_denied_system_path_linux_allows_non_system(monkeypatch, path):
    monkeypatch.setattr(studio_db.platform, "system", lambda: "Linux")
    assert studio_db.is_denied_system_path(path) is False


def test_legacy_and_hub_denylist_agree(monkeypatch):
    monkeypatch.setattr(studio_db.platform, "system", lambda: "Linux")
    monkeypatch.setattr(scan_folders.platform, "system", lambda: "Linux")
    for p in ["/etc", "/proc/1", "/home/u", "/boot", "/opt/x"]:
        assert studio_db.is_denied_system_path(p) == scan_folders.is_denied_system_path(
            p
        )


# is_denied_system_path -- Windows (ntpath-backed), case-insensitive + collisions
@pytest.mark.parametrize(
    "path",
    [
        r"C:\Windows",
        r"C:\Windows\System32",
        r"c:\windows",
        r"C:\WINDOWS\Temp",
        r"C:\Program Files",
        r"C:\Program Files\x",
        r"C:\Program Files (x86)\y",
        r"c:\program files",
    ],
)
def test_is_denied_system_path_windows_denies_system_dirs(path):
    is_denied = _extract_is_denied_windows()
    assert is_denied(path) is True


@pytest.mark.parametrize(
    "path",
    [
        r"C:\Models",
        r"D:\models",
        r"C:\WindowsApps",
        r"C:\ProgramData",
        r"C:\Program Files Extra",
        r"E:\gguf",
        r"C:\Users\me\models",
    ],
)
def test_is_denied_system_path_windows_allows_non_system(path):
    is_denied = _extract_is_denied_windows()
    assert is_denied(path) is False


# _resolve_browse_target -- real-FS integration (legacy browser)
def _extract_resolver():
    """Extract the legacy browse resolver; its inline imports use the real storage.studio_db policy."""
    src = (_BACKEND_ROOT / "routes" / "models.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)
    names = {
        "_is_path_inside_allowlist",
        "_normalize_browse_request_path",
        "_browse_relative_parts",
        "_match_browse_child",
        "_resolve_browse_target",
    }
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.Module(body = funcs, type_ignores = [])
    ast.fix_missing_locations(module)
    ns = {
        "os": os,
        "Path": Path,
        "Optional": Optional,
        "HTTPException": _HTTPException,
        "logger": SimpleNamespace(
            warning = lambda *a, **k: None, debug = lambda *a, **k: None
        ),
    }
    exec(compile(module, "<extracted routes/models.py>", "exec"), ns)
    return ns["_resolve_browse_target"]


def test_resolve_browse_target_blocks_etc_via_root():
    # Registering "/" must not make /etc browsable (Codex #3 regression guard).
    resolve = _extract_resolver()
    with pytest.raises(_HTTPException) as exc:
        resolve("/etc", [Path("/")])
    assert exc.value.status_code == 403


def test_resolve_browse_target_blocks_stale_denied_root(tmp_path, monkeypatch):
    # A stale scan-folder row pointing at a denied dir is refused by the
    # browse-time denylist even though it is its own allowlist root. A tmp-based
    # denied prefix (+ Linux compare) keeps the assertion OS-agnostic: on macOS
    # tmp lives under the already-denied /private/var, masking the message.
    denied = (tmp_path / "sysfake").resolve()
    denied.mkdir()
    monkeypatch.setattr(studio_db.platform, "system", lambda: "Linux")
    monkeypatch.setattr(studio_db, "_denied_path_prefixes", lambda: [str(denied)])
    resolve = _extract_resolver()
    with pytest.raises(_HTTPException) as exc:
        resolve(str(denied), [denied])
    assert exc.value.status_code == 403
    assert "System directories" in exc.value.detail


def test_resolve_browse_target_allows_root_itself():
    resolve = _extract_resolver()
    assert resolve("/", [Path("/")]) == Path("/")


def test_resolve_browse_target_allows_legit_nested_dir(tmp_path, monkeypatch):
    # Force the Linux denylist so the macOS temp location (under the denied
    # /private/var) doesn't reject the tmp fixture; a normal nested dir must not be over-blocked.
    monkeypatch.setattr(studio_db.platform, "system", lambda: "Linux")
    resolve = _extract_resolver()
    base = tmp_path / "allowed"
    sub = base / "models" / "gguf"
    sub.mkdir(parents = True)
    assert resolve(str(sub), [base]) == sub.resolve()


def test_resolve_browse_target_symlink_escape_blocked(tmp_path):
    resolve = _extract_resolver()
    base = tmp_path / "allowed"
    base.mkdir()
    link = base / "escape"
    try:
        link.symlink_to("/etc", target_is_directory = True)
    except OSError:
        pytest.skip("symlinks unsupported on this host")
    with pytest.raises(_HTTPException) as exc:
        resolve(str(link), [base])
    assert exc.value.status_code == 403


# _is_path_inside_allowlist -- bare POSIX root parity (legacy == hub)
def _extract_is_inside(rel_parts, *, os_module = os):
    """Extract a standalone _is_path_inside_allowlist (os/Path only) so both browsers' copies compare without importing their heavy modules."""
    src = _BACKEND_ROOT.joinpath(*rel_parts).read_text(encoding = "utf-8")
    tree = ast.parse(src)
    funcs = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_is_path_inside_allowlist"
    ]
    module = ast.Module(body = funcs, type_ignores = [])
    ast.fix_missing_locations(module)
    ns = {"os": os_module, "Path": Path}
    exec(compile(module, f"<extracted {'/'.join(rel_parts)}>", "exec"), ns)
    return ns["_is_path_inside_allowlist"]


# ntpath semantics with a no-FS realpath, so UNC containment can be driven on a
# POSIX CI (the real realpath cannot resolve \\server\share off Windows).
_WIN_OS = SimpleNamespace(
    sep = ntpath.sep,
    path = SimpleNamespace(
        realpath = lambda p: ntpath.normpath(str(p)),
        normcase = ntpath.normcase,
        splitdrive = ntpath.splitdrive,
        dirname = ntpath.dirname,
        commonpath = ntpath.commonpath,
    ),
)


def test_legacy_and_hub_allowlist_agree_on_posix_root():
    # A bare "/" allowlist entry must authorize only "/" itself in BOTH
    # browsers, never descend into /var, /root, /home (which the denylist does
    # not cover). Guards the hub browser against authorizing every absolute path.
    legacy = _extract_is_inside(["routes", "models.py"])
    hub = _extract_is_inside(["hub", "services", "models", "folder_browser.py"])
    roots = [Path("/")]
    for tgt in ["/var", "/root", "/home", "/usr", "/opt", "/etc"]:
        assert legacy(Path(tgt), roots) is False
        assert hub(Path(tgt), roots) is False
    # "/" itself stays browseable; only its descendants are withheld.
    assert legacy(Path("/"), roots) is True
    assert hub(Path("/"), roots) is True


def test_hub_allowlist_authorizes_normal_nested_dir(tmp_path):
    # The bare-root special case must not over-block a normal allowlist root's descendants.
    hub = _extract_is_inside(["hub", "services", "models", "folder_browser.py"])
    base = tmp_path / "allowed"
    sub = base / "models" / "gguf"
    sub.mkdir(parents = True)
    assert hub(sub, [base]) is True
    assert hub(base, [base]) is True


# add_scan_folder -- filesystem-root rejection parity (legacy == hub)
def test_legacy_add_scan_folder_rejects_filesystem_root(monkeypatch):
    monkeypatch.setattr(studio_db.platform, "system", lambda: "Linux")
    with pytest.raises(ValueError, match = "filesystem root"):
        studio_db.add_scan_folder("/")


def test_hub_add_scan_folder_rejects_filesystem_root(monkeypatch):
    monkeypatch.setattr(scan_folders.platform, "system", lambda: "Linux")
    with pytest.raises(ValueError, match = "filesystem root"):
        scan_folders.add_scan_folder("/")


# is_local_filesystem_root: reject "/" and "C:\\" (roots above denied system dirs),
# but NOT a UNC share root -- registering \\server\share was allowed before this
# guard and has no system dirs under it. _pathmod drives Windows semantics on POSIX CI.
@pytest.mark.parametrize(
    "path, pathmod, expected",
    [
        # Local filesystem roots -> rejected (True).
        ("/", posixpath, True),
        ("C:\\", ntpath, True),
        ("c:\\", ntpath, True),
        ("D:\\", ntpath, True),
        # UNC share roots -> NOT a local root, stay registerable (False).
        (r"\\server\share", ntpath, False),
        (r"\\nas\models", ntpath, False),
        ("//server/share", ntpath, False),
        # Device / extended-length volume roots -> still local roots (rejected),
        # so neither \\?\C:\ nor a drive-letter-less \\?\Volume{GUID}\ can slip
        # past the guard as if it were a share root.
        (r"\\?\C:" + "\\", ntpath, True),
        (r"\\.\C:" + "\\", ntpath, True),
        (r"\\?\C:", ntpath, True),
        (r"\\.\C:", ntpath, True),
        (r"\\?\Volume{2f8e6d31-0000-0000-0000-100000000000}" + "\\", ntpath, True),
        (r"\\.\Volume{2f8e6d31-0000-0000-0000-100000000000}", ntpath, True),
        # Device-namespace UNC share root -> stays registerable (False).
        (r"\\?\UNC\server\share", ntpath, False),
        # Non-root paths (incl. deep device / extended-length) -> not a root (False).
        ("C:\\Models", ntpath, False),
        (r"\\server\share\models", ntpath, False),
        (r"\\?\C:\Users\me\models", ntpath, False),
        (r"\\?\Volume{2f8e6d31-0000-0000-0000-100000000000}\models", ntpath, False),
        ("/home/user", posixpath, False),
    ],
)
def test_is_local_filesystem_root(path, pathmod, expected):
    assert is_local_filesystem_root(path, _pathmod = pathmod) is expected


def test_both_guards_use_the_shared_local_root_helper():
    # Register-root parity: both browsers reject the same roots via one helper, so a
    # UNC-share exemption can never drift between the legacy and hub code paths.
    legacy_src = (_BACKEND_ROOT / "storage" / "studio_db.py").read_text(
        encoding = "utf-8"
    )
    hub_src = (_BACKEND_ROOT / "hub" / "storage" / "scan_folders.py").read_text(
        encoding = "utf-8"
    )
    assert "is_local_filesystem_root(normalized)" in legacy_src
    assert "is_local_filesystem_root(normalized)" in hub_src


# A registered UNC share root must authorize its own descendants in both browsers.
# os.path.commonpath raises "can't mix absolute and relative" on a bare
# \\server\share, so containment falls back to a boundary-safe prefix test; without
# it, registering a UNC share (now allowed) would 403 every folder under it.
@pytest.mark.parametrize(
    "rel_parts",
    [
        ["routes", "models.py"],
        ["hub", "services", "models", "folder_browser.py"],
    ],
)
def test_unc_share_root_authorizes_its_descendants(rel_parts):
    is_inside = _extract_is_inside(rel_parts, os_module = _WIN_OS)
    root = [Path(r"\\server\share")]
    assert is_inside(Path(r"\\server\share"), root) is True  # the root itself
    assert is_inside(Path(r"\\server\share\models"), root) is True  # direct child
    assert is_inside(Path(r"\\server\share\a\b\c"), root) is True  # deep descendant
    assert is_inside(Path(r"\\SERVER\SHARE\Models"), root) is True  # case-insensitive
    assert is_inside(Path(r"\\server\share2\models"), root) is False  # sibling share
    assert is_inside(Path(r"C:\models"), root) is False  # different volume
