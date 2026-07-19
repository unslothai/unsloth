# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from utils.paths import external_media


_BACKEND_ROOT = Path(__file__).resolve().parent.parent


class _HTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _extract_routes_function(name: str, ns_extra: Optional[dict] = None) -> dict:
    """Exec one top-level function from routes/models.py without importing the module (which pulls in FastAPI)."""
    tree = ast.parse((_BACKEND_ROOT / "routes" / "models.py").read_text(encoding = "utf-8"))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body = [fn], type_ignores = [])
    ast.fix_missing_locations(module)
    ns = {"os": os, "Path": Path, "Optional": Optional}
    if ns_extra:
        ns.update(ns_extra)
    exec(compile(module, "<extracted routes/models.py>", "exec"), ns)
    return ns


def _stub_windows(monkeypatch, existing_drives):
    """Simulate Windows exposing only *existing_drives* (e.g. {"C", "D"}) as readable roots, independent of the host FS.

    Overriding _active_windows_drive_bitmask keeps it deterministic even on a
    real Windows host, where live GetLogicalDrives would return the actual layout."""
    monkeypatch.setattr(external_media.platform, "system", lambda: "Windows")
    mask = sum(1 << (ord(d.upper()) - ord("A")) for d in existing_drives)
    monkeypatch.setattr(external_media, "_active_windows_drive_bitmask", lambda: mask)
    present = {f"{d.upper()}:\\" for d in existing_drives}
    monkeypatch.setattr(external_media.os.path, "isdir", lambda p: str(p) in present)
    monkeypatch.setattr(external_media.os, "access", lambda p, _mode: str(p) in present)


def test_windows_drive_roots_empty_off_windows(monkeypatch):
    # Regression guard: the helper is a no-op on Linux/macOS so it can't change the allowlist on the platforms CI runs on.
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


def test_readable_dir_within_times_out(monkeypatch):
    # A probe that outlives the timeout is reported not-readable, so a hung
    # (disconnected mapped network) drive is skipped instead of blocking.
    import time

    monkeypatch.setattr(external_media.os.path, "isdir", lambda p: time.sleep(5) or True)
    monkeypatch.setattr(external_media.os, "access", lambda p, _mode: True)
    start = time.monotonic()
    ok = external_media._readable_dir_within("Z:\\", timeout = 0.2)
    elapsed = time.monotonic() - start
    assert ok is False
    assert elapsed < 3.0  # returned on the timeout, did not wait out the 5s stall


def test_readable_dir_within_reports_fast_probe(monkeypatch):
    monkeypatch.setattr(external_media.os.path, "isdir", lambda p: True)
    monkeypatch.setattr(external_media.os, "access", lambda p, _mode: True)
    assert external_media._readable_dir_within("C:\\", timeout = 2.0) is True


def test_windows_drive_roots_skips_hung_drive(monkeypatch):
    # A disconnected mapped drive stays set in the bitmask and its os.path.isdir
    # stalls; it must be skipped without stalling enumeration. C answers, D hangs,
    # so only C is listed, bounded by the per-drive timeout, not the stall.
    import time

    monkeypatch.setattr(external_media.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        external_media,
        "_active_windows_drive_bitmask",
        lambda: sum(1 << (ord(d) - ord("A")) for d in "CD"),
    )
    monkeypatch.setattr(external_media, "_DRIVE_PROBE_TIMEOUT_S", 0.2)

    def _isdir(p):
        if str(p) == "D:\\":
            time.sleep(5)  # simulate the reconnect stall
            return True
        return str(p) == "C:\\"

    monkeypatch.setattr(external_media.os.path, "isdir", _isdir)
    monkeypatch.setattr(external_media.os, "access", lambda p, _mode: True)

    start = time.monotonic()
    roots = external_media.windows_drive_roots(drive_letters = "CD")
    elapsed = time.monotonic() - start

    assert roots == [Path("C:\\")]
    assert elapsed < 3.0  # bounded by the per-drive timeout, not the 5s stall


def test_windows_drive_roots_probes_hung_drives_in_parallel(monkeypatch):
    # Several disconnected mapped drives must add ~one timeout total, not one
    # per drive: C answers fast, D/E/F stall. The concurrent probe stays bounded
    # by a single deadline where serial probing would cost ~4x the timeout.
    import time

    monkeypatch.setattr(external_media.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        external_media,
        "_active_windows_drive_bitmask",
        lambda: sum(1 << (ord(d) - ord("A")) for d in "CDEF"),
    )
    timeout = 0.2
    monkeypatch.setattr(external_media, "_DRIVE_PROBE_TIMEOUT_S", timeout)

    def _isdir(p):
        if str(p) == "C:\\":
            return True
        time.sleep(5)  # every other drive simulates a reconnect stall
        return True

    monkeypatch.setattr(external_media.os.path, "isdir", _isdir)
    monkeypatch.setattr(external_media.os, "access", lambda p, _mode: True)

    start = time.monotonic()
    roots = external_media.windows_drive_roots(drive_letters = "CDEF")
    elapsed = time.monotonic() - start

    assert roots == [Path("C:\\")]
    # 3 stalled drives probed in parallel finish within ~1 timeout, well under the ~3*timeout a serial probe would take.
    assert elapsed < 3 * timeout


def test_browse_allowlist_includes_windows_drive_roots(monkeypatch, tmp_path):
    # End-to-end wiring: windows_drive_roots() output flows into the browse
    # allowlist built by routes/models.py, mirroring the Linux media-mounts test.
    tree = ast.parse((_BACKEND_ROOT / "routes" / "models.py").read_text(encoding = "utf-8"))
    function_names = {
        "_build_browse_allowlist",
        "_browse_relative_parts",
        "_is_path_inside_allowlist",
        "_match_browse_child",
        "_normalize_browse_request_path",
        "_resolve_browse_target",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    module = ast.Module(body = functions, type_ignores = [])
    ast.fix_missing_locations(module)

    home = tmp_path / "home"
    drive_root = tmp_path / "D_drive"
    model_dir = drive_root / "modelsAI" / "gguf"
    home.mkdir()
    model_dir.mkdir(parents = True)

    fake_paths = SimpleNamespace(
        hf_default_cache_dir = lambda: tmp_path / "missing-default-hf",
        legacy_hf_cache_dir = lambda: tmp_path / "missing-legacy-hf",
        well_known_model_dirs = lambda: [],
        studio_root = lambda: tmp_path / "missing-studio",
        outputs_root = lambda: tmp_path / "missing-outputs",
        exports_root = lambda: tmp_path / "missing-exports",
    )
    fake_external_media = SimpleNamespace(
        linux_run_media_mount_roots = lambda: [],
        windows_drive_roots = lambda: [drive_root],
    )
    fake_studio_db = SimpleNamespace(
        list_scan_folders = lambda: [],
        contains_sensitive_path_component = lambda _p: False,
        # The simulated D:\ root maps to a tmp_path dir, not a denied system path.
        is_denied_system_path = lambda _p: False,
    )
    monkeypatch.setitem(sys.modules, "utils.paths", fake_paths)
    monkeypatch.setitem(sys.modules, "utils.paths.external_media", fake_external_media)
    monkeypatch.setitem(sys.modules, "storage.studio_db", fake_studio_db)

    ns = {
        "HTTPException": _HTTPException,
        "os": os,
        "Path": Path,
        "Optional": Optional,
        "_safe_is_dir": lambda p: Path(p).is_dir(),
        "_resolve_hf_cache_dir": lambda: tmp_path / "missing-hf",
        "logger": SimpleNamespace(debug = lambda *_args, **_kwargs: None),
    }
    exec(compile(module, "<extracted routes/models.py>", "exec"), ns)

    allowlist = ns["_build_browse_allowlist"]()

    # The simulated Windows drive root is now browsable, and a model dir on it resolves.
    assert drive_root.resolve() in allowlist
    assert ns["_resolve_browse_target"](str(model_dir), allowlist) == model_dir.resolve()


def test_build_browse_allowlist_reuses_passed_roots(monkeypatch, tmp_path):
    # Double-probe fix: a browse request probes the drive/media roots once and
    # passes them in, so _build_browse_allowlist must NOT scan
    # windows_drive_roots() again (a disconnected drive would double the stall).
    tree = ast.parse((_BACKEND_ROOT / "routes" / "models.py").read_text(encoding = "utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_browse_allowlist"
    ]
    module = ast.Module(body = functions, type_ignores = [])
    ast.fix_missing_locations(module)

    drive_root = tmp_path / "D_drive"
    drive_root.mkdir()

    calls = {"drive": 0, "media": 0}

    def _drive_roots():
        calls["drive"] += 1
        return [drive_root]

    def _media_roots():
        calls["media"] += 1
        return []

    fake_paths = SimpleNamespace(
        hf_default_cache_dir = lambda: tmp_path / "missing-default-hf",
        legacy_hf_cache_dir = lambda: tmp_path / "missing-legacy-hf",
        well_known_model_dirs = lambda: [],
        studio_root = lambda: tmp_path / "missing-studio",
        outputs_root = lambda: tmp_path / "missing-outputs",
        exports_root = lambda: tmp_path / "missing-exports",
    )
    fake_external_media = SimpleNamespace(
        linux_run_media_mount_roots = _media_roots,
        windows_drive_roots = _drive_roots,
    )
    fake_studio_db = SimpleNamespace(list_scan_folders = lambda: [])
    monkeypatch.setitem(sys.modules, "utils.paths", fake_paths)
    monkeypatch.setitem(sys.modules, "utils.paths.external_media", fake_external_media)
    monkeypatch.setitem(sys.modules, "storage.studio_db", fake_studio_db)

    ns = {
        "os": os,
        "Path": Path,
        "Optional": Optional,
        "_safe_is_dir": lambda p: Path(p).is_dir(),
        "_resolve_hf_cache_dir": lambda: tmp_path / "missing-hf",
        "logger": SimpleNamespace(debug = lambda *_args, **_kwargs: None),
    }
    exec(compile(module, "<extracted routes/models.py>", "exec"), ns)
    build = ns["_build_browse_allowlist"]

    # Roots passed in -> neither helper is probed, but the roots still flow in.
    allowlist = build([], [drive_root])
    assert calls == {"drive": 0, "media": 0}
    assert drive_root.resolve() in allowlist

    # No args -> each helper is probed exactly once.
    build()
    assert calls == {"drive": 1, "media": 1}


def test_is_path_inside_allowlist_real_descendants_and_siblings(tmp_path):
    # Component-wise containment (commonpath): a genuine descendant is allowed,
    # but a sibling sharing only a string prefix ("models_root_evil" vs
    # "models_root") is not, which the old startswith check could miss.
    ns = _extract_routes_function("_is_path_inside_allowlist")
    root = tmp_path / "models_root"
    child = root / "gguf" / "qwen"
    sibling = tmp_path / "models_root_evil"
    child.mkdir(parents = True)
    sibling.mkdir()

    is_inside = ns["_is_path_inside_allowlist"]
    assert is_inside(root, [root]) is True  # the root itself
    assert is_inside(child, [root]) is True  # a genuine descendant
    assert is_inside(sibling, [root]) is False  # prefix-collision sibling


def test_is_path_inside_allowlist_posix_root_does_not_authorize_descendants(monkeypatch):
    # Regression for the reported POSIX "/" unlock: a bare filesystem root may
    # match itself but must NOT authorize arbitrary descendants such as /etc.
    ns = _extract_routes_function("_is_path_inside_allowlist")
    monkeypatch.setattr(os.path, "realpath", lambda p: str(p))  # keep "/" intact

    is_inside = ns["_is_path_inside_allowlist"]
    assert is_inside("/", ["/"]) is True  # the root itself
    assert is_inside("/etc", ["/"]) is False  # not a licensed descendant
    assert is_inside("/root/models", ["/"]) is False


def test_is_path_inside_allowlist_windows_drive_root_descendants():
    # Exercise the Windows drive-root branch on a POSIX host by backing os.path
    # with ntpath and an identity realpath (the simulated drives don't exist
    # here). A drive root authorizes its descendants; a different drive does not.
    import ntpath

    win_os = SimpleNamespace(
        sep = "\\",
        path = SimpleNamespace(
            normcase = ntpath.normcase,
            realpath = lambda p: str(p),
            splitdrive = ntpath.splitdrive,
            dirname = ntpath.dirname,
            commonpath = ntpath.commonpath,
        ),
    )
    ns = _extract_routes_function("_is_path_inside_allowlist", {"os": win_os})
    is_inside = ns["_is_path_inside_allowlist"]

    assert is_inside("D:\\", ["D:\\"]) is True  # drive root itself
    assert is_inside("D:\\models", ["D:\\"]) is True  # descendant on the drive
    assert is_inside("D:\\models\\gguf", ["D:\\"]) is True  # deeper descendant
    assert is_inside("d:\\models", ["D:\\"]) is True  # case-insensitive drive letter
    assert is_inside("C:\\Users", ["D:\\"]) is False  # different drive
    assert is_inside("D:\\models", ["E:\\"]) is False
