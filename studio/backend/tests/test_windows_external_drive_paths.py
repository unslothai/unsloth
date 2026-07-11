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


def test_browse_allowlist_includes_windows_drive_roots(monkeypatch, tmp_path):
    # End-to-end wiring: prove windows_drive_roots() output actually flows into
    # the browse allowlist built by routes/models.py, mirroring the Linux side's
    # test_legacy_browse_allowlist_includes_linux_run_media_mounts.
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
