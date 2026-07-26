# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from hub.storage import scan_folders
from storage import studio_db
from utils.paths import external_media


_BACKEND_ROOT = Path(__file__).resolve().parent.parent


class _ExistingScanFolderConn:
    def __init__(self):
        self.params = ()

    def execute(
        self,
        _sql,
        params = (),
    ):
        self.params = params
        return self

    def fetchone(self):
        return {"id": 1, "path": self.params[0], "created_at": "fake"}

    def commit(self):
        pass

    def close(self):
        pass


class _HTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _stub_linux_path_checks(monkeypatch, module):
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(module.os.path, "realpath", os.path.normpath)
    monkeypatch.setattr(module.os.path, "expanduser", lambda p: p)
    monkeypatch.setattr(module.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(module.os.path, "isdir", lambda _p: True)
    monkeypatch.setattr(module.os, "access", lambda _p, _mode: True)


def _stub_hub_scan_folder_db(monkeypatch):
    monkeypatch.setattr(scan_folders, "_ensure_schema", lambda _conn: None)
    monkeypatch.setattr(scan_folders, "get_connection", _ExistingScanFolderConn)


def _stub_legacy_scan_folder_db(monkeypatch):
    monkeypatch.setattr(studio_db, "get_connection", _ExistingScanFolderConn)


def test_linux_run_media_policy_accepts_mounted_volume_descendants(monkeypatch):
    monkeypatch.setattr(external_media.platform, "system", lambda: "Linux")

    assert external_media.is_linux_run_media_path("/run/media/dspofu/nvmeB")
    assert external_media.is_linux_run_media_path("/run/media/dspofu/nvmeB/modelsAI/gguf/qwen3.6")


@pytest.mark.parametrize(
    "path",
    [
        "/run",
        "/run/media",
        "/run/media/dspofu",
        "/run/user/1000/models",
        "/run/systemd/private",
        "/run/not-media/dspofu/nvmeB",
    ],
)
def test_linux_run_media_policy_rejects_unrelated_run_paths(monkeypatch, path):
    monkeypatch.setattr(external_media.platform, "system", lambda: "Linux")

    assert not external_media.is_linux_run_media_path(path)


def test_linux_run_media_mount_roots_lists_readable_volume_roots(monkeypatch, tmp_path):
    base = tmp_path / "run" / "media"
    mount = base / "dspofu" / "nvmeB"
    sensitive_mount = base / "dspofu" / ".ssh"
    sensitive_aws_mount = base / "dspofu" / ".aws"
    other_user_mount = base / "other" / "backup"
    incomplete = base / "dspofu-only"
    mount.mkdir(parents = True)
    sensitive_mount.mkdir()
    sensitive_aws_mount.mkdir()
    other_user_mount.mkdir(parents = True)
    incomplete.mkdir()
    monkeypatch.setattr(external_media.platform, "system", lambda: "Linux")

    roots = external_media.linux_run_media_mount_roots(base, user = "dspofu")

    assert roots == [mount.resolve()]


def test_linux_run_media_mount_roots_skips_sensitive_resolved_volume_name(monkeypatch, tmp_path):
    base = tmp_path / "run" / "media"
    normal_mount = base / "dspofu" / "nvmeB"
    sensitive_target = base / "dspofu" / ".config"
    normal_mount.mkdir(parents = True)
    sensitive_target.mkdir()
    alias = base / "dspofu" / "config-alias"
    alias.symlink_to(sensitive_target, target_is_directory = True)
    monkeypatch.setattr(external_media.platform, "system", lambda: "Linux")

    roots = external_media.linux_run_media_mount_roots(base, user = "dspofu")

    assert roots == [normal_mount.resolve()]


def test_linux_run_media_mount_roots_skips_sensitive_resolved_descendant(monkeypatch, tmp_path):
    base = tmp_path / "run" / "media"
    normal_mount = base / "dspofu" / "nvmeB"
    sensitive_descendant = normal_mount / ".ssh" / "models"
    sensitive_descendant.mkdir(parents = True)
    alias = base / "dspofu" / "models-alias"
    alias.symlink_to(sensitive_descendant, target_is_directory = True)
    monkeypatch.setattr(external_media.platform, "system", lambda: "Linux")

    roots = external_media.linux_run_media_mount_roots(base, user = "dspofu")

    assert roots == [normal_mount.resolve()]


def test_hub_scan_folder_accepts_linux_run_media_mount(monkeypatch):
    _stub_linux_path_checks(monkeypatch, scan_folders)
    monkeypatch.setattr(external_media.platform, "system", lambda: "Linux")
    _stub_hub_scan_folder_db(monkeypatch)
    target = "/run/media/dspofu/nvmeB/modelsAI/gguf/qwen3.6"

    row = scan_folders.add_scan_folder(target)

    assert row["path"] == target


@pytest.mark.parametrize(
    "target",
    [
        "/run",
        "/run/media",
        "/run/media/dspofu",
        "/run/user/1000/models",
        "/run/systemd/private",
        "/run/not-media/dspofu/nvmeB",
    ],
)
def test_hub_scan_folder_keeps_unrelated_run_paths_blocked(monkeypatch, target):
    _stub_linux_path_checks(monkeypatch, scan_folders)
    monkeypatch.setattr(external_media.platform, "system", lambda: "Linux")
    _stub_hub_scan_folder_db(monkeypatch)

    with pytest.raises(ValueError, match = "Path under /run is not allowed"):
        scan_folders.add_scan_folder(target)


def test_hub_scan_folder_keeps_sensitive_dirs_blocked_under_run_media(monkeypatch):
    _stub_linux_path_checks(monkeypatch, scan_folders)
    monkeypatch.setattr(external_media.platform, "system", lambda: "Linux")
    _stub_hub_scan_folder_db(monkeypatch)

    with pytest.raises(ValueError, match = "Credential or configuration"):
        scan_folders.add_scan_folder("/run/media/dspofu/nvmeB/.ssh/models")


def test_legacy_scan_folder_accepts_linux_run_media_mount(monkeypatch):
    _stub_linux_path_checks(monkeypatch, studio_db)
    monkeypatch.setattr(external_media.platform, "system", lambda: "Linux")
    _stub_legacy_scan_folder_db(monkeypatch)
    target = "/run/media/dspofu/nvmeB/modelsAI/gguf/qwen3.6"

    row = studio_db.add_scan_folder(target)

    assert row["path"] == target


@pytest.mark.parametrize(
    "target",
    [
        "/run",
        "/run/media",
        "/run/media/dspofu",
        "/run/user/1000/models",
        "/run/systemd/private",
        "/run/not-media/dspofu/nvmeB",
    ],
)
def test_legacy_scan_folder_keeps_unrelated_run_paths_blocked(monkeypatch, target):
    _stub_linux_path_checks(monkeypatch, studio_db)
    monkeypatch.setattr(external_media.platform, "system", lambda: "Linux")
    _stub_legacy_scan_folder_db(monkeypatch)

    with pytest.raises(ValueError, match = "Path under /run is not allowed"):
        studio_db.add_scan_folder(target)


def test_legacy_scan_folder_keeps_sensitive_dirs_blocked_under_run_media(monkeypatch):
    _stub_linux_path_checks(monkeypatch, studio_db)
    monkeypatch.setattr(external_media.platform, "system", lambda: "Linux")
    _stub_legacy_scan_folder_db(monkeypatch)

    with pytest.raises(ValueError, match = "Credential or configuration"):
        studio_db.add_scan_folder("/run/media/dspofu/nvmeB/.aws/models")


def test_legacy_browse_allowlist_includes_linux_run_media_mounts(monkeypatch, tmp_path):
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
    media_root = tmp_path / "run" / "media" / "dspofu" / "nvmeB"
    model_dir = media_root / "modelsAI" / "gguf" / "qwen3.6"
    home.mkdir()
    model_dir.mkdir(parents = True)
    (media_root / ".ssh").mkdir()

    fake_paths = SimpleNamespace(
        hf_default_cache_dir = lambda: tmp_path / "missing-default-hf",
        legacy_hf_cache_dir = lambda: tmp_path / "missing-legacy-hf",
        well_known_model_dirs = lambda: [],
        studio_root = lambda: tmp_path / "missing-studio",
        outputs_root = lambda: tmp_path / "missing-outputs",
        exports_root = lambda: tmp_path / "missing-exports",
    )
    fake_external_media = SimpleNamespace(
        linux_run_media_mount_roots = lambda: [media_root],
        macos_volume_roots = lambda: [],
        windows_drive_roots = lambda: [],
    )
    fake_paths.external_media = fake_external_media
    fake_studio_db = SimpleNamespace(
        list_scan_folders = lambda: [],
        contains_sensitive_path_component = studio_db.contains_sensitive_path_component,
        # The media root is a legitimate mount, not denied; the .ssh 403 below
        # comes from the credential check. A False stub keeps this OS-independent
        # (on macOS tmp_path lives under the denied /private/var).
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

    assert media_root.resolve() in allowlist
    assert ns["_resolve_browse_target"](str(model_dir), allowlist) == model_dir.resolve()

    with pytest.raises(_HTTPException) as exc:
        ns["_resolve_browse_target"](str(media_root / ".ssh"), allowlist)
    assert exc.value.status_code == 403

    ssh_root = media_root / ".ssh"
    with pytest.raises(_HTTPException) as exc_root:
        ns["_resolve_browse_target"](str(ssh_root), [ssh_root])
    assert exc_root.value.status_code == 403
