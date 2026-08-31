# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _load_sandbox_module():
    path = Path(__file__).resolve().parents[1] / "core" / "inference" / "sandbox.py"
    spec = importlib.util.spec_from_file_location("_studio_sandbox_profile_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_python_read_paths_rejects_filesystem_roots(monkeypatch):
    sandbox = _load_sandbox_module()
    root = os.path.abspath(os.sep)

    monkeypatch.setattr(sandbox.sys, "prefix", root)
    monkeypatch.setattr(sandbox.sys, "base_prefix", root)
    monkeypatch.setattr(sandbox.site, "getsitepackages", lambda: [root])
    monkeypatch.setattr(sandbox, "_editable_source_paths", lambda: [root])

    assert root not in sandbox._python_read_paths()


def test_linux_ca_mounts_exclude_private_key_directories(tmp_path, monkeypatch):
    sandbox = _load_sandbox_module()
    monkeypatch.setattr(sandbox, "_linux_bwrap_path", "/usr/bin/bwrap")
    monkeypatch.setattr(sandbox, "_python_read_paths", lambda: [])

    argv = sandbox._linux_bwrap_argv(["/usr/bin/true"], str(tmp_path))
    targets = {
        argv[index + 2]
        for index, token in enumerate(argv)
        if token in {"--ro-bind", "--ro-bind-try", "--bind", "--bind-try"} and index + 2 < len(argv)
    }

    assert "/etc/ssl" not in targets
    assert "/etc/pki" not in targets
    assert "/etc/ssl/private" not in targets
    assert "/etc/pki/tls/private" not in targets
    assert "/etc/ssl/certs" in targets
    assert "/etc/pki/tls/certs" in targets


def test_macos_profile_allows_child_signals_and_dev_null_writes(monkeypatch):
    sandbox = _load_sandbox_module()
    monkeypatch.setattr(sandbox, "_python_read_paths", lambda: [])
    monkeypatch.setattr(sandbox.os.path, "realpath", lambda path: path)
    profile = sandbox._macos_seatbelt_profile("/tmp/work")

    assert "(allow signal (target same-sandbox))" in profile
    assert "(allow signal (target self))" not in profile
    assert "(allow file-write-data" in profile
    assert '(path "/dev/null")' in profile
    assert "(vnode-type CHARACTER-DEVICE)" in profile


def test_macos_ca_reads_exclude_private_key_directories(monkeypatch):
    sandbox = _load_sandbox_module()
    monkeypatch.setattr(sandbox, "_python_read_paths", lambda: [])
    monkeypatch.setattr(sandbox.os.path, "realpath", lambda path: path)
    profile = sandbox._macos_seatbelt_profile("/tmp/work")

    assert '(subpath "/private/etc/ssl")' not in profile
    assert "/private/etc/ssl/private" not in profile
    assert '(literal "/private/etc/ssl/cert.pem")' in profile
    assert '(subpath "/private/etc/ssl/certs")' in profile
