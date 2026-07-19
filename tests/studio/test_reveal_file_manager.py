# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Platform guards for the reveal-in-file-manager endpoint.

A stock WSL distro has no Linux desktop, so the generic Linux branch
(``xdg-open``) fails there. Under WSL the reveal must route through Windows
interop (``wslpath -w`` + ``explorer.exe``), fall back to ``xdg-open`` when
interop is unavailable, and leave native Linux behavior unchanged.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest


def _find_repo_root() -> Path | None:
    env = os.environ.get("UNSLOTH_REPO_ROOT")
    if env:
        p = Path(env).resolve()
        if (p / "studio" / "backend").is_dir():
            return p
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "studio" / "backend").is_dir():
            return parent
    return None


_REPO_ROOT = _find_repo_root()
if _REPO_ROOT is None:
    pytest.skip(
        "Could not locate studio/backend. Set UNSLOTH_REPO_ROOT or run from "
        "the repository checkout.",
        allow_module_level = True,
    )

_STUDIO_BACKEND = _REPO_ROOT / "studio" / "backend"
if str(_STUDIO_BACKEND) not in sys.path:
    sys.path.insert(0, str(_STUDIO_BACKEND))

pytest.importorskip("fastapi")
pytest.importorskip("huggingface_hub")

try:
    from routes import models as routes_models
    from utils.paths import path_utils
except Exception as exc:
    pytest.skip(
        f"studio backend import unavailable: {exc}", allow_module_level = True
    )

_WINDOWS_PATH = r"\\wsl.localhost\Distro\cache\model.gguf"


@pytest.fixture()
def linux_host(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "name", "posix")


@pytest.fixture()
def spawned(monkeypatch):
    calls = types.SimpleNamespace(run = [], popen = [], run_error = None)

    def fake_run(cmd, **kwargs):
        calls.run.append(list(cmd))
        if calls.run_error is not None:
            raise calls.run_error
        return types.SimpleNamespace(stdout = _WINDOWS_PATH + "\n")

    def fake_popen(cmd, **kwargs):
        calls.popen.append(list(cmd))
        return types.SimpleNamespace()

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return calls


def test_wsl_file_is_selected_in_explorer(linux_host, spawned, monkeypatch, tmp_path):
    monkeypatch.setattr(path_utils, "_IS_WSL", True)
    target = tmp_path / "model.gguf"
    target.write_bytes(b"gguf")
    routes_models._reveal_in_file_manager(target)
    assert spawned.run == [["wslpath", "-w", str(target)]]
    assert spawned.popen == [["explorer.exe", f"/select,{_WINDOWS_PATH}"]]


def test_wsl_directory_opens_in_explorer(linux_host, spawned, monkeypatch, tmp_path):
    monkeypatch.setattr(path_utils, "_IS_WSL", True)
    routes_models._reveal_in_file_manager(tmp_path)
    assert spawned.popen == [["explorer.exe", _WINDOWS_PATH]]


def test_wsl_without_interop_falls_back_to_xdg_open(
    linux_host, spawned, monkeypatch, tmp_path
):
    monkeypatch.setattr(path_utils, "_IS_WSL", True)
    spawned.run_error = FileNotFoundError("wslpath")
    target = tmp_path / "model.gguf"
    target.write_bytes(b"gguf")
    routes_models._reveal_in_file_manager(target)
    assert spawned.popen == [["xdg-open", str(tmp_path)]]


def test_wsl_empty_conversion_falls_back_to_xdg_open(
    linux_host, spawned, monkeypatch, tmp_path
):
    monkeypatch.setattr(path_utils, "_IS_WSL", True)

    def empty_run(cmd, **kwargs):
        return types.SimpleNamespace(stdout = "\n")

    monkeypatch.setattr(subprocess, "run", empty_run)
    routes_models._reveal_in_file_manager(tmp_path)
    assert spawned.popen == [["xdg-open", str(tmp_path)]]


def test_native_linux_keeps_xdg_open_on_parent_directory(
    linux_host, spawned, monkeypatch, tmp_path
):
    monkeypatch.setattr(path_utils, "_IS_WSL", False)
    target = tmp_path / "model.gguf"
    target.write_bytes(b"gguf")
    routes_models._reveal_in_file_manager(target)
    assert spawned.run == []
    assert spawned.popen == [["xdg-open", str(tmp_path)]]
