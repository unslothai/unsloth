# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A packaged Studio runtime (issue #8954): `unsloth studio` run from /opt/unsloth-studio with
UNSLOTH_STUDIO_HOME naming per-user data serves in-process instead of refusing as not set up."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def studio_cli():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from unsloth_cli.commands import studio as studio_cli
    finally:
        sys.path.pop(0)
    return studio_cli


def _venv(root: Path, owned: bool) -> Path:
    venv = root / "unsloth_studio"
    bindir = venv / ("Scripts" if sys.platform == "win32" else "bin")
    bindir.mkdir(parents = True)
    (bindir / ("python.exe" if sys.platform == "win32" else "python")).write_text("")
    if owned:
        (venv / ".unsloth-studio-owned").write_text("")
    return venv


def test_runtime_venv_with_an_empty_user_home_runs_in_process(studio_cli, tmp_path, monkeypatch):
    runtime = _venv(tmp_path / "opt" / "unsloth-studio", owned = True)
    user_home = tmp_path / "home" / ".unsloth" / "studio"
    monkeypatch.setattr(studio_cli, "STUDIO_HOME", user_home)
    monkeypatch.setattr(sys, "prefix", str(runtime))

    assert not studio_cli._running_in_studio_venv(user_home / "unsloth_studio")
    assert studio_cli._running_in_system_runtime()


def test_a_user_home_with_its_own_venv_keeps_the_reexec(studio_cli, tmp_path, monkeypatch):
    runtime = _venv(tmp_path / "opt" / "unsloth-studio", owned = True)
    user_home = tmp_path / "home" / ".unsloth" / "studio"
    _venv(user_home, owned = True)
    monkeypatch.setattr(studio_cli, "STUDIO_HOME", user_home)
    monkeypatch.setattr(sys, "prefix", str(runtime))

    assert not studio_cli._running_in_system_runtime()


def test_the_desktop_named_runtime_wins_over_a_stale_user_venv(studio_cli, tmp_path, monkeypatch):
    runtime_root = tmp_path / "opt" / "unsloth-studio"
    runtime = _venv(runtime_root, owned = True)
    user_home = tmp_path / "home" / ".unsloth" / "studio"
    _venv(user_home, owned = True)
    monkeypatch.setattr(studio_cli, "STUDIO_HOME", user_home)
    monkeypatch.setattr(sys, "prefix", str(runtime))
    monkeypatch.setenv(studio_cli.SYSTEM_RUNTIME_ENV, str(runtime_root))

    assert studio_cli._running_in_system_runtime()
    monkeypatch.setenv(studio_cli.SYSTEM_RUNTIME_ENV, str(tmp_path / "elsewhere"))
    assert not studio_cli._running_in_system_runtime()


def test_a_direct_runtime_launch_exports_the_runtime_tools(studio_cli, tmp_path, monkeypatch):
    runtime_root = tmp_path / "opt" / "unsloth-studio"
    runtime = _venv(runtime_root, owned = True)
    (runtime_root / "llama.cpp").mkdir()
    (runtime_root / "whisper.cpp").mkdir()
    user_home = tmp_path / "home" / ".unsloth" / "studio"
    monkeypatch.setattr(studio_cli, "STUDIO_HOME", user_home)
    monkeypatch.setattr(studio_cli, "_STUDIO_HOME_IS_CUSTOM", True)
    monkeypatch.setattr(sys, "prefix", str(runtime))
    monkeypatch.setattr(studio_cli, "_recorded_master_root", lambda: None)
    monkeypatch.setattr(studio_cli, "_master_root_llama_dir", lambda: None)
    for name in ("UNSLOTH_HOME", "UNSLOTH_LLAMA_CPP_PATH", "UNSLOTH_AUDIO_CPP_PATH"):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(user_home))
    monkeypatch.setenv("UNSLOTH_WHISPER_CPP_PATH", "/custom/whisper.cpp")

    studio_cli._ensure_studio_env_exported()

    assert os.environ["UNSLOTH_LLAMA_CPP_PATH"] == str(runtime_root / "llama.cpp")
    assert os.environ["UNSLOTH_WHISPER_CPP_PATH"] == "/custom/whisper.cpp"
    assert "UNSLOTH_AUDIO_CPP_PATH" not in os.environ


def test_a_foreign_interpreter_still_reports_not_set_up(studio_cli, tmp_path, monkeypatch):
    # A pip-installed CLI in some other venv: no installer marker, nothing to run in-process.
    other = _venv(tmp_path / "project", owned = False)
    monkeypatch.setattr(studio_cli, "STUDIO_HOME", tmp_path / "home" / ".unsloth" / "studio")
    monkeypatch.setattr(sys, "prefix", str(other))

    assert not studio_cli._running_in_system_runtime()
