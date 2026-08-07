# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Focused regression tests for the Windows Studio updater launcher transaction."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_COMMAND = REPO_ROOT / "unsloth_cli" / "commands" / "studio.py"
ORIGINAL_LAUNCHER = b"MZ-original-launcher"
REAL_MSVCRT = sys.modules.get("msvcrt")


@pytest.fixture
def studio(monkeypatch):
    """Load studio.py without importing the heavyweight unsloth package."""
    package = types.ModuleType("unsloth_cli")
    package.__path__ = [str(REPO_ROOT / "unsloth_cli")]
    commands = types.ModuleType("unsloth_cli.commands")
    commands.__path__ = [str(REPO_ROOT / "unsloth_cli" / "commands")]
    deps = types.ModuleType("unsloth_cli._studio_deps")
    inference = types.ModuleType("unsloth_cli._inference")
    inference.SpeculativeType = str
    password_prompt = types.ModuleType("unsloth_cli.commands._password_prompt")
    commands._password_prompt = password_prompt

    for name, module in (
        ("unsloth_cli", package),
        ("unsloth_cli.commands", commands),
        ("unsloth_cli._studio_deps", deps),
        ("unsloth_cli._inference", inference),
        ("unsloth_cli.commands._password_prompt", password_prompt),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "unsloth_cli.commands.studio_launcher_transaction_test"
    spec = importlib.util.spec_from_file_location(module_name, STUDIO_COMMAND)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _configure_windows(monkeypatch, studio, tmp_path, *, launcher=ORIGINAL_LAUNCHER):
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    python = scripts / "python.exe"
    python.write_bytes(b"python")
    launcher_path = scripts / "unsloth.exe"
    if launcher is not None:
        launcher_path.write_bytes(launcher)

    lock_state = {"locked": False}
    fake_msvcrt = types.ModuleType("msvcrt")
    fake_msvcrt.LK_NBLCK = 1
    fake_msvcrt.LK_UNLCK = 2

    def locking(_fileno, mode, _length):
        if mode == fake_msvcrt.LK_NBLCK:
            if lock_state["locked"]:
                raise OSError("lock conflict")
            lock_state["locked"] = True
        else:
            lock_state["locked"] = False

    fake_msvcrt.locking = locking
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr(studio.platform, "system", lambda: "Windows")
    monkeypatch.setattr(studio.sys, "executable", str(python))
    monkeypatch.setattr(studio, "_ensure_studio_env_exported", lambda: None)
    monkeypatch.setattr(studio, "_windows_hidden_subprocess_kwargs", lambda: {})
    monkeypatch.setattr(studio, "_refresh_desktop_shortcuts", lambda **_kwargs: None)
    monkeypatch.setattr(studio, "_fail_if_install_damaged", lambda: None)
    for name in (
        "SKIP_STUDIO_BASE",
        "STUDIO_PACKAGE_NAME",
        "STUDIO_LOCAL_INSTALL",
        "STUDIO_LOCAL_REPO",
        "UNSLOTH_TAURI_UPDATE",
    ):
        monkeypatch.delenv(name, raising=False)
    return scripts, launcher_path


def _successful_version_run(calls=None):
    def run(argv, **kwargs):
        if calls is not None:
            calls.append((argv, kwargs))
        return types.SimpleNamespace(returncode=0)

    return run


def _update(studio, *, verify=True):
    studio.update(local=False, package="unsloth", verbose=False, verify=verify)


def test_setup_noop_preserves_launcher_and_removes_backup(monkeypatch, studio, tmp_path):
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path)
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: None)
    calls = []
    monkeypatch.setattr(studio.subprocess, "run", _successful_version_run(calls))

    _update(studio)

    assert launcher.read_bytes() == ORIGINAL_LAUNCHER
    assert not (scripts / "unsloth.exe.update-backup").exists()
    assert calls[0][0] == [str(launcher), "--version"]
    assert calls[0][1]["timeout"] == 10


def test_canonical_launcher_exists_while_setup_runs(monkeypatch, studio, tmp_path):
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path)

    def setup(**_kwargs):
        assert launcher.read_bytes() == ORIGINAL_LAUNCHER
        assert (scripts / "unsloth.exe.update-backup").read_bytes() == ORIGINAL_LAUNCHER

    monkeypatch.setattr(studio, "_run_setup_script", setup)
    monkeypatch.setattr(studio.subprocess, "run", _successful_version_run())

    _update(studio)


def test_setup_failure_restores_original_and_propagates(monkeypatch, studio, tmp_path):
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path)

    def setup(**_kwargs):
        launcher.write_bytes(b"MZ-new-but-incomplete")
        raise RuntimeError("setup failed")

    monkeypatch.setattr(studio, "_run_setup_script", setup)

    with pytest.raises(RuntimeError, match="setup failed"):
        _update(studio)

    assert launcher.read_bytes() == ORIGINAL_LAUNCHER
    assert (scripts / "unsloth.exe.update-backup").read_bytes() == ORIGINAL_LAUNCHER


def test_launcher_deleted_during_setup_is_restored_and_update_fails(monkeypatch, studio, tmp_path):
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path)
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: launcher.unlink())

    with pytest.raises(studio.typer.Exit) as exc:
        _update(studio)

    assert exc.value.exit_code == 1
    assert launcher.read_bytes() == ORIGINAL_LAUNCHER
    assert (scripts / "unsloth.exe.update-backup").exists()


@pytest.mark.parametrize("invalid", [b"", b"not-a-pe"])
def test_invalid_launcher_is_restored_and_update_fails(monkeypatch, studio, tmp_path, invalid):
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path)
    monkeypatch.setattr(
        studio, "_run_setup_script", lambda **_kwargs: launcher.write_bytes(invalid)
    )

    with pytest.raises(studio.typer.Exit):
        _update(studio)

    assert launcher.read_bytes() == ORIGINAL_LAUNCHER
    assert (scripts / "unsloth.exe.update-backup").exists()


@pytest.mark.parametrize("outcome", ["nonzero", "timeout"])
def test_runtime_check_failure_restores_launcher(monkeypatch, studio, tmp_path, outcome):
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path)
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: None)

    def run(argv, **kwargs):
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return types.SimpleNamespace(returncode=7)

    monkeypatch.setattr(studio.subprocess, "run", run)

    with pytest.raises(studio.typer.Exit):
        _update(studio)

    assert launcher.read_bytes() == ORIGINAL_LAUNCHER
    assert (scripts / "unsloth.exe.update-backup").exists()


def test_no_verify_still_checks_launcher_but_skips_integrity_scan(monkeypatch, studio, tmp_path):
    _scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path)
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: None)
    calls = []
    monkeypatch.setattr(studio.subprocess, "run", _successful_version_run(calls))
    integrity_calls = []
    monkeypatch.setattr(studio, "_fail_if_install_damaged", lambda: integrity_calls.append(True))

    _update(studio, verify=False)

    assert calls[0][0] == [str(launcher), "--version"]
    assert integrity_calls == []


def test_legacy_backup_recovers_only_when_launcher_is_missing(monkeypatch, studio, tmp_path):
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path, launcher=None)
    legacy = scripts / "unsloth.exe.deleteme"
    legacy.write_bytes(ORIGINAL_LAUNCHER)

    def setup(**_kwargs):
        assert launcher.read_bytes() == ORIGINAL_LAUNCHER

    monkeypatch.setattr(studio, "_run_setup_script", setup)
    monkeypatch.setattr(studio.subprocess, "run", _successful_version_run())

    _update(studio)

    assert launcher.read_bytes() == ORIGINAL_LAUNCHER
    assert not legacy.exists()
    assert not (scripts / "unsloth.exe.update-backup").exists()


def test_lock_contention_exits_before_setup_or_launcher_mutation(monkeypatch, studio, tmp_path):
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path)
    if REAL_MSVCRT is not None:
        # Exercise the real byte-range lock on Windows and the fake elsewhere.
        monkeypatch.setitem(sys.modules, "msvcrt", REAL_MSVCRT)
    first = studio._WindowsLauncherUpdateTransaction()
    first.__enter__()
    before = launcher.read_bytes()
    backup_before = (scripts / "unsloth.exe.update-backup").read_bytes()
    setup_calls = []
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: setup_calls.append(True))
    try:
        with pytest.raises(studio.typer.Exit) as exc:
            _update(studio)
        assert exc.value.exit_code == 1
        assert setup_calls == []
        assert launcher.read_bytes() == before
        assert (scripts / "unsloth.exe.update-backup").read_bytes() == backup_before
    finally:
        first.__exit__(None, None, None)


def test_non_windows_preserves_call_order_without_launcher_operations(
    monkeypatch, studio, tmp_path
):
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")
    monkeypatch.setattr(studio.sys, "executable", str(tmp_path / "bin" / "python"))
    monkeypatch.setattr(studio, "_ensure_studio_env_exported", lambda: None)
    calls = []
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: calls.append("setup"))
    monkeypatch.setattr(studio, "_fail_if_install_damaged", lambda: calls.append("verify"))
    monkeypatch.setattr(
        studio, "_refresh_desktop_shortcuts", lambda **_kwargs: calls.append("refresh")
    )
    for name in (
        "SKIP_STUDIO_BASE",
        "STUDIO_PACKAGE_NAME",
        "STUDIO_LOCAL_INSTALL",
        "STUDIO_LOCAL_REPO",
        "UNSLOTH_TAURI_UPDATE",
    ):
        monkeypatch.delenv(name, raising=False)

    _update(studio)

    assert calls == ["setup", "verify", "refresh"]
    assert list(tmp_path.rglob("unsloth.exe*")) == []
