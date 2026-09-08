# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Focused regression tests for the Windows Unsloth updater launcher transaction."""

from __future__ import annotations

import importlib.util
import inspect
import shutil
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


def _configure_windows(
    monkeypatch,
    studio,
    tmp_path,
    *,
    launcher = ORIGINAL_LAUNCHER,
):
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
    monkeypatch.setattr(studio, "_fail_if_install_damaged", lambda *_args: None)
    # The runtime gate's process scan is Windows-only, so off Windows it never runs and nothing here noticed it was
    # unstubbed. On a real Windows host it shells out to powershell.exe through the same subprocess.run these tests
    # replace, then reads .stdout off a fake that only carries a returncode, and every test in this file dies before
    # reaching what it meant to assert.
    monkeypatch.setattr(
        studio._studio_runtime_gate,
        "ensure_managed_environment_is_idle",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path / "studio_home")
    for name in (
        "SKIP_STUDIO_BASE",
        "STUDIO_PACKAGE_NAME",
        "STUDIO_LOCAL_INSTALL",
        "STUDIO_LOCAL_REPO",
        "UNSLOTH_TAURI_UPDATE",
    ):
        monkeypatch.delenv(name, raising = False)
    return scripts, launcher_path


def _successful_version_run(calls = None):
    def run(argv, **kwargs):
        if calls is not None:
            calls.append((argv, kwargs))
        return types.SimpleNamespace(returncode = 0)

    return run


def _update(studio, *, verify = True):
    studio.update(local = False, package = "unsloth", verbose = False, verify = verify)


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows DLL locking only")
@pytest.mark.parametrize("entry", ["unsloth", "-m"])
@pytest.mark.parametrize("command", ["setup", "update"])
def test_windows_mutating_entry_does_not_load_pydantic_core(entry, command):
    code = f"""
import sys
sys.argv = [{entry!r}, "studio", {command!r}]
from unsloth_cli import app
assert "pydantic_core" not in sys.modules
assert "unsloth_cli.commands.train" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd = REPO_ROOT,
        capture_output = True,
        text = True,
        check = False,
    )
    assert result.returncode == 0, result.stderr


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


def test_a_recoverable_copy_exists_while_setup_runs(monkeypatch, studio, tmp_path):
    # The canonical path is freed so the installer can publish a replacement,
    # but never without a copy to put back if it publishes nothing.
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path)

    def setup(**_kwargs):
        assert not launcher.exists()
        assert (scripts / "unsloth.exe.update-backup").read_bytes() == ORIGINAL_LAUNCHER
        assert (scripts / "unsloth.exe.update-stale").read_bytes() == ORIGINAL_LAUNCHER

    monkeypatch.setattr(studio, "_run_setup_script", setup)
    monkeypatch.setattr(studio.subprocess, "run", _successful_version_run())

    _update(studio)


def test_installer_setup_frees_the_running_launcher_for_metadata_repair(
    monkeypatch, studio, tmp_path
):
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path)
    replacement = b"MZ-reinstalled-launcher"

    def setup(**_kwargs):
        assert not launcher.exists()
        assert (scripts / "unsloth.exe.update-backup").read_bytes() == ORIGINAL_LAUNCHER
        launcher.write_bytes(replacement)

    monkeypatch.setattr(studio, "_run_setup_script", setup)
    monkeypatch.setattr(studio.subprocess, "run", _successful_version_run())

    studio.setup(verbose = False)

    assert launcher.read_bytes() == replacement
    assert not (scripts / "unsloth.exe.update-backup").exists()


def test_setup_failure_restores_original_and_propagates(monkeypatch, studio, tmp_path):
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path)

    def setup(**_kwargs):
        launcher.write_bytes(b"MZ-new-but-incomplete")
        raise RuntimeError("setup failed")

    monkeypatch.setattr(studio, "_run_setup_script", setup)

    with pytest.raises(RuntimeError, match = "setup failed"):
        _update(studio)

    assert launcher.read_bytes() == ORIGINAL_LAUNCHER
    assert (scripts / "unsloth.exe.update-backup").read_bytes() == ORIGINAL_LAUNCHER


def test_setup_publishing_no_launcher_restores_it_and_succeeds(monkeypatch, studio, tmp_path):
    # The bug this transaction exists for: pip finds unsloth already current, writes no launcher, and the old updater
    # then deleted its own .deleteme and left the venv with none at all. Restoring is the right answer, not failing.
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path)
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: None)
    monkeypatch.setattr(studio.subprocess, "run", _successful_version_run())

    _update(studio)

    assert launcher.read_bytes() == ORIGINAL_LAUNCHER
    assert not (scripts / "unsloth.exe.update-stale").exists()


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
        return types.SimpleNamespace(returncode = 7)

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
    monkeypatch.setattr(
        studio, "_fail_if_install_damaged", lambda *_args: integrity_calls.append(True)
    )

    _update(studio, verify = False)

    assert calls[0][0] == [str(launcher), "--version"]
    assert integrity_calls == []


def test_legacy_backup_recovers_only_when_launcher_is_missing(monkeypatch, studio, tmp_path):
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path, launcher = None)
    legacy = scripts / "unsloth.exe.deleteme"
    legacy.write_bytes(ORIGINAL_LAUNCHER)

    def setup(**_kwargs):
        # Recovered from the legacy file, then moved aside for the installer.
        assert (scripts / "unsloth.exe.update-stale").read_bytes() == ORIGINAL_LAUNCHER

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
    before = launcher.read_bytes()
    first = studio._WindowsLauncherUpdateTransaction()
    first.__enter__()
    stale_before = (scripts / "unsloth.exe.update-stale").read_bytes()
    backup_before = (scripts / "unsloth.exe.update-backup").read_bytes()
    setup_calls = []
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: setup_calls.append(True))
    try:
        with pytest.raises(studio.typer.Exit) as exc:
            _update(studio)
        assert exc.value.exit_code == 1
        assert setup_calls == []
        assert stale_before == before
        assert (scripts / "unsloth.exe.update-stale").read_bytes() == stale_before
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
    monkeypatch.setattr(studio, "_fail_if_install_damaged", lambda *_args: calls.append("verify"))
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
        monkeypatch.delenv(name, raising = False)

    _update(studio)

    assert calls == ["setup", "verify", "refresh"]
    assert list(tmp_path.rglob("unsloth.exe*")) == []


def test_an_in_process_update_does_not_stage(monkeypatch, studio, tmp_path):
    """`stage` defaults to typer's OptionInfo, and that sentinel is truthy.

    Only the CLI resolves it to a bool, so a plain `if stage:` sends every
    in-process call down the staging path and skips the update entirely.
    """
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")
    monkeypatch.setattr(studio.sys, "executable", str(tmp_path / "bin" / "python"))
    monkeypatch.setattr(studio, "_ensure_studio_env_exported", lambda: None)
    staged = []
    monkeypatch.setattr(studio, "_stage_update", lambda **kwargs: staged.append(kwargs))
    calls = []
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: calls.append("setup"))
    monkeypatch.setattr(
        studio, "_fail_if_install_damaged", lambda *_a, **_k: calls.append("verify")
    )
    monkeypatch.setattr(
        studio, "_refresh_desktop_shortcuts", lambda **_kwargs: calls.append("refresh")
    )
    for name in (
        "SKIP_STUDIO_BASE",
        "STUDIO_PACKAGE_NAME",
        "STUDIO_LOCAL_INSTALL",
        "STUDIO_LOCAL_REPO",
        "UNSLOTH_TAURI_UPDATE",
        "UNSLOTH_STUDIO_STAGE_ROOT",
    ):
        monkeypatch.delenv(name, raising = False)

    _update(studio)

    assert staged == []
    assert calls == ["setup", "verify", "refresh"]


def _shim(studio, payload = ORIGINAL_LAUNCHER):
    """The hardlinked PATH shim install.ps1 creates beside the managed venv."""
    path = studio.STUDIO_HOME / "bin" / "unsloth.exe"
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(payload)
    return path


def test_a_missing_launcher_is_recovered_from_the_path_shim(monkeypatch, studio, tmp_path):
    # The old updater renamed the launcher away and then unlinked the .deleteme, so an affected install has neither.
    # install.ps1 hardlinks the shim to the same file, so it survives that unlink and can repair the launcher.
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path, launcher = None)
    _shim(studio)
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: None)
    monkeypatch.setattr(studio.subprocess, "run", _successful_version_run())

    _update(studio)

    assert launcher.read_bytes() == ORIGINAL_LAUNCHER


def test_an_invalid_launcher_is_recovered_from_the_backup(monkeypatch, studio, tmp_path):
    # Recovery gated on existence rather than validity left a zero-byte launcher in place while a usable backup sat
    # beside it.
    # The old updater restored on exactly this shape (st_size == 0), so gating on exists() regressed it.
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path, launcher = b"")
    (scripts / "unsloth.exe.update-backup").write_bytes(ORIGINAL_LAUNCHER)
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: None)
    monkeypatch.setattr(studio.subprocess, "run", _successful_version_run())

    _update(studio)

    assert launcher.read_bytes() == ORIGINAL_LAUNCHER


def test_no_launcher_and_no_recovery_source_still_runs_setup(monkeypatch, studio, tmp_path):
    # Refusing here would strand exactly the users the transaction exists for: the previous updater could leave no
    # launcher and no .deleteme, and before this the update simply carried on and let setup reinstall it.
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path, launcher = None)
    ran = []

    def setup(**_kwargs):
        ran.append(True)
        launcher.write_bytes(ORIGINAL_LAUNCHER)

    monkeypatch.setattr(studio, "_run_setup_script", setup)
    monkeypatch.setattr(studio.subprocess, "run", _successful_version_run())

    _update(studio)

    assert ran == [True]
    assert launcher.read_bytes() == ORIGINAL_LAUNCHER


def test_a_backup_failure_does_not_abort_the_update(monkeypatch, studio, tmp_path):
    # A backup is a safety net, not a precondition.
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path)
    ran = []
    original = studio._WindowsLauncherUpdateTransaction._atomic_copy

    def refuse_backup(source, destination):
        if destination.name.endswith(".update-backup"):
            raise OSError("access is denied")
        return original(source, destination)

    # _atomic_copy is a staticmethod, so the stand-in must not bind self either.
    monkeypatch.setattr(
        studio._WindowsLauncherUpdateTransaction, "_atomic_copy", staticmethod(refuse_backup)
    )
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: ran.append(True))
    monkeypatch.setattr(studio.subprocess, "run", _successful_version_run())

    _update(studio)

    assert ran == [True]
    assert launcher.read_bytes() == ORIGINAL_LAUNCHER


def test_an_existing_backup_survives_an_unvalidated_launcher(monkeypatch, studio, tmp_path):
    # A backup outlives __enter__ only when a previous run died before validating, so it holds the last launcher known
    # to run while the canonical file has passed nothing but the two-byte header check. Overwriting it here destroyed
    # the only recovery copy.
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path, launcher = b"MZ-broken")
    backup = scripts / "unsloth.exe.update-backup"
    backup.write_bytes(ORIGINAL_LAUNCHER)
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: None)

    def failing_version(_argv, **_kwargs):
        return types.SimpleNamespace(returncode = 7)

    monkeypatch.setattr(studio.subprocess, "run", failing_version)

    with pytest.raises(studio.typer.Exit):
        _update(studio)

    assert launcher.read_bytes() == ORIGINAL_LAUNCHER


def test_the_launcher_is_resolved_from_the_managed_studio_venv(monkeypatch, studio, tmp_path):
    # A pip-installed or checkout CLI drives an update of the separate managed environment, so sys.executable belongs to
    # the caller while setup.ps1 installs into STUDIO_HOME/unsloth_studio.
    scripts, caller_launcher = _configure_windows(monkeypatch, studio, tmp_path)
    managed = tmp_path / "studio_home" / "unsloth_studio"
    (managed / "Scripts").mkdir(parents = True)
    (managed / "pyvenv.cfg").write_text("home = /usr\n")
    managed_launcher = managed / "Scripts" / "unsloth.exe"
    managed_launcher.write_bytes(b"MZ-managed-launcher")

    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: None)
    calls = []
    monkeypatch.setattr(studio.subprocess, "run", _successful_version_run(calls))

    _update(studio)

    assert calls[0][0] == [str(managed_launcher), "--version"]
    assert managed_launcher.read_bytes() == b"MZ-managed-launcher"


def test_a_replacement_published_by_setup_is_kept(monkeypatch, studio, tmp_path):
    # The point of freeing the canonical path. uv only self-replaces its own executable, so it deletes a third-party
    # console script outright and hard-errors when the file is in use; the pip fallback then no-ops on the
    # already-satisfied bare unsloth and the upgrade is silently skipped.
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path)
    new_launcher = b"MZ-upgraded-launcher"

    monkeypatch.setattr(
        studio, "_run_setup_script", lambda **_kwargs: launcher.write_bytes(new_launcher)
    )
    monkeypatch.setattr(studio.subprocess, "run", _successful_version_run())

    _update(studio)

    assert launcher.read_bytes() == new_launcher
    assert not (scripts / "unsloth.exe.update-stale").exists()
    assert not (scripts / "unsloth.exe.update-backup").exists()


def test_an_invalid_replacement_is_restored_but_still_fails(monkeypatch, studio, tmp_path):
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path)
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: launcher.write_bytes(b""))
    monkeypatch.setattr(studio.subprocess, "run", _successful_version_run())

    with pytest.raises(studio.typer.Exit) as exc:
        _update(studio, verify = False)

    assert exc.value.exit_code == 1
    assert launcher.read_bytes() == ORIGINAL_LAUNCHER


def test_a_backup_that_cannot_run_falls_back_to_the_moved_aside_copy(monkeypatch, studio, tmp_path):
    # Backups are taken after only the two-byte header check, so an interrupted
    # run can leave a PE-shaped but non-runnable one. Preferring it must not
    # strand the working launcher that this run moved aside.
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path)
    bad_backup = b"MZ-unrunnable"
    (scripts / "unsloth.exe.update-backup").write_bytes(bad_backup)
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: None)

    def run(argv, **_kwargs):
        current = Path(argv[0]).read_bytes()
        return types.SimpleNamespace(returncode = 7 if current == bad_backup else 0)

    monkeypatch.setattr(studio.subprocess, "run", run)

    _update(studio)

    assert launcher.read_bytes() == ORIGINAL_LAUNCHER


def test_a_setup_exception_restores_a_runnable_launcher(monkeypatch, studio, tmp_path):
    # Backups are taken after only the two-byte header check, so an interrupted run can leave a PE-shaped but
    # non-runnable backup. __exit__ took the first PE-shaped candidate, so that backup was installed over the
    # working launcher this run had moved aside, and it could also undo a restore validate_launcher just made.
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path)
    bad_backup = b"MZ-unrunnable"
    (scripts / "unsloth.exe.update-backup").write_bytes(bad_backup)

    def setup(**_kwargs):
        raise RuntimeError("setup failed")

    monkeypatch.setattr(studio, "_run_setup_script", setup)

    def run(argv, **_kwargs):
        current = Path(argv[0]).read_bytes()
        return types.SimpleNamespace(returncode = 7 if current == bad_backup else 0)

    monkeypatch.setattr(studio.subprocess, "run", run)

    with pytest.raises(RuntimeError, match = "setup failed"):
        _update(studio)

    assert launcher.read_bytes() == ORIGINAL_LAUNCHER


def test_a_non_runnable_backup_falls_through_to_the_legacy_copy(monkeypatch, studio, tmp_path):
    # An interrupted run can leave a PE-shaped but non-runnable backup while the legacy .deleteme or the PATH shim is
    # still good. Accepting the backup on its MZ header alone and stopping there left the update failing forever with
    # the broken bytes canonical.
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path, launcher = None)
    bad_backup = b"MZ-unrunnable"
    (scripts / "unsloth.exe.update-backup").write_bytes(bad_backup)
    (scripts / "unsloth.exe.deleteme").write_bytes(ORIGINAL_LAUNCHER)
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: None)

    def run(argv, **_kwargs):
        current = Path(argv[0]).read_bytes()
        return types.SimpleNamespace(returncode = 7 if current == bad_backup else 0)

    monkeypatch.setattr(studio.subprocess, "run", run)

    _update(studio)

    assert launcher.read_bytes() == ORIGINAL_LAUNCHER


def test_the_update_lock_lives_outside_the_replaceable_venv(monkeypatch, studio, tmp_path):
    # setup.ps1 removes the whole $VenvDir to rebuild a stale torch, and Windows
    # refuses a recursive delete while a handle inside it is open. A lock under
    # Scripts/ therefore broke exactly the repair path it was meant to guard.
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path)
    seen = {}

    def setup(**_kwargs):
        seen["venv_locks"] = list(scripts.glob("*.update-lock"))
        seen["home_locks"] = list((studio.STUDIO_HOME).glob("*.update-lock"))

    monkeypatch.setattr(studio, "_run_setup_script", setup)
    monkeypatch.setattr(studio.subprocess, "run", _successful_version_run())

    _update(studio)

    assert seen["venv_locks"] == []
    assert len(seen["home_locks"]) == 1


def test_a_failed_move_aside_warns_that_unsloth_may_not_upgrade(
    monkeypatch, studio, tmp_path, capsys
):
    # Aborting here would make an antivirus hold enough to render the environment unupdatable, which main did not do
    # either. But the cost has to be visible: uv cannot replace a launcher it could not move, and the pip fallback
    # drops --upgrade-package, so unsloth stays at its old version.
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path)
    ran = []
    seen = {}

    real_replace = studio.os.replace

    def refuse_move(source, destination):
        # Only the move aside: patching os.replace wholesale would also break _atomic_copy's backup, and the test would
        # then be passing on a compound failure rather than the one it names.
        if str(destination).endswith(".update-stale"):
            raise OSError("access is denied")
        return real_replace(source, destination)

    monkeypatch.setattr(studio.os, "replace", refuse_move)

    def setup(**_kwargs):
        ran.append(True)
        # Sampled here: a successful update unlinks the backup on the way out.
        seen["backup"] = (scripts / "unsloth.exe.update-backup").read_bytes()

    monkeypatch.setattr(studio, "_run_setup_script", setup)
    monkeypatch.setattr(studio.subprocess, "run", _successful_version_run())

    _update(studio)

    assert ran == [True]
    err = capsys.readouterr().err
    assert "could not move the Unsloth launcher aside" in err
    assert "may not be upgraded" in err
    assert "could not back up" not in err
    assert seen["backup"] == ORIGINAL_LAUNCHER


# Issue #8490:


# ── Application Control (issue #8490) ─────────────────────────────────
#
# Windows can deny the generated, unsigned unsloth.exe while the signed
# python.exe beside it still runs. The launcher --version probe is then
# impossible, and reading that as "the update broke" rolled a perfectly good
# install back on every single update.
def _blocked_exe_run(interpreter_result, calls = None):
    """subprocess.run where only the launcher is denied by policy."""

    def run(argv, **kwargs):
        if calls is not None:
            calls.append((argv, kwargs))
        if str(argv[0]).endswith("unsloth.exe"):
            error = OSError(13, "An Application Control policy has blocked this file")
            error.winerror = 1260
            raise error
        return interpreter_result(argv, **kwargs)

    return run


def test_a_policy_blocked_launcher_falls_back_to_the_interpreter(monkeypatch, studio, tmp_path):
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path)
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: None)
    calls = []
    monkeypatch.setattr(
        studio.subprocess,
        "run",
        _blocked_exe_run(lambda argv, **_kwargs: types.SimpleNamespace(returncode = 0), calls),
    )

    _update(studio)

    assert launcher.read_bytes() == ORIGINAL_LAUNCHER
    assert not (scripts / "unsloth.exe.update-backup").exists()
    assert not (scripts / "unsloth.exe.update-stale").exists()

    assert calls[0][0] == [str(launcher), "--version"]
    interpreter_call = calls[1][0]
    # Spelled out rather than imported, so an edit to the constant fails here.
    # -I here and nowhere else in this module: this probe predicts the desktop updater's launch, and
    # build_update_command runs that under Isolation::Isolated with PYTHONHOME/PYTHONPATH cleared.
    assert interpreter_call == [
        str(scripts / "python.exe"),
        "-X",
        "utf8",
        "-I",
        "-c",
        "import sys, os; sys.path[:1] = [x for x in sys.path[:1] if getattr(sys.flags, 'safe_path', False) or x not in ('', os.getcwd())]; "
        "sys.argv[0] = 'unsloth'; from unsloth_cli import app; sys.exit(app())",
        "--version",
    ]
    # The launcher probe gets 10s, a process start.
    assert calls[0][1]["timeout"] == 10
    assert calls[1][1]["timeout"] == studio._MANAGED_CLI_IMPORT_PROBE_TIMEOUT
    assert calls[1][1]["timeout"] > calls[0][1]["timeout"]


def test_a_policy_block_with_a_broken_package_still_fails(monkeypatch, studio, tmp_path):
    """The fallback must not become a blanket "assume it worked"."""
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path)
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: None)
    monkeypatch.setattr(
        studio.subprocess,
        "run",
        _blocked_exe_run(lambda argv, **_kwargs: types.SimpleNamespace(returncode = 3)),
    )

    with pytest.raises(studio.typer.Exit):
        _update(studio)

    assert launcher.read_bytes() == ORIGINAL_LAUNCHER
    assert (scripts / "unsloth.exe.update-backup").exists()


def test_a_policy_block_with_no_interpreter_reports_the_block(
    monkeypatch, studio, tmp_path, capsys
):
    """Nothing left to ask: say what Windows said rather than inventing a cause."""
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path)
    (scripts / "python.exe").unlink()
    monkeypatch.setattr(
        studio, "_run_setup_script", lambda **_kwargs: launcher.write_bytes(b"MZ-new")
    )
    monkeypatch.setattr(
        studio.subprocess,
        "run",
        _blocked_exe_run(lambda argv, **_kwargs: types.SimpleNamespace(returncode = 0)),
    )

    with pytest.raises(studio.typer.Exit):
        _update(studio)

    assert "Application Control policy" in capsys.readouterr().err


def test_an_ordinary_launcher_oserror_is_still_a_failure(monkeypatch, studio, tmp_path):
    """Parity guard: only 1260 takes the new path, everything else is unchanged."""
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path)
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: None)
    interpreter_calls = []

    def run(argv, **_kwargs):
        if str(argv[0]).endswith("unsloth.exe"):
            error = OSError(13, "Access is denied")
            error.winerror = 5
            raise error
        interpreter_calls.append(argv)
        return types.SimpleNamespace(returncode = 0)

    monkeypatch.setattr(studio.subprocess, "run", run)

    with pytest.raises(studio.typer.Exit):
        _update(studio)

    assert interpreter_calls == [], "a non-policy error must not consult the interpreter"
    assert launcher.read_bytes() == ORIGINAL_LAUNCHER
    assert (scripts / "unsloth.exe.update-backup").exists()


def test_the_policy_block_helper_only_matches_1260(studio):
    blocked = OSError(13, "blocked")
    blocked.winerror = 1260
    assert studio._is_application_control_block(blocked)

    denied = OSError(13, "denied")
    denied.winerror = 5
    assert not studio._is_application_control_block(denied)

    # POSIX OSError has no winerror at all.
    assert not studio._is_application_control_block(OSError(13, "denied"))


def test_a_quarantined_away_launcher_falls_back_to_the_interpreter(monkeypatch, studio, tmp_path):
    """Quarantine removes the unsigned stub rather than denying it.

    There is then nothing to run and nothing to put back, and reading that as a
    broken update rolls a good one back exactly as the denial case did. The
    launcher stays gone, which is fine: nothing executes it any more.
    """
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path, launcher = None)
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: None)
    calls = []
    monkeypatch.setattr(studio.subprocess, "run", _successful_version_run(calls))

    _update(studio)

    assert not launcher.exists()
    # A successful update cleans its recovery copies up; a rollback would keep them.
    assert not (scripts / "unsloth.exe.update-backup").exists()
    assert [call[0][0] for call in calls] == [str(scripts / "python.exe")]


def test_only_the_updater_probe_isolates_the_interpreter(studio, tmp_path):
    """Isolation is opt-in, and exactly one caller opts in.

    Every other managed invocation has to stay byte-for-byte what the console
    script did, and -I implies -E and -s: PYTHONPATH, PYTHONWARNINGS,
    PYTHONHASHSEED and user site-packages all stop being honoured. That is an
    observable difference on a machine with no policy at all, so the default
    inherits and only the probe standing in for an already-isolated launch asks
    for it. Mirrors only_the_isolated_flavour_carries_the_isolation_flag in
    studio/src-tauri/src/process.rs.
    """
    python = tmp_path / "python.exe"

    inherited = studio._managed_cli_argv(python, "--version")
    assert inherited[:3] == [str(python), "-X", "utf8"]
    assert "-I" not in inherited

    isolated = studio._managed_cli_argv(python, "--version", isolated = True)
    # -X utf8 first: -I implies -E, so PYTHONUTF8 would be discarded, while a command-line -X survives it.
    assert isolated[:4] == [str(python), "-X", "utf8", "-I"]

    # Isolation is the only difference.
    assert [arg for arg in isolated if arg != "-I"] == inherited


def test_a_quarantined_away_launcher_with_a_broken_package_still_fails(
    monkeypatch, studio, tmp_path
):
    """Parity guard: absence excuses the launcher, never the update."""
    _configure_windows(monkeypatch, studio, tmp_path, launcher = None)
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: None)
    monkeypatch.setattr(
        studio.subprocess,
        "run",
        lambda argv, **_kwargs: types.SimpleNamespace(returncode = 3),
    )

    with pytest.raises(studio.typer.Exit):
        _update(studio)


def test_a_restorable_launcher_is_restored_before_the_interpreter_is_asked(
    monkeypatch, studio, tmp_path
):
    """Absence is only excused once recovery has failed.

    A healthy CLI must not let a launcher that could have been put back stay
    missing, or an ordinary no-op update on an unpoliced machine would quietly
    strip the console script.
    """
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path)
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: None)
    monkeypatch.setattr(studio.subprocess, "run", _successful_version_run())

    _update(studio)

    assert launcher.read_bytes() == ORIGINAL_LAUNCHER


def test_the_package_answers_for_a_quarantined_console_script(monkeypatch, studio, tmp_path):
    """What `studio run` checks instead of the deleted stub, and only on Windows.

    The layout is the fallback for one specific no-verdict case, a probe that
    timed out, so that is what is simulated here. A probe that could not START
    the interpreter is a different answer and is covered separately: the re-exec
    runs that same interpreter, so the layout cannot excuse it.
    """
    scripts = tmp_path / "Scripts"
    site_packages = tmp_path / "Lib" / "site-packages"
    scripts.mkdir(parents = True)
    site_packages.mkdir(parents = True)
    python = scripts / "python.exe"
    python.write_bytes(b"python")

    def timed_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd = "probe", timeout = 60)

    monkeypatch.setattr(studio.subprocess, "run", timed_out)
    monkeypatch.setattr(studio.platform, "system", lambda: "Windows")
    assert not studio._managed_cli_package_present(python)

    (site_packages / "unsloth_cli").mkdir()
    assert studio._managed_cli_package_present(python)

    # An editable install leaves a .pth and no unsloth_cli/ here.
    (site_packages / "unsloth_cli").rmdir()
    (site_packages / "unsloth-2026.8.1.dist-info").mkdir()
    assert studio._managed_cli_package_present(python)

    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")
    assert not studio._managed_cli_package_present(python)


@pytest.fixture(scope = "module")
def real_venv(tmp_path_factory):
    """A real, empty interpreter to ask, rather than a file named python.exe.

    Built without pip so site-packages starts genuinely empty; the layout is
    POSIX here, which is the point -- the check must come from the interpreter,
    not from guessing at Lib\\site-packages.
    """
    root = tmp_path_factory.mktemp("managed_cli_probe") / "venv"
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", str(root)],
            check = True,
            capture_output = True,
            timeout = 300,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        pytest.skip(f"could not build a probe venv: {exc}")
    python = root / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    assert python.is_file()
    site_packages = next(iter(root.glob("lib/python*/site-packages")), None) or (
        root / "Lib" / "site-packages"
    )
    return python, site_packages


def test_orphaned_install_metadata_is_not_a_runnable_cli(monkeypatch, studio, real_venv):
    """Metadata is not an importable package, and this gate cannot accept it.

    An interrupted install, or an editable install whose checkout has since
    moved, leaves an ``unsloth-*.dist-info`` behind with nothing to import. This
    check stands in front of the headless-public strip of .bootstrap_password,
    so answering yes here lands exactly the lockout the gate's placement exists
    to prevent: a public Unsloth with no login page and no recovery credential.
    """
    python, site_packages = real_venv
    windows_layout = python.parent.parent / "Lib" / "site-packages"
    windows_layout.mkdir(parents = True, exist_ok = True)
    (windows_layout / "unsloth-2026.8.1.dist-info").mkdir(exist_ok = True)

    monkeypatch.setattr(studio.platform, "system", lambda: "Windows")
    assert not (site_packages / "unsloth_cli").exists(), "the probe venv must start empty"
    assert not studio._managed_cli_package_present(python)


def test_an_importable_package_still_answers_for_the_quarantined_stub(
    monkeypatch, studio, real_venv
):
    """The quarantine case this fallback exists for keeps working.

    Same venv as above, now with something the interpreter can actually resolve.
    """
    python, site_packages = real_venv
    package = site_packages / "unsloth_cli"
    package.mkdir(parents = True, exist_ok = True)
    (package / "__init__.py").write_text("app = None\n", encoding = "utf-8")

    monkeypatch.setattr(studio.platform, "system", lambda: "Windows")
    assert studio._managed_cli_package_present(python)

    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")
    assert not studio._managed_cli_package_present(python)


@pytest.fixture
def bare_probe_venv(real_venv):
    """The module venv with any unsloth_cli left by a neighbouring test removed.

    These cases each install their own shape of broken package, so they cannot
    inherit one, and they must not leave one behind either.
    """
    python, site_packages = real_venv
    package = site_packages / "unsloth_cli"
    shutil.rmtree(package, ignore_errors = True)
    yield python, site_packages
    shutil.rmtree(package, ignore_errors = True)


@pytest.mark.parametrize(
    "shape, files",
    [
        # An emptied directory. find_spec calls this a namespace package and returns a spec for it, so a spec lookup
        # answers yes to a venv the trampoline's `from unsloth_cli import app` cannot start. This is the shape
        # antivirus leaves when it takes the module files out from under a package it decided it disliked.
        ("an emptied package directory", {}),
        # An interrupted install: the package landed, its dependencies did not.
        (
            "a package whose imports are missing",
            {"__init__.py": "import unsloth_cli_missing_dep\n"},
        ),
        # A partially written __init__ that imports but has no app to hand back.
        ("a package with no app attribute", {"__init__.py": "VERSION = '1'\n"}),
        # An __init__ that raises on import, which no spec lookup ever executes.
        (
            "a package whose import raises",
            {"__init__.py": "raise RuntimeError('half installed')\n"},
        ),
    ],
)
def test_a_package_the_trampoline_cannot_import_is_not_a_runnable_cli(
    monkeypatch, studio, bare_probe_venv, shape, files
):
    """The gate has to fail on everything the launch would fail on.

    It stands in front of the headless-public strip of .bootstrap_password, so a
    yes here that the trampoline then contradicts is a public Unsloth with no
    login page and no plaintext recovery credential. Locating the package is not
    the question; importing it and getting `app` back is, which is why the probe
    runs that exact import rather than a cheaper find_spec.
    """
    python, site_packages = bare_probe_venv
    package = site_packages / "unsloth_cli"
    package.mkdir(parents = True)
    for name, body in files.items():
        (package / name).write_text(body, encoding = "utf-8")

    monkeypatch.setattr(studio.platform, "system", lambda: "Windows")
    assert not studio._managed_cli_package_present(
        python
    ), f"{shape} must not pass the gate: the trampoline cannot start it"

    # Anti-vacuity: the same venv with a package that does import passes, so the assertion above is about the shape and
    # not about the fixture being broken.
    (package / "__init__.py").write_text("app = None\n", encoding = "utf-8")
    assert studio._managed_cli_package_present(python)


def test_a_probe_that_cannot_start_the_interpreter_fails_closed(monkeypatch, studio, tmp_path):
    """No verdict is not the same as no problem, and the two causes differ.

    The re-exec this gate stands in front of runs the same interpreter, so an
    interpreter that will not start means the re-exec will not either, and the
    on-disk layout cannot say otherwise. The caller strips .bootstrap_password
    before that re-exec on a headless public launch, so passing here would leave
    a public Unsloth with no login page and no plaintext recovery credential.
    """
    scripts = tmp_path / "Scripts"
    site_packages = tmp_path / "Lib" / "site-packages"
    scripts.mkdir(parents = True)
    (site_packages / "unsloth_cli").mkdir(parents = True)
    python = scripts / "python.exe"
    python.write_bytes(b"python")
    monkeypatch.setattr(studio.platform, "system", lambda: "Windows")

    assert studio._managed_cli_site_packages_layout(python)

    def blocked(*_args, **_kwargs):
        raise OSError(1260, "An Application Control policy has blocked this file")

    monkeypatch.setattr(studio.subprocess, "run", blocked)
    assert not studio._managed_cli_package_present(python)

    # A timeout is the other kind of no verdict, and it keeps the fallback: slow is not broken, a cold venv under an
    # antivirus scan is exactly this, and the re-exec has no timeout of its own to trip over.
    def slow(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd = "probe", timeout = 60)

    monkeypatch.setattr(studio.subprocess, "run", slow)
    assert studio._managed_cli_package_present(python)


def test_the_interpreter_fallback_waits_as_long_as_the_import_probe(studio):
    """--version through the trampoline is an import, not a process start.

    The launcher's 10 seconds is sized for spawning a built executable. Here the
    same call has to import the whole CLI package, which is the work the import
    probe's ceiling is deliberately generous for, and under the antivirus scan
    that produced the quarantine this path exists to survive, the short ceiling
    would call a healthy update broken and roll it back once per candidate.
    """
    transaction = studio._WindowsLauncherUpdateTransaction
    source = inspect.getsource(transaction._interpreter_health_error)
    assert "_MANAGED_CLI_IMPORT_PROBE_TIMEOUT" in source
    assert "_VERSION_TIMEOUT_SECONDS" not in source
    assert studio._MANAGED_CLI_IMPORT_PROBE_TIMEOUT > transaction._VERSION_TIMEOUT_SECONDS


def test_a_custom_root_survives_the_launcher_being_quarantined(monkeypatch, studio, tmp_path):
    """The sentinel decides which installation every studio command manages.

    Only install.sh writes share/studio.conf, so on a custom-root Windows
    install the generated unsloth.exe was the only sentinel there was. Quarantine
    deletes it, root inference then falls back to ~/.unsloth/studio, and the CLI
    reads and writes the wrong tree while reporting success. The .cmd shim is
    written by the same installer for the same directory and answers the same
    question.
    """
    root = tmp_path / "custom-root"
    (root / "bin").mkdir(parents = True)
    monkeypatch.setattr(studio.platform, "system", lambda: "Windows")

    assert not studio._looks_like_installer_managed_studio_home(root)

    shim = root / "bin" / "unsloth.cmd"
    shim.write_bytes(
        b"@echo off\r\nrem unsloth-studio-managed-launcher\r\n"
        b'"%~dp0..\\unsloth_studio\\Scripts\\python.exe" -X utf8 -c "from unsloth_cli import app" %*\r\n'
    )
    assert studio._looks_like_installer_managed_studio_home(root)

    # The launcher still answers on its own, so an install that never lost it is unaffected either way.
    shim.unlink()
    (root / "bin" / "unsloth.exe").write_bytes(b"MZ")
    assert studio._looks_like_installer_managed_studio_home(root)


@pytest.mark.parametrize(
    "label, body",
    [
        # This decides which tree the CLI manages and the directory is on PATH, so any file of that name would otherwise
        # be enough to redirect a root.
        ("a hand-rolled wrapper", b'@echo off\r\npython -c "from unsloth_cli import app" %*\r\n'),
        ("the marker without the call", b"@echo off\r\nrem unsloth-studio-managed-launcher\r\n"),
        ("an unrelated batch file", b"@echo off\r\necho hello\r\n"),
        ("empty", b""),
    ],
)
def test_only_the_installers_own_cmd_shim_marks_a_root(monkeypatch, studio, tmp_path, label, body):
    root = tmp_path / "root"
    (root / "bin").mkdir(parents = True)
    (root / "bin" / "unsloth.cmd").write_bytes(body)
    monkeypatch.setattr(studio.platform, "system", lambda: "Windows")

    assert not studio._looks_like_installer_managed_studio_home(
        root
    ), f"{label} must not stand in for the installer's own shim"


def test_an_oversized_cmd_shim_is_not_read_into_memory(monkeypatch, studio, tmp_path):
    """Same 8 KB ceiling Test-UnslothCmdShimFile and the uninstaller apply."""
    root = tmp_path / "root"
    (root / "bin").mkdir(parents = True)
    shim = root / "bin" / "unsloth.cmd"
    shim.write_bytes(
        b"rem unsloth-studio-managed-launcher\r\nfrom unsloth_cli import app\r\n" + b"x" * 9000
    )
    monkeypatch.setattr(studio.platform, "system", lambda: "Windows")

    assert not studio._looks_like_installer_managed_studio_home(root)


def test_posix_root_inference_is_unchanged(monkeypatch, studio, tmp_path):
    """A .cmd means nothing off Windows, and the console script still answers."""
    root = tmp_path / "root"
    (root / "bin").mkdir(parents = True)
    (root / "bin" / "unsloth.cmd").write_bytes(
        b"rem unsloth-studio-managed-launcher\r\nfrom unsloth_cli import app\r\n"
    )
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")

    assert not studio._looks_like_installer_managed_studio_home(root)
    (root / "bin" / "unsloth").write_text("#!/bin/sh\n", encoding = "utf-8")
    assert studio._looks_like_installer_managed_studio_home(root)


def test_the_import_probe_performs_the_trampolines_own_import(studio):
    """A spec lookup here would answer a different question than the launch asks.

    Pinned as a source contract because the two failures it prevents are silent:
    an empty directory and a raising __init__ both resolve as specs, and both
    give the gate a yes the launch immediately contradicts.
    """
    assert "from unsloth_cli import app" in studio._MANAGED_CLI_IMPORT_PROBE
    assert "find_spec" not in studio._MANAGED_CLI_IMPORT_PROBE


def test_the_import_probe_scrubs_the_cwd_exactly_as_the_trampoline_does(studio):
    """A drift here would let a checkout in the caller's cwd answer for the venv.

    `-c` puts the cwd on sys.path[0]; the trampoline drops it, so a probe that
    did not would report a CLI the launch cannot then import.
    """
    scrub = (
        "import sys, os; sys.path[:1] = [x for x in sys.path[:1] "
        "if getattr(sys.flags, 'safe_path', False) or x not in ('', os.getcwd())]; "
    )
    assert studio._WINDOWS_CLI_ENTRYPOINT.startswith(scrub)
    assert studio._MANAGED_CLI_IMPORT_PROBE.startswith(scrub)


def test_a_candidate_that_vanishes_mid_copy_does_not_stop_the_next_one(
    monkeypatch, studio, tmp_path
):
    """The header check and the copy open the file separately.

    Antivirus taking the first candidate in that gap says nothing about the rest,
    and giving up there turned a recoverable install into a failed update.
    """
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path, launcher = None)
    backup = scripts / "unsloth.exe.update-backup"
    legacy = scripts / "unsloth.exe.deleteme"
    backup.write_bytes(b"MZ-backup")
    legacy.write_bytes(ORIGINAL_LAUNCHER)

    real_copy = studio._WindowsLauncherUpdateTransaction._atomic_copy

    def flaky_copy(source, destination):
        if source == backup:
            raise OSError(5, "Access is denied")
        return real_copy(source, destination)

    monkeypatch.setattr(
        studio._WindowsLauncherUpdateTransaction, "_atomic_copy", staticmethod(flaky_copy)
    )
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: None)
    monkeypatch.setattr(studio.subprocess, "run", _successful_version_run())

    _update(studio)

    assert launcher.read_bytes() == ORIGINAL_LAUNCHER, "the second candidate was never tried"


def test_every_candidate_failing_is_still_an_error(monkeypatch, studio, tmp_path, capsys):
    """Parity guard: trying them all must not become swallowing them all."""
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path, launcher = None)
    (scripts / "unsloth.exe.update-backup").write_bytes(b"MZ-backup")

    def always_fails(source, destination):
        raise OSError(5, "Access is denied")

    monkeypatch.setattr(
        studio._WindowsLauncherUpdateTransaction, "_atomic_copy", staticmethod(always_fails)
    )
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: None)
    monkeypatch.setattr(studio.subprocess, "run", _successful_version_run())

    with pytest.raises(studio.typer.Exit):
        _update(studio)

    assert "could not recover" in capsys.readouterr().err


def test_an_unrecoverable_launcher_keeps_its_recovery_copies(monkeypatch, studio, tmp_path):
    """Judged healthy through the interpreter is not the same as repaired.

    When every restore attempt failed, the copies are the only material a later
    run has. Deleting them on the way out of a "successful" update would make the
    next one unrecoverable.
    """
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path, launcher = None)
    backup = scripts / "unsloth.exe.update-backup"
    backup.write_bytes(ORIGINAL_LAUNCHER)

    def always_fails(source, destination):
        raise OSError(5, "Access is denied")

    monkeypatch.setattr(
        studio._WindowsLauncherUpdateTransaction, "_atomic_copy", staticmethod(always_fails)
    )
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: None)
    monkeypatch.setattr(studio.subprocess, "run", _successful_version_run())

    with pytest.raises(studio.typer.Exit):
        _update(studio)

    assert backup.exists(), "the only recovery copy was deleted"


def test_a_restored_launcher_still_cleans_up(monkeypatch, studio, tmp_path):
    """Parity guard: keeping copies must not become never cleaning up."""
    scripts, launcher = _configure_windows(monkeypatch, studio, tmp_path)
    monkeypatch.setattr(studio, "_run_setup_script", lambda **_kwargs: None)
    monkeypatch.setattr(studio.subprocess, "run", _successful_version_run())

    _update(studio)

    assert launcher.read_bytes() == ORIGINAL_LAUNCHER
    assert not (scripts / "unsloth.exe.update-backup").exists()
    assert not (scripts / "unsloth.exe.update-stale").exists()
    assert not (scripts / "unsloth.exe.deleteme").exists()
