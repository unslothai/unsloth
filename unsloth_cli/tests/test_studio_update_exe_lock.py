# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The Windows self-lock on unsloth.exe (issue #7697).

Windows locks the directory entry an image was launched from rather than the file
behind it, so the rename here fails exactly when the update is running out of the
copy pip has to replace. These pin that the CLI keeps going, since an update with
no package change never touches the file, that a later setup failure carries a note
explaining the cause without replacing the real error, and that the cleanup after a
successful update never deletes the only copy.
"""

import errno

import pytest
from pathlib import Path

from unsloth_cli.commands import studio


def _as_windows(monkeypatch, scripts_dir: Path):
    monkeypatch.setattr(studio.platform, "system", lambda: "Windows")
    monkeypatch.setattr(studio.sys, "executable", str(scripts_dir / "python.exe"))


def _sharing_violation(*a, **k):
    raise OSError(errno.EACCES, "The process cannot access the file")


def _scripts(tmp_path: Path) -> Path:
    scripts = tmp_path / "unsloth_studio" / "Scripts"
    scripts.mkdir(parents = True)
    return scripts


# ── releasing the lock ───────────────────────────────────────────────


def test_a_locked_exe_is_reported_but_does_not_abort(monkeypatch, tmp_path, capsys):
    """An update with no package change never touches the file and still works from
    here, so this must not become fatal for everyone who runs the exe directly, and
    must not warn on a run that goes on to succeed."""
    scripts = _scripts(tmp_path)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    _as_windows(monkeypatch, scripts)
    monkeypatch.setattr(studio.os, "replace", _sharing_violation)

    err = studio._release_self_exe_lock_windows()

    assert isinstance(err, OSError), "the lock has to be reported to the caller"
    assert capsys.readouterr().out == "", "a successful update must stay quiet"


def test_a_rename_that_works_reports_nothing(monkeypatch, tmp_path, capsys):
    """The supported path, where the update runs through the launcher and this copy
    is free."""
    scripts = _scripts(tmp_path)
    exe = scripts / "unsloth.exe"
    exe.write_bytes(b"MZ")
    _as_windows(monkeypatch, scripts)

    assert studio._release_self_exe_lock_windows() is None
    assert not exe.exists()
    assert (scripts / "unsloth.exe.deleteme").is_file()
    assert capsys.readouterr().err == ""


def test_other_platforms_are_untouched(monkeypatch, tmp_path):
    scripts = tmp_path / "Scripts"
    scripts.mkdir(parents = True)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")
    monkeypatch.setattr(studio.os, "replace", _sharing_violation)

    assert studio._release_self_exe_lock_windows() is None


# ── refusing before pip destroys the install ─────────────────────────


def _shimmed(monkeypatch, tmp_path):
    scripts = _scripts(tmp_path)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    shim = tmp_path / "bin" / "unsloth.exe"
    shim.parent.mkdir(parents = True)
    shim.write_bytes(b"MZ")
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path)
    _as_windows(monkeypatch, scripts)
    return scripts, shim


def test_a_locked_copy_stops_before_pip_runs(monkeypatch, tmp_path, capsys):
    """Continuing is not a failed update. pip uninstalls before it installs, so it
    removes unsloth_cli and only then hits the locked stub, leaving an exe that
    starts and raises ModuleNotFoundError. Nothing has been removed at this point,
    so this is the last moment it is still avoidable."""
    _shimmed(monkeypatch, tmp_path)

    with pytest.raises(studio.typer.Exit) as excinfo:
        studio._refuse_update_that_would_break_the_install(OSError(errno.EACCES, "in use"))

    assert excinfo.value.exit_code == 1
    err = capsys.readouterr().err
    assert "Nothing has been removed" in err
    assert "unsloth.exe" in err, "the reason has to travel with the refusal"


def test_without_a_launcher_it_still_refuses(monkeypatch, tmp_path, capsys):
    """Going on with no launcher does not leave anyone able to update either: it
    destroys the install, so they reinstall anyway, only without being told so."""
    scripts = _scripts(tmp_path)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path)  # no bin/unsloth.exe
    _as_windows(monkeypatch, scripts)

    with pytest.raises(studio.typer.Exit) as excinfo:
        studio._refuse_update_that_would_break_the_install(OSError(errno.EACCES, "in use"))

    assert excinfo.value.exit_code == 1
    err = capsys.readouterr().err
    assert "Nothing has been removed" in err
    assert "reinstall to restore it" in err, "stopping without a way forward is stranding"


def test_an_unusable_launcher_is_still_a_reason_to_stop(monkeypatch, tmp_path, capsys):
    _scripts_dir, shim = _shimmed(monkeypatch, tmp_path)
    shim.write_bytes(b"")

    with pytest.raises(studio.typer.Exit):
        studio._refuse_update_that_would_break_the_install(OSError(errno.EACCES, "in use"))

    err = capsys.readouterr().err
    assert str(shim) not in err, "an unusable launcher was recommended anyway"
    assert "reinstall to restore it" in err


def test_the_refusal_can_be_overridden(monkeypatch, tmp_path):
    """An escape hatch, because this stops updates that would have succeeded: one
    with no package change never touches the file at all."""
    _shimmed(monkeypatch, tmp_path)
    monkeypatch.setenv(studio._ALLOW_LOCKED_ENV, "1")

    studio._refuse_update_that_would_break_the_install(OSError(errno.EACCES, "in use"))


def test_other_platforms_never_refuse(monkeypatch, tmp_path):
    _shimmed(monkeypatch, tmp_path)
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")

    studio._refuse_update_that_would_break_the_install(OSError(errno.EACCES, "in use"))


# ── the note on a later failure ──────────────────────────────────────


def test_the_note_points_at_the_launcher_by_full_path(monkeypatch, tmp_path, capsys):
    """Reaching this lock is evidence the venv Scripts dir may come first on PATH, so
    a bare `unsloth` could resolve straight back to the locked copy."""
    scripts = _scripts(tmp_path)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    shim = tmp_path / "bin" / "unsloth.exe"
    shim.parent.mkdir(parents = True)
    shim.write_bytes(b"MZ")
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path)
    _as_windows(monkeypatch, scripts)

    studio._note_self_exe_locked(OSError(errno.EACCES, "in use"))

    err = capsys.readouterr().err
    assert str(shim) in err, "the note must name the launcher explicitly"
    assert "install.ps1" not in err


def test_the_note_is_conditional_not_a_diagnosis(monkeypatch, tmp_path, capsys):
    """Setup output is streamed, not captured, so this cannot know whether pip ever
    reached unsloth.exe. It says what to check rather than asserting a cause."""
    scripts = _scripts(tmp_path)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path)
    _as_windows(monkeypatch, scripts)

    studio._note_self_exe_locked(OSError(errno.EACCES, "in use"))

    assert "If the failure above mentions" in capsys.readouterr().err


def test_a_local_update_is_retried_as_a_local_update(monkeypatch, tmp_path, capsys):
    """The retry runs in a new shell that inherits neither the flag nor the repo it
    resolved, so without both it quietly becomes a PyPI update and reports success
    with the checkout that prompted the replacement still uninstalled."""
    scripts = _scripts(tmp_path)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    shim = tmp_path / "bin" / "unsloth.exe"
    shim.parent.mkdir(parents = True)
    shim.write_bytes(b"MZ")
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path)
    _as_windows(monkeypatch, scripts)
    repo = tmp_path / "checkout"

    studio._note_self_exe_locked(OSError(errno.EACCES, "in use"), repo_root = repo)

    err = capsys.readouterr().err
    assert "studio update --local" in err
    assert f"$env:STUDIO_LOCAL_REPO = '{repo}'" in err


def test_a_pypi_update_is_not_retried_as_a_local_one(monkeypatch, tmp_path, capsys):
    scripts = _scripts(tmp_path)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    shim = tmp_path / "bin" / "unsloth.exe"
    shim.parent.mkdir(parents = True)
    shim.write_bytes(b"MZ")
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path)
    _as_windows(monkeypatch, scripts)

    studio._note_self_exe_locked(OSError(errno.EACCES, "in use"))

    err = capsys.readouterr().err
    assert "--local" not in err
    assert "STUDIO_LOCAL_REPO" not in err


def test_an_apostrophe_in_the_path_does_not_break_the_powershell_line(
    monkeypatch, tmp_path, capsys
):
    """A custom Studio root is user-chosen, and an unescaped apostrophe ends the
    single-quoted string early, so the prescribed recovery cannot be pasted."""
    home = tmp_path / "O'Brien" / "Studio"
    scripts = home / "unsloth_studio" / "Scripts"
    scripts.mkdir(parents = True)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    shim = home / "bin" / "unsloth.exe"
    shim.parent.mkdir(parents = True)
    shim.write_bytes(b"MZ")
    monkeypatch.setattr(studio, "STUDIO_HOME", home)
    _as_windows(monkeypatch, scripts)

    studio._note_self_exe_locked(OSError(errno.EACCES, "in use"))

    line = [x for x in capsys.readouterr().err.splitlines() if "studio update" in x][0]
    assert "O''Brien" in line, "the apostrophe was not doubled"
    # Every quote in the rendered line has to pair up, or PowerShell reads the
    # rest of the command as a string.
    assert line.count("'") % 2 == 0


def test_a_custom_package_survives_into_the_retry(monkeypatch, tmp_path, capsys):
    """update exports --package as STUDIO_PACKAGE_NAME, so dropping it resets the
    retry to `unsloth` and updates a different thing, then records that package in
    the manifest for later verification to follow."""
    scripts = _scripts(tmp_path)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    shim = tmp_path / "bin" / "unsloth.exe"
    shim.parent.mkdir(parents = True)
    shim.write_bytes(b"MZ")
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path)
    _as_windows(monkeypatch, scripts)

    studio._note_self_exe_locked(OSError(errno.EACCES, "in use"), package = "unsloth-test")

    assert "--package 'unsloth-test'" in capsys.readouterr().err


def test_the_default_package_is_not_spelled_out(monkeypatch, tmp_path, capsys):
    scripts = _scripts(tmp_path)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    shim = tmp_path / "bin" / "unsloth.exe"
    shim.parent.mkdir(parents = True)
    shim.write_bytes(b"MZ")
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path)
    _as_windows(monkeypatch, scripts)

    studio._note_self_exe_locked(OSError(errno.EACCES, "in use"), package = "unsloth")

    assert "--package" not in capsys.readouterr().err


def test_a_zero_byte_launcher_is_not_recommended(monkeypatch, tmp_path, capsys):
    """A damaged shim is a reason to have run the venv copy directly, so pointing
    back at it sends the user to the file that already does not start."""
    scripts = _scripts(tmp_path)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    shim = tmp_path / "bin" / "unsloth.exe"
    shim.parent.mkdir(parents = True)
    shim.write_bytes(b"")
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path)
    _as_windows(monkeypatch, scripts)

    studio._note_self_exe_locked(OSError(errno.EACCES, "in use"))

    err = capsys.readouterr().err
    assert str(shim) not in err, "an unusable launcher was recommended anyway"
    assert "install.ps1" in err, "the reinstall fallback was hidden"


def test_a_truncated_launcher_is_not_recommended(monkeypatch, tmp_path, capsys):
    """Nonempty and readable is not runnable. A half-written Copy-Item fallback
    leaves a file that passes both and that Windows still cannot execute, and
    accepting it aborts an intact update towards a launcher that does not start.
    b"PK" rather than b"" so only the image-header check can reject it."""
    scripts = _scripts(tmp_path)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    shim = tmp_path / "bin" / "unsloth.exe"
    shim.parent.mkdir(parents = True)
    shim.write_bytes(b"PK\x03\x04not an image")
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path)
    _as_windows(monkeypatch, scripts)

    assert studio._is_usable_launcher(shim) is False
    # It still refuses -- the predicate decides which message to print, not
    # whether to stop, or a stricter predicate would widen the destructive path
    # this exists to close.
    with pytest.raises(studio.typer.Exit):
        studio._refuse_update_that_would_break_the_install(OSError(errno.EACCES, "in use"))

    err = capsys.readouterr().err
    assert str(shim) not in err, "an unusable launcher was recommended anyway"
    assert "install.ps1" in err, "the reinstall fallback was hidden"


def test_a_real_image_is_still_accepted(monkeypatch, tmp_path):
    """The header check has to keep the working case working, or it would disable
    the launcher route altogether rather than narrowing it."""
    _scripts_dir, shim = _shimmed(monkeypatch, tmp_path)
    shim.write_bytes(b"MZ\x90\x00" + b"\x00" * 64)

    assert studio._is_usable_launcher(shim) is True


def test_the_header_check_is_windows_only(monkeypatch, tmp_path):
    """Unix has an execute bit and no PE images, so requiring MZ there would reject
    the symlinked shim every time."""
    _scripts_dir, shim = _shimmed(monkeypatch, tmp_path)
    shim.write_bytes(b'#!/bin/sh\nexec unsloth "$@"\n')
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")

    assert studio._is_usable_launcher(shim) is True


def test_the_note_says_another_process_can_hold_the_lock(monkeypatch, tmp_path, capsys):
    """The OSError says the entry is locked, not who by. If a second process
    launched from the same copy holds it, the retry changes nothing, because it is
    pip that has to replace the file."""
    scripts = _scripts(tmp_path)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    shim = tmp_path / "bin" / "unsloth.exe"
    shim.parent.mkdir(parents = True)
    shim.write_bytes(b"MZ")
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path)
    _as_windows(monkeypatch, scripts)

    studio._note_self_exe_locked(OSError(errno.EACCES, "in use"))

    assert "another program is running from" in capsys.readouterr().err


def test_without_a_launcher_the_note_carries_the_install_config(monkeypatch, tmp_path, capsys):
    """A pasted reinstall runs in a new shell with none of this process's env, so a
    custom root has to be spelled out or it repairs the wrong installation."""
    scripts = _scripts(tmp_path)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path)  # no bin/unsloth.exe
    monkeypatch.setattr(studio, "_STUDIO_HOME_IS_CUSTOM", True)
    _as_windows(monkeypatch, scripts)

    studio._note_self_exe_locked(OSError(errno.EACCES, "in use"))

    err = capsys.readouterr().err
    assert "install.ps1" in err
    assert "UNSLOTH_STUDIO_HOME" in err, "a custom root must survive into the command"
    assert str(tmp_path) in err


class _NoTorchManifest:
    @staticmethod
    def recorded_no_torch(root = None):
        return True


class _TorchManifest:
    @staticmethod
    def recorded_no_torch(root = None):
        return False


def test_the_recovery_command_keeps_no_torch_after_the_reader_is_gone(
    monkeypatch, tmp_path, capsys
):
    """recorded_no_torch lives in the package pip uninstalls. On the override path
    setup drops the manifest and pip removes that file before failing, so asking
    afterwards answers nothing -- and answering "no" reinstalls the whole PyTorch
    stack for someone who chose GGUF-only."""
    scripts = _scripts(tmp_path)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path)  # no bin/unsloth.exe
    _as_windows(monkeypatch, scripts)
    monkeypatch.setattr(studio, "_RECORDED_NO_TORCH", None)

    # While the install is intact.
    monkeypatch.setattr(
        studio._studio_deps, "load_install_manifest_module", lambda *a, **k: _NoTorchManifest
    )
    studio._snapshot_recorded_no_torch()

    # Then pip takes the reader away, exactly as it does before hitting the lock.
    def _gone(*a, **k):
        return None

    monkeypatch.setattr(studio._studio_deps, "load_install_manifest_module", _gone)
    studio._note_self_exe_locked(OSError(errno.EACCES, "in use"))

    err = capsys.readouterr().err
    assert "install.ps1" in err, "the reinstall fallback was hidden"
    assert (
        "UNSLOTH_NO_TORCH" in err
    ), "the recovery command lost no-torch mode, so it reinstalls the PyTorch stack"


def test_a_later_call_reflects_the_install_it_is_actually_for(monkeypatch, tmp_path, capsys):
    """The snapshot is a fallback, not a cache. Caching it would answer every later
    call with whatever the first one saw, so a second install root in the same
    process -- or the next test in the file -- inherits the first one's mode."""
    scripts = _scripts(tmp_path)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path)
    _as_windows(monkeypatch, scripts)
    monkeypatch.setattr(studio, "_RECORDED_NO_TORCH", None)

    # A first install that wanted torch, answered and remembered.
    monkeypatch.setattr(
        studio._studio_deps, "load_install_manifest_module", lambda *a, **k: _TorchManifest
    )
    studio._note_self_exe_locked(OSError(errno.EACCES, "in use"))
    assert "UNSLOTH_NO_TORCH" not in capsys.readouterr().err

    # A second one that did not. The manifest is still readable, so the live answer
    # is available and must win.
    monkeypatch.setattr(
        studio._studio_deps, "load_install_manifest_module", lambda *a, **k: _NoTorchManifest
    )
    studio._note_self_exe_locked(OSError(errno.EACCES, "in use"))

    assert "UNSLOTH_NO_TORCH" in capsys.readouterr().err, (
        "the second call answered for the first install"
    )


def test_no_torch_is_not_invented_when_nothing_recorded_it(monkeypatch, tmp_path, capsys):
    """The snapshot must not turn an unknown mode into a claim. A torch install that
    recovered with UNSLOTH_NO_TORCH=1 would come back without torch."""
    scripts = _scripts(tmp_path)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path)
    _as_windows(monkeypatch, scripts)
    monkeypatch.setattr(studio, "_RECORDED_NO_TORCH", None)
    monkeypatch.setattr(studio._studio_deps, "load_install_manifest_module", lambda *a, **k: None)

    studio._note_self_exe_locked(OSError(errno.EACCES, "in use"))

    assert "UNSLOTH_NO_TORCH" not in capsys.readouterr().err


# ── cleanup after a successful update ────────────────────────────────


def test_cleanup_does_not_delete_the_only_copy(monkeypatch, tmp_path):
    """Setup succeeding does not mean pip rewrote unsloth.exe. A dependency pass that
    finds the package already current reinstalls nothing, and the only copy is then
    the one renamed aside before it ran."""
    scripts = _scripts(tmp_path)
    exe = scripts / "unsloth.exe"
    stale = scripts / "unsloth.exe.deleteme"
    stale.write_bytes(b"MZ the only copy")
    _as_windows(monkeypatch, scripts)

    studio._cleanup_self_exe_lock_windows()

    assert exe.is_file(), "the CLI was deleted instead of restored"
    assert exe.read_bytes() == b"MZ the only copy"
    assert not stale.exists()


def test_cleanup_keeps_the_backup_when_the_restore_could_not_run(monkeypatch, tmp_path):
    """os.replace can fail transiently -- antivirus or another process holding the
    destination for a moment is enough -- and the restore reports rather than
    raises. Deleting the backup then is the one outcome with no way back."""
    scripts = _scripts(tmp_path)
    stale = scripts / "unsloth.exe.deleteme"
    stale.write_bytes(b"MZ the only copy")
    _as_windows(monkeypatch, scripts)
    monkeypatch.setattr(studio.os, "replace", _sharing_violation)

    studio._cleanup_self_exe_lock_windows()

    assert stale.is_file(), "the only remaining copy was deleted"
    assert stale.read_bytes() == b"MZ the only copy"


def test_cleanup_keeps_the_backup_when_the_exe_is_still_zero_byte(monkeypatch, tmp_path):
    scripts = _scripts(tmp_path)
    (scripts / "unsloth.exe").write_bytes(b"")
    stale = scripts / "unsloth.exe.deleteme"
    stale.write_bytes(b"MZ the working one")
    _as_windows(monkeypatch, scripts)
    monkeypatch.setattr(studio.os, "replace", _sharing_violation)

    studio._cleanup_self_exe_lock_windows()

    assert stale.is_file(), "the backup went while the destination was still torn"


def test_no_verify_survives_into_the_retry(monkeypatch, tmp_path, capsys):
    """Turning the scan off is a deliberate choice made because the install has
    files it reports and cannot repair, so a retry that turns it back on fails
    after the update it was meant to complete has already succeeded."""
    scripts = _scripts(tmp_path)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    shim = tmp_path / "bin" / "unsloth.exe"
    shim.parent.mkdir(parents = True)
    shim.write_bytes(b"MZ")
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path)
    _as_windows(monkeypatch, scripts)

    studio._note_self_exe_locked(OSError(errno.EACCES, "in use"), verify = False)

    assert "--no-verify" in capsys.readouterr().err


def test_verify_on_is_not_spelled_out(monkeypatch, tmp_path, capsys):
    scripts = _scripts(tmp_path)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    shim = tmp_path / "bin" / "unsloth.exe"
    shim.parent.mkdir(parents = True)
    shim.write_bytes(b"MZ")
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path)
    _as_windows(monkeypatch, scripts)

    studio._note_self_exe_locked(OSError(errno.EACCES, "in use"), verify = True)

    assert "--no-verify" not in capsys.readouterr().err


def test_cleanup_still_clears_the_orphan_when_pip_wrote_a_new_one(monkeypatch, tmp_path):
    """The case it was written for: a fresh binary is there, so the old one goes."""
    scripts = _scripts(tmp_path)
    exe = scripts / "unsloth.exe"
    exe.write_bytes(b"MZ fresh from pip")
    stale = scripts / "unsloth.exe.deleteme"
    stale.write_bytes(b"MZ previous")
    _as_windows(monkeypatch, scripts)

    studio._cleanup_self_exe_lock_windows()

    assert not stale.exists()
    assert exe.read_bytes() == b"MZ fresh from pip", "the new binary was clobbered"


def test_cleanup_replaces_a_zero_byte_exe(monkeypatch, tmp_path):
    """A zero-byte exe is a torn write, not a usable replacement."""
    scripts = _scripts(tmp_path)
    exe = scripts / "unsloth.exe"
    exe.write_bytes(b"")
    stale = scripts / "unsloth.exe.deleteme"
    stale.write_bytes(b"MZ the working one")
    _as_windows(monkeypatch, scripts)

    studio._cleanup_self_exe_lock_windows()

    assert exe.read_bytes() == b"MZ the working one"
