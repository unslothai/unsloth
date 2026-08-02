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


# ── handing over to the launcher ─────────────────────────────────────


def _shimmed(monkeypatch, tmp_path):
    scripts = _scripts(tmp_path)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    shim = tmp_path / "bin" / "unsloth.exe"
    shim.parent.mkdir(parents = True)
    shim.write_bytes(b"MZ")
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path)
    _as_windows(monkeypatch, scripts)
    return scripts, shim


def test_a_locked_copy_hands_over_to_the_launcher(monkeypatch, tmp_path):
    """Continuing is not just a failed update. pip uninstalls before it installs,
    so it removes unsloth_cli and only then hits the locked stub, leaving an exe
    that starts and raises ModuleNotFoundError. Nothing destructive has happened
    at this point, so hand over while that is still true."""
    _scripts_dir, shim = _shimmed(monkeypatch, tmp_path)
    seen = {}

    def fake_run(argv, env = None):
        seen["argv"] = argv
        seen["env"] = env
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(studio.subprocess, "run", fake_run)

    with pytest.raises(studio.typer.Exit):
        studio._reexec_through_launcher_windows(
            local = True, package = "unsloth", verbose = False, verify = False,
        )

    assert seen["argv"][0] == str(shim), "the child did not run through the launcher"
    assert "--local" in seen["argv"] and "--no-verify" in seen["argv"]
    assert seen["env"][studio._REEXEC_ENV] == "1"


def test_the_hand_over_happens_once(monkeypatch, tmp_path):
    """If the launcher resolves back to this same entry the child hits the same
    lock, and without the guard it would do so forever."""
    _shimmed(monkeypatch, tmp_path)
    monkeypatch.setenv(studio._REEXEC_ENV, "1")

    def explode(*a, **k):
        raise AssertionError("re-executed twice")

    monkeypatch.setattr(studio.subprocess, "run", explode)

    studio._reexec_through_launcher_windows(
        local = False, package = "unsloth", verbose = False, verify = True,
    )


def test_the_childs_exit_code_is_this_processs_exit_code(monkeypatch, tmp_path):
    _shimmed(monkeypatch, tmp_path)
    monkeypatch.setattr(
        studio.subprocess, "run",
        lambda argv, env = None: type("R", (), {"returncode": 3})(),
    )

    with pytest.raises(studio.typer.Exit) as excinfo:
        studio._reexec_through_launcher_windows(
            local = False, package = "unsloth", verbose = False, verify = True,
        )
    assert excinfo.value.exit_code == 3


def test_no_launcher_means_carry_on_and_report(monkeypatch, tmp_path):
    """Returning lets the caller reach the failure it can at least explain. A
    broken hand-off must not become a second failure mode on top of the first."""
    scripts = _scripts(tmp_path)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path)  # no bin/unsloth.exe
    _as_windows(monkeypatch, scripts)
    monkeypatch.setattr(
        studio.subprocess, "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    studio._reexec_through_launcher_windows(
        local = False, package = "unsloth", verbose = False, verify = True,
    )


def test_a_launcher_that_will_not_start_is_not_handed_to(monkeypatch, tmp_path):
    _shimmed(monkeypatch, tmp_path)

    def boom(*a, **k):
        raise OSError("cannot exec")

    monkeypatch.setattr(studio.subprocess, "run", boom)

    # Returns rather than raising: the caller still owns the real failure.
    studio._reexec_through_launcher_windows(
        local = False, package = "unsloth", verbose = False, verify = True,
    )


def test_other_platforms_never_hand_over(monkeypatch, tmp_path):
    _shimmed(monkeypatch, tmp_path)
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        studio.subprocess, "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    studio._reexec_through_launcher_windows(
        local = False, package = "unsloth", verbose = False, verify = True,
    )


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
