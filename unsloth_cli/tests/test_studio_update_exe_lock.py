# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The Windows self-lock on unsloth.exe (issue #7697).

Windows locks the directory entry an image was launched from rather than the file
behind it, so the rename here fails exactly when the update is running out of the
copy pip has to replace, and nothing later recovers: pip reaches the same file and
dies mid-uninstall reporting a permissions problem. These pin that the CLI keeps
going, since an update with no package change never touches the file, and that a
setup failure it accounts for is explained rather than reported as a permissions
problem.
"""

import errno
import os
from pathlib import Path

import pytest
import typer

from unsloth_cli.commands import studio


def _as_windows(monkeypatch, scripts_dir: Path):
    monkeypatch.setattr(studio.platform, "system", lambda: "Windows")
    monkeypatch.setattr(studio.sys, "executable", str(scripts_dir / "python.exe"))


def _sharing_violation(*a, **k):
    raise OSError(errno.EACCES, "The process cannot access the file")


def test_a_locked_exe_is_reported_but_does_not_abort(monkeypatch, tmp_path):
    """An update with no package change never touches the file and still works from
    here, so this must not become fatal for everyone who runs the exe directly."""
    scripts = tmp_path / "unsloth_studio" / "Scripts"
    scripts.mkdir(parents = True)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    _as_windows(monkeypatch, scripts)
    monkeypatch.setattr(studio.os, "replace", _sharing_violation)

    err = studio._release_self_exe_lock_windows()

    assert isinstance(err, OSError), "the lock has to be reported to the caller"


def test_the_message_points_at_the_launcher_when_there_is_one(
    monkeypatch, tmp_path, capsys
):
    """The PATH launcher is a second link to the same binary, and the lock follows
    the entry, so running through it leaves this copy replaceable."""
    scripts = tmp_path / "unsloth_studio" / "Scripts"
    scripts.mkdir(parents = True)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    shim = tmp_path / "bin" / "unsloth.exe"
    shim.parent.mkdir(parents = True)
    shim.write_bytes(b"MZ")
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path)
    _as_windows(monkeypatch, scripts)

    with pytest.raises(typer.Exit):
        studio._explain_self_exe_locked(OSError(errno.EACCES, "in use"))

    err = capsys.readouterr().err
    assert "unsloth studio update" in err
    assert str(shim) in err
    assert "install.ps1" not in err, "should not send them to the installer when a launcher exists"


def test_without_a_launcher_it_falls_back_to_the_installer(monkeypatch, tmp_path, capsys):
    scripts = tmp_path / "unsloth_studio" / "Scripts"
    scripts.mkdir(parents = True)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path)  # no bin/unsloth.exe
    _as_windows(monkeypatch, scripts)

    with pytest.raises(typer.Exit):
        studio._explain_self_exe_locked(OSError(errno.EACCES, "in use"))

    assert "install.ps1" in capsys.readouterr().err


def test_a_rename_that_works_stays_silent(monkeypatch, tmp_path, capsys):
    """The supported path, where the update runs through the launcher and this copy
    is free. It must not have become fatal for everyone."""
    scripts = tmp_path / "unsloth_studio" / "Scripts"
    scripts.mkdir(parents = True)
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

    assert studio._release_self_exe_lock_windows() is None  # must not raise


def test_setup_failure_after_a_lock_is_explained_not_left_as_permissions(
    monkeypatch, tmp_path, capsys
):
    """The end-to-end shape: the rename fails, setup then fails on the same file, and
    the user gets the cause instead of pip's 'Check the permissions.'"""
    scripts = tmp_path / "unsloth_studio" / "Scripts"
    scripts.mkdir(parents = True)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    shim = tmp_path / "bin" / "unsloth.exe"
    shim.parent.mkdir(parents = True)
    shim.write_bytes(b"MZ")
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path)
    _as_windows(monkeypatch, scripts)
    monkeypatch.setattr(studio.os, "replace", _sharing_violation)

    lock_err = studio._release_self_exe_lock_windows()
    assert lock_err is not None

    with pytest.raises(typer.Exit):
        studio._explain_self_exe_locked(lock_err)

    err = capsys.readouterr().err
    assert "in use by this process" in err
    assert "unsloth studio update" in err


def test_a_setup_failure_with_no_lock_is_not_explained_away(monkeypatch, tmp_path):
    """A rename that worked means the lock is not the cause, so an unrelated setup
    failure must keep its own error rather than be blamed on this."""
    scripts = tmp_path / "unsloth_studio" / "Scripts"
    scripts.mkdir(parents = True)
    (scripts / "unsloth.exe").write_bytes(b"MZ")
    _as_windows(monkeypatch, scripts)

    assert studio._release_self_exe_lock_windows() is None
