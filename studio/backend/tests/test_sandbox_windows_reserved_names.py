# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A file named nul, con or com1 in the session workdir is a file, not the device (#12473)."""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys

import pytest

from core.inference import os_sandbox

_RESERVED = ("nul", "con", "aux", "prn", "com1", "lpt1")
_EXTENDED = "\\\\?\\"


def _simulate_windows_stat(monkeypatch):
    """A plain Win32 stat of ...\\nul answers for the NUL device; the \\\\?\\ spelling reaches the file."""
    real_lstat, real_stat, real_scandir = os.lstat, os.stat, os.scandir

    def posix(path):
        return path[len(_EXTENDED) :].replace("\\", "/")

    def windows(real):
        def lookup(path, *args, **kwargs):
            path = os.fspath(path)
            if path.startswith(_EXTENDED):
                return real(posix(path), *args, **kwargs)
            stem = os.path.basename(path).split(".")[0].lower()
            if stem in _RESERVED:
                return os.stat_result((stat.S_IFCHR | 0o666, 0, 0, 0, 0, 0, 0, 0, 0, 0))
            return real(path, *args, **kwargs)

        return lookup

    monkeypatch.setattr(os_sandbox, "_NT_PATHS", True, raising = False)
    monkeypatch.setattr(os, "lstat", windows(real_lstat))
    monkeypatch.setattr(os, "stat", windows(real_stat))
    monkeypatch.setattr(
        os,
        "scandir",
        lambda path = ".": real_scandir(posix(path) if str(path).startswith(_EXTENDED) else path),
    )


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("C:\\Users\\a\\work\\nul", "\\\\?\\C:\\Users\\a\\work\\nul"),
        ("C:/Users/a/work/nul", "\\\\?\\C:\\Users\\a\\work\\nul"),
        ("C:\\Users\\a\\work\\.\\sub\\..\\nul", "\\\\?\\C:\\Users\\a\\work\\nul"),
        ("\\\\server\\share\\work\\nul", "\\\\?\\UNC\\server\\share\\work\\nul"),
        ("\\\\?\\C:\\Users\\a\\work\\nul", "\\\\?\\C:\\Users\\a\\work\\nul"),
        ("\\\\?\\UNC\\server\\share\\nul", "\\\\?\\UNC\\server\\share\\nul"),
    ],
)
def test_the_extended_spelling_of_a_workdir_entry(path, expected):
    assert os_sandbox._extended_path(path) == expected


@pytest.mark.skipif(sys.platform == "win32", reason = "simulates Windows stat on a POSIX filesystem")
@pytest.mark.parametrize("name", ["nul", "NUL", "con", "com1", "lpt1", "nul.txt"])
def test_a_reserved_name_file_does_not_refuse_the_session(monkeypatch, tmp_path, name):
    workdir = tmp_path / "work"
    (workdir / "sub").mkdir(parents = True)
    (workdir / name).write_text("hi\n")
    (workdir / "sub" / name).write_text("hi\n")
    _simulate_windows_stat(monkeypatch)

    assert os_sandbox.scan_workdir_for_host_channels(str(workdir)) == ()
    assert (workdir / name).read_text() == "hi\n"


@pytest.mark.skipif(sys.platform == "win32", reason = "simulates Windows stat on a POSIX filesystem")
def test_a_reserved_name_directory_is_walked_and_witnessed(monkeypatch, tmp_path):
    workdir = tmp_path / "work"
    (workdir / "nul").mkdir(parents = True)
    (workdir / "nul" / "out.txt").write_text("hi\n")
    _simulate_windows_stat(monkeypatch)

    witness: list = []
    assert os_sandbox.cache_share_hazard(str(workdir), witness) is None
    assert os_sandbox.directory_witness_matches(witness)
    (workdir / "nul" / "new.txt").write_text("hi\n")
    assert not os_sandbox.directory_witness_matches(witness)


@pytest.mark.skipif(sys.platform == "win32", reason = "simulates Windows stat on a POSIX filesystem")
def test_a_real_device_node_beside_a_nul_file_still_refuses(monkeypatch, tmp_path):
    """The extended spelling reaches what is on disk: a FIFO stays a FIFO."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "nul").write_text("hi\n")
    os.mkfifo(workdir / "planted")
    _simulate_windows_stat(monkeypatch)

    # Named as the caller spelled it, not by the \\?\ path the walk used.
    with pytest.raises(
        os_sandbox.WorkdirUnsafeError,
        match = f"device or IPC node: {re.escape(str(workdir / 'planted'))}$",
    ):
        os_sandbox.scan_workdir_for_host_channels(str(workdir))


@pytest.mark.skipif(sys.platform == "win32", reason = "simulates Windows stat on a POSIX filesystem")
def test_an_outside_hard_link_named_nul_still_refuses(monkeypatch, tmp_path):
    """The \\\\?\\ stat carries the real link count, so the hard-link check still sees the outside name."""
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n")
    workdir = tmp_path / "work"
    workdir.mkdir()
    os.link(outside, workdir / "nul")
    _simulate_windows_stat(monkeypatch)

    with pytest.raises(os_sandbox.WorkdirUnsafeError, match = "hard-linked from outside"):
        os_sandbox.scan_workdir_for_host_channels(str(workdir))


def _remove_extended(path):
    # pytest's own cleanup uses plain paths and cannot remove it.
    extended = os_sandbox._extended_path(path)
    if os.path.isdir(extended):
        shutil.rmtree(extended)
    elif os.path.lexists(extended):
        os.remove(extended)


@pytest.mark.skipif(sys.platform != "win32", reason = "needs the Win32 reserved-name mapping")
def test_a_nul_file_does_not_refuse_the_session_on_windows(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    planted = str(workdir / "nul")
    with open(os_sandbox._extended_path(planted), "w") as handle:
        handle.write("hi\n")
    try:
        # The defect this guards: a plain stat of the file answers for the NUL device.
        assert not stat.S_ISREG(os.lstat(planted).st_mode)
        assert stat.S_ISREG(os.lstat(os_sandbox._extended_path(planted)).st_mode)
        assert os_sandbox.scan_workdir_for_host_channels(str(workdir)) == ()
    finally:
        _remove_extended(planted)


@pytest.mark.skipif(sys.platform != "win32", reason = "needs the Win32 reserved-name mapping")
def test_a_nul_directory_is_walked_on_windows(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    planted = str(workdir / "nul")
    os.mkdir(os_sandbox._extended_path(planted))
    try:
        with open(os_sandbox._extended_path(os.path.join(planted, "out.txt")), "w") as handle:
            handle.write("hi\n")
        assert os_sandbox.scan_workdir_for_host_channels(str(workdir)) == ()
        os.link(
            os_sandbox._extended_path(os.path.join(planted, "out.txt")),
            str(tmp_path / "outside.txt"),
        )
        with pytest.raises(os_sandbox.WorkdirUnsafeError, match = "hard-linked from outside"):
            os_sandbox.scan_workdir_for_host_channels(str(workdir))
    finally:
        _remove_extended(planted)


@pytest.mark.skipif(sys.platform != "win32", reason = "needs Git Bash on Windows")
def test_git_bash_redirect_to_nul_does_not_refuse_the_session(tmp_path):
    from core.inference import tools

    bash = tools._windows_bash()
    if bash is None:
        pytest.skip("no trusted Git Bash on this host")
    workdir = tmp_path / "work"
    workdir.mkdir()
    planted = str(workdir / "nul")
    try:
        subprocess.run([bash, "-c", "echo hi > nul"], cwd = workdir, check = True, timeout = 60)
        assert stat.S_ISREG(
            os.lstat(os_sandbox._extended_path(planted)).st_mode
        ), "Git Bash did not leave a nul file"
        assert os_sandbox.scan_workdir_for_host_channels(str(workdir)) == ()
    finally:
        _remove_extended(planted)
