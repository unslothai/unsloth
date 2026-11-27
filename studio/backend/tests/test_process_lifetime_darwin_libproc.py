# SPDX-License-Identifier: AGPL-3.0-only

"""macOS pid identity from libproc: same lstart fields as `ps -o lstart=,comm=`, no subprocess."""

from __future__ import annotations

import calendar
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from utils import process_lifetime as pl

_START = calendar.timegm((2026, 11, 4, 9, 5, 7, 0, 0, 0))  # single-digit day: padded


def _info(
    pid,
    status = 2,
    start = _START,
    comm = b"python3.12",
):
    return SimpleNamespace(pbi_pid = pid, pbi_status = status, pbi_start_tvsec = start, pbi_comm = comm)


@pytest.fixture
def darwin(monkeypatch):
    monkeypatch.setattr(pl.sys, "platform", "darwin")
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_is_windows", lambda: False)
    for name in ("LC_ALL", "LC_TIME", "LANG"):
        monkeypatch.delenv(name, raising = False)
    spawned = []

    def run(argv, **kwargs):
        spawned.append(argv)
        if "state=" in argv:
            return SimpleNamespace(stdout = "Z+\n")
        # What macOS ps prints under TZ=UTC in the C locale, fixed text.
        return SimpleNamespace(
            stdout = "Wed Nov  4 09:05:07 2026      /opt/homebrew/bin/python3.12\n"
        )

    monkeypatch.setattr(subprocess, "run", run)
    return spawned


def test_libproc_identity_matches_the_ps_line_and_spawns_nothing(darwin, monkeypatch):
    monkeypatch.setattr(pl, "_darwin_bsdinfo", lambda pid: pl._DARWIN_ASK_PS)
    ps_line = pl._pid_identity(4242)
    assert darwin and darwin[-1][:3] == ["ps", "-o", "lstart=,comm="]
    darwin.clear()

    monkeypatch.setattr(pl, "_darwin_bsdinfo", lambda pid: _info(pid))
    native = pl._pid_identity(4242)
    assert darwin == []
    assert native.split()[:5] == ps_line.split()[:5] == ["Wed", "Nov", "4", "09:05:07", "2026"]
    # Records written by either path stay comparable in both directions.
    assert pl._same_identity(ps_line, native) and pl._same_identity(native, ps_line)
    assert not pl._same_identity(ps_line, native.replace("09:05:07", "09:05:08"))


def test_python_s_own_lc_time_does_not_change_the_native_identity(darwin, monkeypatch):
    # ps sees only the inherited environment (C here); Python's LC_TIME may still be set by a library.
    monkeypatch.setattr(pl, "_darwin_bsdinfo", lambda pid: _info(pid))
    expected = pl._pid_identity(4242)
    real_strftime = time.strftime

    def localized(fmt, *args):
        return real_strftime(fmt, *args).replace("Wed", "mer.").replace("Nov", "nov.")

    monkeypatch.setattr(time, "strftime", localized)
    assert pl._pid_identity(4242) == expected
    assert expected.split()[:5] == ["Wed", "Nov", "4", "09:05:07", "2026"]


def test_a_restarted_pid_is_a_different_identity(darwin, monkeypatch):
    monkeypatch.setattr(pl, "_darwin_bsdinfo", lambda pid: _info(pid))
    first = pl._pid_identity(77)
    monkeypatch.setattr(pl, "_darwin_bsdinfo", lambda pid: _info(pid, start = _START + 1))
    assert not pl._same_identity(first, pl._pid_identity(77))


def test_a_gone_pid_has_no_identity_without_asking_ps(darwin, monkeypatch):
    monkeypatch.setattr(pl, "_darwin_bsdinfo", lambda pid: None)
    assert pl._pid_identity(5) is None
    assert pl._pid_is_zombie(5) is False
    assert darwin == []


def test_zombie_state_comes_from_pbi_status(darwin, monkeypatch):
    monkeypatch.setattr(pl, "_darwin_bsdinfo", lambda pid: _info(pid, status = pl._DARWIN_SZOMB))
    assert pl._pid_is_zombie(9) is True
    # A zombie still has its start time, so adoption records it at once instead of retrying.
    assert pl._identity_for_record(9) is not None
    monkeypatch.setattr(pl, "_darwin_bsdinfo", lambda pid: _info(pid, status = 2))
    assert pl._pid_is_zombie(9) is False
    assert darwin == []


def test_libproc_failure_falls_back_to_ps(darwin, monkeypatch):
    monkeypatch.setattr(pl, "_darwin_bsdinfo", lambda pid: pl._DARWIN_ASK_PS)
    assert pl._pid_identity(11).endswith("/opt/homebrew/bin/python3.12")
    assert pl._pid_is_zombie(11) is True
    assert [argv[2] for argv in darwin] == ["lstart=,comm=", "state="]


@pytest.mark.parametrize(
    "value, native",
    [
        ("C", True),
        ("POSIX", True),
        ("C.UTF-8", True),
        ("en_US.UTF-8", True),
        ("de_DE.UTF-8", False),
        ("en_GB.UTF-8", False),
        ("ja_JP.UTF-8", False),
    ],
)
def test_only_locales_whose_percent_c_is_known_skip_ps(darwin, monkeypatch, value, native):
    monkeypatch.setenv("LANG", value)
    monkeypatch.setattr(pl, "_darwin_bsdinfo", lambda pid: _info(pid))
    pl._pid_identity(3)
    assert (darwin == []) is native


def test_lc_all_overrides_lang_like_ps_does(darwin, monkeypatch):
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    monkeypatch.setenv("LC_ALL", "fr_FR.UTF-8")
    monkeypatch.setattr(pl, "_darwin_bsdinfo", lambda pid: _info(pid))
    pl._pid_identity(3)
    assert darwin and darwin[-1][2] == "lstart=,comm="


def test_bsdinfo_layout_matches_xnu():
    """struct proc_bsdinfo is 136 bytes with pbi_start_tvsec at 120 (xnu bsd/sys/proc_info.h)."""
    import ctypes

    info = pl._darwin_bsdinfo_type()
    assert ctypes.sizeof(info) == 136
    assert info.pbi_comm.offset == 48 and info.pbi_start_tvsec.offset == 120


@pytest.mark.parametrize("pid", [True, 0, -1, 2**31, 2**32 + 4242])
def test_pids_outside_pid_t_never_reach_libproc(darwin, monkeypatch, pid):
    monkeypatch.setattr(pl, "_darwin_libproc", lambda: pytest.fail("called libproc"))
    assert pl._darwin_bsdinfo(pid) is pl._DARWIN_ASK_PS


def test_a_native_failure_falls_back_to_ps(darwin, monkeypatch):
    def broken():
        raise OSError("dyld failure")

    monkeypatch.setattr(pl, "_darwin_libproc", broken)
    assert pl._pid_identity(4242).split()[:5] == ["Wed", "Nov", "4", "09:05:07", "2026"]
    assert darwin and darwin[-1][2] == "lstart=,comm="


def test_a_failed_call_that_leaves_errno_alone_asks_ps(monkeypatch):
    import ctypes
    import errno

    ctypes.set_errno(errno.ESRCH)
    struct = pl._darwin_bsdinfo_type()
    lib = SimpleNamespace(proc_pidinfo = lambda *args: 0)
    monkeypatch.setattr(pl, "_darwin_libproc", lambda: (lib, struct))
    assert pl._darwin_bsdinfo(4242) is pl._DARWIN_ASK_PS


def test_ctypes_results_map_to_info_gone_or_ps(monkeypatch):
    import ctypes
    import errno

    struct = pl._darwin_bsdinfo_type()

    def answer(size, pid, err):
        def call(want, flavor, arg, buffer, length):
            assert (flavor, arg, length) == (pl._DARWIN_PROC_PIDTBSDINFO, 1, ctypes.sizeof(struct))
            ctypes.cast(buffer, ctypes.POINTER(struct)).contents.pbi_pid = pid
            ctypes.set_errno(err)
            return size

        return SimpleNamespace(proc_pidinfo = call)

    full = ctypes.sizeof(struct)
    for size, pid, err, expected in (
        (full, 4242, 0, "info"),
        (full, 4243, 0, "ps"),  # another pid's record
        (full - 8, 4242, 0, "ps"),  # short read
        (0, 0, errno.ESRCH, None),
        (0, 0, errno.EPERM, "ps"),
    ):
        monkeypatch.setattr(pl, "_darwin_libproc", lambda: (answer(size, pid, err), struct))
        got = pl._darwin_bsdinfo(4242)
        if expected == "info":
            assert got.pbi_pid == 4242
        elif expected == "ps":
            assert got is pl._DARWIN_ASK_PS
        else:
            assert got is None


@pytest.mark.parametrize(
    ("value", "native"),
    [("en_US.UTF-8@euro", False), ("C.anything", False), ("en_USX", False), ("C.UTF-8", True)],
)
def test_the_locale_gate_admits_only_known_spellings(monkeypatch, value, native):
    monkeypatch.setenv("LC_ALL", value)
    monkeypatch.delenv("PATH_LOCALE", raising = False)
    assert pl._darwin_c_locale_time() is native


def test_custom_locale_data_keeps_ps(monkeypatch):
    for name in ("LC_ALL", "LC_TIME", "LANG"):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("PATH_LOCALE", "/tmp/locales")
    assert pl._darwin_c_locale_time() is False


@pytest.mark.skipif(sys.platform != "darwin", reason = "native libproc and ps exist only on macOS")
def test_native_identity_matches_the_system_ps_on_this_mac():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        native = pl._darwin_native_identity(child.pid)
        assert native is not pl._DARWIN_ASK_PS and native is not None
        out = subprocess.run(
            ["ps", "-o", "lstart=,comm=", "-p", str(child.pid)],
            env = {**os.environ, "TZ": "UTC", "LC_ALL": "C"},
            capture_output = True,
            text = True,
            check = True,
        ).stdout
        assert native.split()[:5] == out.split()[:5]
        assert pl._pid_is_zombie(child.pid) is False
    finally:
        child.kill()
        child.wait()
