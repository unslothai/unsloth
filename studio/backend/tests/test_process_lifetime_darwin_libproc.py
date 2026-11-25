# SPDX-License-Identifier: AGPL-3.0-only

"""macOS pid identity from libproc: same lstart fields as `ps -o lstart=,comm=`, no subprocess."""

from __future__ import annotations

import calendar
import subprocess
import time
from types import SimpleNamespace

import pytest

from utils import process_lifetime as pl

_START = calendar.timegm((2026, 11, 4, 9, 5, 7, 0, 0, 0))  # single-digit day: %e pads it


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
        # What macOS ps prints under TZ=UTC: strftime("%c") left-justified in 28, then comm.
        lstart = time.strftime("%a %b %e %H:%M:%S %Y", time.gmtime(_START))
        return SimpleNamespace(stdout = f"{lstart:<28} /opt/homebrew/bin/python3.12\n")

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
