# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Managed-account tool confinement: capabilities, host mounts, raw devices, macOS profile."""

import ctypes
import os
import re
import subprocess
import sys

import pytest

from core.inference import tool_confinement, tool_path_approval as path_gate, tools
from utils.account_context import run_as

from .test_account_tool_confinement import (  # noqa: F401
    ALICE,
    _keep_the_interpreter_walk_out_of_the_fixtures,
    isolated,
)


class _FakeLibc:
    """Records the pre-exec's libc calls; Landlock calls succeed without touching the process."""

    def __init__(
        self,
        effective = 0xFFFFFFFF,
        last_cap = 40,
        capset_rc = 0,
        drop_rc = 0,
    ):
        self.effective = effective
        self.last_cap = last_cap
        self.capset_rc = capset_rc
        self.drop_rc = drop_rc
        self.dropped: list[int] = []
        self.capset_calls: list[tuple[int, ...]] = []
        self.prctl_calls: list[int] = []
        self._fd = None

    def syscall(self, number, *args):
        if number == tool_confinement._SYS_LANDLOCK_CREATE_RULESET:
            self._fd = os.open(os.devnull, os.O_RDONLY)
            return self._fd
        return 0

    def prctl(self, option, arg, *rest):
        self.prctl_calls.append(option)
        if option == 24:  # PR_CAPBSET_DROP
            if arg > self.last_cap:
                ctypes.set_errno(22)
                return -1
            if self.drop_rc < 0:
                ctypes.set_errno(1)
                return -1
            self.dropped.append(arg)
        return 0

    def capget(self, header, data):
        for i in range(2):
            data[i].effective = self.effective if i == 0 else 0
            data[i].permitted = self.effective if i == 0 else 0
            data[i].inheritable = 0
        return 0

    def capset(self, header, data):
        self.capset_calls.append(
            tuple(
                v
                for i in range(2)
                for v in (data[i].effective, data[i].permitted, data[i].inheritable)
            )
        )
        if self.capset_rc < 0:
            ctypes.set_errno(1)
        return self.capset_rc


def _run_preexec(monkeypatch, libc):
    monkeypatch.setattr(tool_confinement, "_libc", libc)
    tool_confinement._landlock_preexec(
        tool_confinement._handled_mask(6), [], tool_confinement._SCOPE_SIGNAL
    )


def test_managed_preexec_drops_the_bounding_set_and_every_capability(monkeypatch):
    """A root-run Studio must not hand a managed tool its capabilities (raw sockets, admin, ...)."""
    libc = _FakeLibc()
    _run_preexec(monkeypatch, libc)
    assert libc.dropped == list(range(41))
    assert libc.capset_calls == [(0, 0, 0, 0, 0, 0)]
    # Capabilities go before no_new_privs and the Landlock restrict.
    assert libc.prctl_calls.index(tool_confinement._PR_SET_NO_NEW_PRIVS) > libc.prctl_calls.index(
        24
    )


def test_managed_preexec_is_a_no_op_without_capabilities(monkeypatch):
    """A non-root Studio holds no capabilities: nothing to drop, nothing may fail."""
    libc = _FakeLibc(effective = 0, drop_rc = -1, capset_rc = -1)
    _run_preexec(monkeypatch, libc)
    assert libc.dropped == []
    assert libc.capset_calls == []
    assert tool_confinement._PR_SET_NO_NEW_PRIVS in libc.prctl_calls


def test_managed_preexec_fails_closed_when_capabilities_cannot_be_dropped(monkeypatch):
    with pytest.raises(OSError):
        _run_preexec(monkeypatch, _FakeLibc(capset_rc = -1))
    with pytest.raises(OSError):
        _run_preexec(monkeypatch, _FakeLibc(drop_rc = -1, capset_rc = -1))


def test_managed_preexec_still_clears_the_sets_without_setpcap(monkeypatch):
    """A non-root server with an ambient capability cannot touch the bounding set, but still drops it."""
    libc = _FakeLibc(drop_rc = -1)
    _run_preexec(monkeypatch, libc)
    assert libc.dropped == []
    assert libc.capset_calls and all(not any(call) for call in libc.capset_calls)


def _cap_lines(text):
    pairs = (line.split(":", 1) for line in text.splitlines() if line.startswith("Cap"))
    return {k: v.strip() for k, v in pairs}


def test_real_child_loses_capabilities_when_the_server_has_them():
    if sys.platform != "linux" or tool_confinement._libc is None:
        pytest.skip("Linux capabilities")
    parent = _cap_lines(open("/proc/self/status").read())
    if int(parent.get("CapEff", "0"), 16) == 0:
        pytest.skip("this process holds no capabilities (non-root): the drop is a no-op")
    out = subprocess.run(
        [sys.executable, "-c", "print(open('/proc/self/status').read())"],
        preexec_fn = tool_confinement._drop_capabilities,
        capture_output = True,
        text = True,
        check = True,
    ).stdout
    child = _cap_lines(out)
    for key in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"):
        assert int(child[key], 16) == 0, (key, child)


def test_host_and_removable_mounts_are_excluded_from_the_system_grant(tmp_path, monkeypatch):
    """/run/media (USB sticks), /run/host (a container's host root) and /run/mount are user storage."""
    if sys.platform != "linux":
        pytest.skip("Linux rule builder")
    run = tmp_path / "run"
    for name in ("media/alice/usb", "host/etc", "mount/utab.d", "systemd", "lock"):
        (run / name).mkdir(parents = True)
    _keep_the_interpreter_walk_out_of_the_fixtures(tmp_path, monkeypatch)
    monkeypatch.setattr(tool_confinement, "_SYSTEM_READ_ROOTS", (str(run),))
    monkeypatch.setattr(
        tool_confinement,
        "_PRIVATE_RUNTIME_ROOTS",
        tuple(
            str(run / os.path.relpath(p, "/run")) for p in tool_confinement._PRIVATE_RUNTIME_ROOTS
        ),
    )
    run_as(ALICE, tools._get_workdir, "chat")
    rules = run_as(ALICE, tool_confinement._landlock_rules, 6, tools._SANDBOX_SITE_DIR)
    granted = [p for p, _ in rules]
    resolved = str(run.resolve())
    assert f"{resolved}/systemd" in granted
    assert f"{resolved}/lock" in granted
    for private in ("media", "host", "mount"):
        assert not any(
            p == resolved or p == f"{resolved}/{private}" or p.startswith(f"{resolved}/{private}/")
            for p in granted
        ), private


@pytest.mark.skipif(os.sep != "/", reason = "POSIX paths")
def test_raw_device_reads_need_approval_but_pseudo_devices_stay_silent():
    for path in (
        "/dev/sda",
        "/dev/loop1",
        "/dev/nvme0n1",
        "/dev/mem",
        "/dev/kmem",
        "/dev/disk/by-id/x",
    ):
        assert path_gate._path_needs_approval(path) is True, path
    for path in (
        "/dev/null",
        "/dev/zero",
        "/dev/urandom",
        "/dev/random",
        "/dev/tty",
        "/dev/pts/0",
        "/dev/nvidiactl",
        "/dev/nvidia-uvm",
        "/dev/dri/renderD128",
        "/dev/kfd",
    ):
        assert path_gate._path_needs_approval(path) is False, path


@pytest.mark.skipif(os.sep != "/", reason = "POSIX paths")
def test_host_and_removable_mounts_under_run_need_approval():
    for path in ("/run/media/alice/usb/notes.txt", "/run/host/etc/hostname", "/run/mount/utab"):
        assert path_gate._path_needs_approval(path) is True, path
    assert path_gate._path_needs_approval("/run/systemd/resolve/resolv.conf") is False


def test_macos_managed_profile_filters_network_mach_sysctl_and_shm(tmp_path):
    writable = str(tmp_path / "sandbox")
    profile = tool_confinement.macos_profile(
        read_roots = [], hidden_roots = [], writable_roots = [writable]
    )
    lines = profile.splitlines()
    for unfiltered in (
        "(allow network*)",
        "(allow mach-lookup)",
        "(allow sysctl-read)",
        "(allow ipc-posix-shm)",
    ):
        assert unfiltered not in lines, unfiltered
    assert not re.search(r"\(allow network-outbound\)", profile)
    assert '(allow network-outbound (remote ip "*:*"))' in lines
    assert '(allow network-bind (local ip "*:*"))' in lines
    assert "(allow mach-lookup\n" in profile and "(global-name" in profile
    assert "(allow sysctl-read (sysctl-name" in profile
    assert "ipc-posix-name-regex" in profile
    # The account's own sockets (multiprocessing) still work.
    assert f'(allow network-outbound (remote unix-socket (subpath "{writable}")))' in lines
    assert f'(allow network-bind (local unix-socket (subpath "{writable}")))' in lines


@pytest.mark.skipif(
    sys.platform != "darwin" or not os.path.exists("/usr/bin/sandbox-exec"),
    reason = "live Seatbelt check",
)
def test_macos_managed_child_keeps_python_network_and_its_own_files(tmp_path):
    import socket
    import tempfile

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    outside = tempfile.mkdtemp(dir = "/tmp")
    host_socket_path = os.path.join(outside, "s")
    host_socket = socket.socket(socket.AF_UNIX)
    host_socket.bind(host_socket_path)
    host_socket.listen(1)
    code = f"""
import multiprocessing as mp, os, socket
print("cpus", os.cpu_count() is not None)
print("dns", bool(socket.getaddrinfo("localhost", {port})))
socket.create_connection(("127.0.0.1", {port}), timeout = 5).close()
print("tcp", True)
open("own.txt", "w").write("x")
print("write", open("own.txt").read() == "x")
parent, child = mp.Pipe()
proc = mp.get_context("fork").Process(target = lambda conn: conn.send(7), args = (child,))
proc.start(); proc.join(10)
print("fork", parent.recv() == 7)
try:
    socket.socket(socket.AF_UNIX).connect({host_socket_path!r})
    print("host_unix", "reached")
except OSError:
    print("host_unix", "refused")
"""
    workdir = run_as(ALICE, tools._get_workdir, "chat")
    confinement = run_as(ALICE, tools._account_confinement)
    assert confinement.mechanism == "sandbox-exec"
    try:
        result = subprocess.run(
            confinement.wrap([sys.executable, "-c", code]),
            cwd = workdir,
            capture_output = True,
            text = True,
            timeout = 60,
        )
    finally:
        listener.close()
        host_socket.close()
    out = result.stdout + result.stderr
    for line in (
        "cpus True",
        "dns True",
        "tcp True",
        "write True",
        "fork True",
        "host_unix refused",
    ):
        assert line in out, out
