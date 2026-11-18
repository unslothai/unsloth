# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account's tools cannot connect to the host's pathname Unix sockets (docker.sock, the
session D-Bus, ssh-agent): Landlock below ABI 9 does not mediate that connect()."""

import os
import shutil
import socket
import tempfile

import pytest

from core.inference import sandbox_seccomp, tools
from utils.account_context import run_as

from .test_account_tool_confinement import BOB, LANDLOCK, OWNER, auth_env, isolated, matrix  # noqa: F401

_PROBE = (
    "import socket, sys\n"
    "try:\n"
    "    s = socket.socket(socket.AF_UNIX)\n"
    "    s.connect(sys.argv[1]); print('RESULT:' + 'connected')\n"
    "except OSError as e:\n"
    "    print('RESULT:' + 'refused', e.errno)\n"
    "a, b = socket.socketpair(); a.send(b'k'); print('PAIR:' + b.recv(1).decode())\n"
    "try:\n"
    "    d, _ = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)\n"
    "    d.connect(sys.argv[2]); print('DGRAM:' + 'connected')\n"
    "except OSError as e:\n"
    "    print('DGRAM:' + 'refused', e.errno)\n"
    "try:\n"
    "    r, _ = socket.socketpair(socket.AF_UNIX, socket.SOCK_RAW)\n"
    "    r.connect(sys.argv[2]); print('RAW:' + 'connected')\n"
    "except OSError as e:\n"
    "    print('RAW:' + 'refused', e.errno)\n"
    "import ctypes, errno\n"
    "libc = ctypes.CDLL(None, use_errno=True)\n"
    "params = ctypes.create_string_buffer(120)\n"
    "fd = libc.syscall(425, 4, params)\n"
    "print('URING:' + ('refused' if fd < 0 and ctypes.get_errno() == errno.EPERM else 'open'))\n"
)


def _probe(account, host_socket, **kwargs) -> str:
    workdir = run_as(account, tools._get_workdir, "chat")
    with open(os.path.join(workdir, "probe_unix.py"), "w") as f:
        f.write(_PROBE)
    return run_as(
        account,
        tools._bash_exec,
        f"python probe_unix.py {host_socket} {host_socket}.d",
        session_id = "chat",
        **kwargs,
    )


@pytest.fixture
def host_socket():
    # Short and outside every root the account is granted (sun_path caps a path at 108 bytes).
    base = "/dev/shm" if os.path.isdir("/dev/shm") else None
    folder = tempfile.mkdtemp(prefix = "uhs", dir = base)
    path = os.path.join(folder, "s")
    server = socket.socket(socket.AF_UNIX)
    try:
        server.bind(path)
    except OSError as exc:
        server.close()
        shutil.rmtree(folder, ignore_errors = True)
        pytest.skip(f"cannot bind a probe socket: {exc}")
    server.listen(4)
    datagram = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    datagram.bind(path + ".d")
    try:
        yield path
    finally:
        datagram.close()
        server.close()
        shutil.rmtree(folder, ignore_errors = True)


@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
def test_managed_tool_cannot_reach_a_host_unix_socket(host_socket):
    out = _probe(BOB, host_socket)
    assert "RESULT:connected" not in out, out
    assert "RESULT:refused" in out, out
    # socketpair (multiprocessing, asyncio) keeps working.
    assert "PAIR:k" in out, out
    assert "DGRAM:refused" in out, out
    assert "RAW:refused" in out, out
    assert "URING:refused" in out, out


@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
def test_owner_tools_keep_unix_sockets(host_socket):
    out = _probe(OWNER, host_socket, disable_sandbox = True)
    assert "RESULT:connected" in out, out
    assert "DGRAM:connected" in out, out


def test_unix_filter_refuses_only_af_unix_socket():
    for machine in ("x86_64", "aarch64"):
        program = sandbox_seccomp.unix_socket_program(machine)
        socket_nr = sandbox_seccomp._ABIS[machine][0 + 1]
        assert (sandbox_seccomp._JEQ, 0, 4, socket_nr) in program
        assert (sandbox_seccomp._JEQ, 0, 1, sandbox_seccomp._AF_UNIX) in program
        assert program[-1] == (sandbox_seccomp._RET, 0, 0, sandbox_seccomp._ALLOW)
    with pytest.raises(RuntimeError):
        sandbox_seccomp.unix_socket_program("riscv64")
