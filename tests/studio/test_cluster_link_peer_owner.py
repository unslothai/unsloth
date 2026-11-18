# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The loopback relay admits only local clients of this user on macOS and Windows too.

Every local account can reach a 127.0.0.1 port, and the relay authenticates to the worker with this
user's cluster credentials. Linux reads /proc/net/tcp{,6}; macOS asks libproc and Windows asks
GetExtendedTcpTable + the owning process's token. The OS APIs are faked here at the ctypes level
(``ctypes.CDLL`` / ``ctypes.WinDLL``), so these run anywhere; the real-socket tests at the bottom
run the real lookup on whichever OS executes them.
"""

from __future__ import annotations

import builtins
import ctypes
import io
import os
import socket
import struct
import sys
import time
from types import SimpleNamespace

import pytest

from studio.cluster_link import splice as splice_mod

ESCAPE_ENV = "UNSLOTH_CLUSTER_RELAY_ANY_LOCAL"
_LE_64 = sys.byteorder == "little" and struct.calcsize("P") == 8
pytestmark = pytest.mark.skipif(not _LE_64, reason = "fixtures are 64-bit little-endian layouts")

CLIENT = ("127.0.0.1", 50123)
SERVER = ("127.0.0.1", 8090)
CLIENT6 = ("::1", 50123, 0, 0)
SERVER6 = ("::1", 8090, 0, 0)
MAPPED_CLIENT = ("::ffff:127.0.0.1", 50123, 0, 0)
MAPPED_SERVER = ("::ffff:127.0.0.1", 8090, 0, 0)


def _sock(client, server):
    return SimpleNamespace(getpeername = lambda: client, getsockname = lambda: server)


@pytest.fixture(autouse = True)
def _no_escape_hatch(monkeypatch):
    monkeypatch.delenv(ESCAPE_ENV, raising = False)


def _net16(port: int) -> int:
    """A port in network byte order in the low 16 bits of a little-endian DWORD/int."""
    return struct.unpack("<H", struct.pack(">H", port))[0]


# ---------------------------------------------------------------------------------------------
# Windows: GetExtendedTcpTable + OpenProcess / OpenProcessToken / GetTokenInformation / EqualSid
# ---------------------------------------------------------------------------------------------

_ESTAB, _TIME_WAIT = 5, 11
_MY_PID, _OTHER_PID, _DENIED_PID = 4100, 4200, 4300
_MY_SID, _OTHER_SID = 0x5000, 0x6000
_CURRENT = 0xFFFF


def _v4_row(state, local, remote, pid):
    a = struct.unpack("<I", socket.inet_aton(local[0]))[0]
    b = struct.unpack("<I", socket.inet_aton(remote[0]))[0]
    return struct.pack("<IIIIII", state, a, _net16(local[1]), b, _net16(remote[1]), pid)


def _v6_row(state, local, remote, pid):
    a = socket.inet_pton(socket.AF_INET6, local[0])
    b = socket.inet_pton(socket.AF_INET6, remote[0])
    return struct.pack("<16sII16sIIII", a, 0, _net16(local[1]), b, 0, _net16(remote[1]), state, pid)


def _table(rows):
    return struct.pack("<I", len(rows)) + b"".join(rows)


class FakeWindows:
    def __init__(
        self,
        v4 = (),
        v6 = (),
        sids = None,
        fail_af = None,
        denied = (_DENIED_PID,),
    ):
        self.tables = {2: _table(list(v4)), 23: _table(list(v6))}
        self.fail_af = fail_af or {}
        self.sids = {_MY_PID: _MY_SID, _OTHER_PID: _OTHER_SID, **(sids or {})}
        self.denied = set(denied)
        self.calls = {"table": 0, "open": []}

    def dll(self, name, **_kw):
        name = name.lower()
        if name == "iphlpapi":
            return SimpleNamespace(GetExtendedTcpTable = _fn(self.get_table))
        if name == "kernel32":
            return SimpleNamespace(
                OpenProcess = _fn(self.open_process),
                GetCurrentProcess = _fn(lambda: _CURRENT),
                CloseHandle = _fn(lambda h: 1),
            )
        if name == "advapi32":
            return SimpleNamespace(
                OpenProcessToken = _fn(self.open_token),
                GetTokenInformation = _fn(self.token_info),
                EqualSid = _fn(lambda a, b: int(bool(a) and a == b)),
                IsValidSid = _fn(lambda a: int(bool(a))),
            )
        raise OSError(f"no {name}")

    def get_table(self, buf, psize, order, af, cls, reserved):
        self.calls["table"] += 1
        assert cls == 4  # TCP_TABLE_OWNER_PID_CONNECTIONS
        if af in self.fail_af:
            return self.fail_af[af]
        data = self.tables[af]
        size = psize._obj
        if buf is None or size.value < len(data):
            size.value = len(data)
            return 122  # ERROR_INSUFFICIENT_BUFFER
        ctypes.memmove(buf, data, len(data))
        size.value = len(data)
        return 0

    def open_process(self, access, inherit, pid):
        self.calls["open"].append(pid)
        assert access == 0x1000  # PROCESS_QUERY_LIMITED_INFORMATION
        if pid in self.denied or pid not in self.sids:
            return 0
        return 0x10000 + pid

    def open_token(self, handle, access, ptoken):
        pid = _MY_PID if handle == _CURRENT else handle - 0x10000
        ptoken._obj.value = 0x20000 + pid
        return 1

    def token_info(self, token, cls, buf, size, pneed):
        assert cls == 1  # TokenUser
        pneed._obj.value = 16
        if buf is None or size < 16:
            return 0
        sid = self.sids[token - 0x20000]
        ctypes.memmove(buf, struct.pack("<QI", sid, 0), 12)
        return 1


def _fn(f):
    # Real foreign functions convert ctypes scalars (HANDLE, DWORD) to ints via argtypes; do the same.
    def wrapper(*a):
        return f(*(x.value if isinstance(x, ctypes._SimpleCData) else x for x in a))

    return wrapper


@pytest.fixture
def windows(monkeypatch):
    from ctypes import wintypes  # noqa: F401  (imported before the platform is faked)
    def install(fake):
        monkeypatch.setattr(ctypes, "WinDLL", fake.dll, raising = False)
        monkeypatch.setattr(sys, "platform", "win32")
        return fake

    return install


def test_windows_admits_a_client_owned_by_this_user(windows):
    fake = windows(FakeWindows(v4 = [_v4_row(_ESTAB, CLIENT, SERVER, _MY_PID)]))
    assert splice_mod.local_client_is_ours(_sock(CLIENT, SERVER)) is True
    assert fake.calls["table"] and fake.calls["open"] == [_MY_PID]  # the lookup really ran


def test_windows_refuses_a_client_owned_by_another_user(windows):
    windows(FakeWindows(v4 = [_v4_row(_ESTAB, CLIENT, SERVER, _OTHER_PID)]))
    assert splice_mod.local_client_is_ours(_sock(CLIENT, SERVER)) is False


def test_windows_refuses_a_process_it_cannot_open(windows):
    windows(FakeWindows(v4 = [_v4_row(_ESTAB, CLIENT, SERVER, _DENIED_PID)]))
    assert splice_mod.local_client_is_ours(_sock(CLIENT, SERVER)) is False


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param([], id = "no_rows"),
        pytest.param([_v4_row(_TIME_WAIT, CLIENT, SERVER, _MY_PID)], id = "not_established"),
        pytest.param(
            [_v4_row(_ESTAB, ("127.0.0.2", CLIENT[1]), SERVER, _MY_PID)], id = "other_address"
        ),
        pytest.param(
            [_v4_row(_ESTAB, (CLIENT[0], CLIENT[1] + 1), SERVER, _MY_PID)], id = "other_port"
        ),
        pytest.param([_v4_row(_ESTAB, SERVER, CLIENT, _MY_PID)], id = "listener_end_only"),
    ],
)
def test_windows_refuses_when_the_exact_live_tuple_is_not_found(windows, rows):
    fake = windows(FakeWindows(v4 = rows))
    assert splice_mod.local_client_is_ours(_sock(CLIENT, SERVER)) is False
    assert fake.calls["table"]


def test_windows_refuses_a_tuple_two_processes_claim(windows):
    rows = [_v4_row(_ESTAB, CLIENT, SERVER, _MY_PID), _v4_row(_ESTAB, CLIENT, SERVER, _OTHER_PID)]
    windows(FakeWindows(v4 = rows))
    assert splice_mod.local_client_is_ours(_sock(CLIENT, SERVER)) is False


@pytest.mark.parametrize(
    "names, v6",
    [
        pytest.param((CLIENT6, SERVER6), [_v6_row(_ESTAB, CLIENT6, SERVER6, _MY_PID)], id = "ipv6"),
        pytest.param(
            (MAPPED_CLIENT, MAPPED_SERVER),
            [_v6_row(_ESTAB, MAPPED_CLIENT, MAPPED_SERVER, _MY_PID)],
            id = "v4_mapped",
        ),
    ],
)
def test_windows_reads_the_ipv6_table(windows, names, v6):
    fake = windows(FakeWindows(v6 = v6))
    assert splice_mod.local_client_is_ours(_sock(*names)) is True
    assert fake.calls["open"] == [_MY_PID]


def test_windows_ipv6_other_user_refused(windows):
    windows(FakeWindows(v6 = [_v6_row(_ESTAB, CLIENT6, SERVER6, _OTHER_PID)]))
    assert splice_mod.local_client_is_ours(_sock(CLIENT6, SERVER6)) is False


def test_windows_lookup_failure_refuses_unless_the_escape_hatch_is_set(windows, monkeypatch):
    windows(FakeWindows(v4 = [_v4_row(_ESTAB, CLIENT, SERVER, _MY_PID)], fail_af = {2: 87}))
    assert splice_mod.local_client_is_ours(_sock(CLIENT, SERVER)) is False
    monkeypatch.setenv(ESCAPE_ENV, "1")
    assert splice_mod.local_client_is_ours(_sock(CLIENT, SERVER)) is True


def test_windows_missing_iphlpapi_refuses(monkeypatch):
    def no_dll(name, **_kw):
        raise OSError("not here")

    monkeypatch.setattr(ctypes, "WinDLL", no_dll, raising = False)
    monkeypatch.setattr(sys, "platform", "win32")
    assert splice_mod.local_client_is_ours(_sock(CLIENT, SERVER)) is False


# ---------------------------------------------------------------------------------------------
# macOS: libproc proc_listpids / proc_pidinfo / proc_pidfdinfo
# ---------------------------------------------------------------------------------------------

_SOCKET_FDINFO = 792  # sizeof(struct socket_fdinfo)
# Offsets in struct socket_fdinfo (xnu sys/proc_info.h), computed by hand from the C layout:
# proc_fileinfo is 24 bytes, then socket_info: vinfo_stat (136) ... soi_kind at +232, soi_proto at
# +240; in_sockinfo: fport +0, lport +4, vflag +24, faddr +32, laddr +48; tcpsi_state at +80.
_KIND, _PROTO = 24 + 232, 24 + 240
_TSI_ESTABLISHED, _TSI_TIME_WAIT = 4, 10


def _fdinfo(
    local,
    remote,
    state = _TSI_ESTABLISHED,
    kind = 2,
):
    buf = bytearray(_SOCKET_FDINFO)
    struct.pack_into("<i", buf, _KIND, kind)
    v6 = ":" in local[0]
    struct.pack_into("<i", buf, _PROTO + 0, _net16(remote[1]))
    struct.pack_into("<i", buf, _PROTO + 4, _net16(local[1]))
    struct.pack_into("<B", buf, _PROTO + 24, 0x2 if v6 else 0x1)
    for off, host in ((32, remote[0]), (48, local[0])):
        if v6:
            raw = socket.inet_pton(socket.AF_INET6, host)
        else:
            raw = bytes(12) + socket.inet_aton(host)  # in4in6_addr: pad then in_addr
        buf[_PROTO + off : _PROTO + off + 16] = raw
    struct.pack_into("<i", buf, _PROTO + 80, state)
    return bytes(buf)


class FakeLibProc:
    """procs: pid -> (uid, {fd: fdinfo bytes}) for visible processes, None for EPERM ones."""

    def __init__(self, procs):
        self.procs = procs
        self.calls = {"listpids": 0, "fds": [], "uid": []}

    def cdll(self, path, **_kw):
        return SimpleNamespace(
            proc_listpids = _fn(self.listpids),
            proc_pidinfo = _fn(self.pidinfo),
            proc_pidfdinfo = _fn(self.pidfdinfo),
        )

    def listpids(self, kind, info, buf, size):
        self.calls["listpids"] += 1
        assert kind == 1  # PROC_ALL_PIDS
        data = struct.pack(f"<{len(self.procs)}i", *self.procs)
        if buf is None:
            return len(data)
        ctypes.memmove(buf, data, min(len(data), size))
        return min(len(data), size)

    def pidinfo(self, pid, flavor, arg, buf, size):
        proc = self.procs.get(pid)
        if proc is None:
            return 0  # EPERM: another user's process
        uid, fds = proc
        if flavor == 1:  # PROC_PIDLISTFDS
            self.calls["fds"].append(pid)
            data = b"".join(struct.pack("<iI", fd, 2) for fd in fds) + struct.pack("<iI", 99, 1)
        elif flavor == 3:  # PROC_PIDTBSDINFO
            self.calls["uid"].append(pid)
            data = bytearray(136)
            struct.pack_into("<I", data, 20, uid)
            data = bytes(data)
        else:
            raise AssertionError(flavor)
        if buf is None:
            return len(data)
        n = min(len(data), size)
        ctypes.memmove(buf, data, n)
        return n

    def pidfdinfo(self, pid, fd, flavor, buf, size):
        assert flavor == 3  # PROC_PIDFDSOCKETINFO
        data = self.procs[pid][1][fd]
        assert size >= len(data)
        ctypes.memmove(buf, data, len(data))
        return len(data)


@pytest.fixture
def darwin(monkeypatch):
    def install(fake):
        monkeypatch.setattr(ctypes, "CDLL", fake.cdll)
        monkeypatch.setattr(sys, "platform", "darwin")
        return fake

    return install


ME, OTHER = os.getuid() if hasattr(os, "getuid") else 501, 0x7777


def _other_users_listener():
    # The listener's own process holds the accepted end (local=SERVER, remote=CLIENT).
    return (ME, {3: _fdinfo(SERVER, CLIENT), 4: _fdinfo(("127.0.0.1", 8090), ("0.0.0.0", 0), 1)})


@pytest.fixture(autouse = True)
def _getuid(monkeypatch):
    monkeypatch.setattr(os, "getuid", lambda: ME, raising = False)


def test_darwin_admits_a_client_owned_by_this_user(darwin):
    fake = darwin(
        FakeLibProc({10: _other_users_listener(), 20: (ME, {5: _fdinfo(CLIENT, SERVER)})})
    )
    assert splice_mod.local_client_is_ours(_sock(CLIENT, SERVER)) is True
    assert fake.calls["listpids"] and fake.calls["uid"] == [20]  # the lookup really ran


def test_darwin_refuses_a_client_owned_by_another_uid(darwin):
    # Running as root shows every process: the owner is visible but is not us.
    darwin(FakeLibProc({10: _other_users_listener(), 20: (OTHER, {5: _fdinfo(CLIENT, SERVER)})}))
    assert splice_mod.local_client_is_ours(_sock(CLIENT, SERVER)) is False


def test_darwin_refuses_a_client_in_a_process_it_cannot_inspect(darwin):
    # The usual case: another user's process answers EPERM, so the tuple is nowhere to be seen.
    fake = darwin(FakeLibProc({10: _other_users_listener(), 20: None}))
    assert splice_mod.local_client_is_ours(_sock(CLIENT, SERVER)) is False
    assert 10 in fake.calls["fds"]


@pytest.mark.parametrize(
    "info",
    [
        pytest.param(_fdinfo(CLIENT, SERVER, state = _TSI_TIME_WAIT), id = "not_established"),
        pytest.param(_fdinfo(("127.0.0.2", CLIENT[1]), SERVER), id = "other_address"),
        pytest.param(_fdinfo((CLIENT[0], CLIENT[1] + 1), SERVER), id = "other_port"),
        pytest.param(_fdinfo(CLIENT, SERVER, kind = 1), id = "not_tcp"),
    ],
)
def test_darwin_refuses_when_the_exact_live_tuple_is_not_found(darwin, info):
    darwin(FakeLibProc({10: _other_users_listener(), 20: (ME, {5: info})}))
    assert splice_mod.local_client_is_ours(_sock(CLIENT, SERVER)) is False


def test_darwin_refuses_a_tuple_shared_with_another_uid(darwin):
    procs = {
        20: (ME, {5: _fdinfo(CLIENT, SERVER)}),
        30: (OTHER, {7: _fdinfo(CLIENT, SERVER)}),
    }
    darwin(FakeLibProc(procs))
    assert splice_mod.local_client_is_ours(_sock(CLIENT, SERVER)) is False


@pytest.mark.parametrize(
    "names, info",
    [
        pytest.param((CLIENT6, SERVER6), _fdinfo(CLIENT6, SERVER6), id = "ipv6"),
        pytest.param((MAPPED_CLIENT, MAPPED_SERVER), _fdinfo(CLIENT, SERVER), id = "v4_mapped_names"),
    ],
)
def test_darwin_matches_ipv6_and_mapped_names(darwin, names, info):
    fake = darwin(FakeLibProc({20: (ME, {5: info})}))
    assert splice_mod.local_client_is_ours(_sock(*names)) is True
    assert fake.calls["uid"] == [20]


def test_darwin_lookup_failure_refuses_unless_the_escape_hatch_is_set(monkeypatch):
    def no_lib(path, **_kw):
        raise OSError("no libproc")

    monkeypatch.setattr(ctypes, "CDLL", no_lib)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert splice_mod.local_client_is_ours(_sock(CLIENT, SERVER)) is False
    monkeypatch.setenv(ESCAPE_ENV, "1")
    assert splice_mod.local_client_is_ours(_sock(CLIENT, SERVER)) is True


def test_darwin_listpids_failure_refuses(darwin):
    fake = FakeLibProc({20: (ME, {5: _fdinfo(CLIENT, SERVER)})})
    fake.listpids = lambda *a: -1
    darwin(fake)
    assert splice_mod.local_client_is_ours(_sock(CLIENT, SERVER)) is False


def test_an_unknown_platform_refuses_unless_the_escape_hatch_is_set(monkeypatch):
    monkeypatch.setattr(sys, "platform", "freebsd14")
    assert splice_mod.local_client_is_ours(_sock(CLIENT, SERVER)) is False
    monkeypatch.setenv(ESCAPE_ENV, "1")
    assert splice_mod.local_client_is_ours(_sock(CLIENT, SERVER)) is True


# ---------------------------------------------------------------------------------------------
# Linux: unchanged (/proc/net/tcp, no escape hatch, no libproc / iphlpapi)
# ---------------------------------------------------------------------------------------------

_PROC_HEADER = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode"


def _proc_row(uid):
    # 127.0.0.1:50123 -> 127.0.0.1:8090, ESTABLISHED, host-order little-endian words.
    return f"   0: 0100007F:C3CB 0100007F:1F9A 01 00000000:00000000 00:00000000 00000000 {uid:>5}        0 1 1 0 20 4 30 10 -1"


@pytest.mark.parametrize("uid, admitted", [(ME, True), (OTHER, False)])
def test_linux_still_reads_proc_net_tcp_and_ignores_the_escape_hatch(monkeypatch, uid, admitted):
    files = {
        "/proc/net/tcp": f"{_PROC_HEADER}\n{_proc_row(uid)}\n",
        "/proc/net/tcp6": f"{_PROC_HEADER}\n",
    }
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        return io.StringIO(files[path]) if path in files else real_open(path, *args, **kwargs)

    def boom(*_a, **_kw):
        raise AssertionError("Linux must not use libproc / iphlpapi")

    monkeypatch.setattr(splice_mod, "open", fake_open, raising = False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(ctypes, "CDLL", boom)
    monkeypatch.setattr(ctypes, "WinDLL", boom, raising = False)
    monkeypatch.setenv(ESCAPE_ENV, "1")  # Linux has a working lookup: the hatch does not apply
    assert splice_mod.local_client_is_ours(_sock(CLIENT, SERVER)) is admitted


# ---------------------------------------------------------------------------------------------
# The real lookup on the OS running the test (Linux, macOS, Windows)
# ---------------------------------------------------------------------------------------------

_REAL = sys.platform.startswith("linux") or sys.platform in ("darwin", "win32")


def _real_pair(family, host):
    server = socket.socket(family, socket.SOCK_STREAM)
    try:
        server.bind((host, 0))
    except OSError:
        server.close()
        pytest.skip(f"{host} is not bindable here")
    server.listen(1)
    client = socket.create_connection(server.getsockname()[:2])
    conn, _ = server.accept()
    return server, client, conn


@pytest.mark.skipif(not _REAL, reason = "no owner lookup on this OS")
@pytest.mark.parametrize(
    "family, host",
    [
        pytest.param(socket.AF_INET, "127.0.0.1", id = "ipv4"),
        pytest.param(socket.AF_INET6, "::1", id = "ipv6"),
    ],
)
def test_real_loopback_client_of_this_process_is_ours(monkeypatch, family, host):
    monkeypatch.undo()  # the real OS and the real getuid
    monkeypatch.delenv(ESCAPE_ENV, raising = False)
    server, client, conn = _real_pair(family, host)
    try:
        start = time.perf_counter()
        assert splice_mod.local_client_is_ours(conn) is True
        elapsed = time.perf_counter() - start
        print(f"owner lookup on {sys.platform} took {elapsed * 1000:.1f} ms")
        assert elapsed < 5.0
        # A tuple no live connection has (the client's port + 1 is not this connection): refused,
        # so the True above came from finding the tuple, not from a lookup that admits anything.
        peer = conn.getpeername()
        ghost = (peer[0], peer[1] + 1 if peer[1] < 65535 else peer[1] - 1, *peer[2:])
        assert splice_mod.local_client_is_ours(_sock(ghost, conn.getsockname())) is False
    finally:
        for s in (conn, client, server):
            s.close()
