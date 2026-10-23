# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Who owns the client end of a loopback TCP connection, on macOS and Windows.

Every local account can reach a 127.0.0.1 port, so a loopback listener that acts with this user's
credentials must check that the connecting process runs as this user. Linux reads /proc/net/tcp{,6}
(see ``splice._loopback_owner_uid``); this module answers the same question elsewhere:

- Windows: GetExtendedTcpTable (TCP_TABLE_OWNER_PID_CONNECTIONS, IPv4 and IPv6) names the pid that
  owns the exact ESTABLISHED (local, remote) tuple, and that process's token user SID must equal
  ours. A process this user cannot open is not ours.
- macOS: libproc lists every pid and the TCP sockets of each; the pid holding the exact ESTABLISHED
  tuple must run as our uid. Another user's processes cannot be inspected (EPERM), so a tuple not
  found among the visible ones is refused: that is exactly the case being guarded against.

Anything that keeps the lookup itself from working (no libproc, a failing table call, an unknown
platform) refuses the client and logs why once. On a host where the lookup cannot work,
``UNSLOTH_CLUSTER_RELAY_ANY_LOCAL=1`` admits every local client again (the loopback bind alone).
Nothing is cached: each connection is looked up afresh.
"""

from __future__ import annotations

import ctypes
import ipaddress
import logging
import os
import socket
import struct
import sys
from typing import Callable, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

ANY_LOCAL_ENV = "UNSLOTH_CLUSTER_RELAY_ANY_LOCAL"

Endpoint = Tuple[ipaddress._BaseAddress, int]


class OwnerLookupError(Exception):
    """The owner lookup itself cannot run here (as opposed to: the owner is someone else)."""


_logged: set = set()


def _log_once(reason: str) -> None:
    if reason in _logged:
        return
    _logged.add(reason)
    logger.warning(
        "cluster relay: cannot tell which user owns a local client (%s); refusing local clients. "
        "Set %s=1 to admit every local client on this host.",
        reason,
        ANY_LOCAL_ENV,
    )


def any_local_allowed() -> bool:
    return os.environ.get(ANY_LOCAL_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def endpoint(addr, port: int) -> Optional[Endpoint]:
    """Normalise an address (str or packed bytes) + port; IPv4-mapped IPv6 becomes IPv4."""
    try:
        if isinstance(addr, (bytes, bytearray)):
            ip = ipaddress.ip_address(bytes(addr))
        else:
            ip = ipaddress.ip_address(str(addr).split("%", 1)[0])
    except ValueError:
        return None
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip, int(port)


def _net_port(value: int) -> int:
    """A port stored in network byte order in the low 16 bits of a host-order integer."""
    return socket.ntohs(value & 0xFFFF)


def client_is_ours(client: tuple, server: tuple) -> bool:
    """Whether the client end of (client -> server) belongs to a process of this user."""
    if any_local_allowed():
        return True
    want_local = endpoint(*client[:2])
    want_remote = endpoint(*server[:2])
    if want_local is None or want_remote is None:
        return False
    try:
        if sys.platform == "win32":
            return _windows_is_ours(want_local, want_remote)
        if sys.platform == "darwin":
            return _darwin_is_ours(want_local, want_remote)
        raise OwnerLookupError(f"no socket owner lookup on {sys.platform}")
    except OwnerLookupError as exc:
        _log_once(str(exc))
    except Exception as exc:  # noqa: BLE001 - any failure of the lookup refuses, never admits
        _log_once(f"{type(exc).__name__}: {exc}")
    return False


# ---------------------------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------------------------

_AF_INET, _AF_INET6 = 2, 23
_TCP_TABLE_OWNER_PID_CONNECTIONS = 4
_MIB_TCP_STATE_ESTAB = 5
_ERROR_INSUFFICIENT_BUFFER = 122
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_TOKEN_QUERY = 0x0008
_TOKEN_USER = 1

# MIB_TCPROW_OWNER_PID: dwState, dwLocalAddr, dwLocalPort, dwRemoteAddr, dwRemotePort, dwOwningPid.
_ROW4 = struct.Struct("<IIIIII")
# MIB_TCP6ROW_OWNER_PID: ucLocalAddr[16], dwLocalScopeId, dwLocalPort, ucRemoteAddr[16],
# dwRemoteScopeId, dwRemotePort, dwState, dwOwningPid.
_ROW6 = struct.Struct("<16sII16sIIII")
_COUNT = struct.Struct("<I")  # dwNumEntries; rows follow at offset 4


def parse_tcp_table_v4(buf: bytes) -> List[Tuple[int, Endpoint, Endpoint, int]]:
    """(state, local, remote, pid) rows of a MIB_TCPTABLE_OWNER_PID."""
    (count,) = _COUNT.unpack_from(buf, 0)
    if _COUNT.size + count * _ROW4.size > len(buf):
        raise OwnerLookupError("GetExtendedTcpTable returned a truncated IPv4 table")
    rows = []
    for i in range(count):
        state, laddr, lport, raddr, rport, pid = _ROW4.unpack_from(
            buf, _COUNT.size + i * _ROW4.size
        )
        # dwLocalAddr is an in_addr: its bytes in memory are already in network order.
        local = endpoint(struct.pack("<I", laddr), _net_port(lport))
        remote = endpoint(struct.pack("<I", raddr), _net_port(rport))
        if local is not None and remote is not None:
            rows.append((state, local, remote, pid))
    return rows


def parse_tcp_table_v6(buf: bytes) -> List[Tuple[int, Endpoint, Endpoint, int]]:
    """(state, local, remote, pid) rows of a MIB_TCP6TABLE_OWNER_PID."""
    (count,) = _COUNT.unpack_from(buf, 0)
    if _COUNT.size + count * _ROW6.size > len(buf):
        raise OwnerLookupError("GetExtendedTcpTable returned a truncated IPv6 table")
    rows = []
    for i in range(count):
        laddr, _lscope, lport, raddr, _rscope, rport, state, pid = _ROW6.unpack_from(
            buf, _COUNT.size + i * _ROW6.size
        )
        local = endpoint(laddr, _net_port(lport))
        remote = endpoint(raddr, _net_port(rport))
        if local is not None and remote is not None:
            rows.append((state, local, remote, pid))
    return rows


def _windows_table(family: int) -> bytes:
    """Raw GetExtendedTcpTable output for one address family."""
    try:
        iphlpapi = ctypes.WinDLL("iphlpapi", use_last_error = True)  # type: ignore[attr-defined]
        get_table = iphlpapi.GetExtendedTcpTable
    except (AttributeError, OSError) as exc:
        raise OwnerLookupError(f"GetExtendedTcpTable unavailable: {exc}") from None
    get_table.restype = ctypes.c_ulong
    get_table.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.c_ulong,
    ]
    size = ctypes.c_ulong(0)
    for _ in range(8):  # the table can grow between the size query and the read
        buf = ctypes.create_string_buffer(max(size.value, 4))
        rc = get_table(buf, ctypes.byref(size), 0, family, _TCP_TABLE_OWNER_PID_CONNECTIONS, 0)
        if rc == 0:
            return buf.raw[: size.value] if size.value else buf.raw
        if rc != _ERROR_INSUFFICIENT_BUFFER:
            raise OwnerLookupError(f"GetExtendedTcpTable(af={family}) failed with {rc}")
        size = ctypes.c_ulong(size.value + 16 * _ROW6.size)
    raise OwnerLookupError("GetExtendedTcpTable kept growing")


def windows_tcp_rows() -> List[Tuple[int, Endpoint, Endpoint, int]]:
    rows = parse_tcp_table_v4(_windows_table(_AF_INET))
    try:
        v6 = _windows_table(_AF_INET6)
    except OwnerLookupError:
        v6 = None  # a host with IPv6 disabled has no v6 table; v4 still answers
    if v6 is not None:
        rows += parse_tcp_table_v6(v6)
    return rows


class _Win32:
    """Lazily bound advapi32/kernel32 entry points for the token user comparison."""

    def __init__(self) -> None:
        from ctypes import wintypes

        self.k32 = ctypes.WinDLL("kernel32", use_last_error = True)  # type: ignore[attr-defined]
        self.adv = ctypes.WinDLL("advapi32", use_last_error = True)  # type: ignore[attr-defined]
        self.k32.OpenProcess.restype = wintypes.HANDLE
        self.k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.k32.GetCurrentProcess.restype = wintypes.HANDLE
        self.k32.GetCurrentProcess.argtypes = []
        self.k32.CloseHandle.restype = wintypes.BOOL
        self.k32.CloseHandle.argtypes = [wintypes.HANDLE]
        self.adv.OpenProcessToken.restype = wintypes.BOOL
        self.adv.OpenProcessToken.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.HANDLE),
        ]
        self.adv.GetTokenInformation.restype = wintypes.BOOL
        self.adv.GetTokenInformation.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.adv.EqualSid.restype = wintypes.BOOL
        self.adv.EqualSid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.adv.IsValidSid.restype = wintypes.BOOL
        self.adv.IsValidSid.argtypes = [ctypes.c_void_p]
        self.wintypes = wintypes

    def token_user(self, process) -> Optional[ctypes.Array]:
        """TOKEN_USER bytes of a process handle, or None when its token cannot be read."""
        wt = self.wintypes
        token = wt.HANDLE()
        if not self.adv.OpenProcessToken(process, _TOKEN_QUERY, ctypes.byref(token)):
            return None
        try:
            need = wt.DWORD(0)
            self.adv.GetTokenInformation(token, _TOKEN_USER, None, 0, ctypes.byref(need))
            if not need.value:
                return None
            buf = ctypes.create_string_buffer(need.value)
            if not self.adv.GetTokenInformation(token, _TOKEN_USER, buf, need, ctypes.byref(need)):
                return None
            return buf
        finally:
            self.k32.CloseHandle(token)

    @staticmethod
    def sid_of(token_user) -> int:
        # TOKEN_USER starts with SID_AND_ATTRIBUTES { PSID Sid; DWORD Attributes; }.
        return ctypes.c_void_p.from_buffer(token_user).value or 0

    def pid_is_current_user(self, pid: int) -> bool:
        mine = self.token_user(self.k32.GetCurrentProcess())
        if mine is None:
            raise OwnerLookupError("cannot read this process's token user")
        my_sid = self.sid_of(mine)
        if not my_sid or not self.adv.IsValidSid(my_sid):
            raise OwnerLookupError("this process's token user SID is invalid")
        handle = self.k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return False  # another user's (or a protected) process: not ours
        try:
            theirs = self.token_user(handle)
            if theirs is None:
                return False
            their_sid = self.sid_of(theirs)
            return bool(their_sid) and bool(self.adv.EqualSid(my_sid, their_sid))
        finally:
            self.k32.CloseHandle(handle)


def windows_pid_is_current_user(pid: int) -> bool:
    try:
        api = _Win32()
    except (AttributeError, OSError, ImportError) as exc:
        raise OwnerLookupError(f"advapi32/kernel32 unavailable: {exc}") from None
    return api.pid_is_current_user(pid)


def _windows_is_ours(local: Endpoint, remote: Endpoint) -> bool:
    pids = {
        pid
        for state, l, r, pid in windows_tcp_rows()
        if state == _MIB_TCP_STATE_ESTAB and l == local and r == remote
    }
    if len(pids) != 1:
        return False
    return windows_pid_is_current_user(pids.pop())


# ---------------------------------------------------------------------------------------------
# macOS (xnu <sys/proc_info.h>, <libproc.h>)
# ---------------------------------------------------------------------------------------------

_PROC_ALL_PIDS = 1
_PROC_PIDLISTFDS = 1
_PROC_PIDTBSDINFO = 3
_PROC_PIDTBSDINFO_SIZE = 136
_PROC_PIDFDSOCKETINFO = 3
_PROX_FDTYPE_SOCKET = 2
_SOCKINFO_TCP = 2
_TSI_S_ESTABLISHED = 4
_INI_IPV4, _INI_IPV6 = 0x1, 0x2
_PBI_UID_OFFSET = 20  # pbi_flags, pbi_status, pbi_xstatus, pbi_pid, pbi_ppid, then pbi_uid


class ProcFdInfo(ctypes.Structure):
    _fields_ = [("proc_fd", ctypes.c_int32), ("proc_fdtype", ctypes.c_uint32)]


class ProcFileInfo(ctypes.Structure):
    _fields_ = [
        ("fi_openflags", ctypes.c_uint32),
        ("fi_status", ctypes.c_uint32),
        ("fi_offset", ctypes.c_int64),
        ("fi_type", ctypes.c_int32),
        ("fi_guardflags", ctypes.c_uint32),
    ]


class VinfoStat(ctypes.Structure):
    _fields_ = [
        ("vst_dev", ctypes.c_uint32),
        ("vst_mode", ctypes.c_uint16),
        ("vst_nlink", ctypes.c_uint16),
        ("vst_ino", ctypes.c_uint64),
        ("vst_uid", ctypes.c_uint32),
        ("vst_gid", ctypes.c_uint32),
        ("vst_times", ctypes.c_int64 * 8),  # atime, mtime, ctime, birthtime (+ nsec each)
        ("vst_size", ctypes.c_int64),
        ("vst_blocks", ctypes.c_int64),
        ("vst_blksize", ctypes.c_int32),
        ("vst_flags", ctypes.c_uint32),
        ("vst_gen", ctypes.c_uint32),
        ("vst_rdev", ctypes.c_uint32),
        ("vst_qspare", ctypes.c_int64 * 2),
    ]


class SockbufInfo(ctypes.Structure):
    _fields_ = [
        ("sbi_cc", ctypes.c_uint32),
        ("sbi_hiwat", ctypes.c_uint32),
        ("sbi_mbcnt", ctypes.c_uint32),
        ("sbi_mbmax", ctypes.c_uint32),
        ("sbi_lowat", ctypes.c_uint32),
        ("sbi_flags", ctypes.c_short),
        ("sbi_timeo", ctypes.c_short),
    ]


class InSockinfoV6(ctypes.Structure):
    _fields_ = [
        ("in6_hlim", ctypes.c_uint8),
        ("in6_cksum", ctypes.c_int),
        ("in6_ifindex", ctypes.c_ushort),
        ("in6_hops", ctypes.c_short),
    ]


class InSockinfo(ctypes.Structure):
    _fields_ = [
        ("insi_fport", ctypes.c_int),  # network byte order
        ("insi_lport", ctypes.c_int),  # network byte order
        ("insi_gencnt", ctypes.c_uint64),
        ("insi_flags", ctypes.c_uint32),
        ("insi_flow", ctypes.c_uint32),
        ("insi_vflag", ctypes.c_uint8),
        ("insi_ip_ttl", ctypes.c_uint8),
        ("rfu_1", ctypes.c_uint32),
        # in4in6_addr (12 pad bytes + in_addr) or in6_addr: 16 bytes in network order either way.
        ("insi_faddr", ctypes.c_uint8 * 16),
        ("insi_laddr", ctypes.c_uint8 * 16),
        ("insi_v4_tos", ctypes.c_ubyte),
        ("insi_v6", InSockinfoV6),
    ]


class TcpSockinfo(ctypes.Structure):
    _fields_ = [
        ("tcpsi_ini", InSockinfo),
        ("tcpsi_state", ctypes.c_int),
        ("tcpsi_timer", ctypes.c_int * 4),
        ("tcpsi_mss", ctypes.c_int),
        ("tcpsi_flags", ctypes.c_uint32),
        ("rfu_1", ctypes.c_uint32),
        ("tcpsi_tp", ctypes.c_uint64),
    ]


class SoiProto(ctypes.Union):
    # The largest member is un_sockinfo: two 64-bit pointers + two 255-byte address buffers.
    _fields_ = [("pri_tcp", TcpSockinfo), ("_un", ctypes.c_uint64 * 66)]


class SocketInfo(ctypes.Structure):
    _fields_ = [
        ("soi_stat", VinfoStat),
        ("soi_so", ctypes.c_uint64),
        ("soi_pcb", ctypes.c_uint64),
        ("soi_type", ctypes.c_int),
        ("soi_protocol", ctypes.c_int),
        ("soi_family", ctypes.c_int),
        ("soi_options", ctypes.c_short),
        ("soi_linger", ctypes.c_short),
        ("soi_state", ctypes.c_short),
        ("soi_qlen", ctypes.c_short),
        ("soi_incqlen", ctypes.c_short),
        ("soi_qlimit", ctypes.c_short),
        ("soi_timeo", ctypes.c_short),
        ("soi_error", ctypes.c_ushort),
        ("soi_oobmark", ctypes.c_uint32),
        ("soi_rcv", SockbufInfo),
        ("soi_snd", SockbufInfo),
        ("soi_kind", ctypes.c_int),
        ("rfu_1", ctypes.c_uint32),
        ("soi_proto", SoiProto),
    ]


class SocketFdInfo(ctypes.Structure):
    _fields_ = [("pfi", ProcFileInfo), ("psi", SocketInfo)]


SOCKET_FDINFO_SIZE = 792  # sizeof(struct socket_fdinfo) on every 64-bit macOS


def parse_socket_fdinfo(buf: bytes) -> Optional[Tuple[int, Endpoint, Endpoint]]:
    """(tcp state, local, remote) of a PROC_PIDFDSOCKETINFO answer, or None for a non-TCP socket."""
    if len(buf) < ctypes.sizeof(SocketFdInfo):
        return None
    info = SocketFdInfo.from_buffer_copy(buf[: ctypes.sizeof(SocketFdInfo)])
    if info.psi.soi_kind != _SOCKINFO_TCP:
        return None
    tcp = info.psi.soi_proto.pri_tcp
    ini = tcp.tcpsi_ini
    laddr, faddr = bytes(ini.insi_laddr), bytes(ini.insi_faddr)
    if ini.insi_vflag & _INI_IPV4:
        laddr, faddr = laddr[12:], faddr[12:]
    elif not ini.insi_vflag & _INI_IPV6:
        return None
    local = endpoint(laddr, _net_port(ini.insi_lport))
    remote = endpoint(faddr, _net_port(ini.insi_fport))
    if local is None or remote is None:
        return None
    return tcp.tcpsi_state, local, remote


class _LibProc:
    def __init__(self) -> None:
        if ctypes.sizeof(SocketFdInfo) != SOCKET_FDINFO_SIZE:
            raise OwnerLookupError(
                f"socket_fdinfo layout is {ctypes.sizeof(SocketFdInfo)} bytes, not {SOCKET_FDINFO_SIZE}"
            )
        try:
            lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno = True)
        except OSError:
            try:
                lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno = True)
            except OSError as exc:
                raise OwnerLookupError(f"libproc unavailable: {exc}") from None
        try:
            self.listpids = lib.proc_listpids
            self.pidinfo = lib.proc_pidinfo
            self.pidfdinfo = lib.proc_pidfdinfo
        except AttributeError as exc:
            raise OwnerLookupError(f"libproc unavailable: {exc}") from None
        self.listpids.restype = ctypes.c_int
        self.listpids.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int]
        self.pidinfo.restype = ctypes.c_int
        self.pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self.pidfdinfo.restype = ctypes.c_int
        self.pidfdinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]

    def pids(self) -> List[int]:
        need = self.listpids(_PROC_ALL_PIDS, 0, None, 0)
        if need <= 0:
            raise OwnerLookupError(f"proc_listpids failed (errno {ctypes.get_errno()})")
        n = need // ctypes.sizeof(ctypes.c_int) + 64
        arr = (ctypes.c_int * n)()
        got = self.listpids(_PROC_ALL_PIDS, 0, arr, ctypes.sizeof(arr))
        if got <= 0:
            raise OwnerLookupError(f"proc_listpids failed (errno {ctypes.get_errno()})")
        return [p for p in arr[: got // ctypes.sizeof(ctypes.c_int)] if p > 0]

    def socket_fds(self, pid: int) -> List[int]:
        """Socket fds of a pid; [] for a process this user cannot inspect (or that has exited)."""
        need = self.pidinfo(pid, _PROC_PIDLISTFDS, 0, None, 0)
        if need <= 0:
            return []
        n = need // ctypes.sizeof(ProcFdInfo) + 32
        arr = (ProcFdInfo * n)()
        got = self.pidinfo(pid, _PROC_PIDLISTFDS, 0, arr, ctypes.sizeof(arr))
        if got <= 0:
            return []
        return [
            fd.proc_fd
            for fd in arr[: got // ctypes.sizeof(ProcFdInfo)]
            if fd.proc_fdtype == _PROX_FDTYPE_SOCKET
        ]

    def socket_info(self, pid: int, fd: int) -> Optional[bytes]:
        buf = ctypes.create_string_buffer(SOCKET_FDINFO_SIZE + 256)
        got = self.pidfdinfo(pid, fd, _PROC_PIDFDSOCKETINFO, buf, ctypes.sizeof(buf))
        if got < SOCKET_FDINFO_SIZE:
            return None
        return buf.raw[:got]

    def uid(self, pid: int) -> Optional[int]:
        buf = ctypes.create_string_buffer(_PROC_PIDTBSDINFO_SIZE)
        got = self.pidinfo(pid, _PROC_PIDTBSDINFO, 0, buf, _PROC_PIDTBSDINFO_SIZE)
        if got != _PROC_PIDTBSDINFO_SIZE:
            return None
        return struct.unpack_from("<I", buf.raw, _PBI_UID_OFFSET)[0]


def darwin_owner_uids(
    local: Endpoint,
    remote: Endpoint,
    pids: Callable[[], Iterable[int]],
    socket_fds: Callable[[int], Iterable[int]],
    socket_info: Callable[[int, int], Optional[bytes]],
    uid_of: Callable[[int], Optional[int]],
) -> Optional[set]:
    """uids of the visible processes holding the exact ESTABLISHED tuple; None if one is unreadable."""
    uids = set()
    for pid in pids():
        for fd in socket_fds(pid):
            raw = socket_info(pid, fd)
            parsed = parse_socket_fdinfo(raw) if raw else None
            if parsed is None:
                continue
            state, l, r = parsed
            if state == _TSI_S_ESTABLISHED and l == local and r == remote:
                uid = uid_of(pid)
                if uid is None:
                    return None
                uids.add(uid)
                break
    return uids


def _darwin_is_ours(local: Endpoint, remote: Endpoint) -> bool:
    lib = _LibProc()
    uids = darwin_owner_uids(local, remote, lib.pids, lib.socket_fds, lib.socket_info, lib.uid)
    return uids == {os.getuid()}
