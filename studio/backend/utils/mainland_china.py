# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whether this host is in mainland China, decided as install.sh / install.ps1 decide it: no network call."""

from __future__ import annotations

import ctypes
import functools
import os
import re
import sys
from pathlib import Path

_ZONES = (
    "Asia/Shanghai",
    "Asia/Chongqing",
    "Asia/Chungking",
    "Asia/Harbin",
    "Asia/Urumqi",
    "Asia/Kashgar",
    "PRC",
)
_WINDOWS_ZONE = "China Standard Time"
# Mainland public DNS and cloud resolvers.
_MAINLAND_RESOLVER = re.compile(
    r"223\.5\.5\.5|223\.6\.6\.6|119\.29\.29\.29|114\.114\.11[45]\.11[0459]"
    r"|182\.254\.116\.116|119\.28\.28\.28|180\.76\.76\.76|1\.2\.4\.8|210\.2\.4\.8"
    r"|100\.100\.2\.13[68]|183\.60\.8[23]\.(19|98)"
)
_NAMESERVER = re.compile(r"^[ \t]*nameserver[ \t]+(\S+)[ \t]*$", re.MULTILINE)
_RESOLV_CONFS = ("/etc/resolv.conf", "/run/systemd/resolve/resolv.conf")


def _time_zone() -> str:
    zone = os.environ.get("TZ", "").lstrip(":")
    if zone:
        return zone
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\TimeZoneInformation"
            ) as key:
                return str(winreg.QueryValueEx(key, "TimeZoneKeyName")[0])
        except OSError:
            return ""
    try:
        zone = Path("/etc/timezone").read_text(encoding = "utf-8").strip()
    except (OSError, UnicodeDecodeError):
        zone = ""
    if zone:
        return zone
    try:
        return os.readlink("/etc/localtime")
    except OSError:
        return ""


class _Address(ctypes.Structure):
    _fields_ = [("sockaddr", ctypes.POINTER(ctypes.c_ubyte)), ("length", ctypes.c_int)]


class _Server(ctypes.Structure):
    pass


_Server._fields_ = [
    ("header", ctypes.c_ulonglong),
    ("next", ctypes.POINTER(_Server)),
    ("address", _Address),
]


# The leading fields of IP_ADAPTER_ADDRESSES, through OperStatus.
class _Adapter(ctypes.Structure):
    pass


_Adapter._fields_ = [
    ("header", ctypes.c_ulonglong),
    ("next", ctypes.POINTER(_Adapter)),
    ("name", ctypes.c_char_p),
    ("unicast", ctypes.c_void_p),
    ("anycast", ctypes.c_void_p),
    ("multicast", ctypes.c_void_p),
    ("dns", ctypes.POINTER(_Server)),
    ("suffix", ctypes.c_wchar_p),
    ("description", ctypes.c_wchar_p),
    ("friendly_name", ctypes.c_wchar_p),
    ("physical_address", ctypes.c_ubyte * 8),
    ("physical_address_length", ctypes.c_ulong),
    ("flags", ctypes.c_ulong),
    ("mtu", ctypes.c_ulong),
    ("if_type", ctypes.c_ulong),
    ("oper_status", ctypes.c_int),
]
_OPER_STATUS_UP = 1
_AF_INET = 2


def _up_adapter_resolvers(adapter) -> list[str]:
    """IPv4 DNS servers of the adapters that are up, from a GetAdaptersAddresses list."""
    servers: list[str] = []
    while adapter:
        if adapter.contents.oper_status == _OPER_STATUS_UP:
            server = adapter.contents.dns
            while server:
                raw = server.contents.address
                if raw.length >= 8 and raw.sockaddr[0] | raw.sockaddr[1] << 8 == _AF_INET:
                    servers.append(".".join(str(raw.sockaddr[i]) for i in range(4, 8)))
                server = server.contents.next
        adapter = adapter.contents.next
    return servers


def _windows_resolvers() -> list[str]:
    """What the installer's .NET query lists: DNS servers of the adapters that are up."""
    skip_unicast_anycast_multicast, buffer_overflow = 0x7, 111
    try:
        get_adapters = ctypes.WinDLL("iphlpapi").GetAdaptersAddresses
        size = ctypes.c_ulong(16 * 1024)
        for _ in range(3):
            buffer = ctypes.create_string_buffer(size.value)
            result = get_adapters(
                0, skip_unicast_anycast_multicast, None, buffer, ctypes.byref(size)
            )
            if result != buffer_overflow:
                break
    except OSError:
        return []
    if result != 0:
        return []
    return _up_adapter_resolvers(ctypes.cast(buffer, ctypes.POINTER(_Adapter)))


def _resolvers() -> list[str]:
    if sys.platform == "win32":
        return _windows_resolvers()
    servers: list[str] = []
    for path in _RESOLV_CONFS:
        try:
            servers += _NAMESERVER.findall(Path(path).read_text(encoding = "utf-8", errors = "replace"))
        except OSError:
            continue
    return servers


@functools.lru_cache(maxsize = 1)
def in_mainland_china() -> bool:
    zone = _time_zone()
    if zone == _WINDOWS_ZONE or any(zone == z or zone.endswith("/" + z) for z in _ZONES):
        return True
    return any(_MAINLAND_RESOLVER.fullmatch(server) for server in _resolvers())


def china_mirrors_enabled() -> bool:
    """Region detection, overridden by UNSLOTH_MIRROR_FALLBACK: 0 turns it off and 1 on, as for the installer."""
    flag = os.environ.get("UNSLOTH_MIRROR_FALLBACK", "").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    if flag in ("1", "true", "yes", "on"):
        return True
    return in_mainland_china()
