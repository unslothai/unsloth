# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whether this host is in mainland China, decided as install.sh / install.ps1 decide it: no network call."""

from __future__ import annotations

import functools
import itertools
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
_WINDOWS_INTERFACES = r"SYSTEM\CurrentControlSet\Services\Tcpip\Parameters\Interfaces"


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


def _adapter_resolvers(winreg, interface) -> list[str]:
    # A static list replaces the DHCP one, as the resolver does.
    for value in ("NameServer", "DhcpNameServer"):
        try:
            configured = str(winreg.QueryValueEx(interface, value)[0]).strip()
        except OSError:
            continue
        if configured:
            return re.split(r"[\s,]+", configured)
    return []


def _windows_resolvers() -> list[str]:
    # Unlike the installer's .NET query, the registry also lists adapters that are down.
    import winreg

    servers: list[str] = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _WINDOWS_INTERFACES) as interfaces:
            for index in itertools.count():
                try:
                    name = winreg.EnumKey(interfaces, index)
                except OSError:
                    break
                try:
                    with winreg.OpenKey(interfaces, name) as interface:
                        servers += _adapter_resolvers(winreg, interface)
                except OSError:
                    continue
    except OSError:
        pass
    return servers


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
