# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whether this host is in mainland China, decided as install.sh / install.ps1 decide it: no network call."""

from __future__ import annotations

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
_RESOLVER = re.compile(
    r"^[ \t]*nameserver[ \t]+(223\.5\.5\.5|223\.6\.6\.6|119\.29\.29\.29|114\.114\.11[45]\.11[0459]"
    r"|182\.254\.116\.116|119\.28\.28\.28|180\.76\.76\.76|1\.2\.4\.8|210\.2\.4\.8"
    r"|100\.100\.2\.13[68]|183\.60\.8[23]\.(19|98))[ \t]*$",
    re.MULTILINE,
)
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


def _mainland_resolver() -> bool:
    for path in _RESOLV_CONFS:
        try:
            if _RESOLVER.search(Path(path).read_text(encoding = "utf-8", errors = "replace")):
                return True
        except OSError:
            continue
    return False


@functools.lru_cache(maxsize = 1)
def in_mainland_china() -> bool:
    zone = _time_zone()
    if zone == _WINDOWS_ZONE or any(zone == z or zone.endswith("/" + z) for z in _ZONES):
        return True
    return sys.platform != "win32" and _mainland_resolver()


def china_mirrors_enabled() -> bool:
    """Region detection, overridden by UNSLOTH_MIRROR_FALLBACK: 0 turns it off and 1 on, as for the installer."""
    flag = os.environ.get("UNSLOTH_MIRROR_FALLBACK", "").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    if flag in ("1", "true", "yes", "on"):
        return True
    return in_mainland_china()
