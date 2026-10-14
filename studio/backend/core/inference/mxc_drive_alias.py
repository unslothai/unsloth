# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Drive-letter aliases for the session workdir of an isolated cmd Terminal launch.

Inside an MXC AppContainer, Git for Windows cannot resolve its working directory: the drive-letter form
of GetFinalPathNameByHandleW is denied and the GetLongPathNameW fallback has to list every ancestor.
From a drive root there is nothing above to resolve, so a launch runs from ``X:\\`` mapped onto the
canonical workdir with DefineDosDeviceW, the same mapping ``subst`` makes. The mapping only changes how
the workdir is spelled; MXC still grants the canonical folder and nothing else.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import threading
import uuid

logger = logging.getLogger(__name__)

DRIVE_ALIAS_ENV = "UNSLOTH_MXC_DRIVE_ALIAS"
LIMITATION_UNAVAILABLE = "terminal_workdir_drive_alias_unavailable"
LOCK_TIMEOUT_SECONDS = 10
LETTERS = "ZYXWVUTSRQPONMLKJIHG"
DDD_REMOVE_DEFINITION = 0x2
DDD_EXACT_MATCH_ON_REMOVE = 0x4

_lock = threading.Lock()
# normcase(target) -> [letter, target, lease count]; the leases this process holds.
_active: dict[str, list] = {}
# (letter, target) -> [letter, target, lease count]: held mappings another definition now covers. Still removed, by
# exact match, when their last lease ends, so ours never resurfaces once the other one goes.
_shadowed: dict[tuple[str, str], list] = {}
_host = None


def _on_windows() -> bool:
    return os.name == "nt"


def enabled() -> bool:
    return os.environ.get(DRIVE_ALIAS_ENV, "").strip() != "0"


class _Win32:
    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
        self._kernel32.DefineDosDeviceW.argtypes = [
            wintypes.DWORD,
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
        ]
        self._kernel32.DefineDosDeviceW.restype = wintypes.BOOL
        self._kernel32.QueryDosDeviceW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            wintypes.DWORD,
        ]
        self._kernel32.QueryDosDeviceW.restype = wintypes.DWORD
        self._kernel32.GetLogicalDrives.restype = wintypes.DWORD

    def logical_drives(self) -> int:
        return int(self._kernel32.GetLogicalDrives())

    def definitions(self, letter: str) -> list[str]:
        """Every definition of the letter, the one in effect first."""
        buffer = self._ctypes.create_unicode_buffer(32768)
        size = self._kernel32.QueryDosDeviceW(f"{letter}:", buffer, len(buffer))
        if not size:
            return []
        return [item for item in buffer[:size].split("\0") if item]

    def query(self, letter: str) -> str | None:
        found = self.definitions(letter)
        return found[0] if found else None

    def network_letters(self) -> set[str]:
        """Persistent network drives, which stay undefined until they reconnect."""
        import winreg

        letters = set()
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Network") as key:
                index = 0
                while True:
                    try:
                        letters.add(winreg.EnumKey(key, index).upper())
                    except OSError:
                        return letters
                    index += 1
        except OSError:
            return letters

    def define(self, letter: str, target: str) -> bool:
        return bool(self._kernel32.DefineDosDeviceW(0, f"{letter}:", target))

    def remove(self, letter: str, target: str) -> bool:
        flags = DDD_REMOVE_DEFINITION | DDD_EXACT_MATCH_ON_REMOVE
        return bool(self._kernel32.DefineDosDeviceW(flags, f"{letter}:", target))

    def process_alive(self, pid: int, create_time: float) -> bool:
        import psutil
        try:
            return abs(psutil.Process(pid).create_time() - create_time) < 1e-3
        except psutil.NoSuchProcess:
            return False
        except psutil.Error:
            return True  # unknown: never reclaim another process's mapping

    def own_identity(self) -> tuple[int, float]:
        import psutil
        return os.getpid(), psutil.Process(os.getpid()).create_time()


def _get_host():
    global _host
    if _host is None:
        _host = _Win32()
    return _host


def _expected(target: str) -> str:
    return "\\??\\" + target


def record_path() -> Path:
    from .mxc_runtime import _studio_root
    return Path(os.path.abspath(_studio_root() / "mxc-runtime" / "drive-aliases.json"))


def _load_record() -> dict[str, dict]:
    try:
        data = json.loads(record_path().read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return {}
    aliases = data.get("aliases") if isinstance(data, dict) else None
    if not isinstance(aliases, dict):
        return {}
    return {
        str(k): v
        for k, v in aliases.items()
        if isinstance(v, dict) and isinstance(v.get("target"), str)
    }


def _save_record(aliases: dict[str, dict]) -> bool:
    path = record_path()
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps({"aliases": aliases}, indent = 2, sort_keys = True), encoding = "utf-8"
        )
        os.replace(temporary, path)
        return True
    except OSError as exc:
        logger.info("Could not write the MXC drive alias record: %s", exc)
        return False
    finally:
        try:
            temporary.unlink(missing_ok = True)
        except OSError as exc:
            # Raising here would skip the caller's removal of the mapping it just made.
            logger.info("Could not remove %s: %s", temporary, exc)


@contextmanager
def _transaction():
    """The alias record, locked across Studio processes; yields None when it cannot be locked."""
    from filelock import FileLock, Timeout

    path = record_path()
    try:
        path.parent.mkdir(parents = True, exist_ok = True)
        lock = FileLock(str(path.with_suffix(".lock")), timeout = LOCK_TIMEOUT_SECONDS)
        lock.acquire()
    except (Timeout, OSError) as exc:
        logger.info("Could not lock the MXC drive alias record: %s", exc)
        yield None
        return
    try:
        yield _load_record()
    finally:
        lock.release()


def _reclaim(host, aliases: dict[str, dict], own_pid: int) -> None:
    """Drop records whose owner is gone, removing the mapping only while it still names that target."""
    for letter, entry in list(aliases.items()):
        pid, created = entry.get("pid"), entry.get("pid_create_time")
        if not isinstance(pid, int) or not isinstance(created, (int, float)):
            continue
        if pid == own_pid and any(active[0] == letter for active in _active.values()):
            continue
        if pid != own_pid and host.process_alive(pid, created):
            continue
        target = entry["target"]
        if _expected(target) in host.definitions(letter) and not host.remove(letter, target):
            continue  # still defined, even under another definition, and not removable: keep the record
        aliases.pop(letter)


class AliasLease:
    def __init__(self, letter: str, target: str) -> None:
        self.letter = letter
        self.target = target
        self.root = f"{letter}:\\"
        self._released = False

    def release(self) -> None:
        with _lock:
            if self._released:
                return
            self._released = True
            key = os.path.normcase(self.target)
            entry = _active.get(key)
            if entry is not None and entry[0] == self.letter:
                table, slot = _active, key
            else:
                table, slot = _shadowed, (self.letter, key)
                entry = _shadowed.get(slot)
                if entry is None:
                    return
            entry[2] -= 1
            if entry[2] > 0:
                return
            table.pop(slot)
            host = _get_host()
            if not host.remove(self.letter, self.target) and _expected(
                self.target
            ) in host.definitions(self.letter):
                return  # still mapped: keep the record so a later acquire can reclaim it
            with _transaction() as aliases:
                if aliases is not None and self.letter in aliases:
                    aliases.pop(self.letter)
                    _save_record(aliases)


def acquire(workdir: str) -> AliasLease | None:
    """A drive letter naming ``workdir`` for one launch, or None when Studio cannot map one safely."""
    if not _on_windows() or not enabled():
        return None
    target = os.path.abspath(workdir)
    key = os.path.normcase(target)
    try:
        host = _get_host()
        with _lock:
            entry = _active.get(key)
            if entry is not None:
                if host.query(entry[0]) == _expected(entry[1]):
                    entry[2] += 1
                    return AliasLease(entry[0], entry[1])
                # Covered by another definition: never touch that one; ours is removed when its leases end.
                _shadowed[(entry[0], key)] = _active.pop(key)
            with _transaction() as aliases:
                if aliases is None:
                    return None
                own_pid, own_created = host.own_identity()
                _reclaim(host, aliases, own_pid)
                drives = host.logical_drives()
                network = host.network_letters()
                for letter in LETTERS:
                    if (
                        drives & (1 << (ord(letter) - ord("A")))
                        or letter in aliases
                        or letter in network
                    ):
                        continue
                    if host.query(letter) is not None:
                        continue
                    if not host.define(letter, target):
                        continue
                    if host.query(letter) != _expected(target):
                        host.remove(letter, target)
                        continue
                    aliases[letter] = {
                        "target": target,
                        "pid": own_pid,
                        "pid_create_time": own_created,
                    }
                    if not _save_record(aliases):
                        # Without a record a crash would strand the letter; do not map at all.
                        host.remove(letter, target)
                        return None
                    _active[key] = [letter, target, 1]
                    return AliasLease(letter, target)
        logger.info("No free drive letter to alias the MXC workdir %s", target)
        return None
    except Exception as exc:  # noqa: BLE001 - an alias is an optimisation for git, never a launch failure
        logger.info("Could not alias the MXC workdir %s: %s", target, exc)
        return None
