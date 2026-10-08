# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Host RAM this process can use: psutil / meminfo capped by the tightest enforcing cgroup level.

psutil reads the whole machine, so under a container or systemd MemoryMax it admits work the cgroup
OOM-kills. Torch-free: the polled /api/system route calls it."""

from __future__ import annotations

import os
from typing import Optional

CGROUP_ROOT = "/sys/fs/cgroup"
PROC_SELF_CGROUP = "/proc/self/cgroup"
_MIB = 1024 * 1024
# cgroup v1 spells "unlimited" as a near-2^63 sentinel.
_UNLIMITED_FLOOR = 1 << 60


def _first_line(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding = "utf-8") as f:
            return f.readline().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _integer(raw: Optional[str], *, limit: bool = False) -> Optional[int]:
    if not raw or raw == "max":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value < 0 or (limit and value >= _UNLIMITED_FLOOR):
        return None
    return value


def _stat_integer(path: str, *keys: str) -> int:
    """The first requested byte counter present in a ``memory.stat`` file, else 0."""
    values: dict[str, int] = {}
    try:
        with open(path, "r", encoding = "utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[0] in keys:
                    value = _integer(parts[1])
                    if value is not None:
                        values[parts[0]] = value
    except (OSError, UnicodeDecodeError):
        return 0
    return next((values[key] for key in keys if key in values), 0)


def _directories(root: str, relative: Optional[str]) -> list[str]:
    """The process's cgroup directory under ``root`` and every ancestor up to ``root``."""
    root = os.path.abspath(root)
    current = os.path.normpath(os.path.join(root, (relative or "/").lstrip("/")))
    try:
        if os.path.commonpath((root, current)) != root:
            return [root]
    except ValueError:
        return [root]
    out = []
    while True:
        out.append(current)
        if current == root:
            return out
        current = os.path.dirname(current)


def cgroup_memory_budgets(
    cgroup_root: Optional[str] = None, proc_self_cgroup: Optional[str] = None
) -> list[tuple[int, int]]:
    """``(remaining, limit)`` bytes per enforcing limit on this cgroup and its ancestors (an ancestor
    can bind through sibling usage). Usage excludes ``inactive_file``; empty = no finite limit."""
    cgroup_root = CGROUP_ROOT if cgroup_root is None else cgroup_root
    proc_self_cgroup = PROC_SELF_CGROUP if proc_self_cgroup is None else proc_self_cgroup
    try:
        with open(proc_self_cgroup, "r", encoding = "utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except (OSError, UnicodeDecodeError):
        lines = []

    budgets: list[tuple[int, int]] = []
    v2_relative = next((line[3:] for line in lines if line.startswith("0::")), None)
    for directory in _directories(cgroup_root, v2_relative):
        limit = _integer(_first_line(os.path.join(directory, "memory.max")), limit = True)
        if limit is None:
            continue
        used = _integer(_first_line(os.path.join(directory, "memory.current")))
        if used is not None:
            used = max(
                0, used - _stat_integer(os.path.join(directory, "memory.stat"), "inactive_file")
            )
        budgets.append((limit if used is None else limit - used, limit))

    v1_root = os.path.join(cgroup_root, "memory")
    v1_relative = None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) == 3 and "memory" in parts[1].split(","):
            v1_relative = parts[2]
            break
    for directory in _directories(v1_root, v1_relative):
        limit = _integer(_first_line(os.path.join(directory, "memory.limit_in_bytes")), limit = True)
        if limit is None:
            continue
        used = _integer(_first_line(os.path.join(directory, "memory.usage_in_bytes")))
        if used is not None:
            # Hierarchical v1 usage pairs with total_inactive_file.
            used = max(
                0,
                used
                - _stat_integer(
                    os.path.join(directory, "memory.stat"), "total_inactive_file", "inactive_file"
                ),
            )
        budgets.append((limit if used is None else limit - used, limit))
    return budgets


def cgroup_headroom_mib(budgets: Optional[list[tuple[int, int]]] = None) -> Optional[int]:
    """What an enforcing cgroup still lets this process charge (the tightest remainder), else None."""
    budgets = cgroup_memory_budgets() if budgets is None else budgets
    if not budgets:
        return None
    return max(min(remaining for remaining, _limit in budgets), 0) // _MIB


def cgroup_limit_mib(budgets: Optional[list[tuple[int, int]]] = None) -> Optional[int]:
    """The tightest enforcing cgroup limit (not the headroom, which shrinks as it fills), else None."""
    budgets = cgroup_memory_budgets() if budgets is None else budgets
    if not budgets:
        return None
    return min(limit for _remaining, limit in budgets) // _MIB


def _meminfo_mib(key: str) -> Optional[int]:
    try:
        with open("/proc/meminfo", encoding = "utf-8") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) // 1024  # kB -> MiB
    except (OSError, ValueError, IndexError):
        pass
    return None


def system_available_mib() -> Optional[int]:
    """Host-wide available RAM in MiB (psutil, then /proc/meminfo), NOT cgroup-capped, or None."""
    try:
        import psutil
        return int(psutil.virtual_memory().available // _MIB)
    except Exception:  # noqa: BLE001 - fall back to the kernel file
        pass
    return _meminfo_mib("MemAvailable")


def system_total_mib() -> Optional[int]:
    """Host-wide total RAM in MiB (psutil, then /proc/meminfo), NOT cgroup-capped, or None."""
    try:
        import psutil
        return int(psutil.virtual_memory().total // _MIB)
    except Exception:  # noqa: BLE001 - fall back to the kernel file
        pass
    return _meminfo_mib("MemTotal")


def usable_mib(available_mib: Optional[int], headroom_mib: Optional[int]) -> Optional[int]:
    """min() of the two readings; either may be None."""
    if available_mib is None:
        return headroom_mib
    if headroom_mib is None:
        return int(available_mib)
    return min(int(available_mib), int(headroom_mib))


def usable_host_ram_mib() -> Optional[int]:
    """Host RAM this process can still allocate, in MiB: min(system available, cgroup headroom)."""
    return usable_mib(system_available_mib(), cgroup_headroom_mib())


def host_ram_capacity_mib() -> Optional[int]:
    """Host RAM this process may ever charge, in MiB: min(system total, cgroup limit)."""
    return usable_mib(system_total_mib(), cgroup_limit_mib())
