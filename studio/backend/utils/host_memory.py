# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Host RAM this process can actually use, for every place Studio sizes host memory.

``psutil.virtual_memory()`` and ``/proc/meminfo`` describe the whole machine. Under a container or
a systemd ``MemoryMax`` the binding number is the cgroup's: what is left of ``memory.max`` (v2) or
``memory.limit_in_bytes`` (v1), at the tightest level of the process's cgroup and its ancestors.
Sizing from the host reading alone admits work the cgroup then OOM-kills.

- ``usable_host_ram_mib``: min(system available, cgroup headroom). The number to size against.
- ``host_ram_capacity_mib``: min(system total, cgroup limit). What usable RAM can ever become.

With no enforcing limit both return the system reading unchanged. Torch-free and cheap, so the
polled ``/api/system`` route can call it.
"""

from __future__ import annotations

import os
from typing import Optional

CGROUP_ROOT = "/sys/fs/cgroup"
PROC_SELF_CGROUP = "/proc/self/cgroup"
_MIB = 1024 * 1024
# cgroup v1 spells "unlimited" as a near-2^63 sentinel (PAGE_COUNTER_MAX rounded to the page size).
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
    """``(remaining, limit)`` bytes for every enforcing memory limit on this process.

    Walks the process's cgroup plus its ancestors and pairs each limit with that same directory's
    usage: an ancestor slice can be the binding limit and includes sibling usage a leaf does not
    see. Usage is priced as the working set (``inactive_file`` is reclaimable under pressure).
    cgroup v2 and the legacy v1 memory controller. Unlimited, unreadable or malformed levels are
    skipped, so an empty list means "no finite readable limit" and callers keep the host reading.
    """
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
            # memory.current includes file-backed cache. Inactive file pages are reclaimable
            # under pressure, so price against the working set rather than charging cache twice.
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
            # v1 usage is hierarchical when use_hierarchy is enabled, so its matching counter is
            # total_inactive_file. Fall back to the local counter for non-hierarchical controllers.
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
    """The capacity an enforcing cgroup allows (the tightest limit), else None.

    Not interchangeable with the headroom, which shrinks as the cgroup fills."""
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
    """The combination rule: the smaller of the host reading and the cgroup headroom.

    Either side may be unreadable (None); with no limit the host reading passes through untouched."""
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
