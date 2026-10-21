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


# (inactive, active, mapped, dirty, writeback); v1 total_* pairs with usage_in_bytes, per-level is the fallback.
_V2_FILE_KEYS = ("inactive_file", "active_file", "file_mapped", "file_dirty", "file_writeback")
_V1_TOTAL_FILE_KEYS = (
    "total_inactive_file",
    "total_active_file",
    "total_mapped_file",
    "total_dirty",
    "total_writeback",
)
_V1_LOCAL_FILE_KEYS = ("inactive_file", "active_file", "mapped_file", "dirty", "writeback")


def _reclaimable_file_bytes(path: str, *, v1: bool) -> int:
    """Droppable page cache in bytes: inactive_file + (active_file - mapped - dirty - writeback, conservative).

    shmem/tmpfs is on the anon LRU, never counted. Not crediting active_file counted twice-read models as used."""
    values: dict[str, int] = {}
    try:
        with open(path, "r", encoding = "utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    value = _integer(parts[1])
                    if value is not None:
                        values[parts[0]] = value
    except (OSError, UnicodeDecodeError):
        return 0
    if not v1:
        keys = _V2_FILE_KEYS
    elif _V1_TOTAL_FILE_KEYS[0] in values:
        keys = _V1_TOTAL_FILE_KEYS
    else:
        keys = _V1_LOCAL_FILE_KEYS
    inactive, active, mapped, dirty, writeback = (values.get(key, 0) for key in keys)
    return inactive + max(active - mapped - dirty - writeback, 0)


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
    can bind through sibling usage), less reclaimable page cache; empty = no finite limit."""
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
                0,
                used - _reclaimable_file_bytes(os.path.join(directory, "memory.stat"), v1 = False),
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
            used = max(
                0, used - _reclaimable_file_bytes(os.path.join(directory, "memory.stat"), v1 = True)
            )
        budgets.append((limit if used is None else limit - used, limit))
    return budgets


def _v1_reclaimable_usage(directory: str, raw: Optional[int]) -> Optional[int]:
    """A v1 ``*.usage_in_bytes`` reading less reclaimable file cache, as ``cgroup_memory_budgets`` reads it."""
    if raw is None:
        return None
    reclaimable = _reclaimable_file_bytes(os.path.join(directory, "memory.stat"), v1 = True)
    return max(raw - reclaimable, 0)


def cgroup_swap_allowance_bytes(
    free_swap: int,
    cgroup_root: Optional[str] = None,
    proc_self_cgroup: Optional[str] = None,
    *,
    ram_budget: Optional[int] = None,
) -> int:
    """How much of the host's ``free_swap`` this process may still use: bounded by every enforcing cgroup swap
    limit on this cgroup and its ancestors: v2 ``memory.swap.max`` minus ``memory.swap.current`` per level (a
    swap-only counter, so the per-level minimum is exact); v1 the tightest ``memsw`` (RAM+swap) headroom minus the
    RAM the caller actually budgets (``ram_budget``, e.g. min(MemAvailable, cgroup headroom)), else minus the
    tightest cgroup RAM headroom. Unreadable cgroup membership is read as the namespace root, like
    ``cgroup_memory_budgets``, so a root RAM limit without readable swap files still yields 0. 0 when a memory limit is enforced but its swap allowance cannot be read."""
    free_swap = max(int(free_swap or 0), 0)
    cgroup_root = CGROUP_ROOT if cgroup_root is None else cgroup_root
    proc_self_cgroup = PROC_SELF_CGROUP if proc_self_cgroup is None else proc_self_cgroup
    try:
        with open(proc_self_cgroup, "r", encoding = "utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except (OSError, UnicodeDecodeError):
        lines = []
    allowance = free_swap
    v2_relative = next((line[3:] for line in lines if line.startswith("0::")), None)
    if not lines:
        # Membership unknown: judge the namespace root (where cgroup_memory_budgets reads the RAM limit), fail closed.
        v2_relative = "/"
    if v2_relative is not None:
        for directory in _directories(cgroup_root, v2_relative):
            raw = _first_line(os.path.join(directory, "memory.swap.max"))
            if raw is None:
                # No swap accounting here: a RAM limit at this level makes the swap it leaves unknowable.
                if (
                    _integer(_first_line(os.path.join(directory, "memory.max")), limit = True)
                    is not None
                ):
                    return 0
                continue
            limit = _integer(raw, limit = True)
            if limit is None:
                if raw != "max":
                    return 0
                continue
            used = _integer(_first_line(os.path.join(directory, "memory.swap.current")))
            if used is None:
                return 0
            allowance = min(allowance, max(limit - used, 0))
    v1_relative = None if lines else "/"
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) == 3 and "memory" in parts[1].split(","):
            v1_relative = parts[2]
            break
    if v1_relative is not None:
        # v1 memsw is RAM+swap: swap = tightest combined headroom - tightest RAM headroom, never per-level differences.
        v1_root = os.path.join(cgroup_root, "memory")
        # swappiness 0 on an enforcing (limited) level means its reclaim never swaps; unlimited levels are irrelevant.
        for directory in _directories(v1_root, v1_relative):
            enforcing = any(
                _integer(_first_line(os.path.join(directory, name)), limit = True) is not None
                for name in ("memory.limit_in_bytes", "memory.memsw.limit_in_bytes")
            )
            if (
                enforcing
                and _integer(_first_line(os.path.join(directory, "memory.swappiness"))) == 0
            ):
                return 0
        ram_room: Optional[int] = None
        combined_room: Optional[int] = None
        for directory in _directories(v1_root, v1_relative):
            mem_limit = _integer(
                _first_line(os.path.join(directory, "memory.limit_in_bytes")), limit = True
            )
            if mem_limit is None:
                continue
            mem_used = _v1_reclaimable_usage(
                directory, _integer(_first_line(os.path.join(directory, "memory.usage_in_bytes")))
            )
            raw_memsw = _first_line(os.path.join(directory, "memory.memsw.limit_in_bytes"))
            if raw_memsw is None or mem_used is None:
                return 0  # a RAM limit without swap accounting (or usage): the swap it leaves is unknown
            room = mem_limit - mem_used
            ram_room = room if ram_room is None else min(ram_room, room)
            memsw_limit = _integer(raw_memsw, limit = True)
            if memsw_limit is None:
                continue  # RAM+swap unlimited at this level: it bounds RAM only
            memsw_used = _v1_reclaimable_usage(
                directory,
                _integer(_first_line(os.path.join(directory, "memory.memsw.usage_in_bytes"))),
            )
            if memsw_used is None:
                return 0
            room = memsw_limit - memsw_used
            combined_room = room if combined_room is None else min(combined_room, room)
        if combined_room is not None:
            ram_side = ram_room if ram_budget is None else ram_budget
            allowance = min(allowance, max(combined_room - max(ram_side or 0, 0), 0))
    return allowance


def cgroup_v1_memsw_headroom_bytes(
    cgroup_root: Optional[str] = None, proc_self_cgroup: Optional[str] = None
) -> Optional[int]:
    """The tightest cgroup v1 RAM+swap (``memsw``) headroom on this cgroup and its ancestors, the most RAM and swap
    together this process can still charge; None without a finite memsw limit. Unreadable membership is read as
    the namespace root; a finite limit whose usage cannot be read yields 0."""
    cgroup_root = CGROUP_ROOT if cgroup_root is None else cgroup_root
    proc_self_cgroup = PROC_SELF_CGROUP if proc_self_cgroup is None else proc_self_cgroup
    try:
        with open(proc_self_cgroup, "r", encoding = "utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except (OSError, UnicodeDecodeError):
        lines = []
    relative = None if lines else "/"
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) == 3 and "memory" in parts[1].split(","):
            relative = parts[2]
            break
    if relative is None:
        return None
    best: Optional[int] = None
    for directory in _directories(os.path.join(cgroup_root, "memory"), relative):
        limit = _integer(
            _first_line(os.path.join(directory, "memory.memsw.limit_in_bytes")), limit = True
        )
        if limit is None:
            continue
        used = _v1_reclaimable_usage(
            directory, _integer(_first_line(os.path.join(directory, "memory.memsw.usage_in_bytes")))
        )
        room = 0 if used is None else max(limit - used, 0)
        best = room if best is None else min(best, room)
    return best


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
