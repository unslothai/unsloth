# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""One CPU share per pytest-xdist worker for the torch / OpenMP / BLAS thread pools.

Each xdist worker otherwise starts a pool as wide as the machine, so `-n 4` on a 4 vCPU
runner runs 16 spinning threads on 4 cores and CPU training tests slow down by an order
of magnitude. Loaded by tests/conftest.py before torch is imported. Stdlib only.
"""

from __future__ import annotations

import os
from typing import MutableMapping, Optional

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
# Set by apply(): the share it gave each worker, or "explicit" when the caller had already
# chosen a thread count. tests/test_xdist_torch_thread_cap.py reads it.
MARKER = "UNSLOTH_TEST_XDIST_CPU_SHARE"
CGROUP_ROOT = "/sys/fs/cgroup"


def _first_line(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding = "utf-8") as f:
            return f.readline().strip()
    except OSError:
        return None


def _dirs(root: str, rel: Optional[str]) -> list:
    """``root/rel`` and every ancestor up to *root*, innermost first: a parent's limit binds too."""
    dirs = []
    if rel and rel != "/":
        current = os.path.normpath(os.path.join(root, rel.lstrip("/")))
        while current.startswith(root + os.sep):
            dirs.append(current)
            current = os.path.dirname(current)
    dirs.append(root)
    return dirs


def _ratio(quota: Optional[str], period: Optional[str]) -> Optional[float]:
    try:
        q, p = int(quota), int(period)
    except (TypeError, ValueError):
        return None  # "max", missing or unreadable means unlimited
    return q / p if q > 0 and p > 0 else None


def cgroup_cpu_quota(
    root: str = CGROUP_ROOT, proc_cgroup: str = "/proc/self/cgroup"
) -> Optional[float]:
    """The tightest CFS CPU quota over this process's cgroup and its ancestors, in cores.

    None when no quota applies. sched_getaffinity() does not see a quota, so a container
    limited to 4 CPUs on a 64 CPU host would otherwise hand each of 4 workers 16 threads.
    """
    try:
        with open(proc_cgroup, "r", encoding = "utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except OSError:
        lines = []
    quotas = []
    if os.path.isfile(os.path.join(root, "cgroup.controllers")):  # v2
        rel = next((line[3:] for line in lines if line.startswith("0::")), None)
        for directory in _dirs(root, rel):
            raw = (_first_line(os.path.join(directory, "cpu.max")) or "").split()
            if len(raw) == 2:
                quotas.append(_ratio(raw[0], raw[1]))
    else:  # v1
        rel = None
        for line in lines:
            parts = line.split(":", 2)
            if len(parts) == 3 and "cpu" in parts[1].split(","):
                rel = parts[2]
                break
        for mount in ("cpu", "cpu,cpuacct", "cpuacct,cpu"):
            base = os.path.join(root, mount)
            if not os.path.isdir(base):
                continue
            for directory in _dirs(base, rel):
                quotas.append(
                    _ratio(
                        _first_line(os.path.join(directory, "cpu.cfs_quota_us")),
                        _first_line(os.path.join(directory, "cpu.cfs_period_us")),
                    )
                )
            break
    quotas = [q for q in quotas if q is not None]
    return min(quotas) if quotas else None


def usable_cpus() -> int:
    """CPUs this process may actually use: the affinity mask, capped by any cgroup quota."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    quota = cgroup_cpu_quota()
    if quota is not None:
        cpus = min(cpus, max(1, int(quota)))
    return max(1, cpus)


def apply(environ: MutableMapping[str, str], cpus: Optional[int] = None) -> Optional[str]:
    """Give each xdist worker ``cpus // workers`` threads. Returns what it recorded in MARKER.

    Leaves every variable alone when any of them is already set: MKL_NUM_THREADS takes
    precedence over OMP_NUM_THREADS in torch, so filling in the others beside an explicit
    OMP_NUM_THREADS would override the caller's choice.
    """
    workers = int(environ.get("PYTEST_XDIST_WORKER_COUNT") or 0)
    if workers <= 1:
        return None
    if any(environ.get(var) for var in THREAD_VARS):
        environ[MARKER] = "explicit"
        return "explicit"
    share = str(max(1, (cpus if cpus is not None else usable_cpus()) // workers))
    for var in THREAD_VARS:
        environ[var] = share
    environ[MARKER] = share
    return share
