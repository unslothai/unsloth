# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NUMA topology detection and an auto-interleave decision for CPU-only launches.

Linux's first-touch policy faults a model's pages onto one node; if the GGUF is larger
than that node's free RAM the load thrashes/fails despite ample total RAM.
`numactl --interleave=all` spreads pages across nodes (llama.cpp discussion #19102).
Pure stdlib, Linux-only (no-op decision elsewhere), side-effect free.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

_NODE_ROOT = Path("/sys/devices/system/node")


@dataclass(frozen = True)
class NumaTopology:
    """Per-node free memory (MemFree, matching `numactl --hardware`'s 'node N free')."""

    node_free_mib: dict[int, int] = field(default_factory = dict)

    @property
    def node_count(self) -> int:
        return len(self.node_free_mib)

    @property
    def total_free_mib(self) -> int:
        return sum(self.node_free_mib.values())

    @property
    def largest_node_free_mib(self) -> int:
        return max(self.node_free_mib.values(), default = 0)

    @property
    def smallest_node_free_mib(self) -> int:
        return min(self.node_free_mib.values(), default = 0)


def _parse_online(spec: str) -> list[int]:
    """Parse a sysfs cpulist-style range, e.g. '0-1' or '0,2-3', into node ids."""
    out: list[int] = []
    for part in spec.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            try:
                out.extend(range(int(lo), int(hi) + 1))
            except ValueError:
                continue
        else:
            try:
                out.append(int(part))
            except ValueError:
                continue
    return out


def _mems_allowed() -> set[int] | None:
    """NUMA nodes the process may allocate on (cpuset), from /proc/self/status
    Mems_allowed_list. None when unavailable, so callers fall back to the online set.
    numactl --interleave=all only spans these, so a cpuset-limited container must not
    count host nodes it cannot use."""
    try:
        text = Path("/proc/self/status").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("Mems_allowed_list:"):
            return set(_parse_online(line.split(":", 1)[1]))
    return None


def _node_memfree_mib(node: int) -> int | None:
    """MemFree for one node from /sys/.../nodeN/meminfo, in MiB (kB -> MiB)."""
    try:
        text = (_NODE_ROOT / f"node{node}" / "meminfo").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        # "Node 0 MemFree:           465594 kB"
        if "MemFree:" in line:
            parts = line.split()
            try:
                kb = int(parts[parts.index("MemFree:") + 1])
            except (ValueError, IndexError):
                return None
            return kb // 1024
    return None


def read_numa_topology() -> NumaTopology:
    """Read NUMA node free memory from sysfs. Empty topology on non-Linux / no sysfs /
    single-node (a single node is reported but callers treat node_count <= 1 as 'no
    interleave needed')."""
    try:
        online = (_NODE_ROOT / "online").read_text()
    except OSError:
        return NumaTopology()
    # Restrict to cpuset-allowed nodes: under a Docker/systemd cpuset the child can only
    # allocate on these, so counting other host nodes would overstate the fittable RAM.
    allowed = _mems_allowed()
    free: dict[int, int] = {}
    for node in _parse_online(online):
        if allowed is not None and node not in allowed:
            continue
        mib = _node_memfree_mib(node)
        if mib is not None:
            free[node] = mib
    return NumaTopology(node_free_mib = free)


def numactl_available() -> bool:
    return shutil.which("numactl") is not None


@dataclass(frozen = True)
class InterleaveDecision:
    interleave: bool
    reason: str
    prefix: tuple[str, ...] = ()  # argv prefix to apply, e.g. ("numactl", "--interleave=all")


def decide_interleave(
    model_size_bytes: int | None,
    *,
    cpu_only: bool,
    topology: NumaTopology | None = None,
    has_numactl: bool | None = None,
) -> InterleaveDecision:
    """Interleave only when CPU-only, multi-node, and the footprint exceeds the smallest
    node's free RAM but fits across all nodes; otherwise leave placement local. The
    smallest node is the bound because the loader is not pinned, so first-touch may land
    on any node. model_size_bytes should be the resident footprint (weights + KV), not
    weights alone, so a model whose weights fit a node but whose footprint does not still
    interleaves."""
    if not cpu_only:
        return InterleaveDecision(False, "not cpu-only; leaving NUMA placement to the OS")
    if not model_size_bytes or model_size_bytes <= 0:
        return InterleaveDecision(False, "model size unknown; not forcing interleave")

    topo = topology if topology is not None else read_numa_topology()
    if topo.node_count <= 1:
        return InterleaveDecision(False, "single NUMA node; interleave not needed")

    model_mib = model_size_bytes // (1024 * 1024)
    smallest = topo.smallest_node_free_mib
    total = topo.total_free_mib

    # Keep local only when the footprint fits EVERY node (the smallest). We don't bind
    # the loader, so first-touch may land on any node; local placement is safe only when
    # any node can hold it. A footprint that fits only the larger node still interleaves.
    if model_mib <= smallest:
        return InterleaveDecision(
            False,
            f"model ~{model_mib} MiB fits every node's free RAM "
            f"(smallest ~{smallest} MiB); keeping local placement",
        )

    # Impossible across all nodes regardless of numactl: surface the smaller-quant /
    # free-memory path before the numactl hint, so a too-big model is not told to
    # install numactl when interleaving could never make it fit.
    if model_mib > total:
        return InterleaveDecision(
            False,
            f"model ~{model_mib} MiB exceeds total free RAM across all nodes "
            f"(~{total} MiB); interleave cannot help -- free memory or use a smaller quant",
        )

    avail = numactl_available() if has_numactl is None else has_numactl
    if not avail:
        # Needed but unavailable: surface it; caller decides whether to block.
        return InterleaveDecision(
            False,
            f"model ~{model_mib} MiB exceeds the smallest NUMA node's free RAM "
            f"(~{smallest} MiB) and needs interleaving across {topo.node_count} nodes, "
            f"but `numactl` is not installed. Install numactl (e.g. `apt install "
            f"numactl`) or the model may fail to fit a single node.",
        )

    return InterleaveDecision(
        True,
        f"model ~{model_mib} MiB exceeds the smallest NUMA node's free RAM "
        f"(~{smallest} MiB) but fits across {topo.node_count} nodes (~{total} MiB total "
        f"free); wrapping with numactl --interleave=all",
        prefix = ("numactl", "--interleave=all"),
    )
