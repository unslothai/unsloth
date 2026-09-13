#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""One table over what every layer says this GPU's memory is, and the three
statements that table is being collected to settle.

Reads the JSON written by probes/gpu_memory_report_probe.py (one file per arm)
and prints markdown. The statements are checked, not narrated:

  over_report      the number llama.cpp places against exceeds the machine's
                   physical RAM, which no single device can hold.
  sums_heaps       that number equals the heaps added together rather than the
                   largest one, which is how the over-report arises on an
                   integrated part where several heaps alias the same RAM. ggml
                   adds EVERY heap, not only the device-local ones.
  hip_is_vgm_only  the HIP total equals the carve-out the registry records, so
                   the two backends disagree because they are answering
                   different questions, not because one is broken.

Exits non-zero when a reading needed for a statement is missing, so a missing
section can never read as "the statement did not hold".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GIB = 1024**3
# Two readings of the same pool never match to the byte: drivers round, reserve
# and report at different moments. 2% is wide enough for that and far narrower
# than the gaps being distinguished (a heap sum is 2x-3x a single heap).
TOL = 0.02


def gib(value) -> float | None:
    return round(value / GIB, 2) if isinstance(value, (int, float)) and value else None


def close(a, b) -> bool:
    return bool(a and b and abs(a - b) <= TOL * max(a, b))


def sec(doc: dict, name: str) -> dict:
    return (doc.get("sections") or {}).get(name) or {}


def host_numbers(doc: dict) -> dict:
    host = sec(doc, "host")
    vram = 0
    for adapter in host.get("adapters") or []:
        size = adapter.get("qw_memory_size")
        if isinstance(size, int):
            vram = max(vram, size)
    return {
        "ram_gib": gib(host.get("total_phys_bytes") or host.get("memtotal_bytes")),
        "registry_vram_gib": gib(vram),
        "pagefile_gib": gib(host.get("total_pagefile_bytes")),
    }


def rows_for(name: str, doc: dict) -> list[str]:
    out = []
    h = host_numbers(doc)
    out.append(f"| {name} | host: physical RAM | {h['ram_gib']} GiB | GlobalMemoryStatusEx |")
    out.append(
        f"| {name} | host: registry VRAM (carve-out) | {h['registry_vram_gib']} GiB | "
        f"HardwareInformation.qwMemorySize |"
    )
    for d in sec(doc, "hip").get("devices") or []:
        out.append(
            f"| {name} | HIP device {d.get('index')} total | "
            f"{gib(d.get('total_bytes') or d.get('device_total_bytes'))} GiB | "
            f"hipMemGetInfo / hipDeviceTotalMem ({d.get('name', '')}) |"
        )
        out.append(
            f"| {name} | HIP device {d.get('index')} free | {gib(d.get('free_bytes'))} GiB "
            f"| process-scoped on Windows, an optimistic ceiling |"
        )
    for d in sec(doc, "vulkan_raw").get("devices") or []:
        heaps = ", ".join(
            f"{gib(x['size_bytes'])}{'L' if x['device_local'] else ''}"
            for x in d.get("heaps") or []
        )
        out.append(
            f"| {name} | Vulkan raw heaps ({d.get('type')}) | {heaps} GiB | "
            f"vkGetPhysicalDeviceMemoryProperties2, L = DEVICE_LOCAL |"
        )
        out.append(
            f"| {name} | Vulkan device-local sum / max | "
            f"{gib(d.get('device_local_sum_bytes'))} / "
            f"{gib(d.get('device_local_max_bytes'))} GiB | the two candidate readings |"
        )
    for d in sec(doc, "ggml_vulkan").get("devices") or []:
        out.append(
            f"| {name} | ggml Vulkan total (what llama.cpp places against) | "
            f"{gib(d.get('total_bytes'))} GiB | ggml_backend_vk_get_device_memory, "
            f"igpu={d.get('is_igpu')} |"
        )
        out.append(f"| {name} | ggml Vulkan free | {gib(d.get('free_bytes'))} GiB | same call |")
    for line in (sec(doc, "fit").get("list_devices") or {}).get("stdout", "").splitlines():
        if "MiB" in line:
            out.append(f"| {name} | llama-server --list-devices | `{line.strip()}` | the loader |")
    return out


def statements(name: str, doc: dict) -> list[tuple[str, bool | None, str]]:
    h = host_numbers(doc)
    vk_raw = (sec(doc, "vulkan_raw").get("devices") or [None])[0]
    ggml = (sec(doc, "ggml_vulkan").get("devices") or [None])[0]
    hip = (sec(doc, "hip").get("devices") or [None])[0]
    out: list[tuple[str, bool | None, str]] = []

    ggml_total = gib((ggml or {}).get("total_bytes"))
    # The bound is visible RAM PLUS the carve-out, not visible RAM: Windows
    # subtracts the carve-out from what it reports, so a 128 GB machine with a
    # 64 GiB VGM shows about 63 GiB and comparing against that alone would call
    # every honest reading an over-report.
    physical = None
    if h["ram_gib"] is not None:
        physical = round(h["ram_gib"] + (h["registry_vram_gib"] or 0), 2)
    if ggml_total is None or physical is None:
        out.append((f"{name}: over_report", None, "no ggml Vulkan total or no host RAM reading"))
    else:
        out.append(
            (
                f"{name}: over_report",
                ggml_total > physical,
                f"ggml reports {ggml_total} GiB where the machine holds {physical} GiB "
                f"({h['ram_gib']} GiB visible plus a {h['registry_vram_gib']} GiB carve-out)",
            )
        )

    local_sum = gib((vk_raw or {}).get("device_local_sum_bytes"))
    local_max = gib((vk_raw or {}).get("device_local_max_bytes"))
    all_sum = gib((vk_raw or {}).get("all_heaps_sum_bytes"))
    if ggml_total is None or local_sum is None:
        out.append((f"{name}: sums_heaps", None, "no raw Vulkan heaps or no ggml total"))
    else:
        # ggml adds EVERY heap on an integrated device, not only the device-local
        # ones, and on Strix Halo the host-visible aperture is its own heap. An
        # earlier version compared against the device-local sum alone and so
        # answered NO on a machine where the summation was the whole mechanism.
        which = (
            "all heaps"
            if close(ggml_total, all_sum)
            else "the device-local heaps"
            if close(ggml_total, local_sum)
            else None
        )
        out.append(
            (
                f"{name}: sums_heaps",
                bool(which) and not close(ggml_total, local_max),
                f"ggml {ggml_total} GiB against {all_sum} GiB over all heaps, "
                f"{local_sum} GiB device-local and {local_max} GiB largest"
                + (f": it is {which}" if which else ": it matches none of them"),
            )
        )

    hip_total = gib((hip or {}).get("total_bytes") or (hip or {}).get("device_total_bytes"))
    if hip_total is None or h["registry_vram_gib"] is None:
        out.append((f"{name}: hip_is_vgm_only", None, "no HIP total or no registry VRAM"))
    else:
        out.append(
            (
                f"{name}: hip_is_vgm_only",
                close(hip_total, h["registry_vram_gib"]),
                f"HIP {hip_total} GiB against a {h['registry_vram_gib']} GiB carve-out",
            )
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs = "+", help = "probe JSON files, name=path or path")
    ap.add_argument(
        "--require", default = "", help = "comma-separated statements that must be decidable (not None)"
    )
    a = ap.parse_args()

    docs: dict[str, dict] = {}
    for item in a.reports:
        name, _, path = item.partition("=")
        if not path:
            name, path = Path(item).stem.replace("memreport_", ""), item
        try:
            docs[name] = json.loads(Path(path).read_text(encoding = "utf-8"))
        except Exception as e:  # noqa: BLE001
            docs[name] = {"_error": f"{type(e).__name__}: {e}"}

    print("### What each layer reports\n")
    print("| arm | layer | value | source |")
    print("|---|---|---|---|")
    for name, doc in docs.items():
        if doc.get("_error"):
            print(f"| {name} | UNREADABLE | - | {doc['_error']} |")
            continue
        for line in rows_for(name, doc):
            print(line)
        for k, v in (doc.get("errors") or {}).items():
            print(f"| {name} | section {k} FAILED | - | {v} |")

    print("\n### Statements\n")
    print("| statement | holds | evidence |")
    print("|---|---|---|")
    undecided: list[str] = []
    for name, doc in docs.items():
        if doc.get("_error"):
            continue
        for label, ok, why in statements(name, doc):
            print(
                f"| {label} | {'yes' if ok else ('NO' if ok is False else 'undecided')} | {why} |"
            )
            if ok is None:
                undecided.append(label)

    required = [s.strip() for s in a.require.split(",") if s.strip()]
    blocking = [u for u in undecided if any(u.endswith(": " + r) for r in required)]
    if blocking:
        print(
            f"\n**A required statement could not be decided: {blocking}.** An undecidable "
            f"statement is a missing reading, not a negative result."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
