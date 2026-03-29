# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import subprocess
from typing import Any, Optional


def _parse_smi_value(raw: str):
    raw = raw.strip()
    if not raw or raw == "[N/A]":
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def _visible_ordinal_map(
    parent_visible_ids: Optional[list[int]],
) -> Optional[dict[int, int]]:
    if parent_visible_ids is None:
        return None
    return {gpu_id: ordinal for ordinal, gpu_id in enumerate(parent_visible_ids)}


def get_physical_gpu_count() -> int:
    result = subprocess.run(
        ["nvidia-smi", "-L"],
        capture_output = True,
        text = True,
        timeout = 5,
    )
    if result.returncode == 0 and result.stdout.strip():
        return len(result.stdout.strip().splitlines())
    return 1


def get_primary_gpu_utilization() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,temperature.gpu,"
            "memory.used,memory.total,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ],
        capture_output = True,
        text = True,
        timeout = 5,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {"available": False}

    first_line = result.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in first_line.split(",")]
    if len(parts) < 6:
        return {"available": False}

    vram_used_mb = _parse_smi_value(parts[2])
    vram_total_mb = _parse_smi_value(parts[3])
    power_draw = _parse_smi_value(parts[4])
    power_limit = _parse_smi_value(parts[5])

    return {
        "available": True,
        "gpu_utilization_pct": _parse_smi_value(parts[0]),
        "temperature_c": _parse_smi_value(parts[1]),
        "vram_used_gb": round(vram_used_mb / 1024, 2)
        if vram_used_mb is not None
        else None,
        "vram_total_gb": round(vram_total_mb / 1024, 2)
        if vram_total_mb is not None
        else None,
        "vram_utilization_pct": round((vram_used_mb / vram_total_mb) * 100, 1)
        if vram_used_mb is not None and vram_total_mb and vram_total_mb > 0
        else None,
        "power_draw_w": power_draw,
        "power_limit_w": power_limit,
        "power_utilization_pct": round((power_draw / power_limit) * 100, 1)
        if power_draw is not None and power_limit and power_limit > 0
        else None,
    }


def get_visible_gpu_utilization(
    parent_visible_ids: Optional[list[int]],
    parent_cuda_visible_devices: Optional[str] = None,
) -> dict[str, Any]:
    visible_ordinals = _visible_ordinal_map(parent_visible_ids)
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,temperature.gpu,"
            "memory.used,memory.total,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ],
        capture_output = True,
        text = True,
        timeout = 5,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {
            "available": False,
            "backend_cuda_visible_devices": parent_cuda_visible_devices,
            "parent_visible_gpu_ids": parent_visible_ids or [],
            "devices": [],
            "index_kind": "physical",
        }

    devices = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue

        idx = int(parts[0])
        if visible_ordinals is not None and idx not in visible_ordinals:
            continue

        vram_used_mb = _parse_smi_value(parts[3])
        vram_total_mb = _parse_smi_value(parts[4])
        power_draw = _parse_smi_value(parts[5])
        power_limit = _parse_smi_value(parts[6])
        devices.append(
            {
                "index": idx,
                "index_kind": "physical",
                "visible_ordinal": (
                    visible_ordinals[idx]
                    if visible_ordinals is not None
                    else len(devices)
                ),
                "gpu_utilization_pct": _parse_smi_value(parts[1]),
                "temperature_c": _parse_smi_value(parts[2]),
                "vram_used_gb": round(vram_used_mb / 1024, 2)
                if vram_used_mb is not None
                else None,
                "vram_total_gb": round(vram_total_mb / 1024, 2)
                if vram_total_mb is not None
                else None,
                "vram_utilization_pct": round((vram_used_mb / vram_total_mb) * 100, 1)
                if vram_used_mb is not None and vram_total_mb and vram_total_mb > 0
                else None,
                "power_draw_w": power_draw,
                "power_limit_w": power_limit,
                "power_utilization_pct": round((power_draw / power_limit) * 100, 1)
                if power_draw is not None and power_limit and power_limit > 0
                else None,
            }
        )

    return {
        "available": len(devices) > 0,
        "backend_cuda_visible_devices": parent_cuda_visible_devices,
        "parent_visible_gpu_ids": parent_visible_ids or [],
        "devices": devices,
        "index_kind": "physical",
    }


def get_backend_visible_gpu_info(
    parent_visible_ids: Optional[list[int]],
    backend_cuda_visible_devices: Optional[str],
) -> dict[str, Any]:
    visible_ordinals = _visible_ordinal_map(parent_visible_ids)
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output = True,
        text = True,
        timeout = 10,
    )
    if result.returncode != 0:
        return {
            "available": False,
            "backend_cuda_visible_devices": backend_cuda_visible_devices,
            "parent_visible_gpu_ids": parent_visible_ids or [],
            "devices": [],
            "index_kind": "physical",
        }

    devices = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        idx = int(parts[0])
        if visible_ordinals is not None and idx not in visible_ordinals:
            continue
        devices.append(
            {
                "index": idx,
                "index_kind": "physical",
                "visible_ordinal": (
                    visible_ordinals[idx]
                    if visible_ordinals is not None
                    else len(devices)
                ),
                "name": parts[1],
                "memory_total_gb": round(int(parts[2]) / 1024, 2),
            }
        )

    return {
        "available": len(devices) > 0,
        "backend_cuda_visible_devices": backend_cuda_visible_devices,
        "parent_visible_gpu_ids": parent_visible_ids or [],
        "devices": devices,
        "index_kind": "physical",
    }
