# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import subprocess
from typing import Any, Optional

from loggers import get_logger

from utils.native_path_leases import child_env_without_native_path_secret
from utils.subprocess_compat import (
    windows_hidden_subprocess_kwargs as _windows_hidden_subprocess_kwargs,
)

logger = get_logger(__name__)


def _parse_smi_value(raw: str):
    raw = raw.strip()
    if not raw or raw == "[N/A]":
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def _build_gpu_metrics(
    vram_used_mb, vram_total_mb, power_draw, power_limit, **extra
) -> dict[str, Any]:
    return {
        **extra,
        "vram_used_gb": round(vram_used_mb / 1024, 2) if vram_used_mb is not None else None,
        "vram_total_gb": round(vram_total_mb / 1024, 2) if vram_total_mb is not None else None,
        "vram_utilization_pct": round((vram_used_mb / vram_total_mb) * 100, 1)
        if vram_used_mb is not None and vram_total_mb and vram_total_mb > 0
        else None,
        "power_draw_w": power_draw,
        "power_limit_w": power_limit,
        "power_utilization_pct": round((power_draw / power_limit) * 100, 1)
        if power_draw is not None and power_limit and power_limit > 0
        else None,
    }


def _visible_ordinal_map(parent_visible_ids: Optional[list[int]]) -> Optional[dict[int, int]]:
    if parent_visible_ids is None:
        return None
    return {gpu_id: ordinal for ordinal, gpu_id in enumerate(parent_visible_ids)}


def _uuid_visible_ordinal_map(
    parent_cuda_visible_devices: Optional[str], gpu_rows: list[tuple[int, str]]
) -> Optional[dict[int, int]]:
    """Resolve an ordered full-GPU UUID mask against nvidia-smi rows."""
    tokens = [
        token.strip().lower()
        for token in (parent_cuda_visible_devices or "").split(",")
        if token.strip()
    ]
    if not tokens or any(not token.startswith("gpu-") for token in tokens):
        return None

    visible_ordinals: dict[int, int] = {}
    for ordinal, token in enumerate(tokens):
        matches = [idx for idx, gpu_uuid in gpu_rows if gpu_uuid.lower().startswith(token)]
        if len(matches) != 1 or matches[0] in visible_ordinals:
            return None
        visible_ordinals[matches[0]] = ordinal
    return visible_ordinals


def get_physical_gpu_count() -> Optional[int]:
    """Return physical GPU count via nvidia-smi, or None on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 5,
            env = child_env_without_native_path_secret(),
            **_windows_hidden_subprocess_kwargs(),
        )
        if result.returncode == 0 and result.stdout.strip():
            return len(result.stdout.strip().splitlines())
        logger.warning(
            "nvidia-smi -L returned code %d; caller should fall back to torch",
            result.returncode,
        )
    except Exception as e:
        logger.warning("nvidia-smi -L failed: %s; caller should fall back to torch", e)
    return None


def get_primary_gpu_utilization() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,"
                "memory.used,memory.total,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 5,
            env = child_env_without_native_path_secret(),
            **_windows_hidden_subprocess_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning("nvidia-smi query failed in get_primary_gpu_utilization: %s", e)
        return {"available": False}
    if result.returncode != 0 or not result.stdout.strip():
        return {"available": False}

    first_line = result.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in first_line.split(",")]
    if len(parts) < 6:
        return {"available": False}

    return _build_gpu_metrics(
        vram_used_mb = _parse_smi_value(parts[2]),
        vram_total_mb = _parse_smi_value(parts[3]),
        power_draw = _parse_smi_value(parts[4]),
        power_limit = _parse_smi_value(parts[5]),
        available = True,
        gpu_utilization_pct = _parse_smi_value(parts[0]),
        temperature_c = _parse_smi_value(parts[1]),
    )


def get_visible_gpu_utilization(
    parent_visible_ids: Optional[list[int]], parent_cuda_visible_devices: Optional[str] = None
) -> dict[str, Any]:
    visible_ordinals = _visible_ordinal_map(parent_visible_ids)
    includes_uuid = parent_visible_ids is None
    query_fields = "index,"
    if includes_uuid:
        query_fields += "uuid,"
    query_fields += (
        "utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw,power.limit"
    )
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query_fields}",
                "--format=csv,noheader,nounits",
            ],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 5,
            env = child_env_without_native_path_secret(),
            **_windows_hidden_subprocess_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning("nvidia-smi query failed in get_visible_gpu_utilization: %s", e)
        return {
            "available": False,
            "backend_cuda_visible_devices": parent_cuda_visible_devices,
            "parent_visible_gpu_ids": parent_visible_ids or [],
            "devices": [],
            "index_kind": "physical" if parent_visible_ids is not None else "unresolved",
        }
    if result.returncode != 0 or not result.stdout.strip():
        return {
            "available": False,
            "backend_cuda_visible_devices": parent_cuda_visible_devices,
            "parent_visible_gpu_ids": parent_visible_ids or [],
            "devices": [],
            "index_kind": "physical" if parent_visible_ids is not None else "unresolved",
        }

    gpu_rows: list[tuple[int, list[str]]] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < (8 if includes_uuid else 7):
            continue

        try:
            idx = int(parts[0])
        except (ValueError, TypeError):
            continue
        gpu_rows.append((idx, parts))

    if parent_visible_ids is None:
        visible_ordinals = _uuid_visible_ordinal_map(
            parent_cuda_visible_devices,
            [(idx, parts[1]) for idx, parts in gpu_rows],
        )
        if visible_ordinals is None:
            return {
                "available": False,
                "backend_cuda_visible_devices": parent_cuda_visible_devices,
                "parent_visible_gpu_ids": [],
                "devices": [],
                "index_kind": "unresolved",
            }

    devices = []
    field_offset = 1 if includes_uuid else 0
    for idx, parts in gpu_rows:
        if visible_ordinals is not None and idx not in visible_ordinals:
            continue

        visible_ordinal = visible_ordinals[idx] if visible_ordinals is not None else len(devices)
        devices.append(
            _build_gpu_metrics(
                vram_used_mb = _parse_smi_value(parts[3 + field_offset]),
                vram_total_mb = _parse_smi_value(parts[4 + field_offset]),
                power_draw = _parse_smi_value(parts[5 + field_offset]),
                power_limit = _parse_smi_value(parts[6 + field_offset]),
                index = visible_ordinal if includes_uuid else idx,
                index_kind = "relative" if includes_uuid else "physical",
                visible_ordinal = visible_ordinal,
                gpu_utilization_pct = _parse_smi_value(parts[1 + field_offset]),
                temperature_c = _parse_smi_value(parts[2 + field_offset]),
            )
        )

    # nvidia-smi emits physical row order, so a reordering mask would hand back
    # devices whose position contradicts their own visible_ordinal.
    devices.sort(key = lambda d: d["visible_ordinal"])

    return {
        "available": len(devices) > 0,
        "backend_cuda_visible_devices": parent_cuda_visible_devices,
        "parent_visible_gpu_ids": parent_visible_ids or [],
        "devices": devices,
        "index_kind": "relative" if includes_uuid else "physical",
    }


def _query_gpu_inventory(caller: str) -> Optional[list[dict[str, Any]]]:
    """``[{index, name, memory_total_gb}]`` for every GPU nvidia-smi enumerates.

    ``None`` when the query could not be answered at all -- no nvidia-smi on PATH, a
    driver that hung past the timeout, a non-zero exit. Callers report that as
    "unknown", which is not the same as the empty list a working driver with no
    cards returns. Never raises.

    Split out of get_backend_visible_gpu_info so the same rows can be read WITHOUT a
    ``DeviceType.CUDA`` precondition: get_physical_gpu_inventory below is reached on
    exactly the host where torch reports no CUDA device, and that host still has its
    GPUs. Rows a caller cannot make sense of are dropped, not raised on -- a name
    holding commas is rejoined, and a malformed index or memory column skips the row.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 10,
            env = child_env_without_native_path_secret(),
            **_windows_hidden_subprocess_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning("nvidia-smi query failed in %s: %s", caller, e)
        return None
    if result.returncode != 0:
        return None

    rows: list[dict[str, Any]] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
        except (ValueError, TypeError):
            continue
        # Rejoin in case the GPU name contains commas
        name = parts[1] if len(parts) == 3 else ", ".join(parts[1:-1])
        try:
            mem_total_mb = int(parts[-1])
        except (ValueError, TypeError):
            continue
        rows.append(
            {
                "index": idx,
                "name": name,
                "memory_total_gb": round(mem_total_mb / 1024, 2),
            }
        )
    return rows


def get_physical_gpu_inventory() -> dict[str, Any]:
    """Every NVIDIA GPU the driver enumerates, with no visibility mask and no torch.

    Display-only inventory: ``index`` is nvidia-smi's own row number, which is a
    physical id and NOT something a caller may pin, because the whole point of this
    probe is that PyTorch cannot open these devices. A failed probe comes back as a
    structured unavailable result, so this never raises out of an endpoint.
    """
    rows = _query_gpu_inventory("get_physical_gpu_inventory")
    if rows is None:
        return {
            "available": False,
            "source": "nvidia-smi",
            "devices": [],
            "error": "nvidia-smi did not answer",
        }
    return {
        "available": bool(rows),
        "source": "nvidia-smi",
        "devices": [{**row, "vendor": "nvidia", "source": "nvidia-smi"} for row in rows],
        "error": None,
    }


def get_backend_visible_gpu_info(
    parent_visible_ids: Optional[list[int]], backend_cuda_visible_devices: Optional[str]
) -> dict[str, Any]:
    # parent_visible_ids None (UUID/MIG mask): can't map nvidia-smi rows to
    # visible devices.
    if parent_visible_ids is None:
        return {
            "available": False,
            "backend_cuda_visible_devices": backend_cuda_visible_devices,
            "parent_visible_gpu_ids": [],
            "devices": [],
            "index_kind": "unresolved",
        }
    visible_ordinals = _visible_ordinal_map(parent_visible_ids)
    rows = _query_gpu_inventory("get_backend_visible_gpu_info")
    if rows is None:
        return {
            "available": False,
            "backend_cuda_visible_devices": backend_cuda_visible_devices,
            "parent_visible_gpu_ids": parent_visible_ids or [],
            "devices": [],
            "index_kind": "physical",
        }

    devices = []
    for row in rows:
        idx = row["index"]
        if visible_ordinals is not None and idx not in visible_ordinals:
            continue
        devices.append(
            {
                "index": idx,
                "index_kind": "physical",
                "visible_ordinal": (
                    visible_ordinals[idx] if visible_ordinals is not None else len(devices)
                ),
                "name": row["name"],
                "memory_total_gb": row["memory_total_gb"],
            }
        )

    return {
        "available": len(devices) > 0,
        "backend_cuda_visible_devices": backend_cuda_visible_devices,
        "parent_visible_gpu_ids": parent_visible_ids or [],
        "devices": devices,
        "index_kind": "physical",
    }
