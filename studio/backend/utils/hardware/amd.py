# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AMD GPU monitoring via amd-smi.

Mirrors the nvidia.py module structure so hardware.py can swap backends
based on IS_ROCM. All functions return the same dict shapes as their
nvidia.py counterparts.
"""

import json
import subprocess
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)


def _run_amd_smi(*args: str, timeout: int = 5) -> Optional[Any]:
    """Run amd-smi with the given arguments and return parsed JSON, or None."""
    try:
        result = subprocess.run(
            ["amd-smi", *args, "--json"],
            capture_output = True,
            text = True,
            timeout = timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning("amd-smi query failed: %s", e)
        return None
    if result.returncode != 0 or not result.stdout.strip():
        logger.warning("amd-smi returned code %d", result.returncode)
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        logger.warning("Failed to parse amd-smi JSON output")
        return None


def _parse_numeric(value: Any) -> Optional[float]:
    """Extract a numeric value from amd-smi output (may be str, int, float, or dict)."""
    if value is None:
        return None
    # Newer amd-smi versions emit {"value": 10, "unit": "W"}
    if isinstance(value, dict):
        return _parse_numeric(value.get("value"))
    if isinstance(value, (int, float)):
        import math

        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, str):
        # Strip units like "W", "C", "%", "MB", "MiB", "GB", "GiB" etc.
        import re

        cleaned = re.sub(r"\s*[A-Za-z/%]+$", "", value.strip())
        if not cleaned or cleaned.lower() in ("n/a", "none", "unknown"):
            return None
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return None
    return None


def _parse_memory_mb(value: Any) -> Optional[float]:
    """Parse a memory value from amd-smi output and return MB.

    Handles bare numbers (assumed MB), dict-shaped values with units
    ({"value": 192, "unit": "GiB"}), and byte-scale heuristic fallback.
    """
    unit = ""
    raw_value = value

    if isinstance(value, dict):
        unit = str(value.get("unit", "")).strip().lower()
        raw_value = value.get("value")

    num = _parse_numeric(raw_value if isinstance(value, dict) else value)
    if num is None:
        return None

    # Explicit unit conversion
    if "gib" in unit or "gb" in unit:
        return num * 1024
    if "mib" in unit or "mb" in unit:
        return num
    if "kib" in unit or "kb" in unit:
        return num / 1024
    if unit and (
        "b" in unit and "g" not in unit and "m" not in unit and "k" not in unit
    ):
        # Plain bytes
        return num / (1024 * 1024)

    # No explicit unit -- heuristic: values > 10M are likely bytes
    if num > 10_000_000:
        return num / (1024 * 1024)
    return num  # Assume MB


def _extract_gpu_metrics(gpu_data: dict) -> dict[str, Any]:
    """Extract standardized metrics from a single GPU's amd-smi data."""
    # amd-smi metric output structure varies by version; try common paths
    usage = gpu_data.get("usage", gpu_data.get("gpu_activity", {}))
    if isinstance(usage, dict):
        gpu_util = _parse_numeric(
            usage.get("gfx_activity", usage.get("gpu_use_percent"))
        )
    else:
        gpu_util = _parse_numeric(usage)

    # Temperature
    temp_data = gpu_data.get("temperature", {})
    if isinstance(temp_data, dict):
        temp = _parse_numeric(
            temp_data.get(
                "edge",
                temp_data.get(
                    "temperature_edge",
                    temp_data.get("hotspot", temp_data.get("temperature_hotspot")),
                ),
            )
        )
    else:
        temp = _parse_numeric(temp_data)

    # Power
    power_data = gpu_data.get("power", {})
    if isinstance(power_data, dict):
        power_draw = _parse_numeric(
            power_data.get(
                "current_socket_power",
                power_data.get("average_socket_power", power_data.get("socket_power")),
            )
        )
        power_limit = _parse_numeric(
            power_data.get("power_cap", power_data.get("max_power_limit"))
        )
    else:
        power_draw = None
        power_limit = None

    # VRAM -- unit-aware parsing to handle varying amd-smi output formats.
    # Newer amd-smi versions may return {"value": 192, "unit": "GiB"}.
    vram_data = gpu_data.get("vram", gpu_data.get("fb_memory_usage", {}))
    if isinstance(vram_data, dict):
        vram_used_mb = _parse_memory_mb(
            vram_data.get("vram_used", vram_data.get("used"))
        )
        vram_total_mb = _parse_memory_mb(
            vram_data.get("vram_total", vram_data.get("total"))
        )
    else:
        vram_used_mb = None
        vram_total_mb = None

    # Build the standardized dict (same shape as nvidia._build_gpu_metrics)
    vram_used_gb = round(vram_used_mb / 1024, 2) if vram_used_mb is not None else None
    vram_total_gb = (
        round(vram_total_mb / 1024, 2) if vram_total_mb is not None else None
    )
    vram_util = (
        round((vram_used_mb / vram_total_mb) * 100, 1)
        if vram_used_mb is not None and vram_total_mb and vram_total_mb > 0
        else None
    )
    power_util = (
        round((power_draw / power_limit) * 100, 1)
        if power_draw is not None and power_limit and power_limit > 0
        else None
    )

    return {
        "gpu_utilization_pct": gpu_util,
        "temperature_c": temp,
        "vram_used_gb": vram_used_gb,
        "vram_total_gb": vram_total_gb,
        "vram_utilization_pct": vram_util,
        "power_draw_w": power_draw,
        "power_limit_w": power_limit,
        "power_utilization_pct": power_util,
    }


def get_physical_gpu_count() -> Optional[int]:
    """Return physical AMD GPU count via amd-smi, or None on failure."""
    data = _run_amd_smi("list")
    if data is None:
        return None
    if isinstance(data, list):
        return len(data)
    # Some versions return a dict with a "gpu" key
    gpus = data.get("gpu", data.get("gpus", []))
    if isinstance(gpus, list):
        return len(gpus)
    return None


def get_primary_gpu_utilization() -> dict[str, Any]:
    """Return utilization metrics for the primary AMD GPU."""
    data = _run_amd_smi("metric", "-g", "0")
    if data is None:
        return {"available": False}

    # amd-smi may return a list with one entry or a dict
    if isinstance(data, list):
        if len(data) == 0:
            return {"available": False}
        gpu_data = data[0]
    else:
        gpu_data = data

    metrics = _extract_gpu_metrics(gpu_data)
    metrics["available"] = True
    return metrics


def get_visible_gpu_utilization(
    parent_visible_ids: Optional[list[int]],
    parent_cuda_visible_devices: Optional[str] = None,
) -> dict[str, Any]:
    """Return utilization metrics for visible AMD GPUs."""
    if parent_visible_ids is None:
        return {
            "available": False,
            "backend_cuda_visible_devices": parent_cuda_visible_devices,
            "parent_visible_gpu_ids": [],
            "devices": [],
            "index_kind": "unresolved",
        }

    data = _run_amd_smi("metric")
    if data is None:
        return {
            "available": False,
            "backend_cuda_visible_devices": parent_cuda_visible_devices,
            "parent_visible_gpu_ids": parent_visible_ids or [],
            "devices": [],
            "index_kind": "physical",
        }

    gpu_list = (
        data if isinstance(data, list) else data.get("gpus", data.get("gpu", [data]))
    )
    visible_set = set(parent_visible_ids)
    ordinal_map = {gpu_id: ordinal for ordinal, gpu_id in enumerate(parent_visible_ids)}

    devices = []
    for fallback_idx, gpu_data in enumerate(gpu_list):
        # Use AMD-reported GPU ID when available, fall back to enumeration index
        raw_id = (
            gpu_data.get(
                "gpu", gpu_data.get("gpu_id", gpu_data.get("id", fallback_idx))
            )
            if isinstance(gpu_data, dict)
            else fallback_idx
        )
        try:
            idx = int(raw_id)
        except (TypeError, ValueError):
            idx = fallback_idx
        if idx not in visible_set:
            continue
        metrics = _extract_gpu_metrics(gpu_data)
        metrics["index"] = idx
        metrics["index_kind"] = "physical"
        metrics["visible_ordinal"] = ordinal_map.get(idx, len(devices))
        devices.append(metrics)

    return {
        "available": len(devices) > 0,
        "backend_cuda_visible_devices": parent_cuda_visible_devices,
        "parent_visible_gpu_ids": parent_visible_ids or [],
        "devices": devices,
        "index_kind": "physical",
    }
