# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AMD GPU monitoring via amd-smi.

Mirrors the nvidia.py module structure so hardware.py can swap backends
based on IS_ROCM. All functions return the same dict shapes as their
nvidia.py counterparts.
"""

import json
import math
import os
import re
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
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, str):
        # Strip units like "W", "C", "%", "MB", "MiB", "GB", "GiB" etc.
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

    Handles bare numbers (assumed MB -- the amd-smi convention on every
    version we have seen), dict-shaped values with explicit units
    (``{"value": 192, "unit": "GiB"}`` on newer releases), and plain
    strings like ``"8192 MiB"``.
    """
    unit = ""
    raw_value = value

    if isinstance(value, dict):
        unit = str(value.get("unit", "")).strip().lower()
        raw_value = value.get("value")
    elif isinstance(value, str):
        # Extract unit suffix from strings like "192 GiB" or "8192 MB"
        m = re.match(r"^\s*([\d.]+)\s*([A-Za-z]+)\s*$", value.strip())
        if m:
            unit = m.group(2).lower()

    num = _parse_numeric(raw_value if isinstance(value, dict) else value)
    if num is None:
        return None

    # Unit conversion -- GPU tools (including amd-smi) use binary units even
    # when labeling them "GB" or "MB", so treat GB/GiB and MB/MiB the same.
    if "gib" in unit or "gb" in unit:
        return num * 1024
    if "mib" in unit or "mb" in unit:
        return num
    if "kib" in unit or "kb" in unit:
        return num / 1024
    if unit in ("b", "byte", "bytes"):
        # Plain bytes
        return num / (1024 * 1024)

    # No explicit unit -- default to MB, which is the amd-smi convention
    # for bare numeric values. A previous heuristic assumed values above
    # ~10M were bytes, but that misclassifies small VRAM allocations
    # (e.g. 5 MB = 5,242,880 reported without a unit) as ~5 TB. Modern
    # amd-smi always ships explicit units, so the heuristic branch only
    # fired for legacy output where MB was already the convention.
    return num


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

    # Temperature -- try multiple keys in priority order.
    # dict.get() returns "N/A" strings rather than falling through,
    # so we must try each key and check if it parses to a real number.
    temp_data = gpu_data.get("temperature", {})
    temp = None
    if isinstance(temp_data, dict):
        for temp_key in ("edge", "temperature_edge", "hotspot", "temperature_hotspot"):
            temp = _parse_numeric(temp_data.get(temp_key))
            if temp is not None:
                break
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
    # Newer amd-smi uses "mem_usage" with "total_vram" / "used_vram" keys;
    # older versions use "vram" or "fb_memory_usage" with "used" / "total".
    vram_data = gpu_data.get(
        "mem_usage",
        gpu_data.get("vram", gpu_data.get("fb_memory_usage", {})),
    )
    if isinstance(vram_data, dict):
        vram_used_mb = _parse_memory_mb(
            vram_data.get(
                "used_vram", vram_data.get("vram_used", vram_data.get("used"))
            )
        )
        vram_total_mb = _parse_memory_mb(
            vram_data.get(
                "total_vram", vram_data.get("vram_total", vram_data.get("total"))
            )
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
        if vram_used_mb is not None and vram_total_mb is not None and vram_total_mb > 0
        else None
    )
    power_util = (
        round((power_draw / power_limit) * 100, 1)
        if power_draw is not None and power_limit is not None and power_limit > 0
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


def _has_real_metrics(metrics: dict[str, Any]) -> bool:
    """Return True when ``metrics`` contains at least one non-None value.

    ``amd-smi`` can return a zero-exit JSON envelope that is missing every
    expected field (error response, unsupported card, hipless container).
    In that case ``_extract_gpu_metrics`` produces a dict where every value
    is ``None`` -- callers must surface this as ``available: False`` rather
    than ``available: True`` with empty data.
    """
    return any(value is not None for value in metrics.values())


def get_physical_gpu_count() -> Optional[int]:
    """Return physical AMD GPU count via amd-smi, or None on failure."""
    data = _run_amd_smi("list")
    if data is None:
        return None
    if isinstance(data, list):
        return len(data)
    # Some versions return a dict with a "gpu" / "gpus" key. Guard the
    # .get() access with an isinstance check so a malformed scalar /
    # string response from amd-smi cannot raise AttributeError.
    if not isinstance(data, dict):
        return None
    gpus = data.get("gpu", data.get("gpus", []))
    if isinstance(gpus, list):
        return len(gpus)
    return None


def _first_visible_amd_gpu_id() -> Optional[str]:
    """Return the physical AMD GPU id that should be treated as 'primary'.

    Honours HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES
    in that order (HIP respects all three). Returns ``"0"`` when none are
    set, and ``None`` when the env var explicitly narrows to zero GPUs
    ("" or "-1"), so callers can short-circuit to "available: False".
    """
    for env_name in (
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ):
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        raw = raw.strip()
        if raw == "" or raw == "-1":
            return None
        # Filter out empty tokens after splitting. This tolerates minor
        # typos like ``HIP_VISIBLE_DEVICES=",1"`` (leading comma, user
        # clearly meant to narrow to device 1) while still falling
        # through to the next env var when every token is empty
        # (e.g. ``,,,``).
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        if tokens:
            return tokens[0]
    return "0"


def get_primary_gpu_utilization() -> dict[str, Any]:
    """Return utilization metrics for the primary visible AMD GPU."""
    gpu_idx = _first_visible_amd_gpu_id()
    if gpu_idx is None:
        return {"available": False}
    data = _run_amd_smi("metric", "-g", gpu_idx)
    if data is None:
        return {"available": False}

    # amd-smi may return:
    #   - a list of GPU dicts (older versions)
    #   - a dict with a "gpu_data" key wrapping a list (newer versions)
    #   - a single GPU dict (rare)
    if isinstance(data, dict) and "gpu_data" in data:
        data = data["gpu_data"]
    if isinstance(data, list):
        if len(data) == 0:
            return {"available": False}
        gpu_data = data[0]
    else:
        gpu_data = data

    metrics = _extract_gpu_metrics(gpu_data)
    if not _has_real_metrics(metrics):
        # amd-smi returned a JSON envelope with no usable fields (error
        # response or unsupported card). Surface as unavailable rather
        # than available-with-empty-data so the UI does not render a
        # ghost device.
        return {"available": False}
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

    # Extract a device list from amd-smi's envelope. Newer versions return
    # a JSON array directly, older versions return a dict with a "gpus" /
    # "gpu" key wrapping the list. Guard non-dict / non-list envelopes
    # (scalar / string fallbacks from malformed output) so the .get()
    # access cannot raise AttributeError on an unexpected shape.
    if isinstance(data, list):
        gpu_list = data
    elif isinstance(data, dict):
        # Newer amd-smi wraps output in {"gpu_data": [...]}
        gpu_list = data.get("gpu_data", data.get("gpus", data.get("gpu", [data])))
    else:
        gpu_list = [data]
    visible_set = set(parent_visible_ids)
    ordinal_map = {gpu_id: ordinal for ordinal, gpu_id in enumerate(parent_visible_ids)}

    devices = []
    for fallback_idx, gpu_data in enumerate(gpu_list):
        # Skip non-dict entries defensively: if amd-smi ever ships a
        # scalar inside its "gpus" array (observed on some malformed
        # output), _extract_gpu_metrics would raise AttributeError on
        # the first .get() call.
        if not isinstance(gpu_data, dict):
            continue
        # Use AMD-reported GPU ID when available, fall back to enumeration
        # index. Newer amd-smi versions wrap scalars as ``{"value": 0,
        # "unit": "none"}``, so route raw_id through ``_parse_numeric``
        # which already handles bare ints, floats, strings, and that
        # dict shape uniformly.
        raw_id = gpu_data.get(
            "gpu", gpu_data.get("gpu_id", gpu_data.get("id", fallback_idx))
        )
        parsed_id = _parse_numeric(raw_id)
        if parsed_id is None:
            logger.debug(
                "amd-smi GPU id %r could not be parsed; falling back to "
                "enumeration index %d",
                raw_id,
                fallback_idx,
            )
            idx = fallback_idx
        else:
            idx = int(parsed_id)
        if idx not in visible_set:
            continue
        metrics = _extract_gpu_metrics(gpu_data)
        if not _has_real_metrics(metrics):
            # Skip ghost entries: an amd-smi response that decodes to a
            # dict but contains no usable fields (error envelope, etc.)
            # would otherwise show up as a device row with all-None
            # numbers in the UI.
            continue
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
