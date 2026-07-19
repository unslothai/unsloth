# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AMD GPU monitoring via amd-smi.

Mirrors nvidia.py so hardware.py can swap backends based on IS_ROCM.
All functions return the same dict shapes as their nvidia.py counterparts.
"""

import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Optional

from loggers import get_logger
from utils.native_path_leases import child_env_without_native_path_secret
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

logger = get_logger(__name__)

# amd-smi on Windows initialises the full ROCm runtime on first call, which
# can take 15-25 s on cold hardware. Linux is consistently < 2 s.
_AMD_SMI_DEFAULT_TIMEOUT = 30 if platform.system() == "Windows" else 10

# Circuit breaker: stop polling amd-smi after this many consecutive failures
# (each Windows failure may pop a UAC/DiskPart elevation prompt).
_AMD_SMI_FAILURE_LIMIT = 3
_amd_smi_consecutive_failures = 0
_amd_smi_disabled = False


def _path_inside_venv(path: str) -> bool:
    """True if ``path`` is inside the active venv (sys.prefix).

    The venv hipInfo.exe (AMD wheel, put on PATH by main.py/worker.py for
    bitsandbytes) is NOT a HIP SDK (see _hip_sdk_present)."""
    try:
        # realpath (not abspath): resolve symlinks/8.3 names so an aliased venv matches.
        root = os.path.normcase(os.path.realpath(sys.prefix))
        # Guard a root-dir prefix (C:\ or /): commonpath would match every path on
        # it. A venv is never at root, so treat that as outside.
        if os.path.dirname(root) == root:
            return False
        return os.path.normcase(os.path.commonpath([os.path.realpath(path), root])) == root
    except (ValueError, OSError):
        # Different drive / unresolvable -> treat as outside the venv.
        return False


def _external_hipinfo_on_path() -> bool:
    """True if a hipinfo OUTSIDE the venv is on PATH.

    shutil.which returns only the first hit, so the venv hipInfo could shadow a
    real HIP SDK's; scan every PATH entry and skip the venv copy."""
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        directory = directory.strip('"')  # PATH entries can be quoted on Windows
        if not directory:
            continue
        candidate = os.path.join(directory, "hipinfo.exe")
        if os.path.isfile(candidate) and not _path_inside_venv(candidate):
            return True
    return False


def _hip_sdk_present() -> bool:
    """True if a HIP SDK is detectable (hipinfo on PATH or under HIP_PATH/
    ROCM_PATH), so amd-smi has a runtime and runs un-elevated.

    Ignores the venv hipInfo.exe (AMD wheel via the bnb fix): not a HIP SDK, and
    doesn't stop amd-smi's DiskPart UAC."""
    if _external_hipinfo_on_path():
        return True
    for var in ("HIP_PATH", "HIP_PATH_57", "ROCM_PATH"):
        root = os.environ.get(var)
        if not root:
            continue
        candidate = os.path.join(root, "bin", "hipinfo.exe")
        if os.path.exists(candidate) and not _path_inside_venv(candidate):
            return True
    return False


def _amd_smi_allowed() -> bool:
    """Whether it is safe to spawn amd-smi here.

    On Windows without a working HIP runtime, amd-smi elevates a child at
    runtime -- popping a UAC/DiskPart prompt that RunAsInvoker can't suppress
    (its manifest is asInvoker). So only call it on Windows with a HIP SDK
    present or UNSLOTH_ENABLE_AMD_SMI=1. Linux amd-smi never elevates.
    """
    if platform.system() != "Windows":
        return True
    flag = os.environ.get("UNSLOTH_ENABLE_AMD_SMI", "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return True
    if flag in ("0", "false", "no", "off"):
        return False
    return _hip_sdk_present()


def _run_amd_smi(*args: str, timeout: int = _AMD_SMI_DEFAULT_TIMEOUT) -> Optional[Any]:
    """Run amd-smi with the given args and return parsed JSON, or None."""
    global _amd_smi_consecutive_failures, _amd_smi_disabled
    if _amd_smi_disabled:
        return None
    if not _amd_smi_allowed():
        # Permanently skip amd-smi on Windows w/o a HIP SDK: every call would
        # pop a UAC/DiskPart prompt (see _amd_smi_allowed). VRAM polling is then
        # unavailable, but that beats the prompt. Opt back in with
        # UNSLOTH_ENABLE_AMD_SMI=1.
        if not _amd_smi_disabled:
            logger.info(
                "amd-smi disabled on Windows (no HIP SDK detected) to avoid a "
                "UAC/DiskPart elevation prompt; GPU VRAM polling unavailable. "
                "Set UNSLOTH_ENABLE_AMD_SMI=1 to force amd-smi."
            )
            _amd_smi_disabled = True
        return None
    if shutil.which("amd-smi") is None:
        # amd-smi does not exist on Windows (neither Adrenalin nor the HIP SDK
        # ship a CLI) and can be absent on minimal Linux installs. Disable the
        # poller in one step instead of burning the 3-strike circuit breaker
        # on guaranteed FileNotFoundError spawns. Unsloth's VRAM display falls
        # back to torch mem_get_info.
        if not _amd_smi_disabled:
            logger.info(
                "amd-smi not found on PATH; GPU utilization polling via "
                "amd-smi unavailable (VRAM falls back to torch mem_get_info)."
            )
            _amd_smi_disabled = True
        return None
    _amd_env = child_env_without_native_path_secret()
    if platform.system() == "Windows":
        # RunAsInvoker belt-and-suspenders for any manifest-elevating helper;
        # the real guard is _amd_smi_allowed() above. Mirrors install scripts.
        _amd_env = {**_amd_env, "__COMPAT_LAYER": "RunAsInvoker"}
    try:
        result = subprocess.run(
            ["amd-smi", *args, "--json"],
            capture_output = True,
            text = True,
            timeout = timeout,
            env = _amd_env,
            **windows_hidden_subprocess_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        if isinstance(e, FileNotFoundError):
            # Raced a PATH change after the which() check above; absence is
            # expected on Windows (no AMD product ships an amd-smi CLI there).
            logger.debug("amd-smi not found (not in PATH): %s", e)
        else:
            logger.warning("amd-smi query failed: %s", e)
        _amd_smi_consecutive_failures += 1
        if _amd_smi_consecutive_failures >= _AMD_SMI_FAILURE_LIMIT:
            logger.info(
                "amd-smi not available (not installed; expected on HIP SDK-only systems); "
                "GPU VRAM polling disabled"
            )
            _amd_smi_disabled = True
        return None
    if result.returncode != 0:
        logger.warning("amd-smi returned code %d", result.returncode)
        _amd_smi_consecutive_failures += 1
        if _amd_smi_consecutive_failures >= _AMD_SMI_FAILURE_LIMIT:
            logger.info(
                "amd-smi not available (not installed; expected on HIP SDK-only systems); "
                "GPU VRAM polling disabled"
            )
            _amd_smi_disabled = True
        return None
    if not result.stdout.strip():
        # Exit 0 with no output (no GPUs visible, or a version emitting nothing
        # for --json). Not a tool failure, so don't trip the circuit breaker.
        logger.debug("amd-smi exited 0 but returned no output")
        return None
    _amd_smi_consecutive_failures = 0  # reset on success
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        logger.warning("Failed to parse amd-smi JSON output")
        return None


def _parse_numeric(value: Any) -> Optional[float]:
    """Extract a numeric value from amd-smi output (str, int, float, or dict)."""
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
    version seen), dict values with explicit units (``{"value": 192,
    "unit": "GiB"}`` on newer releases), and strings like ``"8192 MiB"``.
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

    # GPU tools use binary units even when labeled "GB"/"MB", so treat GB/GiB
    # and MB/MiB the same.
    if "gib" in unit or "gb" in unit:
        return num * 1024
    if "mib" in unit or "mb" in unit:
        return num
    if "kib" in unit or "kb" in unit:
        return num / 1024
    if unit in ("b", "byte", "bytes"):
        # Plain bytes
        return num / (1024 * 1024)

    # No explicit unit: default to MB (the amd-smi convention for bare numbers).
    # A bytes-above-~10M heuristic was dropped because it misclassified small
    # VRAM allocations; modern amd-smi always ships explicit units.
    return num


def _extract_gpu_metrics(gpu_data: dict) -> dict[str, Any]:
    """Extract standardized metrics from a single GPU's amd-smi data."""
    # Output structure varies by version; try common paths
    usage = gpu_data.get("usage", gpu_data.get("gpu_activity", {}))
    if isinstance(usage, dict):
        gpu_util = _parse_numeric(usage.get("gfx_activity", usage.get("gpu_use_percent")))
    else:
        gpu_util = _parse_numeric(usage)

    # Temperature: try keys in priority order, checking each parses to a real
    # number (dict.get() can return "N/A" strings rather than falling through).
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
        power_limit = _parse_numeric(power_data.get("power_cap", power_data.get("max_power_limit")))
    else:
        power_draw = None
        power_limit = None

    # VRAM: unit-aware parsing across amd-smi formats. Newer versions use
    # "mem_usage" with "total_vram"/"used_vram"; older use "vram" or
    # "fb_memory_usage" with "used"/"total".
    vram_data = gpu_data.get(
        "mem_usage",
        gpu_data.get("vram", gpu_data.get("fb_memory_usage", {})),
    )
    if isinstance(vram_data, dict):
        vram_used_mb = _parse_memory_mb(
            vram_data.get("used_vram", vram_data.get("vram_used", vram_data.get("used")))
        )
        vram_total_mb = _parse_memory_mb(
            vram_data.get("total_vram", vram_data.get("vram_total", vram_data.get("total")))
        )
    else:
        vram_used_mb = None
        vram_total_mb = None

    # Build the standardized dict (same shape as nvidia._build_gpu_metrics)
    vram_used_gb = round(vram_used_mb / 1024, 2) if vram_used_mb is not None else None
    vram_total_gb = round(vram_total_mb / 1024, 2) if vram_total_mb is not None else None
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
    """Return True when ``metrics`` has at least one non-None value.

    amd-smi can return a zero-exit envelope missing every field (error,
    unsupported card, hipless container), yielding an all-None dict; callers must
    surface that as ``available: False``.
    """
    return any(value is not None for value in metrics.values())


def get_physical_gpu_count() -> Optional[int]:
    """Return physical AMD GPU count via amd-smi, or None on failure."""
    data = _run_amd_smi("list")
    if data is None:
        return None
    if isinstance(data, list):
        return len(data)
    # Some versions return a dict with a "gpu"/"gpus" key; guard with isinstance
    # so a malformed scalar/string response can't raise AttributeError.
    if not isinstance(data, dict):
        return None
    gpus = data.get("gpu", data.get("gpus", []))
    if isinstance(gpus, list):
        return len(gpus)
    return None


def _first_visible_amd_gpu_id() -> Optional[str]:
    """Return the physical AMD GPU id treated as 'primary'.

    Honours HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES
    in that order (HIP respects all three). Returns ``"0"`` when none are set,
    and ``None`` when the env var narrows to zero GPUs ("" or "-1"), so callers
    can short-circuit to "available: False".
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
        # Drop empty tokens, tolerating typos like ``",1"`` while still falling
        # through to the next env var when every token is empty (``,,,``).
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
        # Envelope with no usable fields: surface as unavailable so the UI
        # doesn't render a ghost device.
        return {"available": False}
    metrics["available"] = True
    return metrics


def get_visible_gpu_utilization(
    parent_visible_ids: Optional[list[int]], parent_cuda_visible_devices: Optional[str] = None
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

    # Extract a device list across envelope shapes: a JSON array, a dict under
    # "gpu_data"/"gpus"/"gpu", or a guarded scalar/string fallback.
    if isinstance(data, list):
        gpu_list = data
    elif isinstance(data, dict):
        gpu_list = data.get("gpu_data", data.get("gpus", data.get("gpu", [data])))
    else:
        gpu_list = [data]
    visible_set = set(parent_visible_ids)
    ordinal_map = {gpu_id: ordinal for ordinal, gpu_id in enumerate(parent_visible_ids)}

    devices = []
    for fallback_idx, gpu_data in enumerate(gpu_list):
        # Skip non-dict entries (a scalar in the array would raise AttributeError).
        if not isinstance(gpu_data, dict):
            continue
        # Use the AMD-reported GPU ID, else the enumeration index. _parse_numeric
        # handles bare ints/floats/strings and the {"value", "unit"} dict shape.
        raw_id = gpu_data.get("gpu", gpu_data.get("gpu_id", gpu_data.get("id", fallback_idx)))
        parsed_id = _parse_numeric(raw_id)
        if parsed_id is None:
            logger.warning(
                "amd-smi GPU id %r could not be parsed; falling back to enumeration index %d",
                raw_id,
                fallback_idx,
            )
            idx = fallback_idx
        else:
            rounded = round(parsed_id)
            if rounded != parsed_id:
                logger.warning(
                    "amd-smi GPU id %r parsed as non-integer %r; truncating to %d",
                    raw_id,
                    parsed_id,
                    rounded,
                )
            idx = int(rounded)
        if idx not in visible_set:
            continue
        metrics = _extract_gpu_metrics(gpu_data)
        if not _has_real_metrics(metrics):
            # Skip ghost entries (no usable fields) so the UI doesn't show an
            # all-None device row.
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
