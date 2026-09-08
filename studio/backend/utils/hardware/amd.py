# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AMD GPU monitoring via amd-smi.

Mirrors nvidia.py so hardware.py can swap backends based on IS_ROCM.
All functions return the same dict shapes as their nvidia.py counterparts.
"""

import glob
import json
import math
import os
import platform
import re
import shutil
import stat
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
        # Guard a root-dir prefix (C:\ or /): commonpath would match every path on it. A venv is never at root, so treat
        # that as outside.
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


def _run_amd_smi(
    *args: str,
    timeout: int = _AMD_SMI_DEFAULT_TIMEOUT,
    count_failures: bool = True,
) -> Optional[Any]:
    """Run amd-smi with the given args and return parsed JSON, or None.

    ``count_failures = False`` keeps a failure out of the circuit breaker, for a
    subcommand an older amd-smi rejects outright: that exit code says the CLI is old,
    not that the tool is broken, and three of them must not disable VRAM and
    utilization polling for the life of the process.
    """
    global _amd_smi_consecutive_failures, _amd_smi_disabled
    if _amd_smi_disabled:
        return None
    if not _amd_smi_allowed():
        # Permanently skip amd-smi on Windows without a HIP SDK: every call pops a UAC/DiskPart prompt. VRAM polling is
        # then unavailable, which beats the prompt (UNSLOTH_ENABLE_AMD_SMI=1 opts back in).
        if not _amd_smi_disabled:
            logger.info(
                "amd-smi disabled on Windows (no HIP SDK detected) to avoid a "
                "UAC/DiskPart elevation prompt; GPU VRAM polling unavailable. "
                "Set UNSLOTH_ENABLE_AMD_SMI=1 to force amd-smi."
            )
            _amd_smi_disabled = True
        return None
    if shutil.which("amd-smi") is None:
        # amd-smi does not exist on Windows and can be absent on minimal Linux, so disable the poller in one step
        # instead of burning the 3-strike breaker on guaranteed FileNotFoundError spawns.
        # Unsloth's VRAM display falls back to torch mem_get_info.
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
            encoding = "utf-8",
            errors = "replace",
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
        if not count_failures:
            return None
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
        if not count_failures:
            return None
        _amd_smi_consecutive_failures += 1
        if _amd_smi_consecutive_failures >= _AMD_SMI_FAILURE_LIMIT:
            logger.info(
                "amd-smi not available (not installed; expected on HIP SDK-only systems); "
                "GPU VRAM polling disabled"
            )
            _amd_smi_disabled = True
        return None
    if not result.stdout.strip():
        # Exit 0 with no output
        logger.debug("amd-smi exited 0 but returned no output")
        return None
    _amd_smi_consecutive_failures = 0
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
    # A bytes-above-~10M heuristic was dropped because it misclassified small VRAM allocations; modern amd-smi always
    # ships explicit units.
    return num


def _vram_used_total_mb(gpu_data: dict) -> tuple[Optional[float], Optional[float]]:
    """(used, total) VRAM in MB, unit-aware across amd-smi formats.

    Newer versions use "mem_usage" with "total_vram"/"used_vram"; older use
    "vram" or "fb_memory_usage" with "used"/"total". Shared with the VRAM probe
    so both read the same keys.
    """
    vram_data = gpu_data.get(
        "mem_usage",
        gpu_data.get("vram", gpu_data.get("fb_memory_usage", {})),
    )
    if not isinstance(vram_data, dict):
        return None, None
    used = _parse_memory_mb(
        vram_data.get("used_vram", vram_data.get("vram_used", vram_data.get("used")))
    )
    total = _parse_memory_mb(
        vram_data.get("total_vram", vram_data.get("vram_total", vram_data.get("total")))
    )
    return used, total


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

    vram_used_mb, vram_total_mb = _vram_used_total_mb(gpu_data)

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


def _gpu_entries(data: Any) -> list[tuple[int, dict]]:
    """(physical gpu id, gpu dict) pairs from any amd-smi envelope shape.

    A JSON array, a dict under "gpu_data"/"gpus"/"gpu", or a guarded
    scalar/string fallback. The id is amd-smi's own, falling back to the
    enumeration index when it is missing or unparseable.

    An envelope key only counts when its value is really a list. A single-GPU
    response is a bare dict carrying its own numeric ``"gpu"`` id (the shape
    ``get_primary_gpu_utilization`` also handles); reading that key as the
    envelope yields the id itself, and enumerating an int raises TypeError
    instead of falling back to treating the dict as the one entry.
    """
    if isinstance(data, dict):
        gpu_list: Any = [data]
        for _key in ("gpu_data", "gpus", "gpu"):
            _value = data.get(_key)
            if isinstance(_value, list):
                gpu_list = _value
                break
    elif isinstance(data, list):
        gpu_list = data
    else:
        gpu_list = [data]

    entries: list[tuple[int, dict]] = []
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
        entries.append((idx, gpu_data))
    return entries


def get_gpu_vram_report() -> tuple[dict[int, tuple[int, int]], list[int]]:
    """(``get_gpu_vram_mib()``, every amd-smi gpu id the same call enumerated).

    The ids are amd-smi's own, which are NOT HIP's (see
    ``get_hip_id_by_gpu_index``). The second element is what tells a partial answer
    from a complete one: a device whose VRAM does not parse (a shared pool reporting
    total 0) is missing from the dict but present here, and a caller that ranks GPUs
    has to see the whole set or none of it -- what is left of a dropped row is a
    non-empty dict, indistinguishable from a host that really has one card.
    """
    data = _run_amd_smi("metric")
    if data is None:
        return {}, []
    out: dict[int, tuple[int, int]] = {}
    enumerated: list[int] = []
    for idx, gpu_data in _gpu_entries(data):
        enumerated.append(idx)
        used_mb, total_mb = _vram_used_total_mb(gpu_data)
        if used_mb is None or total_mb is None or total_mb <= 0:
            continue
        # Clamp: a used reading above total
        out[idx] = (int(max(0.0, total_mb - used_mb)), int(total_mb))
    return out, enumerated


def get_gpu_vram_mib() -> dict[int, tuple[int, int]]:
    """{amd-smi gpu id: (free MiB, total MiB)} for every AMD GPU amd-smi sees.

    The out-of-process answer to the question ``torch.cuda.mem_get_info`` answers
    in-process. That call creates a HIP primary context the process never gives
    back (~700 MiB measured), which is pure loss in a backend whose GGUF models
    run in a llama-server child. amd-smi reports used rather than free, so free is
    derived; MiB and MB agree here because ``_parse_memory_mb`` normalises the
    binary units amd-smi reports.

    Empty when amd-smi is missing, disabled, or reports no usable VRAM, so callers
    keep whatever fallback they had.
    """
    return get_gpu_vram_report()[0]


def get_hip_id_by_gpu_index() -> Optional[dict[int, int]]:
    """{amd-smi gpu id: HIP device id}, or None when the mapping is not readable.

    Two index spaces, one number. amd-smi's gpu id is an enumeration index in
    discovery order over its KFD/sysfs view; HIP's is what ``HIP_VISIBLE_DEVICES``
    names and what torch reports as ``cuda:N``, and the library derives it from the
    KFD node id instead (``hip_id = node_id - smallest_node_id``). They coincide on
    most hosts and not on all of them, so the number cannot be carried from one
    space to the other without this call.

    ``amd-smi list -e`` is the mapping AMD publishes for exactly this ("mapping
    physical-to-logical GPU IDs"), added in ROCm 6.4.0. None when any device lacks a
    usable id -- an older CLI rejects ``-e`` outright, and ``hip_id`` reads "N/A"
    when the library cannot reach the device's KFD node -- so callers decline rather
    than assume the identity mapping.
    """
    data = _run_amd_smi("list", "-e", count_failures = False)
    if data is None:
        return None
    mapping: dict[int, int] = {}
    for idx, gpu_data in _gpu_entries(data):
        hip_id = _parse_numeric(gpu_data.get("hip_id"))
        if hip_id is None or hip_id < 0 or hip_id != int(hip_id):
            return None
        mapping[idx] = int(hip_id)
    # A collision means the ids describe something other than a 1:1 device mapping.
    if not mapping or len(set(mapping.values())) != len(mapping):
        return None
    return mapping


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

    # amd-smi may return a list of GPU dicts, a dict wrapping one under "gpu_data", or a single GPU dict.
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

    visible_set = set(parent_visible_ids)
    ordinal_map = {gpu_id: ordinal for ordinal, gpu_id in enumerate(parent_visible_ids)}

    devices = []
    for idx, gpu_data in _gpu_entries(data):
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


# /dev/kfd is what HIP opens; /dev/dri/renderD* is what BOTH HIP and the Vulkan loader
# open, so switching backend is no way around a closed render node.
_KFD_NODE = "/dev/kfd"
_DRI_RENDER_GLOB = "/dev/dri/renderD*"


def _render_node_is_amd(path: str) -> bool:
    """Whether a ``/dev/dri/renderD*`` node belongs to an AMD GPU.

    Render nodes are ``root:render`` for EVERY vendor, so an NVIDIA-only host has
    exactly the same closed nodes and none of the problem -- CUDA opens
    ``/dev/nvidia*`` instead. Without this, the render-group advice below would be
    given to every CUDA user whose probe came back empty for an unrelated reason.
    Read from sysfs, which is world-readable, so the answer does not need the access
    this is testing for.
    """
    vendor_file = f"/sys/class/drm/{os.path.basename(path)}/device/vendor"
    try:
        with open(vendor_file, encoding = "utf-8") as fh:
            return fh.read().strip().lower() == "0x1002"
    except (OSError, UnicodeDecodeError):
        return False


def _kfd_topology_has_an_amd_gpu() -> bool:
    """Whether KFD enumerates an AMD GPU node, so ``/dev/kfd`` is one worth opening.

    Mirrors ``hardware._linux_kfd_reports_an_amd_gpu``: gpu_id 0 is the CPU node, and
    NVIDIA's open kernel module registers KFD nodes of its own under vendor_id 4318,
    so AMD ownership is confirmed rather than assumed.
    """
    nodes = "/sys/class/kfd/kfd/topology/nodes"
    try:
        entries = os.listdir(nodes)
    except OSError:
        return False
    for entry in entries:
        try:
            with open(os.path.join(nodes, entry, "properties"), encoding = "utf-8") as fh:
                properties = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
        if re.search(r"\bvendor_id\s+4098\b", properties):
            return True
    return False


def amd_kfd_gpu_node_count() -> Optional[int]:
    """How many AMD GPU agents KFD enumerates, or ``None`` when that cannot be read.

    The ordinal space a visibility mask indexes: HIP numbers GPU agents, so the CPU node
    every KFD topology carries is excluded twice over -- it reports ``vendor_id 0``, the
    guard install.sh already relies on, and a ``simd_count`` of zero. A node that reports
    no ``simd_count`` at all is counted, since dropping it would understate the bound and
    an understated bound is what calls a valid selector a blocker. Read from
    world-readable sysfs, so it answers on the closed-node host this module exists for.

    ``None`` and 0 are both "unknown" to callers by design -- a topology that is missing,
    unreadable, or reports no GPU bounds nothing, and treating it as a bound would make
    every selector on the host look like it names a device that is not there.
    """
    nodes = "/sys/class/kfd/kfd/topology/nodes"
    try:
        entries = os.listdir(nodes)
    except OSError:
        return None
    count = 0
    for entry in entries:
        try:
            with open(os.path.join(nodes, entry, "properties"), encoding = "utf-8") as fh:
                properties = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
        if not re.search(r"\bvendor_id\s+4098\b", properties):
            continue
        _simd = re.search(r"\bsimd_count\s+(\d+)\b", properties)
        if _simd is None or int(_simd.group(1)) > 0:
            count += 1
    return count


def _amd_render_node_exists() -> bool:
    """Whether any AMD render node is present at all.

    Presence, not openability. A container given ``--device /dev/kfd`` and not
    ``--device /dev/dri`` passes every probe in this file, and ROCr opens a render node
    to talk to amdgpu, so it initialises nothing; ``docker/run.sh`` passes both devices
    for that reason. Group membership cannot create the node, so this is an independent
    blocker rather than part of the permission repair.
    """
    return any(_render_node_is_amd(path) for path in glob.glob(_DRI_RENDER_GLOB))


def an_amd_render_node_is_open() -> bool:
    """Whether this user can open at least one AMD render node.

    The counterpart to amd_nodes_closed_to_this_user, and the reason it is not simply
    "closed is empty": a multi-AMD host can have one node shut and another open, and a
    caller explaining an empty GPU probe needs to know that the runtime had a node to
    use. False off Linux, where there are no such nodes to open.
    """
    if platform.system() != "Linux":
        return False
    for path in sorted(glob.glob(_DRI_RENDER_GLOB)):
        try:
            if not _render_node_is_amd(path):
                continue
            if os.access(path, os.R_OK | os.W_OK):
                return True
        except OSError:
            continue
    return False


def amd_nodes_closed_to_this_user() -> list[str]:
    """AMD device nodes that exist on this host and this user cannot open.

    Every AMD probe in this tree tests that the node EXISTS. On a stock Linux
    distribution these are ``root:render`` mode 0660, so a user outside that group
    passes all of them and then cannot open the device: HIP counts zero devices, the
    Vulkan loader enumerates none, and both look exactly like owning no GPU. That is
    the whole of #10466, where a fresh Strix Halo install ran on CPU until the account
    was added to the render and video groups.

    Only AMD-owned nodes count. Every vendor's render node has these permissions, so
    an NVIDIA box reports the identical closed list and has no such problem.

    ``os.access`` rather than a trial ``open()``: opening ``/dev/kfd`` initialises KFD
    state for the process, and this is called from probes that exist to avoid exactly
    that. Empty off Linux, on a host with no AMD nodes, and for root.
    """
    if platform.system() != "Linux":
        return []
    closed = []
    for path in [_KFD_NODE, *sorted(glob.glob(_DRI_RENDER_GLOB))]:
        try:
            if not os.path.exists(path) or os.access(path, os.R_OK | os.W_OK):
                continue
        except OSError:
            continue
        if path == _KFD_NODE:
            if _kfd_topology_has_an_amd_gpu():
                closed.append(path)
        elif _render_node_is_amd(path):
            closed.append(path)
    return closed


def _has_an_access_acl(path: str) -> bool:
    """Whether ``path`` carries a POSIX access ACL, so its mode bits are the ACL mask.

    Read through the xattr rather than by shelling out to ``getfacl``, which is not
    installed everywhere this runs. False on any platform or filesystem that cannot
    answer, which is the direction that keeps the ordinary node prescribed for.

    os.listxattr returns the names as ``str`` for a ``str`` path, so a bytes literal can
    never match one and the check would be dead. Both are accepted rather than assumed,
    because a bytes path yields bytes names and the caller decides the path type.
    """
    try:
        names = os.listxattr(path)
    except (OSError, AttributeError, UnicodeDecodeError):
        return False
    return any(
        (_n.decode("utf-8", "replace") if isinstance(_n, bytes) else _n)
        == "system.posix_acl_access"
        for _n in names
    )


def _groups_that_own(paths: list) -> tuple:
    """How to open ``paths``, read from the nodes: ``(joinable, unnamed, no_group, acl)``.

    "render,video" is not always the right pair, and sometimes no group is the answer at
    all. Three outcomes, because they need three different repairs:

    ``joinable``   group names whose membership WOULD open the node -- the group has read
                   and write on it, so ``usermod -a -G`` is the fix.
    ``unnamed``    GIDs with no entry in the group database, which is the container case
                   ``docker/run.sh`` documents: ``--group-add`` passes the host's numeric
                   gids and no name inside matches them. Naming a bare GID to usermod does
                   NOT work -- shadow 4.13 answers ``group '993' does not exist`` and exits
                   6 -- so these are reported rather than prescribed.
    ``no_group``   nodes whose mode denies the group too, e.g. a udev rule leaving one
                   ``root:render 0600``. Joining render there changes nothing.
    ``acl``        nodes carrying a POSIX access ACL, where the mode's group bits are the
                   ACL mask and the real grant is undecidable from a stat.

    Best effort by construction: a node that cannot be stat'd contributes to none of the
    three rather than raising, since this runs where things are already wrong.
    """
    joinable, unnamed, no_group, acl = [], [], [], []
    for path in paths:
        try:
            _st = os.stat(path)
        except OSError:
            continue
        # acl(5): once a node carries an access ACL, the group-class bits in st_mode are
        # the ACL MASK rather than the owning group's grant, so every reading below is of
        # an upper bound. The mask can allow rw while the group entry denies it, and a
        # named-group entry can grant what the mode hides. Neither is decidable without
        # parsing the ACL, so such a node is reported rather than prescribed for.
        if _has_an_access_acl(path):
            acl.append(path)
            continue
        # Group read AND write: HIP and the Vulkan loader both open the node read-write,
        # which is the same bar amd_nodes_closed_to_this_user() applied to this account.
        if (_st.st_mode & stat.S_IRGRP) == 0 or (_st.st_mode & stat.S_IWGRP) == 0:
            no_group.append(path)
            continue
        try:
            import grp
            name = grp.getgrgid(_st.st_gid).gr_name
        except Exception:  # noqa: BLE001 -- no group database, or no entry for this gid
            if _st.st_gid not in unnamed:
                unnamed.append(_st.st_gid)
            continue
        if name and name not in joinable:
            joinable.append(name)
    return joinable, unnamed, no_group, acl


def amd_node_permission_hint(*, needs_kfd: bool = True) -> Optional[str]:
    """One sentence naming the closed nodes and the command that opens them, or None.

    Kept beside the probe so the capability message, the llama.cpp log and the
    installer all say the same thing, and so a caller that only needs the yes/no does
    not build a string.

    The two nodes do not block the same backends. A closed render node stops every
    one of them, since HIP and the Vulkan loader both open it. ``/dev/kfd`` stops only
    HIP: Vulkan never opens it, so ``needs_kfd = False`` is how a Vulkan-only caller
    says that a closed KFD node is not its problem, and answering otherwise would send
    a Vulkan failure with some other cause after the wrong repair.
    """
    closed = amd_nodes_closed_to_this_user()
    if not needs_kfd:
        closed = [path for path in closed if path != _KFD_NODE]
    # A missing render node is not a permission problem and does not need a closed node to
    # be worth saying: a container given --device /dev/kfd and not --device /dev/dri opens
    # the one node it has, so the closed set is empty and this returned None with nothing
    # working. Gated on the KFD topology, which is world-readable sysfs and names the
    # vendor, so this cannot fire on a host with no AMD card -- the trap a bare
    # "no render node" test would fall into, since every vendor's nodes live under the
    # same glob.
    _render_missing = not _amd_render_node_exists()
    if not closed:
        if _render_missing and _kfd_topology_has_an_amd_gpu():
            return (
                "This host has an AMD GPU in the KFD topology but no AMD render node "
                "(/dev/dri/renderD*), and ROCm and Vulkan both open one, so the device "
                "mapping needs fixing; under Docker that is --device /dev/kfd "
                "--device /dev/dri."
            )
        return None
    # Claim only what the closed set actually blocks.
    blocked = "no GPU backend can use" if any(p != _KFD_NODE for p in closed) else "ROCm cannot use"
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "$USER"
    joinable, unnamed, no_group, acl = _groups_that_own(closed)
    hint = (
        f"This account cannot open {', '.join(closed)}, so {blocked} the "
        f"AMD card even though the driver is loaded."
    )
    # Prescribed only where joining a group is the repair. A host whose nodes could not be
    # stat'd at all still gets the documented pair, since some advice beats none; a host
    # whose nodes were read and offer no joinable group gets the sentences below instead of
    # a command that would fail.
    if joinable or not (unnamed or no_group or acl):
        groups = joinable or ["render", "video"]
        joined = ",".join(groups)
        plural = "group" if len(groups) == 1 else "groups"
        hint += (
            f" Add the account to the {joined} {plural} and then log out and back in: "
            f"sudo usermod -a -G {joined} {user}"
        )
    if unnamed:
        _gids = ", ".join(str(_g) for _g in unnamed)
        # One flag per GID: docker's --group-add takes a single value, so naming only the
        # first leaves every other node shut on a host whose nodes differ in group.
        _adds = " ".join(f"--group-add {_g}" for _g in unnamed)
        _noun = "GID" if len(unnamed) == 1 else "GIDs"
        _verb = "which has" if len(unnamed) == 1 else "which have"
        hint += (
            f" Some of those nodes belong to {_noun} {_gids}, {_verb} no group entry on "
            f"this system, so usermod cannot name them: create a group with that GID, or "
            f"recreate the container passing {_adds}."
        )
    if no_group:
        hint += (
            f" {', '.join(no_group)} does not grant its own group read and write, so no "
            f"membership opens it: fix the udev rule or the node's permissions."
        )
    if acl:
        hint += (
            f" {', '.join(acl)} carries a POSIX ACL, so the group permissions cannot be "
            f"read from its mode: check the real grant with getfacl {acl[0]} before "
            f"changing group membership."
        )
    # Group membership cannot create a device node. A caller that needs /dev/kfd on a
    # host without one has a second, unrelated problem, and the sentence above is then
    # only true of the render node that was found: the DRM driver is loaded, the ROCm
    # kernel stack is not. install.sh already says both; this is the runtime half.
    if needs_kfd and not os.path.exists(_KFD_NODE):
        hint += (
            " ROCm also needs /dev/kfd, which does not exist on this host, so the ROCm "
            "kernel stack has to be installed as well; the groups alone will not create it."
        )
    # The other half of the same pair, and the container shape of it: /dev/kfd mapped
    # without /dev/dri leaves the closed KFD node looking like the whole story while
    # ROCr has no render node to open. Asked whatever needs_kfd said, since a Vulkan
    # caller needs one too.
    if _render_missing:
        hint += (
            " No AMD render node (/dev/dri/renderD*) is present either, and ROCm and "
            "Vulkan both open one, so the device mapping needs fixing too; under Docker "
            "that is --device /dev/kfd --device /dev/dri."
        )
    return hint
