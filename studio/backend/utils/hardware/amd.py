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
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import PurePath
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


_AMD_PCI_VENDOR_ID = "0x1002"


def _render_node_vendor(path: str) -> "str | None":
    """The PCI vendor id behind a ``/dev/dri/renderD*`` node, or None if it cannot be read.

    None is not "some other vendor": a container can map the node while masking or not
    mounting the sysfs entry that names it, and the callers answer differently for the two.
    """
    vendor_file = f"/sys/class/drm/{os.path.basename(path)}/device/vendor"
    try:
        with open(vendor_file, encoding = "utf-8") as fh:
            return fh.read().strip().lower()
    except (OSError, UnicodeDecodeError):
        return None


def _render_node_is_amd(path: str) -> bool:
    """Whether a ``/dev/dri/renderD*`` node belongs to an AMD GPU.

    Render nodes are ``root:render`` for EVERY vendor, so an NVIDIA-only host has
    exactly the same closed nodes and none of the problem -- CUDA opens
    ``/dev/nvidia*`` instead. Without this, the render-group advice below would be
    given to every CUDA user whose probe came back empty for an unrelated reason.
    Read from sysfs, which is world-readable, so the answer does not need the access
    this is testing for.
    """
    return _render_node_vendor(path) == _AMD_PCI_VENDOR_ID


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
            # Unknown propagates, exactly as _amd_render_node_exists treats an unreadable
            # vendor. Skipping the entry would answer with a SMALLER count on a multi-GPU
            # host where one topology entry is momentarily unreadable, and an understated
            # bound is the thing this function's own contract says calls a valid selector
            # a blocker: HIP_VISIBLE_DEVICES=1 against a count of 1 reads as hiding every
            # device, and the user is sent to clear a mask that hides nothing.
            return None
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
    _unreadable = False
    for path in glob.glob(_DRI_RENDER_GLOB):
        _vendor = _render_node_vendor(path)
        if _vendor == _AMD_PCI_VENDOR_ID:
            return True
        _unreadable = _unreadable or _vendor is None
    # A node whose vendor could not be READ is not evidence that no AMD node exists, and the
    # caller turns "does not exist" into "recreate the container with --device /dev/dri" --
    # advice for a device that shape has already mapped. Unknown reads as present, which
    # withdraws a sentence rather than inventing one.
    return _unreadable


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


# RADV radeon_icd.x86_64.json, AMDVLK amd_icd64 / amd_pro_icd64 / amdvlk64, Adrenalin
# amd-vulkan64.json. install_llama_prebuilt._AMD_VULKAN_ICD_NEEDLES is the same list for
# the same reason and a test below holds the two together; it is copied rather than
# imported so the inference path does not pull in the installer.
_AMD_VULKAN_ICD_NEEDLES = ("radeon", "radv", "amdvlk", "amd_icd", "amd_pro", "amd_vulkan")

# Mesa and AMDVLK both register a 32-bit manifest beside the 64-bit one, and a 64-bit
# llama-server cannot load either vendor's. Same rule and same needles as
# install_llama_prebuilt._is_amd_64_bit, which rejects them for the same reason; a test
# holds the two lists together.
_VULKAN_ICD_32_BIT_NEEDLES = ("i686", "i386")


def _vulkan_glob_matches(pattern: str, name: str) -> bool:
    """The loader's four driver-filter globs, case-insensitively: "s", "s*", "*s", "*s*"."""
    pattern, name = pattern.lower(), name.lower()
    starts, ends = pattern.startswith("*"), pattern.endswith("*")
    core = pattern[1 if starts else 0 : len(pattern) - 1 if ends else len(pattern)]
    if starts and ends:
        return core in name
    if starts:
        return name.endswith(core)
    if ends:
        return name.startswith(core)
    return name == core


def _vulkan_loader_allows(path: str) -> bool:
    """Whether the loader's own driver filters leave this manifest loadable.

    They apply to every driver the loader knows, a forced list included, and match the
    manifest's basename, so a list naming AMD alone and then disabling it leaves the loader
    with no driver at all. Disable is read before select precisely so "disable everything,
    then name one back" works, hence select answering alone when it is set.

    install_llama_prebuilt._vulkan_loader_allows is the same rule for the same reason; a
    test below runs the two against one table so they cannot drift.
    """

    def _globs(env_name: str) -> "list[str]":
        value = os.environ.get(env_name) or ""
        return [entry.strip() for entry in value.split(",") if entry.strip()]

    name = PurePath(path).name
    select = _globs("VK_LOADER_DRIVERS_SELECT")
    if select:
        return any(_vulkan_glob_matches(pattern, name) for pattern in select)
    disable = _globs("VK_LOADER_DRIVERS_DISABLE")
    return not any(_vulkan_glob_matches(pattern, name) for pattern in disable)


# ld.so's own defaults, plus the multiarch directories Debian and Ubuntu install into.
_DEFAULT_LIBRARY_DIRS = (
    "/lib",
    "/lib64",
    "/usr/lib",
    "/usr/lib64",
    "/usr/local/lib",
    "/usr/local/lib64",
)

_LD_SO_CONF = "/etc/ld.so.conf"

_ld_cache_sonames_cached: "frozenset[str] | None" = None
_ld_cache_read = False


def _ld_so_conf_dirs(path: str = _LD_SO_CONF, _seen: "set[str] | None" = None) -> "list[str]":
    """The extra library directories /etc/ld.so.conf names, include lines followed.

    Read because a driver installed outside the defaults is the ordinary shape for a
    vendor package -- amdgpu-pro puts its libraries under /opt -- and missing that
    directory would make a live driver look like a stale registration.
    """
    _seen = set() if _seen is None else _seen
    if path in _seen:
        return []
    _seen.add(path)
    dirs: "list[str]" = []
    try:
        with open(path, "r", encoding = "utf-8", errors = "replace") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return dirs
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("include"):
            for entry in sorted(glob.glob(line[len("include") :].strip())):
                dirs.extend(_ld_so_conf_dirs(entry, _seen))
            continue
        dirs.append(line)
    return dirs


def _dynamic_loader_search_dirs() -> "list[str]":
    """Where ld.so would look for a bare soname here, in its own order."""
    dirs = [
        entry
        for entry in (os.environ.get("LD_LIBRARY_PATH") or "").split(os.pathsep)
        if entry.strip()
    ]
    dirs.extend(_ld_so_conf_dirs())
    dirs.extend(_DEFAULT_LIBRARY_DIRS)
    for pattern in ("/usr/lib/*-linux-gnu*", "/lib/*-linux-gnu*"):
        dirs.extend(sorted(glob.glob(pattern)))
    return list(dict.fromkeys(dirs))


def _ld_cache_sonames() -> "frozenset[str] | None":
    """Every soname in the loader's cache, or None when the cache cannot be read.

    None is not an empty set: musl ships no `ldconfig -p` and a minimal container may ship
    no ldconfig at all, and answering "nothing is installed" there would call every bare
    registration stale. Read once, since the diagnosis asks it per manifest.
    """
    global _ld_cache_sonames_cached, _ld_cache_read

    if _ld_cache_read:
        return _ld_cache_sonames_cached
    _ld_cache_read = True
    _ld_cache_sonames_cached = None
    for _candidate in ("ldconfig", "/sbin/ldconfig", "/usr/sbin/ldconfig"):
        _exe = shutil.which(_candidate) if "/" not in _candidate else _candidate
        if not _exe or not os.path.exists(_exe):
            continue
        try:
            _out = subprocess.run(
                [_exe, "-p"],
                capture_output = True,
                text = True,
                # Named rather than inherited, as every other call here does: the default
                # is locale.getencoding(), which is ASCII under the C locale a CI runner
                # or container routinely has, and a library path outside it would then
                # raise instead of being read.
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
                **windows_hidden_subprocess_kwargs(),
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if _out.returncode != 0:
            continue
        # "\tlibfoo.so.1 (libc6,x86-64) => /usr/lib/libfoo.so.1"
        _names = {
            line.strip().split(" ", 1)[0]
            for line in (_out.stdout or "").splitlines()
            if "=>" in line and line.strip()
        }
        # Unconditionally, including the empty case: glibc's ldconfig prints
        # "0 libs found in cache" and exits 0 for a cache that is present and empty, and
        # exits 1 with nothing on stdout when the cache file is absent. The non-zero arm
        # above is what separates those, so storing only a non-empty set collapsed the
        # distinction this function's contract is built on -- a fresh container whose cache
        # has not been built read as "cannot enumerate", and _a_bare_soname_resolves then
        # answered True for a soname that is on no search path and in no cache, which
        # withholds the reinstall half of the repair rather than offering it.
        _ld_cache_sonames_cached = frozenset(_names)
        return _ld_cache_sonames_cached
    return _ld_cache_sonames_cached


def _a_bare_soname_resolves(soname: str) -> bool:
    """Whether a manifest's bare library name still resolves to something on this host.

    A manifest may name its library by soname alone and leave the loader to find it, which
    is what NVIDIA's registration does, so a bare name cannot simply be trusted: the
    package can be removed and leave the manifest behind, and the loader then has one fewer
    driver than the registration count suggests.

    Positive evidence in the negative direction as well, since both answers are load
    bearing. Found on disk or in the loader's cache is a driver. NOT found decides the
    question only when the cache could actually be read; a host whose loader configuration
    this cannot enumerate answers True, because calling a live driver stale would demote
    the node hint on a host whose other vendor really does have a path.
    """
    for _directory in _dynamic_loader_search_dirs():
        try:
            if os.path.isfile(os.path.join(_directory, soname)):
                return True
        except OSError:
            continue
    _cache = _ld_cache_sonames()
    if _cache is None:
        return True
    return soname in _cache


def _icd_manifest_is_usable(path: str) -> bool:
    """Whether a manifest still points at a driver library that is there.

    A leftover or malformed JSON is a registration with no device behind it. A bare soname
    is resolved rather than assumed: it is the form NVIDIA registers under, and a removed
    package leaves the manifest behind, so trusting the name counted a driver that is not
    there and withheld the reinstall half of the repair.

    The three fields checked are the three loader_parse_icd_manifest skips the file for:
    a missing or non-string file_format_version, a missing ICD.library_path, and a missing
    or non-string ICD.api_version. An UNRECOGNISED file_format_version is deliberately not
    one of them -- the loader only logs "may cause errors" there and carries on, so
    refusing it would drop a driver the loader loads.
    """
    try:
        with open(path, "r", encoding = "utf-8") as handle:
            manifest = json.load(handle)
        icd = manifest.get("ICD") or {}
        version = manifest.get("file_format_version")
        library = icd.get("library_path")
        api = icd.get("api_version")
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(version, str) or not version.strip():
        return False
    if not isinstance(api, str) or not api.strip():
        return False
    if not isinstance(library, str) or not library.strip():
        return False
    library = library.strip()
    if not (os.path.isabs(library) or "/" in library or "\\" in library):
        return _a_bare_soname_resolves(library)
    if not os.path.isabs(library):
        # Relative to the manifest's directory, per the loader's interface document.
        library = os.path.join(os.path.dirname(path), library)
    try:
        return os.path.isfile(library)
    except OSError:
        return False


def _is_an_amd_icd_name(path: str) -> bool:
    """Whether a manifest's own filename is one an AMD driver registers under."""
    stem = PurePath(path).stem.lower().replace("-", "_")
    return any(needle in stem for needle in _AMD_VULKAN_ICD_NEEDLES)


def _is_a_32_bit_icd_name(path: str) -> bool:
    """Whether a manifest's own filename marks it as the 32-bit build of a driver.

    Asked of EVERY vendor rather than of AMD alone, unlike the installer's copy: the
    question there is "is an AMD driver installed", and here it is "what can this binary
    load", which a 32-bit NVIDIA or Intel manifest answers no to just as squarely.
    """
    stem = PurePath(path).stem.lower().replace("-", "_")
    return stem.endswith("32") or any(n in stem for n in _VULKAN_ICD_32_BIT_NEEDLES)


def _icd_library_path(path: str) -> "str | None":
    """The library file a manifest points at, when it can be found on disk.

    Separate from _icd_manifest_is_usable, which asks whether the loader has SOMETHING to
    load: this asks which file, so the file itself can be read.
    """
    try:
        with open(path, "r", encoding = "utf-8") as handle:
            library = (json.load(handle).get("ICD") or {}).get("library_path")
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(library, str) or not library.strip():
        return None
    library = library.strip()
    if os.path.isabs(library) or "/" in library or "\\" in library:
        if not os.path.isabs(library):
            library = os.path.join(os.path.dirname(path), library)
        try:
            return library if os.path.isfile(library) else None
        except OSError:
            return None
    # Every match, not the first: a multilib host carries the same soname in both
    # bitnesses, ld.so picks the one matching the process, and the search order does not.
    # sorted(glob) puts /usr/lib/i386-linux-gnu ahead of x86_64, so taking the first hit
    # handed the 32-bit copy to _an_icd_is_32_bit and discarded a driver the loader loads.
    _first: "str | None" = None
    for _directory in _dynamic_loader_search_dirs():
        _candidate = os.path.join(_directory, library)
        try:
            if not os.path.isfile(_candidate):
                continue
        except OSError:
            continue
        if _library_file_is_32_bit(_candidate) is False:
            return _candidate
        if _first is None:
            _first = _candidate
    # No 64-bit copy: hand back whatever is there, so a genuinely 32-bit-only
    # registration is still read from the object rather than from its filename.
    return _first


def _icd_manifest_declares_32_bit(path: str) -> "bool | None":
    """The manifest's own architecture claim, or None where it makes none.

    ICD.library_arch is the loader's own field, a string "32" or "64", and the loader reads
    it for exactly this purpose: to skip a driver whose bitness cannot match the process
    before trying to open it. Optional, and Debian strips it back out of Mesa's manifests
    to keep one file across architectures, so its absence is ordinary and decides nothing.
    """
    try:
        with open(path, "r", encoding = "utf-8") as handle:
            declared = (json.load(handle).get("ICD") or {}).get("library_arch")
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(declared, str):
        return None
    declared = declared.strip()
    if declared == "32":
        return True
    if declared == "64":
        return False
    return None


def _library_file_is_32_bit(path: str) -> "bool | None":
    """The ELF class of a library file, or None where the file does not say.

    e_ident[EI_CLASS] is byte 4 of every ELF file and is 1 for 32-bit, 2 for 64-bit. Read
    rather than inferred, so a manifest that declares nothing and is named neutrally is
    still answered by the object itself.
    """
    try:
        with open(path, "rb") as handle:
            header = handle.read(5)
    except OSError:
        return None
    if len(header) < 5 or header[:4] != b"\x7fELF":
        return None
    if header[4] == 1:
        return True
    if header[4] == 2:
        return False
    return None


def _an_icd_is_32_bit(path: str) -> bool:
    """Whether this manifest registers a driver a 64-bit process cannot load.

    Three sources in order of how much they know. The manifest's declared library_arch is
    the loader's own answer. Failing that the library's ELF class is the object's own, which
    covers the case a filename cannot: Mesa's manifests carry no marker once Debian has
    rewritten them. The filename needles are the last resort, and the only one the installer
    has, since it decides this before any library is resolvable.

    A 64-bit process is assumed, which is what Studio ships; on a 32-bit build the question
    inverts, and no build of that shape exists here.
    """
    declared = _icd_manifest_declares_32_bit(path)
    if declared is not None:
        return declared
    library = _icd_library_path(path)
    if library is not None:
        _elf = _library_file_is_32_bit(library)
        if _elf is not None:
            return _elf
    return _is_a_32_bit_icd_name(path)


def _vulkan_icd_search_dirs() -> "list[str]":
    """The icd.d directories the loader would search, in its own order.

    From the XDG variables, since the loader falls back to the defaults only when one is
    unset: reading the defaults regardless both misses a custom layout's only manifest and
    counts stale ones the loader would never read.
    install_llama_prebuilt._vulkan_icd_search_dirs is the same list, and a test holds them
    together.
    """

    def _paths(var: str, default: str) -> "list[str]":
        value = os.environ.get(var)
        raw = value if (value or "").strip() else default
        return [entry for entry in raw.split(os.pathsep) if entry.strip()]

    def _home(var: str, default: str) -> "list[str]":
        value = os.environ.get(var)
        if (value or "").strip():
            return [value]
        try:
            return [os.path.expanduser(os.path.join("~", default))]
        except Exception:  # noqa: BLE001
            return []

    dirs = [
        *(
            os.path.join(base, "vulkan/icd.d")
            for base in (
                *_home("XDG_CONFIG_HOME", ".config"),
                *_paths("XDG_CONFIG_DIRS", "/etc/xdg"),
            )
        ),
        "/etc/vulkan/icd.d",
        *(
            os.path.join(base, "vulkan/icd.d")
            for base in (
                *_home("XDG_DATA_HOME", ".local/share"),
                *_paths("XDG_DATA_DIRS", "/usr/local/share" + os.pathsep + "/usr/share"),
            )
        ),
    ]
    return list(dict.fromkeys(dirs))


def _vulkan_icd_manifest_paths() -> "list[str]":
    """Every ICD manifest the loader knows about here, before its filters are applied.

    A forced list REPLACES the search rather than adding to it, and VK_DRIVER_FILES
    supersedes VK_ICD_FILENAMES rather than joining it. VK_ADD_DRIVER_FILES is the additive
    one: the loader reads it FIRST and then the search, and ignores it entirely when either
    force list is set. It may name a manifest no search directory holds, so leaving it out
    made a host with an added driver look like one that has only what the walk found.

    Linux only, since the render nodes this is asked about exist nowhere else.
    """
    for var in ("VK_DRIVER_FILES", "VK_ICD_FILENAMES"):
        value = (os.environ.get(var) or "").strip()
        if not value:
            continue
        return [entry.strip() for entry in value.split(os.pathsep) if entry.strip()]
    if platform.system() != "Linux":
        return []
    added = (os.environ.get("VK_ADD_DRIVER_FILES") or "").strip()
    paths = [entry.strip() for entry in added.split(os.pathsep) if entry.strip()]
    for directory in _vulkan_icd_search_dirs():
        try:
            paths.extend(sorted(glob.glob(os.path.join(directory, "*.json"))))
        except OSError:
            continue
    return list(dict.fromkeys(paths))


def the_vulkan_loader_can_only_load_amd() -> bool:
    """Whether every driver this loader would actually load is an AMD one.

    Then no other vendor's driver is ever opened, so its render node is not a path this
    binary has however open it is. Three things decide it and all three are the loader's
    own: which manifests it looks at (a forced list, else the search dirs), its driver
    filters, and whether each manifest still resolves to a library.

    POSITIVE evidence only, so both "nothing could be enumerated" and "the filters leave no
    driver at all" answer False. The second is not an oversight: a loader with no driver
    explains an empty probe by itself, and the closed AMD node is then not the cause either.
    """
    loadable = _loadable_icd_manifests()
    if not loadable:
        return False
    return all(_is_an_amd_icd_name(path) for path in loadable)


def _loadable_icd_manifests() -> "list[str]":
    """The manifests the loader would both find here and be able to load."""
    return [
        path
        for path in _vulkan_icd_manifest_paths()
        # A 32-bit manifest is registered beside the 64-bit one and this binary cannot load
        # it, so it is neither evidence of an AMD driver nor of another vendor's.
        if not _an_icd_is_32_bit(path)
        and _vulkan_loader_allows(path)
        and _icd_manifest_is_usable(path)
    ]


def the_vulkan_loader_has_no_usable_driver() -> bool:
    """Whether the loader would find driver manifests here and load none of them.

    A second blocker rather than a competing explanation: a removed library, a filter that
    disables the last driver, or a 32-bit-only registration leaves the probe empty however
    the render node is owned, so opening the node repairs nothing on its own.

    Positive evidence only, and the two failing answers are different. An enumeration that
    found NOTHING says only that this cannot read the loader's configuration -- a registry
    layout, a distribution that registers drivers some other way -- so it answers False. An
    enumeration that found manifests and could load none of them is the claim itself.
    """
    paths = _vulkan_icd_manifest_paths()
    if not paths:
        return False
    return not _loadable_icd_manifests()


def a_non_amd_render_node_is_open() -> bool:
    """Whether a render node belonging to some OTHER vendor is open to this user.

    Vulkan enumerates any vendor, so on a mixed host an open Intel or NVIDIA render node is
    a complete path for a Vulkan-only build: a closed AMD node is then a second finding
    rather than why the probe came back empty. HIP has no such alternative, which is why
    the caller asks this only for Vulkan. Only nodes whose vendor was READ count.
    """
    if platform.system() != "Linux":
        return False
    for path in sorted(glob.glob(_DRI_RENDER_GLOB)):
        _vendor = _render_node_vendor(path)
        if _vendor is None or _vendor == _AMD_PCI_VENDOR_ID:
            continue
        try:
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
    _amd_in_topology = None
    for path in [_KFD_NODE, *sorted(glob.glob(_DRI_RENDER_GLOB))]:
        try:
            if not os.path.exists(path) or os.access(path, os.R_OK | os.W_OK):
                continue
        except OSError:
            continue
        if _amd_in_topology is None:
            _amd_in_topology = _kfd_topology_has_an_amd_gpu()
        if path == _KFD_NODE:
            if _amd_in_topology:
                closed.append(path)
            continue
        _vendor = _render_node_vendor(path)
        if _vendor == _AMD_PCI_VENDOR_ID:
            closed.append(path)
        elif _vendor is None and _amd_in_topology:
            # A container can map the node and hide the sysfs entry that names its vendor.
            # Dropping it there left a host with a shut node reporting nothing closed, while
            # _amd_render_node_exists reads the same unknown as PRESENT and withdraws the
            # missing-node sentence too -- so #10466's own shape got no diagnosis at all.
            # KFD is the independent evidence, and it is what keeps an NVIDIA-only host
            # silent: its topology reports vendor 0x10DE, so this arm is never reached.
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


# Groups whose membership reaches far beyond a device node. Not exhaustive and does not
# need to be: anything here is reported instead of prescribed, and an unlisted group that
# turns out to be privileged is the status quo rather than a regression.
# docker and lxd are here for the same reason as wheel: membership is root by another
# route, since either can start a container or VM with the host filesystem mounted. A node
# owned by one is a udev mistake to report rather than a group to join for a GPU.
_PRIVILEGED_GROUPS = frozenset(
    {
        "root",
        "wheel",
        "sudo",
        "admin",
        "adm",
        "disk",
        "kmem",
        "shadow",
        "docker",
        "lxd",
    }
)


def _groups_that_own(paths: list) -> tuple:
    """How to open ``paths``, read from the nodes themselves.

    "render,video" is not always the right pair, and sometimes no group is the answer at
    all, so the seven buckets returned each carry a different repair:

    ``joinable``   membership WOULD open it, so ``usermod -a -G`` is the fix.
    ``unnamed``    GIDs with no entry in the group database, the container case
                   ``docker/run.sh`` documents: ``--group-add`` passes the host's numeric
                   gids and no name inside matches. usermod refuses a bare GID (shadow
                   4.13: ``group '993' does not exist``, exit 6), so these are reported.
    ``no_group``   the mode denies the group too, e.g. a udev rule leaving one
                   ``root:render 0600``. Joining render there changes nothing.
    ``acl``        a POSIX access ACL, where the mode's group bits are the ACL mask and
                   the real grant is undecidable from a stat.
    ``owned``      this account owns it, and POSIX stops at the owner class once the uid
                   matches, so no membership opens it however the group bits read.
    ``privileged`` the owning group grants far more than the GPU, so this is a udev
                   misconfiguration to report rather than a membership to prescribe.
    ``already``    this account is ALREADY in the group and the node is still shut: a
                   container device cgroup or an LSM denies it, and usermod would exit 0
                   and change nothing.

    Best effort: a node that cannot be stat'd joins no bucket rather than raising, since
    this runs where things are already wrong.
    """
    joinable, unnamed, no_group, acl, owned, privileged, already = ([], [], [], [], [], [], [])
    try:
        # The account's own gids, read once. getgroups() is the supplementary list and does
        # not always include the primary one, so both are needed.
        _mine = {os.getgid(), *os.getgroups()}
    except (OSError, AttributeError):
        _mine = set()
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
        # POSIX resolves the owner class EXCLUSIVELY once the uid matches, so on a node
        # this account owns the group bits are never consulted and joining the group
        # cannot open it. os.access() already said it is shut; the repair is the mode.
        if _st.st_uid == os.getuid():
            owned.append(path)
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
            name = ""
        # Asked of the GID, and BEFORE the name is required, because the lookup above is
        # exactly what fails in a minimal container: with no entry for gid 0 it raised, the
        # node was filed as an ordinary unnamed GID, and the repair became `groupadd -g 0`
        # plus a usermod into the root group -- the grant this branch exists to refuse.
        # gid 0 is the root group whether or not the database names it.
        if _st.st_gid == 0:
            _root = name or "root"
            if _root not in privileged:
                privileged.append(_root)
            continue
        # Joining one of these would open the node and hand over a great deal else with
        # it, so a device node owned by one is a udev misconfiguration to report rather
        # than a membership to prescribe. An unnamed GID cannot match, so this may sit
        # above the naming branches and keep the shell half's single ordering.
        if name in _PRIVILEGED_GROUPS:
            if name not in privileged:
                privileged.append(name)
            continue
        # os.access already said the node is shut, so if this account is in the owning
        # group the group bits are not what is denying it: a container device cgroup or an
        # LSM is. usermod would exit 0 and leave the node exactly as closed. ABOVE the
        # unnamed branch as well, since `groupadd -g` plus `--group-add` is the same empty
        # promise for a numeric owner this account already carries.
        if _st.st_gid in _mine:
            _held = name or str(_st.st_gid)
            if _held not in already:
                already.append(_held)
            continue
        if not name:
            if _st.st_gid not in unnamed:
                unnamed.append(_st.st_gid)
            continue
        if name not in joinable:
            joinable.append(name)
    return joinable, unnamed, no_group, acl, owned, privileged, already


_RENDER_NODE_GLOB = "/dev/dri/renderD*"


def _amd_nodes_the_runtime_lacks(*, needs_kfd: bool = True) -> "list[str]":
    """The AMD device nodes this backend opens that do not exist at all.

    Gated on the KFD topology, which is world-readable sysfs and names the vendor, so this
    cannot fire on a host with no AMD card -- the trap a bare "no render node" test would
    fall into, since every vendor's nodes live under the same glob. A host that cannot show
    the topology either reports nothing rather than guessing.
    """
    if not _kfd_topology_has_an_amd_gpu():
        return []
    lacks = []
    if needs_kfd and not os.path.exists(_KFD_NODE):
        lacks.append(_KFD_NODE)
    if not _amd_render_node_exists():
        lacks.append(_RENDER_NODE_GLOB)
    return lacks


def amd_closed_nodes_block_the_runtime(*, needs_kfd: bool = True) -> bool:
    """Whether the closed nodes actually leave the runtime with no way in.

    A closed node explains an empty GPU probe only when it is a node the runtime would
    have used. On a multi-AMD host one render node can be shut while a sibling is open,
    and ROCm then had /dev/kfd plus a render node and still enumerated nothing, so the
    closed one is a second finding rather than the cause; a caller returning it as the
    sole diagnosis sends the user after a repair that leaves the probe just as empty.

    ``needs_kfd`` is the caller's backend: HIP opens /dev/kfd as well as a render node,
    Vulkan only the render node. /dev/kfd has no sibling, so a closed one blocks outright.

    True on a host this cannot read, which keeps the closed node as the stated reason and
    is what the callers said before this existed.
    """
    # Missing is not open. A node the runtime needs and that does not exist leaves it with
    # no way in exactly as a shut one does, and amd_node_permission_hint() names the repair
    # for it -- so answering only about CLOSED nodes suppressed that hint at every caller,
    # on the two container shapes where it is the whole diagnosis.
    if _amd_nodes_the_runtime_lacks(needs_kfd = needs_kfd):
        return True
    closed = amd_nodes_closed_to_this_user()
    if not closed:
        return False
    if needs_kfd and _KFD_NODE in closed:
        return True
    # The open sibling only answers for a runtime free to use it. A selector narrowing to
    # particular GPUs may well have selected the CLOSED one, and this cannot tell which,
    # so the sibling stops being evidence: fail closed, as this already does for a host it
    # cannot read, rather than suppressing the repair for the node the run will use.
    # HIP's selectors only, so only for a caller that goes through HIP. Vulkan reads none
    # of these -- which is the whole reason needs_kfd exists -- so a Vulkan probe is still
    # free to use the open sibling, and returning the permission hint as its sole cause
    # would send a Vulkan failure after a group change that cannot empty-probe it.
    if needs_kfd and _a_per_gpu_mask_narrows_the_runtime():
        return True
    return not an_amd_render_node_is_open()


def _selector_exposes_every_gpu(
    value: str,
    count: "int | None",
    *,
    repeat_ends_the_list: bool = False,
) -> bool:
    """Whether every measured GPU survives this selector, so it excludes nothing.

    HIP_VISIBLE_DEVICES=0,1 on a two-GPU host is a selector that selects the whole host:
    the open sibling is still reachable, and reading it as a narrowing hands that host the
    group repair in place of the driver diagnosis it needs.

    ``repeat_ends_the_list`` is ROCr's rule, not clr's: RvdFilter terminates on a token
    that "maps to a device that has been previously selected", so ROCR_VISIBLE_DEVICES=0,0,1
    surfaces ONE device, where clr's parser stops only on a token that is not its own index
    written back out and leaves both. _post_rocr_device_count records the same source.

    An unmappable token TERMINATES the list, it does not discard what came before it: clr
    breaks out of the loop having already pushed every device it accepted, so
    HIP_VISIBLE_DEVICES=0,1,-1 on a two-GPU host exposes both of them. So the prefix is
    what decides this, and False for an unreadable count -- a host this cannot measure
    answers nothing, and unmeasured goes on meaning narrowed.
    """
    if not count:
        return False
    seen = set()
    for token in value.split(","):
        token = token.strip()
        try:
            index = int(token)
        except ValueError:
            break
        # clr's own rule: the token has to be the index written back out.
        if str(index) != token or index < 0 or index >= count:
            break
        if index in seen and repeat_ends_the_list:
            break
        seen.add(index)
    return len(seen) == count


def _a_per_gpu_mask_narrows_the_runtime() -> bool:
    """Whether a selector narrows the runtime away from some of this host's AMD GPUs.

    Read for one purpose only: an OPEN render node is an alternative way in only when the
    runtime is free to use it. Nothing here maps a render node back to the index a mask
    selected it by, so under a NARROWING mask the open node may belong to a GPU the mask
    excludes and stops being evidence.

    Read the way the runtime layers them, which is not "all four at once". ROCr is its own
    layer. The HIP layer then reads HIP_VISIBLE_DEVICES when it is non-empty and
    CUDA_VISIBLE_DEVICES otherwise, so a CUDA value under a HIP one is SHADOWED and narrows
    nothing -- the same precedence _explain_empty_gpu_probe's _hip_layer_var applies.
    GPU_DEVICE_ORDINAL is OpenCL's, and independent of both.
    """
    count = amd_kfd_gpu_node_count()
    _hip_layer = (
        "HIP_VISIBLE_DEVICES"
        if os.environ.get("HIP_VISIBLE_DEVICES", "").strip()
        else "CUDA_VISIBLE_DEVICES"
    )
    for _name in (
        "ROCR_VISIBLE_DEVICES",
        _hip_layer,
        # ROCm's fourth visibility variable, modelled elsewhere in this tree
        # (tests/test_amd_smi_inventory_matches_hip.py, llama_cpp.py's own selector
        # check). Omitting it left one of the four selectors crediting a sibling the
        # runtime had been narrowed away from.
        "GPU_DEVICE_ORDINAL",
    ):
        _value = os.environ.get(_name, "").strip()
        if not _value:
            continue
        if not _selector_exposes_every_gpu(
            _value,
            count,
            # ROCr terminates its list on a repeat; clr does not.
            repeat_ends_the_list = _name == "ROCR_VISIBLE_DEVICES",
        ):
            return True
    return False


def _shell_word(value: str) -> str:
    """``value`` as a single shell word, for a command the user is going to paste.

    NSS names are not identifiers: winbind hands back DOMAIN\\user, and a group name may
    carry whitespace or a metacharacter, so interpolating one raw lets the shell de-escape,
    split or expand it -- and usermod then names an account that is not the one holding the
    node shut, or runs something nobody typed under the sudo the line already carries.
    shlex.quote leaves an ordinary name exactly as it was, so the common command is
    unchanged; install.sh's _shell_quote is the same safe set for the same reason.
    """
    if value == "$USER":
        # The placeholder the no-passwd fallback emits on a platform with no pwd module,
        # and the one value here that is meant to be expanded rather than named.
        return value
    return shlex.quote(value)


def _repair_account() -> Optional[str]:
    """The account the usermod commands must name, or None when there is no such account.

    os.getuid() is who os.access answered for above. USER and LOGNAME are inherited, so a
    container that changes its numeric user without resetting them names somebody else, and
    following the hint then modifies an account that is not the one holding the device shut.

    None rather than a guess. A uid with no passwd entry is the ordinary shape of `docker
    run --user 1234`, and there USER commonly still says root: usermod would then succeed,
    change an identity nothing is running as, and leave the nodes exactly as shut. The
    callers print the container-level repair instead, which is the one that works there.
    """
    try:
        import pwd
        return pwd.getpwuid(os.getuid()).pw_name
    except (KeyError, OSError):
        return None
    except (ImportError, AttributeError):
        # No pwd module at all, which is Windows, where none of these nodes exist.
        return os.environ.get("USER") or os.environ.get("LOGNAME") or "$USER"


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
    # A node the runtime needs and that does not exist blocks exactly as hard as one it
    # cannot open, and needs saying whether or not anything is closed: a container given
    # --device /dev/kfd and not --device /dev/dri opens the one node it has and enumerates
    # nothing, and the mirror image (only /dev/dri) leaves HIP with no /dev/kfd. Both used
    # to be reachable only after a closed node had already produced a sentence.
    missing = _amd_nodes_the_runtime_lacks(needs_kfd = needs_kfd)
    parts: "list[str]" = []
    if closed:
        # Claim only what the closed set actually blocks, and only for the devices it is
        # about. A render node shut beside an OPEN sibling leaves both runtimes a complete
        # path to that other GPU, so "no GPU backend can use the AMD card" is false there --
        # and _explain_empty_gpu_probe reaches exactly that host, appending this sentence
        # after saying the closed node is not why the probe is empty.
        user = _repair_account()
        joinable, unnamed, no_group, acl, owned, privileged, already = _groups_that_own(closed)
        if not any(_p != _KFD_NODE for _p in closed):
            _claim = "so ROCm cannot use the AMD card even though the driver is loaded"
        elif an_amd_render_node_is_open():
            _claim = (
                "so no GPU backend can use the card behind them, even though the driver is "
                "loaded and another AMD render node on this host is open"
            )
        else:
            _claim = "so no GPU backend can use the AMD card even though the driver is loaded"
        parts.append(f"This account cannot open {', '.join(closed)}, {_claim}.")
        # Prescribed only where joining a group is the repair. A host whose nodes could not
        # be stat'd at all still gets the documented pair, since some advice beats none; a
        # host whose nodes were read and offer no joinable group gets the sentences below
        # instead of a command that would fail.
        if joinable or not (unnamed or no_group or acl or owned or privileged or already):
            groups = joinable or ["render", "video"]
            joined = ",".join(groups)
            plural = "group" if len(groups) == 1 else "groups"
            if user is None:
                _joins = " ".join(f"--group-add {_shell_word(_g)}" for _g in groups)
                parts.append(
                    f"This uid has no entry in the passwd database, so usermod has no "
                    f"account to name: recreate the container passing {_joins}, or run it "
                    f"as an account this system knows."
                )
            else:
                parts.append(
                    f"Add the account to the {joined} {plural} and then log out and back "
                    f"in: sudo usermod -a -G {_shell_word(joined)} {_shell_word(user)}"
                )
        if unnamed:
            _gids = ", ".join(str(_g) for _g in unnamed)
            # One flag per GID: docker's --group-add takes a single value, so naming only
            # the first leaves every other node shut on a host whose nodes differ in group.
            _adds = " ".join(f"--group-add {_g}" for _g in unnamed)
            _noun = "GID" if len(unnamed) == 1 else "GIDs"
            _verb = "which has" if len(unnamed) == 1 else "which have"
            # Creating the group only gives the numeric owner a name; the account is
            # still outside it and the node is still shut. Both halves of the bare-host
            # repair, and one per GID: with two unnamed GIDs a singular instruction repairs
            # at most one of the nodes.
            _each = "it" if len(unnamed) == 1 else "each of them"
            # A pair per GID, not just the first: the sentence already says "each of them",
            # and one groupadd names one numeric owner, so a host whose nodes differ in group
            # had every node after the first left shut by the command it was told to run.
            # The name is GENERATED rather than a <name> placeholder, because these are
            # commands to paste: angle brackets are redirection operators, so `groupadd -g
            # 993 <name>` is a shell syntax error before groupadd runs. Derived from the GID,
            # which has no group entry by definition here -- which says nothing about the
            # NAME, so the && is load bearing: on a host that already has an amdgpu<GID>
            # group at a different GID, an unchained usermod would SUCCEED against the wrong
            # group and leave the node shut, having reported success.
            _pairs = "; ".join(
                f"sudo groupadd -g {_g} amdgpu{_g} && "
                f"sudo usermod -a -G amdgpu{_g} {_shell_word(user)}"
                for _g in unnamed
            )
            if user is None:
                # The bare-host half needs an account to add and there is none, so the
                # container half is the whole repair for this shape.
                parts.append(
                    f"Some of those nodes belong to {_noun} {_gids}, {_verb} no group entry "
                    f"on this system, and this uid has no passwd entry either, so neither "
                    f"groupadd nor usermod has anything to name: recreate the container "
                    f"passing {_adds}."
                )
            else:
                parts.append(
                    f"Some of those nodes belong to {_noun} {_gids}, {_verb} no group entry "
                    f"on this system, so usermod cannot name them: create a group for "
                    f"{_each}, add the account to it and then log out and back in "
                    f"({_pairs}), or recreate the container passing {_adds}."
                )
        if no_group:
            parts.append(
                f"{', '.join(no_group)} does not grant its own group read and write, so no "
                f"membership opens it: fix the udev rule or the node's permissions."
            )
        if owned:
            parts.append(
                f"{', '.join(owned)} is owned by this account, and POSIX stops at the owner "
                f"bits once the uid matches, so no group membership opens it however its "
                f"group bits read: fix the mode with chmod, or the udev rule that set it."
            )
        if already:
            parts.append(
                f"This account is already in the {', '.join(already)} "
                f"{'group' if len(already) == 1 else 'groups'} that own those nodes, so "
                f"usermod would change nothing: something outside the file mode is denying "
                f"them, typically a container device cgroup or an LSM such as SELinux or "
                f"AppArmor."
            )
        if privileged:
            parts.append(
                f"Those nodes belong to the {', '.join(privileged)} group, which grants a "
                f"great deal besides the GPU, so joining it is not the repair: fix the udev "
                f"rule so the node is owned by render or video instead."
            )
        if acl:
            parts.append(
                # Every path, not acl[0]: the sentence lists them all, ROCm needs all of
                # them, and their ACLs need not agree, so checking only the first can leave
                # the second blocker undiagnosed. getfacl takes several paths.
                f"{', '.join(acl)} carries a POSIX ACL, so the group permissions cannot be "
                f"read from its mode: check the real grant with getfacl {' '.join(acl)} "
                f"before changing group membership."
            )
    # Group membership cannot create a device node, so these stand whether or not anything
    # above was said. install.sh says the same two things; this is the runtime half.
    if _KFD_NODE in missing:
        # NOT "install the ROCm kernel stack". _amd_nodes_the_runtime_lacks only reports a
        # missing node once the KFD topology names an AMD GPU, and that topology is the
        # amdkfd driver's own sysfs -- so on every host this branch can reach, the kernel
        # stack is already loaded and installing it again changes nothing. What is missing
        # is the node in THIS mount namespace, which is the container shape of the problem.
        parts.append(
            "ROCm needs /dev/kfd, which is not present here, but the KFD topology already "
            "names an AMD GPU, so the kernel driver is loaded and reinstalling ROCm "
            "changes nothing: the node itself is missing. Under Docker, recreate the "
            "container with --device /dev/kfd --device /dev/dri; on a bare host it is a "
            "udev or devtmpfs problem. No group membership creates it."
        )
    if _RENDER_NODE_GLOB in missing:
        # The mapping this caller needs, not both nodes always: Vulkan never opens /dev/kfd,
        # so naming it here hands a container another host device for nothing.
        _devices = "--device /dev/kfd --device /dev/dri" if needs_kfd else "--device /dev/dri"
        parts.append(
            f"No AMD render node (/dev/dri/renderD*) is present, and ROCm and Vulkan both "
            f"open one, so the device mapping needs fixing; under Docker that is "
            f"{_devices}."
        )
    return " ".join(parts) or None
