# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AMD GPU monitoring via amd-smi. Mirrors nvidia.py so hardware.py can swap backends based on IS_ROCM; all functions return the same dict shapes as their nvidia.py counterparts."""

import copy
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
import threading
import time
from pathlib import PurePath
from typing import Any, Optional

from loggers import get_logger
from utils.native_path_leases import child_env_without_native_path_secret
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

logger = get_logger(__name__)

# amd-smi on Windows initialises the full ROCm runtime on first call, which can take 15-25 s on cold hardware. Linux is consistently < 2 s.
_AMD_SMI_DEFAULT_TIMEOUT = 30 if platform.system() == "Windows" else 10

# Circuit breaker: stop polling amd-smi after this many consecutive failures (each Windows failure may pop a UAC/DiskPart elevation prompt).
_AMD_SMI_FAILURE_LIMIT = 3
_amd_smi_consecutive_failures = 0
_amd_smi_disabled = False


def _amd_smi_allowed() -> bool:
    """Windows is opt-in only (UNSLOTH_ENABLE_AMD_SMI=1): amd-smi.exe pops a UAC/DiskPart prompt RunAsInvoker cannot suppress, even with a HIP SDK."""
    if platform.system() != "Windows":
        return True
    flag = os.environ.get("UNSLOTH_ENABLE_AMD_SMI", "").strip().lower()
    return flag in ("1", "true", "yes", "on")


def _run_amd_smi(
    *args: str,
    timeout: int = _AMD_SMI_DEFAULT_TIMEOUT,
    count_failures: bool = True,
) -> Optional[Any]:
    """Run amd-smi with the given args and return parsed JSON, or None. ``count_failures = False`` keeps a failure out of the circuit breaker, for a subcommand an older amd-smi rejects outright: that exit code says the CLI is old, not that the tool is broken, and three of them must not disable VRAM and utilization polling for the life of the process."""
    global _amd_smi_consecutive_failures, _amd_smi_disabled
    if _amd_smi_disabled:
        return None
    if not _amd_smi_allowed():
        if not _amd_smi_disabled:
            logger.info(
                "amd-smi disabled on Windows to avoid a UAC/DiskPart elevation "
                "prompt; VRAM comes from the adapter perf counters. "
                "Set UNSLOTH_ENABLE_AMD_SMI=1 to force amd-smi."
            )
            _amd_smi_disabled = True
        return None
    if shutil.which("amd-smi") is None:
        # amd-smi does not exist on Windows and can be absent on minimal Linux, so disable the poller in one step instead of burning the 3-strike breaker on guaranteed FileNotFoundError spawns; Unsloth's VRAM display falls back to torch mem_get_info.
        if not _amd_smi_disabled:
            logger.info(
                "amd-smi not found on PATH; GPU utilization polling via "
                "amd-smi unavailable (VRAM falls back to torch mem_get_info)."
            )
            _amd_smi_disabled = True
        return None
    _amd_env = child_env_without_native_path_secret()
    if platform.system() == "Windows":
        # RunAsInvoker belt-and-suspenders for any manifest-elevating helper; the real guard is _amd_smi_allowed() above. Mirrors install scripts.
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
            # Raced a PATH change after the which() check above; absence is expected on Windows (no AMD product ships an amd-smi CLI there).
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
        if platform.system() == "Windows":
            # System32 amd-smi.exe exits 0 with no stdout when it can't run, prompting each spawn.
            logger.info(
                "amd-smi exited 0 with no output (%s); GPU utilization polling via amd-smi disabled",
                (result.stderr or "").strip().splitlines()[-1]
                if (result.stderr or "").strip()
                else "no stderr",
            )
            _amd_smi_disabled = True
        else:
            logger.debug("amd-smi exited 0 but returned no output")
        return None
    _amd_smi_consecutive_failures = 0
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        logger.warning("Failed to parse amd-smi JSON output")
        return None


_display_lock = threading.Lock()
_display_cache: dict[tuple[str, ...], tuple[float, int, Any]] = {}


def _run_amd_smi_for_display(*args: str) -> Optional[Any]:
    """``_run_amd_smi`` under gpu_query's display rules; the lock coalesces pollers, failures are never cached."""
    try:
        from utils import gpu_memory_events
        from utils.hardware import gpu_query

        use_cache = gpu_query.display_cache_allowed()
        ttl = gpu_query.ttl_for(gpu_query.DISPLAY)
    except Exception:
        use_cache = False
    if not use_cache:
        return _run_amd_smi(*args)
    with _display_lock:
        cached = _display_cache.get(args)
        if (
            cached is not None
            and time.monotonic() - cached[0] < ttl
            and cached[1] == gpu_memory_events.generation()
        ):
            return copy.deepcopy(cached[2])
        generation = gpu_memory_events.generation()
        data = _run_amd_smi(*args)
        if data is None:
            _display_cache.pop(args, None)
        else:
            _display_cache[args] = (time.monotonic(), generation, copy.deepcopy(data))
        return data


def _parse_numeric(value: Any) -> Optional[float]:
    """Extract a numeric value from amd-smi output (str, int, float, or dict)."""
    if value is None:
        return None
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
    """Parse a memory value from amd-smi output and return MB. Handles bare numbers (assumed MB, the amd-smi convention on every version seen), dict values with explicit units (``{"value": 192, "unit": "GiB"}`` on newer releases), and strings like ``"8192 MiB"``."""
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

    # GPU tools use binary units even when labeled "GB"/"MB", so treat GB/GiB and MB/MiB the same.
    if "gib" in unit or "gb" in unit:
        return num * 1024
    if "mib" in unit or "mb" in unit:
        return num
    if "kib" in unit or "kb" in unit:
        return num / 1024
    if unit in ("b", "byte", "bytes"):
        return num / (1024 * 1024)

    # No explicit unit: default to MB (the amd-smi convention for bare numbers). A bytes-above-~10M heuristic was dropped because it misclassified small VRAM allocations; modern amd-smi always ships explicit units.
    return num


def _vram_used_total_mb(gpu_data: dict) -> tuple[Optional[float], Optional[float]]:
    """(used, total) VRAM in MB, unit-aware across amd-smi formats. Newer versions use "mem_usage" with "total_vram"/"used_vram"; older use "vram" or "fb_memory_usage" with "used"/"total". Shared with the VRAM probe so both read the same keys."""
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
    usage = gpu_data.get("usage", gpu_data.get("gpu_activity", {}))
    if isinstance(usage, dict):
        gpu_util = _parse_numeric(usage.get("gfx_activity", usage.get("gpu_use_percent")))
    else:
        gpu_util = _parse_numeric(usage)

    # Temperature: try keys in priority order, checking each parses to a real number (dict.get() can return "N/A" strings rather than falling through).
    temp_data = gpu_data.get("temperature", {})
    temp = None
    if isinstance(temp_data, dict):
        for temp_key in ("edge", "temperature_edge", "hotspot", "temperature_hotspot"):
            temp = _parse_numeric(temp_data.get(temp_key))
            if temp is not None:
                break
    else:
        temp = _parse_numeric(temp_data)

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
    """Return True when ``metrics`` has at least one non-None value. amd-smi can return a zero-exit envelope missing every field (error, unsupported card, hipless container), yielding an all-None dict; callers must surface that as ``available: False``."""
    return any(value is not None for value in metrics.values())


def get_physical_gpu_count() -> Optional[int]:
    """Return physical AMD GPU count via amd-smi, or None on failure."""
    data = _run_amd_smi("list")
    if data is None:
        return None
    if isinstance(data, list):
        return len(data)
    # Some versions return a dict with a "gpu"/"gpus" key; guard with isinstance so a malformed scalar/string response can't raise AttributeError.
    if not isinstance(data, dict):
        return None
    gpus = data.get("gpu", data.get("gpus", []))
    if isinstance(gpus, list):
        return len(gpus)
    return None


def _gpu_entries(data: Any) -> list[tuple[int, dict]]:
    """(physical gpu id, gpu dict) pairs from any amd-smi envelope shape: a JSON array, a dict under "gpu_data"/"gpus"/"gpu", or a guarded scalar/string fallback. The id is amd-smi's own, falling back to the enumeration index when it is missing or unparseable. An envelope key only counts when its value is really a list: a single-GPU response is a bare dict carrying its own numeric ``"gpu"`` id (the shape ``get_primary_gpu_utilization`` also handles), and reading that key as the envelope yields the id itself, so enumerating an int raises TypeError instead of falling back to treating the dict as the one entry."""
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
        if not isinstance(gpu_data, dict):
            continue
        # Use the AMD-reported GPU ID, else the enumeration index. _parse_numeric handles bare ints/floats/strings and the {"value", "unit"} dict shape.
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
    """(``get_gpu_vram_mib()``, every amd-smi gpu id the same call enumerated). The ids are amd-smi's own, which are NOT HIP's (see ``get_hip_id_by_gpu_index``). The second element is what tells a partial answer from a complete one: a device whose VRAM does not parse (a shared pool reporting total 0) is missing from the dict but present here, and a caller that ranks GPUs has to see the whole set or none of it, since what is left of a dropped row is a non-empty dict, indistinguishable from a host that really has one card."""
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
        out[idx] = (int(max(0.0, total_mb - used_mb)), int(total_mb))
    return out, enumerated


def get_gpu_vram_mib() -> dict[int, tuple[int, int]]:
    """{amd-smi gpu id: (free MiB, total MiB)} for every AMD GPU amd-smi sees. The out-of-process answer to the question ``torch.cuda.mem_get_info`` answers in-process: that call creates a HIP primary context the process never gives back (~700 MiB measured), which is pure loss in a backend whose GGUF models run in a llama-server child. amd-smi reports used rather than free, so free is derived; MiB and MB agree here because ``_parse_memory_mb`` normalises the binary units amd-smi reports. Empty when amd-smi is missing, disabled, or reports no usable VRAM, so callers keep whatever fallback they had."""
    return get_gpu_vram_report()[0]


_HIP_ID_MAP_NONE_TTL_S = 300.0
_hip_id_map_lock = threading.Lock()
_hip_id_map_cache: Optional[
    tuple[float, Optional[dict[int, int]], Optional[int], Optional[tuple]]
] = None


def get_hip_id_by_gpu_index() -> Optional[dict[int, int]]:
    """{amd-smi gpu id: HIP device id}, or None when the mapping is not readable. Two index spaces, one number: amd-smi's gpu id is an enumeration index in discovery order over its KFD/sysfs view, while HIP's is what ``HIP_VISIBLE_DEVICES`` names and what torch reports as ``cuda:N``, derived from the KFD node id instead (``hip_id = node_id - smallest_node_id``). They coincide on most hosts and not on all of them, so the number cannot be carried from one space to the other without this call. ``amd-smi list -e`` is the mapping AMD publishes for exactly this ("mapping physical-to-logical GPU IDs"), added in ROCm 6.4.0. None when any device lacks a usable id (an older CLI rejects ``-e`` outright, and ``hip_id`` reads "N/A" when the library cannot reach the device's KFD node) so callers decline rather than assume the identity mapping. Cached until hardware re-detection; an unreadable answer is retried after five minutes."""
    global _hip_id_map_cache
    static_gen = _static_generation()
    topology = _amd_pci_topology()
    with _hip_id_map_lock:
        cached = _hip_id_map_cache
        if (
            cached is None
            or cached[2] != static_gen
            # Cards changed, or the first readable walk (no baseline); an unreadable walk keeps the map.
            or (topology is not None and cached[3] != topology)
            or (cached[1] is None and time.monotonic() - cached[0] >= _HIP_ID_MAP_NONE_TTL_S)
        ):
            cached = _hip_id_map_cache = (
                time.monotonic(),
                _read_hip_id_by_gpu_index(),
                static_gen,
                topology,
            )
    return None if cached[1] is None else dict(cached[1])


def invalidate_hip_id_map() -> None:
    """Drop the HIP id map, under its lock so a read in flight cannot land after the clear."""
    global _hip_id_map_cache
    with _hip_id_map_lock:
        _hip_id_map_cache = None


def _static_generation() -> Optional[int]:
    """gpu_query's static generation, bumped by hardware re-detection (a swapped or re-seated card)."""
    try:
        from utils.hardware import gpu_query
        return gpu_query._static_gen
    except Exception:
        return None


def _amd_pci_topology() -> Optional[tuple]:
    try:
        from utils.hardware.hardware import _linux_amd_pci_topology
        return _linux_amd_pci_topology()
    except Exception:
        return None


def _read_hip_id_by_gpu_index() -> Optional[dict[int, int]]:
    data = _run_amd_smi("list", "-e", count_failures = False)
    if data is None:
        return None
    mapping: dict[int, int] = {}
    for idx, gpu_data in _gpu_entries(data):
        hip_id = _parse_numeric(gpu_data.get("hip_id"))
        if hip_id is None or hip_id < 0 or hip_id != int(hip_id):
            return None
        mapping[idx] = int(hip_id)
    if not mapping or len(set(mapping.values())) != len(mapping):
        return None
    return mapping


def _first_visible_amd_gpu_id() -> Optional[str]:
    """Return the physical AMD GPU id treated as 'primary'. Honours HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES in that order (HIP respects all three). Returns ``"0"`` when none are set, and ``None`` when the env var narrows to zero GPUs ("" or "-1"), so callers can short-circuit to "available: False"."""
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
        # Drop empty tokens, tolerating typos like ``",1"`` while still falling through to the next env var when every token is empty (``,,,``).
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        if tokens:
            return tokens[0]
    return "0"


def get_primary_gpu_utilization() -> dict[str, Any]:
    """Return utilization metrics for the primary visible AMD GPU."""
    gpu_idx = _first_visible_amd_gpu_id()
    if gpu_idx is None:
        return {"available": False}
    data = _run_amd_smi_for_display("metric", "-g", gpu_idx)
    if data is None:
        return {"available": False}

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
        # Envelope with no usable fields: surface as unavailable so the UI doesn't render a ghost device.
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

    data = _run_amd_smi_for_display("metric")
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
            # Skip ghost entries (no usable fields) so the UI doesn't show an all-None device row.
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


# Both HIP and the Vulkan loader open renderD*, so switching backend does not help.
_KFD_NODE = "/dev/kfd"
_DRI_RENDER_GLOB = "/dev/dri/renderD*"


_AMD_PCI_VENDOR_ID = "0x1002"


def _render_node_vendor(path: str) -> "str | None":
    """None means unreadable, not another vendor, since a container can mask the sysfs entry."""
    vendor_file = f"/sys/class/drm/{os.path.basename(path)}/device/vendor"
    try:
        with open(vendor_file, encoding = "utf-8") as fh:
            return fh.read().strip().lower()
    except (OSError, UnicodeDecodeError):
        return None


def _render_node_is_amd(path: str) -> bool:
    """Render nodes are root:render for every vendor, so only AMD nodes may earn the render-group advice."""
    return _render_node_vendor(path) == _AMD_PCI_VENDOR_ID


def _kfd_topology_amd_state() -> "bool | None":
    """None when unreadable, which is no evidence either way; a container may hide the sysfs entry."""
    nodes = "/sys/class/kfd/kfd/topology/nodes"
    try:
        entries = os.listdir(nodes)
    except OSError:
        return None
    _read_one = False
    _missed_one = False
    for entry in entries:
        try:
            with open(os.path.join(nodes, entry, "properties"), encoding = "utf-8") as fh:
                properties = fh.read()
        except (OSError, UnicodeDecodeError):
            _missed_one = True
            continue
        _read_one = True
        if re.search(r"\bvendor_id\s+4098\b", properties):
            return True
    # A partial read is unknown, not False; install.sh's _kfd_gfx_targets uses the same rule.
    return None if _missed_one or not _read_one else False


def _a_confirmed_amd_render_node_exists() -> bool:
    """Requires the vendor to be read: an unknown vendor would let an NVIDIA-only host pass as AMD."""
    return any(_render_node_is_amd(path) for path in glob.glob(_DRI_RENDER_GLOB))


def _kfd_topology_has_an_amd_gpu() -> bool:
    """Mirrors hardware's KFD check: NVIDIA's KFD nodes (vendor_id 4318) must not count as AMD."""
    return _kfd_topology_amd_state() is True


def amd_kfd_gpu_node_count() -> Optional[int]:
    """None and 0 both mean unknown: an unreadable or GPU-less topology must not bound valid selectors."""
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
            # Unknown propagates: skipping would understate the count and flag a valid selector.
            return None
        if not re.search(r"\bvendor_id\s+4098\b", properties):
            continue
        _simd = re.search(r"\bsimd_count\s+(\d+)\b", properties)
        if _simd is None or int(_simd.group(1)) > 0:
            count += 1
    return count


def amd_kfd_gpu_gfx_targets(
    nodes: str = "/sys/class/kfd/kfd/topology/nodes",
) -> Optional[list[str]]:
    """The gfx target of each AMD GPU agent KFD enumerates, in HIP device order, or ``None``
    when the topology cannot be read. HIP numbers GPU agents by KFD node id, so position N is
    physical device N. ``gfx_target_version`` is written by amdkfd itself, so it is immune to
    HSA_OVERRIDE_GFX_VERSION; it encodes major * 10000 + minor * 100 + stepping, the stepping
    in hex (110501 is gfx1151, 90010 is gfx90a), as install_python_stack._kfd_gfx_targets reads
    it. Unlike that one, an unreadable node fails the whole answer, since it would shift ordinals."""
    try:
        entries = sorted((int(e), e) for e in os.listdir(nodes) if e.isdigit())
    except OSError:
        return None
    targets = []
    for _node, entry in entries:
        try:
            with open(os.path.join(nodes, entry, "properties"), encoding = "utf-8") as fh:
                properties = fh.read()
        except (OSError, UnicodeDecodeError):
            return None
        simd = re.search(r"\bsimd_count\s+(\d+)\b", properties)
        if not re.search(r"\bvendor_id\s+4098\b", properties) or (
            simd is not None and int(simd.group(1)) == 0
        ):
            continue
        version = re.search(r"\bgfx_target_version\s+(\d+)\b", properties)
        raw = int(version.group(1)) if version else 0
        major, minor, step = raw // 10000, (raw // 100) % 100, raw % 100
        targets.append(
            f"gfx{major}{minor}{step:x}" if raw > 0 and minor <= 9 and step <= 15 else ""
        )
    return targets


def _amd_render_node_exists() -> bool:
    """Presence, not openability: /dev/kfd without /dev/dri passes every probe yet initialises nothing."""
    _unreadable = False
    for path in glob.glob(_DRI_RENDER_GLOB):
        _vendor = _render_node_vendor(path)
        if _vendor == _AMD_PCI_VENDOR_ID:
            return True
        _unreadable = _unreadable or _vendor is None
    # An unreadable vendor counts as present, so no wrong --device advice is given.
    return _unreadable


def an_amd_render_node_is_open() -> bool:
    """Not just closed-is-empty: a multi-AMD host can shut one render node while another is open."""
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


# Keep in sync with install_llama_prebuilt._AMD_VULKAN_ICD_NEEDLES (copied, not imported).
_AMD_VULKAN_ICD_NEEDLES = ("radeon", "radv", "amdvlk", "amd_icd", "amd_pro", "amd_vulkan")

# A 64-bit llama-server cannot load 32-bit ICDs; same needles as install_llama_prebuilt.
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
    """VK_LOADER_DRIVERS_DISABLE beats the select list: a driver both selected and disabled never loads."""

    def _globs(env_name: str) -> "list[str]":
        value = os.environ.get(env_name) or ""
        return [entry.strip() for entry in value.split(",") if entry.strip()]

    name = PurePath(path).name
    disable = _globs("VK_LOADER_DRIVERS_DISABLE")
    if any(_vulkan_glob_matches(pattern, name) for pattern in disable):
        return False
    select = _globs("VK_LOADER_DRIVERS_SELECT")
    return any(_vulkan_glob_matches(pattern, name) for pattern in select) if select else True


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
    """ld.so.conf library dirs, includes followed: vendor drivers install outside the defaults."""
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
    return list(dict.fromkeys(dirs + _system_library_dirs()))


def _system_library_dirs() -> "list[str]":
    """The loader's search directories that do not come from LD_LIBRARY_PATH."""
    dirs = list(_ld_so_conf_dirs())
    dirs.extend(_DEFAULT_LIBRARY_DIRS)
    for pattern in ("/usr/lib/*-linux-gnu*", "/lib/*-linux-gnu*"):
        dirs.extend(sorted(glob.glob(pattern)))
    return list(dict.fromkeys(dirs))


def _ld_cache_sonames() -> "frozenset[str] | None":
    """None when the cache is unreadable, not empty: that would make every bare registration look stale."""
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
                # Explicit utf-8: the default is ASCII under the C locale and would raise on a path.
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
        # Store even an empty set: an empty cache (exit 0) differs from an absent one (exit 1).
        _ld_cache_sonames_cached = frozenset(_names)
        return _ld_cache_sonames_cached
    return _ld_cache_sonames_cached


def _a_bare_soname_resolves(
    soname: str,
    *,
    with_ld_library_path: bool = True,
    when_unknown: bool = True,
) -> bool:
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

    ``with_ld_library_path = False`` asks for a child that does not inherit LD_LIBRARY_PATH;
    ``when_unknown`` is the answer when the cache cannot be read (False for a hard requirement).
    """
    for _directory in (
        _dynamic_loader_search_dirs() if with_ld_library_path else _system_library_dirs()
    ):
        try:
            if os.path.isfile(os.path.join(_directory, soname)):
                return True
        except OSError:
            continue
    _cache = _ld_cache_sonames()
    if _cache is None:
        return when_unknown
    return soname in _cache


def _icd_manifest_is_usable(path: str) -> bool:
    """Usable if its JSON fields are valid and its library resolves; unknown file_format_version passes."""
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
    """Applies to every vendor, not just AMD: a 32-bit NVIDIA or Intel manifest cannot load here either."""
    stem = PurePath(path).stem.lower().replace("-", "_")
    return stem.endswith("32") or any(n in stem for n in _VULKAN_ICD_32_BIT_NEEDLES)


def _icd_library_path(path: str) -> "str | None":
    """The library a manifest points at, if on disk; asks which file, so the file itself can be read."""
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
    # Every match: multilib hosts carry both bitnesses and sorted glob puts i386 first.
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
    # No 64-bit copy: return whatever exists, so a 32-bit-only registration is read from the object.
    return _first


def _icd_manifest_declares_32_bit(path: str) -> "bool | None":
    """The optional ICD.library_arch claim, or None: Debian strips it, so absence decides nothing."""
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
    """e_ident[EI_CLASS] byte: 1 is 32-bit, 2 is 64-bit; read from the object itself."""
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
    """Checks library_arch, then the ELF class, then filename needles; assumes a 64-bit process."""
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
    """icd.d dirs from the XDG variables, falling back to defaults only when unset, as the loader does."""

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
    """VK_DRIVER_FILES or VK_ICD_FILENAMES replaces the search; VK_ADD_DRIVER_FILES is ignored then."""
    for var in ("VK_DRIVER_FILES", "VK_ICD_FILENAMES"):
        if (os.environ.get(var) or "").strip():
            return _forced_icd_manifest_paths(var)
    return _searched_vulkan_icd_manifest_paths()


def _searched_vulkan_icd_manifest_paths() -> "list[str]":
    """Search plus additive list, the half a forced list replaces; split out for the attribution check."""
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
    """Positive evidence only: at least one loadable manifest, and every loadable one is AMD."""
    loadable = _loadable_icd_manifests()
    if not loadable:
        return False
    return all(_is_an_amd_icd_name(path) for path in loadable)


def _loadable_icd_manifests(paths: "list[str] | None" = None) -> "list[str]":
    """Manifests the loader would find and load here; paths checks a candidate list instead."""
    return [
        path
        for path in (_vulkan_icd_manifest_paths() if paths is None else paths)
        # A 32-bit manifest is unloadable here, so it is neither evidence of AMD nor of another vendor.
        if not _an_icd_is_32_bit(path)
        and _vulkan_loader_allows(path)
        and _icd_manifest_is_usable(path)
    ]


def the_vulkan_loader_has_no_usable_driver() -> bool:
    """Manifests were found but none is loadable; finding none at all answers False, not True."""
    paths = _vulkan_icd_manifest_paths()
    if not paths:
        return False
    return not _loadable_icd_manifests()


_DRIVER_OVERRIDES = (
    "VK_DRIVER_FILES",
    "VK_ICD_FILENAMES",
    "VK_LOADER_DRIVERS_SELECT",
    "VK_LOADER_DRIVERS_DISABLE",
)


def _vulkan_override_patterns(var: str) -> "list[str]":
    value = os.environ.get(var) or ""
    return [entry.strip() for entry in value.split(",") if entry.strip()]


def _the_loader_would_have_a_driver_without(cleared: "frozenset[str]") -> bool:
    """Would a driver still load with these overrides unset? A filename allowance alone is not enough."""
    forced = [var for var in ("VK_DRIVER_FILES", "VK_ICD_FILENAMES") if _is_set(var)]
    if not set(forced) & cleared:
        candidates = _vulkan_icd_manifest_paths()
    else:
        remaining = [var for var in forced if var not in cleared]
        candidates = (
            _forced_icd_manifest_paths(remaining[0])
            if remaining
            else _searched_vulkan_icd_manifest_paths()
        )
    disable = (
        []
        if "VK_LOADER_DRIVERS_DISABLE" in cleared
        else _vulkan_override_patterns("VK_LOADER_DRIVERS_DISABLE")
    )
    select = (
        []
        if "VK_LOADER_DRIVERS_SELECT" in cleared
        else _vulkan_override_patterns("VK_LOADER_DRIVERS_SELECT")
    )
    for path in candidates:
        name = PurePath(path).name
        if any(_vulkan_glob_matches(pattern, name) for pattern in disable):
            continue
        if select and not any(_vulkan_glob_matches(pattern, name) for pattern in select):
            continue
        if not _an_icd_is_32_bit(path) and _icd_manifest_is_usable(path):
            return True
    return False


def _is_set(var: str) -> bool:
    return bool((os.environ.get(var) or "").strip())


def _forced_icd_manifest_paths(var: str) -> "list[str]":
    value = (os.environ.get(var) or "").strip()
    return [entry.strip() for entry in value.split(os.pathsep) if entry.strip()]


def the_vulkan_loader_override_to_blame() -> "str | None":
    """Smallest override set whose removal leaves a loadable driver; None when no override is to blame."""
    if not _vulkan_icd_manifest_paths():
        return None
    overrides = [var for var in _DRIVER_OVERRIDES if _is_set(var)]
    if not overrides:
        return None
    for var in overrides:
        if _the_loader_would_have_a_driver_without(frozenset([var])):
            return var
    if len(overrides) > 1 and _the_loader_would_have_a_driver_without(frozenset(overrides)):
        return " and ".join(overrides) + " together"
    return None


def a_non_amd_render_node_is_open() -> bool:
    """Vulkan can use any vendor's node, so an open non-AMD one is a path; HIP has no such fallback."""
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
    """AMD nodes this user cannot open; uses os.access because open() on /dev/kfd initialises KFD."""
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
            # DRM is consulted only when KFD topology is unreadable; vendor-confirmed, never assumed.
            if _amd_in_topology or (
                _kfd_topology_amd_state() is None and _a_confirmed_amd_render_node_exists()
            ):
                closed.append(path)
            continue
        _vendor = _render_node_vendor(path)
        if _vendor == _AMD_PCI_VENDOR_ID:
            closed.append(path)
        elif _vendor is None and _amd_in_topology:
            # A container can hide the node's vendor sysfs; KFD keeps NVIDIA-only hosts silent.
            closed.append(path)
    return closed


def _has_an_access_acl(path: str) -> bool:
    """Detects a POSIX ACL via the xattr list; the names are str, so a bytes literal would never match."""
    try:
        names = os.listxattr(path)
    except (OSError, AttributeError, UnicodeDecodeError):
        return False
    return any(
        (_n.decode("utf-8", "replace") if isinstance(_n, bytes) else _n)
        == "system.posix_acl_access"
        for _n in names
    )


# Membership here grants far more than a device node (docker/lxd are root by another route).
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


# usermod-safe names only: commas split groups and install.sh parses with awk -F'|'.
_GROUP_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\$?\Z")


def _group_name_is_prescribable(name: str) -> bool:
    """Whether ``name`` can be pasted into a repair command as a single group."""
    return bool(name) and _GROUP_NAME_RE.match(name) is not None


def _groups_that_own(paths: list) -> tuple:
    """Sorts nodes into buckets that each need a different repair; usermod refuses a bare GID."""
    joinable, unnamed, no_group, acl, owned, privileged, already, external = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    try:
        # getgroups() does not always include the primary gid, so both are needed.
        _mine = {os.getgid(), *os.getgroups()}
    except (OSError, AttributeError):
        _mine = set()
    for path in paths:
        try:
            _st = os.stat(path)
        except OSError:
            continue
        # With an access ACL the group mode bits are the ACL mask, so report instead of prescribe.
        if _has_an_access_acl(path):
            acl.append(path)
            continue
        # POSIX uses only the owner class once the uid matches, so joining the group cannot help.
        if _st.st_uid == os.getuid():
            # Owner bits without rw mean the node is shut, not the mode; counted external like `already`.
            if _st.st_mode & stat.S_IRUSR and _st.st_mode & stat.S_IWUSR:
                external.append(path)
            else:
                owned.append(path)
            continue
        # Other class also resolves exclusively: rw other bits mean the mode is not the denial.
        if _st.st_gid not in _mine and (_st.st_mode & stat.S_IROTH and _st.st_mode & stat.S_IWOTH):
            external.append(path)
            continue
        # HIP and the Vulkan loader both open the node read-write.
        if (_st.st_mode & stat.S_IRGRP) == 0 or (_st.st_mode & stat.S_IWGRP) == 0:
            no_group.append(path)
            continue
        try:
            import grp
            name = grp.getgrgid(_st.st_gid).gr_name
        except Exception:  # noqa: BLE001 -- no group database, or no entry for this gid
            name = ""
        # A name that cannot be pasted as one group is reported by GID instead.
        if name and not _group_name_is_prescribable(name):
            name = ""
        # gid 0 is the root group even when the lookup fails in a minimal container.
        if _st.st_gid == 0:
            _root = name or "root"
            if _root not in privileged:
                privileged.append(_root)
            continue
        # A node owned by a privileged group is a udev misconfiguration to report.
        if name in _PRIVILEGED_GROUPS:
            if name not in privileged:
                privileged.append(name)
            continue
        # Already a member: a cgroup or LSM denies it, so usermod would not help.
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
    return joinable, unnamed, no_group, acl, owned, privileged, already, external


_RENDER_NODE_GLOB = "/dev/dri/renderD*"


def _amd_nodes_the_runtime_lacks(*, needs_kfd: bool = True) -> "list[str]":
    """Missing AMD nodes, gated on AMD evidence so an NVIDIA-only host cannot trigger it."""
    if not _kfd_topology_has_an_amd_gpu() and not (
        _kfd_topology_amd_state() is None and _a_confirmed_amd_render_node_exists()
    ):
        return []
    lacks = []
    if needs_kfd and not os.path.exists(_KFD_NODE):
        lacks.append(_KFD_NODE)
    if not _amd_render_node_exists():
        lacks.append(_RENDER_NODE_GLOB)
    return lacks


def amd_closed_nodes_block_the_runtime(*, needs_kfd: bool = True) -> bool:
    """A closed node blocks only with no open sibling; /dev/kfd has no sibling, so it blocks HIP
    outright."""
    # Missing nodes block like closed ones, so they must not suppress the hint.
    if _amd_nodes_the_runtime_lacks(needs_kfd = needs_kfd):
        return True
    closed = amd_nodes_closed_to_this_user()
    if not closed:
        return False
    if needs_kfd and _KFD_NODE in closed:
        return True
    # A HIP selector may pick the closed GPU, so the open sibling stops counting; not for Vulkan.
    if needs_kfd and _a_per_gpu_mask_narrows_the_runtime():
        return True
    return not an_amd_render_node_is_open()


def _selector_exposes_every_gpu(
    value: str,
    count: "int | None",
    *,
    repeat_ends_the_list: bool = False,
) -> bool:
    """ROCr ends its list at a repeated token (not clr), so ROCR_VISIBLE_DEVICES=0,0,1 shows one GPU."""
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
    """A narrowing mask makes an open node no evidence; HIP_VISIBLE_DEVICES shadows CUDA_VISIBLE_DEVICES."""
    count = amd_kfd_gpu_node_count()
    _hip_layer = (
        "HIP_VISIBLE_DEVICES"
        if os.environ.get("HIP_VISIBLE_DEVICES", "").strip()
        else "CUDA_VISIBLE_DEVICES"
    )
    for _name in (
        "ROCR_VISIBLE_DEVICES",
        _hip_layer,
        # ROCm's fourth visibility variable; see tests/test_amd_smi_inventory_matches_hip.py.
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
        # The no-pwd placeholder is meant to be expanded, not named.
        return value
    return shlex.quote(value)


def _repair_account() -> Optional[str]:
    """From getuid, not the inherited USER; None when the uid has no passwd entry (docker --user)."""
    try:
        import pwd
        return pwd.getpwuid(os.getuid()).pw_name
    except (KeyError, OSError):
        return None
    except (ImportError, AttributeError):
        # No pwd module at all, which is Windows, where none of these nodes exist.
        return os.environ.get("USER") or os.environ.get("LOGNAME") or "$USER"


def amd_node_permission_hint(*, needs_kfd: bool = True) -> Optional[str]:
    """Names the closed nodes and the command that opens them; needs_kfd=False drops /dev/kfd for Vulkan."""
    closed = amd_nodes_closed_to_this_user()
    if not needs_kfd:
        closed = [path for path in closed if path != _KFD_NODE]
    # A missing node needs reporting even when nothing is closed (partial --device mappings).
    missing = _amd_nodes_the_runtime_lacks(needs_kfd = needs_kfd)
    parts: "list[str]" = []
    trailing_command = ""
    if closed:
        # Claim only what the closed set blocks: an open sibling still gives a complete path.
        user = _repair_account()
        joinable, unnamed, no_group, acl, owned, privileged, already, external = _groups_that_own(
            closed
        )
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
        # Prescribe only where joining a group is the repair; unstat-able hosts get the default pair.
        if joinable or not (
            unnamed or no_group or acl or owned or privileged or already or external
        ):
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
                # Kept at the end so the pasteable command is not run into by later text.
                trailing_command = (
                    f"Add the account to the {joined} {plural} and then log out and back "
                    f"in: sudo usermod -a -G {_shell_word(joined)} {_shell_word(user)}"
                )
        if unnamed:
            _gids = ", ".join(str(_g) for _g in unnamed)
            # docker --group-add takes one value, so one flag per GID.
            _adds = " ".join(f"--group-add {_g}" for _g in unnamed)
            _noun = "GID" if len(unnamed) == 1 else "GIDs"
            _verb = "which has" if len(unnamed) == 1 else "which have"
            # Creating the group names only the GID; the account and node still need fixing, one per GID.
            _each = "it" if len(unnamed) == 1 else "each of them"
            _pairs = "; ".join(
                f"sudo groupadd -g {_g} amdgpu{_g} && "
                f"sudo usermod -a -G amdgpu{_g} {_shell_word(user)}"
                for _g in unnamed
            )
            # One groupadd/usermod pair per GID; && stops usermod hitting a same-named wrong group.
            if user is None:
                # No passwd account to add, so the bare-host half cannot run and the container half is the repair.
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
        if external:
            # Worded by the class that applies: owner or other, not always owner.
            parts.append(
                f"{', '.join(external)} is already granted read and write by the permission "
                f"bits that apply to this account, so the mode is not what is shutting it: "
                f"something outside the file mode is denying it, typically a container "
                f"device cgroup or an LSM such as SELinux or AppArmor."
            )
        if privileged:
            parts.append(
                f"Those nodes belong to the {', '.join(privileged)} group, which grants a "
                f"great deal besides the GPU, so joining it is not the repair: fix the udev "
                f"rule so the node is owned by render or video instead."
            )
        if acl:
            parts.append(
                # Every path: their ACLs need not agree; getfacl takes several paths.
                f"{', '.join(acl)} carries a POSIX ACL, so the group permissions cannot be "
                f"read from its mode: check the real grant with getfacl {' '.join(acl)} "
                f"before changing group membership."
            )
    # Group membership cannot create a device node; install.sh says the same.
    if _KFD_NODE in missing:
        # Only a topology naming an AMD GPU proves the driver is loaded, hence two wordings.
        if _kfd_topology_amd_state() is True:
            parts.append(
                "ROCm needs /dev/kfd, which is not present here, but the KFD topology "
                "already names an AMD GPU, so the kernel driver is loaded and reinstalling "
                "ROCm changes nothing: the node itself is missing. Under Docker, recreate "
                "the container with --device /dev/kfd --device /dev/dri; on a bare host it "
                "is a udev or devtmpfs problem. No group membership creates it."
            )
        else:
            parts.append(
                "ROCm needs /dev/kfd, which is not present here, while an AMD render node "
                "is. Under Docker, recreate the container with --device /dev/kfd --device "
                "/dev/dri; on a bare host, check that the amdgpu kernel module is loaded. "
                "No group membership creates it."
            )
    if _RENDER_NODE_GLOB in missing:
        # Vulkan never opens /dev/kfd, so only map it when needed.
        _devices = "--device /dev/kfd --device /dev/dri" if needs_kfd else "--device /dev/dri"
        parts.append(
            f"No AMD render node (/dev/dri/renderD*) is present, and ROCm and Vulkan both "
            f"open one, so the device mapping needs fixing; under Docker that is "
            f"{_devices}."
        )
    if trailing_command:
        parts.append(trailing_command)
    return " ".join(parts) or None
