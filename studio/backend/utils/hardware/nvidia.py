# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import platform
import shutil
import subprocess
import threading
import time
from typing import Any, Optional

from loggers import get_logger

from . import gpu_query

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
    tokens = [token.strip().lower() for token in (parent_cuda_visible_devices or "").split(",")]
    # CUDA stops at the first empty entry: a leading one hides every GPU, so leave it unresolved.
    if "" in tokens:
        tokens = tokens[: tokens.index("")]
    if not tokens or any(not token.startswith("gpu-") for token in tokens):
        return None

    visible_ordinals: dict[int, int] = {}
    for ordinal, token in enumerate(tokens):
        matches = [idx for idx, gpu_uuid in gpu_rows if gpu_uuid.lower().startswith(token)]
        if len(matches) != 1 or matches[0] in visible_ordinals:
            return None
        visible_ordinals[matches[0]] = ordinal
    return visible_ordinals


_UUID_MASK_TTL_S = 30.0
_uuid_mask_cache: dict[tuple[str, int], tuple[float, Optional[list[int]]]] = {}


def resolve_uuid_mask(parent_cuda_visible_devices: str) -> Optional[list[int]]:
    """Physical nvidia-smi indices for a full-GPU UUID mask, in mask order; None if any token is unresolvable."""
    # gpu_query never caches a failure, and the parent GPU spec is read on every poll and load:
    # without this a hung nvidia-smi costs its full timeout per call.
    # Keyed on the static generation so invalidate_static() (eGPU hotplug, driver reset) drops it too.
    key = (parent_cuda_visible_devices, gpu_query._static_gen)
    hit = _uuid_mask_cache.get(key)
    if hit is not None and time.monotonic() - hit[0] < _UUID_MASK_TTL_S:
        return None if hit[1] is None else list(hit[1])
    ids = _query_uuid_mask(parent_cuda_visible_devices)
    _uuid_mask_cache.clear()
    _uuid_mask_cache[key] = (time.monotonic(), ids)
    return None if ids is None else list(ids)


def _query_uuid_mask(parent_cuda_visible_devices: str) -> Optional[list[int]]:
    try:
        result = gpu_query.run_nvidia_smi(
            [_nvidia_smi_executable(), "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 5,
            env = child_env_without_native_path_secret(),
            **_windows_hidden_subprocess_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning("nvidia-smi query failed while resolving a UUID mask: %s", e)
        return None
    if result.returncode != 0:
        return None
    gpu_rows = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdecimal():
            gpu_rows.append((int(parts[0]), parts[1]))
    visible_ordinals = _uuid_visible_ordinal_map(parent_cuda_visible_devices, gpu_rows)
    if visible_ordinals is None:
        return None
    return sorted(visible_ordinals, key = visible_ordinals.__getitem__)


def get_physical_gpu_count(gpu_rows_only: bool = False) -> Optional[int]:
    """Return physical GPU count via nvidia-smi, or None on failure. ``gpu_rows_only`` skips the indented MIG child rows."""
    gen = gpu_query._static_gen
    started = time.monotonic()
    try:
        result = gpu_query.run_nvidia_smi(
            [_nvidia_smi_executable(), "-L"],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 5,
            env = child_env_without_native_path_secret(),
            **_windows_hidden_subprocess_kwargs(),
        )
        if result.returncode == 0 and result.stdout.strip():
            # Same listing get_mig_gpu_indices parses: publish it so memory polls reuse it.
            at = _listing_taken_at(started, result.stdout)
            if at is not None:
                _store_mig_gpu_indices(_parse_mig_gpu_indices(result.stdout), gen, at = at)
            lines = result.stdout.strip().splitlines()
            if gpu_rows_only:
                return sum(1 for line in lines if line.startswith("GPU "))
            return len(lines)
        logger.warning(
            "nvidia-smi -L returned code %d; caller should fall back to torch",
            result.returncode,
        )
    except Exception as e:
        logger.warning("nvidia-smi -L failed: %s; caller should fall back to torch", e)
    return None


# A failed listing is retried after this, not on every poll.
_MIG_FAILURE_RETRY_S = 5.0
# (static generation, monotonic stamp, indices, answered); answered False is a failed probe.
_mig_indices_cache: Optional[tuple[int, float, frozenset[int], bool]] = None


def _store_mig_gpu_indices(
    indices: frozenset[int],
    gen: int,
    at: Optional[float] = None,
    answered: bool = True,
) -> None:
    """Publish a listing under the generation read before its query started; one the
    inventory was invalidated during is dropped, so the next caller re-reads. ``at`` is
    when the listing was taken, so a gpu_query cache hit does not renew its age."""
    global _mig_indices_cache
    if gen == gpu_query._static_gen:
        _mig_indices_cache = (gen, time.monotonic() if at is None else at, indices, answered)


def _listing_taken_at(started: float, stdout: Any) -> Optional[float]:
    """When the ``-L`` answer ``stdout`` was produced: the start of the gpu_query entry
    holding exactly that answer, or when this query began with the cache off. None when
    the cache no longer holds it (a newer listing replaced it): not safe to publish."""
    if not gpu_query.enabled():
        return started
    try:
        at = gpu_query.answered_at([_nvidia_smi_executable(), "-L"], stdout)
    except Exception:
        return None
    return None if at is None else min(at, started)


def _parse_mig_gpu_indices(stdout: Any) -> frozenset[int]:
    if not isinstance(stdout, str):
        return frozenset()
    indices: set[int] = set()
    current: Optional[int] = None
    for line in stdout.splitlines():
        if line.startswith("GPU "):
            head = line[4:].split(":", 1)[0].strip()
            current = int(head) if head.isdecimal() else None
        elif current is not None and line.strip().startswith("MIG "):
            indices.add(current)
    return frozenset(indices)


def get_mig_gpu_indices() -> set[int]:
    """Physical indices of GPUs with MIG instances (indented ``MIG`` rows under their
    ``GPU N:`` row in ``nvidia-smi -L``). CUDA sees such a card as one slice, so its
    ``--query-gpu`` memory is the whole card's, not the device's.

    Read on every memory poll, so the parsed set (empty included) is cached until the
    listing is gpu_query's static TTL old, and dropped with the static generation (hotplug, driver reset);
    the physical count probe refreshes it from the same listing. A failed probe keeps the
    last answer of this generation (empty if none) and is retried after a short backoff."""
    hit = _mig_indices_cache
    now = time.monotonic()
    if hit is not None and hit[0] == gpu_query._static_gen:
        # Same lifetime as gpu_query's static inventory, so UNSLOTH_GPU_QUERY_STATIC_TTL
        # (0 included) bounds it too.
        ttl = gpu_query.ttl_for(gpu_query.STATIC)
        if now - hit[1] < (ttl if hit[3] else min(ttl, _MIG_FAILURE_RETRY_S)):
            return set(hit[2])
    gen = gpu_query._static_gen
    answer = _query_mig_gpu_indices()
    if answer is None:
        last = hit[2] if hit is not None and hit[0] == gen else frozenset()
        _store_mig_gpu_indices(last, gen, at = time.monotonic(), answered = False)
        return set(last)
    indices, at = answer
    if at is not None:
        _store_mig_gpu_indices(indices, gen, at = at)
    return set(indices)


def _query_mig_gpu_indices() -> Optional[tuple[frozenset[int], Optional[float]]]:
    """The parsed listing and when it was taken (None: superseded, do not cache), or None
    when nvidia-smi could not answer. Empty output is a failed answer, as in gpu_query."""
    started = time.monotonic()
    try:
        result = gpu_query.run_nvidia_smi(
            [_nvidia_smi_executable(), "-L"],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 5,
            env = child_env_without_native_path_secret(),
            **_windows_hidden_subprocess_kwargs(),
        )
    except Exception:
        return None
    if result.returncode != 0 or not isinstance(result.stdout, str) or not result.stdout.strip():
        return None
    return _parse_mig_gpu_indices(result.stdout), _listing_taken_at(started, result.stdout)


def get_primary_gpu_utilization() -> dict[str, Any]:
    try:
        result = gpu_query.run_nvidia_smi(
            [
                _nvidia_smi_executable(),
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
        result = gpu_query.run_nvidia_smi(
            [
                _nvidia_smi_executable(),
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

    # nvidia-smi emits physical row order, so a reordering mask would hand back devices whose position contradicts their own visible_ordinal.
    devices.sort(key = lambda d: d["visible_ordinal"])

    return {
        "available": len(devices) > 0,
        "backend_cuda_visible_devices": parent_cuda_visible_devices,
        "parent_visible_gpu_ids": parent_visible_ids or [],
        "devices": devices,
        "index_kind": "relative" if includes_uuid else "physical",
    }


_WSL_NVIDIA_SMI = "/usr/lib/wsl/lib/nvidia-smi"


def _nvidia_smi_executable() -> str:
    """The nvidia-smi to run, resolving the standard Windows locations off PATH. A driver install can leave nvidia-smi.exe in the NVSMI directory or the driver store without putting either on PATH, and a bare "nvidia-smi" then raises FileNotFoundError, leaving the physical inventory empty on exactly the host this inventory exists for: real GPUs, a PyTorch that cannot see them. Same two locations setup.ps1 falls back to. Returns the bare name when nothing better is found, so the caller's existing OSError handling still applies."""
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if platform.system() != "Windows":
        # WSL: often off PATH (secure_path strips it), and no /proc/driver/nvidia fallback.
        if platform.system() == "Linux" and os.path.isfile(_WSL_NVIDIA_SMI):
            return _WSL_NVIDIA_SMI
        return "nvidia-smi"
    for base, tail in (
        (os.environ.get("ProgramFiles"), r"NVIDIA Corporation\NVSMI\nvidia-smi.exe"),
        (os.environ.get("SystemRoot"), r"System32\nvidia-smi.exe"),
    ):
        if not base:
            continue
        candidate = os.path.join(base, tail)
        if os.path.isfile(candidate):
            return candidate
    return "nvidia-smi"


# "nvidia-smi is not on this machine" is a conclusive answer, not a failed probe: it is the normal state of every CPU-only, AMD and Intel host, and the installers read the same absence the same way. Distinct from None, which means a probe that WAS found could not answer (a hung driver, a permission fault, a non-zero exit).
NVIDIA_SMI_ABSENT = object()


# This thread's last inventory exit code: exit 6 is nvidia-smi's own "No devices were found".
_inventory_exit = threading.local()


def _query_gpu_inventory(caller: str) -> Any:
    """``[{index, name, memory_total_gb}]`` for every GPU nvidia-smi enumerates.

    ``None`` when the query could not be answered at all (no nvidia-smi on PATH, a driver that hung past the timeout, a non-zero exit), which callers report as "unknown" and is not the same as the empty list a working driver with no cards returns. Never raises.

    Split out of get_backend_visible_gpu_info so the same rows can be read WITHOUT a ``DeviceType.CUDA`` precondition: get_physical_gpu_inventory below is reached on exactly the host where torch reports no CUDA device, and that host still has its GPUs. Rows a caller cannot make sense of are dropped rather than raised on: a name holding commas is rejoined, and a malformed index or memory column skips the row.
    """
    try:
        result = gpu_query.run_nvidia_smi(
            [
                _nvidia_smi_executable(),
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
    except FileNotFoundError as e:
        # No nvidia-smi at all, the NORMAL state of every CPU-only, AMD and Intel host. This is called on a 60 second refresh reached from the health and system polls, so warning here would log a line every minute on machines that are working correctly.
        logger.debug("nvidia-smi is not installed (%s): %s", caller, e)
        return NVIDIA_SMI_ABSENT
    except (OSError, subprocess.TimeoutExpired) as e:
        # Past this point an nvidia-smi WAS found, so a failure is a real fault on this host.
        logger.warning("nvidia-smi query failed in %s: %s", caller, e)
        return None
    _inventory_exit.code = result.returncode
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
        # A capacity this build of nvidia-smi will not report ("[N/A]", as _parse_smi_value already recognises) is a missing metric, not a missing card. Dropping the row hid the GPU from the whole inventory, and the Linux procfs fallback does not run either since it answers for a query that FAILED rather than one that came back short: the host lost its mismatch and its repair guidance over an unknown size.
        mem_total_mb = _parse_smi_value(parts[-1])
        rows.append(
            {
                "index": idx,
                "name": name,
                "memory_total_gb": round(mem_total_mb / 1024, 2)
                if mem_total_mb is not None
                else None,
            }
        )
    return rows


def _linux_nvidia_procfs_gpu_count() -> int:
    """How many GPUs the NVIDIA kernel driver enumerates under /proc, or 0. One subdirectory per GPU, published whatever nvidia-smi's state is, which is why the installer falls back to it too. Never raises; 0 on any platform without it."""
    if platform.system() != "Linux":
        return 0
    try:
        entries = os.listdir("/proc/driver/nvidia/gpus")
    except OSError:
        return 0
    return len(entries)


# Native tensor-core support: FP8 from Ada / Hopper (sm_89), NVFP4 from Blackwell (sm_100, sm_120).
_CHECKPOINT_QUANT_MIN_CAPABILITY = (("fp8", (8, 9)), ("nvfp4", (10, 0)))


def get_checkpoint_quant_formats() -> list[str]:
    """Prequantized checkpoint formats every NVIDIA card runs natively (Hub Recommended filter).
    A load may land on any card, so the least capable one answers; an unreadable card counts as none."""
    try:
        result = gpu_query.run_nvidia_smi(
            [_nvidia_smi_executable(), "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 5,
            env = child_env_without_native_path_secret(),
            **_windows_hidden_subprocess_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    capabilities: list[tuple[int, int]] = []
    for line in result.stdout.strip().splitlines():
        major, _, minor = line.strip().partition(".")
        if not (major.isdigit() and minor.isdigit()):
            return []
        capabilities.append((int(major), int(minor)))
    if not capabilities:
        return []
    floor = min(capabilities)
    return [fmt for fmt, needed in _CHECKPOINT_QUANT_MIN_CAPABILITY if floor >= needed]


def get_compute_capabilities() -> dict[int, tuple[int, int]]:
    """nvidia-smi row index -> (major, minor), no mask and no torch; {} when unreadable."""
    try:
        result = gpu_query.run_nvidia_smi(
            [_nvidia_smi_executable(), "--query-gpu=index,compute_cap", "--format=csv,noheader"],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 5,
            env = child_env_without_native_path_secret(),
            **_windows_hidden_subprocess_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if result.returncode != 0:
        return {}
    capabilities: dict[int, tuple[int, int]] = {}
    for line in result.stdout.strip().splitlines():
        index, _, cap = (part.strip() for part in line.partition(","))
        major, _, minor = cap.partition(".")
        if not (index.isdigit() and major.isdigit() and minor.isdigit()):
            return {}
        capabilities[int(index)] = (int(major), int(minor))
    return capabilities


def get_physical_gpu_inventory() -> dict[str, Any]:
    """Every NVIDIA GPU the driver enumerates, with no visibility mask and no torch. Display-only inventory: ``index`` is nvidia-smi's own row number, a physical id and NOT something a caller may pin, because the whole point of this probe is that PyTorch cannot open these devices. A failed probe comes back as a structured unavailable result, so this never raises out of an endpoint."""
    rows = _query_gpu_inventory("get_physical_gpu_inventory")
    # Either way the CLI could not answer. The kernel driver publishes its cards regardless, and on a cold start there is no settled verdict for the resulting unknown to protect.
    if (rows is NVIDIA_SMI_ABSENT or rows is None) and _linux_nvidia_procfs_gpu_count():
        # The kernel driver is loaded and enumerating cards; only the CLI is missing. _has_usable_nvidia_gpu() reads the same directory, so without this the installer can repair a CUDA wheel on a host the backend insists has no card. No name and no capacity: procfs gives neither, and an invented one would be worse than an honest blank.
        return {
            "available": True,
            "source": "proc-driver-nvidia",
            "devices": [
                {
                    "vendor": "nvidia",
                    "index": ordinal,
                    "name": None,
                    "memory_total_gb": None,
                    "source": "proc-driver-nvidia",
                }
                for ordinal in range(_linux_nvidia_procfs_gpu_count())
            ],
            "error": None,
            "absent": False,
        }
    if rows is NVIDIA_SMI_ABSENT:
        # An answer, and the caller must not read it as "some probe failed": an AMD-only host has no nvidia-smi by design.
        return {
            "available": False,
            "source": "nvidia-smi",
            "devices": [],
            "error": "nvidia-smi is not installed",
            "absent": True,
        }
    if rows is None:
        return {
            "available": False,
            "source": "nvidia-smi",
            "devices": [],
            "error": "nvidia-smi did not answer",
            "absent": False,
        }
    return {
        "available": bool(rows),
        "source": "nvidia-smi",
        "devices": [{**row, "vendor": "nvidia", "source": "nvidia-smi"} for row in rows],
        "error": None,
        "absent": False,
    }


def get_backend_visible_gpu_info(
    parent_visible_ids: Optional[list[int]], backend_cuda_visible_devices: Optional[str]
) -> dict[str, Any]:
    # parent_visible_ids None (UUID/MIG mask): cannot map nvidia-smi rows to visible devices.
    if parent_visible_ids is None:
        return {
            "available": False,
            "backend_cuda_visible_devices": backend_cuda_visible_devices,
            "parent_visible_gpu_ids": [],
            "devices": [],
            "index_kind": "unresolved",
        }
    visible_ordinals = _visible_ordinal_map(parent_visible_ids)
    _inventory_exit.code = None
    rows = _query_gpu_inventory("get_backend_visible_gpu_info")
    if rows is None or rows is NVIDIA_SMI_ABSENT:
        out = {
            "available": False,
            "backend_cuda_visible_devices": backend_cuda_visible_devices,
            "parent_visible_gpu_ids": parent_visible_ids or [],
            "devices": [],
            "index_kind": "physical",
        }
        if rows is NVIDIA_SMI_ABSENT:
            out["smi_absent"] = True
        elif getattr(_inventory_exit, "code", None) != 6:
            # No answer is unknown, not "no cards"; exit 6 ("No devices were found") is an answer.
            out["probe_failed"] = True
        return out

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
