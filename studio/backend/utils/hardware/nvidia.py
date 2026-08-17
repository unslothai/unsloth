# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import subprocess
import time
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


# Failures aren't cached forever: a hung/missing nvidia-smi at Studio startup
# (driver still initializing, momentary system load) can recover mid-session,
# and the picker shouldn't stay hidden until a restart on the strength of one
# bad probe. Successes also expire, just on a much longer horizon -- GPU
# topology and MIG mode are effectively static, but not provably immutable
# for a whole Studio session (an admin can toggle MIG on a running host) --
# so a resolution is revalidated periodically rather than trusted forever.
_FAILED_RESOLUTION_TTL_SECONDS = 30
_RESOLVED_MASK_TTL_SECONDS = 300
_uuid_mask_resolution_cache: dict[tuple[str, ...], tuple[Optional[list[int]], float]] = {}

_GPU_UUID_PREFIX = "GPU-"


def _query_uuid_to_ordinal() -> Optional[dict[str, tuple[int, bool]]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,mig.mode.current",
                "--format=csv,noheader",
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
        logger.warning("nvidia-smi query failed while resolving a UUID mask: %s", e)
        return None
    if result.returncode != 0:
        return None

    rows = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        idx_str, uuid, bus_id, mig_mode = parts
        try:
            idx = int(idx_str)
        except (ValueError, TypeError):
            continue
        rows.append((idx, uuid, bus_id, mig_mode))

    # get_visible_gpu_utilization() and get_backend_visible_gpu_info() below,
    # and the "GPU index ordering" pin in hardware.py, all assume nvidia-smi's
    # own index column already matches PCI bus order -- that's the load-bearing
    # invariant this whole file leans on. NVIDIA's own docs stop short of
    # guaranteeing it, so verify it here instead of silently trusting it (or
    # silently computing an independent ordinal that would then disagree with
    # every other index-keyed probe in this file): sort by pci.bus_id
    # ourselves -- its fixed-width "domain:bus:device.function" hex format
    # sorts correctly as plain text -- and confirm every row's index equals
    # its position in that sort. If it does, resolving to nvidia-smi's own
    # index is safe and stays consistent with those other probes. If it
    # doesn't, the assumption this file depends on doesn't hold on this host
    # and no ordinal from either source can be trusted to line up with the
    # others, so fail closed rather than resolve to a value that only agrees
    # with half the codebase's telemetry.
    by_pci_order = sorted(rows, key = lambda row: row[2])
    for expected_idx, (idx, _uuid, bus_id, _mig_mode) in enumerate(by_pci_order):
        if idx != expected_idx:
            logger.warning(
                "nvidia-smi index does not match PCI bus order (index=%s, "
                "pci.bus_id=%s, expected position=%s); declining to resolve "
                "UUID masks on this host",
                idx,
                bus_id,
                expected_idx,
            )
            return None

    # (index, is_mig_enabled) per UUID, for every row -- including MIG-enabled
    # roots. A MIG-enabled root is never itself a valid resolution (CUDA
    # exposes its MIG instances instead of the whole card), but it must still
    # be visible during *ambiguity* checking below: excluding it here first
    # would let a prefix shared with a MIG root look falsely unambiguous.
    uuid_info: dict[str, tuple[int, bool]] = {}
    for idx, uuid, _bus_id, mig_mode in rows:
        uuid_info[uuid] = (idx, mig_mode == "Enabled")
    return uuid_info


def _resolve_uuid_token(token: str, uuid_info: dict[str, tuple[int, bool]]) -> Optional[int]:
    exact = uuid_info.get(token)
    if exact is not None:
        idx, is_mig_enabled = exact
        return None if is_mig_enabled else idx
    # NVIDIA accepts an unambiguous UUID prefix (e.g. "GPU-abcdef12") as a
    # device identifier. Require the token to actually look like a GPU UUID
    # before attempting a prefix match -- otherwise a malformed token like
    # "G" or "GPU" (no trailing "-") could coincidentally prefix a real UUID
    # on a single-GPU host and get treated as a valid, intentional selector.
    if not (token.startswith(_GPU_UUID_PREFIX) and len(token) > len(_GPU_UUID_PREFIX)):
        return None
    # Checked against every root UUID, MIG-enabled or not -- a prefix shared
    # by more than one card, MIG or otherwise, can't be trusted either way.
    prefix_matches = [uuid for uuid in uuid_info if uuid.startswith(token)]
    if len(prefix_matches) != 1:
        return None
    idx, is_mig_enabled = uuid_info[prefix_matches[0]]
    return None if is_mig_enabled else idx


def resolve_gpu_uuid_mask(tokens: list[str]) -> Optional[list[int]]:
    """Resolve a CUDA_VISIBLE_DEVICES mask that mixes numeric indices and GPU
    UUIDs (e.g. ["0", "GPU-<uuid>"]) to physical indices, so a UUID mask is
    exactly as selectable as the equivalent numeric one. Numeric tokens are
    validated against the GPUs nvidia-smi actually reports (real CUDA
    semantics truncate enumeration at the first negative or out-of-range
    member of a mixed mask, hiding everything after it -- rather than
    replicate that exact truncation point, an invalid numeric member fails
    the whole resolution); UUID tokens are resolved to the same index
    get_visible_gpu_utilization() and get_backend_visible_gpu_info() already
    key their own nvidia-smi rows by (see _query_uuid_to_ordinal()'s PCI-bus
    cross-check). Order is preserved to match the mask, since it defines the
    visible-ordinal mapping downstream. Returns None -- and the caller falls
    back to relative ordinals -- if nvidia-smi is unavailable, its index
    doesn't match PCI bus order, a numeric token is negative or not a
    queried GPU, any UUID token doesn't match exactly one non-MIG physical
    device (an actual MIG instance UUID, a MIG-enabled root's UUID, or a
    UUID/prefix ambiguous across cards -- checked against every root
    regardless of MIG state, so a MIG root can't make an otherwise-ambiguous
    prefix look unambiguous), or two tokens resolve to the same physical ID
    (a repeated or aliased UUID/index pair): _visible_ordinal_map() is keyed
    by physical ID, so a duplicate silently collapses one of the mask's
    visible ordinals rather than giving it its own device entry. Cached per
    token tuple -- successes for _RESOLVED_MASK_TTL_SECONDS, since GPU
    topology/MIG mode can change while Studio keeps running (an admin
    enabling MIG on a previously plain card, without a restart); failures
    for the much shorter _FAILED_RESOLUTION_TTL_SECONDS, since a hung or
    missing nvidia-smi would otherwise cost its full timeout on every caller
    of _get_parent_visible_gpu_spec() every poll cycle."""
    cache_key = tuple(tokens)
    cached = _uuid_mask_resolution_cache.get(cache_key)
    if cached is not None:
        value, cached_at = cached
        ttl = _RESOLVED_MASK_TTL_SECONDS if value is not None else _FAILED_RESOLUTION_TTL_SECONDS
        if time.monotonic() - cached_at < ttl:
            return value

    uuid_info = _query_uuid_to_ordinal()
    if uuid_info is None:
        _uuid_mask_resolution_cache[cache_key] = (None, time.monotonic())
        return None
    valid_indices = {idx for idx, _is_mig_enabled in uuid_info.values()}

    resolved = []
    for token in tokens:
        try:
            numeric_idx = int(token)
        except ValueError:
            idx = _resolve_uuid_token(token, uuid_info)
            if idx is None:
                _uuid_mask_resolution_cache[cache_key] = (None, time.monotonic())
                return None
            resolved.append(idx)
            continue
        # CUDA truncates enumeration at the first invalid member of a mixed
        # mask (negative or not a real device) -- everything listed after it
        # in the real CUDA_VISIBLE_DEVICES semantics is not visible at all,
        # not merely skipped. Replicating that exact truncation point would
        # mean returning a list shorter than the mask, which every caller of
        # _get_parent_visible_gpu_spec() would then have to special-case, so
        # fail the whole resolution instead: falling back to relative
        # ordinals (no explicit selection) is safer than resolving a later
        # UUID token that real CUDA_VISIBLE_DEVICES parsing would have hidden.
        if numeric_idx < 0 or numeric_idx not in valid_indices:
            _uuid_mask_resolution_cache[cache_key] = (None, time.monotonic())
            return None
        resolved.append(numeric_idx)

    if len(set(resolved)) != len(resolved):
        logger.warning(
            "GPU mask %r resolved to duplicate physical IDs %r; a physical "
            "ID mask can't represent this many distinct visible ordinals",
            tokens,
            resolved,
        )
        _uuid_mask_resolution_cache[cache_key] = (None, time.monotonic())
        return None

    _uuid_mask_resolution_cache[cache_key] = (resolved, time.monotonic())
    return resolved


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
    # parent_visible_ids None (UUID/MIG mask): can't map nvidia-smi rows to
    # visible devices, so return empty rather than exposing all physical GPUs.
    if parent_visible_ids is None:
        return {
            "available": False,
            "backend_cuda_visible_devices": parent_cuda_visible_devices,
            "parent_visible_gpu_ids": [],
            "devices": [],
            "index_kind": "unresolved",
        }
    visible_ordinals = _visible_ordinal_map(parent_visible_ids)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,temperature.gpu,"
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
        logger.warning("nvidia-smi query failed in get_visible_gpu_utilization: %s", e)
        return {
            "available": False,
            "backend_cuda_visible_devices": parent_cuda_visible_devices,
            "parent_visible_gpu_ids": parent_visible_ids or [],
            "devices": [],
            "index_kind": "physical",
        }
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

        try:
            idx = int(parts[0])
        except (ValueError, TypeError):
            continue

        if visible_ordinals is not None and idx not in visible_ordinals:
            continue

        devices.append(
            _build_gpu_metrics(
                vram_used_mb = _parse_smi_value(parts[3]),
                vram_total_mb = _parse_smi_value(parts[4]),
                power_draw = _parse_smi_value(parts[5]),
                power_limit = _parse_smi_value(parts[6]),
                index = idx,
                index_kind = "physical",
                visible_ordinal = (
                    visible_ordinals[idx] if visible_ordinals is not None else len(devices)
                ),
                gpu_utilization_pct = _parse_smi_value(parts[1]),
                temperature_c = _parse_smi_value(parts[2]),
            )
        )

    return {
        "available": len(devices) > 0,
        "backend_cuda_visible_devices": parent_cuda_visible_devices,
        "parent_visible_gpu_ids": parent_visible_ids or [],
        "devices": devices,
        "index_kind": "physical",
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
        logger.warning("nvidia-smi query failed in get_backend_visible_gpu_info: %s", e)
        return {
            "available": False,
            "backend_cuda_visible_devices": backend_cuda_visible_devices,
            "parent_visible_gpu_ids": parent_visible_ids or [],
            "devices": [],
            "index_kind": "physical",
        }
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
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
        except (ValueError, TypeError):
            continue
        if visible_ordinals is not None and idx not in visible_ordinals:
            continue
        # Rejoin in case the GPU name contains commas
        name = parts[1] if len(parts) == 3 else ", ".join(parts[1:-1])
        try:
            mem_total_mb = int(parts[-1])
        except (ValueError, TypeError):
            continue
        devices.append(
            {
                "index": idx,
                "index_kind": "physical",
                "visible_ordinal": (
                    visible_ordinals[idx] if visible_ordinals is not None else len(devices)
                ),
                "name": name,
                "memory_total_gb": round(mem_total_mb / 1024, 2),
            }
        )

    return {
        "available": len(devices) > 0,
        "backend_cuda_visible_devices": backend_cuda_visible_devices,
        "parent_visible_gpu_ids": parent_visible_ids or [],
        "devices": devices,
        "index_kind": "physical",
    }
