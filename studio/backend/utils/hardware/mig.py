# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""R570+ MIG: CUDA device ordinals (one per GPU instance) joined to NVML slice memory by UUID."""

import json
import os
import re
import subprocess
import sys
import threading
import time

from . import gpu_query, nvidia
from utils.child_stdio import utf8_child_env
from utils.native_path_leases import child_env_without_native_path_secret
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

# key -> (monotonic time, rows, state); state "rows", "legacy" (old rule) or "failed" (rows keep identities, zero memory).
_cache = {}
_flights = {}
_flights_lock = threading.Lock()
_FAILED_BACKOFF_S = 5.0
_ROWS_TTL_S = 60.0
_FRESH_SHARE_S = 2.0
# Parent's ordinal -> MIG uuid map; workers pin from it and never probe.
_ENV = "UNSLOTH_MIG_CUDA_DEVICES"


def possible() -> bool:
    """MIG needs bare-metal Linux with the NVIDIA kernel driver (not WSL); elsewhere nothing starts."""
    return sys.platform == "linux" and os.path.isfile("/proc/driver/nvidia/version")


def rows_from_payload(payload):
    """None = legacy/unknown; [] = CUDA confirmed an empty set. Uses CUDA's UUIDs, not MIG row
    counts: several CIs share a GI, and containers may expose only some."""
    if not isinstance(payload, dict) or payload.get("source") != "nvml":
        return None
    match = re.match(r"(\d+)\.", str(payload.get("driver_version") or ""))
    if not match or int(match[1]) < 570:
        return None
    cuda_devices = payload.get("cuda_devices")
    if not isinstance(cuda_devices, list):
        return None
    inventory = payload.get("devices") or []
    if not any(r.get("mig") for r in inventory):
        return None
    by_uuid = {str(r.get("uuid", ""))[-36:].lower(): r for r in inventory if r.get("uuid")}
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    tokens = [t.strip() for t in (raw or "").split(",")]
    numeric = raw is not None and all(re.fullmatch(r"\d+", t) for t in tokens)
    ids = [int(t) for t in tokens] if numeric and len(tokens) == len(cuda_devices) else None
    rows = []
    seen = set()
    for ordinal, device in enumerate(cuda_devices):
        key = str(device.get("uuid", ""))[-36:].lower()
        row = by_uuid.get(key)
        if row is None or key in seen:
            return None  # never guess an ordinal-to-UUID mapping
        seen.add(key)
        rows.append(
            {
                **row,
                "physical_index": row["index"],
                "index": str(ids[ordinal] if ids is not None else ordinal),
            }
        )
    return rows


def _mig_instances(listing):
    """Cards with MIG instances from the cached ``nvidia-smi -L``. Only ``listing`` callers (GGUF
    placement) may always relist; others list once per hardware generation, then reuse the answer."""
    if listing:
        return nvidia.get_mig_gpu_indices_or_none()
    cached = nvidia._mig_cached()
    if cached is not nvidia._MISS:
        return cached
    entry = nvidia._mig_entry
    if entry is not None and entry[3] is not None and not entry[3]:
        return None
    return nvidia.get_mig_gpu_indices_or_none()


def visible_devices(*, fresh = False, listing = False):
    """R570+ MIG rows from a bounded out-of-process probe, or None (old rule). ``fresh`` for memory."""
    if not possible() or os.environ.get("UNSLOTH_NVIDIA_LIBRARY_PROBE", "1") == "0":
        return None
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not (raw and "mig-" in raw.lower()) and not _mig_instances(listing):
        os.environ.pop(_ENV, None)
        return None
    key = _current_key()
    hit = _cached_answer(key, fresh)
    if hit is not _MISS:
        return hit
    # Single flight per key.
    with _flights_lock:
        flight = _flights.get(key)
        leader = flight is None
        if leader:
            flight = _flights[key] = threading.Event()
    if not leader:
        flight.wait(15)
        entry = _cache.get(key)
        return entry[1] if entry is not None else None
    try:
        return _refresh(key, raw)
    finally:
        with _flights_lock:
            _flights.pop(key, None)
        flight.set()


_MISS = object()


def _current_key():
    return (
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        os.environ.get("CUDA_DEVICE_ORDER"),
        gpu_query._static_gen,
        os.environ.get("UNSLOTH_NVIDIA_PROBE"),
    )


def _cached_answer(key, fresh):
    entry = _cache.get(key)
    if entry is None:
        return _MISS
    age = time.monotonic() - entry[0]
    state = entry[2]
    if state == "failed":
        return entry[1] if age < _FAILED_BACKOFF_S else _MISS
    if state == "legacy":
        return None if age < _ROWS_TTL_S else _MISS
    if age < _FRESH_SHARE_S or (not fresh and age < _ROWS_TTL_S):
        return entry[1]
    return _MISS


def _probe():
    """(rows, answered). ``answered`` is False when the probe child gave no NVML answer."""
    try:
        from utils.prebuilt.update_flow import find_installer_script

        script = find_installer_script(
            env_var = "UNSLOTH_NVIDIA_PROBE", script_name = "nvidia_probe.py"
        )
        if script is None:
            return None, False
        result = subprocess.run(
            [sys.executable, "-I", str(script), "--json"],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 10,
            env = utf8_child_env(child_env_without_native_path_secret()),
            **windows_hidden_subprocess_kwargs(),
        )
        if result.returncode != 0:
            return None, False
        payload = json.loads(result.stdout)
        # Only NVML answers; an R570+ MIG payload without ``cuda_devices`` has not answered either.
        answered = isinstance(payload, dict) and payload.get("source") == "nvml"
        if answered and not isinstance(payload.get("cuda_devices"), list):
            match = re.match(r"(\d+)\.", str(payload.get("driver_version") or ""))
            if (
                match
                and int(match[1]) >= 570
                and any(r.get("mig") for r in payload.get("devices") or [])
            ):
                answered = False
        return rows_from_payload(payload), answered
    except Exception:
        return None, False


def _refresh(key, raw):
    previous = _cache.get(key)
    rows, answered = _probe()
    if rows is not None:
        state = "rows"
    elif answered:
        state = "legacy"
    else:
        state = "failed"
        if previous is not None and previous[1] is not None:
            # Never collapse a known set to physical cards; keep identities, drop stale memory.
            rows = [{**r, "memory_free_mib": "0", "memory_total_mib": "0"} for r in previous[1]]
    _cache[key] = (time.monotonic(), rows, state)
    if key != _current_key():
        # Retired while it ran: the worker map belongs to the current key only.
        return rows
    if rows is not None:
        os.environ[_ENV] = json.dumps(
            {
                "key": [raw, os.environ.get("CUDA_DEVICE_ORDER")],
                "uuids": {str(r["index"]): r["uuid"] for r in rows},
            }
        )
    else:
        os.environ.pop(_ENV, None)
    return rows


def _inherited_uuids():
    """The parent's map if made under this process's mask and order, else None."""
    try:
        data = json.loads(os.environ.get(_ENV) or "null")
        if data["key"] != [
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            os.environ.get("CUDA_DEVICE_ORDER"),
        ]:
            return None
        return {int(k): str(v) for k, v in data["uuids"].items()}
    except Exception:
        return None


def mask_for(gpu_ids):
    """CUDA_VISIBLE_DEVICES pinning these ids on an R570+ MIG host, else None. Reads only the
    published map, never probes."""
    by_id = _inherited_uuids()
    return None if by_id is None else _pin(by_id, gpu_ids)


def rows_for(gpu_ids):
    """MIG rows for these ids in order, None off R570+ MIG; ValueError outside the visible set."""
    rows = visible_devices(fresh = True)
    if rows is None:
        return None
    by_id = {int(r["index"]): r for r in rows}
    ids = [int(i) for i in gpu_ids]
    if any(i not in by_id for i in ids):
        raise ValueError(f"GPU IDs {ids} are outside the visible CUDA device set {list(by_id)}")
    return [by_id[i] for i in ids]


def visibility_for(rows, gpu_ids):
    return _pin({int(r["index"]): r["uuid"] for r in rows}, gpu_ids)


def _pin(by_id, gpu_ids):
    ids = [int(i) for i in gpu_ids]
    if len(ids) != len(set(ids)) or any(i not in by_id for i in ids):
        raise ValueError(f"GPU IDs {ids} are outside the visible CUDA device set {list(by_id)}")
    return ",".join(by_id[i] for i in ids)
