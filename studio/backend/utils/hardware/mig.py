# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""R570 MIG device identities. CUDA enumerates; NVML only supplies memory.

A physical card index and a CUDA device ordinal are separate namespaces. Keep the
UUID with each ordinal so a worker can reproduce exactly the set that was sized.
"""

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

# key -> (monotonic time, rows, state); state is "rows", "legacy" (the probe answered: keep the
# old rule) or "failed" (no answer; rows may hold retained identities with zero memory).
_cache = {}
_flights = {}
_flights_lock = threading.Lock()
_FAILED_BACKOFF_S = 5.0
_ROWS_TTL_S = 60.0
# A fresh sample this young is shared: polling callers never start more than one child per window.
_FRESH_SHARE_S = 2.0
# The parent's ordinal -> MIG uuid map, inherited by spawned workers: their apply_gpu_ids runs
# before any probe there, so it pins the exact instances the parent sized without re-probing.
_ENV = "UNSLOTH_MIG_CUDA_DEVICES"


def possible() -> bool:
    """MIG runs only on bare-metal Linux with the NVIDIA kernel driver loaded (not WSL), so
    every other host skips the listing and the probe without starting a process."""
    return sys.platform == "linux" and os.path.isfile("/proc/driver/nvidia/version")


def rows_from_payload(payload):
    """None means legacy/unknown; [] means CUDA confirmed an empty visible set.

    Use the driver's actual UUID enumeration, rather than counting MIG child rows:
    several CIs can share a GI (and its memory), and containers may expose only some.
    """
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
    """Physical cards with MIG instances, from the ``nvidia-smi -L`` the NVIDIA paths already
    read. ``listing`` callers (GGUF placement, which lists on every host anyway) always may start
    one. The rest reuse a fresh answer, skip when the last answer of any age found none, and
    start one only when no listing has run this hardware generation, so the first read on a MIG
    host already uses the instance namespace instead of switching it after the first placement."""
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
    """Bounded out-of-process probe, gated off ordinary hosts; cache identities only.

    Memory callers request a fresh sample. Failed probes back off and retire at the
    hardware generation boundary, like the other NVIDIA inventory probes.
    """
    if not possible() or os.environ.get("UNSLOTH_NVIDIA_LIBRARY_PROBE", "1") == "0":
        return None
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not (raw and "mig-" in raw.lower()) and not _mig_instances(listing):
        # The cached ``nvidia-smi -L`` the NVIDIA paths already read: no MIG instances (or
        # unknown) keeps the existing rule, and the NVML probe never starts.
        os.environ.pop(_ENV, None)
        return None
    key = _current_key()
    hit = _cached_answer(key, fresh)
    if hit is not _MISS:
        return hit
    # One probe child per key at a time: concurrent callers wait for it and share its answer.
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
        # A failed probe backs off even for fresh callers, retained identities or not.
        return entry[1] if age < _FAILED_BACKOFF_S else _MISS
    if state == "legacy":
        # The probe answered that the old rule applies; nothing here carries memory to refresh.
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
        # Only NVML answers the MIG question; the CUDA fallback means NVML itself failed. An R570+
        # MIG host whose strict CUDA enumeration failed (no ``cuda_devices``) has not answered either.
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
            # A failed memory/identity refresh cannot collapse a known device set to
            # physical cards. Retain identities, but never reuse stale free memory.
            rows = [{**r, "memory_free_mib": "0", "memory_total_mib": "0"} for r in previous[1]]
    _cache[key] = (time.monotonic(), rows, state)
    if key != _current_key():
        # Retired while it ran (re-detection, a new mask): the worker map belongs to the
        # current key's probe, never to this one.
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
    """The parent's map when it was made under this process's own mask and order, else None."""
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
    """The CUDA_VISIBLE_DEVICES value pinning these CUDA device ids on an R570+ MIG host, or
    None when this is not one (callers keep their own rule). Workers call this before anything
    else, so it reads only the map the parent published and never starts a probe."""
    by_id = _inherited_uuids()
    return None if by_id is None else _pin(by_id, gpu_ids)


def rows_for(gpu_ids):
    """The current MIG rows for these CUDA device ids, in order, or None when this is not an R570+
    MIG host. Raises ValueError for an id outside the visible set."""
    rows = visible_devices(fresh = True)
    if rows is None:
        return None
    by_id = {int(r["index"]): r for r in rows}
    ids = [int(i) for i in gpu_ids]
    if any(i not in by_id for i in ids):
        raise ValueError(f"GPU IDs {ids} are outside the visible CUDA device set {list(by_id)}")
    return [by_id[i] for i in ids]


def visibility_for(rows, gpu_ids):
    """Pin only UUIDs in the current parent's enumerated device set."""
    return _pin({int(r["index"]): r["uuid"] for r in rows}, gpu_ids)


def _pin(by_id, gpu_ids):
    ids = [int(i) for i in gpu_ids]
    if len(ids) != len(set(ids)) or any(i not in by_id for i in ids):
        raise ValueError(f"GPU IDs {ids} are outside the visible CUDA device set {list(by_id)}")
    return ",".join(by_id[i] for i in ids)
