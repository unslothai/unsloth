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
import time

from . import gpu_query, nvidia
from utils.child_stdio import utf8_child_env
from utils.native_path_leases import child_env_without_native_path_secret
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

_cache = {}
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
        rows.append({**row, "physical_index": row["index"],
                     "index": str(ids[ordinal] if ids is not None else ordinal)})
    return rows


def _mig_instances(listing):
    """Physical cards with MIG instances, from the ``nvidia-smi -L`` the NVIDIA paths already
    read. Only ``listing`` callers (GGUF placement, which lists on every host anyway) may start
    one; the rest reuse a fresh answer, and skip when the last answer of any age found none."""
    if listing:
        return nvidia.get_mig_gpu_indices_or_none()
    cached = nvidia._mig_cached()
    if cached is not nvidia._MISS:
        return cached
    entry = nvidia._mig_entry
    if entry is not None and entry[3] is not None and not entry[3]:
        return None
    return nvidia.get_mig_gpu_indices_or_none()


def visible_devices(*, fresh=False, listing=False):
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
    key = (raw, os.environ.get("CUDA_DEVICE_ORDER"), gpu_query._static_gen,
           os.environ.get("UNSLOTH_NVIDIA_PROBE"))
    cached = _cache.get(key)
    if cached is not None and time.monotonic() - cached[0] < (5 if cached[1] is None else 60):
        if not fresh or cached[1] is None:
            return cached[1]
    rows = None
    try:
        from utils.prebuilt.update_flow import find_installer_script
        script = find_installer_script(env_var="UNSLOTH_NVIDIA_PROBE", script_name="nvidia_probe.py")
        if script is not None:
            result = subprocess.run(
                [sys.executable, "-I", str(script), "--json"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
                env=utf8_child_env(child_env_without_native_path_secret()),
                **windows_hidden_subprocess_kwargs(),
            )
            if result.returncode == 0:
                rows = rows_from_payload(json.loads(result.stdout))
    except Exception:
        pass
    if rows is None and cached is not None and cached[1] is not None:
        # A failed memory/identity refresh cannot collapse a known device set to
        # physical cards. Retain identities, but never reuse stale free memory.
        rows = [{**r, "memory_free_mib": "0", "memory_total_mib": "0"} for r in cached[1]]
    _cache[key] = (time.monotonic(), rows)
    if rows is not None:
        os.environ[_ENV] = json.dumps(
            {"key": [raw, os.environ.get("CUDA_DEVICE_ORDER")],
             "uuids": {str(r["index"]): r["uuid"] for r in rows}}
        )
    else:
        os.environ.pop(_ENV, None)
    return rows


def _inherited_uuids():
    """The parent's map when it was made under this process's own mask and order, else None."""
    try:
        data = json.loads(os.environ.get(_ENV) or "null")
        if data["key"] != [os.environ.get("CUDA_VISIBLE_DEVICES"), os.environ.get("CUDA_DEVICE_ORDER")]:
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


def visibility_for(rows, gpu_ids):
    """Pin only UUIDs in the current parent's enumerated device set."""
    return _pin({int(r["index"]): r["uuid"] for r in rows}, gpu_ids)


def _pin(by_id, gpu_ids):
    ids = [int(i) for i in gpu_ids]
    if len(ids) != len(set(ids)) or any(i not in by_id for i in ids):
        raise ValueError(f"GPU IDs {ids} are outside the visible CUDA device set {list(by_id)}")
    return ",".join(by_id[i] for i in ids)
