# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Decide GPU offload from three probes; a CPU fallback is silent, so text output proves nothing."""

from __future__ import annotations

import json
import re
from pathlib import Path

# CUDA context plus scratch alone is tens of MiB, so less than this is not real offload.
MIN_PROCESS_VRAM_MIB = 96

# Device-wide growth across the load that no CPU-resident model explains.
MIN_DEVICE_VRAM_DELTA_MIB = 256

# Checked before loading: a truncated file passes os.path.exists.
GGUF_MAGIC = b"GGUF"

_OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers?\s+to\s+GPU")

_CUDA_BUFFER_RE = re.compile(
    r"(CUDA\d+|ROCm\d+)\s+model buffer size\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*MiB",
    re.IGNORECASE,
)

CUDA_INSTALL_KINDS = frozenset({"linux-cuda", "linux-arm64-cuda"})

# Anchored on the digit so "cudart" or a repo name containing "cuda" cannot match.
_CUDA_RUNTIME_RE = re.compile(r"cuda\d+")


def parse_compute_apps(csv_text: str) -> dict[int, int]:
    """Parse compute-apps CSV to {pid: MiB}; a unit suffix is accepted, since getting it wrong yields {}."""
    apps: dict[int, int] = {}
    for line in csv_text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("pid"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        mem = parts[1].replace("MiB", "").replace("MB", "").strip()
        if mem.lower() in ("[n/a]", "n/a", ""):
            continue
        try:
            apps[pid] = int(float(mem))
        except ValueError:
            continue
    return apps


def count_listed_pids(csv_text: str) -> int:
    """Count pids nvidia-smi listed, with or without a memory figure; [N/A] rows cannot be attributed."""
    n = 0
    for line in csv_text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("pid"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if not parts:
            continue
        try:
            int(parts[0])
        except ValueError:
            continue
        n += 1
    return n


def listed_pids(csv_text: str) -> set[int]:
    """Every pid listed, figure or not; a mixed listing can hide the one process being checked."""
    pids: set[int] = set()
    for line in csv_text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("pid"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if not parts:
            continue
        try:
            pids.add(int(parts[0]))
        except ValueError:
            continue
    return pids


def offloaded_layers(log_text: str) -> tuple[int, int] | None:
    """The last 'offloaded N/M layers to GPU' line, since a session may load the model more than once."""
    matches = _OFFLOAD_RE.findall(log_text or "")
    if not matches:
        return None
    offloaded, total = matches[-1]
    return int(offloaded), int(total)


def cuda_buffer_mib(log_text: str) -> float | None:
    """Largest device model-buffer allocation llama.cpp reported, in MiB."""
    matches = _CUDA_BUFFER_RE.findall(log_text or "")
    if not matches:
        return None
    return max(float(size) for _, size in matches)


def install_kind(marker_path: Path | None) -> str | None:
    """Read runtime_line from the marker, not install_kind (absent from it); asset is the fallback."""
    if marker_path is None:
        return None
    try:
        payload = json.loads(Path(marker_path).read_text(encoding = "utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(payload, dict):
        return None
    runtime_line = payload.get("runtime_line")
    if runtime_line:
        return str(runtime_line)
    asset = payload.get("asset")
    return str(asset) if asset else None


def is_cuda_install(kind: str | None) -> bool:
    """Match any cuda runtime line, not a fixed set, so a new CUDA major never reads as non-CUDA."""
    if not kind:
        return False
    lowered = str(kind).lower()
    return bool(_CUDA_RUNTIME_RE.search(lowered)) or lowered in CUDA_INSTALL_KINDS


# Most specific first; the canonical location is install_llama_prebuilt.py's default.
def llama_cpp_marker(studio_home: Path) -> Path | None:
    """Marker in ~/.unsloth/llama.cpp (where the installer writes) or studio_home/llama.cpp; else None."""
    candidates = (
        Path(studio_home) / "llama.cpp" / "UNSLOTH_PREBUILT_INFO.json",
        Path.home() / ".unsloth" / "llama.cpp" / "UNSLOTH_PREBUILT_INFO.json",
    )
    for marker in candidates:
        if marker.is_file():
            return marker
    return None


def gguf_magic_ok(path: Path) -> bool:
    """Does this file start with GGUF's magic? Cheap, and catches a truncation."""
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == GGUF_MAGIC
    except OSError:
        return False


def offload_verdict(
    *,
    server_pid: int | None,
    compute_apps: dict[int, int] | None,
    log_text: str,
    device_vram_delta_mib: float | None,
    status: dict | None,
    server_pids: list[int] | None = None,
) -> dict:
    """Negative signals: cpu_fallback_reason, or gpu_layers of 0; gpu_layers -1 (Auto) is neutral."""
    evidence: list[str] = []
    failures: list[str] = []
    positives: list[str] = []

    status = status or {}
    fallback = status.get("cpu_fallback_reason")
    if fallback:
        failures.append(f"Unsloth reported a CPU fallback: cpu_fallback_reason={fallback!r}")

    effective_layers = status.get("gpu_layers")
    if isinstance(effective_layers, int):
        evidence.append(f"status.gpu_layers={effective_layers}")
        if effective_layers == 0:
            failures.append(
                "Unsloth reports gpu_layers=0, so nothing was placed on the GPU "
                "even though the load asked for it"
            )

    counts = offloaded_layers(log_text)
    if counts is None:
        evidence.append("llama.cpp log: no offload line found")
    else:
        offloaded, total = counts
        evidence.append(f"llama.cpp log: offloaded {offloaded}/{total} layers to GPU")
        if offloaded <= 0:
            failures.append(
                f"llama.cpp offloaded {offloaded}/{total} layers, which is the CPU path"
            )
        else:
            positives.append(f"llama.cpp offloaded {offloaded}/{total} layers")

    buffer_mib = cuda_buffer_mib(log_text)
    if buffer_mib is not None:
        evidence.append(f"llama.cpp device model buffer: {buffer_mib:.0f} MiB")

    # The status body carries no pid, so discovered pids are accepted too.
    candidates: list[int] = []
    for pid in [server_pid, *(server_pids or [])]:
        if isinstance(pid, int) and pid not in candidates:
            candidates.append(pid)

    if compute_apps is None:
        evidence.append("nvidia-smi compute-apps: unreadable")
    elif not compute_apps:
        evidence.append("nvidia-smi compute-apps: no process listed")
    elif not candidates:
        evidence.append(
            f"nvidia-smi compute-apps: {len(compute_apps)} process(es) listed, but "
            f"no llama-server pid was found to match them against"
        )
    else:
        matched = {pid: compute_apps[pid] for pid in candidates if pid in compute_apps}
        if not matched:
            evidence.append(
                f"nvidia-smi compute-apps: llama-server pid(s) "
                f"{', '.join(str(p) for p in candidates)} are not among the "
                f"{len(compute_apps)} process(es) holding GPU memory"
            )
        for pid, used in matched.items():
            evidence.append(f"nvidia-smi compute-apps: pid {pid} holds {used} MiB")
            if used >= MIN_PROCESS_VRAM_MIB:
                positives.append(f"llama-server pid {pid} holds {used} MiB of VRAM")
            else:
                evidence.append(
                    f"that is below the {MIN_PROCESS_VRAM_MIB} MiB floor, so it is "
                    f"consistent with a bare CUDA context and no weights"
                )

    if device_vram_delta_mib is None:
        evidence.append("device VRAM delta: unreadable")
    else:
        evidence.append(f"device VRAM delta across the load: {device_vram_delta_mib:.0f} MiB")
        if device_vram_delta_mib >= MIN_DEVICE_VRAM_DELTA_MIB:
            positives.append(f"device VRAM in use grew by {device_vram_delta_mib:.0f} MiB")

    if not failures and not positives:
        failures.append(
            "no probe could show the GPU was used: the process was not visible to "
            "nvidia-smi, llama.cpp logged no offload line, and device VRAM did not "
            "move. Text was returned, but nothing here distinguishes that from a "
            "CPU fallback, so this is a failure rather than a pass"
        )

    return {
        "passed": not failures,
        "failures": failures,
        "positives": positives,
        "evidence": evidence,
    }
