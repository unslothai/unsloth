# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared GPU host-capability detection for the prebuilt installers.

whisper's original `detect_host` only recorded a boolean `has_usable_nvidia`; it
never read the GPU compute capabilities or the driver's CUDA version, so it could
not select an SM-appropriate CUDA bundle. This module provides the detection
llama already does -- compute caps + driver CUDA version from nvidia-smi, honoring
CUDA_VISIBLE_DEVICES -- so both installers share one probe.

Lifted from install_llama_prebuilt.py (`detect_host` NVIDIA block,
`parse_cuda_visible_devices`, `select_visible_gpu_rows`,
`detect_torch_cuda_runtime_preference`).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

from .selection import normalize_compute_cap


def _windows_hidden_kwargs() -> dict:
    """Suppress a console window for child probes on Windows; no-op elsewhere."""
    if sys.platform.startswith("win"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        return {"startupinfo": startupinfo, "creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def _run(args: list[str], timeout: int = 20) -> subprocess.CompletedProcess | None:
    """Best-effort subprocess capture; None on any failure (never raises)."""
    try:
        return subprocess.run(
            args,
            capture_output = True,
            text = True,
            timeout = timeout,
            **_windows_hidden_kwargs(),
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def parse_cuda_visible_devices(value: str | None) -> list[str] | None:
    """Parse CUDA_VISIBLE_DEVICES. None => not set (all visible); [] => none
    visible ('' or '-1'); else the ordered token list."""
    if value is None:
        return None
    raw = value.strip()
    if not raw or raw == "-1":
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def supports_explicit_visible_device_matching(visible_devices: list[str] | None) -> bool:
    """True when every CUDA_VISIBLE_DEVICES token is an index or a GPU-UUID
    ('gpu-...') selector -- tokens we can map deterministically to nvidia-smi rows.
    When such tokens match no row, the GPU is explicitly hidden. A non-index/UUID
    selector (e.g. a MIG-only UUID we can't enumerate) returns False so the caller
    keeps the host usable rather than declaring it hidden. Lifted from
    install_llama_prebuilt.py."""
    if not visible_devices:
        return False
    for token in visible_devices:
        lowered = token.lower()
        if token.isdigit() or lowered.startswith("gpu-"):
            continue
        return False
    return True


def _select_visible_rows(
    rows: list[tuple[str, str, str]], visible_devices: list[str] | None
) -> list[tuple[str, str, str]]:
    """Filter nvidia-smi (index, uuid, compute_cap) rows by CUDA_VISIBLE_DEVICES,
    matching by index or GPU-UUID token (with the 'gpu-' prefix optional). Tokens
    that map to no row are skipped -- the host-usable decision for non-mappable
    selectors is made by the caller via supports_explicit_visible_device_matching,
    not here. Lifted from install_llama_prebuilt.py `select_visible_gpu_rows`."""
    if visible_devices is None:
        return list(rows)
    if not visible_devices:
        return []
    by_index = {index: (index, uuid, cap) for index, uuid, cap in rows}
    by_uuid = {uuid.lower(): (index, uuid, cap) for index, uuid, cap in rows}
    selected: list[tuple[str, str, str]] = []
    seen_indices: set[str] = set()
    for token in visible_devices:
        row = by_index.get(token)
        if row is None:
            normalized_token = token.lower()
            row = by_uuid.get(normalized_token)
            if row is None and not normalized_token.startswith("gpu-"):
                row = by_uuid.get("gpu-" + normalized_token)
        if row is None:
            continue
        index = row[0]
        if index in seen_indices:
            continue
        seen_indices.add(index)
        selected.append(row)
    return selected


@dataclass
class NvidiaCaps:
    has_usable_nvidia: bool
    compute_caps: list[str]
    driver_cuda_version: tuple[int, int] | None
    has_physical_nvidia: bool = False


def detect_nvidia_caps(*, is_linux: bool | None = None) -> NvidiaCaps:
    """Detect visible NVIDIA GPUs, their compute caps, and the driver's CUDA
    version via nvidia-smi, honoring CUDA_VISIBLE_DEVICES. Mirrors
    install_llama_prebuilt.py `detect_host`'s NVIDIA block, including the
    explicit-visible-device-matching decision (so `CUDA_VISIBLE_DEVICES=7` on a
    single-GPU host reports the GPU hidden, not usable) and the Linux
    /proc/driver/nvidia/gpus fallback for an absent/wedged nvidia-smi. All-empty /
    None when no NVIDIA GPU is present."""
    if is_linux is None:
        is_linux = sys.platform.startswith("linux")
    nvidia_smi = shutil.which("nvidia-smi")
    visible_tokens = parse_cuda_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    has_physical = False
    has_usable = False
    driver_cuda_version: tuple[int, int] | None = None
    compute_caps: list[str] = []

    if nvidia_smi:
        # -L must actually list a GPU: the banner prints even when the driver can't
        # be reached, which would misclassify an AMD/ROCm host as NVIDIA.
        listing = _run([nvidia_smi, "-L"])
        if listing is not None:
            gpu_lines = [line for line in listing.stdout.splitlines() if line.startswith("GPU ")]
            if gpu_lines:
                has_physical = True
                has_usable = visible_tokens != []

        banner = _run([nvidia_smi])
        if banner is not None:
            merged = "\n".join(part for part in (banner.stdout, banner.stderr) if part)
            # Newer drivers print "CUDA UMD Version: X.Y" instead of "CUDA Version: X.Y".
            match = re.search(r"CUDA(?: UMD)? Version:\s*(\d+)\.(\d+)", merged)
            if match is not None:
                driver_cuda_version = (int(match.group(1)), int(match.group(2)))

        caps = _run([nvidia_smi, "--query-gpu=index,uuid,compute_cap", "--format=csv,noheader"])
        if caps is not None:
            rows: list[tuple[str, str, str]] = []
            for raw in caps.stdout.splitlines():
                parts = [part.strip() for part in raw.split(",")]
                if len(parts) == 3:
                    rows.append((parts[0], parts[1], parts[2]))
            visible_rows = _select_visible_rows(rows, visible_tokens)
            for _index, _uuid, cap in visible_rows:
                normalized = normalize_compute_cap(cap)
                if normalized is not None and normalized not in compute_caps:
                    compute_caps.append(normalized)
            if visible_rows:
                has_usable = True
                # Older nvidia-smi (pre -L) misses the -L block but still lists
                # caps here; keep has_physical consistent for downstream routing.
                if not has_physical:
                    has_physical = True
            elif visible_tokens == []:
                has_usable = False
            elif supports_explicit_visible_device_matching(visible_tokens):
                # All tokens are index/UUID selectors but none matched a row: the
                # GPU(s) are explicitly hidden from this process.
                has_usable = False
            elif has_physical:
                # Non-mappable selector (e.g. a MIG UUID we can't enumerate) on a
                # host with a physical NVIDIA GPU: can't rule the GPU out.
                has_usable = True

    # Linux /proc fallback: the driver exposes one subdir per GPU regardless of
    # nvidia-smi state, so a host whose nvidia-smi is absent/wedged is still
    # recognised as NVIDIA. compute_caps / driver_cuda_version stay unset here, so
    # downstream CUDA selection treats unknown SMs as "prefer portable" and an
    # unknown driver line as "no CUDA match". Mirrors install_llama_prebuilt.py.
    if is_linux and not has_physical:
        try:
            proc_gpu_dir = "/proc/driver/nvidia/gpus"
            if os.path.isdir(proc_gpu_dir) and os.listdir(proc_gpu_dir):
                has_physical = True
                has_usable = visible_tokens != []
        except OSError:
            pass

    return NvidiaCaps(
        has_usable_nvidia = has_usable,
        compute_caps = compute_caps,
        driver_cuda_version = driver_cuda_version,
        has_physical_nvidia = has_physical,
    )


def parse_macos_version(value: str | None) -> tuple[int, int] | None:
    """Parse a macOS product version string into (major, minor). Handles
    "14.7.1", "15.5", "26.0" and bare "26". Returns None when empty or unparseable
    (callers then defer to runtime validation rather than rejecting a prebuilt).
    Lifted from install_llama_prebuilt.py."""
    if not value:
        return None
    match = re.match(r"\s*(\d+)(?:\.(\d+))?", str(value))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def _runtime_line_from_cuda_version(cuda_version: str | None) -> str | None:
    if not cuda_version:
        return None
    major, _, _ = str(cuda_version).strip().partition(".")
    if major == "12":
        return "cuda12"
    if major == "13":
        return "cuda13"
    return None


def detect_torch_cuda_runtime_line() -> str | None:
    """The CUDA runtime line torch is built against ('cuda12'/'cuda13'), used as a
    tie-break preference on non-Blackwell hosts. None if torch is absent, has no
    CUDA build, reports CUDA unavailable, or reports an unsupported major. Lifted
    from llama `detect_torch_cuda_runtime_preference` (best-effort, never raises)."""
    try:
        import torch
    except Exception:
        return None
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    if not isinstance(cuda_version, str) or not cuda_version.strip():
        return None
    try:
        if not bool(torch.cuda.is_available()):
            return None
    except Exception:
        return None
    return _runtime_line_from_cuda_version(cuda_version)
