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


def _select_visible_rows(
    rows: list[tuple[str, str, str]], visible_devices: list[str] | None
) -> list[tuple[str, str, str]]:
    """Filter nvidia-smi (index, uuid, compute_cap) rows by CUDA_VISIBLE_DEVICES,
    matching by index or GPU-UUID token. Non-index/uuid selectors (unusual) leave
    all rows visible."""
    if visible_devices is None:
        return rows
    if not visible_devices:
        return []
    # A non-index/non-uuid selector means we can't map deterministically; keep all.
    for token in visible_devices:
        lowered = token.lower()
        if not (token.isdigit() or lowered.startswith("gpu-")):
            return rows
    by_index = {index: (index, uuid, cap) for index, uuid, cap in rows}
    by_uuid = {uuid.lower(): (index, uuid, cap) for index, uuid, cap in rows}
    selected: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for token in visible_devices:
        row = by_index.get(token) or by_uuid.get(token.lower())
        if row is not None and row[0] not in seen:
            seen.add(row[0])
            selected.append(row)
    return selected


@dataclass
class NvidiaCaps:
    has_usable_nvidia: bool
    compute_caps: list[str]
    driver_cuda_version: tuple[int, int] | None


def detect_nvidia_caps() -> NvidiaCaps:
    """Detect visible NVIDIA GPUs, their compute caps, and the driver's CUDA
    version via nvidia-smi, honoring CUDA_VISIBLE_DEVICES. All-empty/None when no
    usable NVIDIA GPU is present."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return NvidiaCaps(has_usable_nvidia = False, compute_caps = [], driver_cuda_version = None)

    visible_tokens = parse_cuda_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    has_usable = False

    # -L must actually list a GPU: the banner prints even when the driver can't be
    # reached, which would misclassify an AMD/ROCm host as NVIDIA.
    listing = _run([nvidia_smi, "-L"])
    if listing is not None:
        gpu_lines = [line for line in listing.stdout.splitlines() if line.startswith("GPU ")]
        if gpu_lines:
            has_usable = visible_tokens != []

    driver_cuda_version: tuple[int, int] | None = None
    banner = _run([nvidia_smi])
    if banner is not None:
        merged = "\n".join(part for part in (banner.stdout, banner.stderr) if part)
        # Newer drivers print "CUDA UMD Version: X.Y" instead of "CUDA Version: X.Y".
        match = re.search(r"CUDA(?: UMD)? Version:\s*(\d+)\.(\d+)", merged)
        if match is not None:
            driver_cuda_version = (int(match.group(1)), int(match.group(2)))

    compute_caps: list[str] = []
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
        elif visible_tokens == []:
            has_usable = False

    return NvidiaCaps(
        has_usable_nvidia = has_usable,
        compute_caps = compute_caps,
        driver_cuda_version = driver_cuda_version,
    )


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
