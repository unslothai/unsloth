# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared GPU host-capability detection for the prebuilt installers.

Reads compute caps + driver CUDA version (honoring CUDA_VISIBLE_DEVICES) and the
ROCm gfx target so SM-appropriate bundle selection works, giving both installers
one probe. Lifted from install_llama_prebuilt.py.
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
    """True when every CUDA_VISIBLE_DEVICES token is an index or GPU-UUID we can map
    to a row (so an unmatched token means the GPU is hidden). A non-mappable token
    (e.g. a MIG UUID) returns False, keeping the host usable. From llama."""
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
    """Filter nvidia-smi (index, uuid, cap) rows by CUDA_VISIBLE_DEVICES, matching
    by index or GPU-UUID ('gpu-' prefix optional); unmatched tokens are skipped.
    Lifted from install_llama_prebuilt.py `select_visible_gpu_rows`."""
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
    """Detect visible NVIDIA GPUs, compute caps, and driver CUDA version via
    nvidia-smi, honoring CUDA_VISIBLE_DEVICES (so a hidden GPU reads as not usable)
    with a /proc/driver/nvidia/gpus fallback. Mirrors install_llama_prebuilt.py's
    NVIDIA block; all-empty/None when no NVIDIA GPU is present."""
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
                if not has_physical:  # older nvidia-smi lists caps but not -L
                    has_physical = True
            elif visible_tokens == []:
                has_usable = False
            elif supports_explicit_visible_device_matching(visible_tokens):
                has_usable = False  # index/UUID tokens matched nothing -> hidden
            elif has_physical:
                has_usable = True  # non-mappable token (MIG UUID) -> can't rule out

    # Linux /proc fallback: the driver exposes a subdir per GPU regardless of
    # nvidia-smi state, so an absent/wedged nvidia-smi is still recognised as
    # NVIDIA (caps/driver stay unset -> selection prefers portable). From llama.
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


def pick_rocm_gfx_target(output: str) -> str | None:
    """gfx target for the ACTIVE AMD GPU from rocminfo/hipinfo output. Honors
    HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES (empty/'-1'
    -> None, else the first GPU) so a mixed APU + dGPU host gets the arch HIP runs
    on. Splits per-GPU 'Agent N'/'device#N' sections (one token each, right for
    same-arch cards), else insertion-order dedup; the nonzero-first-digit regex
    skips the CPU agent and generic ISA. From install_llama_prebuilt.py."""
    sections = re.split(
        r"(?mi)^\s*\*+\s*$\s*agent\s+\d+\s*$|\bdevice\s*#\s*\d+\b",
        output,
    )
    if len(sections) > 1:
        tokens: list[str] = []
        for section in sections[1:]:
            match = re.search(r"gfx[1-9][0-9a-z]{2,3}", section.lower())
            if match:
                tokens.append(match.group(0))
    else:
        tokens = list(dict.fromkeys(re.findall(r"gfx[1-9][0-9a-z]{2,3}", output.lower())))

    if not tokens:
        return None

    visible_raw = None
    # AMD's HIP runtime honors all three env vars with identical semantics.
    for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        value = os.environ.get(name)
        if value is not None:
            visible_raw = value
            break
    if visible_raw is not None:
        visible = visible_raw.strip()
        if visible == "" or visible == "-1":
            return None
        first = visible.split(",")[0].strip()
        try:
            index = int(first)
            if 0 <= index < len(tokens):
                return tokens[index]
        except ValueError:
            pass
    return tokens[0]


def parse_macos_version(value: str | None) -> tuple[int, int] | None:
    """Parse a macOS version ("14.7.1", "15.5", "26") into (major, minor), or None
    when unparseable (callers then defer to runtime validation). From llama."""
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
    """The CUDA runtime line torch is built against ('cuda12'/'cuda13'), a tie-break
    on non-Blackwell hosts; None if torch is absent/CPU-only/unsupported.
    Best-effort, never raises. From llama's detect_torch_cuda_runtime_preference."""
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
