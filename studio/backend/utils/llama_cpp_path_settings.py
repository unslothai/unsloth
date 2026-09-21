# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persist and validate the llama.cpp directory selected in Unsloth settings."""

from __future__ import annotations

import os
import re
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

CUSTOM_LLAMA_CPP_PATH_SETTING_KEY = "custom_llama_cpp_path"
MAX_CUSTOM_LLAMA_CPP_PATH_LENGTH = 32767
MANAGED_LLAMA_CPP_PATH_MARKER = "UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH"

_settings_lock = threading.RLock()
_path_revision = 0


def mark_managed_llama_cpp_path(directory: Path | str) -> bool:
    """Mark Unsloth's inherited install path without hiding a real env override."""
    configured = os.environ.get("UNSLOTH_LLAMA_CPP_PATH", "").strip()
    if not configured:
        os.environ.pop(MANAGED_LLAMA_CPP_PATH_MARKER, None)
        return False
    try:
        managed = Path(directory).expanduser().resolve(strict = False)
        inherited = Path(configured).expanduser().resolve(strict = False)
        is_managed = inherited == managed
    except (OSError, RuntimeError, ValueError):
        is_managed = False
    if is_managed:
        os.environ[MANAGED_LLAMA_CPP_PATH_MARKER] = "1"
    else:
        os.environ.pop(MANAGED_LLAMA_CPP_PATH_MARKER, None)
    return is_managed


@contextmanager
def llama_cpp_path_selection_guard() -> Iterator[None]:
    """Serialize a runtime path snapshot with a settings write.

    Model loads and UI saves share this lock so reload status sees one snapshot.
    """
    with _settings_lock:
        yield


def llama_server_binary_name(platform: Optional[str] = None) -> str:
    return "llama-server.exe" if (platform or sys.platform) == "win32" else "llama-server"


def llama_server_candidates(
    directory: Path | str, *, platform: Optional[str] = None
) -> tuple[Path, ...]:
    """Supported llama.cpp build layouts, in the runtime's search order."""
    root = Path(directory)
    binary_name = llama_server_binary_name(platform)
    windows = (platform or sys.platform) == "win32"
    candidates = [root / binary_name]
    # build/ first, then the per-backend build dirs a source checkout keeps side by side (#5941).
    for build in ("build", "build-cuda", "build-hip", "build-rocm", "build-vulkan"):
        candidates.append(root / build / "bin" / binary_name)
        if windows:
            candidates.append(root / build / "bin" / "Release" / binary_name)
    return tuple(candidates)


def _usable_binary(path: Path, *, platform: Optional[str] = None) -> bool:
    try:
        if not path.is_file():
            return False
    except OSError:
        return False
    return (platform or sys.platform) == "win32" or os.access(path, os.X_OK)


_GPU_BACKEND_LIB_RE = re.compile(
    r"^(?:lib)?ggml-(cann|cuda|hip|metal|musa|opencl|sycl|virtgpu|vulkan)"
    r"(?:\.dll|\.so(?:\.\d+)*|(?:\.\d+)*\.dylib)$"
)
_CPU_BACKEND_LIB_RE = re.compile(r"^(?:lib)?ggml-(?:cpu|base)(?:[-.]|$)")
# Backends bound to one vendor; vulkan, opencl and virtgpu run on any GPU.
_BACKEND_VENDOR = {
    "cuda": "nvidia",
    "hip": "amd",
    "sycl": "intel",
    "metal": "apple",
    "musa": "mthreads",
    "cann": "huawei",
}
_DRM_VENDOR = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel"}
_WINDOWS_VENDOR_DLLS = {
    "nvidia": ("nvcuda.dll",),
    "amd": ("amdhip64*.dll", "amdocl64.dll"),
    "intel": ("ze_intel_gpu64.dll", "igdrcl64.dll"),
}
_DRM_ROOT = "/sys/class/drm"
# WSL2 reaches the GPU through /dev/dxg (no DRM card, no /dev/kfd or /dev/nvidiactl): the
# vendor is the runtime that drives it, librocdxg for AMD and the WSL libcuda for NVIDIA.
_WSL_ROCM_LIB_DIRS = ("/opt/rocm/lib", "/opt/rocm/lib64")
_WSL_CUDA_LIB_DIR = "/usr/lib/wsl/lib"
_HOST: Any = object()  # prefer_gpu_capable's default: read the vendors from this host


def binary_gpu_backends(binary: Path | str) -> Optional[set[str]]:
    """GPU backend libraries beside the binary; empty for a split build with only CPU
    libraries, None when the layout says nothing (a static build or a wrapper)."""
    try:
        names = [p.name for p in Path(binary).resolve().parent.iterdir() if p.is_file()]
    except OSError:
        return None
    gpu = {m.group(1) for m in map(_GPU_BACKEND_LIB_RE.match, names) if m}
    if gpu or any(_CPU_BACKEND_LIB_RE.match(name) for name in names):
        return gpu
    return None


def binary_gpu_verdict(binary: Path | str) -> str:
    """``gpu``, ``cpu`` (a split-library build with no GPU backend beside it) or ``unknown``."""
    backends = binary_gpu_backends(binary)
    return "unknown" if backends is None else ("gpu" if backends else "cpu")


def _mask_hides_all(*names: str) -> bool:
    """Whether the first set mask among names (their precedence order) hides every device."""
    mask = next((os.environ[n] for n in names if n in os.environ), "x")
    return mask.strip() in ("", "-1")


def _amd_mask_hides_all() -> bool:
    """HIP_ and ROCR_VISIBLE_DEVICES stack (ROCR filters the agents HIP then indexes), so an
    empty one at either layer hides everything; CUDA_VISIBLE_DEVICES stands in only when
    neither is set (_active_gpu_visibility_mask)."""
    if any(n in os.environ for n in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")):
        return _mask_hides_all("HIP_VISIBLE_DEVICES") or _mask_hides_all("ROCR_VISIBLE_DEVICES")
    return _mask_hides_all("CUDA_VISIBLE_DEVICES")


def host_gpu_vendors() -> Optional[set[str]]:
    """GPU vendors this process can reach: DRM sysfs vendors whose device node is present and
    whose visibility mask is not empty on Linux, the vendors' driver DLLs on Windows. None when
    nothing was detected at all (an unknown host filters nothing); an empty set when GPUs were
    detected but none is reachable, so only vendor-agnostic backends fit."""
    vendors: set[str] = set()
    if sys.platform == "darwin":
        return {"apple"}
    if sys.platform == "win32":
        system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
        for vendor, patterns in _WINDOWS_VENDOR_DLLS.items():
            if any(next(system32.glob(pattern), None) for pattern in patterns):
                vendors.add(vendor)
    else:
        for vendor_file in Path(_DRM_ROOT).glob("card*/device/vendor"):
            try:
                vendor = _DRM_VENDOR.get(vendor_file.read_text(encoding = "utf-8").strip().lower())
            except OSError:
                continue
            if vendor:
                vendors.add(vendor)
        if os.path.isdir("/proc/driver/nvidia"):
            vendors.add("nvidia")
        wsl = os.path.exists("/dev/dxg")
        wsl_amd = wsl and any(
            os.path.exists(os.path.join(d, n))
            for d in _WSL_ROCM_LIB_DIRS
            for n in ("librocdxg.so", "librocdxg.so.1")
        )
        wsl_nvidia = wsl and any(Path(_WSL_CUDA_LIB_DIR).glob("libcuda.so*"))
        vendors |= {v for v, on in (("amd", wsl_amd), ("nvidia", wsl_nvidia)) if on}
        detected = set(vendors)
        # A container that exposes only one vendor's device nodes, or a mask hiding a vendor,
        # must not make its backend look runnable on a hybrid box.
        if "nvidia" in vendors and (
            not (os.path.exists("/dev/nvidiactl") or wsl_nvidia)
            or _mask_hides_all("CUDA_VISIBLE_DEVICES")
        ):
            vendors.discard("nvidia")
        if "amd" in vendors and (
            not (os.path.exists("/dev/kfd") or wsl_amd) or _amd_mask_hides_all()
        ):
            vendors.discard("amd")
        return vendors if detected else None
    return vendors or None


def _fits_host(backends: set[str], vendors: Optional[set[str]]) -> bool:
    if vendors is None:
        return True
    return any(_BACKEND_VENDOR.get(b) is None or _BACKEND_VENDOR[b] in vendors for b in backends)


def prefer_gpu_capable(
    candidates: Iterable[Path],
    usable: Callable[[Path], bool],
    vendors: Any = _HOST,
) -> list[Path]:
    """Search order, except that a first hit proven CPU-only, or proven built for a GPU vendor
    this host lacks, yields to a later usable build shipping a GPU backend this host's vendor
    runs (#5941: a CPU build/ beside build-cuda/; a stale build-cuda/ must not shadow
    build-hip/ on an AMD box, whether or not a build/ precedes it). Unknown layouts stay put."""
    ordered = list(candidates)
    first = next((c for c in ordered if usable(c)), None)
    if first is None:
        return ordered
    verdict = binary_gpu_verdict(first)
    if verdict == "unknown":
        return ordered
    if vendors is _HOST:
        vendors = host_gpu_vendors()
    if verdict == "gpu" and _fits_host(binary_gpu_backends(first), vendors):
        return ordered
    for candidate in ordered:
        if candidate == first or not usable(candidate):
            continue
        backends = binary_gpu_backends(candidate)
        if backends and _fits_host(backends, vendors):
            return [candidate] + [c for c in ordered if c != candidate]
    return ordered


def resolve_llama_server_binary(
    directory: Path | str, *, platform: Optional[str] = None
) -> Optional[Path]:
    """Return the first executable llama-server in a supported layout, GPU-capable preferred."""
    usable = lambda candidate: _usable_binary(candidate, platform = platform)  # noqa: E731
    ordered = prefer_gpu_capable(llama_server_candidates(directory, platform = platform), usable)
    return next((candidate for candidate in ordered if usable(candidate)), None)


def get_stored_custom_llama_cpp_path() -> Optional[Path]:
    """The Unsloth-selected directory, or ``None`` when automatic discovery is active."""
    try:
        from storage.studio_db import get_app_setting
        from utils.account_context import OWNER, run_as
        value = run_as(OWNER, get_app_setting, CUSTOM_LLAMA_CPP_PATH_SETTING_KEY, None)
    except Exception:
        # A settings DB problem must not take the bundled runtime down with it.
        return None
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > MAX_CUSTOM_LLAMA_CPP_PATH_LENGTH:
        return None
    return Path(value).expanduser()


def _environment_override() -> tuple[Optional[str], Optional[str], bool]:
    """``(path, variable, direct_binary)`` for the existing environment pins."""
    direct = os.environ.get("LLAMA_SERVER_PATH", "").strip()
    if direct:
        return direct, "LLAMA_SERVER_PATH", True
    directory = os.environ.get("UNSLOTH_LLAMA_CPP_PATH", "").strip()
    if directory and os.environ.get(MANAGED_LLAMA_CPP_PATH_MARKER) != "1":
        return directory, "UNSLOTH_LLAMA_CPP_PATH", False
    return None, None, False


def custom_llama_cpp_path_source() -> str:
    """The active custom-path authority: environment, studio, or default."""
    env_path, _variable, _direct = _environment_override()
    if env_path is not None:
        return "environment"
    if get_stored_custom_llama_cpp_path() is not None:
        return "studio"
    return "default"


def _canonical_directory(value: str) -> Path:
    raw = value.strip()
    if not raw:
        raise ValueError("Choose a llama.cpp folder or use the bundled runtime.")
    if len(raw) > MAX_CUSTOM_LLAMA_CPP_PATH_LENGTH:
        raise ValueError("The llama.cpp folder path is too long.")
    try:
        directory = Path(raw).expanduser().resolve(strict = True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("The llama.cpp folder does not exist or cannot be accessed.") from exc
    if not directory.is_dir():
        raise ValueError("The custom llama.cpp path must be a folder.")
    if resolve_llama_server_binary(directory) is None:
        binary_name = llama_server_binary_name()
        raise ValueError(
            f"No executable {binary_name} was found in that folder or its build/bin directory."
        )
    return directory


def set_custom_llama_cpp_path(value: Optional[str]) -> Optional[Path]:
    """Store a validated directory. ``None`` restores automatic discovery."""
    global _path_revision
    env_path, variable, _direct = _environment_override()
    if env_path is not None:
        raise RuntimeError(f"The llama.cpp path is managed by the {variable} environment variable.")
    directory = _canonical_directory(value) if value is not None else None
    with _settings_lock:
        from storage.studio_db import upsert_app_settings
        upsert_app_settings(
            {CUSTOM_LLAMA_CPP_PATH_SETTING_KEY: (str(directory) if directory is not None else None)}
        )
        _path_revision += 1
    return directory


def custom_llama_cpp_path_revision() -> int:
    """In-process revision used to retire sidecars launched before a path save."""
    with _settings_lock:
        return _path_revision


def custom_llama_cpp_path_status() -> dict:
    """UI payload describing the effective custom-path selection."""
    env_path, variable, direct_binary = _environment_override()
    source = "default"
    path: Optional[Path] = None
    binary: Optional[Path] = None

    if env_path is not None:
        source = "environment"
        path = Path(env_path).expanduser()
        if direct_binary:
            binary = path if _usable_binary(path) else None
        else:
            binary = resolve_llama_server_binary(path)
    else:
        path = get_stored_custom_llama_cpp_path()
        if path is not None:
            source = "studio"
            binary = resolve_llama_server_binary(path)

    return {
        "path": str(path) if path is not None else None,
        "source": source,
        "editable": source != "environment",
        "available": source == "default" or binary is not None,
        "resolved_binary": str(binary) if binary is not None else None,
        "environment_variable": variable,
    }
