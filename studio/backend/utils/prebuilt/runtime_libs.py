# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""On-disk CUDA runtime-line detection for the prebuilt installers.

The prebuilt CUDA bundles are dynamically linked and do NOT ship the CUDA runtime
(libcudart / libcublas) -- they load the same runtime the host already has (the
libraries torch ships, a system CUDA toolkit, ...). So the driver's advertised
CUDA version is only an upper bound: a cuda13 bundle still needs cuda13 runtime
libraries actually present on disk. This module scans for them and reports which
`cuda<major>` lines are usable, so selection can intersect
detected(on-disk) with driver-compatible lines -- exactly what
install_llama_prebuilt.py does (`detected_linux_runtime_lines` /
`detected_windows_runtime_lines`). Ported from there.
"""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path
from typing import Iterable

from .hosts import _run
from .selection import _MIN_CUDA_MAJOR

# Highest CUDA major we probe for installed runtime libraries. Detection is
# generated per major so a new toolkit (cuda14, ...) needs no code change while
# the cudart64_<major>.dll / libcudart.so.<major> naming holds.
_MAX_PROBE_CUDA_MAJOR = 19


def glob_paths(*patterns: str) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        if any(char in pattern for char in "*?[]"):
            matches.extend(str(path) for path in Path("/").glob(pattern.lstrip("/")))
        else:
            matches.append(pattern)
    return matches


def dedupe_existing_dirs(paths: Iterable[str | Path]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.is_dir():
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def python_runtime_dirs() -> list[str]:
    """CUDA runtime dirs shipped inside Python wheels (torch + nvidia-* wheels)."""
    candidates: list[Path] = []
    search_roots = [Path(entry) for entry in sys.path if entry]
    try:
        search_roots.extend(Path(path) for path in site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            search_roots.append(Path(user_site))
    except Exception:
        pass

    for root in search_roots:
        if not root.is_dir():
            continue
        candidates.extend(root.glob("nvidia/*/lib"))  # Linux convention
        candidates.extend(root.glob("nvidia/*/bin"))  # legacy modular Windows wheels
        candidates.extend(root.glob("nvidia/*/bin/x86_64"))  # CUDA 13 Windows wheel layout
        candidates.extend(root.glob("nvidia/*/bin/x64"))
        candidates.extend(root.glob("nvidia/*/Library/bin"))  # conda-style repacks
        candidates.extend(root.glob("nvidia/*/Library/bin/x86_64"))
        candidates.extend(root.glob("nvidia/*/Library/bin/x64"))
        candidates.extend(root.glob("torch/lib"))
    return dedupe_existing_dirs(candidates)


def ldconfig_runtime_dirs(required_libraries: Iterable[str]) -> list[str]:
    result = _run(["ldconfig", "-p"], timeout = 20)
    if result is None:
        return []
    required = set(required_libraries)
    candidates: list[str] = []
    for line in result.stdout.splitlines():
        if "=>" not in line:
            continue
        library, _, location = line.partition("=>")
        library = library.strip().split()[0]
        if required and library not in required:
            continue
        candidates.append(str(Path(location.strip()).parent))
    return dedupe_existing_dirs(candidates)


def _linux_runtime_dirs_for_required_libraries(required_libraries: Iterable[str]) -> list[str]:
    required = [library for library in required_libraries if library]
    candidates: list[str | Path] = []

    env_dirs = os.environ.get("CUDA_RUNTIME_LIB_DIR", "")
    if env_dirs:
        candidates.extend(part for part in env_dirs.split(os.pathsep) if part)
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if ld_library_path:
        candidates.extend(part for part in ld_library_path.split(os.pathsep) if part)

    cuda_roots: list[Path] = []
    for name in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"):
        value = os.environ.get(name)
        if value:
            cuda_roots.append(Path(value))
    cuda_roots.extend(Path(path) for path in glob_paths("/usr/local/cuda", "/usr/local/cuda-*"))
    for root in cuda_roots:
        candidates.extend([root / "lib", root / "lib64", root / "targets" / "x86_64-linux" / "lib"])

    candidates.extend(
        Path(path)
        for path in glob_paths(
            "/lib",
            "/lib64",
            "/usr/lib",
            "/usr/lib64",
            "/usr/local/lib",
            "/usr/local/lib64",
            "/lib/x86_64-linux-gnu",
            "/usr/lib/x86_64-linux-gnu",
        )
    )
    candidates.extend(
        Path(path) for path in glob_paths("/usr/local/lib/ollama/cuda_v*", "/usr/lib/wsl/lib")
    )
    candidates.extend(Path(path) for path in python_runtime_dirs())
    candidates.extend(Path(path) for path in ldconfig_runtime_dirs(required))
    return dedupe_existing_dirs(candidates)


def _glob_hit(directories: Iterable[str], pattern: str) -> bool:
    """True when `pattern` matches at least one file in any of `directories`.
    (Note: `any(Path(d).glob(p) for d in dirs)` is always truthy because each
    element is a generator object -- the match must be consumed per directory.)"""
    return any(any(Path(directory).glob(pattern)) for directory in directories)


def detected_linux_runtime_lines() -> list[str]:
    """`cuda<major>` lines whose libcudart + libcublas are present on disk, newest
    major first. Ported from install_llama_prebuilt.py."""
    detected: list[str] = []
    for major in range(_MAX_PROBE_CUDA_MAJOR, _MIN_CUDA_MAJOR - 1, -1):
        required = [f"libcudart.so.{major}", f"libcublas.so.{major}"]
        dirs = _linux_runtime_dirs_for_required_libraries(required)
        if all(_glob_hit(dirs, f"{library}*") for library in required):
            detected.append(f"cuda{major}")
    return detected


def _windows_runtime_dirs() -> list[str]:
    candidates: list[str | Path] = []
    env_dirs = os.environ.get("CUDA_RUNTIME_DLL_DIR", "")
    if env_dirs:
        candidates.extend(part for part in env_dirs.split(os.pathsep) if part)
    path_dirs = os.environ.get("PATH", "")
    if path_dirs:
        candidates.extend(part for part in path_dirs.split(os.pathsep) if part)

    cuda_roots: list[Path] = []
    for name in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        value = os.environ.get(name)
        if value:
            cuda_roots.append(Path(value))
    for root in cuda_roots:
        candidates.extend([root / "bin", root / "lib" / "x64"])

    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    toolkit_base = Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA"
    if toolkit_base.is_dir():
        candidates.extend(toolkit_base.glob("v*/bin"))
        candidates.extend(toolkit_base.glob("v*/lib/x64"))

    candidates.extend(Path(path) for path in python_runtime_dirs())
    return dedupe_existing_dirs(candidates)


def detected_windows_runtime_lines() -> list[str]:
    """`cuda<major>` lines whose cudart/cublas DLLs are present on disk, newest
    major first. Ported from install_llama_prebuilt.py `detected_windows_runtime_lines`."""
    dirs = _windows_runtime_dirs()
    detected: list[str] = []
    for major in range(_MAX_PROBE_CUDA_MAJOR, _MIN_CUDA_MAJOR - 1, -1):
        patterns = (f"cudart64_{major}*.dll", f"cublas64_{major}*.dll", f"cublasLt64_{major}*.dll")
        if all(_glob_hit(dirs, pattern) for pattern in patterns):
            detected.append(f"cuda{major}")
    return detected


def detected_cuda_runtime_lines(*, is_windows: bool) -> list[str]:
    """Platform-appropriate on-disk CUDA runtime-line detection (newest first)."""
    return detected_windows_runtime_lines() if is_windows else detected_linux_runtime_lines()
