# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CUDA runtime dirs the dynamic linker does not find on its own, for the managed
servers' child envs: Python wheels, and dirs another application ships privately.

Kept in sync with install_llama_prebuilt.py's python_runtime_dirs and
prebuilt_core.py's linux_runtime_dirs_for_required_libraries; the backend cannot
import the studio/ installer scripts, so this small copy stays importable with
only the backend root on sys.path.
"""

from __future__ import annotations

import glob
import os
import platform
import re
import shutil
import site
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional


def dedupe_existing_dirs(paths: Iterable[str | Path]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        if not raw:
            continue
        try:
            path = Path(raw).expanduser()
            if not path.is_dir():
                continue
            resolved = str(path.resolve())
        except (OSError, ValueError):
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


# Debian, Ubuntu and their derivatives put every library in an architecture
# subdirectory and bake that directory into ld.so's own search path: glibc prints
# /lib/x86_64-linux-gnu and /usr/lib/x86_64-linux-gnu ahead of /lib and /usr/lib as
# its first system dirs, with no ldconfig involved. The glob is used rather than
# the machine name because that is what these layouts are actually named, on every
# architecture, and a layout that is not installed cannot hold a runtime anyway.
_MULTIARCH_LIB_GLOBS: tuple[str, ...] = ("/lib/*-linux-gnu*", "/usr/lib/*-linux-gnu*")


def _multiarch_lib_dirs() -> list[str]:
    """The arch-specific dirs this host's loader searches without being asked.

    Counted alongside the generic defaults below, because prebuilt_core.py's
    linux_runtime_dirs_for_required_libraries already counts them when it picks a
    runtime line: a system CUDA runtime in one of them is not absent just because
    no ``ldconfig -p`` can be read, and calling it absent would add the vendored
    dir over the top of the very runtime the loader probe exists to protect.
    """
    found: list[str] = []
    for pattern in _MULTIARCH_LIB_GLOBS:
        try:
            found.extend(sorted(glob.glob(pattern)))
        except OSError:
            continue
    return found


_LOADER_DEFAULT_LIB_DIRS: tuple[str, ...] = (
    "/lib",
    "/lib64",
    "/usr/lib",
    "/usr/lib64",
    # ld.so searches the architecture-specific dirs below ahead of these generic dirs.
    *_multiarch_lib_dirs(),
)


def _ld_cache_entries() -> tuple[tuple[str, str, str], ...] | None:
    """Cached (soname, ABI, path) entries, or None when ldconfig cannot be read."""
    for candidate in ("ldconfig", "/sbin/ldconfig", "/usr/sbin/ldconfig"):
        exe = shutil.which(candidate) if "/" not in candidate else candidate
        if not exe or not os.path.exists(exe):
            continue
        try:
            result = subprocess.run(
                [exe, "-p"],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode != 0:
            continue
        entries: list[tuple[str, str, str]] = []
        for line in (result.stdout or "").splitlines():
            match = re.match(r"\s*(\S+)\s+\(([^)]*)\)\s+=>\s+(\S+)", line)
            if match:
                soname, abi_text, path = match.groups()
                abi = abi_text.strip().lower()
                entries.append((soname, abi, path))
        return tuple(entries)
    return None


def _loader_resolves_sonames(sonames: tuple[str, ...]) -> bool:
    """Ask the dynamic loader in a child process, without loading CUDA into Studio."""
    script = "import ctypes, sys; [ctypes.CDLL(name) for name in sys.argv[1:]]"
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", script, *sonames],
            capture_output = True,
            timeout = 10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


_NATIVE_LOADER_ABIS: dict[str, frozenset[str]] = {
    "x86_64": frozenset({"libc6,x86-64", "libc6,x86-64-v2", "libc6,x86-64-v3", "libc6,x86-64-v4"}),
    "amd64": frozenset({"libc6,x86-64", "libc6,x86-64-v2", "libc6,x86-64-v3", "libc6,x86-64-v4"}),
    "i386": frozenset({"libc6,i686", "libc6,i586", "libc6,i486", "libc6,i386"}),
    "i686": frozenset({"libc6,i686", "libc6,i586", "libc6,i486", "libc6,i386"}),
    "aarch64": frozenset({"libc6,aarch64"}),
    "arm64": frozenset({"libc6,aarch64"}),
    "armv7l": frozenset({"libc6,armhf"}),
    "ppc64le": frozenset({"libc6,ppc64le"}),
    "s390x": frozenset({"libc6,s390x"}),
    "riscv64": frozenset({"libc6,riscv64"}),
}


def _loader_already_provides_runtime(major: str) -> bool:
    """Whether the loader finds this CUDA major's pair without the vendored dir.

    LD_LIBRARY_PATH outranks both the cache and default dirs, so never add the
    vendored pair if either source already provides both native-ABI libraries.
    """
    sonames = (f"libcudart.so.{major}", f"libcublas.so.{major}")
    cached = _ld_cache_entries()
    if cached is None and _loader_resolves_sonames(sonames):
        return True
    native_abis = _NATIVE_LOADER_ABIS.get(platform.machine().lower(), frozenset())
    for soname in sonames:
        cache_has_compatible = cached is not None and any(
            cached_soname == soname
            and abi in native_abis
            and os.path.isfile(path)
            and os.access(path, os.R_OK)
            for cached_soname, abi, path in cached
        )
        if cache_has_compatible:
            continue
        if not any(
            os.path.isfile(os.path.join(directory, soname))
            and os.access(os.path.join(directory, soname), os.R_OK)
            for directory in _LOADER_DEFAULT_LIB_DIRS
        ):
            return False
    return True


_VENDORED_CUDA_ROOTS: tuple[tuple[Path, str], ...] = (
    (Path("/usr/local/lib/ollama"), "cuda_v{major}"),
)


def vendored_cuda_runtime_dirs(
    marker: object, *, roots: Optional[Iterable[tuple[Path, str]]] = None
) -> list[str]:
    """CUDA runtime dirs an installed build selects from a private root.

    ``marker`` is the build's install marker; its ``runtime_line`` ("cuda13")
    picks the CUDA major. A dir qualifies only with both libcudart and libcublas
    for that exact major, and is yielded only while the loader cannot already
    find that pair. Callers place the result last: it rescues hosts with no
    other copy of the runtime, and must never displace the one the build picked.
    """
    if not sys.platform.startswith("linux"):
        return []
    runtime_line = marker.get("runtime_line") if isinstance(marker, dict) else None
    match = re.fullmatch(r"cuda(\d+)", runtime_line if isinstance(runtime_line, str) else "")
    if match is None:
        return []
    major = match.group(1)
    if _loader_already_provides_runtime(major):
        return []
    found: list[Path] = []
    for root, prefix_template in _VENDORED_CUDA_ROOTS if roots is None else roots:
        prefix = prefix_template.format(major = major)
        # cuda_v130 must not answer a cuda_v13 glob; cuda_v13.0 still should.
        exact = re.compile(rf"{re.escape(prefix)}(?:[._-].*)?")
        try:
            found.extend(
                directory
                for directory in sorted(Path(root).glob(f"{prefix}*"))
                if exact.fullmatch(directory.name)
                and all(
                    (directory / f"lib{name}.so.{major}").is_file()
                    and os.access(directory / f"lib{name}.so.{major}", os.R_OK)
                    for name in ("cudart", "cublas")
                )
            )
        except OSError:
            continue
    return dedupe_existing_dirs(found)


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
        # A sys.path entry this user cannot stat makes is_dir() raise, and the
        # caller turns that into an empty dir list, dropping every CUDA wheel dir.
        # Kept in sync with install_llama_prebuilt.py's python_runtime_dirs.
        try:
            if not root.is_dir():
                continue
        except (OSError, ValueError):
            continue
        candidates.extend(root.glob("nvidia/*/lib"))
        candidates.extend(root.glob("nvidia/*/bin"))
        candidates.extend(root.glob("nvidia/*/bin/x86_64"))  # CUDA 13 Windows wheel layout
        candidates.extend(root.glob("nvidia/*/bin/x64"))
        candidates.extend(root.glob("nvidia/*/Library/bin"))
        candidates.extend(root.glob("nvidia/*/Library/bin/x86_64"))
        candidates.extend(root.glob("nvidia/*/Library/bin/x64"))
        candidates.extend(root.glob("torch/lib"))
    return dedupe_existing_dirs(candidates)
