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

import os
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


_LOADER_DEFAULT_LIB_DIRS: tuple[str, ...] = (
    "/lib",
    "/lib64",
    "/usr/lib",
    "/usr/lib64",
    "/usr/local/lib",
    "/usr/local/lib64",
)


def _ld_cache_sonames() -> "frozenset[str] | None":
    """Every soname in the loader's cache, or None when no ldconfig can be read."""
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
        # "\tlibfoo.so.1 (libc6,x86-64) => /usr/lib/libfoo.so.1"
        return frozenset(
            line.strip().split(" ", 1)[0]
            for line in (result.stdout or "").splitlines()
            if "=>" in line and line.strip()
        )
    return None


def _loader_already_provides_runtime(major: str) -> bool:
    """Whether the loader finds this CUDA major's pair without the vendored dir.

    The vendored dir joins LD_LIBRARY_PATH, which outranks the loader's cache and its
    default dirs however late the entry sits, so it is safe only where nothing else
    provides both libs. A cache that cannot be read is ignorance rather than absence:
    the default dirs answer instead, and the rescue stands.
    """
    cached = _ld_cache_sonames()
    for soname in (f"libcudart.so.{major}", f"libcublas.so.{major}"):
        if cached is not None:
            if soname not in cached:
                return False
            continue
        if not any(
            os.path.isfile(os.path.join(directory, soname))
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
                and any(directory.glob(f"libcudart.so.{major}*"))
                and any(directory.glob(f"libcublas.so.{major}*"))
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
