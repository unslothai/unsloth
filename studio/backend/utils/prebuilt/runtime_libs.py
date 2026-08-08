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

import re
import site
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


# (root, per-CUDA-major subdir prefix) for apps shipping a private CUDA runtime.
# Off the linker's search path, but the installer counts them when it picks a
# runtime line (prebuilt_core.py: linux_runtime_dirs_for_required_libraries), so
# a launcher must restore the matching one or ggml silently drops to CPU.
_VENDORED_CUDA_ROOTS: tuple[tuple[Path, str], ...] = (
    (Path("/usr/local/lib/ollama"), "cuda_v{major}"),
)


def vendored_cuda_runtime_dirs(
    marker: object, *, roots: Optional[Iterable[tuple[Path, str]]] = None
) -> list[str]:
    """CUDA runtime dirs private to another application, for an installed build.

    ``marker`` is the build's parsed install marker; its ``runtime_line``
    ("cuda13") picks the CUDA major. Anything else yields nothing. A dir
    qualifies only with both libcudart and libcublas for that exact major.

    Callers must place the result last: it rescues hosts with no other copy of
    the runtime, and must never displace the one that qualified the build.
    """
    # Every root below is a Linux path. Revisit if a Windows/macOS entry is added.
    if not sys.platform.startswith("linux"):
        return []

    runtime_line = marker.get("runtime_line") if isinstance(marker, dict) else None
    match = re.fullmatch(r"cuda(\d+)", runtime_line if isinstance(runtime_line, str) else "")
    if match is None:
        return []

    major = match.group(1)
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
        candidates.extend(root.glob("nvidia/*/lib"))  # Linux convention
        candidates.extend(root.glob("nvidia/*/bin"))  # legacy modular Windows wheels
        candidates.extend(root.glob("nvidia/*/bin/x86_64"))  # CUDA 13 Windows wheel layout
        candidates.extend(root.glob("nvidia/*/bin/x64"))
        candidates.extend(root.glob("nvidia/*/Library/bin"))  # conda-style repacks
        candidates.extend(root.glob("nvidia/*/Library/bin/x86_64"))
        candidates.extend(root.glob("nvidia/*/Library/bin/x64"))
        candidates.extend(root.glob("torch/lib"))
    return dedupe_existing_dirs(candidates)
