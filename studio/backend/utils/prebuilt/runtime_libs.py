# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where the CUDA runtime lives on this host, for the managed servers' child envs.

Two sources the dynamic linker does not find on its own: Python wheels
(``python_runtime_dirs``) and directories another application ships privately
(``vendored_cuda_runtime_dirs``).

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


# Directories where another application ships a private copy of the CUDA runtime.
# They are not on the dynamic linker's search path, but the prebuilt installer
# counts them when it decides which CUDA runtime line this host can run
# (prebuilt_core.py: linux_runtime_dirs_for_required_libraries), so a launcher
# has to put the matching one back or the CUDA backend fails to dlopen and ggml
# drops to CPU. Entries are (root, per-CUDA-major subdirectory prefix).
_VENDORED_CUDA_ROOTS: tuple[tuple[Path, str], ...] = (
    (Path("/usr/local/lib/ollama"), "cuda_v{major}"),
)


def vendored_cuda_runtime_dirs(
    marker: object,
    *,
    roots: Optional[Iterable[tuple[Path, str]]] = None,
) -> list[str]:
    """CUDA runtime dirs private to another application, for an installed build.

    ``marker`` is the parsed install marker of the build being launched; its
    ``runtime_line`` ("cuda13") says which CUDA major that build needs. Anything
    else (a CPU or Vulkan build, a source build with no marker, a corrupt one)
    yields nothing. Only a directory holding both libcudart and libcublas for
    that exact major qualifies, so an unrelated CUDA major never reaches the
    loader path.

    Callers must place these *last*: they are a rescue for hosts where no other
    copy of the runtime exists, never a replacement for the wheel or system one
    that qualified the build.
    """
    # Every root below is a Linux path, and only Linux uses LD_LIBRARY_PATH to
    # find them. Revisit if a Windows or macOS entry is ever added.
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
        # Guard against cuda_v130 answering a cuda_v13 glob; a trailing minor
        # (cuda_v13.0) is still the right major.
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
