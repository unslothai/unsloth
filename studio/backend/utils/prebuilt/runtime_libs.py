# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CUDA runtime dirs shipped inside Python wheels, for the STT sidecar's child env.

Kept in sync with install_llama_prebuilt.py's python_runtime_dirs; the backend
cannot import the studio/ installer scripts, so this small copy stays importable
with only the backend root on sys.path.
"""

from __future__ import annotations

import site
import sys
from pathlib import Path
from typing import Iterable


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
