# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Gives each xdist worker its own TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR; import before torch."""

from __future__ import annotations

import os
import pathlib
import tempfile

WORKER_ENV = "PYTEST_XDIST_WORKER"


def isolate_compile_caches() -> str | None:
    """Point this xdist worker's inductor and Triton caches at its own dir; returns None outside xdist."""
    worker = os.environ.get(WORKER_ENV)
    if not worker:
        return None

    base = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if base:
        # Respect an explicit location and split underneath it; CI may point it at a cached path.
        root = pathlib.Path(base)
    else:
        root = pathlib.Path(tempfile.gettempdir()) / f"torchinductor_{os.environ.get('USER', 'ci')}"

    mine = root / f"xdist_{worker}"
    mine.mkdir(parents = True, exist_ok = True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(mine)
    os.environ["TRITON_CACHE_DIR"] = str(mine / "triton")
    return str(mine)


isolate_compile_caches()
