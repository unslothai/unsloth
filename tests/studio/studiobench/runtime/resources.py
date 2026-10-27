# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reads packaged data from a directory or a zipapp; Path(__file__) paths do not exist in a zipapp."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

PACKAGE = "tests.studio.studiobench"


def read_bytes(relative: str) -> bytes:
    """`relative` is a POSIX path under the studiobench package, e.g. `scene/dom.js`."""
    direct = Path(__file__).resolve().parents[1] / relative
    if direct.exists():
        return direct.read_bytes()
    from importlib.resources import files

    resource = files(PACKAGE)
    for part in relative.split("/"):
        resource = resource.joinpath(part)
    return resource.read_bytes()


def read_text(relative: str) -> str:
    return read_bytes(relative).decode("utf-8")


def exists(relative: str) -> bool:
    try:
        read_bytes(relative)
        return True
    except (FileNotFoundError, OSError, ModuleNotFoundError):
        return False


def iter_lines(relative: str):
    for line in read_text(relative).splitlines():
        if line.strip():
            yield line


def writable_dir(preferred: Optional[Path] = None) -> Path:
    """A directory for output, never inside the package: a zipapp is read-only and a checkout may be."""
    target = Path(preferred) if preferred else Path.cwd() / "studiobench-out"
    target.mkdir(parents = True, exist_ok = True)
    return target
