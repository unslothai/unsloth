# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read packaged data whether the package is a directory or inside a zipapp.

`Path(__file__).parent / "thing.js"` works in a checkout and returns a path that does not exist
inside `studiobench.pyz`, because a zipapp's contents are not on the filesystem. Caught by running
the built artifact's own `--doctor`, which reported the frozen corpus missing at a path ending
`studiobench.pyz/tests/studio/studiobench/fixture/corpus/frozen/manifest.json` -- a path that can
never exist. Every packaged file this tool reads at run time goes through here.

The filesystem is tried FIRST, so a checkout run picks up an edit to a .js file without a
reinstall, which is most of what makes the JS pleasant to work on.
"""

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
    """A directory the tool may WRITE to, which is never inside the package.

    A zipapp is read-only, and a checkout may be. Anything the tool produces belongs in the run's
    output directory, not next to its own source.
    """
    target = Path(preferred) if preferred else Path.cwd() / "studiobench-out"
    target.mkdir(parents = True, exist_ok = True)
    return target
