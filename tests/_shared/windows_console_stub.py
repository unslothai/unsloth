# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A genuinely runnable Windows .exe that is not a renamed system binary.

Tests that need a real PE at a chosen name used to copy System32's where.exe there.
A Microsoft-signed tool renamed to llama-server.exe or trusted-signing-cli.exe is
itself an AV heuristic, so this builds the file the way pip builds every
Scripts\\*.exe: distlib's console launcher, a shebang naming this interpreter, and a
zipped __main__.py that exits with the requested code. The launcher reads the
shebang from its own file, so the bytes can be written anywhere and still start.
"""

from __future__ import annotations

import io
import platform
import struct
import sys
import zipfile
from pathlib import Path


def _launcher_dir() -> Path | None:
    for name in ("pip._vendor.distlib", "distlib"):
        try:
            module = __import__(name, fromlist = ["scripts"])
        except ImportError:
            continue
        return Path(module.__file__).parent
    return None


def console_stub_bytes(exit_code: int = 0, *, source: str | None = None) -> bytes | None:
    """Launcher + shebang + zip, or None where no distlib launcher ships. `source` replaces the exit-only body."""
    directory = _launcher_dir()
    if directory is None:
        return None
    if platform.machine().upper() in ("ARM64", "AARCH64"):
        launcher = "t64-arm.exe"
    else:
        launcher = "t64.exe" if struct.calcsize("P") == 8 else "t32.exe"
    path = directory / launcher
    if not path.is_file():
        return None
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        body = source if source is not None else f"import sys\nsys.exit({int(exit_code)})\n"
        archive.writestr("__main__.py", body)
    shebang = b'#!"' + sys.executable.encode("utf-8") + b'"\n'
    return path.read_bytes() + shebang + stream.getvalue()
