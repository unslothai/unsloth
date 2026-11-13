# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A stand-in llama-server that exits 0 at once on every OS (Windows has no /usr/bin/true)."""

from __future__ import annotations

import sys
from pathlib import Path


def fake_llama_server(directory: Path) -> str:
    directory.mkdir(parents = True, exist_ok = True)
    if sys.platform == "win32":
        fake = directory / "llama-server.cmd"
        fake.write_text("@exit /b 0\r\n", encoding = "utf-8")
    else:
        fake = directory / "llama-server"
        fake.write_text("#!/bin/sh\nexit 0\n", encoding = "utf-8")
        fake.chmod(0o755)
    return str(fake)
