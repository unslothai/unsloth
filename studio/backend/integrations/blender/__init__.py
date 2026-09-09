# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


"""Studio-managed Blender MCP integration."""

import sys
from pathlib import Path

from .runtime import runtime_path

MIN_PYTHON_VERSION = (3, 10)
MIN_BLENDER_VERSION = "5.1.0"


def launch_command() -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve().with_name("launcher.py")),
        str(runtime_path()),
    ]
