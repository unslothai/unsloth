# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Make a Python child agree with the parent that its pipes are UTF-8.

A child's ``sys.stdout`` uses ``locale.getpreferredencoding()``, which on
Windows is the ANSI code page. Reading that pipe as UTF-8 would then mangle any
non-ASCII the child prints, so the child has to be told which encoding to emit.
Only needed for Python children; llama.cpp and node already emit UTF-8.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional


def utf8_child_env(env: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Copy *env* (or the current environment) with UTF-8 stdio forced."""
    child = dict(os.environ if env is None else env)
    child["PYTHONIOENCODING"] = "utf-8"
    return child
