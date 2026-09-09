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
    if env is None and child.get("UNSLOTH_ZOO_DISABLE_GPU_INIT") == "1":
        # The Xet shim sets this process-wide for one optional import, and a child spawned in that window inherits it
        # for life, running against unsloth_zoo's triton/bitsandbytes STUBS.
        try:
            from utils.hf_xet_fallback import gpu_init_override_active
            if gpu_init_override_active():
                child.pop("UNSLOTH_ZOO_DISABLE_GPU_INIT", None)
        except Exception:  # noqa: BLE001 - never fail a spawn over this
            pass
    return child
