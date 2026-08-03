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
        # The Xet shim sets this process-wide for the duration of one optional import, so a child
        # spawned in that window would inherit it for its whole life -- and unsloth_zoo injects
        # triton and bitsandbytes STUBS when it is set, so a training child would run against
        # no-ops. An operator who set it deliberately is not affected: only the loader's own
        # transient value is dropped.
        try:
            from utils.hf_xet_fallback import gpu_init_override_active
            if gpu_init_override_active():
                child.pop("UNSLOTH_ZOO_DISABLE_GPU_INIT", None)
        except Exception:  # noqa: BLE001 - never fail a spawn over this
            pass
    return child
