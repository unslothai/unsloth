# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tells Python children to emit UTF-8: on Windows their stdout defaults to the ANSI code page"""

from __future__ import annotations

import os
from typing import Mapping, Optional


def utf8_child_env(env: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Copy *env* (or the current environment) with UTF-8 stdio forced."""
    child = dict(os.environ if env is None else env)
    child["PYTHONIOENCODING"] = "utf-8"
    if env is None and child.get("UNSLOTH_ZOO_DISABLE_GPU_INIT") == "1":
        # A child spawned while the Xet shim sets this would run against zoo stubs for life.
        try:
            from utils.hf_xet_fallback import gpu_init_override_active
            if gpu_init_override_active():
                child.pop("UNSLOTH_ZOO_DISABLE_GPU_INIT", None)
        except Exception:  # noqa: BLE001 - never fail a spawn over this
            pass
    return child
