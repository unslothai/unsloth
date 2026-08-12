# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Probe torch allocation in a child so driver crashes do not kill the backend.

Only a killed or hung child marks a device unusable. Ordinary Python errors are
left to the in-process loader. Set ``UNSLOTH_STUDIO_DISABLE_DEVICE_PROBE=1`` to
skip the probe.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from functools import lru_cache

from utils.child_stdio import utf8_child_env
from utils.native_path_leases import child_env_without_native_path_secret
from utils.process_lifetime import child_popen_kwargs
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

logger = logging.getLogger(__name__)

DISABLE_ENV_VAR = "UNSLOTH_STUDIO_DISABLE_DEVICE_PROBE"

# Allow for a cold torch import and driver initialization on a busy host.
PROBE_TIMEOUT_SECONDS = 120.0

# The matmul also tests vendor BLAS initialization, which a bare allocation misses.
_PROBE_SCRIPT = (
    "import sys, torch; "
    "t = torch.ones((8, 8), dtype = torch.float16, device = sys.argv[1]); "
    "(t @ t).sum().item()"
)


def _died_by_signal(returncode: int) -> bool:
    """Return whether the code represents a POSIX signal or fatal Windows NTSTATUS."""
    if returncode < 0:
        return True
    return os.name == "nt" and (returncode & 0xC0000000) == 0xC0000000


@lru_cache(maxsize = None)
def device_can_allocate(device: str) -> bool:
    """Return false only when the cached child probe crashes or times out.

    Spawn failures and ordinary child exceptions return true so the existing
    in-process loader can report them accurately.
    """
    if os.environ.get(DISABLE_ENV_VAR) == "1":
        return True
    try:
        probe = subprocess.run(
            [sys.executable, "-c", _PROBE_SCRIPT, device],
            stdout = subprocess.DEVNULL,
            stderr = subprocess.PIPE,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = PROBE_TIMEOUT_SECONDS,
            env = utf8_child_env(child_env_without_native_path_secret()),
            # Keep a wedged child from outliving the backend.
            **child_popen_kwargs(),
            **windows_hidden_subprocess_kwargs(),
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "torch allocation probe on %s did not finish in %.0fs; treating the "
            "device as unusable",
            device,
            PROBE_TIMEOUT_SECONDS,
        )
        return False
    except Exception:  # noqa: BLE001 - no child ran, so nothing was proven
        logger.debug("torch allocation probe on %s could not run", device, exc_info = True)
        return True
    if _died_by_signal(probe.returncode):
        stderr = (probe.stderr or "").strip()
        logger.warning(
            "torch allocation probe on %s was killed (exit %s); this torch build "
            "cannot use the device without crashing the process%s",
            device,
            probe.returncode,
            f": {stderr}" if stderr else "",
        )
        return False
    return True
