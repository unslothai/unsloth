# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Early CPU thread-pool configuration for Unsloth processes."""

import os
from typing import MutableMapping, Optional


_THREAD_POOL_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_cpu_threads(env: Optional[MutableMapping[str, str]] = None) -> None:
    """Apply ``UNSLOTH_CPU_THREADS`` to native CPU pools when configured.

    Must run before importing libraries that initialize an OpenMP or BLAS
    pool. Library-specific vars are left untouched so users can override a
    single runtime independently.
    """
    environ = os.environ if env is None else env
    configured = environ.get("UNSLOTH_CPU_THREADS", "").strip()
    if not configured:
        return

    try:
        thread_count = int(configured)
    except ValueError as exc:
        raise ValueError("UNSLOTH_CPU_THREADS must be a positive integer") from exc
    if thread_count < 1:
        raise ValueError("UNSLOTH_CPU_THREADS must be a positive integer")

    value = str(thread_count)
    for variable in _THREAD_POOL_ENV_VARS:
        environ.setdefault(variable, value)
