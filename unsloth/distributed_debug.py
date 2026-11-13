# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import os
import sys

__all__ = ["apply_debug_distributed_env", "keep_debug_nccl_env"]

# Read at process group / communicator creation, so they only need to precede init_process_group.
_DEBUG_DISTRIBUTED_ENV = (
    ("TORCH_NCCL_DESYNC_DEBUG", "1"),
    ("TORCH_NCCL_DUMP_ON_TIMEOUT", "1"),
    ("NCCL_DEBUG", "INFO"),
)
# Flight recorder ring size. Both names are needed: the recorder reads TORCH_FR_BUFFER_SIZE first
# (torch >= 2.8), but ProcessGroupNCCL gates dump on timeout on TORCH_NCCL_TRACE_BUFFER_SIZE alone
# (ProcessGroupNCCL.hpp, through 2.13), the only name torch 2.6 / 2.7 know. Costs torch >= 2.8 one
# deprecation warning.
_FR_BUFFER_NAMES = ("TORCH_FR_BUFFER_SIZE", "TORCH_NCCL_TRACE_BUFFER_SIZE")
_FR_BUFFER_SIZE = "2000"


def _enabled(environ):
    return environ.get("UNSLOTH_DEBUG_DISTRIBUTED", "0").lower() in ("1", "true", "yes", "on")


def apply_debug_distributed_env(environ = None):
    """Under UNSLOTH_DEBUG_DISTRIBUTED=1, enable NCCL / c10d hang diagnostics.

    Variables the user already set are left alone. The two flight recorder names are
    one setting: a value given under either is copied to the other, so an explicit
    TORCH_NCCL_TRACE_BUFFER_SIZE=0 is not overridden. Returns the names that were set.
    """
    environ = os.environ if environ is None else environ
    if not _enabled(environ):
        return []
    dist = sys.modules.get("torch.distributed")
    try:
        initialized = dist is not None and dist.is_available() and dist.is_initialized()
    except Exception:
        initialized = False
    if initialized:
        # Too late to take effect, and ProcessGroupNCCL threads read the environment
        # concurrently (setenv races getenv), so leave it untouched.
        print(
            "Unsloth: UNSLOTH_DEBUG_DISTRIBUTED=1 is set but the process group already exists, "
            "so it was not applied. Import unsloth before init_process_group, or export "
            "the variables in the launcher.",
            file = sys.stderr,
        )
        return []
    changed = []
    for name, value in _DEBUG_DISTRIBUTED_ENV:
        if name not in environ:
            environ[name] = value
            changed.append(name)
    given = [environ[name] for name in _FR_BUFFER_NAMES if name in environ]
    for name in _FR_BUFFER_NAMES:
        if name not in environ:
            environ[name] = given[0] if given else _FR_BUFFER_SIZE
            changed.append(name)
    return changed


def keep_debug_nccl_env(environ = None):
    """Under UNSLOTH_DEBUG_DISTRIBUTED=1, snapshot NCCL_DEBUG and return a callable restoring it.

    unsloth_zoo's import sets NCCL_DEBUG=WARN unconditionally when UNSLOTH_ENABLE_LOGGING=1 or
    UNSLOTH_COMPILE_DEBUG=1, which would drop the INFO logs (or the user's own value) chosen here.
    Off, or with NCCL_DEBUG unset, the callable does nothing.
    """
    environ = os.environ if environ is None else environ
    saved = environ.get("NCCL_DEBUG") if _enabled(environ) else None

    def restore():
        if saved is not None and environ.get("NCCL_DEBUG") != saved:
            environ["NCCL_DEBUG"] = saved

    return restore
