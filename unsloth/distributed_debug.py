# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import os
import sys

__all__ = ["apply_debug_distributed_env"]

# Read at process group / communicator creation, so they only need to precede init_process_group.
_DEBUG_DISTRIBUTED_ENV = (
    ("TORCH_NCCL_DESYNC_DEBUG", "1"),
    ("TORCH_NCCL_DUMP_ON_TIMEOUT", "1"),
    ("NCCL_DEBUG", "INFO"),
)
# Flight recorder ring size: torch >= 2.8 reads TORCH_FR_BUFFER_SIZE and warns that the fallback
# TORCH_NCCL_TRACE_BUFFER_SIZE is deprecated, the only name torch 2.6 / 2.7 know. 2.7 / 2.8
# default it to 0 (off).
_FR_BUFFER_NAMES = ("TORCH_FR_BUFFER_SIZE", "TORCH_NCCL_TRACE_BUFFER_SIZE")
_FR_BUFFER_SIZE = "2000"


def _flight_recorder_buffer_name(torch_version = None):
    if torch_version is None:
        try:
            from importlib.metadata import version

            torch_version = version("torch")
        except Exception:
            return _FR_BUFFER_NAMES[0]
    try:
        major, minor = (int(x) for x in torch_version.split("+")[0].split(".")[:2])
    except ValueError:
        return _FR_BUFFER_NAMES[0]
    return _FR_BUFFER_NAMES[0] if (major, minor) >= (2, 8) else _FR_BUFFER_NAMES[1]


def apply_debug_distributed_env(environ = None, torch_version = None):
    """Under UNSLOTH_DEBUG_DISTRIBUTED=1, enable NCCL / c10d hang diagnostics.

    Variables the user already set are left alone, and the two flight recorder names
    are treated as one setting, so an explicit TORCH_NCCL_TRACE_BUFFER_SIZE=0 is not
    overridden through the newer name. Returns the names that were set.
    """
    environ = os.environ if environ is None else environ
    if environ.get("UNSLOTH_DEBUG_DISTRIBUTED", "0").lower() not in ("1", "true", "yes", "on"):
        return []
    changed = []
    for name, value in _DEBUG_DISTRIBUTED_ENV:
        if name not in environ:
            environ[name] = value
            changed.append(name)
    if not any(name in environ for name in _FR_BUFFER_NAMES):
        name = _flight_recorder_buffer_name(torch_version)
        environ[name] = _FR_BUFFER_SIZE
        changed.append(name)
    dist = sys.modules.get("torch.distributed")
    try:
        initialized = dist is not None and dist.is_available() and dist.is_initialized()
    except Exception:
        initialized = False
    if initialized and changed:
        print(
            "Unsloth: UNSLOTH_DEBUG_DISTRIBUTED=1 was applied after the process group was "
            "created, so torch.distributed will not see it. Import unsloth before "
            "init_process_group, or export the variables in the launcher.",
            file = sys.stderr,
        )
    return changed
