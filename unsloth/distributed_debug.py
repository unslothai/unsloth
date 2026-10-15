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
# Flight recorder ring size. Both names are needed: the recorder reads TORCH_FR_BUFFER_SIZE first
# (torch >= 2.8), but ProcessGroupNCCL gates dump on timeout on TORCH_NCCL_TRACE_BUFFER_SIZE alone
# (ProcessGroupNCCL.hpp, through 2.13), the only name torch 2.6 / 2.7 know. Costs torch >= 2.8 one
# deprecation warning.
_FR_BUFFER_NAMES = ("TORCH_FR_BUFFER_SIZE", "TORCH_NCCL_TRACE_BUFFER_SIZE")
_FR_BUFFER_SIZE = "2000"


def apply_debug_distributed_env(environ = None):
    """Under UNSLOTH_DEBUG_DISTRIBUTED=1, enable NCCL / c10d hang diagnostics.

    Variables the user already set are left alone. The two flight recorder names are
    one setting: a value given under either is copied to the other, so an explicit
    TORCH_NCCL_TRACE_BUFFER_SIZE=0 is not overridden. Returns the names that were set.
    """
    environ = os.environ if environ is None else environ
    if environ.get("UNSLOTH_DEBUG_DISTRIBUTED", "0").lower() not in ("1", "true", "yes", "on"):
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
