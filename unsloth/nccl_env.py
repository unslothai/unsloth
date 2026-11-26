# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import os
import sys

__all__ = ["apply_nccl_default_env"]

# Communicators torch splits from the default one (the dims of a multi-dimensional DeviceMesh, as
# in dp x tp, pp x dp or ep_rep x ep layouts) reuse its channels and buffers instead of allocating
# their own: two 2-D meshes on 2 T4s drop from 1196 to 444 MiB per rank on torch 2.15. A 1-D mesh
# over the world reuses the default group, so nothing changes there. NCCL reads it at communicator
# creation, so it has to be in the environment before init_process_group. An explicit value,
# including NCCL_COMM_SPLIT_SHARE_RESOURCES=0, is kept.
_NCCL_DEFAULT_ENV = (("NCCL_COMM_SPLIT_SHARE_RESOURCES", "1"),)


def apply_nccl_default_env(environ = None):
    """Set Unsloth's NCCL defaults the user has not set. Returns the names that were set.

    Nothing is changed once the default process group exists: NCCL already read its
    configuration, and ProcessGroupNCCL threads read the environment concurrently.
    """
    environ = os.environ if environ is None else environ
    dist = sys.modules.get("torch.distributed")
    try:
        if dist is not None and dist.is_available() and dist.is_initialized():
            return []
    except Exception:
        return []
    changed = []
    for name, value in _NCCL_DEFAULT_ENV:
        if name not in environ:
            environ[name] = value
            changed.append(name)
    return changed
